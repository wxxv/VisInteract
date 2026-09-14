"""
Visualization code execution and chart saving tool

Uses subprocess isolation to prevent memory leaks and process crashes from affecting the main process.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import altair as alt

from vis_interact.config import settings

logger = logging.getLogger(__name__)

_NON_XML_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_for_xml(obj):
    """Recursively clear illegal XML control characters in strings to prevent vl-convert Rust side panic."""
    if isinstance(obj, str):
        return _NON_XML_CHAR_RE.sub('', obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_xml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_xml(item) for item in obj]
    return obj


def _exec_code_worker(code: str, db_path: str, result_pipe):
    """Execute visualization code in a subprocess, return results through pipe."""
    import altair as _alt
    import pandas as _pd
    import sqlite3 as _sqlite3

    exec_globals = {
        "alt": _alt,
        "pd": _pd,
        "sqlite3": _sqlite3,
        "__builtins__": __builtins__,
    }

    chart_types = (
        _alt.Chart, _alt.LayerChart, _alt.HConcatChart,
        _alt.VConcatChart, _alt.FacetChart,
    )

    try:
        lines = code.rstrip().split("\n")
        last_line = lines[-1].strip() if lines else ""

        chart = None
        if last_line and not last_line.startswith(("#", "import", "from")) and "=" not in last_line:
            body = "\n".join(lines[:-1])
            exec(body, exec_globals)
            try:
                result = eval(last_line, exec_globals)
                if isinstance(result, chart_types):
                    chart = result
            except Exception:
                exec(last_line, exec_globals)
        else:
            exec(code, exec_globals)

        if chart is None:
            chart = exec_globals.get("chart")

        if chart is None:
            for _name, obj in exec_globals.items():
                if isinstance(obj, chart_types):
                    chart = obj

        if chart is None:
            result_pipe.send((False, None, "No chart object found in the code"))
        else:
            spec_json = chart.to_json()
            result_pipe.send((True, spec_json, None))
    except Exception as e:
        result_pipe.send((False, None, str(e)))
    finally:
        result_pipe.close()


_spawn_ctx = __import__("multiprocessing").get_context("spawn")


def execute_visualization_code(
    code: str,
    db_id: str,
    timeout: int = 30,
) -> Tuple[bool, Optional[alt.Chart], Optional[dict], Optional[str]]:
    """Execute visualization code (subprocess isolation, timeout can be forced to terminate).

    Returns:
        (success, chart, spec, error_message)
    """
    db_path = str(Path(settings.paths.database_path).resolve())
    code = code.replace("'./databases/", f"'{db_path}/")
    code = code.replace('"./databases/', f'"{db_path}/')

    parent_conn, child_conn = _spawn_ctx.Pipe(duplex=False)
    proc = _spawn_ctx.Process(target=_exec_code_worker, args=(code, db_path, child_conn))
    proc.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout):
            ok, payload, error = parent_conn.recv()
        else:
            proc.kill()
            proc.join(timeout=5)
            logger.warning(f"Code execution timeout ({timeout}s), subprocess terminated")
            return False, None, None, f"Code execution timeout ({timeout}s)"
    except EOFError:
        return False, None, None, "Worker process crashed unexpectedly"
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)

    if not ok:
        return False, None, None, error

    try:
        spec = json.loads(payload)
        chart = alt.Chart.from_dict(spec)
        return True, chart, None, None
    except Exception as e:
        return False, None, None, f"Failed to reconstruct chart from spec: {e}"


def _save_chart_in_subprocess(spec_json: str, output_path: str, result_queue):
    """Execute chart.save() in a subprocess, isolate the memory of the V8 engine of vl-convert."""
    try:
        import altair as _alt
        spec = json.loads(spec_json)
        _alt.Chart.from_dict(spec).save(output_path)
        result_queue.put(("success", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def save_chart_as_image(
    chart: alt.Chart, output_path: str, timeout: int = 60,
) -> Tuple[bool, str]:
    """Save Altair chart as image (subprocess isolation, using spawn to avoid fork+thread problems).

    Returns:
        (success, reason) — when failed, reason contains the specific reason.
    """
    try:
        spec = chart.to_dict()
        spec = _sanitize_for_xml(spec)
        spec_json = json.dumps(spec)
    except Exception as e:
        reason = f"Chart serialization failed: {e}"
        logger.warning(reason)
        return False, reason

    result_queue = _spawn_ctx.Queue()
    proc = _spawn_ctx.Process(
        target=_save_chart_in_subprocess,
        args=(spec_json, output_path, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        reason = f"Render subprocess timed out ({timeout}s)"
        logger.warning(f"{reason}: {output_path}")
        return False, reason

    if proc.exitcode != 0:
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            logger.info(f"Subprocess exit code non-zero but image written: {output_path}")
            return True, ""
        reason = f"Render subprocess crashed (exit code {proc.exitcode})"
        logger.warning(f"{reason}: {output_path}")
        return False, reason

    try:
        if not result_queue.empty():
            status, error = result_queue.get_nowait()
            if status == "success":
                return True, ""
            reason = f"vl-convert render error: {error}"
            logger.warning(reason)
            return False, reason
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            logger.info(f"Queue is empty but image written: {output_path}")
            return True, ""
        reason = "Render subprocess returned no result and no image file was produced"
        logger.warning(reason)
        return False, reason
    except Exception as e:
        reason = f"Failed to retrieve render result: {e}"
        logger.warning(reason)
        return False, reason
