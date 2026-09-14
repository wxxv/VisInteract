"""
OpenAI Function Calling format tool definition and dispatcher

Standard tool set:
  execute_sql      — Execute SQL query
  execute_altair   — Execute Altair visualization code, return image_url
  ask_user_text    — Ask user (pure text clarification)
  ask_user_vis     — Show chart to get visual feedback

Termination tool:
  finish           — Submit final visualization code
"""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vis_interact.config import settings
from vis_interact.agents.user_sim import (
    VisualFeedback,
    ask_user_text as _ask_user_text,
    ask_user_vis as _ask_user_vis,
)
from vis_interact.utils.code_exec import execute_visualization_code, save_chart_as_image

logger = logging.getLogger(__name__)


# ── 运行时上下文 ──────────────────────────────────────────────────────────────

@dataclass
class ToolContext:
    """Runtime context for tool execution (maintained by Agent)"""
    sample_id: str
    db_id: str
    key_features: List[dict]
    gt_code: str
    model: str = ""
    user_text_model: str = ""
    user_vis_model: str = ""
    work_dir: Optional[Path] = None
    text_budget_remaining: int = 3
    vis_budget_remaining: int = 1
    gt_chart_path: Optional[str] = None
    verified_sql: Optional[str] = None
    path_qa_history: List[tuple] = field(default_factory=list)
    generated_images: List[str] = field(default_factory=list)
    vis_scores: List[int] = field(default_factory=list)
    last_altair_code: Optional[str] = None
    _image_counter: int = 0
    _image_path_override: Optional[str] = None

    def __post_init__(self):
        if not self.model:
            self.model = settings.models.default_model
        if not self.user_text_model:
            self.user_text_model = settings.models.user_text_model
        if not self.user_vis_model:
            self.user_vis_model = settings.models.user_vis_model

    def next_image_path(self) -> str:
        d = self.work_dir or Path(tempfile.mkdtemp())
        d.mkdir(parents=True, exist_ok=True)
        if self._image_path_override:
            return str(d / self._image_path_override)
        self._image_counter += 1
        return str(d / f"chart_{self._image_counter}.png")


# ── Tool Schemas ──────────────────────────────────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a SQL query against the SQLite database and return "
                "up to 5 preview rows. Note: this is SQLite — functions like "
                "STDDEV, STDEV, VARIANCE are NOT available; compute them manually "
                "if needed (e.g. AVG(x*x) - AVG(x)*AVG(x) for variance)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A valid SQL query to execute."},
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_altair",
            "description": (
                "Execute Altair visualization Python code in a sandboxed environment. "
                "The code must produce an Altair chart object. "
                "Returns the image URL (file path) of the rendered chart on success, "
                "or an error message on failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete, runnable Python visualization code using Altair.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_text",
            "description": (
                "Ask the user ONE focused question in natural language. "
                "Use this to confirm your plan, verify assumptions, seek preferences, "
                "or resolve ambiguity. Talking to the user leads to better results — "
                "do not hesitate to ask. Costs 1 text-budget unit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The single clarification question to ask."},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_vis",
            "description": (
                "Show the user your current visualization attempt and ask for visual feedback. "
                "Pass your complete, runnable Python visualization code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete, executable Python visualization code to render and show.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Submit the final visualization. Takes NO parameters — "
                "it submits the code from your most recent successful "
                "`execute_altair` call. Can ONLY be called after a "
                "successful `execute_altair`."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ── Tool implementation ──────────────────────────────────────────────────────────────────

_SQL_TIMEOUT_SEC = 10


def _impl_execute_sql(ctx: ToolContext, sql: str) -> str:
    db_path = Path(settings.paths.database_path) / f"{ctx.db_id}.sqlite"
    if not db_path.exists():
        return f"ERROR: Database '{ctx.db_id}' not found at {db_path}."
    try:
        conn = sqlite3.connect(str(db_path))
        deadline = time.monotonic() + _SQL_TIMEOUT_SEC
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(5)
        conn.close()
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return f"SQL ERROR: Query timed out ({_SQL_TIMEOUT_SEC}s). Simplify your query — avoid expensive JOINs with IN/OR on multiple columns; consider using UNION ALL to flatten first, then JOIN."
        return f"SQL ERROR: {e}"
    except Exception as e:
        return f"SQL ERROR: {e}"

    if not columns:
        return "Query executed successfully (no result set)."
    if not rows:
        return "Query returned 0 rows — check your filters or table name."

    ctx.verified_sql = sql
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    suffix = f"\n(showing first {len(rows)} rows)" if len(rows) == 20 else ""
    return f"{header}\n{body}{suffix}"


def _impl_execute_altair(ctx: ToolContext, code: str) -> str:
    if not code or not code.strip():
        return "ERROR: No code provided to execute."
    try:
        success, chart, _spec, error = execute_visualization_code(code, ctx.db_id)
        if not success or chart is None:
            return f"EXECUTION FAILED: {error or 'Unknown error'}"

        img_path = ctx.next_image_path()
        saved, reason = save_chart_as_image(chart, img_path)
        if saved and Path(img_path).exists():
            ctx.generated_images.append(img_path)
            ctx.last_altair_code = code
            return f"SUCCESS: image_url={img_path}"
        return f"EXECUTION FAILED: Code ran but chart image could not be saved. {reason}"
    except Exception as e:
        return f"EXECUTION FAILED: {e}"


def _impl_ask_user_text(ctx: ToolContext, question: str) -> str:
    if ctx.text_budget_remaining <= 0:
        return "BUDGET EXCEEDED: No text-question budget remaining. Proceed with your best guess."
    ctx.text_budget_remaining -= 1
    try:
        answer = _ask_user_text(
            question=question,
            key_features=ctx.key_features,
            gt_code=ctx.gt_code,
            db_id=ctx.db_id,
            model=ctx.user_text_model or ctx.model,
            gt_chart_path=ctx.gt_chart_path,
        )
        ctx.path_qa_history.append((question, answer))
        logger.info(f"[ask_user_text] Q: {question!r} | A: {answer!r}")
        return answer
    except Exception as e:
        logger.error(f"[ask_user_text] error: {e}")
        return "I'm not sure, use your best judgment."


def _impl_ask_user_vis(ctx: ToolContext, code: str) -> str:
    if ctx.vis_budget_remaining <= 0:
        return "BUDGET EXCEEDED: No visual-feedback budget remaining."
    if not code or not code.strip():
        return "ERROR: No code provided to render."
    ctx.vis_budget_remaining -= 1
    try:
        result: VisualFeedback = _ask_user_vis(
            key_features=ctx.key_features,
            gt_code=ctx.gt_code,
            ask_code=code,
            db_id=ctx.db_id,
            model=ctx.user_vis_model or ctx.model,
            gt_chart_path=ctx.gt_chart_path,
        )
        ctx.vis_scores.append(result.score)
        logger.info(f"[ask_user_vis] feedback={result.feedback!r}, score={result.score}")
        return f"[Score: {result.score}/10]\n\n{result.feedback}"
    except Exception as e:
        logger.error(f"[ask_user_vis] error: {e}")
        return "I can't see the chart clearly, please try again."


def _impl_finish(ctx: ToolContext) -> str:
    code = ctx.last_altair_code
    if not code or not code.strip():
        return "FINISH FAILED: No prior successful execute_altair code found. Call execute_altair first."
    try:
        success, chart, _, error = execute_visualization_code(code, ctx.db_id)
        if not success or chart is None:
            return f"FINISH FAILED: Code execution error: {error}"
        img_path = ctx.next_image_path()
        saved, reason = save_chart_as_image(chart, img_path)
        if saved and Path(img_path).exists():
            ctx.generated_images.append(img_path)
            return f"FINISH: Code validated and chart rendered. image_url={img_path}"
        return f"FINISH FAILED: Code ran but chart image could not be saved. {reason}"
    except Exception as e:
        return f"FINISH FAILED: {e}"


# ── ToolRegistry ──────────────────────────────────────────────────────────────

class ToolRegistry:
    """Bind tool schema and execution logic to a ToolContext instance.

    Parameters
    ----------
    ctx : ToolContext
        Runtime context for tool execution.
    exclude : set[str] | None
        Set of tool names to exclude, used by Agent to prune the tool set as needed.
    """

    def __init__(self, ctx: ToolContext, exclude: Optional[set] = None) -> None:
        self.ctx = ctx
        self._finished: bool = False
        self._dispatch_table: Dict[str, Any] = {
            "execute_sql": _impl_execute_sql,
            "execute_altair": _impl_execute_altair,
            "ask_user_text": _impl_ask_user_text,
            "ask_user_vis": _impl_ask_user_vis,
            "finish": self._finish_wrapper,
        }

        exclude = exclude or set()
        self.schemas: List[Dict[str, Any]] = [
            s for s in TOOL_SCHEMAS
            if s["function"]["name"] not in exclude
        ]
        for name in exclude:
            self._dispatch_table.pop(name, None)

    def register_tool(
        self, name: str, schema: Dict[str, Any], handler,
    ) -> None:
        """Register or override a tool (schema + handler)."""
        self._dispatch_table[name] = handler
        self.schemas = [
            s for s in self.schemas
            if s["function"]["name"] != name
        ]
        self.schemas.append(schema)

    def dispatch(self, tool_call: Any) -> str:
        name: str = tool_call.function.name
        try:
            kwargs: Dict[str, Any] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"ERROR: Cannot parse arguments for '{name}': {e}"

        handler = self._dispatch_table.get(name)
        if handler is None:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self._dispatch_table)}"

        try:
            return handler(self.ctx, **kwargs)
        except TypeError as e:
            return f"ERROR: Wrong arguments for tool '{name}': {e}"
        except Exception as e:
            logger.exception(f"[ToolRegistry] tool '{name}' raised an exception")
            return f"ERROR: Tool '{name}' raised: {e}"

    def make_tool_message(self, tool_call: Any) -> Dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": self.dispatch(tool_call),
        }

    @property
    def is_finished(self) -> bool:
        return self._finished

    def _finish_wrapper(self, ctx: ToolContext, **kwargs) -> str:
        self._finished = True
        return _impl_finish(ctx)
