"""
VisInteract MCTS Agent Runner

Usage:
    python main.py                          # Run all samples
    python main.py --max-samples 1         # Run only first 1 samples
    python main.py --max-samples 10 --workers 10 # Run only first 10 samples with 10 workers
    python main.py --samples q5_c1         # Run specific samples
    python main.py --output results/mcts    # Custom output directory
    python main.py --n-rollout 12 --b-vis 3 # Override MCTS params
"""

import argparse
import concurrent.futures
import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from vis_interact.agents.vis_mcts import MCTSAgent, AgentResult
from vis_interact.utils.code_exec import execute_visualization_code, save_chart_as_image
from vis_interact.utils.dataset import load_dataset

LOG_FMT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(LOG_FMT)
    logger.addHandler(_console)


GT_CHART_DIR = Path("VisInteractBench/gt_chart/png")


class _SampleLogFilter(logging.Filter):
    """Pass only log records whose message contains [sample_id]."""

    def __init__(self, sample_id: str):
        super().__init__()
        self._tag = f"[{sample_id}]"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._tag in record.getMessage()


def run_sample(agent: MCTSAgent, sample: dict, output_dir: Path) -> dict:
    sample_id = sample["sample_id"]
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    gt_src = GT_CHART_DIR / f"{sample_id}.png"
    if gt_src.exists():
        sample["gt_chart_path"] = str(gt_src)
        gt_dst = sample_dir / "gt.png"
        if not gt_dst.exists():
            shutil.copy2(str(gt_src), str(gt_dst))
            logger.info(f"[{sample_id}] GT chart copied -> {gt_dst}")

    log_path = sample_dir / f"{sample_id}.log"
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(LOG_FMT)
    fh.setLevel(logging.INFO)
    fh.addFilter(_SampleLogFilter(sample_id))
    logging.getLogger("vis_interact.agents.vis_mcts").addHandler(fh)

    start = time.time()
    try:
        result: AgentResult = agent.run(sample, sample_dir=sample_dir)
    except Exception as e:
        logger.error(f"[{sample_id}] Agent failed: {e}", exc_info=True)
        result = AgentResult(sample_id=sample_id, error_message=str(e))
    elapsed = time.time() - start

    logging.getLogger("vis_interact.agents.vis_mcts").removeHandler(fh)
    fh.close()

    code_renderable = False
    if result.final_code:
        code_path = sample_dir / "pred.py"
        code_path.write_text(result.final_code, encoding="utf-8")

        ok, chart, _, err = execute_visualization_code(result.final_code, sample["db_id"])
        if ok and chart is not None:
            img_path = str(sample_dir / "pred.png")
            saved, _reason = save_chart_as_image(chart, img_path)
            code_renderable = saved and Path(img_path).exists()

    summary = {
        "sample_id": sample_id,
        "db_id": sample["db_id"],
        "query": sample["initial_question"],
        "code_renderable": code_renderable,
        "best_score": result.best_score,
        "total_rollouts": result.total_rollouts,
        "vis_questions_asked": result.vis_questions_asked,
        "text_questions_asked": result.text_questions_asked,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "error": result.error_message,
    }

    (sample_dir / "result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run VisInteract MCTS Agent")
    parser.add_argument("--samples", nargs="*", default=None, help="Sample IDs to run (default: all)")
    parser.add_argument("--output", type=str, default="results/mcts_run", help="Output directory")
    parser.add_argument("--model", type=str, default="", help="LLM model name")
    parser.add_argument("--n-rollout", type=int, default=10, help="MCTS rollout count (bounded by b_vis)")
    parser.add_argument("--b-vis", type=int, default=10, help="Visual feedback budget (VLM evaluations)")
    parser.add_argument("--b-text", type=int, default=10, help="Text clarification budget")
    parser.add_argument("--max-steps", type=int, default=10, help="Global tool-execution step limit")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to run (default: all)")
    parser.add_argument("--temperature", type=float, default=0.8, help="LLM sampling temperature")
    parser.add_argument("--c-pw", type=float, default=2.0, help="Progressive widening constant")
    parser.add_argument("--alpha-pw", type=float, default=0.5, help="Progressive widening exponent")
    parser.add_argument("--max-children", type=int, default=3, help="Max children per node")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for sample-level concurrency")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    if args.samples:
        sample_ids = args.samples
    else:
        sample_ids = list(dataset.keys())

    samples = []
    for sid in sample_ids:
        if sid in dataset:
            samples.append(dataset[sid])
        else:
            logger.warning(f"Sample '{sid}' not found, skip")

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    logger.info(f"Running {len(samples)} samples, output -> {output_dir}")

    agent = MCTSAgent(
        model=args.model,
        b_vis=args.b_vis,
        b_text=args.b_text,
        n_rollout=args.n_rollout,
        max_steps=args.max_steps,
        temperature=args.temperature,
        c_pw=args.c_pw,
        alpha_pw=args.alpha_pw,
        max_children=args.max_children,
    )

    result_map: dict[str, dict] = {}
    pending = []

    for sample in samples:
        sid = sample["sample_id"]
        existing_result_path = output_dir / sid / "result.json"
        if existing_result_path.exists():
            try:
                cached = json.loads(existing_result_path.read_text(encoding="utf-8"))
                result_map[sid] = cached
                logger.info(f"Skipped (cached): {sid}")
                continue
            except Exception:
                logger.warning(f"Corrupt result for {sid}, will re-run")
        pending.append(sample)

    skipped = len(samples) - len(pending)

    def _log_done(sid, summary, done_count, total):
        score_str = f"score={summary['best_score']}" if summary["best_score"] else "no score"
        logger.info(
            f"[{done_count}/{total}] Done: {sid} "
            f"({score_str}, {summary['elapsed_sec']}s, "
            f"renderable={summary['code_renderable']})"
        )

    if pending:
        logger.info(f"{len(pending)} sample(s) to run, {args.workers} worker(s)")

        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_map = {
                    pool.submit(run_sample, agent, s, output_dir): s
                    for s in pending
                }
                done_count = 0
                for future in concurrent.futures.as_completed(future_map):
                    sample = future_map[future]
                    sid = sample["sample_id"]
                    done_count += 1
                    try:
                        summary = future.result()
                    except Exception as e:
                        logger.error(f"[{sid}] Worker failed: {e}", exc_info=True)
                        summary = {
                            "sample_id": sid, "db_id": sample["db_id"],
                            "query": sample["initial_question"],
                            "code_renderable": False, "best_score": None,
                            "total_rollouts": 0, "vis_questions_asked": 0,
                            "text_questions_asked": 0, "prompt_tokens": 0,
                            "completion_tokens": 0, "total_tokens": 0,
                            "elapsed_sec": 0, "error": str(e),
                        }
                    result_map[sid] = summary
                    _log_done(sid, summary, done_count, len(pending))
        else:
            for i, sample in enumerate(pending):
                sid = sample["sample_id"]
                logger.info(f"[{i+1}/{len(pending)}] Running {sid}...")
                summary = run_sample(agent, sample, output_dir)
                result_map[sid] = summary
                _log_done(sid, summary, i + 1, len(pending))

    if skipped:
        logger.info(f"Skipped {skipped} already-completed samples")

    all_results = [result_map[s["sample_id"]] for s in samples if s["sample_id"] in result_map]

    run_results = {
        "track": "mcts",
        "model": agent.model,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_rollout": args.n_rollout,
            "b_vis": args.b_vis,
            "b_text": args.b_text,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "c_pw": args.c_pw,
            "alpha_pw": args.alpha_pw,
            "max_children": args.max_children,
        },
        "total_samples": len(all_results),
        "code_renderable": sum(1 for r in all_results if r["code_renderable"]),
        "avg_score": _safe_avg([r["best_score"] for r in all_results if r["best_score"]]),
        "samples": all_results,
    }

    results_path = output_dir / "run_results.json"
    results_path.write_text(
        json.dumps(run_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Results saved to {results_path}")

    renderable = run_results["code_renderable"]
    total = run_results["total_samples"]
    avg = run_results["avg_score"]
    print(f"\n{'='*50}")
    print(f"MCTS Run Complete")
    print(f"{'='*50}")
    print(f"Samples: {total}")
    print(f"Renderable: {renderable}/{total} ({renderable/total:.1%})")
    print(f"Avg score: {avg:.1f}" if avg else "Avg score: N/A")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")


def _safe_avg(values):
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
