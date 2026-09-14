"""
Evaluation Runner - Performs batch evaluations of existing run results.


How to use:
    from evaluate import EvaluatorRunner
    runner = EvaluatorRunner(eval_model="qwen-plus")
    results = runner.evaluate(result_dir="results/pomdp_run", save_dir="results/pomdp_run")
"""

from __future__ import annotations

import json
import time
import logging
import threading
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import main
from vis_interact.config import settings
from vis_interact.evaluation.evaluator import Evaluator
from vis_interact.evaluation.models import (
    KeyFeatureDetail,
    EvalSampleResult,
    EvalResults,
)

LOG_FMT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def _setup_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        console = logging.StreamHandler()
        console.setFormatter(LOG_FMT)
        log.addHandler(console)
    return log

logger = _setup_logger(__name__)
_setup_logger("vis_interact.evaluation")
_setup_logger("vis_interact.evaluation.evaluator")
_setup_logger("vis_interact.utils")


# ── Logging context ──────────────────────────────────────────────────────────────

class _ThreadFilter(logging.Filter):
    def __init__(self, tid: int):
        super().__init__()
        self.tid = tid

    def filter(self, record):
        return threading.current_thread().ident == self.tid


class EvalSampleLogContext:
    """Writes evaluation logs to sample folder, supports concurrent thread isolation."""

    LOGGER_NAMES = ("__main__", "evaluate", "vis_interact.evaluation", "vis_interact.utils")

    def __init__(self, sample_dir: Path, sample_id: str):
        self.log_path = sample_dir / f"eval_{sample_id}.log"
        self._handles: list = []

    def __enter__(self):
        tid = threading.current_thread().ident
        filt = _ThreadFilter(tid)
        for name in self.LOGGER_NAMES:
            log = logging.getLogger(name)
            fh = logging.FileHandler(str(self.log_path), mode="w", encoding="utf-8")
            fh.setFormatter(LOG_FMT)
            fh.setLevel(logging.INFO)
            fh.addFilter(filt)
            log.addHandler(fh)
            self._handles.append((log, fh))
        return self

    def __exit__(self, *exc):
        for log, fh in self._handles:
            log.removeHandler(fh)
            fh.close()
        return False


# ── Core evaluation logic ──────────────────────────────────────────────────────────────

def _do_eval_core(
    sample: dict,
    sample_id: str,
    result_path: Path,
    evaluator: Evaluator,
    skip_chart_judge: bool,
) -> tuple:
    """Single sample evaluation core logic, independent of EvaluatorRunner instance."""
    start_time = time.time()

    pred_image_path: Optional[str] = None
    pred_png = result_path / sample_id / "pred.png"
    if pred_png.exists():
        pred_image_path = str(pred_png)

    gt_chart_path = Path("VisInteractBench/gt_chart/png") / f"{sample_id}.png"
    gt_image_path = str(gt_chart_path) if gt_chart_path.exists() else None

    key_features = sample.get("key_features", [])
    kf_must_total = sum(1 for kf in key_features if kf.get("must", True))
    empty_result = EvalSampleResult(
        sample_id=sample_id,
        kf_total=len(key_features),
        kf_must_total=kf_must_total,
        all_eval_success=True,
        eval_time=0.0,
    )

    if not sample.get("code_renderable", False) or not pred_image_path or not gt_image_path:
        logging.getLogger(__name__).info(f"[{sample_id}] Skipping evaluation (code not executable or missing images)")
        return empty_result, True

    try:
        eval_result = evaluator.evaluate(
            sample=sample,
            pred_image_path=pred_image_path,
            gt_image_path=gt_image_path,
            skip_chart_judge=skip_chart_judge,
        )

        kf_details = []
        for kf_result in eval_result.key_feature_results:
            code_j = kf_result.code_judgment
            chart_j = kf_result.chart_judgment
            kf_details.append(KeyFeatureDetail(
                feature_id=kf_result.feature_id,
                feature_text=kf_result.feature_text,
                feature_type=kf_result.feature_type,
                is_must=kf_result.is_must,
                code_satisfied=code_j.satisfied if code_j else False,
                code_confidence=code_j.confidence if code_j else 0.0,
                code_reasoning=code_j.reasoning if code_j else "",
                code_eval_success=code_j.eval_success if code_j else False,
                chart_satisfied=chart_j.satisfied if chart_j else False,
                chart_confidence=chart_j.confidence if chart_j else 0.0,
                chart_reasoning=chart_j.reasoning if chart_j else "",
                chart_eval_success=chart_j.eval_success if chart_j else False,
                merge_satisfied=kf_result.satisfied_merge,
            ))

        all_eval_success = (
            len(eval_result.code_eval_failed_kf_ids) == 0
            and len(eval_result.chart_eval_failed_kf_ids) == 0
        )

        return EvalSampleResult(
            sample_id=sample_id,
            kf_total=eval_result.kf_total,
            kf_must_total=eval_result.kf_must_total,
            code_kf_pass_count=eval_result.code_kf_pass_count,
            code_must_pass_count=eval_result.code_must_pass_count,
            code_kf_pass_rate=eval_result.code_kf_pass_rate,
            code_must_pass_rate=eval_result.code_must_pass_rate,
            code_strict_success=eval_result.code_strict_success,
            chart_kf_pass_count=eval_result.chart_kf_pass_count,
            chart_must_pass_count=eval_result.chart_must_pass_count,
            chart_kf_pass_rate=eval_result.chart_kf_pass_rate,
            chart_must_pass_rate=eval_result.chart_must_pass_rate,
            chart_strict_success=eval_result.chart_strict_success,
            merge_kf_pass_count=eval_result.merge_kf_pass_count,
            merge_must_pass_count=eval_result.merge_must_pass_count,
            merge_kf_pass_rate=eval_result.merge_kf_pass_rate,
            merge_must_pass_rate=eval_result.merge_must_pass_rate,
            merge_strict_success=eval_result.merge_strict_success,
            all_eval_success=all_eval_success,
            code_eval_failed_kf_ids=eval_result.code_eval_failed_kf_ids,
            chart_eval_failed_kf_ids=eval_result.chart_eval_failed_kf_ids,
            key_feature_details=kf_details,
            eval_time=time.time() - start_time,
        ), all_eval_success

    except Exception as e:
        logging.getLogger(__name__).error(f"[{sample_id}] Evaluation failed: {e}")
        return EvalSampleResult(
            sample_id=sample_id,
            kf_total=len(key_features),
            kf_must_total=kf_must_total,
            all_eval_success=False,
            eval_time=0.0,
        ), False


# ── EvaluatorRunner ──────────────────────────────────────────────────────────────────

class EvaluatorRunner:
    """Evaluation runner - Performs batch evaluations of existing run results (supports serial/concurrent, incremental recovery)."""

    def __init__(
        self,
        eval_model: str = "",
        skip_chart_judge: bool = False,
        max_workers: int = 1,
        force: bool = False,
    ):
        self.eval_model = eval_model or settings.models.eval_model
        self.skip_chart_judge = skip_chart_judge
        self.max_workers = max_workers
        self.force = force
        self._results_lock = threading.Lock()
        self._global_log_handler: Optional[logging.Handler] = None
        logger.info(
            f"Initializing EvaluatorRunner: eval_model={self.eval_model}, "
            f"skip_chart_judge={skip_chart_judge}, max_workers={max_workers}"
        )

    def _create_evaluator(self) -> Evaluator:
        return Evaluator(
            code_judge_model=self.eval_model,
            chart_judge_model=self.eval_model,
        )

    # ── Internal aggregation and saving ──────────────────────────────────────────────────────

    def _save_intermediate(
        self,
        eval_results_path: Path,
        track: str,
        model: str,
        result_path: Path,
        eval_sample_results: List[EvalSampleResult],
        evaluated_count: int,
        total_samples: int,
        code_renderable_count: int = 0,
    ) -> EvalResults:
        eval_failed_ids = [r.sample_id for r in eval_sample_results if not r.all_eval_success]
        valid = [r for r in eval_sample_results if r.all_eval_success]

        kf_total = sum(r.kf_total for r in valid)
        kf_must_total = sum(r.kf_must_total for r in valid)

        def _rate(num, den):
            return num / den if den > 0 else 0.0

        code_kf = sum(r.code_kf_pass_count for r in valid)
        code_must = sum(r.code_must_pass_count for r in valid)
        code_strict = sum(1 for r in valid if r.code_strict_success)

        chart_kf = sum(r.chart_kf_pass_count for r in valid)
        chart_must = sum(r.chart_must_pass_count for r in valid)
        chart_strict = sum(1 for r in valid if r.chart_strict_success)

        merge_kf = sum(r.merge_kf_pass_count for r in valid)
        merge_must = sum(r.merge_must_pass_count for r in valid)
        merge_strict = sum(1 for r in valid if r.merge_strict_success)

        eval_results = EvalResults(
            track=track,
            model=model,
            eval_model=self.eval_model,
            timestamp=datetime.now().isoformat(),
            result_dir=str(result_path),
            total_samples=total_samples,
            code_renderable_count=code_renderable_count,
            code_renderable_rate=_rate(code_renderable_count, total_samples),
            evaluated_samples=evaluated_count,
            valid_samples=len(valid),
            valid_kf_total=kf_total,
            valid_kf_must_total=kf_must_total,
            code_dataset_score=_rate(code_kf, kf_total),
            code_must_dataset_score=_rate(code_must, kf_must_total),
            code_strict_success_rate=_rate(code_strict, evaluated_count),
            chart_dataset_score=_rate(chart_kf, kf_total),
            chart_must_dataset_score=_rate(chart_must, kf_must_total),
            chart_strict_success_rate=_rate(chart_strict, evaluated_count),
            merge_dataset_score=_rate(merge_kf, kf_total),
            merge_must_dataset_score=_rate(merge_must, kf_must_total),
            merge_strict_success_rate=_rate(merge_strict, evaluated_count),
            eval_failed_sample_ids=eval_failed_ids,
            sample_results=eval_sample_results,
        )

        with self._results_lock:
            tmp = eval_results_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(eval_results), f, indent=2, ensure_ascii=False, default=str)
            tmp.replace(eval_results_path)

        return eval_results

    # ── Single sample evaluation ──────────────────────────────────────────────────────────

    def _evaluate_single_sample(
        self,
        sample: dict,
        result_path: Path,
        evaluator: Evaluator,
    ) -> tuple:
        sample_id = sample["sample_id"]
        sample_dir = result_path / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        with EvalSampleLogContext(sample_dir, sample_id):
            return self._do_evaluate(sample, sample_id, result_path, evaluator)

    def _do_evaluate(
        self,
        sample: dict,
        sample_id: str,
        result_path: Path,
        evaluator: Evaluator,
    ) -> tuple:
        return _do_eval_core(sample, sample_id, result_path, evaluator, self.skip_chart_judge)

    # ── Public entry ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        result_dir: str,
        save_dir: str,
        sample_ids: Optional[List[str]] = None,
    ) -> EvalResults:
        """
        Batch evaluate existing run results.

        Args:
            result_dir: Run result directory (contains run_results.json and sample subdirectories)
            save_dir: Evaluation result save directory
            sample_ids: Specify the sample IDs to evaluate (None means all)

        Returns:
            EvalResults object
        """
        result_path = Path(result_dir).expanduser().resolve()
        save_path = Path(save_dir).expanduser().resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        global_log_path = result_path / "eval_log.log"
        global_fh = logging.FileHandler(str(global_log_path), encoding="utf-8")
        global_fh.setFormatter(LOG_FMT)
        global_fh.setLevel(logging.INFO)
        for name in EvalSampleLogContext.LOGGER_NAMES:
            logging.getLogger(name).addHandler(global_fh)
        self._global_log_handler = global_fh

        try:
            return self._run_evaluate(result_path, save_path, sample_ids)
        finally:
            self._global_log_handler = None
            for name in EvalSampleLogContext.LOGGER_NAMES:
                logging.getLogger(name).removeHandler(global_fh)
            global_fh.close()

    def _run_evaluate(
        self,
        result_path: Path,
        save_path: Path,
        sample_ids: Optional[List[str]],
    ) -> EvalResults:
        run_results_path = result_path / "run_results.json"
        if not run_results_path.exists():
            raise FileNotFoundError(f"Run result file not found: {run_results_path}")

        with open(run_results_path, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        track = run_data["track"]
        model = run_data["model"]
        samples = run_data["samples"]

        dataset_path = Path(settings.paths.dataset_path)
        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            dataset_map = {s["sample_id"]: s for s in dataset}
            for sample in samples:
                ds_entry = dataset_map.get(sample["sample_id"], {})
                if "key_features" not in sample and "key_features" in ds_entry:
                    sample["key_features"] = ds_entry["key_features"]
                if "ground_truth_code" not in sample and "ground_truth_code" in ds_entry:
                    sample["ground_truth_code"] = ds_entry["ground_truth_code"]
            logger.info(f"Key features, etc. fields have been supplemented from dataset {dataset_path}")
        else:
            logger.warning(f"Dataset file not found: {dataset_path}, using fields from run results")

        all_executable = sum(1 for s in samples if s.get("code_renderable", False))
        total_count = len(samples)

        if sample_ids:
            id_set = set(sample_ids)
            samples = [s for s in samples if s["sample_id"] in id_set]
            logger.info(f"Specified {len(sample_ids)} samples, matched {len(samples)} samples")

        eval_results_path = save_path / f"eval_results_{self.eval_model}.json"

        completed_results: dict = {}
        if eval_results_path.exists() and not self.force:
            try:
                with open(eval_results_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for sr in existing.get("sample_results", []):
                    if sr.get("all_eval_success"):
                        completed_results[sr["sample_id"]] = EvalSampleResult(**{
                            k: v for k, v in sr.items()
                            if k in EvalSampleResult.__dataclass_fields__
                        })
                logger.info(f"Restored {len(completed_results)} completed evaluations")
            except Exception as e:
                logger.warning(f"Failed to load existing evaluation results: {e}")

        if self.force:
            completed_results = {}
        elif sample_ids:
            force_ids = {s["sample_id"] for s in samples}
            for sid in force_ids:
                completed_results.pop(sid, None)

        pending = [s for s in samples if s["sample_id"] not in completed_results]
        eval_sample_results = list(completed_results.values())
        evaluated_count = len(eval_sample_results)

        logger.info(
            f"Starting evaluation: total {len(samples)} samples, completed {len(completed_results)},"
            f"pending {len(pending)} samples, concurrent workers: {self.max_workers}"
        )

        eval_results: EvalResults

        if not pending:
            logger.info("All samples have been evaluated")
            eval_results = self._save_intermediate(
                eval_results_path, track, model, result_path,
                eval_sample_results, evaluated_count, total_count, all_executable,
            )
        elif self.max_workers <= 1:
            evaluator = self._create_evaluator()
            for sample in tqdm(pending, desc="Evaluating"):
                sample_id = sample["sample_id"]
                eval_result, is_ok = self._evaluate_single_sample(sample, result_path, evaluator)
                eval_sample_results.append(eval_result)
                if is_ok:
                    evaluated_count += 1
                eval_results = self._save_intermediate(
                    eval_results_path, track, model, result_path,
                    eval_sample_results, evaluated_count, total_count, all_executable,
                )
                logger.info(f"[{sample_id}] Saved ({len(eval_sample_results)}/{len(samples)})")
        else:
            results_dict: dict = {}
            success_dict: dict = {}
            evaluator = self._create_evaluator()

            logger.info(
                f"Using ThreadPoolExecutor for concurrent evaluation, workers={self.max_workers}"
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_sample = {
                    executor.submit(
                        self._evaluate_single_sample,
                        sample, result_path, evaluator,
                    ): (idx, sample["sample_id"])
                    for idx, sample in enumerate(pending)
                }

                with tqdm(total=len(pending), desc="Concurrent evaluation") as pbar:
                    for future in as_completed(future_to_sample):
                        idx, sample_id = future_to_sample[future]
                        try:
                            res, is_ok = future.result()
                            results_dict[idx] = res
                            success_dict[idx] = is_ok
                        except Exception as e:
                            logger.error(f"Evaluation failed for sample {sample_id}: {e}")
                            key_features = pending[idx].get("key_features", [])
                            results_dict[idx] = EvalSampleResult(
                                sample_id=sample_id,
                                kf_total=len(key_features),
                                kf_must_total=sum(1 for kf in key_features if kf.get("must", True)),
                                all_eval_success=False,
                                eval_time=0.0,
                            )
                            success_dict[idx] = False

                        pbar.update(1)
                        logger.info(
                            f"[{sample_id}] Evaluation completed "
                            f"({len(results_dict)}/{len(pending)})"
                        )
                        new_results = [results_dict[i] for i in sorted(results_dict)]
                        merged = list(completed_results.values()) + new_results
                        ev_cnt = (
                            len(completed_results)
                            + sum(1 for i in sorted(success_dict) if success_dict[i])
                        )
                        eval_results = self._save_intermediate(
                            eval_results_path, track, model, result_path,
                            merged, ev_cnt, total_count, all_executable,
                        )
                        eval_sample_results = merged

        logger.info(f"Evaluation results saved to: {eval_results_path}")
        self._print_summary(eval_results)
        return eval_results

    def _print_summary(self, result: EvalResults) -> None:
        total = result.total_samples
        if total == 0:
            print("No evaluation results")
            return
        print("\n" + "=" * 70)
        print(f"Evaluation summary - {result.track} Track")
        print("=" * 70)
        print(f"Running model: {result.model}   Evaluation model: {result.eval_model}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Result directory: {result.result_dir}")
        print("-" * 70)
        print(f"Total samples: {total}")
        print(f"Code renderable: {result.code_renderable_count}/{total} "
              f"({result.code_renderable_rate:.4f})")
        print(f"Evaluation successful: {result.evaluated_samples}   Evaluation failed: {len(result.eval_failed_sample_ids)}")
        print("-" * 70)
        print("Three-view evaluation metrics (only counting successful evaluations):")
        fmt = "  {:<12} {:<12.4f} {:<12.4f} {:<12.4f}"
        print(f"  {'View':<12} {'KF Score':<12} {'Must Score':<12} {'Strict Rate':<12}")
        print(fmt.format("Code-Level", result.code_dataset_score, result.code_must_dataset_score, result.code_strict_success_rate))
        print(fmt.format("Chart-Level", result.chart_dataset_score, result.chart_must_dataset_score, result.chart_strict_success_rate))
        print(fmt.format("Merge", result.merge_dataset_score, result.merge_must_dataset_score, result.merge_strict_success_rate))
        print("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch evaluate existing run results")
    parser.add_argument("--result-dir", type=str, default="results/qwen3.5-flash/vis_mcts_qwen3.5-flash_10r_20steps", help="Run result directory (contains run_results.json)")
    parser.add_argument("--save-dir", type=str, default="results/qwen3.5-flash/vis_mcts_qwen3.5-flash_10r_20steps", help="Evaluation result save directory (default same as result-dir)")
    parser.add_argument("--eval-model", type=str, default=None, help="Evaluation model name, default is eval_model in config.toml")
    parser.add_argument("--max-workers", type=int, default=64, help="Concurrent thread count (default 32)")
    parser.add_argument("--skip-chart-judge", action="store_true", help="Skip chart-level evaluation")
    parser.add_argument("--force", action="store_true", help="Ignore existing evaluation results, re-evaluate all")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Specify the sample IDs to evaluate")

    args = parser.parse_args()

    runner = EvaluatorRunner(
        eval_model=args.eval_model,
        skip_chart_judge=args.skip_chart_judge,
        max_workers=args.max_workers,
        force=args.force,
    )
    runner.evaluate(
        result_dir=args.result_dir,
        save_dir=args.save_dir or args.result_dir,
        sample_ids=args.sample_ids,
    )
