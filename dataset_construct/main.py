#!/usr/bin/env python3
"""Vis-Interact Dataset Construction - CLI (BIRD Mini-Dev / SQLite)

Usage:
    python main.py generate --max-instances 10 --num-candidates 5
    python main.py list-charts
    python main.py list-instances --limit 30
    python main.py validate samples.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_path():
    """Add this directory to sys.path for imports."""
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> Path:
    """Configure logging for the application."""
    from core.config import get_config

    level = logging.DEBUG if verbose else logging.INFO

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["VIS_INTERACT_RUN_ID"] = run_id

    if log_file is None:
        config = get_config()
        base_log_dir = config.paths.logs_dir
        
        # Use hierarchical structure: logs/{run_id}/app.log
        run_log_dir = base_log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_log_dir / "app.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    os.environ["VIS_INTERACT_APP_LOG_PATH"] = str(log_file)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    # LLM I/O logs are now candidate-specific (created inside candidate_logging_context)
    # No run-level llm_io file is created

    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    return log_file


setup_path()

from core.config import init_config, get_config
from utils.logging_context import install_candidate_filter_to_logger

logger = logging.getLogger(__name__)

# Install candidate context filter to root logger for automatic prefixing
install_candidate_filter_to_logger(logging.getLogger())


def cmd_generate(args) -> int:
    from pipeline import create_pipeline

    project_root = Path(args.project_root) if args.project_root else None
    pipeline = create_pipeline(project_root)
    config = get_config()

    # Run pipeline (automatically resumes from samples_full.json)
    results = pipeline.run(
        max_instances=args.max_instances,
        num_candidates=args.num_candidates,
        db_filter=args.db_id,
        difficulty_filter=args.difficulty,
        question_id=args.question_id,  # Single question_id for debugging
        show_progress=True,
        parallel=args.parallel
    )

    out_dir = config.paths.output_dir
    samples_path = out_dir / "samples.json"
    samples_full_path = out_dir / "samples_full.json"
    statistic_path = out_dir / "dataset_statistic.json"

    # Final outputs are already saved by pipeline._save_final_outputs()
    # But we can log the paths for user reference
    logger.info(f"Output files:")
    logger.info(f"  - samples.json: {samples_path}")
    logger.info(f"  - samples_full.json: {samples_full_path}")
    logger.info(f"  - dataset_statistic.json: {statistic_path}")
    
    # Count total samples from samples_full.json
    total_samples = 0
    if samples_full_path.exists():
        try:
            with open(samples_full_path, "r", encoding="utf-8") as f:
                samples_data = json.load(f)
                total_samples = len(samples_data) if isinstance(samples_data, list) else 0
        except Exception as e:
            logger.warning(f"Failed to count samples: {e}")
    
    logger.info(f"Total samples in dataset: {total_samples}")
    return 0


def cmd_list_charts(args) -> int:
    from utils.chart_contracts import get_contract_manager

    project_root = Path(args.project_root) if args.project_root else None
    if project_root:
        init_config(project_root)

    config = get_config()
    chart_example_dir = config.paths.chart_example_dir
    if not chart_example_dir.exists():
        logger.error(f"chart_example directory not found at {chart_example_dir}")
        return 1

    manager = get_contract_manager()
    categories = manager.get_chart_categories()

    print("=" * 60)
    print("Available Chart Types (from chart_example)")
    print("=" * 60)

    for cat, types in sorted(categories.items()):
        print(f"\n{cat} ({len(types)} types)")
        for ct in sorted(types):
            print(f"  - {ct}")

    total = sum(len(t) for t in categories.values())
    print(f"\nTotal: {total} chart types in {len(categories)} categories")
    return 0


def cmd_list_instances(args) -> int:
    project_root = Path(args.project_root) if args.project_root else None
    if project_root:
        init_config(project_root)

    config = get_config()
    data_path = config.paths.minidev_json
    if not data_path.exists():
        logger.error(f"mini_dev_sqlite.json not found at {data_path}")
        return 1

    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    limit = args.limit
    logger.info("=" * 60)
    logger.info(f"BIRD Mini-Dev instances (showing first {limit})")
    logger.info("=" * 60)

    for i, item in enumerate(items[:limit]):
        qid = item.get("question_id")
        db_id = item.get("db_id")
        diff = item.get("difficulty")
        q = (item.get("question") or "").strip().replace("\n", " ")
        logger.info(f"{i+1}. qid={qid} db_id={db_id} difficulty={diff}")
        logger.info(f"   {q[:160]}")

    logger.info(f"Total instances: {len(items)}")
    return 0


def cmd_validate(args) -> int:
    sample_path = Path(args.sample_file)
    if not sample_path.exists():
        logger.error(f"File not found: {sample_path}")
        return 1

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else [data]

    required_fields = [
        "sample_id",
        "source_question_id",
        "candidate_id",
        "vis_question",
        "vis_sql",
        "db_id",
        "data",
        "altair_code",
        "vega_lite_spec",
    ]

    failed = 0
    for i, s in enumerate(samples):
        missing = [k for k in required_fields if k not in s]
        if missing:
            failed += 1
            logger.info(f"Sample {i+1}: MISSING fields: {missing}")

    if failed == 0:
        logger.info(f"All {len(samples)} sample(s) look structurally valid.")
        return 0

    logger.info(f"{failed}/{len(samples)} sample(s) failed structural validation.")
    return 1


def main() -> int:
    # Load default config for argument defaults
    from core.config import get_config
    default_config = get_config()

    parser = argparse.ArgumentParser(description="Vis-Interact Dataset Construction (BIRD Mini-Dev / SQLite)")

    parser.add_argument("--project-root", type=str, default=None, help="Path to project root (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    gen = subparsers.add_parser("generate", help="Generate samples")
    gen.add_argument("--max-instances", "-n", type=int, default=5000, help="Number of BIRD instances to process")
    
    # Use config default for num_candidates
    gen.add_argument("--num-candidates", type=int, default=default_config.pipeline.num_candidates, 
                    help=f"Candidates per instance (default: {default_config.pipeline.num_candidates})")
    
    gen.add_argument("--db-id", type=str, default=None, help="Filter by db_id")
    gen.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty (simple/moderate/challenging)")
    gen.add_argument("--question-id", "-q", type=int, default=None, help="Process only this specific question_id (for debugging)")
    
    # Parallel processing arguments
    gen.add_argument("--parallel", action="store_true", default=default_config.pipeline.enable_parallel,
                    help=f"Enable parallel processing (default: {default_config.pipeline.enable_parallel})")
    gen.add_argument("--no-parallel", action="store_false", dest="parallel", help="Disable parallel processing")

    subparsers.add_parser("list-charts", help="List chart types (from chart_example)")

    li = subparsers.add_parser("list-instances", help="List BIRD instances")
    li.add_argument("--limit", "-l", type=int, default=9999999999999999, help="Number of instances to show")

    val = subparsers.add_parser("validate", help="Validate a generated samples.json")
    val.add_argument("sample_file", type=str, help="Path to samples.json")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "list-charts":
        return cmd_list_charts(args)
    if args.command == "list-instances":
        return cmd_list_instances(args)
    if args.command == "validate":
        return cmd_validate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
