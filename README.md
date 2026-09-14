# VisInteract: Towards Dynamic Interactive Text-to-Visualization under Imperfect Queries

## Repository Layout

```
VisInteract/
├── main.py                    # MCTS agent runner
├── evaluate.py                # Dual-judge evaluator
├── vis_interact/              # Core library: agents, tools, prompts, evaluator, config
├── dataset_construct/         # Dataset construction pipeline (see its own README)
├── VisInteractBench_example/  # 3-sample example benchmark + SQLite DB
└── VisInteractBench/          # full benchmark (download from OneDrive, not in git)
```

## Installation

Requires **Python ≥ 3.11**.

```bash
conda create -n VisInteract python=3.11
conda activate VisInteract
pip install -e .
```



## Configuration

Create a `config.toml` at the repo root. Environment variables override TOML values.

```toml
[api.openai]
base_url = "https://api.openai.com/v1"
api_key  = ""

[api.qwen]
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key  = ""

[api.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key  = ""

[models]
default_model   = "qwen3.5-flash"   # the agent
user_text_model = "qwen3.5-flash"   # textual user simulator
user_vis_model  = "qwen3.5-flash"   # visual user simulator (VLM)
eval_model      = "qwen3.5-plus"    # dual-judge evaluator

[paths]
database_path = "VisInteractBench_example/databases"
dataset_path  = "VisInteractBench_example/test_example.json"
```



## VisInteractBench

The full benchmark (test set, SQLite databases, and ground-truth charts) is available on [OneDrive](https://1drv.ms/u/c/E720D72F2F8559F9/IQAqTVjjIooAQq_yLPoKBaJUAbWfYNsFuoOJycAyXlDMlmA?e=6xG4Fd). Download and extract it as `VisInteractBench/` at the repo root, then point `config.toml` at the full data:

```toml
[paths]
database_path = "VisInteractBench/databases"
dataset_path  = "VisInteractBench/test.json"
```

A 3-sample subset lives in `VisInteractBench_example/` so the pipeline can run without the full download.

## Quickstart

`VisInteractBench_example/` ships with 3 samples and a SQLite database, so the pipeline runs out of the box.

```bash
# Run the agent
python main.py --max-samples 3 --output results/example

# Evaluate the run with the dual judge (code-level + chart-level)
python evaluate.py --result-dir results/example --save-dir results/example
```

Per-sample artifacts (`pred.py`, `pred.png`, `result.json`, log) are written under `results/example/<sample_id>/`. The aggregate `run_results.json` is saved atomically, so re-running with the same `--output` resumes where it left off.

Common flags:


| Flag               | Default | Meaning                         |
| ------------------ | ------- | ------------------------------- |
| `--samples ID ...` | all     | Run only the listed sample ids. |
| `--max-samples N`  | none    | Cap samples for this run.       |
| `--n-rollout`      | 10      | MCTS rollouts per sample.       |
| `--workers`        | 1       | Sample-level concurrency.       |




## Data Format

Each sample in the dataset JSON looks like:

```json
{
  "sample_id": "q5_c1",
  "db_id": "california_schools",
  "initial_question": "...ambiguous user request...",
  "ground_truth_code": "import altair as alt\n...",
  "key_features": [
    { "id": "kf_mark_errorbar", "type": "mark", "text": "Use error bars ...", "must": true },
    { "id": "kf_composition_point", "type": "composition", "text": "...", "must": false }
  ]
}
```

Predictions are scored against `key_features` from two perspectives:

- **Code judge** (LLM) inspects the generated Python/Altair code.
- **Chart judge** (VLM) inspects the rendered PNG.

For each view the evaluator reports KF pass-rate, *must*-KF pass-rate, and a strict success rate (1 iff every required KF is satisfied).