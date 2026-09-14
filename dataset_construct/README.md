# dataset_construct

`dataset_construct` is the subdirectory in this repository dedicated to "dataset construction". It is not the top-level README of the entire project, but rather the usage guide for this set of data construction scripts, configurations, database copies, and output files.

The goal of this directory is to convert the original SQL question-answer samples from BIRD Mini-Dev into data samples that can be used for visualization interaction evaluation.

## What's in this directory

```
dataset_construct/
├── main.py                    # CLI entry point
├── pipeline.py                # Main pipeline orchestration
├── core/                      # Configuration and data models
├── preprocessing/             # Data loading, schema parsing, SQL semantic analysis
├── generation/                # Candidate generation, SQL/NL rewriting, ambiguity injection, chart generation
├── execution/                 # SQLite execution logic
├── validation/                # Sample validation
├── utils/                     # Concurrency, logging, state management, LLM client, etc.
├── databases/                 # Local SQLite database copies used by this directory
├── output/                    # Generated results, statistics, logs, temporary files
├── prepare_review_data.py     # Data preparation script for review
├── test_review_system.py      # Test scripts for the review system
├── GENERATE_FINAL_DATASET.md  # Instructions for final dataset generation
├── requirements.txt
└── README.md
```

## Directory responsibilities

This directory is mainly responsible for the following tasks:

- Reading the original BIRD data and schema from `minidev/MINIDEV` under the project root.
- Executing and validating generated queries using the local SQLite files in `dataset_construct/databases`.
- Transforming original Q&A pairs into visualization tasks and generating multiple differentiated candidate samples.
- Producing result files such as `samples.json` and `samples_full.json` for subsequent review or final dataset assembly.

## External resources required

Although this is the README for `dataset_construct`, this directory still depends on several resources from the repository root at runtime:

- `minidev/MINIDEV/mini_dev_sqlite.json`
- `minidev/MINIDEV/dev_tables.json`
- `minidev/MINIDEV/dev_databases/`
- `chart_example/`

In other words, this directory is "a subsystem within the project", not a fully standalone Python package.

## Quick start

From the project root, enter this directory:

```bash
cd dataset_construct
pip install -r requirements.txt
```

Set environment variables and you're ready to run:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

```bash
export OPENAI_API_KEY="your-api-key"
```

Common optional environment variables:

| Variable | Description |
|------|------|
| `OPENAI_API_KEY` | API key required for text model calls |
| `OPENAI_BASE_URL` | Custom API endpoint |
| `VLM_API_KEY` | Vision model API key (can differ from text model) |
| `VLM_BASE_URL` | Vision model endpoint |
| `VIS_INTERACT_DEBUG` | Set to `1` or `true` to enable debug mode |

## Command-line entry

The CLI entry point of this directory is `main.py`:

```bash
cd dataset_construct
python main.py [global options] <command> [command options]
```

Global options:

| Option | Short | Description |
|------|------|------|
| `--project-root` |  | Explicitly specify the project root directory |
| `--verbose` | `-v` | Output more verbose logs |

### `generate`

Generate visualization samples.

```bash
python main.py generate [options]
```

Common parameters:

| Parameter | Short | Description |
|------|------|------|
| `--max-instances` | `-n` | Maximum number of original instances to process |
| `--num-candidates` |  | Number of candidates to generate per instance |
| `--db-id` |  | Filter by database |
| `--difficulty` |  | Filter by difficulty |
| `--resume` |  | Resume from interrupted state |
| `--no-resume` |  | Ignore existing resume state and start over |
| `--parallel` |  | Enable parallel processing |
| `--no-parallel` |  | Disable parallel processing |

Examples:

```bash
python main.py generate
python main.py generate -n 10
python main.py generate -q 1471 --num-candidates 3
python main.py -v generate --db-id california_schools --difficulty moderate
python main.py generate --resume
```

### `list-instances`

View processable original instances:

```bash
python main.py list-instances
python main.py list-instances -l 50
```

### `list-charts`

List the currently available chart types:

```bash
python main.py list-charts
```

### `validate`

Validate the format of an output sample file:

```bash
python main.py validate output/samples.json
```

## Most common workflows

### 1. Small-scale trial run

```bash
cd dataset_construct
python main.py list-instances -l 20
python main.py generate -n 3
python main.py validate output/samples.json
```

### 2. Formal generation

```bash
cd dataset_construct
python main.py generate -n 100 --num-candidates 5
```

### 3. Resume after interruption

```bash
cd dataset_construct
python main.py generate --resume
```

## Output files

Generation tasks typically write results into `output/`:

| File | Description |
|------|------|
| `output/samples.json` | Main sample output |
| `output/samples_full.json` | More complete sample output |
| `output/dataset_statistic.json` | Dataset statistics |
| `output/logs/` | Run logs |
| `output/temp_images/` | Temporary images from the validation stage |

Some files may only appear in certain run modes or specific pipeline versions, such as files related to checkpoint recovery.

## Parallelism and resume

The current pipeline supports parallel processing and interrupt recovery, suitable for long-running tasks:

- Semantic analysis and candidate generation are reused as much as possible to reduce duplicate calls.
- Candidate samples support parallel processing to improve overall throughput.
- Completed results are saved on interruption, making it easy to recover later via `--resume`.

Parallelism-related parameters live in `PipelineConfig` inside `core/config.py`.

## Data flow overview

```
Original BIRD instance
    -> schema / SQL semantic analysis
    -> candidate visualization plan generation
    -> SQL and question rewriting
    -> execute data processing code
    -> generate Altair / Vega-Lite
    -> inject controlled ambiguity
    -> validation
    -> write to output/
```

## Code structure

| Path | Purpose |
|------|------|
| `core/config.py` | Unified configuration and path management |
| `core/models.py` | Data model definitions |
| `preprocessing/loader.py` | Load original data |
| `preprocessing/schema.py` | Schema parsing |
| `preprocessing/sql_analyzer.py` | SQL semantic analysis |
| `generation/candidates.py` | Candidate generation and difference control |
| `generation/transform.py` | SQL/NL rewriting |
| `generation/spec.py` | Altair / spec generation |
| `generation/ambiguity.py` | Ambiguity injection |
| `execution/sqlite_client.py` | SQLite execution wrapper |
| `validation/validator.py` | Sample validation |
| `utils/state_manager.py` | Incremental save and resume |

## Other scripts

In addition to the main generation pipeline, this directory contains several auxiliary files:

- `prepare_review_data.py`: Prepares data for manual review or post-processing.
- `test_review_system.py`: Checks whether the review-related pipeline works.
- `GENERATE_FINAL_DATASET.md`: Explains the steps to assemble the final dataset.
- `run_generate_final.bat`: Helper launch script for Windows.

## Frequently asked questions

### Cannot find data files

First, verify that the following paths exist under the project root:

- `minidev/MINIDEV/mini_dev_sqlite.json`
- `minidev/MINIDEV/dev_tables.json`
- `minidev/MINIDEV/dev_databases/`

Also confirm that this directory contains:

- `databases/`

### LLM or VLM calls fail

Check the following first:

- Whether `OPENAI_API_KEY` / `VLM_API_KEY` are set correctly
- Whether `OPENAI_BASE_URL` / `VLM_BASE_URL` are reachable
- Whether the current concurrency configuration is too high

### SQL execution fails

Common things to investigate:

- Whether the target database file is intact
- Whether the rewritten SQL is compatible with SQLite
- Whether detailed errors are recorded in `output/logs/`

## Related documents

- Upper-level design notes: `../Dataset-Construction.md`
- Final dataset assembly guide: `GENERATE_FINAL_DATASET.md`

## One-line summary

If you only care about this directory itself, you can think of it as:

> `dataset_construct` = a complete subsystem that "reads original BIRD data -> generates visualization candidate samples -> validates and writes to output".
