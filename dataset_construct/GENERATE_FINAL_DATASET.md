# Generate Final Dataset Guide

## 📋 Overview

The script `generate_final_dataset.py` has been created to produce the final dataset based on review results.

## 🚀 Usage

### Run from the terminal

```bash
cd dataset_construct
python generate_final_dataset.py
```

### Expected behavior

The script will:
1. Load `output/samples.json` (original sample data)
2. Load `review_status.json` (review results)
3. Filter out all samples marked as `"pass"`
4. Group by question ID and renumber them
5. Save the final dataset to `output/final_dataset.json`

## 📊 Data format

### Output fields

The final dataset contains only the following fields:

```json
{
  "sample_id": "q100_c0",
  "db_id": "debit_card_specializing",
  "chart_type": "area",
  "chart_category": "composition",
  "initial_question": "Original vis_question content...",
  "ground_truth_code": "Full altair_code...",
  "key_features": [...]
}
```

### Field descriptions

| Field | Description |
|------|------|
| `sample_id` | Renumbered ID (`q{question_id}_c{index}`) |
| `db_id` | Database ID |
| `chart_type` | Chart type (extracted from `key_features`) |
| `chart_category` | Chart category (inferred from `key_features`) |
| `initial_question` | Original question (i.e. `vis_question`) |
| `ground_truth_code` | Ground-truth code (i.e. `altair_code`) |
| `key_features` | List of key features |

## 🔢 Renumbering logic

### Example

Original `sample_id` → review result → final `sample_id`

```
q1473_c0 → pass → q1473_c0
q1473_c1 → fail → (removed)
q1473_c2 → pass → q1473_c1  ← renumbered!
```

### Rules

1. Group by `question_id`
2. Keep only samples that passed review
3. Renumber from `c0` within each question
4. Keep `question_id` unchanged
5. Sort by numeric `question_id`

## 📈 Statistics

The script will print detailed statistics:

- Number of passed / failed samples
- Counts by database
- Counts by chart type
- Counts by chart category
- Distribution of candidate count per question

## 📁 Output file

### Location
```
dataset_construct/output/final_dataset.json
```

### Size
The file size depends on how many samples passed the review.

## ⚠️ Notes

### Required files

Make sure the following files exist before running:
- `output/samples.json` ✓ (already exists)
- `review_status.json` ✓ (already exists, since you have completed review)

### `chart_type` and `chart_category`

- If these two fields exist in the original data, they are used directly
- Otherwise, they are derived from `key_features`:
  - `chart_type`: extracted from the `mark`-type feature
  - `chart_category`: inferred from feature types

## 🔍 Verification

After running, check:

```bash
# Show sample count
python -c "import json; data = json.load(open('output/final_dataset.json')); print(f'Total samples: {len(data)}')"

# Show the first sample
python -c "import json; data = json.load(open('output/final_dataset.json')); print(json.dumps(data[0], indent=2, ensure_ascii=False))"

# Verify required fields are present
python -c "import json; data = json.load(open('output/final_dataset.json')); required_fields = ['sample_id', 'db_id', 'chart_type', 'chart_category', 'initial_question', 'ground_truth_code', 'key_features']; sample = data[0]; missing = [f for f in required_fields if f not in sample]; print('All fields present!' if not missing else f'Missing: {missing}')"
```

## 🐛 Troubleshooting

### Issue 1: File not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'output/samples.json'
```
**Fix**: Make sure you run the script from the `dataset_construct` directory.

### Issue 2: JSON parsing error
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
**Fix**: Check that the format of `review_status.json` is correct.

### Issue 3: No samples passed
```
Filtered 0 passed samples
```
**Fix**: Verify that `review_status.json` contains records with `"status": "pass"`.

## 📊 Example output

```
============================================================
Generate final dataset
============================================================
2025-12-19 20:00:00,000 - INFO - Loading sample data: output/samples.json
2025-12-19 20:00:05,000 - INFO - Total samples: 1350
2025-12-19 20:00:05,000 - INFO - Loading review status: review_status.json
2025-12-19 20:00:05,000 - INFO - Reviewed: 1350 samples
2025-12-19 20:00:05,000 - INFO - Passed: 850
2025-12-19 20:00:05,000 - INFO - Failed: 500
2025-12-19 20:00:05,000 - INFO - Generating final dataset...
2025-12-19 20:00:05,000 - INFO - Filtered 850 passed samples
2025-12-19 20:00:05,000 - INFO - Covering 450 distinct questions
2025-12-19 20:00:05,000 - INFO - Final dataset contains 850 samples
============================================================
Dataset statistics
============================================================

By database:
  debit_card_specializing: 120 samples
  california_schools: 95 samples
  ...

By chart type:
  bar: 200 samples
  line: 150 samples
  ...

By chart category:
  composition: 300 samples
  transform: 250 samples
  ...

Candidate count distribution per question:
  1 candidate:  200 questions
  2 candidates: 180 questions
  3 candidates: 70 questions
============================================================

Saving final dataset to: output/final_dataset.json
============================================================
✅ Final dataset generation complete!
- Output file: output/final_dataset.json
- Sample count: 850
============================================================
```

## 🎯 Next steps

After generating the final dataset, you can:

1. **Verify data quality**
   ```bash
   python -c "import json; data = json.load(open('output/final_dataset.json')); print(f'✅ Successfully loaded {len(data)} samples')"
   ```

2. **Inspect statistics**
   - Check sample distribution across databases
   - Review chart type distribution
   - Analyze candidate counts

3. **Use the dataset**
   - Train models
   - Perform data analysis
   - Export to other formats

---

**Created**: 2025-12-19  
**Script location**: `dataset_construct/generate_final_dataset.py`  
**Output location**: `dataset_construct/output/final_dataset.json`
