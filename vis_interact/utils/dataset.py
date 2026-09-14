"""
Dataset and database tool

Provides dataset loading, sample retrieval, and database schema query functions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from vis_interact.config import settings

logger = logging.getLogger(__name__)

_dataset_cache: Optional[dict] = None

CODE_STRUCTURE = """
## Code Structure Requirements
**Important: Make sure to import all packages you need (e.g., numpy, scipy, etc.) at the beginning of your code.**

```python
import altair as alt
import sqlite3
import pandas as pd
# Import other packages as needed (e.g., import numpy as np, import scipy, etc.)

# Stage 1: SQL - Data retrieval
conn = sqlite3.connect('./databases/{db_id}.sqlite')
SQL = \"\"\"
-- Your SQL query here
\"\"\"
df = pd.read_sql_query(SQL, conn)
conn.close()

# Stage 2: Pandas Processing (if needed)

# Stage 3: Visualization
chart = alt.Chart(df).mark_xxx().encode(...).properties(...)

chart
```

**Important: The final chart object MUST be assigned to a variable named `chart`. If you layer or compose multiple charts, assign the final result to `chart`.**
Example: `chart = base_chart + text_labels`"""


def load_dataset(path: str = "") -> dict:
    """Load dataset and create index of sample_id -> sample."""
    global _dataset_cache
    if _dataset_cache is None:
        dataset_path = path or str(settings.paths.dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _dataset_cache = {item["sample_id"]: item for item in data}
        logger.info(f"Dataset loaded, {len(_dataset_cache)} samples")
    return _dataset_cache


def get_sample(sample_id: str) -> dict:
    """Get sample information by sample_id."""
    dataset = load_dataset()
    if sample_id not in dataset:
        raise ValueError(f"Sample ID '{sample_id}' not found in dataset")
    return dataset[sample_id]


def get_database_detailed_schema(db_id: str) -> str:
    """Get database schema text: each line `table_name(col1, col2, ...)`, for writing to prompt."""
    db_path = Path(settings.paths.database_path) / f"{db_id}.sqlite"
    if not db_path.exists():
        return f"Database {db_id} not found."

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    tables = [
        row[0]
        for row in cursor.fetchall()
        if row[0] and not str(row[0]).startswith("sqlite_")
    ]

    lines: List[str] = []
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        rows = cursor.fetchall()
        names = [row[1] for row in rows]
        cols = ", ".join(names) if names else ""
        lines.append(f"Table: {table}({cols})")

    conn.close()
    return "\n".join(lines) if lines else "(no tables)"


def format_key_features(key_features: List[dict]) -> str:
    """Format key_features as readable text."""
    lines = []
    for i, kf in enumerate(key_features, 1):
        must_tag = "[Required]" if kf.get("must", False) else "[Optional]"
        lines.append(f"{i}. {must_tag} {kf['text']}")
    return "\n".join(lines)
