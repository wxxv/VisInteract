"""
Transform Processor for Vis-Interact Dataset Construction.

Performs SQL and NL transformation with multi-turn validation:
1. SQL Transformation: Query SQL → Visualization SQL (with execution validation)
2. NL Transformation: Query intent → Visualization intent with ambiguity injection

Features:
- Multi-turn conversation for SQL error correction
- Returns execution results as markdown table for confirmation
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import pandas as pd

from core.models import (
    AmbiguityProfile, AmbiguityType, PreferredInterface, Difficulty,
    FeatureCandidate, SemanticContext, DatabaseInfo,
)
from utils.llm_client import get_llm_client
from core.config import get_config

logger = logging.getLogger(__name__)


def _format_vis_effects_static(candidate: "FeatureCandidate") -> str:
    """Format visualization effects for prompt (semantic version)."""
    sections = []
    
    # Data mapping structure
    if candidate.data_mapping:
        sections.append("- **Data Mapping**:")
        mapping = candidate.data_mapping
        
        if "primary" in mapping:
            primary_desc = ", ".join(f"{k}: {v}" for k, v in mapping["primary"].items())
            sections.append(f"  - Primary encodings: {primary_desc}")
        
        if "secondary" in mapping:
            secondary_desc = ", ".join(f"{k}: {v}" for k, v in mapping["secondary"].items())
            sections.append(f"  - Secondary encodings (dual-axis): {secondary_desc}")
        
        if "layers" in mapping:
            for i, layer in enumerate(mapping["layers"], 1):
                if isinstance(layer, dict):
                    layer_desc = ", ".join(f"{k}: {v}" for k, v in layer.items())
                    sections.append(f"  - Layer {i}: {layer_desc}")
        
        if "facet" in mapping:
            if isinstance(mapping["facet"], dict):
                facet_desc = ", ".join(f"{k}: {v}" for k, v in mapping["facet"].items())
                sections.append(f"  - Facet: {facet_desc}")
    
    # Visualization features (semantic list)
    if candidate.visualization_features:
        sections.append("- **Visualization Features**:")
        for feat in candidate.visualization_features:
            sections.append(f"  - {feat}")
    
    if sections:
        return "\n" + "\n".join(sections) + "\n"
    return ""


def _format_list_of_dicts(items: List[Dict[str, Any]], title: str) -> str:
    """Format a list of dicts as compact markdown bullet list."""
    lines: List[str] = []
    for idx, item in enumerate(items, 1):
        kv = _format_inline_kv(item)
        lines.append(f"  - {title} {idx}: {kv}")
    return "\n".join(lines)


def _format_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """Format a dict as compact markdown bullet list."""
    lines: List[str] = []
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            inline = _format_inline_kv(v)
            lines.append(f"{prefix}- {k}: {inline}")
        else:
            lines.append(f"{prefix}- {k}: {v}")
    return "\n".join(lines)


def _format_kv_lines(d: Dict[str, Any], indent: int = 0) -> List[str]:
    """Helper to format key-value pairs."""
    lines: List[str] = []
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}- {k}: {_format_inline_kv(v)}")
        else:
            lines.append(f"{prefix}- {k}: {v}")
    return lines


def _format_inline_kv(d: Dict[str, Any]) -> str:
    """Format dict as inline key: value pairs separated by '; '."""
    parts = []
    for k, v in d.items():
        if isinstance(v, list):
            parts.append(f"{k}: {v}")
        elif isinstance(v, dict):
            parts.append(f"{k}: {json.dumps(v)}")
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


@dataclass
class TransformResult:
    """Result of data processing code generation and validation."""
    # Complete data processing code (import + SQL + Pandas)
    data_processing_code: str
    # Natural language
    vis_question: str
    # Execution results (for validation)
    processed_df: pd.DataFrame
    # Metadata
    transformation_notes: str = ""
    has_pandas_processing: bool = False
    ambiguity_profile: Optional[AmbiguityProfile] = None
    # Validation status
    success: bool = True
    error: Optional[str] = None
    execution_validated: bool = False
    data_quality_passed: bool = False
    data_quality_report: Optional[Dict[str, Any]] = None


class TransformProcessor:
    """
    Unified processor for SQL and NL transformation.
    
    Performs both transformations with multi-turn validation:
    1. Semantic consistency between SQL and question
    2. SQL execution validation with error feedback
    3. Better context awareness for ambiguity injection
    """
    
    def __init__(self, sql_executor: Optional[Callable[[str, str], pd.DataFrame]] = None):
        """
        Initialize transform processor.
        
        Args:
            sql_executor: Function(db_id, sql) -> DataFrame for executing SQL
        """
        self.llm = get_llm_client()
        self._sql_executor = sql_executor
        self._sql_executor_with_error = None  # Function(db_id, sql) -> (DataFrame, error_msg)
    
    def set_sql_executor(self, executor: Callable[[str, str], pd.DataFrame]):
        """Set the SQL executor function (legacy, no error reporting)."""
        self._sql_executor = executor
    
    def set_sql_executor_with_error(self, executor: Callable[[str, str], Tuple[Optional[pd.DataFrame], Optional[str]]]):
        """Set the SQL executor function that also returns error messages."""
        self._sql_executor_with_error = executor
    
    def _build_transform_context(
        self,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        schema_info: DatabaseInfo,
        min_rows: int,
        max_rows: int,
        inject_ambiguity: bool
    ) -> Dict[str, Any]:
        """
        Build reusable context for SQL/NL transformation prompts.
        
        Note: Ambiguity injection is now handled separately by AmbiguityInjector
        after Altair code generation. When inject_ambiguity=False (recommended),
        this generates a clear NL question without ambiguity.
        """
        instance = semantic_ctx.instance
        # Only plan and include ambiguity guidance if explicitly requested
        # Note: Prefer inject_ambiguity=False and use AmbiguityInjector post-Altair
        if inject_ambiguity:
            ambiguity_profile = self._plan_ambiguity(candidate, semantic_ctx)
            ambiguity_guidance = self._get_ambiguity_guidance(ambiguity_profile)
        else:
            ambiguity_profile = None
            ambiguity_guidance = ""
        schema_digest = self._build_schema_context(schema_info)
        vis_effects = _format_vis_effects_static(candidate)
        
        # Execute original SQL and generate markdown table
        original_sql_result = self._execute_and_format_original_sql(
            instance.db_id, 
            instance.gold_sql
        )
        
        return {
            "original_question": instance.question,
            "original_sql": instance.gold_sql,
            "original_sql_result": original_sql_result,
            "evidence": instance.evidence or "None",
            "db_id": instance.db_id,
            "chart_type": candidate.chart_type,
            "dimensions": ', '.join(candidate.dimensions or []),
            "measures": ', '.join(candidate.measures or []),
            "aggregation": candidate.aggregation,
            "vis_intent": candidate.vis_intent,
            "vis_intent_type": candidate.vis_intent_type.value,
            "vis_effects": vis_effects,
            "schema_digest": schema_digest,
            "min_rows": min_rows,
            "max_rows": max_rows,
            "ambiguity_guidance": ambiguity_guidance,
            "ambiguity_profile": ambiguity_profile,
        }
    
    def _execute_and_format_original_sql(self, db_id: str, sql: str) -> str:
        """
        Execute original SQL and format result as markdown table.
        
        Args:
            db_id: Database ID
            sql: SQL query to execute
            
        Returns:
            Formatted markdown with data sample table
        """
        try:
            from execution.sqlite_client import get_sqlite_client
            sqlite_client = get_sqlite_client()
            df = sqlite_client.execute_safe(db_id, sql, limit=10)
            
            if df is not None and not df.empty:
                # Generate markdown table
                table_text = self._df_to_markdown_table(df.head(5), full_df=df)
                
                # Add row count info
                total_rows = len(df)
                if total_rows > 5:
                    table_text += f"\n... +{total_rows - 5} more rows"
                
                result = f"\n## Data Sample (first 5 rows of total {total_rows} rows, with distinct counts per column)\n{table_text}"
            else:
                result = "\n*(SQL execution returned no results)*"
                
        except Exception as e:
            logger.warning(f"Failed to execute original SQL: {e}")
            result = f"\n*(SQL execution failed: {e})*"
        
        return result
    
    def _df_to_markdown_table(self, df: pd.DataFrame, max_cols: int = 10, full_df: pd.DataFrame = None) -> str:
        """
        Convert DataFrame to markdown table format with column statistics.
        
        Args:
            df: DataFrame to display (usually a head() subset)
            max_cols: Maximum columns to display
            full_df: Full DataFrame for computing distinct counts (if None, uses df)
        """
        if df.empty:
            return "*(empty result)*"
        
        # Use full_df for statistics if provided
        stats_df = full_df if full_df is not None else df
        
        # Limit columns for display
        display_df = df.iloc[:, :max_cols] if len(df.columns) > max_cols else df
        stats_display_df = stats_df.iloc[:, :max_cols] if len(stats_df.columns) > max_cols else stats_df
        
        # Build header
        headers = list(display_df.columns)
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        
        # Build distinct count row
        distinct_counts = []
        for col in display_df.columns:
            distinct_count = stats_display_df[col].nunique()
            distinct_counts.append(f"*{distinct_count} distinct*")
        distinct_row = "| " + " | ".join(distinct_counts) + " |"
        
        # Build data rows
        data_rows = []
        for _, row in display_df.iterrows():
            values = []
            for val in row:
                # Truncate long values
                str_val = str(val) if pd.notna(val) else "NULL"
                if len(str_val) > 30:
                    str_val = str_val[:27] + "..."
                values.append(str_val)
            data_rows.append("| " + " | ".join(values) + " |")
        
        result = "\n".join([header_row, separator, distinct_row] + data_rows)
        
        if len(df.columns) > max_cols:
            result += f"\n\n*(showing {max_cols} of {len(df.columns)} columns)*"
        
        return result
    
    def _build_initial_transform_prompt(self, ctx: Dict[str, Any]) -> str:
        """Build the initial SQL/NL transformation prompt (first attempt)."""
        return f"""Perform TWO transformations for creating a visualization dataset sample.

## Task Overview
Transform BOTH the SQL query AND the natural language question to create a visualization-focused sample.

## Original Information
- **Original Question**: {ctx['original_question']}
- **Original SQL**: 
```sql
{ctx['original_sql']}
```

- **Original SQL Result**:
{ctx['original_sql_result']}

- **Domain Evidence**: {ctx['evidence']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions** (X-axis / categories): {ctx['dimensions']}
- **Measures** (Y-axis / values): {ctx['measures']}
- **Aggregation**: {ctx['aggregation']}
- **Composition**: {ctx['composition']}
- **Visualization Intent**: {ctx['vis_intent']}
- **Intent Type**: {ctx['vis_intent_type']}
{ctx['vis_effects']}
## Database Schema
{ctx['schema_digest']}

## Transformation Requirements

### SQL Transformation
1. Result should have {ctx['min_rows']}-{ctx['max_rows']} rows (suitable for visualization)
2. SELECT must include dimension and measure fields
3. If original has LIMIT 1, remove or expand it
4. If original returns single value, add GROUP BY for multiple data points
5. Use {ctx['aggregation']} aggregation for measures
6. Ensure valid SQLite syntax

### IMPORTANT: Minimize SQL Complexity
- **Prefer minimal changes** - only modify what's necessary for visualization
- **Keep similar structure** to the original SQL (don't restructure unnecessarily)
- **Avoid adding** CTEs, subqueries, or window functions unless the original already uses them
- **Don't over-engineer** - a simple GROUP BY is better than complex RANK()/ROW_NUMBER()
- If original SQL is already suitable, use it with minimal adjustments (e.g., just change LIMIT)

### Question Transformation
1. Change from "find/query/get" to "show/visualize/display"
2. Focus on visualization insight ({ctx['vis_intent_type']})
3. Do NOT mention specific chart types
{ctx['ambiguity_guidance']}

## Output Format
Return a JSON object:
```json
{{
    "vis_sql": "transformed SQL query",
    "vis_question": "transformed visualization question",
    "transformation_notes": "brief notes on what was changed"
}}
```"""
    
    def _build_iteration_transform_prompt(
        self,
        ctx: Dict[str, Any],
        previous_sql: str,
        previous_question: str,
        error: Optional[str] = None,
        zero_rows: bool = False,
        quality_issues: Optional[List[str]] = None,
        quality_suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Build a single-turn iteration prompt for SQL/NL transformation.
        
        Includes all context + previous attempt + specific error/feedback.
        """
        # Determine feedback section
        if error:
            feedback_section = f"""## SQL Execution Error
The previous SQL failed with this error:
```
{error}
```

Common issues to check:
- Table or column names may be incorrect
- SQLite syntax errors
- Missing quotes around string values"""
        elif zero_rows:
            feedback_section = """## SQL Execution Result
The SQL executed successfully but returned **0 rows**.

Please modify the SQL to return data. Consider:
- Relaxing WHERE conditions
- Using different table joins
- Checking if the table has data"""
        elif quality_issues:
            issues_text = chr(10).join(f'- {issue}' for issue in quality_issues)
            suggestions_text = chr(10).join(f'- {s}' for s in (quality_suggestions or []))
            feedback_section = f"""## Data Quality Validation Failed
The SQL executed successfully but the resulting data is not ideal for visualization.

**Issues found**:
{issues_text}

**Suggestions**:
{suggestions_text}

Please modify the SQL query to address these issues."""
        else:
            feedback_section = "## Issue\nThe previous transformation needs improvement."
        
        return f"""Fix the SQL/NL transformation based on the feedback below.

## Original Information
- **Original Question**: {ctx['original_question']}
- **Original SQL**: 
```sql
{ctx['original_sql']}
```
- **Original SQL Result**:
{ctx['original_sql_result']}

- **Domain Evidence**: {ctx['evidence']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions** (X-axis / categories): {ctx['dimensions']}
- **Measures** (Y-axis / values): {ctx['measures']}
- **Aggregation**: {ctx['aggregation']}
- **Composition**: {ctx['composition']}
- **Visualization Intent**: {ctx['vis_intent']}
- **Intent Type**: {ctx['vis_intent_type']}
{ctx['vis_effects']}
## Database Schema
{ctx['schema_digest']}

## Previous Attempt (has issues)
**Previous SQL**:
```sql
{previous_sql}
```

**Previous Question**: {previous_question}

{feedback_section}

## Transformation Requirements
1. Result should have {ctx['min_rows']}-{ctx['max_rows']} rows (suitable for visualization)
2. SELECT must include dimension and measure fields
3. Use {ctx['aggregation']} aggregation for measures
4. Ensure valid SQLite syntax
5. Question should focus on visualization insight, NOT mention chart types
{ctx['ambiguity_guidance']}

## Output Format
Return a JSON object:
```json
{{
    "vis_sql": "fixed SQL query",
    "vis_question": "fixed visualization question",
    "transformation_notes": "brief notes on what was fixed"
}}
```"""
    
    def _build_initial_data_processing_prompt(self, ctx: Dict[str, Any]) -> str:
        """Build prompt for generating complete data processing code."""
        return f"""Generate complete data processing code and visualization question.

## Task
Generate TWO components:
1. **Data processing code** - Complete Python script (import + SQL + Pandas)
2. **Visualization question** - Natural language

## Original Information
- **Original Question**: {ctx['original_question']}
- **Original SQL**: 
```sql
{ctx['original_sql']}
```

- **Original SQL Result**:
{ctx['original_sql_result']}

- **Domain Evidence**: {ctx['evidence']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions**: {ctx['dimensions']}
- **Measures**: {ctx['measures']}
- **Aggregation**: {ctx['aggregation']}
- **Visualization Intent**: {ctx['vis_intent']}
{ctx['vis_effects']}

## Database Information
- **Database Path**: `{ctx['db_path']}`
- **Schema**:
{ctx['schema_digest']}

## Requirements

### Part 1: Data Processing Code

Generate a **COMPLETE Python script** that can be executed independently.

**Required Structure**:

```python
import sqlite3
import pandas as pd

# Stage 1: SQL - Connect and retrieve data
conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()

# Stage 2: Pandas Processing (optional, only if needed)
# Type conversions
df['date'] = pd.to_datetime(df['date'])

# Complex operations (ranking, window, pivot, etc.)
df['rank'] = df.groupby('category')['value'].rank()

# Column renaming for display
df = df.rename(columns={{'old_col': 'Display Name'}})

# Result: DataFrame 'df' ready for visualization
```

**SQL Guidelines** (Stage 1: Data Retrieval):

✅ **DO**:
- Table JOINs, WHERE filters
- Basic GROUP BY (SUM, COUNT, AVG, MIN, MAX)
- ORDER BY for sorting
- LIMIT for {ctx['min_rows']}-{ctx['max_rows']} rows

❌ **AVOID** (use Pandas instead):
- Window functions (ROW_NUMBER, RANK, DENSE_RANK)
- Complex CTEs for reshaping
- PIVOT operations

**Pandas Guidelines** (Stage 2: Processing - Optional):

✅ **USE Pandas for**:
- Ranking: `df['rank'] = df.groupby('cat')['val'].rank()`
- Percentages: `df['pct'] = df['val'] / df['val'].sum()`
- Rolling window: `df['ma'] = df['val'].rolling(7).mean()`
- Pivoting: `df_pivot = df.pivot_table(...)`
- Type conversions: `df['date'] = pd.to_datetime(df['date'])`
- Column renaming: `df.rename(columns={{...}})`

❌ **SKIP Pandas if**:
- SQL result is already perfect for visualization

**Important**:
- Code must be **complete and runnable** with `exec()`
- Use actual database path: `{ctx['db_path']}`
- Final DataFrame must be named `df`
- Add comments explaining each step
- **DO NOT use `try ... except` blocks** - code should be straightforward without error handling

### Part 2: Visualization Question

**Requirements**:
- Change from "find/query/get" to "show/visualize/display"
- Focus on visualization insight ({ctx['vis_intent_type']})
- Do NOT mention chart types
- Keep clear and concise

## Output Format

Return TWO markdown blocks in this exact order:

1. **Python code block** - Complete data processing code:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()
```

2. **Plain text block** - Visualization question:

```plaintext
Show me the top 10 products by sales
```

**Important**:
- First block MUST be Python code (```python ... ```)
- Second block MUST be plain text (```plaintext ... ```)
- Code must produce a DataFrame named `df`
- Use triple quotes for SQL ('''...''')
"""
    
    def _build_iteration_data_processing_prompt(
        self,
        ctx: Dict[str, Any],
        previous_code: str,
        previous_question: str,
        execution_error: Optional[str] = None,
        quality_issues: Optional[List[str]] = None,
        quality_suggestions: Optional[List[str]] = None
    ) -> str:
        """Build iteration prompt with feedback from execution or quality validation."""
        # Determine feedback section
        if execution_error:
            feedback_section = f"""## Execution Error

The previous code failed to execute:

```
{execution_error}
```

**Common issues to check**:
- Syntax errors in Python code
- SQL syntax errors (check quotes, table/column names)
- DataFrame operations on non-existent columns
- Type conversion errors

**How to fix**:
- Review the code carefully for syntax issues
- Verify table and column names against schema
- Ensure SQL query is valid for SQLite
- Test data operations step by step
"""
        elif quality_issues:
            issues_text = '\n'.join(f'- {issue}' for issue in quality_issues)
            suggestions_text = '\n'.join(f'- {s}' for s in (quality_suggestions or []))
            
            feedback_section = f"""## Data Quality Validation Failed

The code executed successfully, but the resulting data is not ideal for visualization.

**Issues found**:
{issues_text}

**Suggestions to fix**:
{suggestions_text}

**How to fix**:
- Modify SQL query to adjust data selection/aggregation
- Add/modify Pandas operations to transform data appropriately
- Ensure output DataFrame has proper structure for visualization
"""
        else:
            feedback_section = "## Issue\nThe previous code needs improvement."
        
        return f"""Fix the data processing code based on the feedback below.

## Original Information
- **Original Question**: {ctx['original_question']}
- **Original SQL**: 
```sql
{ctx['original_sql']}
```
- **Domain Evidence**: {ctx['evidence']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions**: {ctx['dimensions']}
- **Measures**: {ctx['measures']}
- **Aggregation**: {ctx['aggregation']}
- **Visualization Intent**: {ctx['vis_intent']}
{ctx['vis_effects']}

## Database Information
- **Database Path**: `{ctx['db_path']}`
- **Schema**:
{ctx['schema_digest']}

## Previous Code (has issues)

```python
{previous_code}
```

**Previous Question**: {previous_question}

{feedback_section}

## Requirements

Generate **FIXED** data processing code that addresses the issues above.

**Code Structure** (same as before):

```python
import sqlite3
import pandas as pd

# Stage 1: SQL
conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()

# Stage 2: Pandas (if needed)
# Your fixed pandas operations here
```

**Important**:
- Fix the specific issues mentioned above
- Keep code complete and runnable
- Final DataFrame must be named `df`
- Result should have {ctx['min_rows']}-{ctx['max_rows']} rows
- Data should be suitable for {ctx['chart_type']} visualization
- **DO NOT use `try ... except` blocks** - code should be straightforward without error handling

## Output Format

Return TWO markdown blocks in this exact order:

1. **Python code block** - Fixed data processing code:

```python
import sqlite3
import pandas as pd

# Fixed code here
```

2. **Plain text block** - Visualization question:

```plaintext
Show me the ...
```

**Important**: First block is Python (```python), second block is plain text (```plaintext)
"""
    
    def _parse_code_and_question_from_markdown(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse data processing code and visualization question from markdown blocks.
        
        Expected format:
        ```python
        # data processing code
        ```
        ```plaintext
        visualization question
        ```
        
        Args:
            text: Response text with markdown blocks
            
        Returns:
            Tuple of (data_processing_code, vis_question)
        """
        # Find all code blocks with optional language tag
        # Pattern: ```language\n...\n``` or ```\n...\n```
        pattern = r'```(\w*)\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            logger.warning("No markdown code blocks found in response")
            return None, None
        
        data_processing_code = None
        vis_question = None
        
        # Process blocks in order
        for i, (lang, content) in enumerate(matches):
            content = content.strip()
            
            # First block should be Python code
            if i == 0:
                if lang.lower() in ['python', 'py']:
                    data_processing_code = content
                elif not lang:
                    # Be lenient - accept blocks without language tag
                    logger.warning("First block has no language tag, assuming Python")
                    data_processing_code = content
                else:
                    logger.warning(f"First block has unexpected language: {lang}")
            
            # Second block should be plaintext (question)
            elif i == 1:
                # Expect plaintext tag, but be lenient with other text formats
                if lang.lower() in ['plaintext', 'text', 'txt']:
                    vis_question = content
                elif not lang:
                    # Accept blocks without language tag
                    logger.warning("Second block has no language tag, assuming plaintext")
                    vis_question = content
                else:
                    # Be lenient - accept it anyway if it looks like a question
                    logger.warning(f"Second block has unexpected language: {lang}, accepting anyway")
                    vis_question = content
                break  # Only need first two blocks
        
        if not data_processing_code:
            logger.warning("Failed to extract data processing code")
        if not vis_question:
            logger.warning("Failed to extract visualization question")
        
        return data_processing_code, vis_question
    
    def _execute_data_processing_code(self, code: str, db_path: str) -> Optional[pd.DataFrame]:
        """
        Execute complete data processing code and extract DataFrame.
        
        Args:
            code: Complete Python code (import + SQL + Pandas) with relative paths
            db_path: Absolute path to database for execution
        
        Returns:
            DataFrame named 'df' from executed code
        """
        try:
            import sqlite3
            import pandas as pd
            import re
            
            # Replace relative database paths with absolute paths
            # Pattern: ./databases/{db_id}.sqlite
            # Use forward slashes to avoid Windows backslash escape issues in exec()
            db_path_safe = db_path.replace('\\', '/')
            code_to_execute = re.sub(
                r'["\']\.\/databases\/(\w+)\.sqlite["\']',
                f'"{db_path_safe}"',
                code
            )
            
            # Create execution namespace
            namespace = {
                'sqlite3': sqlite3,
                'pd': pd,
                'pandas': pd
            }
            
            # Execute the code
            exec(code_to_execute, namespace)
            
            # Extract the 'df' variable
            if 'df' not in namespace:
                logger.error("Data processing code did not create 'df' variable")
                return None
            
            df = namespace['df']
            
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Variable 'df' is not a DataFrame: {type(df)}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Data processing code execution failed: {e}")
            raise
    
    def _has_pandas_operations(self, code: str) -> bool:
        """Check if code contains Pandas operations beyond basic SQL read."""
        # Simple heuristic: check for pandas operations after read_sql
        lines = code.split('\n')
        found_read_sql = False
        
        for line in lines:
            if 'read_sql' in line:
                found_read_sql = True
            elif found_read_sql and 'df' in line and any(
                op in line for op in ['rank(', 'rolling(', 'pivot', 'groupby', 'rename(', 'to_datetime']
            ):
                return True
        
        return False
    
    def transform(
        self,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext,
        schema_info: DatabaseInfo,
        min_rows: int = 5,
        max_rows: int = 100,
        db_path: Optional[str] = None,
        validate_execution: bool = True,
        max_retries: int = 3
    ) -> TransformResult:
        """
        Transform with complete data processing code generation and validation.
        
        Flow:
        1. LLM generates: complete data_processing_code + vis_question
        2. Execute code with exec()
        3. Check if DataFrame 'df' was created
        4. Check for errors → retry if needed
        5. LLM validates data quality
        6. If quality check fails → retry with suggestions
        7. Return code string + processed DataFrame
        
        Args:
            candidate: Target visualization candidate
            semantic_ctx: Semantic context with instance info
            schema_info: Database schema
            min_rows: Minimum rows for visualization
            max_rows: Maximum rows for visualization
            db_path: Path to SQLite database (if None, will try to get from context)
            validate_execution: Whether to validate by executing code
            max_retries: Maximum retry attempts
            
        Returns:
            TransformResult with data_processing_code, vis_question, and processed_df
        """
        instance = semantic_ctx.instance
        
        # Get db_path if not provided
        # For prompts, use relative path; for execution, use absolute path
        from execution.sqlite_client import get_sqlite_client
        sqlite_client = get_sqlite_client()
        
        if not db_path:
            db_path = sqlite_client.get_db_path(instance.db_id)  # Absolute path for execution
        
        db_path_for_llm = sqlite_client.get_db_path_for_llm(instance.db_id)  # Relative path for prompts
        
        # Build base context (reused across all attempts)
        ctx = self._build_transform_context(
            candidate=candidate,
            semantic_ctx=semantic_ctx,
            schema_info=schema_info,
            min_rows=min_rows,
            max_rows=max_rows,
            inject_ambiguity=False  # No ambiguity in transform
        )
        ctx['db_path'] = db_path_for_llm  # Use relative path in prompts
        
        # System prompt
        system_prompt = """You are a data visualization expert. Generate complete data processing code and visualization question.
Return the code in a Python code block (```python), followed by the question in a plaintext block (```plaintext)."""
        
        config = get_config()
        enable_data_quality_check = config.validation.enable_data_quality_check
        max_validation_retries = config.validation.max_validation_retries
        
        # Track feedback for iteration
        last_error = None
        last_quality_issues = None
        last_quality_suggestions = None
        data_processing_code = ""
        vis_question = ""
        df = None
        
        for attempt in range(max_retries + 1):
            # Build prompt based on feedback
            if attempt == 0:
                # First attempt: initial generation
                prompt = self._build_initial_data_processing_prompt(ctx)
            else:
                # Retry: provide feedback
                prompt = self._build_iteration_data_processing_prompt(
                    ctx=ctx,
                    previous_code=data_processing_code,
                    previous_question=vis_question,
                    execution_error=last_error,
                    quality_issues=last_quality_issues,
                    quality_suggestions=last_quality_suggestions
                )
            
            # Reset feedback
            last_error = None
            last_quality_issues = None
            last_quality_suggestions = None
            
            # LLM generates complete data processing code
            response = self.llm.complete(
                prompt,
                system_prompt=system_prompt,
                process_name="transform_processor.transform",
                metadata={"attempt": attempt, "is_retry": attempt > 0}
            )
            
            # Parse markdown blocks
            data_processing_code, vis_question = self._parse_code_and_question_from_markdown(response.content)
            
            if not data_processing_code:
                last_error = "Failed to extract data processing code from response"
                continue
            
            if not vis_question:
                last_error = "Failed to extract visualization question from response"
                continue
            
            # No explicit notes in markdown format, set a default
            notes = f"Generated {'on retry' if attempt > 0 else 'initially'}"
            
            if not data_processing_code:
                last_error = "Empty data processing code"
                continue
            
            # Execute the complete code
            if validate_execution:
                try:
                    df = self._execute_data_processing_code(data_processing_code, db_path)
                    
                    if df is None:
                        last_error = "Code executed but df is None"
                        continue
                    
                    if df.empty or len(df) == 0:
                        last_error = "DataFrame is empty (0 rows)"
                        continue
                    
                    execution_validated = True
                    
                    # Data quality validation with LLM
                    if enable_data_quality_check:
                        quality_passed, quality_report, suggestions = \
                            self._validate_data_quality(
                                df=df,
                                candidate=candidate,
                                vis_question=vis_question,
                                data_processing_code=data_processing_code,
                                min_rows=min_rows,
                                max_rows=max_rows
                            )
                        
                        if not quality_passed:
                            # Check if we've exceeded validation retries
                            if attempt >= max_validation_retries:
                                logger.warning(f"Data quality validation failed after {attempt + 1} attempts, proceeding anyway")
                                data_quality_passed = False
                                break
                            
                            # Store quality feedback for next iteration
                            last_quality_issues = quality_report.get("issues", [])
                            last_quality_suggestions = suggestions
                            logger.debug(f"Attempt {attempt}: Data quality issues: {last_quality_issues}")
                            continue
                        
                        data_quality_passed = True
                        data_quality_report = quality_report
                    else:
                        data_quality_passed = True
                        data_quality_report = None
                    
                    # Success!
                    return TransformResult(
                        data_processing_code=data_processing_code,
                        vis_question=vis_question,
                        processed_df=df,
                        has_pandas_processing=self._has_pandas_operations(data_processing_code),
                        transformation_notes=notes,
                        success=True,
                        execution_validated=True,
                        data_quality_passed=data_quality_passed,
                        data_quality_report=data_quality_report
                    )
                    
                except Exception as e:
                    # Provide detailed error message with exception type
                    error_type = type(e).__name__
                    last_error = f"{error_type}: {str(e)}"
                    logger.debug(f"Data processing code execution failed ({error_type}):\n{data_processing_code}")
                    continue
            else:
                # No validation, just return the code
                break
        
        # Failed after retries
        if not vis_question:
            vis_question = f"Show me {candidate.vis_intent}"
        
        return TransformResult(
            data_processing_code=data_processing_code,
            vis_question=vis_question,
            processed_df=df if df is not None else pd.DataFrame(),
            transformation_notes=notes,
            success=False,
            error=last_error or "Max retries reached"
        )
    
    def _plan_ambiguity(
        self,
        candidate: FeatureCandidate,
        semantic_ctx: SemanticContext
    ) -> AmbiguityProfile:
        """Plan what kind of ambiguity to inject."""
        # Determine ambiguity type based on candidate
        if candidate.vis_intent_type.value in ["trend", "distribution"]:
            ambiguity_type = AmbiguityType.DATA
        elif candidate.vis_intent_type.value in ["comparison", "ranking"]:
            ambiguity_type = AmbiguityType.VISUALIZATION
        else:
            ambiguity_type = AmbiguityType.INFO_COMPLETION
        
        # Determine target paths
        target_paths = []
        
        if candidate.aggregation:
            target_paths.append("encoding.y.aggregate")
        
        if any("date" in d.lower() or "time" in d.lower() for d in candidate.dimensions):
            target_paths.append("encoding.x.timeUnit")
        
        if len(candidate.dimensions) > 1:
            target_paths.append("encoding.color.field")
        
        target_paths.append("mark.type")
        target_paths = target_paths[:2]  # Limit for moderate difficulty
        
        # Determine preferred interface
        if ambiguity_type == AmbiguityType.VISUALIZATION:
            preferred_interface = PreferredInterface.VIS
        else:
            preferred_interface = PreferredInterface.TEXT
        
        # Determine difficulty
        if len(target_paths) == 1:
            difficulty = Difficulty.EASY
        elif len(target_paths) == 2:
            difficulty = Difficulty.MEDIUM
        else:
            difficulty = Difficulty.HARD
        
        return AmbiguityProfile(
            ambiguity_types=[ambiguity_type],  # Wrap in list for new format
            template_names=[],  # No specific template names in this context
            target_feature_paths=target_paths,
            preferred_interface=preferred_interface,
            difficulty=difficulty
        )
    
    def _get_ambiguity_guidance(self, profile: AmbiguityProfile) -> str:
        """Get ambiguity injection guidance for the prompt."""
        guidance_parts = ["\n### Ambiguity Injection"]
        
        # Handle multiple ambiguity types
        for amb_type in profile.ambiguity_types:
            if amb_type == AmbiguityType.DATA:
                guidance_parts.append("""
Make the question ambiguous about DATA aspects:
- Don't specify exact aggregation (sum vs average vs count)
- Don't specify exact time granularity (by day vs month vs year)
- Use vague terms like "overall", "generally", "typically"
""")
            elif amb_type == AmbiguityType.VISUALIZATION:
                guidance_parts.append("""
Make the question ambiguous about VISUALIZATION aspects:
- Don't suggest any specific chart type
- Don't specify how to encode data (what goes on x-axis, etc.)
- Keep focus on what insight is needed, not how to display it
""")
            elif amb_type == AmbiguityType.INFO_COMPLETION:
                guidance_parts.append("""
Make the question INCOMPLETE:
- Omit some important filtering criteria
- Leave some context unclear
- The user will need to ask clarifying questions
""")
        
        guidance_parts.append(f"Target difficulty: {profile.difficulty.value}")
        
        return "\n".join(guidance_parts)
    
    def _build_schema_context(self, schema_info: DatabaseInfo) -> str:
        """
        Build schema context for LLM prompt using FK-organized table groups with sample data.
        
        This format organizes tables by foreign key relationships and includes
        sample data to help LLM understand the data content.
        """
        if not schema_info:
            return "Schema not available"
        
        from preprocessing.schema import get_schema_processor
        schema_processor = get_schema_processor()
        return schema_processor.format_schema_with_groups_and_samples(schema_info)
    
    def _df_to_markdown_table(self, df: pd.DataFrame, max_cols: int = 8, full_df: pd.DataFrame = None) -> str:
        """
        Convert DataFrame to markdown table format with column statistics.
        
        Args:
            df: DataFrame to display (usually a head() subset)
            max_cols: Maximum columns to display
            full_df: Full DataFrame for computing distinct counts (if None, uses df)
        """
        if df is None or df.empty:
            return "*(empty result)*"
        
        # Use full_df for statistics if provided
        stats_df = full_df if full_df is not None else df
        
        # Limit columns for display
        display_df = df.iloc[:, :max_cols] if len(df.columns) > max_cols else df
        stats_display_df = stats_df.iloc[:, :max_cols] if len(stats_df.columns) > max_cols else stats_df
        
        # Build header
        headers = list(display_df.columns)
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        
        # Build distinct count row
        distinct_counts = []
        for col in display_df.columns:
            distinct_count = stats_display_df[col].nunique()
            distinct_counts.append(f"*{distinct_count} distinct*")
        distinct_row = "| " + " | ".join(distinct_counts) + " |"
        
        # Build data rows
        data_rows = []
        for _, row in display_df.iterrows():
            values = []
            for val in row:
                str_val = str(val) if pd.notna(val) else "NULL"
                if len(str_val) > 25:
                    str_val = str_val[:22] + "..."
                values.append(str_val)
            data_rows.append("| " + " | ".join(values) + " |")
        
        result = "\n".join([header_row, separator, distinct_row] + data_rows)
        
        if len(df.columns) > max_cols:
            result += f"\n\n*(showing {max_cols} of {len(df.columns)} columns)*"
        
        return result
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL output."""
        if not sql:
            return ""
        
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()
        
        # Find SELECT statement
        match = re.search(r'\bSELECT\b', sql, re.IGNORECASE)
        if match:
            sql = sql[match.start():]
        
        return sql
    
    def _clean_question(self, question: str) -> str:
        """Clean and normalize the question."""
        if not question:
            return ""
        
        question = question.strip().strip('"\'')
        
        if question.startswith("```"):
            question = question.strip("`").strip()
        
        if not question.endswith(('?', '.', '!')):
            question += '?'
        
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        candidate: FeatureCandidate,
        vis_question: str,
        data_processing_code: Optional[str] = None,
        min_rows: int = 5,
        max_rows: int = 100
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Use LLM to validate if the SQL result data is suitable for visualization.
        
        Checks:
        - Row count within acceptable range
        - Dimension field cardinality suitable for chart type
        - Measure field distribution is meaningful
        - Data can answer the vis_question
        
        Args:
            df: DataFrame with SQL execution result
            candidate: Target visualization candidate
            vis_question: The visualization question to answer
            data_processing_code: The data processing code that generated this data (for context)
            min_rows: Minimum expected rows
            max_rows: Maximum expected rows
            
        Returns:
            - passed: Whether validation passed
            - quality_report: Detailed quality report
            - suggestions: List of improvement suggestions
        """
        config = get_config()
        
        # Quick structural checks
        row_count = len(df)
        if row_count < min_rows:
            return False, {
                "suitable": False,
                "score": 0.0,
                "issues": [f"Too few rows: {row_count} < {min_rows}"]
            }, [f"Modify SQL to return at least {min_rows} rows"]
        
        if row_count > max_rows * 2:
            # Warning but not failure - will be truncated
            pass
        
        # Build data statistics for LLM
        columns = list(df.columns)
        dim_cardinality = {}
        measure_stats = {}
        
        for dim in candidate.dimensions:
            if dim in df.columns:
                dim_cardinality[dim] = int(df[dim].nunique())
        
        for measure in candidate.measures:
            if measure in df.columns:
                col = df[measure]
                if pd.api.types.is_numeric_dtype(col):
                    measure_stats[measure] = {
                        "min": float(col.min()) if pd.notna(col.min()) else None,
                        "max": float(col.max()) if pd.notna(col.max()) else None,
                        "mean": float(col.mean()) if pd.notna(col.mean()) else None,
                        "null_count": int(col.isna().sum())
                    }
        
        # Build markdown table of first 5 rows (with distinct counts from full data)
        preview_rows = min(5, df.shape[0])
        data_table = self._df_to_markdown_table(df.head(preview_rows), full_df=df)

        if df.shape[0] > 5:
            omitted_count = df.shape[0] - 5
            data_table += f"\n... +{omitted_count} more rows"
        
        # Include data processing code if provided for better context
        code_section = ""
        if data_processing_code:
            code_section = f"""
## Data Processing Code
```python
{data_processing_code}
```
"""

        prompt = f"""Analyze if this SQL result data is suitable for creating a {candidate.chart_type} visualization.

## User Question
{vis_question}

## Visualization Design
- Chart Type: {candidate.chart_type}
- Dimensions: {candidate.dimensions}
- Measures: {candidate.measures}
- Aggregation: {candidate.aggregation}
- Intent: {candidate.vis_intent}
{code_section}
## Data Sample (first {preview_rows} rows of total {df.shape[0]} rows, with distinct counts per column)
{data_table}

## Data Statistics
- Total rows: {row_count}
- Columns: {columns}
- Dimension cardinality: {json.dumps(dim_cardinality)}
- Measure statistics: {json.dumps(measure_stats)}

## Evaluation Criteria
1. **Row count**: Is {row_count} rows appropriate for this chart type?
2. **Dimension diversity**: Are there enough distinct values for meaningful comparison?
3. **Measure distribution**: Is the data range meaningful (not all zeros, not all same value)?
4. **Chart fit**: Does the data structure match the chart type requirements?
5. **Question alignment**: Can this data answer the user's question?

Return a JSON object:
{{
    "suitable": true/false,
    "score": 0.0-1.0,
    "issues": ["list of specific issues found"],
    "suggestions": ["specific suggestions to improve the data processing code (SQL queries or pandas transformations)"]
}}

**Note**: If providing suggestions, reference the specific part of the data processing code that needs modification.
"""
        
        result = self.llm.complete_json(
            prompt,
            process_name="transform_processor.validate_data_quality"
        )
        
        if not result:
            # LLM failed, assume pass with warning
            return True, {
                "suitable": True,
                "score": 0.5,
                "issues": ["LLM validation failed, using default pass"]
            }, []
        
        passed = result.get("suitable", True)
        score = result.get("score", 0.5)
        
        # Check against minimum score threshold
        min_score = config.validation.min_data_diversity_score
        if score < min_score:
            passed = False
        
        return passed, result, result.get("suggestions", [])
    
    def _create_fallback_result(
        self,
        instance,
        candidate: FeatureCandidate,
        db_path: str,
        error: Optional[str] = None
    ) -> TransformResult:
        """Create fallback result when LLM fails."""
        # Use original SQL with simple modifications
        sql = instance.gold_sql
        if "LIMIT 1" in sql.upper():
            sql = re.sub(r'\bLIMIT\s+1\b', 'LIMIT 20', sql, flags=re.IGNORECASE)
        
        question = f"Show me the {candidate.vis_intent_type.value} of the data."
        
        # Create fallback data processing code
        fallback_code = f"""import sqlite3
import pandas as pd

conn = sqlite3.connect('{db_path}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()
"""
        
        return TransformResult(
            data_processing_code=fallback_code,
            vis_question=question,
            processed_df=pd.DataFrame(),
            transformation_notes="Fallback: LLM transformation failed",
            success=False,
            error=error
        )


# Singleton instance
_processor: Optional[TransformProcessor] = None


def get_transform_processor() -> TransformProcessor:
    """Get or create singleton transform processor."""
    global _processor
    if _processor is None:
        _processor = TransformProcessor()
    return _processor
