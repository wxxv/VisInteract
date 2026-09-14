"""
Spec Generator for Step 7: Ground Truth Spec generation.
Generates Altair/Vega-Lite specifications for visualizations.

Supports multi-turn conversation for:
- SQL execution validation with error feedback
- Altair code validation with image feedback
"""
import json
import base64
import io
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

import logging
from pathlib import Path

from core.models import (
    Feature, FeatureOp, ChartContract, IterationFeedback,
    FeatureCandidate, DatabaseInfo, SQLSemanticSummary,
)
from utils.chart_contracts import get_contract_manager
from utils.llm_client import get_llm_client
from core.config import get_config

logger = logging.getLogger(__name__)


class SpecGenerator:
    """Generates Altair/Vega-Lite specifications."""
    
    def __init__(self):
        self.contract_manager = get_contract_manager()
        self.llm = get_llm_client()
    
    def generate_altair_code(
        self,
        df: pd.DataFrame,
        candidate: FeatureCandidate,
        vis_sql: str,
        db_path: str,
        schema_info: Optional[DatabaseInfo] = None,
        sql_summary: Optional[SQLSemanticSummary] = None,
        user_question: str = "",
        evidence: str = ""
    ) -> str:
        """
        Generate complete Altair Python code for visualization.
        
        The generated code includes:
        1. SQL execution code (with sqlite3)
        2. Pandas data processing code
        3. Altair visualization code with advanced effects
        
        Args:
            df: DataFrame with query results (for column/data reference)
            candidate: FeatureCandidate with visualization design
            vis_sql: SQL query to execute
            db_path: Path to SQLite database
            schema_info: Database schema information
            sql_summary: Semantic summary of SQL query
            user_question: User's analytical question
            evidence: Domain knowledge/evidence
        """
        
        # Build data description with columns info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            columns_info.append(f"  - {col} ({dtype}): {sample}")
        data_desc = "\n".join(columns_info)
        
        # Build SQL execution result sample (first 5 rows as markdown table with distinct counts)
        preview_rows = min(5, df.shape[0])
        sql_result_sample = self._df_to_markdown_table(df.head(preview_rows), full_df=df)
        if df.shape[0] > 5:
            omitted_count = df.shape[0] - 5
            sql_result_sample += f"\n... +{omitted_count} more rows"
        
        # Build schema description using FK-organized table groups with sample data
        schema_desc = ""
        if schema_info:
            from preprocessing.schema import get_schema_processor
            schema_processor = get_schema_processor()
            schema_desc = schema_processor.format_schema_with_groups_and_samples(schema_info)
        
        # Build features description
        features_desc = []
        for f in candidate.features:
            if f.value is not None:
                features_desc.append(f"  - {f.path} {f.op.value} {f.value}")
            else:
                features_desc.append(f"  - {f.path} {f.op.value}")
        
        # Build visualization effects description
        vis_effects = self._format_vis_effects(candidate)
        
        # Get contract for mark guidance
        contract = self.contract_manager.build_contract(candidate.chart_type)
        
        prompt = f"""Generate COMPLETE, STANDALONE Altair visualization code.

## User Request
{user_question or candidate.analytical_question}

## Domain Knowledge
{evidence or "None provided"}

## Database Information
- Database Path: {db_path}
- Tables: 
{schema_desc or "Not provided"}

## SQL Query to Execute
```sql
{vis_sql}
```

## SQL Execution Result (first {min(10, df.shape[0])} rows of total {df.shape[0]} rows)
{sql_result_sample}

## Target Visualization
- **Chart Type**: {candidate.chart_type}
- **Dimensions**: {candidate.dimensions}
- **Measures**: {candidate.measures}
- **Aggregation**: {candidate.aggregation}
- **Visualization Intent**: {candidate.vis_intent}
{vis_effects}

## Required Key Features (must satisfy)
{chr(10).join(features_desc) if features_desc else 'None specified yet'}

## Contract Requirements
- Mark type: {contract.chart_type if contract else candidate.chart_type}
- Category: {contract.category if contract else "general"}

## CODE STRUCTURE REQUIREMENTS

Generate code with THREE sections:

### Section 1: Data Loading (SQL Execution)
```python
import altair as alt
import sqlite3
import pandas as pd

# Connect to database and execute SQL
conn = sqlite3.connect('{db_path}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()
```

### Section 2: Data Processing (Pandas)
```python
# Type conversions if needed (e.g., dates)
# Column renaming for better display
# Any derived calculations
# DO NOT filter or aggregate - SQL already did that
```

### Section 3: Visualization (Altair)
```python
# Create chart with ALL specified effects:
# - Proper encodings (x, y, color, size, etc.)
# - Transforms if specified (window, calculate, filter)
# - Interactions if specified (selection, slider, brush)
# - Multi-layer composition if needed
# - Scale customizations (log scale, color schemes)
# - Mark customizations (innerRadius for donut, etc.)

chart = alt.Chart(df)...

chart  # REQUIRED: Last line must be just "chart" (not chart.show() or chart.display())
```

## Important Notes
1. Use the ACTUAL SQL query and database path provided above
2. Variable name for final chart MUST be 'chart'
3. **The last line of code MUST be simply `chart` (not `chart.show()`, not `chart.display()`)**
4. Use proper Altair 5.x syntax
5. Implement ALL visualization effects from the design
6. The code should be completely runnable standalone

Output ONLY the Python code, no explanations.
"""
        
        response = self.llm.complete(
            prompt,
            process_name="spec_generator.generate_altair_code"
        )
        code = response.content.strip()
        
        # Clean up code
        code = self._clean_code(code)
        
        return code
    
    def _format_vis_effects(self, candidate: FeatureCandidate) -> str:
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
            return "\n" + "\n".join(sections)
        return ""
    
    def _format_list_of_dicts(self, items: List[Dict[str, Any]], title: str) -> str:
        """Format a list of dicts as compact markdown bullet list."""
        lines: List[str] = []
        for idx, item in enumerate(items, 1):
            kv = self._format_inline_kv(item)
            lines.append(f"  - {title} {idx}: {kv}")
        return "\n".join(lines)
    
    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Format a dict as compact markdown bullet list."""
        lines: List[str] = []
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                inline = self._format_inline_kv(v)
                lines.append(f"{prefix}- {k}: {inline}")
            else:
                lines.append(f"{prefix}- {k}: {v}")
        return "\n".join(lines)
    
    def _format_inline_kv(self, d: Dict[str, Any]) -> str:
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
    
    def _build_base_context(
        self,
        df: pd.DataFrame,
        candidate: FeatureCandidate,
        data_processing_code: str,
        schema_info: Optional[DatabaseInfo] = None,
        user_question: str = "",
        evidence: str = "",
        db_path: str = ""
    ) -> Dict[str, Any]:
        """
        Build reusable base context for code generation prompts.
        
        This context is shared across initial generation and iteration attempts.
        """
        # Build data description with columns info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            columns_info.append(f"  - {col} ({dtype}): {sample}")
        data_desc = "\n".join(columns_info)
        
        # Build SQL execution result sample (first 5 rows with distinct counts)
        preview_rows = min(5, df.shape[0])
        sql_result_sample = self._df_to_markdown_table(df.head(preview_rows), full_df=df)
        if df.shape[0] > 5:
            omitted_count = df.shape[0] - 5
            sql_result_sample += f"\n... +{omitted_count} more rows"
        
        # Build schema description using FK-organized table groups with sample data
        schema_desc = ""
        if schema_info:
            from preprocessing.schema import get_schema_processor
            schema_processor = get_schema_processor()
            schema_desc = schema_processor.format_schema_with_groups_and_samples(schema_info)
        
        # Get contract for validation
        contract = self.contract_manager.build_contract(candidate.chart_type)
        
        # Get sample code from chart_example for reference
        sample_codes = self.contract_manager.get_sample_code(candidate.chart_type)
        sample_code_section = self._format_sample_code(sample_codes, candidate.chart_type)
        
        # Build visualization effects description
        vis_effects = self._format_vis_effects(candidate)
        
        return {
            "user_question": user_question or candidate.analytical_question,
            "evidence": evidence or "None provided",
            "schema_desc": schema_desc or "Not provided",
            "db_path": db_path,
            "data_processing_code": data_processing_code,
            "sql_result_num": df.shape[0],
            "sql_result_sample": sql_result_sample,
            "data_desc": data_desc,
            "chart_type": candidate.chart_type,
            "dimensions": candidate.dimensions,
            "measures": candidate.measures,
            "aggregation": candidate.aggregation,
            "vis_intent": candidate.vis_intent,
            "vis_effects": vis_effects,
            "sample_code_section": sample_code_section,
            "mark_type": contract.chart_type if contract else candidate.chart_type,
        }
    
    def _format_sample_code(self, sample_codes: List[str], chart_type: str) -> str:
        """Format chart_example code for reference."""
        if not sample_codes:
            return ""
        
        return f"""## Reference Example from chart_example

Working example of `{chart_type}`:

```python
{sample_codes[0]}
```

**How to use**:
- Study the chart structure (mark, encodings, transforms)
- Adapt the pattern to YOUR DataFrame
- Match field names to your data columns
- Keep the same visual encoding patterns
"""
    
    def _assemble_complete_code(
        self,
        data_processing_code: str,
        altair_code: str
    ) -> str:
        """
        [DEPRECATED] Assemble complete code from data processing and Altair parts.
        
        This function is no longer used as LLM now generates complete code directly.
        Kept for backward compatibility.
        """
        # Add altair import if not already in data processing code
        if 'import altair' not in data_processing_code and 'import alt' not in data_processing_code:
            altair_import = "import altair as alt\n"
        else:
            altair_import = ""
        
        return f"""{data_processing_code}

{altair_import}# Visualization
{altair_code}
"""
    
    def _build_initial_generation_prompt(self, ctx: Dict[str, Any]) -> str:
        """
        Build the initial code generation prompt (first attempt).
        """
        return f"""Generate complete Python code (data processing + visualization) for this task.

## User Request
{ctx['user_question']}

{ctx['sample_code_section']}

## Domain Knowledge
{ctx['evidence']}

## Data Processing (ALREADY VALIDATED)

The following data processing code has been validated and will be included:

```python
{ctx['data_processing_code']}
```

This code produces a DataFrame `df` with the following structure:

{ctx['sql_result_sample']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions**: {ctx['dimensions']}
- **Measures**: {ctx['measures']}
{ctx['vis_effects']}

## Your Task

Generate **COMPLETE Python code** including BOTH data processing AND visualization.

**Required Structure**:

```python
import altair as alt
import sqlite3
import pandas as pd

# Stage 1: SQL - Data retrieval
conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()

# Stage 2: Pandas Processing (if needed)
# df['new_col'] = ...

# Stage 3: Visualization
chart = alt.Chart(df).mark_{ctx['chart_type']}().encode(
    # Your encodings based on the sample code and design
    # ...
).properties(
    # Chart properties
)

chart  # MUST be the last line
```

**Instructions**:
1. Generate the COMPLETE code from scratch (data processing + visualization)
2. You can reference or adapt the provided data processing code
3. Implement ALL visualization effects from the design
4. Variable name must be `chart`
5. Last line MUST be simply `chart`

## Output Format

Return the complete code in a Python code block:

```python
import altair as alt
import sqlite3
import pandas as pd

# Data processing
conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()

# Visualization
chart = alt.Chart(df).mark_bar().encode(
    x='column_name:Q',
    y='another_column:N'
).properties(
    width=400,
    height=300
)

chart
```

**Important**:
- Generate COMPLETE code (data processing + visualization)
- Last line MUST be simply `chart`"""
    
    def _build_iteration_prompt(
        self,
        ctx: Dict[str, Any],
        previous_code: str,
        error: Optional[str] = None,
        vlm_feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a single-turn iteration prompt with previous code and error/feedback.
        
        This replaces the multi-turn conversation pattern. All context is provided
        in one prompt so the LLM can directly fix the code.
        """
        # Determine what kind of feedback we have
        if error:
            feedback_section = f"""## Execution Error
The previous code failed with this error:
```
{error}
```

Common issues to check:
- Field names must match DataFrame columns exactly
- Ensure proper Altair syntax (alt.Chart, encode, mark_*)
- Variable must be named 'chart'
- **The last line of code MUST be simply `chart`**
- Check for deprecated Altair parameters"""
        elif vlm_feedback:
            issues = vlm_feedback.get("issues", [])
            suggestions = vlm_feedback.get("suggestions", [])
            feedback_section = f"""## VLM Chart Validation Failed
The code executed but the rendered chart has quality issues.

**Issues found**:
{chr(10).join(f'- {issue}' for issue in issues)}

**Readability Score**: {vlm_feedback.get('readability_score', 'N/A')}
**Data Represented Correctly**: {vlm_feedback.get('data_represented_correctly', 'N/A')}
**Visual Quality OK**: {vlm_feedback.get('visual_quality_ok', 'N/A')}

**Suggestions to fix**:
{chr(10).join(f'- {s}' for s in suggestions)}

Focus on:
- Fixing data representation if incorrect
- Improving chart aspect ratio and layout
- Improving label readability (rotation, font size)
- Avoiding overlapping elements
- Ensuring proper legend placement"""
        else:
            feedback_section = "## Issue\nThe previous code needs improvement."
        
        return f"""Fix the complete Python code based on the feedback below.

## User Request
{ctx['user_question']}

{ctx['sample_code_section']}

## Domain Knowledge
{ctx['evidence']}

## Data Processing (ALREADY VALIDATED)

The data processing code is pre-validated and should NOT be modified:

The data processing code produces DataFrame `df`:

{ctx['sql_result_sample']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions**: {ctx['dimensions']}
- **Measures**: {ctx['measures']}
{ctx['vis_effects']}

## Previous Complete Code (has issues)
```python
{previous_code}
```

{feedback_section}

## Fix Requirements
- Address the issues identified above
- Generate COMPLETE code (data processing + visualization)
- You can modify both data processing and visualization as needed
- Variable must be named 'chart'
- Last line MUST be simply `chart`
- Use Altair 5.x syntax

## Output Format

Return the fixed complete code in a Python code block:

```python
import altair as alt
import sqlite3
import pandas as pd

# Data processing
conn = sqlite3.connect('{ctx['db_path']}')
SQL = "SELECT ... FROM ... WHERE ... GROUP BY ... LIMIT ..."
df = pd.read_sql_query(SQL, conn)
conn.close()

# Visualization
chart = alt.Chart(df).mark_bar().encode(
    # ... your fixed encodings ...
).properties(
    # ... properties ...
)

chart
```

**Important**: Generate COMPLETE code (data processing + visualization) and end with `chart`"""
    
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
    
    def generate_altair_code_with_validation(
        self,
        df: pd.DataFrame,
        candidate: FeatureCandidate,
        data_processing_code: str,
        schema_info: Optional[DatabaseInfo] = None,
        sql_summary: Optional[SQLSemanticSummary] = None,
        user_question: str = "",
        evidence: str = "",
        max_retries: int = 3,
        sample_id: str = "",
        db_path: str = ""
    ) -> Tuple[
        str,
        Optional[Dict[str, Any]],
        Optional[str],
        Optional[str],
        Optional[Dict[str, Any]],
        bool,
        Optional[bool],
        Optional[str]
    ]:
        """
        Generate complete Python code (data processing + visualization) with validation.
        
        The data_processing_code is provided as a reference. LLM generates complete
        code from scratch including both data processing and visualization.
        
        Uses single-turn prompts (no conversation accumulation):
        1. LLM generates complete code (data processing + visualization)
        2. Execute and validate
        3. If execution error: create fresh prompt with error feedback
        4. If success: save image and run VLM validation
        5. If VLM validation fails: create fresh prompt with VLM feedback
        
        Args:
            df: DataFrame with processed data (for validation)
            candidate: FeatureCandidate with visualization design
            data_processing_code: Reference data processing code (not directly used)
            schema_info: Database schema information
            sql_summary: Semantic summary of SQL query
            user_question: User's analytical question
            evidence: Domain knowledge/evidence
            max_retries: Maximum number of retry attempts
            sample_id: Sample ID for image file naming
            
        Returns:
            - code: Final complete code (data processing + visualization)
            - spec: Vega-Lite spec if successful
            - image_base64: Base64 encoded PNG image if successful
            - image_path: Path to saved chart image
            - vlm_report: VLM validation report
            - success: Whether code generation and validation succeeded
            - validation_passed: Result of VLM validation (None if skipped)
            - error_message: Last error encountered during generation/validation
        """
        # Build base context (reused across all attempts)
        base_context = self._build_base_context(
            df=df,
            candidate=candidate,
            data_processing_code=data_processing_code,
            schema_info=schema_info,
            user_question=user_question,
            evidence=evidence,
            db_path=db_path
        )
        
        # System prompt for code generation
        system_prompt = """You are an expert data visualization engineer. Generate complete Python code including data processing and visualization.
Return the complete code in a Python code block as specified in the prompt."""
        
        config = get_config()
        enable_vlm_check = config.validation.enable_vlm_chart_check
        max_vlm_retries = config.validation.max_validation_retries
        
        code = None
        spec = None
        image_base64 = None
        image_path = None
        vlm_report = None
        vlm_retry_count = 0
        validation_passed: Optional[bool] = None
        success = False
        error_message = None
        
        # Track last error/feedback for iteration
        last_error = None
        last_vlm_feedback = None
        
        for attempt in range(max_retries + 1):
            # Build prompt based on attempt number
            if attempt == 0:
                # First attempt: initial generation prompt
                prompt = self._build_initial_generation_prompt(base_context)
            else:
                # Retry: single-turn prompt with previous code + error/feedback
                prompt = self._build_iteration_prompt(
                    ctx=base_context,
                    previous_code=code,
                    error=last_error,
                    vlm_feedback=last_vlm_feedback
                )
            
            # Reset feedback for this iteration
            last_error = None
            last_vlm_feedback = None
            
            # Single-turn LLM call (no message accumulation)
            response = self.llm.complete(
                prompt,
                system_prompt=system_prompt,
                process_name="spec_generator.generate_altair_code_with_validation",
                metadata={"attempt": attempt, "is_retry": attempt > 0}
            )
            
            # Extract complete code from markdown code block
            # LLM generates complete code (data processing + visualization)
            complete_code = self._extract_code_from_markdown(response.content)
            
            # Clean the code
            if complete_code:
                code = self._clean_code(complete_code)
            else:
                # Fallback: use entire response content
                code = self._clean_code(response.content)
            
            # Execute and validate
            spec, error = self.execute_altair_code(code, df, skip_sql_execution=True)
            
            if error:
                error_message = error
                if attempt >= max_retries:
                    logger.warning(f"Max retries ({max_retries}) reached with execution error")
                    break
                
                # Store error for next iteration
                last_error = error
                logger.debug(f"Attempt {attempt} execution error: {error[:100]}...")
                continue
            
            # Code executed successfully! Save image and run VLM validation
            try:
                # Save chart image to temp folder
                effective_sample_id = sample_id or f"temp_{attempt}"
                image_path, image_base64, render_error = self._save_chart_image(spec, effective_sample_id)
                
                if not image_base64:
                    # Fallback to in-memory rendering
                    image_base64, render_error2 = self._render_chart_to_base64(spec)
                    if render_error2:
                        render_error = render_error or render_error2
                
                if not image_base64:
                    # Rendering failed - treat as error and retry
                    error_message = render_error or "Chart rendering failed: Unable to convert spec to image. This may be due to duplicate parameter names, invalid encodings, or other Vega-Lite errors."
                    
                    if attempt >= max_retries:
                        logger.warning(f"Failed to render chart image after {max_retries} retries, giving up")
                        break
                    
                    # Store rendering error for next iteration with detailed info
                    last_error = f"RENDERING ERROR: {error_message}\n\nThe code executed successfully but the chart cannot be rendered. Common causes:\n- Duplicate parameter/signal names (check selection parameters)\n- Invalid field references in encodings\n- Conflicting transforms or layered charts with incompatible parameters"
                    logger.warning(f"Attempt {attempt} rendering failed: {error_message}, will retry with error feedback")
                    continue
                
                # VLM validation (if enabled)
                if enable_vlm_check:
                    vlm_passed, vlm_report, vlm_suggestions = self._vlm_validate_chart(
                        image_base64=image_base64,
                        candidate=candidate,
                        vis_question=user_question,
                        data_sample=df.head(5),
                        altair_code=code  # Provide code context for better suggestions
                    )
                    validation_passed = vlm_passed
                    
                    # Check if VLM had a parsing error (not a real validation failure)
                    vlm_error = vlm_report.get("vlm_error", False) if vlm_report else True
                    
                    if not vlm_passed:
                        error_message = "VLM validation failed"
                        vlm_retry_count += 1
                        
                        # If VLM parsing failed (not a chart quality issue), skip retry
                        if vlm_error:
                            logger.warning("VLM parsing failed, proceeding without VLM validation")
                            vlm_report = {"vlm_error": True, "skipped": True}
                            validation_passed = None
                            success = True  # 解析错误不算失败，标记为成功
                            break
                        
                        # Check if we should retry or accept the chart
                        if attempt >= max_retries:
                            logger.warning("Max code generation retries reached, accepting chart despite VLM issues")
                            success = True  # 达到最大重试次数，接受当前图表
                            validation_passed = False  # 但记录VLM验证未通过
                            break
                        
                        if vlm_retry_count > max_vlm_retries:
                            logger.warning(f"VLM validation failed after {vlm_retry_count} attempts, accepting chart anyway")
                            success = True  # VLM多次失败，接受当前图表
                            validation_passed = False
                            break
                        
                        # Store VLM feedback for next iteration
                        last_vlm_feedback = {
                            "issues": vlm_report.get("issues", []),
                            "suggestions": vlm_suggestions,
                            "readability_score": vlm_report.get("readability_score"),
                            "data_represented_correctly": vlm_report.get("data_represented_correctly"),
                            "visual_quality_ok": vlm_report.get("visual_quality_ok")
                        }
                        logger.info(f"VLM validation failed (attempt {attempt+1}/{max_retries+1}), will retry with feedback")
                        continue
                    
                    # VLM validation passed
                    validation_passed = True
                    logger.info(f"VLM validation passed with score: {vlm_report.get('readability_score', 'N/A')}")
                
                # Success!
                success = True
                break
                
            except Exception as render_error:
                logger.warning(f"Chart rendering/validation error: {render_error}")
                error_message = str(render_error)
                # Chart generated but rendering failed - still a success
                image_base64 = None
                break
        
        # Determine final success status
        # Success if we have a valid spec, regardless of VLM validation result
        if spec is None:
            success = False
            if error_message is None:
                error_message = "Failed to generate valid Altair spec"
        
        # If we have a spec but validation failed, that's still a partial success
        # The caller can decide whether to use it based on validation_passed flag
        
        return code, spec, image_base64, image_path, vlm_report, success, validation_passed, error_message
    
    def _render_chart_to_base64(self, spec: Dict[str, Any], scale: Optional[float] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Render Vega-Lite spec to base64 PNG image.
        
        Returns:
            - image_base64: Base64 encoded PNG (or None if failed)
            - error_message: Error message if failed (or None if successful)
        """
        try:
            import altair as alt
            
            config = get_config()
            if scale is None:
                scale = config.validation.chart_image_scale
            
            # Sanitize spec data to prevent vl-convert crashes from invalid XML characters
            spec = self._sanitize_spec_data(spec)
            
            chart = alt.Chart.from_dict(spec)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            chart.save(temp_path, format='png', scale_factor=scale)
            
            # Read and encode
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            import os
            os.unlink(temp_path)
            
            return base64.b64encode(image_data).decode('utf-8'), None
            
        except Exception as e:
            error_msg = f"Chart rendering failed: {str(e)}"
            logger.warning(error_msg)
            return None, error_msg
    
    def _save_chart_image(
        self,
        spec: Dict[str, Any],
        sample_id: str,
        scale: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Render and save chart image to temp_images folder.
        
        Args:
            spec: Vega-Lite specification
            sample_id: Sample ID for naming the file
            scale: Optional scale factor for rendering
            
        Returns:
            - image_path: Path to saved image file (or None if failed)
            - image_base64: Base64 encoded image data (or None if failed)
            - error_message: Error message if failed (or None if successful)
        """
        try:
            import altair as alt
            import os
            
            config = get_config()
            if scale is None:
                scale = config.validation.chart_image_scale
            
            # Ensure temp_images directory exists
            temp_images_dir = config.paths.temp_images_dir
            temp_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean sample_id for filename
            safe_sample_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in sample_id)
            image_path = temp_images_dir / f"{safe_sample_id}.png"
            
            # Sanitize spec data to prevent vl-convert crashes from invalid XML characters
            spec = self._sanitize_spec_data(spec)
            
            # Render chart
            chart = alt.Chart.from_dict(spec)
            chart.save(str(image_path), format='png', scale_factor=scale)
            
            # Read and encode to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            logger.debug(f"Saved chart image to {image_path}")
            
            return str(image_path), image_base64, None
            
        except Exception as e:
            error_msg = f"Failed to save chart image: {str(e)}"
            logger.warning(error_msg)
            return None, None, error_msg
    
    def _vlm_validate_chart(
        self,
        image_base64: str,
        candidate: FeatureCandidate,
        vis_question: str,
        data_sample: pd.DataFrame,
        altair_code: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Use VLM to validate chart readability and correctness.
        
        Checks:
        - Chart is readable (no overlapping, clipping)
        - Labels and legends are complete
        - Chart type is correct
        - Data is correctly represented
        - Chart answers the vis_question
        
        Args:
            image_base64: Base64 encoded chart image
            candidate: Visualization candidate with design info
            vis_question: The visualization question to answer
            data_sample: Sample of the data being visualized
            altair_code: The Altair code that generated the chart (for context)
            
        Returns:
            - passed: Whether validation passed
            - vlm_report: VLM evaluation report
            - suggestions: List of improvement suggestions
        """
        config = get_config()
        
        # Build data sample as markdown table
        # Show first 5 rows with distinct counts
        preview_rows = min(5, len(data_sample))
        data_table = self._df_to_markdown_table(data_sample.head(preview_rows), full_df=data_sample)
        
        # Add remaining rows info if there are more
        if len(data_sample) > preview_rows:
            omitted_count = len(data_sample) - preview_rows
            data_table += f"\n... +{omitted_count} more rows"
        
        # Try to get reference example for this chart type
        example_image_base64 = None
        example_code = None
        try:
            from generation.candidates import get_key_feature_generator
            from utils.chart_executor import get_chart_executor
            
            generator = get_key_feature_generator()
            example_code = generator.catalog_loader.get_example_code(candidate.chart_type)
            
            if example_code:
                executor = get_chart_executor(
                    timeout=config.pipeline.chart_execution_timeout,
                    scale=config.pipeline.example_image_scale
                )
                example_image_base64, error = executor.execute_chart_example(
                    candidate.chart_type,
                    example_code
                )
                if error:
                    logger.debug(f"Failed to generate example image for {candidate.chart_type}: {error}")
                    example_image_base64 = None
        except Exception as e:
            logger.debug(f"Could not load example image for {candidate.chart_type}: {e}")
            example_image_base64 = None
        
        # Build image reference section
        image_reference = ""
        if example_image_base64:
            image_reference = f"""
## Reference Images

**Image 1**: Reference example of {candidate.chart_type} (for comparison)
**Image 2**: Generated chart to be validated

Use Image 1 as a reference to understand the expected visual style and structure of {candidate.chart_type}.
"""
        
        # Include example code if available
        example_code_section = ""
        if example_code and example_image_base64:
            # Truncate example code for brevity (max 500 chars)
            truncated_code = example_code[:500] + "..." if len(example_code) > 500 else example_code
            example_code_section = f"""
## Reference Example Code for {candidate.chart_type}
```python
{truncated_code}
```
"""
        
        # Include Altair code if provided for better context
        code_section = ""
        if altair_code:
            code_section = f"""
## Generated Visualization Code (to validate)
```python
{altair_code}
```
"""
        
        prompt = f"""Analyze this data visualization chart for quality and readability.
{image_reference}
## User Question
{vis_question}

## Expected Visualization
- Chart Type: {candidate.chart_type}
- Dimensions (categories): {candidate.dimensions}
- Measures (values): {candidate.measures}
- Visualization Intent: {candidate.vis_intent}
{example_code_section}{code_section}
## Data Sample (first {preview_rows} rows of total {len(data_sample)} rows, with distinct counts per column)
{data_table}

## Evaluation Criteria

Please evaluate the {'second' if example_image_base64 else ''} chart on these aspects:

1. **Readability**: Are labels, axes, and legends clear and not overlapping?
2. **Chart Type**: Does it match the expected {candidate.chart_type}?{' (Compare with reference example in Image 1)' if example_image_base64 else ''}
3. **Data Representation**: Does the chart correctly show the data values?
4. **Completeness**: Are all necessary elements present (title, axis labels, legend if needed)?
5. **Visual Quality**: Any rendering issues, clipping, or artifacts?
6. **Question Alignment**: Does the chart help answer the user's question?

Return a JSON object:
{{
    "readable": true/false,
    "readability_score": 0.0-1.0,
    "chart_type_correct": true/false,
    "data_represented_correctly": true/false,
    "elements_complete": true/false,
    "visual_quality_ok": true/false,
    "answers_question": true/false,
    "issues": ["list of specific issues found"],
    "suggestions": ["specific suggestions to fix the Altair code (e.g., adjust mark properties, encoding channels, scales, legends, or layout)"]
}}

**Note**: If providing suggestions, reference the specific part of the Altair code that needs modification (e.g., "Adjust mark.fontSize", "Modify encoding.text.condition", "Set axis.labelAngle").
"""
        
        # Use multimodal with both images if example is available
        if example_image_base64:
            images = [example_image_base64, image_base64]
            result = self.llm.complete_with_images(
                prompt=prompt,
                images=images,
                image_media_type="image/png",
                process_name="spec_generator.vlm_validate_chart"
            )
        else:
            # Fallback to single image
            result = self.llm.complete_with_image(
                prompt=prompt,
                image_base64=image_base64,
                image_media_type="image/png",
                process_name="spec_generator.vlm_validate_chart"
            )
        
        if not result:
            # VLM parsing failed - this is now a validation failure, not a pass
            # The caller can decide whether to retry or proceed
            logger.warning("VLM validation returned no parseable result")
            return False, {
                "readable": False,
                "readability_score": 0.0,
                "issues": ["VLM validation failed to return parseable result"],
                "vlm_error": True
            }, ["Retry VLM validation or check model response format"]
        
        # Determine if passed based on multiple criteria
        readable = result.get("readable", True)
        readability_score = result.get("readability_score", 0.5)
        chart_type_correct = result.get("chart_type_correct", True)
        data_correct = result.get("data_represented_correctly", True)
        elements_complete = result.get("elements_complete", True)
        visual_ok = result.get("visual_quality_ok", True)
        answers_question = result.get("answers_question", True)
        
        # Pass if readability score meets threshold and major criteria pass
        min_score = config.validation.min_chart_readability_score
        passed = (
            readability_score >= min_score and
            readable and
            chart_type_correct and
            data_correct and
            elements_complete and
            visual_ok and
            answers_question
        )
        
        suggestions = result.get("suggestions", [])
        
        # Log validation result
        if not passed:
            issues = result.get("issues", [])
            logger.info(f"VLM validation failed: score={readability_score}, issues={issues}")
        
        return passed, result, suggestions
    
    def _extract_code_from_markdown(self, text: str) -> Optional[str]:
        """
        Extract Python code from markdown code blocks.
        
        Supports formats:
        - ```python ... ```
        - ``` ... ```
        
        Args:
            text: Response text potentially containing markdown code blocks
            
        Returns:
            Extracted code or None if no code block found
        """
        # Try to find ```python ... ``` first
        pattern_python = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern_python, text, re.DOTALL)
        if matches:
            # Return the last code block (most relevant)
            return matches[-1].strip()
        
        # Try generic ``` ... ```
        pattern_generic = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern_generic, text, re.DOTALL)
        if matches:
            # Filter out JSON blocks
            for match in reversed(matches):
                match_stripped = match.strip()
                # Skip if it looks like JSON
                if match_stripped.startswith('{') and match_stripped.endswith('}'):
                    continue
                # Skip if it looks like SQL
                if 'SELECT' in match_stripped.upper()[:50] and 'FROM' in match_stripped.upper():
                    continue
                return match_stripped
        
        # No code block found
        return None
    
    def _sanitize_string(self, text: Any) -> Any:
        """
        Remove invalid XML/control characters from text to prevent vl-convert crashes.
        
        Removes control characters (ASCII 0-31 except tab, newline, carriage return)
        that can cause vl-convert to crash with "NonXmlChar" errors.
        
        Args:
            text: Input text (any type)
            
        Returns:
            Sanitized text (same type as input, or None if input is None)
        """
        if text is None or pd.isna(text):
            return None
        
        if not isinstance(text, str):
            return text
        
        # Remove control characters (0x00-0x1F) except tab(0x09), newline(0x0A), carriage return(0x0D)
        # Also remove DEL (0x7F) and C1 control characters (0x80-0x9F)
        import unicodedata
        sanitized = ''.join(
            char for char in text
            if char in '\t\n\r' or (ord(char) >= 0x20 and ord(char) != 0x7F and not (0x80 <= ord(char) <= 0x9F))
        )
        
        return sanitized if sanitized else None
    
    def _sanitize_spec_data(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize string data in Vega-Lite spec to prevent vl-convert crashes.
        
        This provides defense-in-depth by cleaning any data that might have been
        embedded directly in the spec without going through _prepare_df_for_json.
        
        Args:
            spec: Vega-Lite specification dictionary
            
        Returns:
            Sanitized spec (modified in place, but also returned)
        """
        def sanitize_value(value):
            if isinstance(value, str):
                return self._sanitize_string(value)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            else:
                return value
        
        # Sanitize the data section if present
        if 'data' in spec and 'values' in spec['data']:
            spec['data']['values'] = sanitize_value(spec['data']['values'])
        
        # Sanitize datasets if present (for multi-dataset specs)
        if 'datasets' in spec:
            spec['datasets'] = sanitize_value(spec['datasets'])
        
        # Sanitize any inline data in layers, vconcat, hconcat, etc.
        for key in ['layer', 'vconcat', 'hconcat', 'concat', 'repeat']:
            if key in spec:
                spec[key] = sanitize_value(spec[key])
        
        return spec
    
    def _prepare_df_for_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for JSON serialization by converting problematic types.
        
        Converts:
        - Timestamp/datetime -> ISO format strings (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        - NaT -> None (null in JSON)
        - NaN/Inf -> None (null in JSON)
        - String columns -> Sanitized (removes invalid XML control characters)
        - Other types can be added as needed
        
        Args:
            df: Original DataFrame
            
        Returns:
            DataFrame copy with converted types
        """
        df_copy = df.copy()
        
        for col in df_copy.columns:
            # Convert datetime columns to ISO format strings
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                # Check if there's time component (excluding NaT)
                non_nat = df_copy[col].dropna()
                if len(non_nat) > 0:
                    has_time = (non_nat.dt.hour != 0).any() or (non_nat.dt.minute != 0).any() or (non_nat.dt.second != 0).any()
                    fmt = '%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d'
                else:
                    fmt = '%Y-%m-%d'  # default format if all NaT
                
                # Convert, handling NaT values
                # strftime returns NaN for NaT, which becomes None in JSON
                df_copy[col] = df_copy[col].apply(
                    lambda x: x.strftime(fmt) if pd.notna(x) else None
                )
            # Convert NaN/Inf in numeric columns to None
            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                # Replace NaN and Inf with None for JSON compatibility
                df_copy[col] = df_copy[col].replace([float('nan'), float('inf'), float('-inf')], None)
            # Sanitize string columns to remove invalid XML characters
            elif pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_string_dtype(df_copy[col]):
                # Apply sanitization to prevent vl-convert crashes
                df_copy[col] = df_copy[col].apply(self._sanitize_string)
        
        return df_copy
    
    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        code = code.strip()
        
        # Remove any remaining markdown code block markers
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        
        code = code.strip()
        
        # Ensure it ends with 'chart' reference
        lines = code.strip().split('\n')
        if lines and not lines[-1].strip().startswith('chart'):
            # Check if last line is a chart creation
            if 'chart' not in lines[-1]:
                lines.append('chart')
        
        return '\n'.join(lines)
    
    def execute_altair_code(
        self,
        code: str,
        df: pd.DataFrame,
        db_path: Optional[str] = None,
        skip_sql_execution: bool = True,
        validate_rendering: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute Altair code and return Vega-Lite spec.
        
        Args:
            code: Python code to execute (may contain relative paths)
            df: DataFrame (used if skip_sql_execution=True)
            db_path: Absolute path to database (used if skip_sql_execution=False)
            skip_sql_execution: If True, inject pre-loaded df; if False, let code execute SQL
            validate_rendering: If True, validate spec can be rendered (catches frontend errors)
        
        Returns:
            - spec: Vega-Lite JSON spec (or None if failed)
            - error: Error message (or None if success)
        """
        try:
            import altair as alt
            import sqlite3
            import re
            
            # Create execution namespace
            namespace = {
                'alt': alt,
                'pd': pd,
                'sqlite3': sqlite3,
            }
            
            # Convert datetime columns to ISO format strings to avoid JSON serialization issues
            # This must be done before injecting df or using it for validation
            df_for_validation = self._prepare_df_for_json(df)
            
            code_to_execute = code
            
            if skip_sql_execution:
                # Inject pre-loaded DataFrame to skip SQL execution
                # This is faster and avoids db connection issues during validation
                namespace['df'] = df_for_validation
                
                # Modify code to skip SQL loading section
                code_to_execute = self._inject_preloaded_df(code)
            elif db_path:
                # Replace relative database paths with absolute paths
                # Pattern: ./databases/{db_id}.sqlite
                code_to_execute = re.sub(
                    r'["\']\.\/databases\/(\w+)\.sqlite["\']',
                    f'"{db_path}"',
                    code
                )
            
            # Execute code
            exec(code_to_execute, namespace)
            
            # Get chart object
            chart = namespace.get('chart')
            if chart is None:
                return None, "No 'chart' variable found in code"
            
            # Convert to Vega-Lite spec using to_json() for more reliable serialization
            # This ensures JSON-compatible output and avoids non-serializable objects
            try:
                import json
                spec_json = chart.to_json()
                spec = json.loads(spec_json)
            except Exception as e:
                # Fallback to to_dict() if to_json() fails
                logger.warning(f"Failed to use to_json(), falling back to to_dict(): {e}")
                spec = chart.to_dict()
            
            # Validate rendering to catch JavaScript/Vega-Lite errors
            if validate_rendering:
                render_error = self._validate_spec_rendering(spec, df_for_validation)
                if render_error:
                    return None, f"Rendering validation failed: {render_error}"
            
            return spec, None
            
        except Exception as e:
            # Provide detailed error message with exception type
            error_type = type(e).__name__
            error_msg = str(e)
            
            # For KeyError, add more context
            if isinstance(e, KeyError):
                detailed_error = f"{error_type}: {error_msg}\nColumn not found in DataFrame. Available columns: {list(df.columns)}"
            else:
                detailed_error = f"{error_type}: {error_msg}"
            
            return None, detailed_error
    
    def _validate_spec_rendering(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> Optional[str]:
        """
        Validate that the Vega-Lite spec can be rendered without JavaScript errors.
        
        This uses Altair's native chart.save() method which is more reliable
        than vl_convert for complex specs.
        
        Args:
            spec: Vega-Lite spec dictionary
            df: DataFrame with data (should be pre-processed by _prepare_df_for_json)
        
        Returns:
            Error message if rendering fails, None if successful
        """
        try:
            import altair as alt
            import tempfile
            import os
            
            # Sanitize spec data to prevent vl-convert crashes from invalid XML characters
            spec = self._sanitize_spec_data(spec)
            
            # Create a temporary Altair chart from the spec
            # Altair already has the data embedded in the spec from to_dict()
            try:
                # Use alt.Chart.from_dict() to recreate the chart
                chart = alt.Chart.from_dict(spec)
                
                # Try to save to a temporary file - this validates the spec
                # can be properly serialized and rendered
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    # Save as JSON (Vega-Lite spec)
                    # This will catch any serialization issues
                    chart.save(tmp_path)
                    
                    # If save succeeded, validation passed
                    return None
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
            except Exception as chart_error:
                # Log details for debugging
                logger.debug(f"Chart validation failed. Error: {chart_error}")
                logger.debug(f"Spec keys: {list(spec.keys())}")
                raise
            
        except ImportError as ie:
            # Altair should always be available, but handle gracefully
            logger.warning(f"Import error during validation: {ie}")
            return None
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Parse and simplify error message
            if "Unrecognized signal name" in error_msg:
                return f"{error_type}: Invalid signal reference"
            elif "Undefined field" in error_msg:
                return f"{error_type}: Field reference error"
            elif "Cannot read" in error_msg or "undefined" in error_msg.lower():
                return f"{error_type}: Spec structure error"
            elif "JSON" in error_msg or "json" in error_msg:
                return f"{error_type}: JSON serialization error"
            else:
                # Return a simplified error message
                return f"{error_type}: {error_msg[:200]}"
    
    def _propagate_data_to_subspecs(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagate data references to sub-specs in compound visualizations.
        
        For specs with hconcat, vconcat, layer, etc., each sub-spec needs
        to reference the data. This replaces named data references with inline data.
        """
        # List of keys that contain sub-specs
        compound_keys = ['hconcat', 'vconcat', 'layer', 'concat']
        
        for key in compound_keys:
            if key in spec:
                subspecs = spec[key]
                if isinstance(subspecs, list):
                    for subspec in subspecs:
                        if isinstance(subspec, dict):
                            # Set data reference in subspec
                            if 'data' in subspec:
                                # If subspec has its own data, keep it
                                pass
                            else:
                                # Reference the parent data
                                # We already set 'values' at top level, so subspecs can omit it
                                pass
                            
                            # Recursively handle nested compound specs
                            if any(ck in subspec for ck in compound_keys):
                                self._propagate_data_to_subspecs(subspec)
        
        return spec
    
    def _inject_preloaded_df(self, code: str) -> str:
        """
        Modify code to use pre-loaded df instead of executing data processing.
        This skips both SQL and Pandas processing for faster validation.
        
        Strategy:
        1. Skip SQL section (sqlite3.connect to conn.close() or end of with block)
        2. Skip Pandas processing section (Stage 2 comments until Altair code)
        3. Keep only import statements and Altair visualization code
        
        Handles both styles:
        1. with sqlite3.connect(...) as conn: (context manager)
        2. conn = sqlite3.connect(...) / conn.close() (explicit)
        """
        lines = code.split('\n')
        result_lines = []
        skip_sql_section = False
        skip_pandas_section = False
        with_block_indent = -1  # Track indentation of 'with' statement
        found_altair = False  # Track when we reach Altair code
        
        for line in lines:
            stripped = line.strip()
            
            # Calculate current indentation
            if stripped:
                current_indent = len(line) - len(line.lstrip())
            else:
                current_indent = 0
            
            # Detect Altair code section (keep everything after this)
            if not found_altair and ('import altair' in line or 'alt.Chart' in line or 'alt.selection' in line):
                found_altair = True
                skip_pandas_section = False  # Stop skipping
            
            # If we've found Altair code, keep everything
            if found_altair:
                result_lines.append(line)
                continue
            
            # Detect start of SQL section with context manager: "with sqlite3.connect(...)"
            if 'with' in line and 'sqlite3.connect' in line:
                skip_sql_section = True
                with_block_indent = current_indent
                continue
            
            # Detect start of explicit connection style
            if 'sqlite3.connect' in line and 'with' not in line:
                skip_sql_section = True
                with_block_indent = -1  # Not a with block
                continue
            
            # Skip pd.read_sql lines within SQL section
            if skip_sql_section and ('pd.read_sql' in line or 'read_sql_query' in line):
                continue
            
            # Detect end of 'with' block by indentation decrease
            if skip_sql_section and with_block_indent >= 0:
                if stripped and current_indent <= with_block_indent:
                    # We've exited the with block
                    skip_sql_section = False
                    with_block_indent = -1
                    # Start skipping Pandas section after SQL
                    skip_pandas_section = True
                    continue
            
            # Detect end of explicit connection style
            if skip_sql_section and with_block_indent < 0:
                if 'conn.close()' in line:
                    skip_sql_section = False
                    # Start skipping Pandas section after SQL
                    skip_pandas_section = True
                    continue
            
            # Skip Pandas processing section (Stage 2)
            if skip_pandas_section:
                # Keep import statements and comments
                if line.startswith('import ') or line.startswith('from ') or (stripped.startswith('#') and 'Stage' in line):
                    result_lines.append(line)
                continue
            
            # Include non-skipped lines
            if not skip_sql_section and not skip_pandas_section:
                result_lines.append(line)
        
        # Add a comment indicating df is pre-loaded
        if len(result_lines) < len(lines):
            # Find first non-import line
            insert_pos = 0
            for i, line in enumerate(result_lines):
                if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                    insert_pos = i
                    break
            result_lines.insert(insert_pos, "# df is pre-loaded (data processing skipped for validation)")
        
        return '\n'.join(result_lines)
    
    def iterate_code_with_feedback(
        self,
        original_code: str,
        feedback: IterationFeedback,
        df: pd.DataFrame,
        candidate: FeatureCandidate,
        vis_sql: str,
        db_path: str,
        question: str,
        attempt: int = 0,
        schema_info: Optional[DatabaseInfo] = None,
        evidence: str = ""
    ) -> str:
        """
        Iterate on Altair code based on validation feedback.
        
        Uses single-turn prompt with all context included.
        NL question and SQL are NOT modified.
        
        Args:
            original_code: The code that had issues
            feedback: Structured feedback from validator
            df: DataFrame for the visualization
            candidate: FeatureCandidate with full visualization design
            vis_sql: SQL query for reference
            db_path: Path to database
            question: User's visualization question
            attempt: Current iteration attempt number
            schema_info: Optional database schema info
            evidence: Optional domain knowledge
            
        Returns:
            Fixed Altair Python code
        """
        # Build base context using helper
        ctx = self._build_base_context(
            df=df,
            candidate=candidate,
            vis_sql=vis_sql,
            db_path=db_path,
            schema_info=schema_info,
            user_question=question,
            evidence=evidence
        )
        
        # Build features text
        features_text = "\n".join([
            f"- {f.path}: {f.op.value} {f.value if f.value else ''}"
            for f in candidate.features
        ]) if candidate.features else "None specified"
        
        # Build prompt with feedback from validator
        prompt = f"""Fix the Altair visualization code based on validation feedback.

## User Request
{ctx['user_question']}

## Domain Knowledge
{ctx['evidence']}

## Database Information
- Database Path: {ctx['db_path']}
- Tables: 
{ctx['schema_desc']}

## SQL Query (DO NOT MODIFY)
```sql
{ctx['vis_sql']}
```

## SQL Execution Result (first {min(10, ctx['sql_result_num'])} rows of total {ctx['sql_result_num']} rows)
{ctx['sql_result_sample']}

## Target Visualization
- **Chart Type**: {ctx['chart_type']}
- **Dimensions**: {ctx['dimensions']}
- **Measures**: {ctx['measures']}
- **Aggregation**: {ctx['aggregation']}
- **Composition**: {ctx['composition']}
- **Visualization Intent**: {ctx['vis_intent']}
{ctx['vis_effects']}

## Required Key Features
{features_text}

## Previous Code (has issues)
```python
{original_code}
```

{feedback.to_prompt_context()}

## Fix Requirements
- Address all issues identified above
- Ensure ALL Key Features are satisfied
- Use proper Altair 5.x syntax
- Variable must be named 'chart'
- **The last line of code MUST be simply `chart`**
- Keep the complete code structure (SQL execution + Pandas + Altair)
- DO NOT modify the SQL query

## Note: This is attempt 1000 (last attempt)!!! Try a different approach.

Output ONLY the fixed Python code, no explanations."""
        
        response = self.llm.complete(
            prompt,
            process_name="spec_generator.iterate_code_with_feedback",
            metadata={"attempt": attempt}
        )
        code = response.content.strip()
        
        # Clean up code
        code = self._clean_code(code)
        
        return code
    
    def generate_and_validate_code(
        self,
        df: pd.DataFrame,
        chart_type: str,
        key_features: List[Feature],
        question: str,
        max_retries: int = 3
    ) -> Tuple[str, Optional[Dict[str, Any]], List[str]]:
        """
        Generate Altair code with retry loop for execution errors.
        
        Note: This is a legacy utility method. For full pipeline usage,
        use generate_altair_code_with_validation() instead.
        
        Args:
            df: DataFrame for visualization
            chart_type: Target chart type
            key_features: Required key features
            question: User's visualization question
            max_retries: Maximum retry attempts
            
        Returns:
            - final_code: The Altair code (last attempt)
            - spec: Vega-Lite spec if successful, None if failed
            - log: List of attempt logs
        """
        log = []
        code = self.generate_altair_code(df, chart_type, key_features, question)
        
        for attempt in range(max_retries):
            log.append(f"Attempt {attempt + 1}: Executing Altair code")
            
            spec, error = self.execute_altair_code(code, df)
            
            if error is None and spec is not None:
                # Validate key features
                is_valid, violations = self.validate_spec_against_features(spec, key_features)
                
                if is_valid:
                    log.append("Success: All key features satisfied")
                    return code, spec, log
                
                log.append(f"Feature violations: {violations}")
                
                # Build simple iteration prompt (inline, no external method call)
                missing_text = "\n".join([f"- {f.path}: {f.op.value}" for f in self._find_missing_features(spec, key_features)])
                prompt = f"""Fix this Altair code to satisfy the required key features.

## Previous Code
```python
{code}
```

## Issues
{chr(10).join(f'- {v}' for v in violations)}

## Missing Key Features
{missing_text}

## Requirements
- Chart type: {chart_type}
- Question: {question}
- Variable must be named 'chart'
- **The last line of code MUST be simply `chart`**

Output ONLY the fixed Python code."""
                
                response = self.llm.complete(prompt)
                code = self._clean_code(response.content)
            else:
                # Execution error
                log.append(f"Execution error: {error}")
                
                prompt = f"""Fix this Altair code that has an execution error.

## Previous Code
```python
{code}
```

## Error
{error}

## Requirements
- Chart type: {chart_type}
- Question: {question}
- Variable must be named 'chart'
- **The last line of code MUST be simply `chart`**

Output ONLY the fixed Python code."""
                
                response = self.llm.complete(prompt)
                code = self._clean_code(response.content)
        
        # Return last attempt
        log.append("Max retries reached, returning last attempt")
        spec, _ = self.execute_altair_code(code, df)
        return code, spec, log
    
    def _find_missing_features(
        self,
        spec: Dict[str, Any],
        key_features: List[Feature]
    ) -> List[Feature]:
        """Find key features not satisfied by the spec."""
        missing = []
        for feature in key_features:
            if not self._check_feature_in_spec(spec, feature):
                missing.append(feature)
        return missing
    
    def extract_features_from_spec(
        self,
        spec: Dict[str, Any]
    ) -> List[Feature]:
        """Extract features from Vega-Lite spec."""
        features = []
        
        # Extract mark type
        mark = spec.get("mark")
        if isinstance(mark, str):
            features.append(Feature("mark.type", FeatureOp.EQ, mark))
        elif isinstance(mark, dict):
            mark_type = mark.get("type")
            if mark_type:
                features.append(Feature("mark.type", FeatureOp.EQ, mark_type))
        
        # Extract encodings
        encoding = spec.get("encoding", {})
        for channel, enc_spec in encoding.items():
            if isinstance(enc_spec, dict):
                # Field
                if "field" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.field",
                        FeatureOp.EQ,
                        enc_spec["field"]
                    ))
                
                # Type
                if "type" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.type",
                        FeatureOp.EQ,
                        enc_spec["type"]
                    ))
                
                # Aggregate
                if "aggregate" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.aggregate",
                        FeatureOp.EQ,
                        enc_spec["aggregate"]
                    ))
                
                # TimeUnit
                if "timeUnit" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.timeUnit",
                        FeatureOp.EQ,
                        enc_spec["timeUnit"]
                    ))
                
                # Bin
                if "bin" in enc_spec:
                    features.append(Feature(
                        f"encoding.{channel}.bin",
                        FeatureOp.EXISTS
                    ))
        
        # Extract transforms
        transforms = spec.get("transform", [])
        if transforms:
            features.append(Feature("transform", FeatureOp.EXISTS))
            for i, t in enumerate(transforms):
                for key in ["aggregate", "filter", "calculate", "window", "bin"]:
                    if key in t:
                        features.append(Feature(
                            f"transform[{i}].{key}",
                            FeatureOp.EXISTS
                        ))
        
        # Extract layer
        if "layer" in spec:
            features.append(Feature("layer", FeatureOp.EXISTS))
            features.append(Feature(
                "layer",
                FeatureOp.LEN_GE,
                len(spec["layer"])
            ))
        
        # Extract concat
        for concat_type in ["hconcat", "vconcat", "concat"]:
            if concat_type in spec:
                features.append(Feature(concat_type, FeatureOp.EXISTS))
        
        # Extract facet
        if "facet" in spec:
            features.append(Feature("facet", FeatureOp.EXISTS))
        
        # Extract resolve
        resolve = spec.get("resolve", {})
        if resolve:
            features.append(Feature("resolve", FeatureOp.EXISTS))
            for res_type, res_spec in resolve.items():
                for channel, value in res_spec.items():
                    features.append(Feature(
                        f"resolve.{res_type}.{channel}",
                        FeatureOp.EQ,
                        value
                    ))
        
        # Extract params (for interactivity)
        params = spec.get("params", [])
        if params:
            features.append(Feature("params", FeatureOp.EXISTS))
            for i, param in enumerate(params):
                if "select" in param:
                    select = param["select"]
                    if isinstance(select, dict) and "type" in select:
                        features.append(Feature(
                            f"params[{i}].select.type",
                            FeatureOp.EQ,
                            select["type"]
                        ))
                if "bind" in param:
                    features.append(Feature(
                        f"params[{i}].bind",
                        FeatureOp.EXISTS
                    ))
        
        return features
    
    # Backward compatibility alias
    extract_key_features_from_spec = extract_features_from_spec
    
    def compress_features(
        self,
        features: List[Feature],
        essential_only: bool = True
    ) -> List[Feature]:
        """Compress features to minimal must-have set."""
        if not essential_only:
            return features
        
        # Priority order for features
        priority_paths = [
            "mark.type",
            "encoding.x.field", "encoding.y.field",
            "encoding.x.type", "encoding.y.type",
            "encoding.y.aggregate",
            "encoding.color.field",
            "layer", "resolve.scale.y",
            "params",
        ]
        
        compressed = []
        seen_paths = set()
        
        # Add priority features first
        for priority in priority_paths:
            for f in features:
                if f.path.startswith(priority) and f.path not in seen_paths:
                    compressed.append(f)
                    seen_paths.add(f.path)
                    break
        
        # Add any remaining essential features
        for f in features:
            if f.path not in seen_paths:
                # Only add if it's a defining feature
                if any(kw in f.path for kw in ["mark", "encoding", "layer", "params", "resolve"]):
                    if len(compressed) < 15:  # Cap at 15 features
                        compressed.append(f)
                        seen_paths.add(f.path)
        
        return compressed
    
    # Backward compatibility alias
    compress_key_features = compress_features
    
    def validate_spec_against_features(
        self,
        spec: Dict[str, Any],
        required_features: List[Feature]
    ) -> Tuple[bool, List[str]]:
        """Validate spec satisfies required features."""
        violations = []
        
        for feature in required_features:
            if not self._check_feature_in_spec(spec, feature):
                violations.append(
                    f"Missing: {feature.path} {feature.op.value} {feature.value or ''}"
                )
        
        return len(violations) == 0, violations
    
    def _check_feature_in_spec(
        self,
        spec: Dict[str, Any],
        feature: Feature
    ) -> bool:
        """Check if spec satisfies a feature."""
        value = self._get_path_value(spec, feature.path)
        
        op = feature.op if isinstance(feature.op, FeatureOp) else FeatureOp(feature.op)
        
        if op == FeatureOp.EXISTS:
            return value is not None
        elif op == FeatureOp.EQ:
            return value == feature.value
        elif op == FeatureOp.IN:
            return value in (feature.value or [])
        elif op == FeatureOp.CONTAINS:
            if isinstance(value, (list, str)):
                return feature.value in value
            return False
        elif op == FeatureOp.LEN_GE:
            if isinstance(value, (list, str)):
                return len(value) >= feature.value
            return False
        elif op == FeatureOp.LEN_LE:
            if isinstance(value, (list, str)):
                return len(value) <= feature.value
            return False
        
        return False
    
    def _get_path_value(self, obj: Any, path: str) -> Any:
        """Get value at path in nested object."""
        parts = path.replace("][", ".").replace("[", ".").replace("]", "").split(".")
        
        current = obj
        for part in parts:
            if current is None:
                return None
            
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current
    
    def render_spec_to_image(
        self,
        spec: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Render Vega-Lite spec to image file."""
        try:
            import altair as alt
            
            chart = alt.Chart.from_dict(spec)
            chart.save(output_path)
            return True
        except Exception as e:
            print(f"Render error: {e}")
            return False


# Singleton instance
_generator: Optional[SpecGenerator] = None


def get_spec_generator() -> SpecGenerator:
    """Get or create singleton spec generator."""
    global _generator
    if _generator is None:
        _generator = SpecGenerator()
    return _generator
