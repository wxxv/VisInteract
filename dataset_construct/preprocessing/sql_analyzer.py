"""
SQL Processor for BIRD Mini-Dev SQLite databases.
Provides SQL parsing, semantic summary extraction, and SQL transformation for visualization.
"""
from typing import Any, Dict, List, Optional
import json
import re
import logging

try:
    import sqlglot
    from sqlglot import exp, parse_one
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from core.models import SQLSemanticSummary, FeatureCandidate, DatabaseInfo
from utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class SQLProcessor:
    """Processes SQL queries for semantic analysis and transformation."""
    
    def __init__(self):
        if not HAS_SQLGLOT:
            logger.warning("sqlglot not installed, using LLM fallback for SQL parsing")
        self.dialect = "sqlite"  # Changed from snowflake to sqlite
    
    def parse_sql(self, sql: str) -> Optional[exp.Expression]:
        """Parse SQL string into AST."""
        if not HAS_SQLGLOT:
            return None
        
        try:
            return parse_one(sql, dialect=self.dialect)
        except Exception as e:
            logger.debug(f"SQL parse error: {e}")
            return None
    
    def canonicalize_sql(self, sql: str) -> str:
        """Convert SQL to canonical form."""
        if not HAS_SQLGLOT:
            return sql.strip()
        
        try:
            ast = parse_one(sql, dialect=self.dialect)
            return ast.sql(dialect=self.dialect, pretty=True)
        except Exception:
            return sql.strip()
    
    def extract_tables(self, ast: exp.Expression) -> List[str]:
        """Extract table names from AST."""
        tables = []
        for table in ast.find_all(exp.Table):
            if table.name:
                tables.append(table.name)
        return list(set(tables))
    
    def extract_columns(self, ast: exp.Expression) -> List[str]:
        """Extract column references from AST."""
        columns = []
        for col in ast.find_all(exp.Column):
            columns.append(col.sql(dialect=self.dialect))
        return list(set(columns))
    
    def extract_select_fields(self, ast: exp.Expression) -> List[str]:
        """Extract SELECT field expressions."""
        fields = []
        select = ast.find(exp.Select)
        if select:
            for expr in select.expressions:
                fields.append(expr.sql(dialect=self.dialect))
        return fields
    
    def extract_joins(self, ast: exp.Expression) -> List[Dict[str, str]]:
        """Extract JOIN clauses from AST."""
        joins = []
        for join in ast.find_all(exp.Join):
            try:
                on_clause = join.args.get("on")
                join_info = {
                    "type": join.kind or "INNER",
                    "table": join.this.sql(dialect=self.dialect) if join.this else "",
                    "on": on_clause.sql(dialect=self.dialect) if on_clause else ""
                }
                joins.append(join_info)
            except Exception:
                # Skip malformed join clauses
                continue
        return joins
    
    def extract_filters(self, ast: exp.Expression) -> List[str]:
        """Extract WHERE/HAVING conditions from AST."""
        filters = []
        
        try:
            # WHERE clause
            where = ast.find(exp.Where)
            if where and where.this:
                filters.append(f"WHERE: {where.this.sql(dialect=self.dialect)}")
            
            # HAVING clause
            having = ast.find(exp.Having)
            if having and having.this:
                filters.append(f"HAVING: {having.this.sql(dialect=self.dialect)}")
        except Exception:
            pass
        
        return filters
    
    def extract_groupby(self, ast: exp.Expression) -> List[str]:
        """Extract GROUP BY dimensions from AST."""
        dims = []
        group = ast.find(exp.Group)
        if group:
            for expr in group.expressions:
                dims.append(expr.sql(dialect=self.dialect))
        return dims
    
    def extract_aggregations(self, ast: exp.Expression) -> List[Dict[str, str]]:
        """Extract aggregate functions from AST."""
        aggs = []
        agg_funcs = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX', 'TOTAL', 'GROUP_CONCAT']
        
        for func in ast.find_all(exp.Func):
            func_name = func.sql_name().upper()
            if func_name in agg_funcs:
                # Filter out non-expression args (e.g., distinct=True is a bool)
                args = []
                for arg in func.args.values():
                    if arg and hasattr(arg, 'sql'):
                        args.append(arg.sql(dialect=self.dialect))
                aggs.append({
                    "function": func_name,
                    "args": args
                })
        
        return aggs
    
    def extract_order_limit(self, ast: exp.Expression) -> Optional[Dict[str, Any]]:
        """Extract ORDER BY and LIMIT from AST."""
        result = {}
        
        try:
            # ORDER BY
            order = ast.find(exp.Order)
            if order:
                order_items = []
                for item in order.expressions:
                    if item.this:
                        order_items.append({
                            "expr": item.this.sql(dialect=self.dialect),
                            "desc": item.args.get("desc", False)
                        })
                if order_items:
                    result["order_by"] = order_items
            
            # LIMIT
            limit = ast.find(exp.Limit)
            if limit and limit.this:
                result["limit"] = limit.this.sql(dialect=self.dialect)
        except Exception:
            pass
        
        return result if result else None
    
    def extract_window_ops(self, ast: exp.Expression) -> List[str]:
        """Extract window function operations from AST."""
        windows = []
        try:
            for window in ast.find_all(exp.Window):
                if window:
                    windows.append(window.sql(dialect=self.dialect))
        except Exception:
            pass
        return windows
    
    def extract_ctes(self, ast: exp.Expression) -> List[str]:
        """Extract CTE names from AST."""
        ctes = []
        with_clause = ast.find(exp.With)
        if with_clause:
            for cte in with_clause.expressions:
                if isinstance(cte, exp.CTE):
                    ctes.append(cte.alias)
        return ctes
    
    def build_semantic_summary(self, sql: str) -> SQLSemanticSummary:
        """Build structured semantic summary from SQL."""
        ast = self.parse_sql(sql)
        
        if ast:
            return SQLSemanticSummary(
                tables=self.extract_tables(ast),
                joins=self.extract_joins(ast),
                filters=self.extract_filters(ast),
                groupby_dims=self.extract_groupby(ast),
                measures=self.extract_aggregations(ast),
                order_limit=self.extract_order_limit(ast),
                window_ops=self.extract_window_ops(ast),
                ctes=self.extract_ctes(ast),
                select_fields=self.extract_select_fields(ast),
                sql=sql  # Store original SQL query
            )
        else:
            # Fallback to LLM
            summary = self._llm_extract_summary(sql)
            summary.sql = sql  # Store original SQL query
            return summary
    
    def _llm_extract_summary(self, sql: str) -> SQLSemanticSummary:
        """Use LLM to extract semantic summary when parser fails."""
        llm = get_llm_client()
        
        prompt = f"""Analyze this SQLite SQL query and extract its semantic structure.

SQL:
```sql
{sql}
```

Return a JSON object with these fields:
- tables: list of table names used
- joins: list of {{"type", "table", "on"}} for each join
- filters: list of WHERE/HAVING conditions as strings
- groupby_dims: list of GROUP BY columns
- measures: list of {{"function", "args"}} for aggregations (e.g., SUM, COUNT, AVG)
- order_limit: {{"order_by": [{{"expr", "desc"}}], "limit": number}} or null
- window_ops: list of window function expressions
- ctes: list of CTE names
- select_fields: list of SELECT expressions

Only output the JSON object, no explanations.
"""
        
        result = llm.complete_json(
            prompt,
            process_name="sql_processor.llm_extract_summary"
        )
        
        if result:
            return SQLSemanticSummary(
                tables=result.get("tables", []),
                joins=result.get("joins", []),
                filters=result.get("filters", []),
                groupby_dims=result.get("groupby_dims", []),
                measures=result.get("measures", []),
                order_limit=result.get("order_limit"),
                window_ops=result.get("window_ops", []),
                ctes=result.get("ctes", []),
                select_fields=result.get("select_fields", [])
            )
        else:
            return SQLSemanticSummary()
    
    def validate_columns_exist(
        self,
        sql: str,
        available_columns: List[str]
    ) -> List[str]:
        """Check which columns in SQL don't exist in available columns."""
        ast = self.parse_sql(sql)
        if not ast:
            return []
        
        sql_columns = self.extract_columns(ast)
        available_upper = [c.upper() for c in available_columns]
        
        missing = []
        for col in sql_columns:
            # Extract just the column name (not table prefix)
            col_name = col.split(".")[-1].strip('"').upper()
            if col_name not in available_upper and col_name != '*':
                missing.append(col)
        
        return missing
    
    def get_intent_text(self, summary: SQLSemanticSummary) -> str:
        """Generate human-readable intent text from summary."""
        parts = []
        
        if summary.tables:
            parts.append(f"Tables: {', '.join(summary.tables)}")
        
        if summary.measures:
            measures_str = ", ".join(
                f"{m['function']}({', '.join(m.get('args', []))})"
                for m in summary.measures
            )
            parts.append(f"Measures: {measures_str}")
        
        if summary.groupby_dims:
            parts.append(f"Grouped by: {', '.join(summary.groupby_dims)}")
        
        if summary.filters:
            parts.append(f"Filters: {'; '.join(summary.filters)}")
        
        if summary.order_limit:
            if "order_by" in summary.order_limit:
                order_parts = [
                    f"{o['expr']} {'DESC' if o.get('desc') else 'ASC'}"
                    for o in summary.order_limit["order_by"]
                ]
                parts.append(f"Ordered by: {', '.join(order_parts)}")
            if "limit" in summary.order_limit:
                parts.append(f"Limited to: {summary.order_limit['limit']}")
        
        return "\n".join(parts) if parts else "Unable to extract intent"
    
    # =========================================================================
    # SQL Transformation for Visualization
    # =========================================================================
    
    def transform_sql_for_vis(
        self,
        gold_sql: str,
        candidate: FeatureCandidate,
        schema_info: DatabaseInfo,
        min_rows: int = 5,
        max_rows: int = 100
    ) -> str:
        """
        Transform a Gold SQL (designed for precise answers) into a visualization-friendly SQL.
        
        Typical transformations:
        - Remove LIMIT 1 or expand to Top-K
        - Add GROUP BY to get multiple rows
        - Adjust aggregations based on vis intent
        - Ensure dimensions and measures are in SELECT
        
        Args:
            gold_sql: Original SQL from BIRD dataset
            candidate: Target visualization candidate
            schema_info: Database schema information
            min_rows: Minimum desired result rows
            max_rows: Maximum desired result rows
            
        Returns:
            Transformed SQL suitable for visualization
        """
        llm = get_llm_client()
        
        # Build schema context
        schema_digest = self._build_schema_context(schema_info)
        
        prompt = f"""Transform the following SQL to make it suitable for generating a {candidate.chart_type} visualization.

## Original SQL (from text-to-SQL task)
```sql
{gold_sql}
```

## Target Visualization
- Chart Type: {candidate.chart_type}
- Dimensions (X-axis / categories): {', '.join(candidate.dimensions)}
- Measures (Y-axis / values): {', '.join(candidate.measures)}
- Aggregation: {candidate.aggregation}
- Visualization Intent: {candidate.vis_intent}

## Database Schema
{schema_digest}

## Transformation Requirements
1. The result should have {min_rows}-{max_rows} rows (suitable for visualization)
2. SELECT clause must include the dimension and measure fields
3. If the original SQL has LIMIT 1, consider removing it or expanding to show more data
4. If the original SQL returns a single value, add GROUP BY to get multiple data points
5. Use {candidate.aggregation} aggregation for measures
6. Keep the semantic relationship with the original query (same tables, related filters)
7. Ensure the SQL is valid SQLite syntax

## Output
Return ONLY the transformed SQL code, no explanations or markdown.
"""
        
        response = llm.complete(
            prompt,
            process_name="sql_processor.transform_sql_for_vis"
        )
        
        # Extract content from LLMResponse and clean up
        result_text = response.content if response and response.content else ""
        transformed_sql = self._clean_sql_output(result_text)
        
        return transformed_sql if transformed_sql else gold_sql
    
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
    
    def _clean_sql_output(self, output: str) -> str:
        """Clean LLM SQL output, removing markdown and extra text."""
        if not output:
            return ""
        
        # Remove markdown code blocks
        output = re.sub(r'```sql\s*', '', output)
        output = re.sub(r'```\s*', '', output)
        
        # Remove leading/trailing whitespace
        output = output.strip()
        
        # Remove any text before SELECT (if there's explanation)
        match = re.search(r'\bSELECT\b', output, re.IGNORECASE)
        if match:
            output = output[match.start():]
        
        return output
    
    def quick_transform_remove_limit1(self, sql: str) -> str:
        """Quick transformation: remove LIMIT 1 from SQL."""
        # Simple regex replacement for LIMIT 1
        transformed = re.sub(r'\bLIMIT\s+1\b', '', sql, flags=re.IGNORECASE)
        return transformed.strip()
    
    def quick_transform_expand_limit(self, sql: str, new_limit: int = 20) -> str:
        """Quick transformation: expand LIMIT to show more rows."""
        # Replace existing LIMIT with new value
        transformed = re.sub(
            r'\bLIMIT\s+\d+\b', 
            f'LIMIT {new_limit}', 
            sql, 
            flags=re.IGNORECASE
        )
        
        # If no LIMIT was found, add one
        if 'LIMIT' not in transformed.upper():
            transformed = f"{transformed.rstrip().rstrip(';')} LIMIT {new_limit}"
        
        return transformed
    
    def analyze_sql_vis_compatibility(self, sql: str) -> Dict[str, Any]:
        """
        Analyze SQL to determine its visualization compatibility.
        
        Returns:
            Dict with:
            - is_aggregated: bool - has aggregate functions
            - has_groupby: bool - has GROUP BY
            - estimated_rows: str - 'single', 'few', 'many'
            - limit_value: int or None
            - suggestions: list of improvement suggestions
        """
        summary = self.build_semantic_summary(sql)
        
        is_aggregated = len(summary.measures) > 0
        has_groupby = len(summary.groupby_dims) > 0
        
        # Determine estimated rows
        limit_value = None
        if summary.order_limit and 'limit' in summary.order_limit:
            try:
                limit_value = int(summary.order_limit['limit'])
            except (ValueError, TypeError):
                pass
        
        if limit_value == 1:
            estimated_rows = 'single'
        elif limit_value and limit_value < 10:
            estimated_rows = 'few'
        elif is_aggregated and not has_groupby:
            estimated_rows = 'single'
        else:
            estimated_rows = 'many'
        
        # Generate suggestions
        suggestions = []
        if estimated_rows == 'single':
            suggestions.append("Add GROUP BY to get multiple data points")
            suggestions.append("Remove LIMIT 1 or expand to show top N")
        if not is_aggregated and not has_groupby:
            suggestions.append("Consider adding aggregation for visualization")
        
        return {
            'is_aggregated': is_aggregated,
            'has_groupby': has_groupby,
            'estimated_rows': estimated_rows,
            'limit_value': limit_value,
            'suggestions': suggestions,
            'tables': summary.tables,
            'measures': summary.measures,
            'groupby_dims': summary.groupby_dims
        }


# Singleton instance
_processor: Optional[SQLProcessor] = None


def get_sql_processor() -> SQLProcessor:
    """Get or create singleton SQL processor."""
    global _processor
    if _processor is None:
        _processor = SQLProcessor()
    return _processor
