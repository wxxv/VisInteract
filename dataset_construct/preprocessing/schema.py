"""
Schema Processor for BIRD Mini-Dev SQLite databases.
Parses dev_tables.json and builds DatabaseInfo for each database.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.models import ColumnInfo, TableInfo, DatabaseInfo, ForeignKeyInfo, TableGroup
from core.config import get_config
from execution.sqlite_client import get_sqlite_client


class SchemaProcessor:
    """
    Processes database schemas from BIRD Mini-Dev resources.
    Primary source: dev_tables.json
    Secondary source: Live SQLite queries for sample values
    """
    
    def __init__(
        self, 
        dev_tables_path: Optional[Path] = None,
        databases_dir: Optional[Path] = None
    ):
        """
        Initialize schema processor.
        
        Args:
            dev_tables_path: Path to dev_tables.json
            databases_dir: Path to dev_databases directory
        """
        config = get_config()
        self.dev_tables_path = dev_tables_path or config.paths.dev_tables_json
        self.databases_dir = databases_dir or config.paths.dev_databases_dir
        
        # Schema cache by db_id
        self._schema_cache: Dict[str, DatabaseInfo] = {}
        
        # Load dev_tables.json
        self._dev_tables_data: List[Dict[str, Any]] = []
        self._db_id_to_schema: Dict[str, Dict[str, Any]] = {}
        self._load_dev_tables()
    
    def _load_dev_tables(self):
        """Load and index dev_tables.json."""
        if not self.dev_tables_path.exists():
            raise FileNotFoundError(f"dev_tables.json not found: {self.dev_tables_path}")
        
        with open(self.dev_tables_path, "r", encoding="utf-8") as f:
            self._dev_tables_data = json.load(f)
        
        # Index by db_id
        for schema_data in self._dev_tables_data:
            db_id = schema_data.get("db_id", "")
            if db_id:
                self._db_id_to_schema[db_id] = schema_data
    
    def get_available_db_ids(self) -> List[str]:
        """Get list of all available database IDs."""
        return list(self._db_id_to_schema.keys())
    
    def build_schema_catalog(self, db_id: str, include_sample_values: bool = False) -> Optional[DatabaseInfo]:
        """
        Build complete schema catalog for a database.
        
        Args:
            db_id: Database identifier
            include_sample_values: Whether to query sample values from SQLite
            
        Returns:
            DatabaseInfo or None if db_id not found
        """
        # Check cache first
        cache_key = f"{db_id}_{include_sample_values}"
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        
        schema_data = self._db_id_to_schema.get(db_id)
        if not schema_data:
            return None
        
        # Parse the BIRD dev_tables.json format
        db_info = self._parse_bird_schema(schema_data, include_sample_values)
        
        self._schema_cache[cache_key] = db_info
        return db_info
    
    def _parse_bird_schema(
        self, 
        schema_data: Dict[str, Any],
        include_sample_values: bool = False
    ) -> DatabaseInfo:
        """
        Parse BIRD dev_tables.json format into DatabaseInfo.
        
        BIRD format:
        {
            "db_id": "california_schools",
            "table_names_original": ["frpm", "satscores", "schools"],
            "column_names_original": [[-1, "*"], [0, "CDSCode"], [0, "Academic Year"], ...],
            "column_types": ["text", "integer", ...],
            "foreign_keys": [[col_idx1, col_idx2], ...],
            "primary_keys": [col_idx, ...]
        }
        """
        db_id = schema_data.get("db_id", "")
        table_names = schema_data.get("table_names_original", [])
        column_names = schema_data.get("column_names_original", [])
        column_types = schema_data.get("column_types", [])
        foreign_keys = schema_data.get("foreign_keys", [])
        primary_keys = schema_data.get("primary_keys", [])
        
        # Convert primary_keys to set for quick lookup
        # Handle composite primary keys (nested lists like [21, 22])
        pk_indices = set()
        for pk in primary_keys:
            if isinstance(pk, list):
                # Composite primary key - add all column indices
                pk_indices.update(pk)
            else:
                pk_indices.add(pk)
        
        # Build column index to table/column mapping
        # column_names format: [[table_idx, col_name], ...]
        # table_idx -1 means the special "*" column, skip it
        
        # Group columns by table
        table_columns: Dict[int, List[Tuple[int, str, str]]] = {}
        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx < 0:
                continue  # Skip special "*" column
            
            col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
            
            if table_idx not in table_columns:
                table_columns[table_idx] = []
            
            table_columns[table_idx].append((col_idx, col_name, col_type))
        
        # Build TableInfo list
        tables: List[TableInfo] = []
        for table_idx, table_name in enumerate(table_names):
            columns: List[ColumnInfo] = []
            
            for col_idx, col_name, col_type in table_columns.get(table_idx, []):
                # Check if this column is a primary key
                is_pk = col_idx in pk_indices
                
                # Check if this column is part of a foreign key
                is_fk = any(col_idx in fk_pair for fk_pair in foreign_keys)
                
                column = ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    table_name=table_name,
                    is_primary_key=is_pk,
                    is_foreign_key=is_fk
                )
                
                columns.append(column)
            
            table_info = TableInfo(
                name=table_name,
                columns=columns
            )
            tables.append(table_info)
        
        # Parse foreign key relationships
        fk_list: List[ForeignKeyInfo] = []
        for fk_pair in foreign_keys:
            if len(fk_pair) != 2:
                continue
            
            from_idx, to_idx = fk_pair
            
            # Find table and column for each index
            from_info = self._get_table_column_by_idx(column_names, table_names, from_idx)
            to_info = self._get_table_column_by_idx(column_names, table_names, to_idx)
            
            if from_info and to_info:
                fk_list.append(ForeignKeyInfo(
                    from_table=from_info[0],
                    from_column=from_info[1],
                    to_table=to_info[0],
                    to_column=to_info[1]
                ))
        
        db_info = DatabaseInfo(
            db_id=db_id,
            tables=tables,
            foreign_keys=fk_list
        )
        
        # Optionally get sample values from SQLite
        if include_sample_values:
            self._add_sample_values(db_info)
        
        return db_info
    
    def _get_table_column_by_idx(
        self,
        column_names: List[List],
        table_names: List[str],
        col_idx: int
    ) -> Optional[Tuple[str, str]]:
        """Get (table_name, column_name) by column index."""
        if col_idx < 0 or col_idx >= len(column_names):
            return None
        
        table_idx, col_name = column_names[col_idx]
        if table_idx < 0 or table_idx >= len(table_names):
            return None
        
        return (table_names[table_idx], col_name)
    
    def _add_sample_values(self, db_info: DatabaseInfo):
        """Add sample values to columns by querying SQLite."""
        try:
            client = get_sqlite_client()
        except Exception:
            return  # Skip if client not available
        
        for table in db_info.tables:
            for column in table.columns:
                try:
                    samples = client.get_sample_values(
                        db_info.db_id, 
                        table.name, 
                        column.name, 
                        limit=5
                    )
                    column.sample_values = samples
                except Exception:
                    pass  # Skip on error
            
            # Get row count
            try:
                table.row_count = client.get_row_count(db_info.db_id, table.name)
            except Exception:
                pass
    
    def build_schema_slice(
        self,
        db_info: DatabaseInfo,
        relevant_tables: List[str]
    ) -> DatabaseInfo:
        """Create a slice of schema containing only relevant tables."""
        relevant_tables_lower = [t.lower() for t in relevant_tables]
        
        filtered_tables = [
            t for t in db_info.tables
            if t.name.lower() in relevant_tables_lower
        ]
        
        # Also filter foreign keys to only include relevant tables
        filtered_fks = [
            fk for fk in db_info.foreign_keys
            if fk.from_table.lower() in relevant_tables_lower
            and fk.to_table.lower() in relevant_tables_lower
        ]
        
        return DatabaseInfo(
            db_id=db_info.db_id,
            tables=filtered_tables,
            foreign_keys=filtered_fks,
            evidence=db_info.evidence
        )
    
    def get_schema_digest(
        self, 
        db_info: DatabaseInfo, 
        max_cols: int = 10,
        include_fks: bool = True
    ) -> str:
        """Generate schema digest for LLM context."""
        lines = [f"Database: {db_info.db_id}"]
        lines.append("")
        
        for table in db_info.tables:
            # Table header
            row_info = f" ({table.row_count} rows)" if table.row_count else ""
            lines.append(f"Table: {table.name}{row_info}")
            
            # Columns
            cols = table.columns[:max_cols]
            for col in cols:
                pk_marker = " [PK]" if col.is_primary_key else ""
                fk_marker = " [FK]" if col.is_foreign_key else ""
                lines.append(f"  - {col.name}: {col.data_type}{pk_marker}{fk_marker}")
            
            if len(table.columns) > max_cols:
                lines.append(f"  ... +{len(table.columns) - max_cols} more columns")
            
            lines.append("")
        
        # Foreign key relationships
        if include_fks and db_info.foreign_keys:
            lines.append("Foreign Keys:")
            for fk in db_info.foreign_keys:
                lines.append(f"  {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
        
        return "\n".join(lines)
    
    def get_schema_digest_compact(
        self,
        db_info: DatabaseInfo,
        instruction: str = "",
        max_tables: int = 20,
        max_cols: int = 8
    ) -> str:
        """
        Generate compact schema digest for LLM context.
        Prioritizes tables relevant to the instruction.
        """
        if not db_info:
            return ""
        
        total_tables = len(db_info.tables)
        lines = [f"Database: {db_info.db_id} ({total_tables} tables)"]
        
        # Select relevant tables if instruction provided
        if instruction:
            selected_names = self.select_relevant_tables_by_text(
                db_info, instruction, max_tables=max_tables
            )
        else:
            selected_names = [t.name for t in db_info.tables[:max_tables]]
        
        # Build table lookup
        table_map = {t.name.lower(): t for t in db_info.tables}
        
        for name in selected_names:
            t = table_map.get(name.lower())
            if not t:
                continue
            
            cols = t.columns[:max_cols]
            col_strs = [f"{c.name}:{c.data_type}" for c in cols]
            if len(t.columns) > max_cols:
                col_strs.append(f"+{len(t.columns) - max_cols}")
            
            lines.append(f"  {t.name}: {', '.join(col_strs)}")
        
        if total_tables > len(selected_names):
            lines.append(f"  (omitted {total_tables - len(selected_names)} tables)")
        
        return "\n".join(lines)
    
    def select_relevant_tables_by_text(
        self,
        db_info: DatabaseInfo,
        text: str,
        max_tables: int = 12
    ) -> List[str]:
        """
        Select relevant tables based on text overlap with table/column names.
        """
        if not db_info or not db_info.tables:
            return []
        
        query_tokens = set(self._tokenize(text))
        if not query_tokens:
            return [t.name for t in db_info.tables[:max_tables]]
        
        scored: List[Tuple[float, str]] = []
        
        for table in db_info.tables:
            name_tokens = set(self._tokenize(table.name))
            col_tokens = set()
            for c in table.columns[:15]:
                col_tokens.update(self._tokenize(c.name))
            
            # Score: name overlap * 2 + column overlap
            overlap = len(query_tokens & name_tokens) * 2 + len(query_tokens & col_tokens)
            scored.append((float(overlap), table.name))
        
        # Sort by score descending
        scored.sort(key=lambda x: (-x[0], x[1]))
        
        # If all scores are 0, return first N tables
        if scored and scored[0][0] <= 0:
            return [t.name for t in db_info.tables[:max_tables]]
        
        return [name for _, name in scored[:max_tables]]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for matching."""
        if not text:
            return []
        tokens = re.split(r"[^A-Za-z0-9_]+", text.upper())
        return [t for t in tokens if len(t) >= 2]
    
    def find_tables_for_columns(
        self,
        db_info: DatabaseInfo,
        column_names: List[str]
    ) -> List[str]:
        """Find tables containing specified columns."""
        column_names_lower = [c.lower() for c in column_names]
        matching_tables = []
        
        for table in db_info.tables:
            table_cols_lower = [c.name.lower() for c in table.columns]
            if any(col in table_cols_lower for col in column_names_lower):
                matching_tables.append(table.name)
        
        return matching_tables
    
    def get_column_by_name(
        self,
        db_info: DatabaseInfo,
        column_name: str
    ) -> Optional[Tuple[str, ColumnInfo]]:
        """Find a column by name across all tables. Returns (table_name, column_info)."""
        column_name_lower = column_name.lower()
        
        for table in db_info.tables:
            for col in table.columns:
                if col.name.lower() == column_name_lower:
                    return (table.name, col)
        
        return None
    
    def infer_column_semantics(self, column: ColumnInfo) -> str:
        """
        Infer semantic type of a column based on name and type.
        Returns: 'temporal', 'categorical', 'numeric', 'identifier'
        """
        name_lower = column.name.lower()
        type_lower = column.data_type.lower()
        
        # Temporal patterns
        temporal_patterns = ['date', 'time', 'year', 'month', 'day', 'created', 'updated', 'timestamp']
        if any(p in name_lower for p in temporal_patterns):
            return 'temporal'
        
        # Numeric types
        if any(t in type_lower for t in ['int', 'real', 'float', 'double', 'decimal', 'numeric']):
            # Check if it's an ID
            if name_lower.endswith('id') or name_lower == 'id':
                return 'identifier'
            return 'numeric'
        
        # Categorical (text/string types)
        if any(t in type_lower for t in ['text', 'varchar', 'char', 'string']):
            # Check if it looks like an ID
            if name_lower.endswith('id') or name_lower.endswith('code'):
                return 'identifier'
            return 'categorical'
        
        return 'categorical'  # Default
    
    # =========================================================================
    # Table Groups (FK-based clustering)
    # =========================================================================
    
    def build_table_groups(self, db_info: DatabaseInfo) -> List[TableGroup]:
        """
        Build table groups by clustering tables connected via foreign keys.
        
        Uses Union-Find algorithm to group FK-connected tables together.
        Standalone tables (no FK connections) are returned as single-table groups.
        
        Args:
            db_info: Complete database info with tables and foreign keys
            
        Returns:
            List of TableGroup, each containing related tables and their FKs
        """
        if not db_info.tables:
            return []
        
        # Build table name -> index mapping
        table_names = [t.name for t in db_info.tables]
        name_to_idx = {name.lower(): i for i, name in enumerate(table_names)}
        
        # Union-Find data structure
        parent = list(range(len(table_names)))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union tables connected by foreign keys
        for fk in db_info.foreign_keys:
            from_idx = name_to_idx.get(fk.from_table.lower())
            to_idx = name_to_idx.get(fk.to_table.lower())
            if from_idx is not None and to_idx is not None:
                union(from_idx, to_idx)
        
        # Group tables by their root parent
        groups_dict: Dict[int, List[int]] = {}
        for i in range(len(table_names)):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(i)
        
        # Build TableGroup objects
        result: List[TableGroup] = []
        for indices in groups_dict.values():
            group_tables = [db_info.tables[i] for i in indices]
            group_table_names_lower = {t.name.lower() for t in group_tables}
            
            # Filter FKs to only those within this group
            group_fks = [
                fk for fk in db_info.foreign_keys
                if fk.from_table.lower() in group_table_names_lower
                and fk.to_table.lower() in group_table_names_lower
            ]
            
            result.append(TableGroup(
                tables=group_tables,
                foreign_keys=group_fks
            ))
        
        # Sort groups: larger groups first, then alphabetically by first table name
        result.sort(key=lambda g: (-len(g.tables), g.tables[0].name.lower() if g.tables else ""))
        
        return result
    
    def format_schema_with_groups(
        self,
        db_info: DatabaseInfo,
        max_cols_per_table: int = 12
    ) -> str:
        """
        Format database schema organized by table groups.
        
        Tables connected by FKs are grouped together, making it easier
        for LLM to understand relationships and potential JOINs.
        
        Args:
            db_info: Complete database info
            max_cols_per_table: Maximum columns to show per table
            
        Returns:
            Formatted schema string with table groups
        """
        if not db_info or not db_info.tables:
            return "No schema available"
        
        groups = self.build_table_groups(db_info)
        lines = [f"Database: {db_info.db_id} ({len(db_info.tables)} tables)"]
        lines.append("")
        
        group_num = 1
        standalone_tables = []
        
        for group in groups:
            if len(group.tables) == 1 and not group.foreign_keys:
                # Standalone table - collect for later
                standalone_tables.append(group.tables[0])
            else:
                # Multi-table group
                lines.append(f"## Table Group {group_num}")
                group_num += 1
                
                for table in group.tables:
                    lines.append(self._format_table_entry(table, max_cols_per_table))
                
                # Show FK relationships within group
                if group.foreign_keys:
                    lines.append("")
                    lines.append("Relationships:")
                    for fk in group.foreign_keys:
                        lines.append(f"  {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
                
                lines.append("")
        
        # Add standalone tables section
        if standalone_tables:
            lines.append("## Standalone Tables")
            for table in standalone_tables:
                lines.append(self._format_table_entry(table, max_cols_per_table))
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_table_entry(self, table: TableInfo, max_cols: int = 12) -> str:
        """Format a single table entry for schema display."""
        cols = table.columns[:max_cols]
        col_parts = []
        
        for col in cols:
            col_str = col.name
            # Add type info
            col_str += f" ({col.data_type})"
            # Add PK/FK markers
            if col.is_primary_key:
                col_str += " [PK]"
            if col.is_foreign_key:
                col_str += " [FK]"
            col_parts.append(col_str)
        
        if len(table.columns) > max_cols:
            col_parts.append(f"... +{len(table.columns) - max_cols} more")
        
        return f"- {table.name}: {', '.join(col_parts)}"
    
    def format_schema_with_groups_and_samples(
        self,
        db_info: DatabaseInfo,
        sample_rows: int = 3
    ) -> str:
        """
        Format database schema organized by table groups with sample data.
        
        Tables connected by FKs are grouped together, and each table includes
        a sample of the first few rows to help LLM understand the data.
        All columns are displayed.
        
        Args:
            db_info: Complete database info
            sample_rows: Number of sample rows to show per table
            
        Returns:
            Formatted schema string with table groups and sample data
        """
        if not db_info or not db_info.tables:
            return "No schema available"
        
        # Get sqlite client for fetching samples
        sqlite_client = get_sqlite_client()
        
        groups = self.build_table_groups(db_info)
        lines = [f"Database: {db_info.db_id} ({len(db_info.tables)} tables)"]
        lines.append("")
        
        group_num = 1
        standalone_tables = []
        
        for group in groups:
            if len(group.tables) == 1 and not group.foreign_keys:
                # Standalone table - collect for later
                standalone_tables.append(group.tables[0])
            else:
                # Multi-table group
                lines.append(f"## Table Group {group_num}")
                group_num += 1
                
                for table in group.tables:
                    lines.append(self._format_table_entry_with_samples(
                        table, db_info.db_id, sqlite_client, sample_rows
                    ))
                
                # Show FK relationships within group
                if group.foreign_keys:
                    lines.append("")
                    lines.append("Relationships:")
                    for fk in group.foreign_keys:
                        lines.append(f"  {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
                
                lines.append("")
        
        # Add standalone tables section
        if standalone_tables:
            lines.append("## Standalone Tables")
            for table in standalone_tables:
                lines.append(self._format_table_entry_with_samples(
                    table, db_info.db_id, sqlite_client, sample_rows
                ))
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_table_entry_with_samples(
        self,
        table: TableInfo,
        db_id: str,
        sqlite_client,
        sample_rows: int = 3
    ) -> str:
        """Format a single table as markdown table with schema header and sample rows."""
        cols = table.columns  # Show all columns
        
        # Get total row count
        total_rows = 0
        try:
            count_sql = f'SELECT COUNT(*) as cnt FROM "{table.name}"'
            count_df = sqlite_client.execute_safe(db_id, count_sql)
            if count_df is not None and not count_df.empty:
                total_rows = int(count_df.iloc[0]['cnt'])
        except Exception:
            pass
        
        # Build header row: column_name (type) [PK/FK]
        header_parts = []
        for col in cols:
            h = col.name
            h += f" ({col.data_type})"
            if col.is_primary_key:
                h += " [PK]"
            if col.is_foreign_key:
                h += " [FK]"
            header_parts.append(h)
        
        # Build markdown table with row count in title
        lines = [f"### {table.name} ({total_rows} rows)"]
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")
        
        # Add distinct count row
        distinct_counts = []
        for col in cols:
            try:
                distinct_sql = (
                    f'SELECT COUNT(DISTINCT "{col.name}") as cnt '
                    f'FROM "{table.name}" WHERE "{col.name}" IS NOT NULL'
                )
                distinct_df = sqlite_client.execute_safe(db_id, distinct_sql)
                cnt = int(distinct_df.iloc[0]["cnt"]) if distinct_df is not None and not distinct_df.empty else 0
                distinct_counts.append(f"*{cnt} distinct*")
            except Exception:
                distinct_counts.append("(error)")
        if distinct_counts:
            lines.append("| " + " | ".join(distinct_counts) + " |")
        
        # Fetch sample data
        try:
            col_names = [c.name for c in cols]
            col_names_str = ", ".join([f'"{c}"' for c in col_names])
            sample_sql = f'SELECT {col_names_str} FROM "{table.name}" LIMIT {sample_rows}'
            
            df = sqlite_client.execute_safe(db_id, sample_sql)
            
            if df is not None and not df.empty:
                for _, row in df.head(sample_rows).iterrows():
                    row_vals = []
                    for col_name in col_names:
                        val = row.get(col_name)
                        val_str = str(val) if val is not None else "NULL"
                        # Truncate long values and escape pipes
                        if len(val_str) > 25:
                            val_str = val_str[:22] + "..."
                        val_str = val_str.replace("|", "\\|")
                        row_vals.append(val_str)
                    lines.append("| " + " | ".join(row_vals) + " |")
                
                # Show remaining rows indicator if there are more
                remaining = total_rows - min(len(df), sample_rows)
                if remaining > 0:
                    lines.append(f"... +{remaining} more rows")
            else:
                empty_row = ["(empty)"] + [""] * (len(header_parts) - 1)
                lines.append("| " + " | ".join(empty_row) + " |")
        except Exception as e:
            err_row = [f"(error: {str(e)[:30]})"] + [""] * (len(header_parts) - 1)
            lines.append("| " + " | ".join(err_row) + " |")
        
        return "\n".join(lines)


# Singleton instance
_processor: Optional[SchemaProcessor] = None


def get_schema_processor() -> SchemaProcessor:
    """Get or create singleton schema processor."""
    global _processor
    if _processor is None:
        _processor = SchemaProcessor()
    return _processor


def init_schema_processor(
    dev_tables_path: Optional[Path] = None,
    databases_dir: Optional[Path] = None
) -> SchemaProcessor:
    """Initialize schema processor with custom paths."""
    global _processor
    _processor = SchemaProcessor(
        dev_tables_path=dev_tables_path,
        databases_dir=databases_dir
    )
    return _processor
