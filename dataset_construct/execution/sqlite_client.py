"""
SQLite client for executing queries against BIRD Mini-Dev databases.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from core.config import get_config

logger = logging.getLogger(__name__)


class SQLiteClient:
    """
    Client for connecting to and querying BIRD Mini-Dev SQLite databases.
    """
    
    def __init__(self, databases_dir: Optional[Path] = None):
        """
        Initialize SQLite client.
        
        Args:
            databases_dir: Path to local databases directory (flat structure). 
                          If None, uses config default (dataset_construct/databases).
        """
        if databases_dir is None:
            databases_dir = get_config().paths.local_databases_dir
        self.databases_dir = Path(databases_dir)
        
        if not self.databases_dir.exists():
            raise ValueError(f"Databases directory does not exist: {self.databases_dir}")
        
        # Cache of db_id -> db_path mappings
        self._db_paths: Dict[str, Path] = {}
    
    def _get_db_path(self, db_id: str) -> Path:
        """
        Get the path to a SQLite database file.
        Uses flat structure: databases/{db_id}.sqlite
        """
        if db_id in self._db_paths:
            return self._db_paths[db_id]
        
        # Flat structure: databases/{db_id}.sqlite
        db_path = self.databases_dir / f"{db_id}.sqlite"
        
        self._db_paths[db_id] = db_path
        return db_path
    
    def get_db_path(self, db_id: str) -> str:
        """
        Get the full absolute path to a SQLite database file as a string.
        
        Args:
            db_id: Database identifier (e.g., 'california_schools')
            
        Returns:
            Full absolute path to the database file as string
        """
        return str(self._get_db_path(db_id))
    
    def get_db_path_for_llm(self, db_id: str) -> str:
        """
        Get the relative database path for LLM prompts.
        
        Args:
            db_id: Database identifier (e.g., 'california_schools')
            
        Returns:
            Relative path string like './databases/{db_id}.sqlite'
        """
        return f"./databases/{db_id}.sqlite"
    
    def execute(
        self, 
        db_id: str, 
        sql: str, 
        limit: Optional[int] = None,
        timeout: int = 30
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            db_id: Database identifier (e.g., 'california_schools')
            sql: SQL query to execute
            limit: Optional row limit (applied after query)
            timeout: Query timeout in seconds
            
        Returns:
            DataFrame with query results
            
        Raises:
            FileNotFoundError: If database file doesn't exist
            sqlite3.Error: If query execution fails
        """
        db_path = self._get_db_path(db_id)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=timeout)
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Execute query
            df = pd.read_sql_query(sql, conn)
            
            # Apply limit if specified
            if limit is not None and len(df) > limit:
                df = df.head(limit)
            
            logger.debug(f"Executed SQL on {db_id}: {len(df)} rows returned")
            return df
            
        finally:
            if conn:
                conn.close()
    
    def execute_safe(
        self, 
        db_id: str, 
        sql: str, 
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Execute SQL with error handling, returning None on failure.
        
        Args:
            db_id: Database identifier
            sql: SQL query to execute
            limit: Optional row limit
            
        Returns:
            DataFrame on success, None on failure
        """
        try:
            return self.execute(db_id, sql, limit)
        except FileNotFoundError as e:
            logger.error(f"Database not found for {db_id}: {e}")
            return None
        except sqlite3.Error as e:
            logger.warning(f"SQL execution failed for {db_id}: {e}\nSQL: {sql[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing SQL for {db_id}: {e}")
            return None
    
    def execute_with_error(
        self, 
        db_id: str, 
        sql: str, 
        limit: Optional[int] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL with error handling, returning both result and error message.
        
        Args:
            db_id: Database identifier
            sql: SQL query to execute
            limit: Optional row limit
            
        Returns:
            Tuple of (DataFrame or None, error_message or None)
        """
        try:
            df = self.execute(db_id, sql, limit)
            return df, None
        except FileNotFoundError as e:
            error_msg = f"Database not found: {e}"
            logger.error(f"Database not found for {db_id}: {e}")
            return None, error_msg
        except sqlite3.Error as e:
            error_msg = f"SQL error: {e}"
            logger.warning(f"SQL execution failed for {db_id}: {e}\nSQL: {sql[:200]}...")
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error executing SQL for {db_id}: {e}")
            return None, error_msg
    
    def get_table_names(self, db_id: str) -> List[str]:
        """Get list of table names in a database."""
        sql = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
        try:
            df = self.execute(db_id, sql)
            return df['name'].tolist()
        except Exception as e:
            logger.error(f"Failed to get table names for {db_id}: {e}")
            return []
    
    def get_table_schema(self, db_id: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table.
        
        Returns list of dicts with: name, type, notnull, pk
        """
        sql = f"PRAGMA table_info('{table_name}');"
        try:
            df = self.execute(db_id, sql)
            columns = []
            for _, row in df.iterrows():
                columns.append({
                    "name": row["name"],
                    "type": row["type"],
                    "notnull": bool(row["notnull"]),
                    "pk": bool(row["pk"])
                })
            return columns
        except Exception as e:
            logger.error(f"Failed to get schema for {db_id}.{table_name}: {e}")
            return []
    
    def get_sample_values(
        self, 
        db_id: str, 
        table_name: str, 
        column_name: str, 
        limit: int = 5
    ) -> List[Any]:
        """Get sample distinct values from a column."""
        sql = f"""
        SELECT DISTINCT "{column_name}" 
        FROM "{table_name}" 
        WHERE "{column_name}" IS NOT NULL 
        LIMIT {limit};
        """
        try:
            df = self.execute(db_id, sql)
            return df[column_name].tolist()
        except Exception as e:
            logger.debug(f"Failed to get sample values for {table_name}.{column_name}: {e}")
            return []
    
    def get_row_count(self, db_id: str, table_name: str) -> int:
        """Get row count for a table."""
        sql = f'SELECT COUNT(*) as cnt FROM "{table_name}";'
        try:
            df = self.execute(db_id, sql)
            return int(df['cnt'].iloc[0])
        except Exception as e:
            logger.debug(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def validate_sql(self, db_id: str, sql: str) -> Tuple[bool, str]:
        """
        Validate if SQL is syntactically correct without executing.
        
        Returns:
            (is_valid, error_message)
        """
        db_path = self._get_db_path(db_id)
        
        if not db_path.exists():
            return False, f"Database not found: {db_path}"
        
        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            
            # Use EXPLAIN to validate without executing
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            
            return True, ""
            
        except sqlite3.Error as e:
            return False, str(e)
        finally:
            if conn:
                conn.close()
    
    def list_available_databases(self) -> List[str]:
        """
        List all available database IDs.
        Scans flat structure for {db_id}.sqlite files.
        """
        db_ids = []
        for item in self.databases_dir.iterdir():
            if item.is_file() and item.suffix == '.sqlite':
                # Extract db_id from filename (remove .sqlite extension)
                db_ids.append(item.stem)
        return sorted(db_ids)


# Global client instance
_client: Optional[SQLiteClient] = None


def get_sqlite_client() -> SQLiteClient:
    """Get global SQLite client instance."""
    global _client
    if _client is None:
        _client = SQLiteClient()
    return _client


def init_sqlite_client(databases_dir: Optional[Path] = None) -> SQLiteClient:
    """Initialize global SQLite client with custom path."""
    global _client
    _client = SQLiteClient(databases_dir=databases_dir)
    return _client