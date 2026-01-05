"""
Database client for TimescaleDB with connection pooling.
"""
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
from pathlib import Path

from .config import load_config, get_config_value


class DatabaseClient:
    """TimescaleDB client with connection pooling."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database client.
        
        Args:
            config_path: Path to config.yaml file. If None, uses environment variables.
        """
        # Load config
        config = load_config(config_path)
        db_config = config.get('database', {})
        
        # Parse config values, handling environment variable placeholders
        # Priority: environment variable > config file > default
        host = os.getenv('DB_HOST') or get_config_value(config, 'database.host', 'localhost')
        port_str = os.getenv('DB_PORT') or str(get_config_value(config, 'database.port', 5432))
        database = os.getenv('DB_NAME') or get_config_value(config, 'database.name', 'energy_forecasting')
        user = os.getenv('DB_USER') or get_config_value(config, 'database.user', 'ml_app_user')
        password = os.getenv('DB_PASSWORD') or get_config_value(config, 'database.password', 'devpassword123')
        
        # Convert port to int
        try:
            port = int(port_str)
        except (ValueError, TypeError):
            port = 5432
        
        self.config = {
            'host': str(host),
            'port': port,
            'database': str(database),
            'user': str(user),
            'password': str(password),
        }
        
        pool_size = db_config.get('pool_size', 10)
        max_overflow = db_config.get('max_overflow', 20)
        
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size + max_overflow,
                **self.config
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create connection pool: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = self.connection_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor from the connection pool."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            List of dictionaries with query results
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: Optional[tuple] = None) -> Optional[int]:
        """
        Execute an INSERT query and return the number of rows affected.
        
        Args:
            query: SQL INSERT query
            params: Query parameters (tuple or dict)
        
        Returns:
            Number of rows affected (or None if error)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                return cursor.rowcount if cursor.rowcount > 0 else None
            except Exception as e:
                # Log error but don't raise - let caller handle it
                return None
            finally:
                cursor.close()
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an UPDATE query.
        
        Args:
            query: SQL UPDATE query
            params: Query parameters
        
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                return cursor.rowcount
            finally:
                cursor.close()
    
    def execute_batch(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a batch insert/update.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
        
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, params_list)
                return cursor.rowcount
            finally:
                cursor.close()
    
    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()


# Global database client instance
_db_client: Optional[DatabaseClient] = None


def get_db_client(config_path: Optional[str] = None) -> DatabaseClient:
    """Get or create a global database client instance."""
    global _db_client
    if _db_client is None:
        if config_path is None:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'config.yaml'
            if not config_path.exists():
                config_path = None
        _db_client = DatabaseClient(config_path)
    return _db_client

