"""
Setup database schema.
"""
import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.database import get_db_client


def setup_database():
    """Initialize database schema."""
    print("Setting up database schema...")
    
    # Read SQL schema file
    schema_file = project_root / 'sql' / 'schemas' / '01_create_tables.sql'
    
    if not schema_file.exists():
        print(f"Error: Schema file not found at {schema_file}")
        return False
    
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    # Get database connection
    try:
        db = get_db_client()
        
        # Execute schema SQL
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            conn.commit()
            cursor.close()
        
        print("Database schema created successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False


if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)

