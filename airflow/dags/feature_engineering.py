"""
Feature Engineering DAG - Computes features from raw data.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.database import get_db_client
from src.feature_engineering import FeatureEngineer


def compute_features():
    """Compute engineered features from raw data."""
    db = get_db_client()
    feature_engineer = FeatureEngineer(db)
    
    # Compute features for last 7 days to ensure we have enough data for rolling windows
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Engineer features
    features_df = feature_engineer.engineer_features(start_date, end_date)
    
    if features_df.empty:
        raise Exception("No features computed - check if raw data exists")
    
    # Save to database
    count = feature_engineer.save_features_to_db(features_df)
    
    return f"Features computed and saved: {count} records"


# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'feature_engineering',
    default_args=default_args,
    description='Compute rolling averages, lag features, and time-based features',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['feature_engineering', 'ml'],
)

# Define task
compute_features_task = PythonOperator(
    task_id='compute_features',
    python_callable=compute_features,
    dag=dag,
)

compute_features_task
