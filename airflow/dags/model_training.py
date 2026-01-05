"""
Model Training DAG - Trains and registers ML models.
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
from src.training import ModelTrainer


def train_xgboost_model():
    """Train XGBoost model with MLflow tracking."""
    db = get_db_client()
    trainer = ModelTrainer(db)
    
    result = trainer.train_with_mlflow(
        model_name='xgboost',
        hyperparameter_tuning=False  # Set to True for tuning (takes longer)
    )
    
    return f"XGBoost model trained. R2: {result['metrics']['r2']:.4f}, RMSE: {result['metrics']['rmse']:.2f}"


def train_lightgbm_model():
    """Train LightGBM model with MLflow tracking."""
    db = get_db_client()
    trainer = ModelTrainer(db)
    
    result = trainer.train_with_mlflow(
        model_name='lightgbm',
        hyperparameter_tuning=False
    )
    
    return f"LightGBM model trained. R2: {result['metrics']['r2']:.4f}, RMSE: {result['metrics']['rmse']:.2f}"


# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# Define the DAG
dag = DAG(
    'model_training',
    default_args=default_args,
    description='Train energy demand forecasting models (XGBoost, LightGBM)',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['model_training', 'ml', 'mlflow'],
)

# Define tasks
train_xgboost_task = PythonOperator(
    task_id='train_xgboost',
    python_callable=train_xgboost_model,
    dag=dag,
)

train_lightgbm_task = PythonOperator(
    task_id='train_lightgbm',
    python_callable=train_lightgbm_model,
    dag=dag,
)

# Models can be trained in parallel
train_xgboost_task
train_lightgbm_task
