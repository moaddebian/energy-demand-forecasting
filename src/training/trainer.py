"""
Model training pipeline with MLflow integration.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training pipeline with MLflow tracking."""
    
    def __init__(self, db_client, config_path: Optional[str] = None):
        """
        Initialize model trainer.
        
        Args:
            db_client: Database client instance
            config_path: Path to config.yaml file
        """
        self.db = db_client
        
        from src.common.config import load_config, get_config_value
        
        # Load config with environment variable parsing
        config = load_config(config_path)
        
        self.mlflow_config = config.get('mlflow', {})
        self.model_config = config.get('model', {})
        self.training_config = config.get('model', {}).get('training', {})
        self.hyperparameter_config = config.get('hyperparameter_tuning', {})
        
        # Initialize MLflow with parsed config values
        # Default to file-based tracking (no server required)
        default_tracking_uri = './mlruns'
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if not tracking_uri:
            tracking_uri = get_config_value(config, 'mlflow.tracking_uri', default_tracking_uri)
            # If it's still the placeholder string, use file-based
            if isinstance(tracking_uri, str) and tracking_uri.startswith('${'):
                tracking_uri = default_tracking_uri
        
        experiment_name = get_config_value(
            config, 'mlflow.experiment_name', 'energy_demand_forecasting'
        )
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def load_features(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load feature data from database.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        query = """
            SELECT * FROM feature_data
        """
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= %s")
                params.append(end_date)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        data = self.db.execute_query(query, tuple(params) if params else None)
        
        if not data:
            logger.error("No feature data found")
            return pd.DataFrame(), pd.Series()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Separate features and target
        target_col = 'demand'
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'demand', 'created_at']]
        
        # Convert numeric columns to proper types
        numeric_cols = [col for col in feature_cols if col not in ['is_weekend', 'is_holiday']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(bool)
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(bool)
        
        X = df[feature_cols].fillna(0)
        y = pd.to_numeric(df[target_col], errors='coerce')
        
        return X, y
    
    def train_test_split_time_series(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split time series data maintaining temporal order.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
        
        Returns:
            Train/test split
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                     hyperparameters: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameters: Model hyperparameters
        
        Returns:
            Trained XGBoost model
        """
        params = self.model_config.get('xgboost', {})
        if hyperparameters:
            params.update(hyperparameters)
        
        model = xgb.XGBRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                      hyperparameters: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameters: Model hyperparameters
        
        Returns:
            Trained LightGBM model
        """
        params = self.model_config.get('lightgbm', {})
        if hyperparameters:
            params.update(hyperparameters)
        
        model = lgb.LGBMRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        return metrics
    
    def train_with_mlflow(self, model_name: str = 'xgboost', 
                          hyperparameter_tuning: bool = False) -> Dict:
        """
        Train model with MLflow tracking.
        
        Args:
            model_name: Name of model to train ('xgboost', 'lightgbm', 'prophet')
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training for {model_name}")
        
        # Load data
        X, y = self.load_features()
        if X.empty:
            raise ValueError("No feature data available for training")
        
        # Split data
        test_size = self.training_config.get('test_size', 0.2)
        val_size = self.training_config.get('validation_size', 0.1)
        
        X_train, X_test, y_train, y_test = self.train_test_split_time_series(X, y, test_size)
        
        # Further split for validation
        val_split_idx = int(len(X_train) * (1 - val_size))
        X_train_split, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
        y_train_split, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Hyperparameter tuning
            if hyperparameter_tuning and model_name in ['xgboost', 'lightgbm']:
                logger.info("Starting hyperparameter tuning")
                best_params = self.tune_hyperparameters(model_name, X_train_split, y_train_split, X_val, y_val)
            else:
                best_params = None
            
            # Train model
            if model_name == 'xgboost':
                model = self.train_xgboost(X_train_split, y_train_split, X_val, y_val, best_params)
                mlflow.xgboost.log_model(model, "model")
            elif model_name == 'lightgbm':
                model = self.train_lightgbm(X_train_split, y_train_split, X_val, y_val, best_params)
                mlflow.lightgbm.log_model(model, "model")
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            if best_params:
                mlflow.log_params(best_params)
            
            # Log model info
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("training_date", datetime.now().isoformat())
            
            # Register model if metrics meet threshold
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_name_registry = self.mlflow_config.get('model_name', 'energy_demand_model')
            
            # Check if model meets performance threshold
            min_r2 = 0.7  # Can be configured
            if metrics['r2'] >= min_r2:
                mlflow.register_model(model_uri, model_name_registry)
                logger.info(f"Model registered with R2 score: {metrics['r2']:.4f}")
            
            # Save metrics to database
            self.save_training_metrics(model_name, metrics, mlflow.active_run().info.run_id)
            
            return {
                'model': model,
                'metrics': metrics,
                'run_id': mlflow.active_run().info.run_id,
                'model_uri': model_uri
            }
    
    def tune_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Hyperparameter tuning with Optuna.
        
        Args:
            model_name: Model name
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = lgb.LGBMRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unsupported model for tuning: {model_name}")
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return rmse
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        n_trials = self.hyperparameter_config.get('n_trials', 50)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def save_training_metrics(self, model_name: str, metrics: Dict, run_id: str):
        """
        Save training metrics to database.
        
        Args:
            model_name: Model name
            metrics: Metrics dictionary
            run_id: MLflow run ID
        """
        try:
            for metric_name, metric_value in metrics.items():
                query = """
                    INSERT INTO model_metrics 
                    (model_name, model_version, metric_name, metric_value, evaluation_timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, model_version, metric_name, evaluation_timestamp) 
                    DO UPDATE SET metric_value = EXCLUDED.metric_value
                """
                self.db.execute_insert(
                    query,
                    (model_name, 'latest', metric_name, float(metric_value), datetime.now())
                )
            
            logger.info("Training metrics saved to database")
        except Exception as e:
            logger.error(f"Failed to save training metrics: {str(e)}")

