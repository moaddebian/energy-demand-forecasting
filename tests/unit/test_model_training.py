"""
Unit tests for model training module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    @patch('src.training.trainer.mlflow.set_tracking_uri')
    @patch('src.training.trainer.mlflow.set_experiment')
    def test_trainer_init(self, mock_set_experiment, mock_set_tracking, mock_db_client):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(mock_db_client)
        assert trainer.db == mock_db_client
        assert hasattr(trainer, 'mlflow_config')
        assert hasattr(trainer, 'model_config')
    
    def test_load_features(self, mock_db_client, sample_features_df):
        """Test loading features from database."""
        # Mock database query result
        mock_data = []
        for _, row in sample_features_df.iterrows():
            mock_data.append(row.to_dict())
        
        mock_db_client.execute_query.return_value = mock_data
        
        trainer = ModelTrainer(mock_db_client)
        X, y = trainer.load_features()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)
    
    def test_load_features_empty(self, mock_db_client):
        """Test loading features with empty database."""
        mock_db_client.execute_query.return_value = []
        
        trainer = ModelTrainer(mock_db_client)
        X, y = trainer.load_features()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 0
        assert len(y) == 0
    
    def test_train_test_split_time_series(self, mock_db_client, sample_features_df):
        """Test time series train/test split."""
        trainer = ModelTrainer(mock_db_client)
        
        # Prepare data
        feature_cols = [col for col in sample_features_df.columns 
                       if col not in ['timestamp', 'demand', 'created_at']]
        X = sample_features_df[feature_cols]
        y = sample_features_df['demand']
        
        X_train, X_test, y_train, y_test = trainer.train_test_split_time_series(X, y, test_size=0.2)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        
        # Check temporal order is maintained
        assert X_train.index[-1] < X_test.index[0]
    
    @patch('src.training.trainer.xgb.XGBRegressor')
    def test_train_xgboost(self, mock_xgb, mock_db_client, sample_features_df):
        """Test XGBoost model training."""
        trainer = ModelTrainer(mock_db_client)
        
        # Prepare data
        feature_cols = [col for col in sample_features_df.columns 
                       if col not in ['timestamp', 'demand', 'created_at']]
        X_train = sample_features_df[feature_cols].iloc[:20]
        y_train = sample_features_df['demand'].iloc[:20]
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_xgb.return_value = mock_model
        
        model = trainer.train_xgboost(X_train, y_train)
        
        assert model is not None
        assert mock_model.fit.called
    
    @patch('src.training.trainer.lgb.LGBMRegressor')
    def test_train_lightgbm(self, mock_lgb, mock_db_client, sample_features_df):
        """Test LightGBM model training."""
        trainer = ModelTrainer(mock_db_client)
        
        # Prepare data
        feature_cols = [col for col in sample_features_df.columns 
                       if col not in ['timestamp', 'demand', 'created_at']]
        X_train = sample_features_df[feature_cols].iloc[:20]
        y_train = sample_features_df['demand'].iloc[:20]
        
        # Mock LightGBM model
        mock_model = Mock()
        mock_lgb.return_value = mock_model
        
        model = trainer.train_lightgbm(X_train, y_train)
        
        assert model is not None
        assert mock_model.fit.called
    
    def test_evaluate_model(self, mock_db_client, sample_features_df):
        """Test model evaluation."""
        trainer = ModelTrainer(mock_db_client)
        
        # Prepare data
        feature_cols = [col for col in sample_features_df.columns 
                       if col not in ['timestamp', 'demand', 'created_at']]
        X_test = sample_features_df[feature_cols].iloc[:10]
        y_test = sample_features_df['demand'].iloc[:10]
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = y_test.values + np.random.normal(0, 10, len(y_test))
        
        metrics = trainer.evaluate_model(mock_model, X_test, y_test)
        
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert isinstance(metrics['r2'], float)
        assert isinstance(metrics['rmse'], float)
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0

