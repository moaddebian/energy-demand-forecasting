"""
Pytest configuration and fixtures for testing.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'
os.environ['DB_NAME'] = 'energy_forecasting_test'
os.environ['DB_USER'] = 'ml_app_user'
os.environ['DB_PASSWORD'] = 'devpassword123'
os.environ['MLFLOW_TRACKING_URI'] = './mlruns'


@pytest.fixture
def sample_energy_data():
    """Sample energy data for testing."""
    return [
        {
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'demand': 1000 + (i * 10),
            'region': 'default'
        }
        for i in range(24)
    ]


@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing."""
    return [
        {
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'temperature': 20.0 + (i * 0.1),
            'humidity': 65.0 + (i * 0.2),
            'pressure': 1013.25 + (i * 0.05),
            'wind_speed': 10.5 + (i * 0.1),
            'location': 'London,UK'
        }
        for i in range(24)
    ]


@pytest.fixture
def sample_features_df():
    """Sample features DataFrame for testing."""
    dates = pd.date_range(start=datetime.utcnow() - timedelta(hours=23), 
                          end=datetime.utcnow(), freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'demand': [1000 + i * 10 for i in range(24)],
        'lag_1': [950 + i * 10 for i in range(24)],
        'lag_7': [980 + i * 10 for i in range(24)],
        'lag_24': [1020 + i * 10 for i in range(24)],
        'lag_168': [1050 + i * 10 for i in range(24)],
        'rolling_avg_7': [990 + i * 10 for i in range(24)],
        'rolling_avg_24': [1005 + i * 10 for i in range(24)],
        'rolling_avg_168': [1010 + i * 10 for i in range(24)],
        'rolling_std_7': [25.5] * 24,
        'rolling_std_24': [30.2] * 24,
        'hour': [i % 24 for i in range(24)],
        'day_of_week': [i % 7 for i in range(24)],
        'day_of_month': [4] * 24,
        'month': [1] * 24,
        'is_weekend': [False] * 24,
        'is_holiday': [False] * 24,
        'temperature': [20.5 + i * 0.1 for i in range(24)],
        'humidity': [65.0 + i * 0.2 for i in range(24)],
        'pressure': [1013.25 + i * 0.05 for i in range(24)],
        'wind_speed': [10.5 + i * 0.1 for i in range(24)],
        'temp_rolling_avg_24': [20.0] * 24,
        'humidity_rolling_avg_24': [65.0] * 24
    })
    return df


@pytest.fixture
def mock_db_client():
    """Mock database client for testing."""
    mock_db = Mock()
    mock_db.execute_query = Mock(return_value=[])
    mock_db.execute_insert = Mock(return_value=1)
    mock_db.get_connection = Mock()
    return mock_db


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request for API testing."""
    return {
        "features": [
            950.0, 980.0, 1020.0, 1050.0, 990.0, 1005.0, 1010.0,
            25.5, 30.2, 14, 1, 4, 1, 0, 0,
            20.5, 65.0, 1013.25, 10.5, 20.0, 65.0
        ],
        "model_name": "xgboost"
    }

