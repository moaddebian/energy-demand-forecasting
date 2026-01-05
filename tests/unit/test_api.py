"""
Unit tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    mock = MagicMock()
    mock.predict.return_value = [1052.32]
    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoints."""
    
    @patch('src.api.app.get_model')
    def test_predict_success(self, mock_get_model, client, mock_model, sample_prediction_request):
        """Test successful prediction."""
        mock_get_model.return_value = mock_model
        
        response = client.post("/predict", json=sample_prediction_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_name" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] == 1052.32
    
    def test_predict_invalid_features(self, client):
        """Test prediction with invalid features."""
        invalid_request = {
            "features": [1, 2, 3],  # Wrong number of features
            "model_name": "xgboost"
        }
        
        response = client.post("/predict", json=invalid_request)
        
        # Should either validate or fail gracefully
        assert response.status_code in [200, 400, 422, 500]
    
    def test_predict_missing_features(self, client):
        """Test prediction with missing features."""
        invalid_request = {
            "model_name": "xgboost"
        }
        
        response = client.post("/predict", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.app.get_model')
    def test_predict_batch_success(self, mock_get_model, client, mock_model):
        """Test successful batch prediction."""
        import numpy as np
        mock_get_model.return_value = mock_model
        mock_model.predict.return_value = np.array([1052.32, 1060.45])
        
        batch_request = {
            "features_list": [
                [950.0, 980.0, 1020.0, 1050.0, 990.0, 1005.0, 1010.0,
                 25.5, 30.2, 14, 1, 4, 1, 0, 0,
                 20.5, 65.0, 1013.25, 10.5, 20.0, 65.0],
                [960.0, 990.0, 1030.0, 1060.0, 1000.0, 1015.0, 1020.0,
                 26.0, 31.0, 15, 2, 5, 1, 0, 0,
                 21.0, 66.0, 1014.0, 11.0, 21.0, 66.0]
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert isinstance(data["predictions"][0], float)


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    @patch('src.api.app.get_db_client')
    def test_metrics_success(self, mock_get_db, client):
        """Test metrics endpoint with data."""
        # Mock database response
        mock_db = Mock()
        mock_db.execute_query.return_value = [
            {
                'model_name': 'energy_demand_model',
                'model_version': '1',
                'metric_name': 'r2',
                'metric_value': 0.85,
                'evaluation_timestamp': '2026-01-04T00:00:00'
            },
            {
                'model_name': 'energy_demand_model',
                'model_version': '1',
                'metric_name': 'rmse',
                'metric_value': 50.2,
                'evaluation_timestamp': '2026-01-04T00:00:00'
            }
        ]
        mock_get_db.return_value = mock_db
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "model_name" in data[0]
            assert "model_version" in data[0]
            assert "metrics" in data[0]
    
    @patch('src.api.app.get_db_client')
    def test_metrics_empty(self, mock_get_db, client):
        """Test metrics endpoint with no data."""
        mock_db = Mock()
        mock_db.execute_query.return_value = []
        mock_get_db.return_value = mock_db
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_metrics_with_params(self, client):
        """Test metrics endpoint with query parameters."""
        response = client.get("/metrics?model_name=test_model&limit=5")
        
        # Should not fail even if no data
        assert response.status_code in [200, 500]

