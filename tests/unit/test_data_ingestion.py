"""
Unit tests for data ingestion modules.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.weather_client import WeatherAPIClient
from src.data_ingestion.energy_client import EnergyDataClient, EnergyDataIngester


class TestWeatherAPIClient:
    """Tests for WeatherClient."""
    
    def test_weather_client_init(self):
        """Test WeatherAPIClient initialization."""
        client = WeatherAPIClient()
        assert client is not None
        assert hasattr(client, 'api_key')
        assert hasattr(client, 'api_url')
    
    @patch('src.data_ingestion.weather_client.requests.get')
    def test_get_weather_data_success(self, mock_get):
        """Test successful weather data retrieval."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'main': {
                'temp': 20.5,
                'humidity': 65.0,
                'pressure': 1013.25
            },
            'wind': {
                'speed': 10.5
            }
        }
        mock_get.return_value = mock_response
        
        client = WeatherAPIClient()
        client.api_key = 'test_key'
        
        result = client.get_weather_data(
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow()
        )
        
        assert result is not None
        assert len(result) > 0
        assert 'temperature' in result[0]
        assert 'humidity' in result[0]
    
    @patch('src.data_ingestion.weather_client.requests.get')
    def test_get_weather_data_api_error(self, mock_get):
        """Test weather data retrieval with API error."""
        mock_get.side_effect = Exception("API Error")
        
        client = WeatherAPIClient()
        client.api_key = 'test_key'
        
        result = client.get_weather_data(
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow()
        )
        
        # Should return empty list on error
        assert result == []


class TestEnergyDataClient:
    """Tests for EnergyDataClient."""
    
    def test_energy_client_init(self):
        """Test EnergyDataClient initialization."""
        client = EnergyDataClient()
        assert client is not None
    
    @patch('src.data_ingestion.energy_client.requests.get')
    def test_get_energy_data_success(self, mock_get):
        """Test successful energy data retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'demand': 1000.0,
                    'region': 'default'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = EnergyDataClient()
        client.api_key = 'test_key'
        
        result = client.get_energy_data(
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
            'default'
        )
        
        assert result is not None
        assert len(result) > 0
        assert 'demand' in result[0]
    
    def test_generate_sample_data(self, mock_db_client):
        """Test sample data generation."""
        ingester = EnergyDataIngester(mock_db_client)
        
        result = ingester.generate_sample_data(days=1)
        
        assert result > 0
        assert mock_db_client.execute_insert.called


class TestEnergyDataIngester:
    """Tests for EnergyDataIngester."""
    
    def test_ingester_init(self, mock_db_client):
        """Test EnergyDataIngester initialization."""
        ingester = EnergyDataIngester(mock_db_client)
        assert ingester.db == mock_db_client
        assert ingester.energy_client is not None
    
    @patch('src.data_ingestion.energy_client.EnergyDataClient.get_energy_data')
    def test_ingest_energy_data(self, mock_get_data, mock_db_client):
        """Test energy data ingestion."""
        # Mock energy data
        mock_get_data.return_value = [
            {
                'timestamp': datetime.utcnow(),
                'demand': 1000.0,
                'region': 'default'
            }
        ]
        
        ingester = EnergyDataIngester(mock_db_client)
        ingester.energy_client.get_energy_data = mock_get_data
        
        result = ingester.ingest_energy_data(
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
            'default'
        )
        
        assert result >= 0
        assert mock_db_client.execute_insert.called
    
    def test_generate_sample_data_count(self, mock_db_client):
        """Test sample data generation returns correct count."""
        ingester = EnergyDataIngester(mock_db_client)
        
        # Mock execute_insert to return success
        mock_db_client.execute_insert.return_value = 1
        
        result = ingester.generate_sample_data(days=1)
        
        # Should generate 24 records (1 day * 24 hours)
        assert result == 24
        assert mock_db_client.execute_insert.call_count == 24

