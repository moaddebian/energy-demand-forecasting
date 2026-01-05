"""
Weather API client for fetching weather data.
"""
import requests
import time
from typing import Dict, Optional, List
from datetime import datetime
import logging
from pathlib import Path
import yaml
import os

logger = logging.getLogger(__name__)


class WeatherAPIClient:
    """Client for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize weather API client.
        
        Args:
            config_path: Path to config.yaml file
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                weather_config = config.get('data_ingestion', {}).get('weather', {})
        else:
            weather_config = {}
        
        self.api_key = os.getenv('WEATHER_API_KEY', weather_config.get('api_key', ''))
        self.api_url = os.getenv('WEATHER_API_URL', weather_config.get('api_url', 'https://api.openweathermap.org/data/2.5'))
        self.location = os.getenv('WEATHER_LOCATION', weather_config.get('location', 'London,UK'))
        self.retry_attempts = weather_config.get('retry_attempts', 3)
        self.retry_delay = weather_config.get('retry_delay_seconds', 5)
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.warning("Weather API key not configured. Set WEATHER_API_KEY environment variable.")
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
        
        Returns:
            API response as dictionary or None if failed
        """
        url = f"{self.api_url}/{endpoint}"
        params['appid'] = self.api_key
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Weather API request failed (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All weather API retry attempts failed: {str(e)}")
                    return None
        
        return None
    
    def get_current_weather(self) -> Optional[Dict]:
        """
        Get current weather data.
        
        Returns:
            Dictionary with weather data or None if failed
        """
        params = {
            'q': self.location,
            'units': 'metric'
        }
        
        data = self._make_request('weather', params)
        
        if data:
            return {
                'timestamp': datetime.utcnow(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'wind_direction': data.get('wind', {}).get('deg'),
                'cloud_cover': data.get('clouds', {}).get('all', 0)
            }
        
        return None
    
    def get_forecast(self, hours: int = 24) -> List[Dict]:
        """
        Get weather forecast.
        
        Args:
            hours: Number of hours to forecast (default: 24)
        
        Returns:
            List of forecast data points
        """
        params = {
            'q': self.location,
            'units': 'metric',
            'cnt': min(hours // 3, 40)  # API returns 3-hour intervals, max 40
        }
        
        data = self._make_request('forecast', params)
        
        if data and 'list' in data:
            forecasts = []
            for item in data['list']:
                forecasts.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item.get('wind', {}).get('speed', 0),
                    'wind_direction': item.get('wind', {}).get('deg'),
                    'cloud_cover': item.get('clouds', {}).get('all', 0)
                })
            return forecasts
        
        return []


class WeatherDataIngester:
    """Ingest weather data and store in database."""
    
    def __init__(self, db_client, weather_client: Optional[WeatherAPIClient] = None):
        """
        Initialize weather data ingester.
        
        Args:
            db_client: Database client instance
            weather_client: Weather API client instance
        """
        self.db = db_client
        self.weather_client = weather_client or WeatherAPIClient()
    
    def ingest_current_weather(self) -> bool:
        """
        Ingest current weather data.
        
        Returns:
            True if successful, False otherwise
        """
        weather_data = self.weather_client.get_current_weather()
        
        if not weather_data:
            logger.error("Failed to fetch weather data")
            return False
        
        try:
            query = """
                INSERT INTO weather_data 
                (timestamp, temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover)
                VALUES (%(timestamp)s, %(temperature)s, %(humidity)s, %(pressure)s, 
                        %(wind_speed)s, %(wind_direction)s, %(cloud_cover)s)
                ON CONFLICT (timestamp) DO UPDATE SET
                    temperature = EXCLUDED.temperature,
                    humidity = EXCLUDED.humidity,
                    pressure = EXCLUDED.pressure,
                    wind_speed = EXCLUDED.wind_speed,
                    wind_direction = EXCLUDED.wind_direction,
                    cloud_cover = EXCLUDED.cloud_cover
            """
            
            self.db.execute_insert(query, tuple(weather_data.values()))
            logger.info(f"Successfully ingested weather data for {weather_data['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert weather data: {str(e)}")
            return False
    
    def ingest_forecast(self) -> int:
        """
        Ingest weather forecast data.
        
        Returns:
            Number of records inserted
        """
        forecasts = self.weather_client.get_forecast()
        
        if not forecasts:
            logger.warning("No forecast data received")
            return 0
        
        inserted = 0
        for forecast in forecasts:
            try:
                query = """
                    INSERT INTO weather_data 
                    (timestamp, temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover)
                    VALUES (%(timestamp)s, %(temperature)s, %(humidity)s, %(pressure)s, 
                            %(wind_speed)s, %(wind_direction)s, %(cloud_cover)s)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        temperature = EXCLUDED.temperature,
                        humidity = EXCLUDED.humidity,
                        pressure = EXCLUDED.pressure,
                        wind_speed = EXCLUDED.wind_speed,
                        wind_direction = EXCLUDED.wind_direction,
                        cloud_cover = EXCLUDED.cloud_cover
                """
                
                self.db.execute_insert(query, tuple(forecast.values()))
                inserted += 1
                
            except Exception as e:
                logger.error(f"Failed to insert forecast data: {str(e)}")
        
        logger.info(f"Successfully ingested {inserted} forecast records")
        return inserted

