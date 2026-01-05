"""
Energy data ingestion client.
Supports both generic energy APIs and ENTSO-E Transparency Platform.
"""
import requests
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
import yaml
import os
from pathlib import Path

from src.data_ingestion.entsoe_client import ENTSOEClient

logger = logging.getLogger(__name__)


class EnergyDataClient:
    """Client for fetching energy consumption data.
    
    Supports:
    - ENTSO-E Transparency Platform API (preferred, real European data)
    - Generic energy API (fallback)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize energy data client.
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                energy_config = config.get('data_ingestion', {}).get('energy', {})
        else:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    energy_config = config.get('data_ingestion', {}).get('energy', {})
            else:
                energy_config = {}
        
        # Generic API configuration
        self.api_url = os.getenv('ENERGY_API_URL', energy_config.get('api_url', ''))
        self.api_key = os.getenv('ENERGY_API_KEY', energy_config.get('api_key', ''))
        self.retry_attempts = energy_config.get('retry_attempts', 3)
        self.retry_delay = energy_config.get('retry_delay_seconds', 5)
        
        # ENTSO-E client (preferred if API key is available)
        self.use_entsoe = os.getenv('ENTSOE_API_KEY', '') or energy_config.get('use_entsoe', False)
        if self.use_entsoe:
            self.entsoe_client = ENTSOEClient(config_path)
            logger.info("ENTSO-E client initialized")
        else:
            self.entsoe_client = None
            logger.info("Using generic energy API client")
    
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
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Energy API request failed (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All energy API retry attempts failed: {str(e)}")
                    return None
        
        return None
    
    def get_energy_data(self, start_date: datetime, end_date: datetime, region: str = 'default') -> List[Dict]:
        """
        Get energy consumption data for a date range.
        
        Priority:
        1. ENTSO-E API (if configured and available)
        2. Generic energy API (if configured)
        3. Returns empty list (fallback to sample data generation)
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            region: Region identifier (country code for ENTSO-E, e.g., 'FR', 'DE')
        
        Returns:
            List of energy data points with 'timestamp', 'demand', 'region'
        """
        # Try ENTSO-E first (preferred for European data)
        if self.entsoe_client:
            try:
                entsoe_data = self.entsoe_client.get_actual_load(start_date, end_date, domain=region)
                if entsoe_data:
                    # Convert ENTSO-E format to our format
                    return [
                        {
                            'timestamp': item['timestamp'],
                            'demand': item['demand'],
                            'region': region
                        }
                        for item in entsoe_data
                    ]
            except Exception as e:
                logger.warning(f"ENTSO-E API failed, trying generic API: {str(e)}")
        
        # Fallback to generic API
        if self.api_url and self.api_key:
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'region': region
            }
            
            data = self._make_request('consumption', params)
            
            if data and 'data' in data:
                return [
                    {
                        'timestamp': datetime.fromisoformat(item['timestamp']),
                        'demand': float(item['demand']),
                        'region': region
                    }
                    for item in data['data']
                ]
        
        # No data available
        logger.warning("No energy data available from APIs")
        return []


class EnergyDataIngester:
    """Ingest energy data and store in database."""
    
    def __init__(self, db_client, energy_client: Optional[EnergyDataClient] = None):
        """
        Initialize energy data ingester.
        
        Args:
            db_client: Database client instance
            energy_client: Energy API client instance
        """
        self.db = db_client
        self.energy_client = energy_client or EnergyDataClient()
    
    def ingest_energy_data(self, start_date: datetime, end_date: datetime, region: str = 'default') -> int:
        """
        Ingest energy consumption data.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            region: Region identifier
        
        Returns:
            Number of records inserted
        """
        energy_data = self.energy_client.get_energy_data(start_date, end_date, region)
        
        if not energy_data:
            logger.warning("No energy data received")
            return 0
        
        inserted = 0
        for data_point in energy_data:
            try:
                query = """
                    INSERT INTO energy_data (timestamp, demand, region)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (timestamp, region) DO UPDATE SET
                        demand = EXCLUDED.demand
                """
                
                result = self.db.execute_insert(query, (data_point['timestamp'], data_point['demand'], data_point['region']))
                if result is not None:
                    inserted += 1
                
            except Exception as e:
                logger.error(f"Failed to insert energy data: {str(e)}")
        
        logger.info(f"Successfully ingested {inserted} energy data records")
        return inserted
    
    def generate_sample_data(self, days: int = 30) -> int:
        """
        Generate sample energy data for testing (when API is not available).
        
        Args:
            days: Number of days of sample data to generate
        
        Returns:
            Number of records inserted
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info(f"Generating {days} days of sample energy data")
        
        now = datetime.utcnow()
        data_points = []
        
        # Generate hourly data with realistic patterns
        for i in range(days * 24):
            timestamp = now - timedelta(hours=i)
            hour = timestamp.hour
            
            # Base demand with daily and weekly patterns
            base_demand = 1000
            daily_pattern = 200 * np.sin(2 * np.pi * hour / 24 - np.pi/2)  # Peak during day
            weekly_pattern = 100 if timestamp.weekday() < 5 else -50  # Lower on weekends
            noise = np.random.normal(0, 50)
            
            demand = max(0, base_demand + daily_pattern + weekly_pattern + noise)
            
            data_points.append({
                'timestamp': timestamp,
                'demand': round(demand, 2),
                'region': 'default'
            })
        
        inserted = 0
        for data_point in data_points:
            try:
                query = """
                    INSERT INTO energy_data (timestamp, demand, region)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (timestamp, region) DO UPDATE SET
                        demand = EXCLUDED.demand
                """
                
                result = self.db.execute_insert(query, (data_point['timestamp'], data_point['demand'], data_point['region']))
                if result is not None:
                    inserted += 1
                
            except Exception as e:
                logger.error(f"Failed to insert sample energy data: {str(e)}")
        
        logger.info(f"Successfully generated {inserted} sample energy data records")
        return inserted

