"""
ENTSO-E Transparency Platform API client for energy data.
"""
import requests
import time
import xml.etree.ElementTree as ET
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import os

logger = logging.getLogger(__name__)

# ENTSO-E Domain codes for European countries
ENTSOE_DOMAINS = {
    'FR': '10YFR-RTE------C',  # France
    'DE': '10Y1001A1001A83F',  # Germany
    'IT': '10YIT-GRTN-----B',  # Italy
    'ES': '10YES-REE------0',  # Spain
    'NL': '10YNL----------L',  # Netherlands
    'BE': '10YBE----------2',  # Belgium
    'AT': '10YAT-APG------L',  # Austria
    'CH': '10YCH-SWISSGRIDZ',  # Switzerland
    'GB': '10YGB----------A',  # Great Britain
    'PL': '10YPL-AREA-----S',  # Poland
    'PT': '10YPT-REN------W',  # Portugal
    'DK': '10YDK-1--------W',  # Denmark
    'SE': '10YSE-1--------K',  # Sweden
    'NO': '10YNO-0--------C',  # Norway
    'FI': '10YFI-1--------U',  # Finland
    'CZ': '10YCZ-CEPS-----N',  # Czech Republic
    'HU': '10YHU-MAVIR----U',  # Hungary
    'RO': '10YRO-TEL------P',  # Romania
    'GR': '10YGR-HTSO-----Y',  # Greece
    'IE': '10YIE-1001A00010',  # Ireland
    'default': '10YFR-RTE------C'  # Default to France
}

# ENTSO-E Document Types
DOCUMENT_TYPES = {
    'actual_total_load': 'A65',  # Actual Total Load
    'day_ahead_total_load_forecast': 'A61',  # Day-ahead Total Load Forecast
    'week_ahead_total_load_forecast': 'A62',  # Week-ahead Total Load Forecast
    'month_ahead_total_load_forecast': 'A63',  # Month-ahead Total Load Forecast
    'year_ahead_total_load_forecast': 'A64',  # Year-ahead Total Load Forecast
}


class ENTSOEClient:
    """Client for ENTSO-E Transparency Platform API."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ENTSO-E client.
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                entsoe_config = config.get('data_ingestion', {}).get('entsoe', {})
        else:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    entsoe_config = config.get('data_ingestion', {}).get('entsoe', {})
            else:
                entsoe_config = {}
        
        self.api_url = os.getenv('ENTSOE_API_URL', entsoe_config.get('api_url', 'https://web-api.tp.entsoe.eu/api'))
        self.api_key = os.getenv('ENTSOE_API_KEY', entsoe_config.get('api_key', ''))
        self.retry_attempts = entsoe_config.get('retry_attempts', 3)
        self.retry_delay = entsoe_config.get('retry_delay_seconds', 5)
        self.default_domain = entsoe_config.get('default_domain', 'FR')
        
        if not self.api_key:
            logger.warning("ENTSO-E API key not provided. Set ENTSOE_API_KEY environment variable or in config.yaml")
    
    def _make_request(self, params: Dict) -> Optional[str]:
        """
        Make API request to ENTSO-E with retry logic.
        
        Args:
            params: Request parameters
            
        Returns:
            XML response as string or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"ENTSO-E API request failed (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All ENTSO-E API retry attempts failed: {str(e)}")
                    return None
        
        return None
    
    def _parse_xml_response(self, xml_string: str) -> List[Dict]:
        """
        Parse ENTSO-E XML response and extract time series data.
        
        Args:
            xml_string: XML response from API
            
        Returns:
            List of data points with timestamp and value
        """
        try:
            root = ET.fromstring(xml_string)
            
            # ENTSO-E XML namespace
            ns = {
                'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0',
                'md': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:3:0'
            }
            
            data_points = []
            
            # Find all TimeSeries elements
            for time_series in root.findall('.//ns:TimeSeries', ns):
                # Get period
                period = time_series.find('.//ns:Period', ns)
                if period is None:
                    continue
                
                # Get start time
                time_interval = period.find('.//ns:timeInterval', ns)
                if time_interval is None:
                    continue
                
                start_elem = time_interval.find('.//ns:start', ns)
                if start_elem is None:
                    continue
                
                start_time = datetime.fromisoformat(start_elem.text.replace('Z', '+00:00'))
                
                # Get resolution (PT60M = 60 minutes, PT15M = 15 minutes)
                resolution_elem = period.find('.//ns:resolution', ns)
                resolution_minutes = 60  # Default
                if resolution_elem is not None:
                    resolution_str = resolution_elem.text
                    if 'PT15M' in resolution_str:
                        resolution_minutes = 15
                    elif 'PT30M' in resolution_str:
                        resolution_minutes = 30
                    elif 'PT60M' in resolution_str:
                        resolution_minutes = 60
                
                # Get all points
                points = period.findall('.//ns:Point', ns)
                for i, point in enumerate(points):
                    position_elem = point.find('.//ns:position', ns)
                    quantity_elem = point.find('.//ns:quantity', ns)
                    
                    if position_elem is not None and quantity_elem is not None:
                        # Calculate timestamp based on position
                        timestamp = start_time + timedelta(minutes=resolution_minutes * (int(position_elem.text) - 1))
                        value = float(quantity_elem.text)
                        
                        data_points.append({
                            'timestamp': timestamp,
                            'demand': value,  # MW
                            'position': int(position_elem.text)
                        })
            
            return data_points
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse ENTSO-E XML response: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing ENTSO-E XML response: {str(e)}")
            return []
    
    def get_actual_load(self, start_date: datetime, end_date: datetime, 
                       domain: Optional[str] = None) -> List[Dict]:
        """
        Get actual total load (consumption) data from ENTSO-E.
        
        Args:
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            domain: Country domain code (e.g., 'FR', 'DE') or full domain code
            
        Returns:
            List of data points with timestamp and demand
        """
        if not self.api_key:
            logger.warning("ENTSO-E API key not configured. Cannot fetch data.")
            return []
        
        # Get domain code
        if domain is None:
            domain = self.default_domain
        
        if domain in ENTSOE_DOMAINS:
            domain_code = ENTSOE_DOMAINS[domain]
        else:
            # Assume it's already a full domain code
            domain_code = domain
        
        # Format dates for API (YYYYMMDDHHmm format)
        start_str = start_date.strftime('%Y%m%d%H%M')
        end_str = end_date.strftime('%Y%m%d%H%M')
        
        # Build request parameters
        params = {
            'securityToken': self.api_key,
            'documentType': DOCUMENT_TYPES['actual_total_load'],
            'processType': 'A16',  # Realised
            'in_Domain': domain_code,
            'out_Domain': domain_code,
            'periodStart': start_str,
            'periodEnd': end_str
        }
        
        logger.info(f"Fetching ENTSO-E data for domain {domain_code} from {start_date} to {end_date}")
        
        # Make request
        xml_response = self._make_request(params)
        
        if not xml_response:
            logger.warning("No response from ENTSO-E API")
            return []
        
        # Parse XML
        data_points = self._parse_xml_response(xml_response)
        
        if not data_points:
            logger.warning("No data points extracted from ENTSO-E response")
            return []
        
        logger.info(f"Successfully fetched {len(data_points)} data points from ENTSO-E")
        
        return data_points
    
    def get_day_ahead_forecast(self, start_date: datetime, end_date: datetime,
                              domain: Optional[str] = None) -> List[Dict]:
        """
        Get day-ahead load forecast from ENTSO-E.
        
        Args:
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            domain: Country domain code
            
        Returns:
            List of forecast data points
        """
        if not self.api_key:
            logger.warning("ENTSO-E API key not configured. Cannot fetch data.")
            return []
        
        # Get domain code
        if domain is None:
            domain = self.default_domain
        
        if domain in ENTSOE_DOMAINS:
            domain_code = ENTSOE_DOMAINS[domain]
        else:
            domain_code = domain
        
        # Format dates
        start_str = start_date.strftime('%Y%m%d%H%M')
        end_str = end_date.strftime('%Y%m%d%H%M')
        
        params = {
            'securityToken': self.api_key,
            'documentType': DOCUMENT_TYPES['day_ahead_total_load_forecast'],
            'processType': 'A01',  # Day ahead
            'in_Domain': domain_code,
            'out_Domain': domain_code,
            'periodStart': start_str,
            'periodEnd': end_str
        }
        
        xml_response = self._make_request(params)
        
        if not xml_response:
            return []
        
        return self._parse_xml_response(xml_response)

