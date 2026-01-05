"""
Generate sample data for testing.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.database import get_db_client
from src.data_ingestion import EnergyDataIngester, WeatherDataIngester


def generate_sample_data(days: int = 30):
    """Generate sample energy and weather data."""
    print(f"Generating {days} days of sample data...")
    
    db = get_db_client()
    
    # Generate energy data
    print("Generating energy data...")
    energy_ingester = EnergyDataIngester(db)
    energy_count = energy_ingester.generate_sample_data(days=days)
    print(f"Generated {energy_count} energy data records")
    
    # Generate weather data (simplified)
    print("Generating weather data...")
    # Note: Weather data generation would require API or synthetic generation
    # For now, we'll skip this or use the weather API if available
    print("Weather data generation requires API key or manual generation")
    
    print(f"\nSample data generation complete!")
    print(f"   - Energy records: {energy_count}")
    print(f"   - Time range: Last {days} days")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample data")
    parser.add_argument("--days", type=int, default=30, help="Number of days of data to generate")
    args = parser.parse_args()
    
    generate_sample_data(args.days)

