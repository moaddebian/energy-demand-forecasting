"""
Run feature engineering pipeline.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.database import get_db_client
from src.feature_engineering import FeatureEngineer


def run_feature_engineering():
    """Run feature engineering pipeline."""
    print("Running feature engineering pipeline...")
    
    db = get_db_client()
    feature_engineer = FeatureEngineer(db)
    
    # Compute features for last 7 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    try:
        features_df = feature_engineer.engineer_features(start_date, end_date)
        
        if features_df.empty:
            print("WARNING: No features computed - check if raw data exists")
            return False
        
        print(f"Features computed: {len(features_df)} records")
        
        # Save to database
        count = feature_engineer.save_features_to_db(features_df)
        print(f"Features saved to database: {count} records")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Feature engineering failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_feature_engineering()
    sys.exit(0 if success else 1)

