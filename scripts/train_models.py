"""
Train models script.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.database import get_db_client
from src.training import ModelTrainer


def train_model(model_name: str, hyperparameter_tuning: bool = False):
    """Train a model."""
    print(f"Training {model_name} model...")
    
    db = get_db_client()
    trainer = ModelTrainer(db)
    
    try:
        result = trainer.train_with_mlflow(
            model_name=model_name,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        print(f"\nModel training complete!")
        print(f"   Model: {model_name}")
        print(f"   R2 Score: {result['metrics']['r2']:.4f}")
        print(f"   RMSE: {result['metrics']['rmse']:.2f}")
        print(f"   MAE: {result['metrics']['mae']:.2f}")
        print(f"   MAPE: {result['metrics']['mape']:.4f}")
        print(f"   MLflow Run ID: {result['run_id']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Model training failed: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--model", type=str, choices=['xgboost', 'lightgbm'], 
                       default='xgboost', help="Model to train")
    parser.add_argument("--tune", action='store_true', 
                       help="Enable hyperparameter tuning")
    args = parser.parse_args()
    
    success = train_model(args.model, args.tune)
    sys.exit(0 if success else 1)

