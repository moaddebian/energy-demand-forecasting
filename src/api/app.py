"""
FastAPI application for energy demand forecasting.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import numpy as np
import os
from datetime import datetime
from typing import List
import logging

from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, MetricsResponse
)
from src.common.database import get_db_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Energy Demand Forecasting API",
    description="API for predicting energy demand using ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model will be loaded lazily on first request
_model = None
_model_name = None
_model_version = None

def get_model(model_name: str = None):
    """Lazy load the model on first use."""
    global _model, _model_name, _model_version
    if _model is None:
        # Priority 1: Try environment variable for explicit path
        local_path = os.getenv("MLFLOW_MODEL_LOCAL_PATH")
        if local_path and os.path.exists(local_path):
            try:
                _model = mlflow.pyfunc.load_model(local_path)
                _model_name = "energy_demand_model"
                _model_version = "local"
                logger.info(f"Model loaded from local path: {local_path}")
                return _model
            except Exception as e:
                logger.warning(f"Failed to load from MLFLOW_MODEL_LOCAL_PATH: {str(e)}")
        
        # Priority 2: Try to find latest model in mlruns directory
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        mlruns_dir = project_root / "mlruns"
        
        if mlruns_dir.exists():
            # Find the latest run with a model
            experiment_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name != "models"]
            latest_model_path = None
            latest_run_id = None
            
            for exp_dir in experiment_dirs:
                if (exp_dir / "meta.yaml").exists():
                    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
                    for run_dir in run_dirs:
                        model_path = run_dir / "artifacts" / "model"
                        if model_path.exists() and (model_path / "MLmodel").exists():
                            # Use the most recent run (by directory name/number)
                            if latest_run_id is None or run_dir.name > latest_run_id:
                                latest_model_path = str(model_path)
                                latest_run_id = run_dir.name
            
            if latest_model_path:
                try:
                    _model = mlflow.pyfunc.load_model(latest_model_path)
                    _model_name = "energy_demand_model"
                    _model_version = latest_run_id[:8] if latest_run_id else "latest"
                    logger.info(f"Model loaded from mlruns: {latest_model_path}")
                    return _model
                except Exception as e:
                    logger.warning(f"Failed to load from mlruns: {str(e)}")
        
        # Priority 3: Try model registry (if MLflow server is running)
        try:
            model_uri = os.getenv(
                "MLFLOW_MODEL_URI", 
                f"models:/energy_demand_model/{model_name or 'production'}"
            )
            _model = mlflow.pyfunc.load_model(model_uri)
            
            # Extract model info
            if "models:/" in model_uri:
                parts = model_uri.split("/")
                _model_name = parts[1] if len(parts) > 1 else "energy_demand_model"
                _model_version = parts[2] if len(parts) > 2 else "production"
            
            logger.info(f"Model loaded from registry: {model_uri}")
            return _model
        except Exception as e:
            logger.error(f"Failed to load model from registry: {str(e)}")
        
        # If all methods failed
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train a model first using: python scripts/train_models.py --model xgboost"
        )
    return _model

def check_database_health() -> str:
    """Check database connection health."""
    try:
        db = get_db_client()
        db.execute_query("SELECT 1")
        return "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return "disconnected"

@app.get("/", tags=["Root"])
def root():
    """Root endpoint."""
    return {
        "message": "Energy Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint."""
    db_status = check_database_health()
    model_loaded = _model is not None
    
    status = "healthy" if db_status == "connected" and model_loaded else "degraded"
    
    return HealthResponse(
        status=status,
        database=db_status,
        model_loaded=model_loaded,
        timestamp=datetime.utcnow()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):
    """Make a single prediction."""
    try:
        model = get_model(request.model_name)
        
        # Convert features to the required format (NumPy array)
        feature_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_name=_model_name,
            model_version=_model_version,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        model = get_model(request.model_name)
        
        # Convert features to NumPy array
        feature_array = np.array(request.features_list)
        
        # Make predictions
        predictions = model.predict(feature_array)
        
        # Convert to list if it's a numpy array, otherwise use as-is
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions_list],
            model_name=_model_name,
            model_version=_model_version,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics", response_model=List[MetricsResponse], tags=["Metrics"])
def get_metrics(model_name: str = "energy_demand_model", limit: int = 10):
    """Get model performance metrics."""
    try:
        db = get_db_client()
        query = """
            SELECT model_name, model_version, metric_name, metric_value, evaluation_timestamp
            FROM model_metrics
            WHERE model_name = %s
            ORDER BY evaluation_timestamp DESC
            LIMIT %s
        """
        results = db.execute_query(query, (model_name, limit))
        
        # Group metrics by model version
        metrics_by_version = {}
        for row in results:
            version = row['model_version']
            if version not in metrics_by_version:
                metrics_by_version[version] = {
                    'model_name': row['model_name'],
                    'model_version': version,
                    'metrics': {},
                    'evaluation_timestamp': row['evaluation_timestamp']
                }
            metrics_by_version[version]['metrics'][row['metric_name']] = float(row['metric_value'])
        
        return [
            MetricsResponse(**data)
            for data in metrics_by_version.values()
        ]
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")
