"""
API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_name: Optional[str] = Field(None, description="Model name to use (default: latest)")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    features_list: List[List[float]] = Field(..., description="List of feature vectors")
    model_name: Optional[str] = Field(None, description="Model name to use (default: latest)")


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    prediction: float = Field(..., description="Predicted energy demand")
    model_name: Optional[str] = Field(None, description="Model used for prediction")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[float] = Field(..., description="List of predictions")
    model_name: Optional[str] = Field(None, description="Model used for prediction")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database connection status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class MetricsResponse(BaseModel):
    """Model metrics response."""
    model_name: str
    model_version: str
    metrics: dict
    evaluation_timestamp: datetime

