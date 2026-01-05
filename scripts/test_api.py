"""
Script de test simple pour l'API de prédiction.
"""
import requests
import json
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def test_health_check() -> bool:
    """Test du endpoint de health check."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Database: {data['database']}")
        print(f"  Model loaded: {data['model_loaded']}")
        return True
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def test_predict(features: list) -> bool:
    """Test du endpoint de prédiction."""
    print("\nTesting /predict endpoint...")
    try:
        payload = {
            "features": features,
            "model_name": "xgboost"
        }
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        print(f"  Prediction: {data['prediction']:.2f}")
        print(f"  Model: {data['model_name']}")
        print(f"  Version: {data['model_version']}")
        return True
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False


def test_batch_predict(features_list: list) -> bool:
    """Test du endpoint de prédiction par lot."""
    print("\nTesting /predict/batch endpoint...")
    try:
        payload = {
            "features_list": features_list
        }
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        print(f"  Predictions: {len(data['predictions'])}")
        print(f"  First prediction: {data['predictions'][0]:.2f}")
        return True
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def test_metrics() -> bool:
    """Test du endpoint de métriques."""
    print("\nTesting /metrics endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/metrics?limit=5")
        response.raise_for_status()
        data = response.json()
        print(f"  Metrics found: {len(data)}")
        if data:
            for metric in data:
                print(f"  Model: {metric['model_name']}, Version: {metric['model_version']}")
                print(f"    Metrics: {list(metric['metrics'].keys())}")
        return True
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def main():
    """Exécuter tous les tests."""
    print("=" * 50)
    print("API Testing Script")
    print("=" * 50)
    
    # Test health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("\nHealth check failed. Is the API running?")
        print("Start the API with: uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Exemple de features (21 features - sans demand qui est la target)
    # Format: [lag_1, lag_7, lag_24, lag_168, rolling_avg_7, rolling_avg_24, 
    #          rolling_avg_168, rolling_std_7, rolling_std_24, hour, day_of_week, 
    #          day_of_month, month, is_weekend, is_holiday, temperature, humidity, 
    #          pressure, wind_speed, temp_rolling_avg_24, humidity_rolling_avg_24]
    sample_features = [
        950.0,   # lag_1
        980.0,   # lag_7
        1020.0,  # lag_24
        1050.0,  # lag_168
        990.0,   # rolling_avg_7
        1005.0,  # rolling_avg_24
        1010.0,  # rolling_avg_168
        25.5,    # rolling_std_7
        30.2,    # rolling_std_24
        14,      # hour
        1,       # day_of_week
        4,       # day_of_month
        1,       # month
        0,       # is_weekend (0 or 1)
        0,       # is_holiday (0 or 1)
        20.5,    # temperature
        65.0,    # humidity
        1013.25, # pressure
        10.5,    # wind_speed
        20.0,    # temp_rolling_avg_24
        65.0     # humidity_rolling_avg_24
    ]
    
    # Test prédiction simple
    predict_ok = test_predict(sample_features)
    
    # Test prédiction par lot
    batch_features = [sample_features, sample_features]
    batch_ok = test_batch_predict(batch_features)
    
    # Test métriques
    metrics_ok = test_metrics()
    
    # Résumé
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Predict: {'PASS' if predict_ok else 'FAIL'}")
    print(f"Batch Predict: {'PASS' if batch_ok else 'FAIL'}")
    print(f"Metrics: {'PASS' if metrics_ok else 'FAIL'}")
    
    all_passed = all([health_ok, predict_ok, batch_ok, metrics_ok])
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()

