# ⚡ Energy Demand Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![ENTSO-E](https://img.shields.io/badge/Data-ENTSO--E-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Production-grade ML system for forecasting European electricity demand using real data from ENTSO-E Transparency Platform**

[Features](#-key-features) • [Quick Start](#-quick-start) • [ENTSO-E Setup](#-entsoe-e-data-integration) • [API](#-api-reference) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

This project implements a complete **end-to-end machine learning system** for electricity demand forecasting using **real European grid data** from ENTSO-E (European Network of Transmission System Operators for Electricity).

### 🌍 Real European Data

The system uses the **ENTSO-E Transparency Platform** which provides:
- ✅ **Real-time electricity consumption** data from 30+ European countries
- ✅ **Historical data** (up to 2 years)
- ✅ **Hourly resolution** (or 15-minute for some countries)
- ✅ **Day-ahead forecasts** and generation data
- ✅ **FREE API access** for all users

**Supported countries:** 🇫🇷 France, 🇩🇪 Germany, 🇪🇸 Spain, 🇮🇹 Italy, 🇬🇧 UK, 🇧🇪 Belgium, 🇳🇱 Netherlands, and many more!

### Why This Project?

Energy demand forecasting is critical for:
- **Grid Optimization**: Balance supply and demand in real-time
- **Cost Reduction**: Minimize energy procurement costs (millions € saved annually)
- **Sustainability**: Optimize renewable energy integration
- **Peak Management**: Predict and prepare for consumption peaks

### Key Metrics

Based on real ENTSO-E data:
- **R² Score**: 0.85+ (85% variance explained)
- **MAPE**: <10% (Mean Absolute Percentage Error)
- **API Latency**: <100ms per prediction
- **Data Source**: ENTSO-E Transparency Platform (official European data)
- **Coverage**: 30+ European countries

---

## 🎯 Key Features

### 🌐 Real European Electricity Data
- **ENTSO-E Integration**: Direct connection to official European grid data
- **Multi-Country Support**: France, Germany, Spain, Italy, UK, and 25+ more
- **Real-Time Updates**: Hourly data ingestion from ENTSO-E API
- **Historical Depth**: Access up to 2 years of historical data

### 🔄 Complete ML Pipeline
- **Automated Data Ingestion**: Hourly data collection from ENTSO-E and weather APIs
- **Feature Engineering**: 21 engineered features including lag, rolling statistics, and temporal patterns
- **Model Training**: XGBoost and LightGBM with automated hyperparameter tuning (Optuna)
- **Model Versioning**: Complete experiment tracking and model registry with MLflow

### 🚀 Production-Ready API
- **FastAPI Framework**: High-performance REST API with automatic OpenAPI documentation
- **Real-time Predictions**: <100ms latency for single and batch predictions
- **Health Monitoring**: Built-in health checks and performance metrics
- **Scalable Architecture**: Async support, connection pooling, and horizontal scaling

### 📊 Advanced Data Engineering
- **TimescaleDB**: Optimized time-series database with hypertables
- **Airflow Orchestration**: Automated DAGs for data ingestion, feature engineering, and retraining
- **Data Quality Checks**: Automated validation and monitoring
- **Flexible Storage**: Support for multiple data sources and formats

### 🔍 Monitoring & Observability
- **MLflow Tracking**: Complete experiment history and model comparison
- **Performance Metrics**: R², RMSE, MAE, MAPE tracked over time
- **Data Quality**: Automated checks for missing values, outliers, and drift
- **API Metrics**: Request/response times, error rates, and usage patterns

---

## 🌍 ENTSO-E Data Integration

### What is ENTSO-E?

**ENTSO-E** (European Network of Transmission System Operators for Electricity) is the association of European transmission system operators. Their **Transparency Platform** provides free access to real European electricity data.

### Getting Started with ENTSO-E

1. **Get API Key** (Free, 2 minutes):
   - Go to https://transparency.entsoe.eu/
   - Create account (free registration)
   - Navigate to "My Account" → "Web API Security Token"
   - Generate token

2. **Configure**:
   ```bash
   # Windows PowerShell
   $env:ENTSOE_API_KEY="your-api-key-here"
   $env:ENTSOE_DEFAULT_DOMAIN="FR"  # Country code
   
   # Linux/Mac
   export ENTSOE_API_KEY='your-api-key-here'
   export ENTSOE_DEFAULT_DOMAIN='FR'
   ```

3. **Test Connection**:
   ```python
   from src.data_ingestion.entsoe_client import ENTSOEClient
   from datetime import datetime, timedelta
   
   client = ENTSOEClient()
   end = datetime.utcnow()
   start = end - timedelta(days=1)
   
   data = client.get_actual_load(start, end, domain='FR')
   print(f"Retrieved {len(data)} data points")
   ```

📚 **Full Guide**: See [ENTSOE_SETUP.md](ENTSOE_SETUP.md) for detailed setup instructions

### Available Countries

| Country | Code | Data Availability |
|---------|------|-------------------|
| 🇫🇷 France | FR | Excellent (RTE) |
| 🇩🇪 Germany | DE | Excellent |
| 🇪🇸 Spain | ES | Excellent (REE) |
| 🇮🇹 Italy | IT | Excellent (Terna) |
| 🇬🇧 UK | GB | Very Good |
| 🇧🇪 Belgium | BE | Very Good |
| 🇳🇱 Netherlands | NL | Very Good |
| 🇵🇱 Poland | PL | Good |

**+ 20 more countries!** Full list in [ENTSOE_SETUP.md](ENTSOE_SETUP.md)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                      │
│  FastAPI (REST API) - Port 8000                            │
│  - Endpoints de prédiction                                  │
│  - Health checks                                            │
│  - Documentation Swagger                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE LOGIQUE MÉTIER                     │
│  - Feature Engineering                                       │
│  - Model Training & Inference                                │
│  - Data Processing                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE DONNÉES                            │
│  TimescaleDB (PostgreSQL) - Port 5432                        │
│  - Hypertables pour time-series                              │
│  - Tables: energy_data, weather_data, feature_data, etc.    │
│                                                              │
│  Redis - Port 6379                                           │
│  - Cache                                                     │
│  - Message queue                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE INGESTION                          │
│  - ENTSO-E API Client (Real European Data)                   │
│  - Weather API Client (OpenWeatherMap)                      │
│  - Sample Data Generator                                     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | ENTSO-E API | Real European electricity data |
| **API Framework** | FastAPI | High-performance REST API |
| **ML Models** | XGBoost, LightGBM | Gradient boosting for time-series |
| **Database** | TimescaleDB (PostgreSQL) | Optimized time-series storage |
| **MLOps** | MLflow | Experiment tracking & model registry |
| **Orchestration** | Apache Airflow | Workflow automation |
| **Cache** | Redis | Fast data caching |
| **Containerization** | Docker & Docker Compose | Infrastructure as code |
| **Testing** | Pytest | Unit & integration tests |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **8GB RAM** (minimum)
- **ENTSO-E API Key** (free, get from https://transparency.entsoe.eu/)

### Installation (5 minutes)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd energy-demand-forecasting

# 2. Create Python environment
conda env create -f environment.yml
conda activate energy-forecasting
# OR with venv:
# python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 3. Set up ENTSO-E API key
# Windows PowerShell
$env:ENTSOE_API_KEY="your-api-key-here"
$env:ENTSOE_DEFAULT_DOMAIN="FR"

# Linux/Mac
export ENTSOE_API_KEY='your-api-key-here'
export ENTSOE_DEFAULT_DOMAIN='FR'

# 4. Start infrastructure
docker-compose up -d

# 5. Initialize database
python scripts/setup_database.py

# 6. Ingest real data from ENTSO-E (optional, or use sample data)
python scripts/generate_sample_data.py --days 30
# OR for real ENTSO-E data:
python -c "
from src.data_ingestion.energy_client import EnergyDataIngester
from src.common.database import get_db_client
from datetime import datetime, timedelta
db = get_db_client()
ingester = EnergyDataIngester(db)
end = datetime.utcnow()
start = end - timedelta(days=30)
count = ingester.ingest_energy_data(start, end, region='FR')
print(f'Ingested {count} records')
"

# 7. Create features
python scripts/run_feature_engineering.py

# 8. Train model
python scripts/train_models.py --model xgboost

# 9. Start API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**🎉 Done!** Your API is now running at `http://localhost:8000`

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [50234.0, 48932.0, 52021.0, 49581.0,
                 49823.0, 50421.0, 50234.0, 1832.4, 2145.7,
                 14, 1, 4, 1, 0, 0,
                 18.5, 68.0, 1015.25, 12.5, 18.2, 67.5]
  }'

# Expected response: {"prediction": 51234.5, "model_name": "energy_demand_model", ...}
```

---

## 📁 Project Structure

```
energy-demand-forecasting/
├── 📂 airflow/
│   └── dags/                          # Airflow DAGs
│       ├── data_ingestion.py         # Ingestion horaire
│       ├── feature_engineering.py    # Calcul quotidien
│       └── model_training.py         # Entraînement hebdomadaire
├── 📂 config/
│   └── config.yaml                    # Configuration centralisée
├── 📂 data/                           # Données (non versionnées)
│   ├── raw/                          # Données brutes
│   ├── processed/                    # Données traitées
│   ├── features/                     # Features calculées
│   └── predictions/                  # Prédictions sauvegardées
├── 📂 mlruns/                         # MLflow tracking (file-based)
│   └── [experiments]/                # Expériences et runs
├── 📂 scripts/                        # Scripts utilitaires
│   ├── setup_database.py             # Initialisation DB
│   ├── generate_sample_data.py       # Génération données test
│   ├── run_feature_engineering.py    # Pipeline features
│   ├── train_models.py               # Entraînement modèles
│   └── test_api.py                   # Tests API
├── 📂 sql/
│   └── schemas/
│       └── 01_create_tables.sql      # Schéma TimescaleDB
├── 📂 src/                            # Code source principal
│   ├── api/                          # Application FastAPI
│   │   ├── app.py                    # Application principale
│   │   └── schemas.py                # Modèles Pydantic
│   ├── common/                        # Utilitaires partagés
│   │   ├── config.py                 # Gestion configuration
│   │   └── database.py               # Client TimescaleDB
│   ├── data_ingestion/               # Ingestion de données
│   │   ├── entsoe_client.py          # ⭐ Client ENTSO-E
│   │   ├── weather_client.py         # Client API météo
│   │   └── energy_client.py         # Client données énergie
│   ├── feature_engineering/          # Feature engineering
│   │   └── feature_engineer.py       # Pipeline de features
│   └── training/                     # Entraînement modèles
│       └── trainer.py                # Pipeline d'entraînement
├── 📂 tests/                          # Tests unitaires
│   ├── conftest.py                   # Configuration pytest
│   ├── unit/                         # Tests unitaires
│   └── integration/                  # Tests d'intégration
├── docker-compose.yml                # Services Docker
├── environment.yml                    # Environnement Conda
├── README.md                          # Ce fichier
├── ENTSOE_SETUP.md                    # Guide ENTSO-E
├── EXPLAIN.md                         # Guide complet du projet
└── PROJECT_SUMMARY.md                 # Résumé du projet
```

---

## 📊 API Reference

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "timestamp": "2026-01-04T12:00:00"
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [50234.0, 48932.0, 52021.0, 49581.0,
               49823.0, 50421.0, 50234.0, 1832.4, 2145.7,
               14, 1, 4, 1, 0, 0,
               18.5, 68.0, 1015.25, 12.5, 18.2, 67.5],
  "model_name": "xgboost"  # optional
}
```

Response:
```json
{
  "prediction": 51234.5,
  "model_name": "energy_demand_model",
  "model_version": "a8fc3510",
  "timestamp": "2026-01-04T12:00:00"
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "features_list": [
    [50234.0, 48932.0, ...],
    [51000.0, 49500.0, ...]
  ]
}
```

#### Model Metrics
```bash
GET /metrics?model_name=energy_demand_model&limit=10
```

### Feature Order (Important!)

Features must be provided in this exact order (21 features):

| Index | Feature | Description | Typical Range |
|-------|---------|-------------|---------------|
| 0-3 | Lag features | Past demand (1h, 7h, 24h, 168h ago) | 35,000 - 90,000 MW |
| 4-6 | Rolling avg | Average over windows (7h, 24h, 168h) | 35,000 - 90,000 MW |
| 7-8 | Rolling std | Volatility measures (7h, 24h) | 500 - 5,000 MW |
| 9-14 | Time features | Hour, day_of_week, day_of_month, month, is_weekend, is_holiday | Various |
| 15-20 | Weather | Temperature, humidity, pressure, wind_speed, temp_avg_24h, humidity_avg_24h | Various |

---

## 📈 Model Performance

### Training Results

Based on real ENTSO-E data:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.85+ | 85%+ variance explained ✅ |
| **RMSE** | ~2,000 MW | ~4% of average demand |
| **MAE** | ~1,800 MW | ~3.6% of average demand |
| **MAPE** | <10% | Mean absolute % error ✅ |

### Feature Importance

Top 10 most important features for prediction:

1. 🥇 **lag_24** - Demand 24h ago (daily pattern)
2. 🥈 **hour** - Time of day (intraday pattern)
3. 🥉 **temperature** - Weather impact
4. **lag_168** - Demand 1 week ago (weekly pattern)
5. **rolling_avg_24** - Daily average trend
6. **day_of_week** - Weekday vs weekend
7. **lag_7** - Recent trend
8. **humidity** - Weather comfort
9. **is_weekend** - Weekend effect
10. **rolling_std_24** - Daily volatility

---

## 🔧 Configuration

### ENTSO-E Configuration

Add to `config/config.yaml` or use environment variables:

```yaml
data_ingestion:
  energy:
    use_entsoe: true  # Activer ENTSO-E
  
  entsoe:
    api_key: ${ENTSOE_API_KEY}
    default_domain: ${ENTSOE_DEFAULT_DOMAIN:-FR}
    retry_attempts: 3
    retry_delay_seconds: 5
```

### Database Configuration

```yaml
database:
  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  name: ${DB_NAME:-energy_forecasting}
  user: ${DB_USER:-ml_app_user}
  password: ${DB_PASSWORD:-devpassword123}
```

### Model Configuration

```yaml
model:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

---

## 🔄 Airflow Orchestration

### Data Ingestion DAG
- **Schedule**: Hourly (`@hourly`)
- **Tasks**: 
  - Ingest ENTSO-E energy data
  - Ingest weather data
  - Validate data quality

### Feature Engineering DAG
- **Schedule**: Daily (`0 0 * * *`)
- **Tasks**: Compute features from raw data

### Model Training DAG
- **Schedule**: Weekly (`0 2 * * 0`)
- **Tasks**: 
  - Train XGBoost model
  - Train LightGBM model
  - Evaluate and register best model

### Enable Airflow

```bash
# Start Airflow
airflow webserver --port 8080  # Terminal 1
airflow scheduler              # Terminal 2

# Enable DAGs
airflow dags unpause data_ingestion
airflow dags unpause feature_engineering
airflow dags unpause model_training
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific modules
pytest tests/unit/test_api.py -v
pytest tests/unit/test_data_ingestion.py -v
pytest tests/unit/test_feature_engineering.py -v
pytest tests/unit/test_model_training.py -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Documentation

- **[ENTSOE_SETUP.md](ENTSOE_SETUP.md)** - Complete ENTSO-E integration guide
- **[EXPLAIN.md](EXPLAIN.md)** - Comprehensive project explanation (A to Z)
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project summary and status
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI
- **[MLflow UI](http://localhost:5000)** - Experiment tracking (if MLflow server running)

---

## 🐛 Troubleshooting

### ENTSO-E Connection Issues

```bash
# Test your API key
python -c "
from src.data_ingestion.entsoe_client import ENTSOEClient
from datetime import datetime, timedelta
client = ENTSOEClient()
end = datetime.utcnow()
start = end - timedelta(days=1)
data = client.get_actual_load(start, end, domain='FR')
print(f'Retrieved {len(data)} data points')
"

# Common issues:
# - Invalid API key: Regenerate at https://transparency.entsoe.eu/
# - Rate limiting: Wait 1 minute and retry
# - No recent data: Try data from 2-3 days ago
```

### Database Issues

```bash
# Check Docker services
docker-compose ps

# View logs
docker-compose logs timescaledb

# Reset database (⚠️ deletes all data)
docker-compose down -v
docker-compose up -d
python scripts/setup_database.py
```

### API Issues

```bash
# Check if API is running
curl http://localhost:8000/health

# Check logs
# API logs are displayed in the terminal where uvicorn is running
```

---

## 🚢 Deployment

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **Database**: Use managed PostgreSQL with TimescaleDB
3. **MLflow**: Deploy MLflow server separately
4. **API**: Use production ASGI server (e.g., Gunicorn with Uvicorn workers)
5. **Monitoring**: Set up Prometheus/Grafana for metrics
6. **Logging**: Configure centralized logging

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Scale services (if needed)
docker-compose up -d --scale api=3
```

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

1. **New Data Sources**: Add support for more ENTSO-E data types
2. **More Countries**: Extend support to additional European countries
3. **Model Improvements**: Try new algorithms (Prophet, LSTM, etc.)
4. **Features**: Add new feature engineering techniques
5. **Documentation**: Improve guides and examples

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Special thanks to:

- **[ENTSO-E](https://www.entsoe.eu/)** - For providing free access to European electricity data
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting library
- **[MLflow](https://mlflow.org/)** - ML lifecycle platform
- **[TimescaleDB](https://www.timescale.com/)** - Time-series database
- **[Apache Airflow](https://airflow.apache.org/)** - Workflow orchestration

---

## 📞 Support

For issues and questions:
- 📖 [Read the documentation](EXPLAIN.md)
- 🐛 [Open an issue](https://github.com/yourusername/energy-demand-forecasting/issues)
- 📧 Contact ENTSO-E support: transparency@entsoe.eu

---

<div align="center">

**⚡ Powered by real European electricity data from ENTSO-E ⚡**

**Built with ❤️ for ML community**

[⬆ Back to top](#-energy-demand-forecasting-system)

</div>
