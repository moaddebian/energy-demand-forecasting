-- ============================================================================
-- Energy Demand Forecasting - Database Schema
-- ============================================================================
-- TimescaleDB extension for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- Raw Data Tables
-- ============================================================================

-- Weather data table
CREATE TABLE IF NOT EXISTS weather_data (
    timestamp TIMESTAMPTZ NOT NULL,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(7, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    cloud_cover DECIMAL(5, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp)
);

-- Energy consumption data table
CREATE TABLE IF NOT EXISTS energy_data (
    timestamp TIMESTAMPTZ NOT NULL,
    demand DECIMAL(10, 2) NOT NULL,
    region VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, region)
);

-- ============================================================================
-- Feature Engineering Tables
-- ============================================================================

-- Feature data table (engineered features)
CREATE TABLE IF NOT EXISTS feature_data (
    timestamp TIMESTAMPTZ NOT NULL,
    demand DECIMAL(10, 2) NOT NULL,
    
    -- Lag features
    lag_1 DECIMAL(10, 2),
    lag_7 DECIMAL(10, 2),
    lag_24 DECIMAL(10, 2),
    lag_168 DECIMAL(10, 2),  -- Weekly lag
    
    -- Rolling statistics
    rolling_avg_7 DECIMAL(10, 2),
    rolling_avg_24 DECIMAL(10, 2),
    rolling_avg_168 DECIMAL(10, 2),
    rolling_std_7 DECIMAL(10, 2),
    rolling_std_24 DECIMAL(10, 2),
    
    -- Time-based features
    hour INTEGER,
    day_of_week INTEGER,
    day_of_month INTEGER,
    month INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    
    -- Weather features
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(7, 2),
    wind_speed DECIMAL(5, 2),
    
    -- Weather rolling averages
    temp_rolling_avg_24 DECIMAL(5, 2),
    humidity_rolling_avg_24 DECIMAL(5, 2),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp)
);

-- ============================================================================
-- Model & Prediction Tables
-- ============================================================================

-- Model predictions table
CREATE TABLE IF NOT EXISTS predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction DECIMAL(10, 2) NOT NULL,
    confidence_interval_lower DECIMAL(10, 2),
    confidence_interval_upper DECIMAL(10, 2),
    actual_value DECIMAL(10, 2),
    prediction_error DECIMAL(10, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, model_name, model_version)
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10, 4) NOT NULL,
    evaluation_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (evaluation_timestamp, model_name, model_version, metric_name)
);

-- Model training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    training_started_at TIMESTAMPTZ NOT NULL,
    training_completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running',
    hyperparameters JSONB,
    mlflow_run_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Data Quality Tables
-- ============================================================================

-- Data quality checks table
CREATE TABLE IF NOT EXISTS data_quality_checks (
    check_timestamp TIMESTAMPTZ NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    check_status VARCHAR(20) NOT NULL,  -- 'pass', 'fail', 'warning'
    check_message TEXT,
    records_checked INTEGER,
    records_failed INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (check_timestamp, table_name, check_name)
);

-- ============================================================================
-- Create Hypertables (TimescaleDB optimization)
-- ============================================================================

-- Convert regular tables to hypertables for time-series optimization
SELECT create_hypertable('weather_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('energy_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('feature_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('model_metrics', 'evaluation_timestamp', if_not_exists => TRUE);
SELECT create_hypertable('data_quality_checks', 'check_timestamp', if_not_exists => TRUE);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Indexes on timestamp columns (already created by hypertable, but adding for clarity)
CREATE INDEX IF NOT EXISTS idx_weather_data_timestamp ON weather_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_energy_data_timestamp ON energy_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feature_data_timestamp ON feature_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, model_version);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_feature_data_time_features ON feature_data(hour, day_of_week, month);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model ON model_metrics(model_name, model_version, evaluation_timestamp DESC);

-- ============================================================================
-- Retention Policies (Optional - for data management)
-- ============================================================================

-- Keep raw data for 2 years
-- SELECT add_retention_policy('weather_data', INTERVAL '2 years', if_not_exists => TRUE);
-- SELECT add_retention_policy('energy_data', INTERVAL '2 years', if_not_exists => TRUE);

-- Keep predictions for 1 year
-- SELECT add_retention_policy('predictions', INTERVAL '1 year', if_not_exists => TRUE);

