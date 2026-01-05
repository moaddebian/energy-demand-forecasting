"""
Feature engineering for energy demand forecasting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for energy demand forecasting."""
    
    def __init__(self, db_client, config_path: Optional[str] = None):
        """
        Initialize feature engineer.
        
        Args:
            db_client: Database client instance
            config_path: Path to config.yaml file
        """
        self.db = db_client
        
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.feature_config = config.get('feature_engineering', {})
        else:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.feature_config = config.get('feature_engineering', {})
            else:
                self.feature_config = {}
        
        self.lags = self.feature_config.get('lags', [1, 7, 24, 168])
        self.rolling_windows = self.feature_config.get('rolling_windows', [7, 24, 168])
        self.time_features_config = self.feature_config.get('time_features', {})
        self.weather_features_config = self.feature_config.get('weather_features', {})
    
    def load_energy_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load energy data from database.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        
        Returns:
            DataFrame with energy data
        """
        query = "SELECT timestamp, demand, region FROM energy_data"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= %s")
                params.append(end_date)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        data = self.db.execute_query(query, tuple(params) if params else None)
        
        if not data:
            logger.warning("No energy data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        return df
    
    def load_weather_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load weather data from database.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        
        Returns:
            DataFrame with weather data
        """
        query = """
            SELECT timestamp, temperature, humidity, pressure, wind_speed
            FROM weather_data
        """
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= %s")
                params.append(end_date)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        data = self.db.execute_query(query, tuple(params) if params else None)
        
        if not data:
            logger.warning("No weather data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        """
        Create lag features.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
        
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for lag in self.lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        """
        Create rolling average and standard deviation features.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
        
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for window in self.rolling_windows:
            df[f'rolling_avg_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        if self.time_features_config.get('include_hour', True):
            df['hour'] = df.index.hour
        
        if self.time_features_config.get('include_day_of_week', True):
            df['day_of_week'] = df.index.dayofweek
        
        if self.time_features_config.get('include_day_of_month', True):
            df['day_of_month'] = df.index.day
        
        if self.time_features_config.get('include_month', True):
            df['month'] = df.index.month
        
        if self.time_features_config.get('include_is_weekend', True):
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        if self.time_features_config.get('include_is_holiday', True):
            # Simple holiday detection (can be enhanced with holiday calendar)
            df['is_holiday'] = 0  # Placeholder - implement holiday calendar
        
        return df
    
    def merge_weather_features(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather features with energy data.
        
        Args:
            energy_df: Energy data DataFrame
            weather_df: Weather data DataFrame
        
        Returns:
            Merged DataFrame
        """
        if weather_df.empty:
            logger.warning("No weather data to merge")
            return energy_df
        
        # Resample weather data to match energy data frequency (hourly)
        weather_resampled = weather_df.resample('H').mean()
        
        # Merge on timestamp
        merged_df = energy_df.merge(
            weather_resampled,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Forward fill missing weather values
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in weather_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Create weather rolling averages if configured
        if self.weather_features_config.get('rolling_avg_window'):
            window = self.weather_features_config['rolling_avg_window']
            if 'temperature' in merged_df.columns:
                merged_df['temp_rolling_avg_24'] = merged_df['temperature'].rolling(window=window, min_periods=1).mean()
            if 'humidity' in merged_df.columns:
                merged_df['humidity_rolling_avg_24'] = merged_df['humidity'].rolling(window=window, min_periods=1).mean()
        
        return merged_df
    
    def engineer_features(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            start_date: Start date for feature engineering
            end_date: End date for feature engineering
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Load data
        energy_df = self.load_energy_data(start_date, end_date)
        if energy_df.empty:
            logger.error("No energy data available for feature engineering")
            return pd.DataFrame()
        
        weather_df = self.load_weather_data(start_date, end_date)
        
        # Create features
        logger.info("Creating lag features")
        energy_df = self.create_lag_features(energy_df)
        
        logger.info("Creating rolling features")
        energy_df = self.create_rolling_features(energy_df)
        
        logger.info("Creating time features")
        energy_df = self.create_time_features(energy_df)
        
        logger.info("Merging weather features")
        energy_df = self.merge_weather_features(energy_df, weather_df)
        
        # Drop rows with NaN values (from lag features)
        initial_rows = len(energy_df)
        energy_df = energy_df.dropna()
        dropped_rows = initial_rows - len(energy_df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        logger.info(f"Feature engineering complete. Final dataset shape: {energy_df.shape}")
        
        return energy_df
    
    def save_features_to_db(self, features_df: pd.DataFrame) -> int:
        """
        Save engineered features to database.
        
        Args:
            features_df: DataFrame with engineered features
        
        Returns:
            Number of records inserted
        """
        if features_df.empty:
            logger.warning("No features to save")
            return 0
        
        # Reset index to get timestamp as column
        df = features_df.reset_index()
        
        inserted = 0
        errors = 0
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                for _, row in df.iterrows():
                    try:
                        query = """
                            INSERT INTO feature_data (
                                timestamp, demand,
                                lag_1, lag_7, lag_24, lag_168,
                                rolling_avg_7, rolling_avg_24, rolling_avg_168,
                                rolling_std_7, rolling_std_24,
                                hour, day_of_week, day_of_month, month, is_weekend, is_holiday,
                                temperature, humidity, pressure, wind_speed,
                                temp_rolling_avg_24, humidity_rolling_avg_24
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (timestamp) DO UPDATE SET
                                demand = EXCLUDED.demand,
                                lag_1 = EXCLUDED.lag_1,
                                lag_7 = EXCLUDED.lag_7,
                                lag_24 = EXCLUDED.lag_24,
                                lag_168 = EXCLUDED.lag_168,
                                rolling_avg_7 = EXCLUDED.rolling_avg_7,
                                rolling_avg_24 = EXCLUDED.rolling_avg_24,
                                rolling_avg_168 = EXCLUDED.rolling_avg_168,
                                rolling_std_7 = EXCLUDED.rolling_std_7,
                                rolling_std_24 = EXCLUDED.rolling_std_24,
                                hour = EXCLUDED.hour,
                                day_of_week = EXCLUDED.day_of_week,
                                day_of_month = EXCLUDED.day_of_month,
                                month = EXCLUDED.month,
                                is_weekend = EXCLUDED.is_weekend,
                                is_holiday = EXCLUDED.is_holiday,
                                temperature = EXCLUDED.temperature,
                                humidity = EXCLUDED.humidity,
                                pressure = EXCLUDED.pressure,
                                wind_speed = EXCLUDED.wind_speed,
                                temp_rolling_avg_24 = EXCLUDED.temp_rolling_avg_24,
                                humidity_rolling_avg_24 = EXCLUDED.humidity_rolling_avg_24
                        """
                        
                        # Convert boolean values properly
                        is_weekend_val = row.get('is_weekend', None)
                        if is_weekend_val is not None:
                            is_weekend_val = bool(is_weekend_val)
                        
                        is_holiday_val = row.get('is_holiday', None)
                        if is_holiday_val is not None:
                            is_holiday_val = bool(is_holiday_val)
                        
                        values = (
                            row['timestamp'],
                            row.get('demand', None),
                            row.get('lag_1', None), row.get('lag_7', None),
                            row.get('lag_24', None), row.get('lag_168', None),
                            row.get('rolling_avg_7', None), row.get('rolling_avg_24', None),
                            row.get('rolling_avg_168', None),
                            row.get('rolling_std_7', None), row.get('rolling_std_24', None),
                            row.get('hour', None), row.get('day_of_week', None),
                            row.get('day_of_month', None), row.get('month', None),
                            is_weekend_val, is_holiday_val,
                            row.get('temperature', None), row.get('humidity', None),
                            row.get('pressure', None), row.get('wind_speed', None),
                            row.get('temp_rolling_avg_24', None),
                            row.get('humidity_rolling_avg_24', None)
                        )
                        
                        cursor.execute(query, values)
                        inserted += 1
                        
                    except Exception as e:
                        errors += 1
                        if errors <= 3:  # Only log first few errors
                            logger.error(f"Failed to insert feature data: {str(e)}")
                        conn.rollback()
                        # Start a new transaction for the next insert
                        continue
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {str(e)}")
            finally:
                cursor.close()
        
        logger.info(f"Successfully saved {inserted} feature records to database")
        return inserted

