import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# Google Earth Engine and Satellite Data Integration
try:
    import ee
    import geemap
except ImportError:
    print("Google Earth Engine not available. Install with: pip install earthengine-api geemap")


class AirQualityPredictor:
    """
    Air Quality Prediction Model using real pollutant measurements
    Handles multiple cities and pollutants (PM2.5, PM10, NO2, etc.)
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        # Use a simple dict for stable city encoding across train/predict
        self.city_mapping = None

    def load_data_from_folder(self, folder_path="data"):
        """Load all CSV files from the specified data folder"""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found!")
            print("Please create a 'data' folder and place your CSV files there.")
            return None

        csv_pattern = os.path.join(folder_path, "*.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"No CSV files found in '{folder_path}' folder!")
            return None

        print(f"Found {len(csv_files)} CSV files in '{folder_path}' folder:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")

        return self.process_air_quality_data(csv_files)

 def process_air_quality_data(self, csv_files):
        """Process air quality CSV files into training dataset"""
        all_data = []

        for file_path in csv_files:
            try:
                print(f"Processing: {os.path.basename(file_path)}")

                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                print(f"Columns found: {list(df.columns)}")

                required_columns = ['City', 'Date']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    print(f"Warning: Missing required columns {missing_columns} in {os.path.basename(file_path)}")
                    continue

                # Date handling
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])

                # Pollutant coercion
                pollutant_columns = [
                    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                    'Benzene', 'Toluene', 'Xylene', 'AQI'
                ]
                for col in pollutant_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Temporal
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                df['DayOfYear'] = df['Date'].dt.dayofyear
                # cast to int to avoid pandas UInt types leaking into models
                df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

                # Seasonal and weekend flags
                df['Season'] = df['Month'].apply(self.get_season)
                df['Is_Winter'] = (df['Month'].isin([12, 1, 2])).astype(int)
                df['Is_Monsoon'] = (df['Month'].isin([6, 7, 8, 9])).astype(int)
                df['Is_Summer'] = (df['Month'].isin([3, 4, 5])).astype(int)
                df['Is_PostMonsoon'] = (df['Month'].isin([10, 11])).astype(int)
                df['Is_Weekend'] = (df['DayOfWeek'].isin([5, 6])).astype(int)

                all_data.append(df)
                print(f"Successfully processed {len(df)} records from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not all_data:
            print("No data was successfully processed!")
            return None

        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['City', 'Date']).reset_index(drop=True)

        # Lags/rolls
        combined_data = self.add_lagged_features(combined_data)

        print(f"\nTotal records loaded: {len(combined_data)}")
        print(f"Cities: {combined_data['City'].nunique()}")
        print(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")

        if 'AQI' in combined_data.columns:
            aqi_stats = combined_data['AQI'].describe()
            print(f"AQI range: {aqi_stats['min']:.1f} to {aqi_stats['max']:.1f}")
            print(f"Average AQI: {aqi_stats['mean']:.1f}")

        return combined_data

    def get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
