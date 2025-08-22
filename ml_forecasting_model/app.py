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
