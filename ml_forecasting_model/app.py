import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        self.label_encoders = {}
        
    def load_data_from_folder(self, folder_path="data"):
        """Load all CSV files from the specified data folder"""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found!")
            print("Please create a 'data' folder and place your CSV files there.")
            return None
        
        # Find all CSV files in the data folder
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
                
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Clean column names (remove any extra spaces)
                df.columns = df.columns.str.strip()
                
                print(f"Columns found: {list(df.columns)}")
                
                # Check if required columns exist
                required_columns = ['City', 'Date']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"Warning: Missing required columns {missing_columns} in {os.path.basename(file_path)}")
                    continue
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
                
                # Remove rows with invalid dates
                df = df.dropna(subset=['Date'])
                
                # Handle missing values in pollutant columns
                pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                                   'Benzene', 'Toluene', 'Xylene', 'AQI']
                
                # Convert pollutant columns to numeric, replacing non-numeric with NaN
                for col in pollutant_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add temporal features
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                df['DayOfYear'] = df['Date'].dt.dayofyear
                df['Week'] = df['Date'].dt.isocalendar().week
                
                # Add seasonal features
                df['Season'] = df['Month'].apply(self.get_season)
                df['Is_Winter'] = (df['Month'].isin([12, 1, 2])).astype(int)
                df['Is_Monsoon'] = (df['Month'].isin([6, 7, 8, 9])).astype(int)
                df['Is_Summer'] = (df['Month'].isin([3, 4, 5])).astype(int)
                df['Is_PostMonsoon'] = (df['Month'].isin([10, 11])).astype(int)
                
                # Add weekend indicator
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
        
        # Sort by date
        combined_data = combined_data.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Add lagged features for time series patterns
        combined_data = self.add_lagged_features(combined_data)
        
        print(f"\nTotal records loaded: {len(combined_data)}")
        print(f"Cities: {combined_data['City'].nunique()}")
        print(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        
        # Print data summary
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
    
    def add_lagged_features(self, df):
        """Add lagged features for time series modeling"""
        df_with_lags = df.copy()
        
        # Sort by city and date
        df_with_lags = df_with_lags.sort_values(['City', 'Date'])
        
        # Define pollutants for which to create lags
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Group by city and create lags
                df_with_lags[f'{pollutant}_lag1'] = df_with_lags.groupby('City')[pollutant].shift(1)
                df_with_lags[f'{pollutant}_lag3'] = df_with_lags.groupby('City')[pollutant].shift(3)
                df_with_lags[f'{pollutant}_lag7'] = df_with_lags.groupby('City')[pollutant].shift(7)
                
                # Rolling averages
                df_with_lags[f'{pollutant}_rolling_3'] = df_with_lags.groupby('City')[pollutant].rolling(window=3).mean().reset_index(0, drop=True)
                df_with_lags[f'{pollutant}_rolling_7'] = df_with_lags.groupby('City')[pollutant].rolling(window=7).mean().reset_index(0, drop=True)
        
        return df_with_lags
    
    def simulate_meteorological_features(self, df):
        """
        Simulate meteorological features based on date and location
        In production, these would come from weather APIs or satellite data
        """
        np.random.seed(42)
        n_samples = len(df)
        
        # Temperature simulation based on month and city
        base_temp = 25 + 10 * np.sin(2 * np.pi * df['Month'] / 12)
        city_temp_offset = df['City'].map({
            'Ahmedabad': 5,    # Generally hotter
            'Delhi': 0,        # Reference
            'Mumbai': 3,       # Coastal, moderate
            'Kolkata': 2,      # Humid
            'Chennai': 8,      # Very hot
            'Bangalore': -5,   # Cooler
            'Hyderabad': 2,
            'Pune': -2
        }).fillna(0)
        
        df['Temperature'] = base_temp + city_temp_offset + np.random.normal(0, 3, n_samples)
        
        # Humidity simulation
        base_humidity = 60 + 20 * np.sin(2 * np.pi * (df['Month'] - 3) / 12)
        df['Humidity'] = np.clip(base_humidity + np.random.normal(0, 10, n_samples), 20, 95)
        
        # Wind speed simulation
        df['Wind_Speed'] = np.maximum(2, 8 + np.random.normal(0, 3, n_samples))
        
        # Rainfall simulation (higher during monsoon)
        df['Rainfall'] = np.where(df['Is_Monsoon'], 
                                 np.random.exponential(50, n_samples),
                                 np.random.exponential(5, n_samples))
        
        # Pressure simulation
        df['Pressure'] = 1013 + np.random.normal(0, 10, n_samples)
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix for ML models"""
        df_processed = df.copy()
        
        # Encode categorical variables
        if 'City' in df.columns:
            le_city = LabelEncoder()
            df_processed['City_encoded'] = le_city.fit_transform(df_processed['City'])
            self.label_encoders['City'] = le_city
        
        # Encode season
        season_mapping = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Post-Monsoon': 3}
        df_processed['Season_encoded'] = df_processed['Season'].map(season_mapping)
        
        # Add simulated meteorological features
        df_processed = self.simulate_meteorological_features(df_processed)
        
        # Define feature columns
        feature_columns = [
            'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
            'Season_encoded', 'Is_Winter', 'Is_Monsoon', 'Is_Summer', 'Is_PostMonsoon',
            'Is_Weekend', 'City_encoded',
            'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall', 'Pressure'
        ]
        
        # Add available pollutant features (excluding the target)
        pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                             'Benzene', 'Toluene', 'Xylene']
        
        # Add lagged features
        for pollutant in pollutant_features + ['AQI']:
            lag_features = [f'{pollutant}_lag1', f'{pollutant}_lag3', f'{pollutant}_lag7',
                           f'{pollutant}_rolling_3', f'{pollutant}_rolling_7']
            feature_columns.extend([col for col in lag_features if col in df_processed.columns])
        
        # Add available pollutant features to feature list
        available_pollutants = [col for col in pollutant_features if col in df_processed.columns]
        
        return df_processed, feature_columns, available_pollutants
    
    def train_models(self, df, target_columns=None):
        """Train ML models for different pollutants"""
        if target_columns is None:
            # Default targets - use available columns
            possible_targets = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
            target_columns = [col for col in possible_targets if col in df.columns]
        
        if not target_columns:
            print("No valid target columns found!")
            return None, None
        
        # Prepare features
        df_processed, feature_columns, available_pollutants = self.prepare_features(df)
        
        results = {}
        
        for target in target_columns:
            if target not in df_processed.columns:
                print(f"Target '{target}' not found in data. Skipping...")
                continue
                
            print(f"\nTraining model for {target}...")
            
            # For each target, exclude it and its lagged versions from features
            target_features = [col for col in feature_columns 
                             if not col.startswith(target) and col in df_processed.columns]
            
            # If predicting AQI, include other pollutants as features
            if target == 'AQI':
                target_features.extend([col for col in available_pollutants if col != target])
            # If predicting a pollutant, exclude AQI to avoid data leakage
            elif target in available_pollutants:
                other_pollutants = [col for col in available_pollutants if col != target]
                target_features.extend(other_pollutants)
            
            # Remove duplicates and ensure all features exist in dataframe
            target_features = list(set([col for col in target_features if col in df_processed.columns]))
            
            print(f"Using {len(target_features)} features for {target}")
            
            # Prepare data
            X = df_processed[target_features].copy()
            y = df_processed[target].copy()
            
            # Remove rows with missing target values
            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:  # Need minimum samples
                print(f"Not enough valid samples for {target} ({len(X)} samples). Skipping...")
                continue
            
            print(f"Training with {len(X)} valid samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target] = scaler
            
            # Try multiple algorithms
            models_to_try = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('-inf')
            
            for model_name, model in models_to_try.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    avg_cv_score = cv_scores.mean()
                    
                    # Fit model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    print(f"{model_name} - CV Score: {avg_cv_score:.3f}, R²: {r2:.3f}, RMSE: {rmse:.3f}")
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        
                        results[target] = {
                            'model': model,
                            'model_name': model_name,
                            'r2': r2,
                            'rmse': rmse,
                            'mae': mae,
                            'cv_score': avg_cv_score,
                            'predictions': y_pred,
                            'actual': y_test.values,
                            'features': target_features
                        }
                
                except Exception as e:
                    print(f"Error training {model_name} for {target}: {e}")
                    continue
            
            if best_model is not None:
                self.models[target] = best_model
                
                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': target_features,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[target] = importance_df
                    
                    print(f"\nTop 5 features for {target}:")
                    print(importance_df.head())
        
        return results, df_processed
    
    def predict_for_city(self, city_name, date, pollutant_values=None):
        """Make predictions for a specific city and date"""
        predictions = {}
        
        # Create input data
        input_data = {
            'City': city_name,
            'Date': pd.to_datetime(date),
            'Year': pd.to_datetime(date).year,
            'Month': pd.to_datetime(date).month,
            'Day': pd.to_datetime(date).day,
            'DayOfWeek': pd.to_datetime(date).dayofweek,
            'DayOfYear': pd.to_datetime(date).dayofyear,
            'Week': pd.to_datetime(date).isocalendar().week,
            'Season': self.get_season(pd.to_datetime(date).month),
            'Is_Winter': int(pd.to_datetime(date).month in [12, 1, 2]),
            'Is_Monsoon': int(pd.to_datetime(date).month in [6, 7, 8, 9]),
            'Is_Summer': int(pd.to_datetime(date).month in [3, 4, 5]),
            'Is_PostMonsoon': int(pd.to_datetime(date).month in [10, 11]),
            'Is_Weekend': int(pd.to_datetime(date).dayofweek in [5, 6])
        }
        
        # Add pollutant values if provided
        if pollutant_values:
            input_data.update(pollutant_values)
        
        input_df = pd.DataFrame([input_data])
        processed_df, _, _ = self.prepare_features(input_df)
        
        for target, model in self.models.items():
            try:
                features_used = self.feature_importance[target]['feature'].tolist()
                X_input = processed_df[features_used].fillna(0)  # Fill NaN with 0 for missing lags
                X_scaled = self.scalers[target].transform(X_input)
                pred = model.predict(X_scaled)[0]
                predictions[target] = max(0, pred)  # Ensure non-negative predictions
            except Exception as e:
                print(f"Error predicting {target}: {e}")
                predictions[target] = None
        
        return predictions
    
    def create_city_risk_map(self, df, date=None):
        """Create interactive map showing air quality across cities"""
        if date is None:
            # Use latest available date
            date = df['Date'].max()
        
        # Filter data for the specified date (or closest available)
        date_data = df[df['Date'] == date]
        if len(date_data) == 0:
            # If no data for exact date, use latest data for each city
            date_data = df.loc[df.groupby('City')['Date'].idxmax()]
        
        # City coordinates (approximate)
        city_coords = {
            'Ahmedabad': [23.0225, 72.5714],
            'Delhi': [28.6139, 77.2090],
            'Mumbai': [19.0760, 72.8777],
            'Kolkata': [22.5726, 88.3639],
            'Chennai': [13.0827, 80.2707],
            'Bangalore': [12.9716, 77.5946],
            'Hyderabad': [17.3850, 78.4867],
            'Pune': [18.5204, 73.8567]
        }
        
        # Create base map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Color mapping for AQI levels
        def get_aqi_color(aqi):
            if pd.isna(aqi):
                return 'gray'
            elif aqi <= 50:
                return 'green'
            elif aqi <= 100:
                return 'yellow'
            elif aqi <= 150:
                return 'orange'
            elif aqi <= 200:
                return 'red'
            elif aqi <= 300:
                return 'purple'
            else:
                return 'maroon'
        
        # Add city markers
        for _, row in date_data.iterrows():
            city = row['City']
            if city in city_coords:
                coords = city_coords[city]
                aqi = row.get('AQI', np.nan)
                color = get_aqi_color(aqi)
                
                # Create popup text
                popup_text = f"<b>{city}</b><br>Date: {row['Date'].strftime('%Y-%m-%d')}"
                if not pd.isna(aqi):
                    popup_text += f"<br>AQI: {aqi:.0f}"
                if 'PM2.5' in row and not pd.isna(row['PM2.5']):
                    popup_text += f"<br>PM2.5: {row['PM2.5']:.1f} μg/m³"
                if 'PM10' in row and not pd.isna(row['PM10']):
                    popup_text += f"<br>PM10: {row['PM10']:.1f} μg/m³"
                if 'NO2' in row and not pd.isna(row['NO2']):
                    popup_text += f"<br>NO2: {row['NO2']:.1f} μg/m³"
                
                folium.CircleMarker(
                    location=coords,
                    radius=15,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 180px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>AQI Levels</h4>
        <p><i style="color:green">●</i> Good (0-50)</p>
        <p><i style="color:yellow">●</i> Satisfactory (51-100)</p>
        <p><i style="color:orange">●</i> Moderate (101-150)</p>
        <p><i style="color:red">●</i> Poor (151-200)</p>
        <p><i style="color:purple">●</i> Very Poor (201-300)</p>
        <p><i style="color:maroon">●</i> Severe (300+)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def forecast_pollution(self, city, days_ahead=7):
        """Generate pollution forecasts for a specific city"""
        if not self.models:
            print("No trained models available for forecasting!")
            return None
        
        current_date = pd.Timestamp.now()
        forecast_dates = pd.date_range(current_date, periods=days_ahead, freq='D')
        
        forecasts = []
        for date in forecast_dates:
            # Make predictions (simplified - in practice would use more sophisticated time series methods)
            predictions = self.predict_for_city(city, date)
            
            forecast_row = {
                'Date': date,
                'City': city
            }
            forecast_row.update(predictions)
            
            # Add risk assessment
            aqi = predictions.get('AQI', 100)
            if pd.isna(aqi) or aqi is None:
                risk = 'Unknown'
            elif aqi <= 100:
                risk = 'Low'
            elif aqi <= 150:
                risk = 'Moderate'
            elif aqi <= 200:
                risk = 'High'
            else:
                risk = 'Very High'
            
            forecast_row['Risk_Level'] = risk
            forecasts.append(forecast_row)
        
        return pd.DataFrame(forecasts)
    
    def plot_results(self, results, df):
        """Create visualization plots for model results"""
        n_targets = len(results)
        if n_targets == 0:
            print("No results to plot!")
            return None
        
        # Create subplots
        cols = min(2, n_targets)
        rows = (n_targets + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        for target, result in results.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Actual vs Predicted scatter plot
            ax.scatter(result['actual'], result['predictions'], alpha=0.6)
            
            # Perfect prediction line
            min_val = min(result['actual'].min(), result['predictions'].min())
            max_val = max(result['actual'].max(), result['predictions'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel(f'Actual {target}')
            ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'{target} Prediction\n(R² = {result["r2"]:.3f}, RMSE = {result["rmse"]:.1f})')
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def analyze_data_quality(self, df):
        """Analyze and report data quality issues"""
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Missing value analysis
        print("\nMissing Values:")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data / len(df) * 100).round(2)
        
        for col in missing_data.index:
            if missing_data[col] > 0:
                print(f"  {col}: {missing_data[col]} ({missing_percent[col]}%)")
        
        # Date range by city
        print(f"\nDate Range by City:")
        for city in df['City'].unique():
            city_data = df[df['City'] == city]
            print(f"  {city}: {city_data['Date'].min()} to {city_data['Date'].max()} ({len(city_data)} records)")
        
        # Outlier analysis for key pollutants
        print(f"\nOutlier Analysis (values > 99th percentile):")
        pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        for pollutant in pollutants:
            if pollutant in df.columns:
                q99 = df[pollutant].quantile(0.99)
                outliers = (df[pollutant] > q99).sum()
                if outliers > 0:
                    print(f"  {pollutant}: {outliers} values > {q99:.1f}")

def main():
    """Main function to demonstrate the air quality prediction system"""
    
    # Initialize the predictor
    predictor = AirQualityPredictor()
    
    # Load data from CSV files in the 'data' folder
    print("Loading air quality data from 'data' folder...")
    df = predictor.load_data_from_folder("data")
    
    if df is None:
        print("\n" + "="*60)
        print("CREATING SAMPLE DATA FOR DEMONSTRATION")
        print("="*60)
        print("Since no data folder was found, creating sample data...")
        print("To use your own data:")
        print("1. Create a folder named 'data' in your working directory")
        print("2. Place your CSV files with columns: City, Date, PM2.5, PM10, NO2, etc.")
        print("3. Run the script again")
        print("\n" + "="*60)
        
        # Create sample data based on your format
        cities = ['Ahmedabad', 'Delhi', 'Mumbai', 'Chennai']
        sample_data = []
        
        # Create sample data for 2 years
        date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        
        for date in date_range:
            for city in cities:
                # Simulate pollutant values based on seasonal patterns
                month = date.month
                base_aqi = 100 + np.random.normal(0, 30)
                
                # Seasonal adjustments
                if month in [11, 12, 1, 2]:  # Winter - higher pollution
                    base_aqi += 50
                elif month in [6, 7, 8, 9]:  # Monsoon - lower pollution
                    base_aqi -= 30
                
                # City-specific adjustments
                city_factor = {'Ahmedabad': 1.1, 'Delhi': 1.5, 'Mumbai': 1.0, 'Chennai': 0.9}
                base_aqi *= city_factor.get(city, 1.0)
                
                base_aqi = max(50, min(300, base_aqi))
                
                # Generate related pollutants
                pm25 = base_aqi * 0.4 + np.random.normal(0, 5)
                pm10 = pm25 * 1.8 + np.random.normal(0, 8)
                no2 = base_aqi * 0.3 + np.random.normal(0, 10)
                co = np.random.uniform(0.5, 2.0)
                so2 = np.random.uniform(5, 25)
                o3 = np.random.uniform(20, 100)
                
                sample_data.append({
                    'City': city,
                    'Date': date,
                    'PM2.5': max(0, pm25),
                    'PM10': max(0, pm10),
                    'NO': np.random.uniform(10, 50),
                    'NO2': max(0, no2),
                    'NOx': max(0, no2 * 1.5),
                    'NH3': np.random.uniform(20, 80),
                    'CO': max(0, co),
                    'SO2': max(0, so2),
                    'O3': max(0, o3),
                    'Benzene': np.random.uniform(0, 5),
                    'Toluene': np.random.uniform(0, 10),
                    'Xylene': np.random.uniform(0, 8),
                    'AQI': base_aqi
                })
        
        df = pd.DataFrame(sample_data)
        print(f"Sample dataset created with {len(df)} records for demonstration.")
    
    # Analyze data quality
    predictor.analyze_data_quality(df)
    
    # Train models
    print("\nTraining machine learning models...")
    results, processed_df = predictor.train_models(df)
    
    if not results:
        print("No models were successfully trained!")
        return None, None, None, None
    
    # Generate forecasts for each city
    print("\nGenerating pollution forecasts...")
    all_forecasts = {}
    cities = df['City'].unique()
    
    for city in cities[:3]:  # Limit to first 3 cities for demo
        try:
            forecast = predictor.forecast_pollution(city, days_ahead=7)
            if forecast is not None:
                all_forecasts[city] = forecast
                print(f"\nForecast for {city}:")
                print(forecast[['Date', 'AQI', 'PM2.5', 'Risk_Level']].head())
        except Exception as e:
            print(f"Error generating forecast for {city}: {e}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        fig = predictor.plot_results(results, df)
        if fig:
            plt.show()
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Create risk map
    print("\nCreating risk assessment map...")
    try:
        risk_map = predictor.create_city_risk_map(df)
        # Save map to HTML file
        map_filename = "air_quality_risk_map.html"
        risk_map.save(map_filename)
        print(f"Risk map saved as '{map_filename}'")
    except Exception as e:
        print(f"Error creating risk map: {e}")
        risk_map = None
    
    # Model performance summary
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    for target, result in results.items():
        print(f"{target}:")
        print(f"  Model: {result['model_name']}")
        print(f"  R² Score: {result['r2']:.3f}")
        print(f"  RMSE: {result['rmse']:.2f}")
        print(f"  MAE: {result['mae']:.2f}")
        print(f"  CV Score: {result['cv_score']:.3f}")
        print()
    
    # Feature importance summary
    print("TOP FEATURES BY TARGET:")
    print("-" * 30)
    for target in results.keys():
        if target in predictor.feature_importance:
            top_features = predictor.feature_importance[target].head(3)
            print(f"{target}: {', '.join(top_features['feature'].tolist())}")
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    if len(cities) > 0:
        example_city = cities[0]
        example_date = '2024-12-01'
        
        print(f"Predicting air quality for {example_city} on {example_date}:")
        
        try:
            predictions = predictor.predict_for_city(example_city, example_date)
            for pollutant, value in predictions.items():
                if value is not None:
                    print(f"  {pollutant}: {value:.2f}")
        except Exception as e:
            print(f"Error making example prediction: {e}")
    
    print("\n" + "="*60)
    print("DATA LOADING INSTRUCTIONS")
    print("="*60)
    print("To use your own CSV files:")
    print("1. Create a 'data' folder in your current directory")
    print("2. Place CSV files with the following columns:")
    print("   Required: City, Date")
    print("   Optional: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,")
    print("            Benzene, Toluene, Xylene, AQI, AQI_Bucket")
    print("3. Date format should be readable by pandas (e.g., 1/1/2015)")
    print("4. The script will automatically detect and load all CSV files")
    print("="*60)
    
    return predictor, df, results, all_forecasts

# Additional utility functions for enhanced analysis
def analyze_seasonal_patterns(df):
    """Analyze seasonal pollution patterns"""
    if 'AQI' not in df.columns or 'Season' not in df.columns:
        return None
    
    seasonal_stats = df.groupby('Season')['AQI'].agg(['mean', 'std', 'min', 'max']).round(2)
    
    print("\n" + "="*40)
    print("SEASONAL POLLUTION PATTERNS")
    print("="*40)
    print(seasonal_stats)
    
    # Plot seasonal patterns
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Boxplot by season
    plt.subplot(2, 2, 1)
    df.boxplot(column='AQI', by='Season', ax=plt.gca())
    plt.title('AQI Distribution by Season')
    plt.suptitle('')
    
    # Subplot 2: Monthly averages
    if 'Month' in df.columns:
        plt.subplot(2, 2, 2)
        monthly_avg = df.groupby('Month')['AQI'].mean()
        monthly_avg.plot(kind='bar')
        plt.title('Average AQI by Month')
        plt.xlabel('Month')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)
    
    # Subplot 3: City comparison
    if 'City' in df.columns:
        plt.subplot(2, 2, 3)
        city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        city_avg.plot(kind='bar')
        plt.title('Average AQI by City')
        plt.xlabel('City')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)
    
    # Subplot 4: Time series if enough data
    plt.subplot(2, 2, 4)
    if len(df) > 365:  # If we have more than a year of data
        # Sample data for plotting (to avoid overcrowding)
        sample_df = df.sample(min(1000, len(df))).sort_values('Date')
        plt.plot(sample_df['Date'], sample_df['AQI'], alpha=0.6)
        plt.title('AQI Time Series (Sample)')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return seasonal_stats

def create_pollutant_correlation_analysis(df):
    """Analyze correlations between different pollutants"""
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
    available_pollutants = [col for col in pollutants if col in df.columns]
    
    if len(available_pollutants) < 2:
        print("Not enough pollutants available for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_pollutants].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Pollutant Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*40)
    print("POLLUTANT CORRELATIONS")
    print("="*40)
    
    # Find strongest correlations
    correlations = []
    for i in range(len(available_pollutants)):
        for j in range(i+1, len(available_pollutants)):
            corr_val = corr_matrix.iloc[i, j]
            correlations.append({
                'Pollutant1': available_pollutants[i],
                'Pollutant2': available_pollutants[j],
                'Correlation': corr_val
            })
    
    # Sort by absolute correlation value
    correlations = sorted(correlations, key=lambda x: abs(x['Correlation']), reverse=True)
    
    print("Strongest correlations:")
    for corr in correlations[:5]:
        print(f"  {corr['Pollutant1']} - {corr['Pollutant2']}: {corr['Correlation']:.3f}")
    
    return corr_matrix

def generate_air_quality_report(predictor, df, results):
    """Generate a comprehensive air quality analysis report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE AIR QUALITY ANALYSIS REPORT")
    print("="*60)
    
    # Dataset overview
    print(f"\nDATASET OVERVIEW:")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Cities covered: {df['City'].nunique()}")
    print(f"Cities: {', '.join(df['City'].unique())}")
    
    # Data completeness
    print(f"\nDATA COMPLETENESS:")
    key_columns = ['AQI', 'PM2.5', 'PM10', 'NO2']
    for col in key_columns:
        if col in df.columns:
            completeness = (1 - df[col].isnull().mean()) * 100
            print(f"  {col}: {completeness:.1f}%")
    
    # Pollution level distribution
    if 'AQI' in df.columns:
        print(f"\nAQI DISTRIBUTION:")
        aqi_ranges = [
            (0, 50, "Good"),
            (51, 100, "Satisfactory"),
            (101, 150, "Moderate"),
            (151, 200, "Poor"),
            (201, 300, "Very Poor"),
            (301, 500, "Severe")
        ]
        
        for min_val, max_val, category in aqi_ranges:
            count = ((df['AQI'] >= min_val) & (df['AQI'] <= max_val)).sum()
            percentage = (count / len(df[df['AQI'].notna()])) * 100
            if count > 0:
                print(f"  {category} ({min_val}-{max_val}): {count:,} records ({percentage:.1f}%)")
    
    # City rankings
    if 'City' in df.columns and 'AQI' in df.columns:
        print(f"\nCITY RANKINGS (by average AQI):")
        city_rankings = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        for i, (city, avg_aqi) in enumerate(city_rankings.items(), 1):
            print(f"  {i}. {city}: {avg_aqi:.1f}")
    
    # Model performance
    if results:
        print(f"\nMODEL PERFORMANCE:")
        for target, result in results.items():
            accuracy_rating = "Excellent" if result['r2'] > 0.8 else "Good" if result['r2'] > 0.6 else "Fair" if result['r2'] > 0.4 else "Poor"
            print(f"  {target}: R² = {result['r2']:.3f} ({accuracy_rating})")
    
    # Recommendations
    print(f"\nRECOMMendations:")
    print("  1. Focus monitoring on cities with highest pollution levels")
    print("  2. Implement seasonal pollution control measures during high-risk periods")
    print("  3. Develop early warning systems based on model predictions")
    if results and any(result['r2'] < 0.6 for result in results.values()):
        print("  4. Collect additional meteorological data to improve model accuracy")
    print("  5. Regular calibration of monitoring equipment recommended")
    
    return True

if __name__ == "__main__":
    # Run the main demonstration
    try:
        predictor, data, results, forecasts = main()
        
        if data is not None and len(data) > 0:
            # Additional analyses
            print("\nPerforming additional analyses...")
            
            # Seasonal analysis
            analyze_seasonal_patterns(data)
            
            # Correlation analysis
            create_pollutant_correlation_analysis(data)
            
            # Comprehensive report
            generate_air_quality_report(predictor, data, results)
        
        print("\n" + "="*50)
        print("AIR QUALITY PREDICTION SYSTEM READY")
        print("="*50)
        print("Features:")
        print("✓ Multi-city air quality analysis")
        print("✓ ML models for AQI, PM2.5, PM10, and other pollutants")
        print("✓ Seasonal pattern analysis")
        print("✓ Pollution forecasting")
        print("✓ Interactive risk mapping")
        print("✓ Comprehensive reporting")
        print("✓ Data quality assessment")
        print("\nNext steps:")
        print("1. Add real meteorological data integration")
        print("2. Implement real-time data feeds")
        print("3. Develop mobile app interface")
        print("4. Add health impact assessments")
        print("5. Deploy as web service")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()