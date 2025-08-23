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

    def add_lagged_features(self, df):
        """Add lagged and rolling features for time series modeling"""
        df_with_lags = df.sort_values(['City', 'Date']).copy()

        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
        for pollutant in pollutants:
            if pollutant in df.columns:
                grp = df_with_lags.groupby('City')[pollutant]
                df_with_lags[f'{pollutant}_lag1'] = grp.shift(1)
                df_with_lags[f'{pollutant}_lag3'] = grp.shift(3)
                df_with_lags[f'{pollutant}_lag7'] = grp.shift(7)
                df_with_lags[f'{pollutant}_rolling_3'] = grp.rolling(window=3).mean().reset_index(0, drop=True)
                df_with_lags[f'{pollutant}_rolling_7'] = grp.rolling(window=7).mean().reset_index(0, drop=True)

        return df_with_lags

    def simulate_meteorological_features(self, df):
        """
        Simulate meteorological features based on date and location
        In production, these would come from weather APIs or satellite data
        """
        np.random.seed(42)
        n_samples = len(df)

        base_temp = 25 + 10 * np.sin(2 * np.pi * df['Month'] / 12.0)
        city_temp_offset = df['City'].map({
            'Ahmedabad': 5, 'Delhi': 0, 'Mumbai': 3, 'Kolkata': 2, 'Chennai': 8,
            'Bangalore': -5, 'Hyderabad': 2, 'Pune': -2
        }).fillna(0)

        df['Temperature'] = base_temp + city_temp_offset + np.random.normal(0, 3, n_samples)

        base_humidity = 60 + 20 * np.sin(2 * np.pi * (df['Month'] - 3) / 12.0)
        df['Humidity'] = np.clip(base_humidity + np.random.normal(0, 10, n_samples), 20, 95)

        df['Wind_Speed'] = np.maximum(2, 8 + np.random.normal(0, 3, n_samples))

        df['Rainfall'] = np.where(
            df['Is_Monsoon'] == 1,
            np.random.exponential(50, n_samples),
            np.random.exponential(5, n_samples)
        )

        df['Pressure'] = 1013 + np.random.normal(0, 10, n_samples)

        return df

    def _encode_city(self, series: pd.Series, training: bool) -> pd.Series:
        """Stable integer encoding for City with unknowns handled."""
        if training:
            unique_cities = pd.Index(series.dropna().unique())
            self.city_mapping = {city: i for i, city in enumerate(sorted(unique_cities))}
        if self.city_mapping is None:
            # No mapping yet (edge case: predict before train); create temp
            tmp_mapping = {city: i for i, city in enumerate(sorted(series.dropna().unique()))}
            return series.map(tmp_mapping).fillna(-1).astype(int)
        return series.map(self.city_mapping).fillna(-1).astype(int)

    def prepare_features(self, df, *, training: bool):
        """Prepare feature matrix for ML models"""
        df_processed = df.copy()

        # City encoding (stable across train/predict)
        if 'City' in df_processed.columns:
            df_processed['City_encoded'] = self._encode_city(df_processed['City'], training=training)
        else:
            df_processed['City_encoded'] = -1

        # Encode season (robust to missing)
        season_mapping = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Post-Monsoon': 3}
        if 'Season' not in df_processed.columns and 'Month' in df_processed.columns:
            df_processed['Season'] = df_processed['Month'].apply(self.get_season)
        df_processed['Season_encoded'] = df_processed['Season'].map(season_mapping).fillna(-1).astype(int)

        # Simulated meteorology
        # Ensure temporal fields exist for simulations if this is for a single-row predict
        for col, func in [('Year', lambda d: d['Date'].dt.year),
                          ('Month', lambda d: d['Date'].dt.month),
                          ('Day', lambda d: d['Date'].dt.day),
                          ('DayOfWeek', lambda d: d['Date'].dt.dayofweek),
                          ('DayOfYear', lambda d: d['Date'].dt.dayofyear),
                          ('Week', lambda d: d['Date'].dt.isocalendar().week.astype(int))]:
            if col not in df_processed.columns and 'Date' in df_processed.columns:
                df_processed[col] = func(df_processed)

        df_processed = self.simulate_meteorological_features(df_processed)

        # Base feature columns
        feature_columns = [
            'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
            'Season_encoded', 'Is_Winter', 'Is_Monsoon', 'Is_Summer', 'Is_PostMonsoon',
            'Is_Weekend', 'City_encoded',
            'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall', 'Pressure'
        ]

        pollutant_features = [
            'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
            'Benzene', 'Toluene', 'Xylene'
        ]

        # Add lag/rolling features where present
        for pollutant in pollutant_features + ['AQI']:
            for suffix in ['lag1', 'lag3', 'lag7', 'rolling_3', 'rolling_7']:
                col = f'{pollutant}_{suffix}'
                if col in df_processed.columns:
                    feature_columns.append(col)

        available_pollutants = [col for col in pollutant_features if col in df_processed.columns]

        return df_processed, feature_columns, available_pollutants

    def train_models(self, df, target_columns=None):
        """Train ML models for different pollutants"""
        if target_columns is None:
            possible_targets = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
            target_columns = [col for col in possible_targets if col in df.columns]

        if not target_columns:
            print("No valid target columns found!")
            return None, None

        # Prepare features (training=True to fit city mapping)
        df_processed, feature_columns, available_pollutants = self.prepare_features(df, training=True)

        results = {}

        for target in target_columns:
            if target not in df_processed.columns:
                print(f"Target '{target}' not found in data. Skipping...")
                continue

            print(f"\nTraining model for {target}...")

            # Exclude target's self-lags/rolls to avoid leakage
            target_features = [c for c in feature_columns if not c.startswith(target) and c in df_processed.columns]

            if target == 'AQI':
                target_features.extend([c for c in available_pollutants if c != target])
            elif target in available_pollutants:
                other_pollutants = [c for c in available_pollutants if c != target]
                target_features.extend(other_pollutants)

            target_features = sorted(set([c for c in target_features if c in df_processed.columns]))

            print(f"Using {len(target_features)} features for {target}")

            X = df_processed[target_features].copy()
            y = df_processed[target].copy()

            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            X, y = X[valid_mask], y[valid_mask]

            if len(X) < 50:
                print(f"Not enough valid samples for {target} ({len(X)} samples). Skipping...")
                continue

            print(f"Training with {len(X)} valid samples")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler

            models_to_try = {
                'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
            }

            best_model = None
            best_score = float('-inf')

            for model_name, model in models_to_try.items():
                try:
                    # Make sure we have at least 3 folds if data is small (but >=50 so OK)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    avg_cv_score = cv_scores.mean()

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    print(f"{model_name} - CV: {avg_cv_score:.3f}, R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

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

        # Build a single-row input frame
        date_ts = pd.to_datetime(date)
        input_data = {
            'City': city_name,
            'Date': date_ts,
            'Year': date_ts.year,
            'Month': date_ts.month,
            'Day': date_ts.day,
            'DayOfWeek': date_ts.dayofweek,
            'DayOfYear': date_ts.dayofyear,
            'Week': int(pd.Timestamp(date_ts).isocalendar().week),
            'Season': self.get_season(date_ts.month),
            'Is_Winter': int(date_ts.month in [12, 1, 2]),
            'Is_Monsoon': int(date_ts.month in [6, 7, 8, 9]),
            'Is_Summer': int(date_ts.month in [3, 4, 5]),
            'Is_PostMonsoon': int(date_ts.month in [10, 11]),
            'Is_Weekend': int(date_ts.dayofweek in [5, 6]),
        }

        if pollutant_values:
            input_data.update(pollutant_values)

        input_df = pd.DataFrame([input_data])

        # IMPORTANT: do NOT fit encoders here; use training=False to keep mapping
        processed_df, _, _ = self.prepare_features(input_df, training=False)

        for target, model in self.models.items():
            try:
                # Use the same features as training used for this target
                if target in self.feature_importance:
                    features_used = [f for f in self.feature_importance[target]['feature'].tolist()
                                     if f in processed_df.columns]
                else:
                    # Fallback: union of model features if importance not available
                    features_used = [c for c in processed_df.columns if c in self.scalers[target].feature_names_in_]

                if not features_used:
                    predictions[target] = None
                    continue

                X_input = processed_df[features_used].fillna(0)
                X_scaled = self.scalers[target].transform(X_input)
                pred = float(model.predict(X_scaled)[0])
                predictions[target] = max(0.0, pred)
            except Exception as e:
                print(f"Error predicting {target}: {e}")
                predictions[target] = None

        return predictions

    def create_city_risk_map(self, df, date=None):
        """Create interactive map showing air quality across cities"""
        if date is None:
            date = df['Date'].max()

        # Try exact date; otherwise fall back to latest per city
        date_data = df[df['Date'] == date]
        if len(date_data) == 0:
            date_data = df.loc[df.groupby('City')['Date'].idxmax()]

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

        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

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

        for _, row in date_data.iterrows():
            city = row['City']
            if city in city_coords:
                coords = city_coords[city]
                aqi = row.get('AQI', np.nan)
                color = get_aqi_color(aqi)

                popup_text = f"<b>{city}</b><br>Date: {pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}"
                if 'AQI' in row.index and not pd.isna(row['AQI']):
                    popup_text += f"<br>AQI: {row['AQI']:.0f}"
                if 'PM2.5' in row.index and not pd.isna(row['PM2.5']):
                    popup_text += f"<br>PM2.5: {row['PM2.5']:.1f} μg/m³"
                if 'PM10' in row.index and not pd.isna(row['PM10']):
                    popup_text += f"<br>PM10: {row['PM10']:.1f} μg/m³"
                if 'NO2' in row.index and not pd.isna(row['NO2']):
                    popup_text += f"<br>NO2: {row['NO2']:.1f} μg/m³"

                folium.CircleMarker(
                    location=coords,
                    radius=15,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fillOpacity=0.8
                ).add_to(m)

        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 210px; 
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

        current_date = pd.Timestamp.now().normalize()
        forecast_dates = pd.date_range(current_date, periods=days_ahead, freq='D')

        forecasts = []
        for date in forecast_dates:
            predictions = self.predict_for_city(city, date)

            aqi = predictions.get('AQI', 100)
            if aqi is None or pd.isna(aqi):
                risk = 'Unknown'
            elif aqi <= 100:
                risk = 'Low'
            elif aqi <= 150:
                risk = 'Moderate'
            elif aqi <= 200:
                risk = 'High'
            else:
                risk = 'Very High'

            row = {'Date': date, 'City': city, **predictions, 'Risk_Level': risk}
            forecasts.append(row)

        return pd.DataFrame(forecasts)

    def plot_results(self, results, df):
        """Create visualization plots for model results"""
        n_targets = len(results)
        if n_targets == 0:
            print("No results to plot!")
            return None

        cols = min(2, n_targets)
        rows = (n_targets + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))

        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        plot_idx = 0
        for target, result in results.items():
            if plot_idx >= (len(axes) if isinstance(axes, (list, np.ndarray)) else 1):
                break

            ax = axes[plot_idx] if isinstance(axes, (list, np.ndarray)) else axes
            ax.scatter(result['actual'], result['predictions'], alpha=0.6)

            min_val = min(result['actual'].min(), result['predictions'].min())
            max_val = max(result['actual'].max(), result['predictions'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel(f'Actual {target}')
            ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'{target} Prediction\n(R² = {result["r2"]:.3f}, RMSE = {result["rmse"]:.1f})')

            plot_idx += 1

        if isinstance(axes, (list, np.ndarray)):
            for i in range(plot_idx, len(axes)):
                try:
                    axes[i].set_visible(False)
                except Exception:
                    pass

        plt.tight_layout()
        return fig

    def analyze_data_quality(self, df):
        """Analyze and report data quality issues"""
        print("\n" + "=" * 50)
        print("DATA QUALITY ANALYSIS")
        print("=" * 50)

        print("\nMissing Values:")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data / len(df) * 100).round(2)

        for col in missing_data.index:
            if missing_data[col] > 0:
                print(f"  {col}: {missing_data[col]} ({missing_percent[col]}%)")

        print(f"\nDate Range by City:")
        for city in df['City'].dropna().unique():
            city_data = df[df['City'] == city]
            print(f"  {city}: {city_data['Date'].min()} to {city_data['Date'].max()} ({len(city_data)} records)")

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

    predictor = AirQualityPredictor()

    print("Loading air quality data from 'data' folder...")
    df = predictor.load_data_from_folder("data")

    if df is None:
        print("\n" + "=" * 60)
        print("CREATING SAMPLE DATA FOR DEMONSTRATION")
        print("=" * 60)
        print("Since no data folder was found, creating sample data...")
        print("To use your own data:")
        print("1. Create a folder named 'data' in your working directory")
        print("2. Place your CSV files with columns: City, Date, PM2.5, PM10, NO2, etc.")
        print("3. Run the script again")
        print("\n" + "=" * 60)

        cities = ['Ahmedabad', 'Delhi', 'Mumbai', 'Chennai']
        sample_data = []

        date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')

        rng = np.random.default_rng(42)
        for date in date_range:
            for city in cities:
                month = date.month
                base_aqi = 100 + rng.normal(0, 30)

                if month in [11, 12, 1, 2]:
                    base_aqi += 50
                elif month in [6, 7, 8, 9]:
                    base_aqi -= 30

                city_factor = {'Ahmedabad': 1.1, 'Delhi': 1.5, 'Mumbai': 1.0, 'Chennai': 0.9}
                base_aqi *= city_factor.get(city, 1.0)

                base_aqi = max(50, min(300, base_aqi))

                pm25 = base_aqi * 0.4 + rng.normal(0, 5)
                pm10 = pm25 * 1.8 + rng.normal(0, 8)
                no2 = base_aqi * 0.3 + rng.normal(0, 10)
                co = rng.uniform(0.5, 2.0)
                so2 = rng.uniform(5, 25)
                o3 = rng.uniform(20, 100)

                sample_data.append({
                    'City': city,
                    'Date': date,
                    'PM2.5': max(0, pm25),
                    'PM10': max(0, pm10),
                    'NO': rng.uniform(10, 50),
                    'NO2': max(0, no2),
                    'NOx': max(0, no2 * 1.5),
                    'NH3': rng.uniform(20, 80),
                    'CO': max(0, co),
                    'SO2': max(0, so2),
                    'O3': max(0, o3),
                    'Benzene': rng.uniform(0, 5),
                    'Toluene': rng.uniform(0, 10),
                    'Xylene': rng.uniform(0, 8),
                    'AQI': base_aqi
                })

        df = pd.DataFrame(sample_data)
        print(f"Sample dataset created with {len(df)} records for demonstration.")

    predictor.analyze_data_quality(df)

    print("\nTraining machine learning models...")
    results, processed_df = predictor.train_models(df)

    if not results:
        print("No models were successfully trained!")
        return None, None, None, None

    print("\nGenerating pollution forecasts...")
    all_forecasts = {}
    cities = df['City'].dropna().unique()

    for city in cities[:3]:  # demo limit
        try:
            forecast = predictor.forecast_pollution(city, days_ahead=7)
            if forecast is not None:
                all_forecasts[city] = forecast
                with_cols = [c for c in ['Date', 'AQI', 'PM2.5', 'Risk_Level'] if c in forecast.columns]
                print(f"\nForecast for {city}:")
                print(forecast[with_cols].head())
        except Exception as e:
            print(f"Error generating forecast for {city}: {e}")

    print("\nGenerating visualizations...")
    try:
        fig = predictor.plot_results(results, df)
        if fig:
            plt.show()
    except Exception as e:
        print(f"Error creating plots: {e}")

    print("\nCreating risk assessment map...")
    try:
        risk_map = predictor.create_city_risk_map(df)
        map_filename = "air_quality_risk_map.html"
        risk_map.save(map_filename)
        print(f"Risk map saved as '{map_filename}'")
    except Exception as e:
        print(f"Error creating risk map: {e}")
        risk_map = None

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    for target, result in results.items():
        print(f"{target}:")
        print(f"  Model: {result['model_name']}")
        print(f"  R² Score: {result['r2']:.3f}")
        print(f"  RMSE: {result['rmse']:.2f}")
        print(f"  MAE: {result['mae']:.2f}")
        print(f"  CV Score: {result['cv_score']:.3f}")
        print()

    print("TOP FEATURES BY TARGET:")
    print("-" * 30)
    for target in results.keys():
        if target in predictor.feature_importance:
            top_features = predictor.feature_importance[target].head(3)
            print(f"{target}: {', '.join(top_features['feature'].tolist())}")

    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTION")
    print("=" * 50)
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

    print("\n" + "=" * 60)
    print("DATA LOADING INSTRUCTIONS")
    print("=" * 60)
    print("To use your own CSV files:")
    print("1. Create a 'data' folder in your current directory")
    print("2. Place CSV files with the following columns:")
    print("   Required: City, Date")
    print("   Optional: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,")
    print("            Benzene, Toluene, Xylene, AQI, AQI_Bucket")
    print("3. Date format should be readable by pandas (e.g., 1/1/2015)")
    print("4. The script will automatically detect and load all CSV files")
    print("=" * 60)

    return predictor, df, results, all_forecasts


# Additional utility functions for enhanced analysis
def analyze_seasonal_patterns(df):
    """Analyze seasonal pollution patterns"""
    if 'AQI' not in df.columns or 'Season' not in df.columns:
        return None

    seasonal_stats = df.groupby('Season')['AQI'].agg(['mean', 'std', 'min', 'max']).round(2)

    print("\n" + "=" * 40)
    print("SEASONAL POLLUTION PATTERNS")
    print("=" * 40)
    print(seasonal_stats)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    df.boxplot(column='AQI', by='Season', ax=plt.gca())
    plt.title('AQI Distribution by Season')
    plt.suptitle('')

    if 'Month' in df.columns:
        plt.subplot(2, 2, 2)
        monthly_avg = df.groupby('Month')['AQI'].mean()
        monthly_avg.plot(kind='bar')
        plt.title('Average AQI by Month')
        plt.xlabel('Month')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)

    if 'City' in df.columns:
        plt.subplot(2, 2, 3)
        city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        city_avg.plot(kind='bar')
        plt.title('Average AQI by City')
        plt.xlabel('City')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    if len(df) > 365:
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

    corr_matrix = df[available_pollutants].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Pollutant Correlation Matrix')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 40)
    print("POLLUTANT CORRELATIONS")
    print("=" * 40)

    correlations = []
    for i in range(len(available_pollutants)):
        for j in range(i + 1, len(available_pollutants)):
            corr_val = corr_matrix.iloc[i, j]
            correlations.append({
                'Pollutant1': available_pollutants[i],
                'Pollutant2': available_pollutants[j],
                'Correlation': corr_val
            })

    correlations = sorted(correlations, key=lambda x: abs(x['Correlation']), reverse=True)

    print("Strongest correlations:")
    for corr in correlations[:5]:
        print(f"  {corr['Pollutant1']} - {corr['Pollutant2']}: {corr['Correlation']:.3f}")

    return corr_matrix


def generate_air_quality_report(predictor, df, results):
    """Generate a comprehensive air quality analysis report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE AIR QUALITY ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nDATASET OVERVIEW:")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Cities covered: {df['City'].nunique()}")
    print(f"Cities: {', '.join(df['City'].dropna().unique())}")

    print(f"\nDATA COMPLETENESS:")
    key_columns = ['AQI', 'PM2.5', 'PM10', 'NO2']
    for col in key_columns:
        if col in df.columns:
            completeness = (1 - df[col].isnull().mean()) * 100
            print(f"  {col}: {completeness:.1f}%")

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
            mask = (df['AQI'] >= min_val) & (df['AQI'] <= max_val)
            total = df['AQI'].notna().sum()
            if total > 0:
                count = mask.sum()
                percentage = (count / total) * 100
                if count > 0:
                    print(f"  {category} ({min_val}-{max_val}): {count:,} records ({percentage:.1f}%)")

    if 'City' in df.columns and 'AQI' in df.columns:
        print(f"\nCITY RANKINGS (by average AQI):")
        city_rankings = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        for i, (city, avg_aqi) in enumerate(city_rankings.items(), 1):
            print(f"  {i}. {city}: {avg_aqi:.1f}")

    if results:
        print(f"\nMODEL PERFORMANCE:")
        for target, result in results.items():
            r2 = result['r2']
            if r2 > 0.8:
                acc = "Excellent"
            elif r2 > 0.6:
                acc = "Good"
            elif r2 > 0.4:
                acc = "Fair"
            else:
                acc = "Poor"
            print(f"  {target}: R² = {r2:.3f} ({acc})")

    print(f"\nRECOMMENDATIONS:")
    print("  1. Focus monitoring on cities with highest pollution levels")
    print("  2. Implement seasonal pollution control measures during high-risk periods")
    print("  3. Develop early warning systems based on model predictions")
    if results and any(result['r2'] < 0.6 for result in results.values()):
        print("  4. Collect additional meteorological data to improve model accuracy")
    print("  5. Regular calibration of monitoring equipment recommended")

    return True


if __name__ == "__main__":
    try:
        predictor, data, results, forecasts = main()

        if data is not None and len(data) > 0:
            print("\nPerforming additional analyses...")
            analyze_seasonal_patterns(data)
            create_pollutant_correlation_analysis(data)
            generate_air_quality_report(predictor, data, results)

        print("\n" + "=" * 50)
        print("AIR QUALITY PREDICTION SYSTEM READY")
        print("=" * 50)
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

