from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Load API key from .env
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found in .env file")
BASE_URL = "http://api.openweathermap.org/data/2.5"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
HISTORICAL_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

class EnvironmentalDataProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pollution_history = defaultdict(list)
    
    def get_air_pollution(self, lat, lon):
        """Get current air pollution data"""
        try:
            url = f"{AIR_POLLUTION_URL}?lat={lat}&lon={lon}&appid={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching air pollution data: {e}")
            return None
    
    def get_historical_air_pollution(self, lat, lon, start, end):
        """Get historical air pollution data"""
        try:
            url = f"{HISTORICAL_POLLUTION_URL}?lat={lat}&lon={lon}&start={start}&end={end}&appid={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_weather_data(self, lat, lon):
        """Get current weather data for water quality indicators"""
        try:
            url = f"{BASE_URL}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def analyze_pollution_trend(self, pollution_data):
        """Analyze pollution trends and detect anomalies"""
        if not pollution_data or 'list' not in pollution_data:
            return None
        
        aqi_values = []
        timestamps = []
        
        for entry in pollution_data['list']:
            aqi_values.append(entry['main']['aqi'])
            timestamps.append(entry['dt'])
        
        if len(aqi_values) < 2:
            return None
        
        # Calculate moving average for anomaly detection
        df = pd.DataFrame({'aqi': aqi_values, 'timestamp': timestamps})
        df['rolling_mean'] = df['aqi'].rolling(window=min(5, len(aqi_values))).mean()
        df['anomaly'] = abs(df['aqi'] - df['rolling_mean']) > df['aqi'].std()
        
        return {
            'trend': 'increasing' if aqi_values[-1] > aqi_values[0] else 'decreasing',
            'anomalies': df[df['anomaly']].to_dict('records'),
            'avg_aqi': np.mean(aqi_values),
            'max_aqi': max(aqi_values),
            'min_aqi': min(aqi_values)
        }
    
    def estimate_water_pollution(self, weather_data, air_pollution):
        """Estimate water pollution based on available data"""
        if not weather_data or not air_pollution:
            return None
        
        # Simple estimation based on air quality, humidity, and precipitation
        humidity = weather_data.get('main', {}).get('humidity', 50)
        rain = weather_data.get('rain', {}).get('1h', 0)
        aqi = air_pollution['list'][0]['main']['aqi'] if air_pollution.get('list') else 3
        
        # Estimation formula (simplified)
        water_quality_index = max(1, min(5, aqi + (100 - humidity) / 20 - rain * 0.1))
        
        return {
            'estimated_wqi': round(water_quality_index, 1),
            'factors': {
                'air_quality_influence': aqi,
                'humidity': humidity,
                'precipitation': rain
            }
        }
    
    def detect_crop_burning(self, lat, lon, weather_data, air_pollution):
        """Detect potential crop burning based on pollution spikes and weather"""
        if not weather_data or not air_pollution:
            return None
        
        current_month = datetime.now().month
        burning_season = current_month in [3, 4, 5, 10, 11]  # Common burning seasons
        
        aqi = air_pollution['list'][0]['main']['aqi'] if air_pollution.get('list') else 1
        pm25 = air_pollution['list'][0]['components'].get('pm2_5', 0) if air_pollution.get('list') else 0
        
        # Detection criteria
        high_pollution = aqi >= 4  # Unhealthy levels
        high_pm25 = pm25 > 35  # WHO guideline threshold
        low_wind = weather_data.get('wind', {}).get('speed', 10) < 5  # Low wind dispersal
        
        probability = 0
        if burning_season:
            probability += 30
        if high_pollution:
            probability += 25
        if high_pm25:
            probability += 25
        if low_wind:
            probability += 20
        
        return {
            'burning_probability': min(100, probability),
            'factors': {
                'burning_season': burning_season,
                'high_pollution': high_pollution,
                'high_pm25': high_pm25,
                'low_wind_dispersal': low_wind
            },
            'pm25_level': pm25,
            'wind_speed': weather_data.get('wind', {}).get('speed', 0)
        }

# Initialize processor
processor = EnvironmentalDataProcessor(OPENWEATHER_API_KEY)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/pollution/<float:lat>/<float:lon>')
def get_pollution_data(lat, lon):
    """API endpoint to get comprehensive pollution data"""
    
    # Get current data
    air_pollution = processor.get_air_pollution(lat, lon)
    weather_data = processor.get_weather_data(lat, lon)
    
    # Get historical data (last 7 days)
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=7)).timestamp())
    historical_pollution = processor.get_historical_air_pollution(lat, lon, start_time, end_time)
    
    # Process data
    trend_analysis = processor.analyze_pollution_trend(historical_pollution)
    water_estimation = processor.estimate_water_pollution(weather_data, air_pollution)
    crop_burning = processor.detect_crop_burning(lat, lon, weather_data, air_pollution)
    
    # Prepare response
    response_data = {
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
        'air_pollution': air_pollution,
        'water_pollution': water_estimation,
        'weather': weather_data,
        'trend_analysis': trend_analysis,
        'crop_burning': crop_burning,
        'status': 'success'
    }
    
    return jsonify(response_data)

@app.route('/api/locations')
def get_sample_locations():
    """Get sample locations for testing"""
    locations = [
        {'name': 'Delhi, India', 'lat': 28.6139, 'lon': 77.2090},
        {'name': 'Mumbai, India', 'lat': 19.0760, 'lon': 72.8777},
        {'name': 'Beijing, China', 'lat': 39.9042, 'lon': 116.4074},
        {'name': 'Los Angeles, USA', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'London, UK', 'lat': 51.5074, 'lon': -0.1278},
        {'name': 'São Paulo, Brazil', 'lat': -23.5505, 'lon': -46.6333},
        {'name': 'Nagpur, India', 'lat': 21.1458, 'lon': 79.0882},

  {'name': 'Bengaluru, India', 'lat': 12.9716, 'lon': 77.5946},
  {'name': 'Hyderabad, India', 'lat': 17.3850, 'lon': 78.4867},
  {'name': 'Ahmedabad, India', 'lat': 23.0225, 'lon': 72.5714},
  {'name': 'Chennai, India', 'lat': 13.0827, 'lon': 80.2707},
  {'name': 'Kolkata, India', 'lat': 22.5726, 'lon': 88.3639},
  {'name': 'Surat, India', 'lat': 21.1702, 'lon': 72.8311},
  {'name': 'Pune, India', 'lat': 18.5204, 'lon': 73.8567},
  {'name': 'Jaipur, India', 'lat': 26.9124, 'lon': 75.7873},
  {'name': 'Lucknow, India', 'lat': 26.8467, 'lon': 80.9462},
  {'name': 'Kanpur, India', 'lat': 26.4499, 'lon': 80.3319},
  {'name': 'Nagpur, India', 'lat': 21.1458, 'lon': 79.0882},
  {'name': 'Indore, India', 'lat': 22.7196, 'lon': 75.8577},
  {'name': 'Bhopal, India', 'lat': 23.2599, 'lon': 77.4126}
    ]
    return jsonify(locations)

if __name__ == '__main__':
    app.run(debug=True)
