from flask import Flask, render_template, request
import ee
import geemap.foliumap as geemap
import traceback

# Initialize Earth Engine
try:
    ee.Initialize(project="rishishahubuggers")
    print("Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")

app = Flask(__name__)

def get_forecast_map(days, layer_choice, pollutant_type):
    try:
        # Get country boundaries
        countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
        india = countries.filter(ee.Filter.eq('country_na', 'India'))
        
        if pollutant_type == "NO2":
            return get_no2_map(days, layer_choice, india)
        else:  # PM2.5
            return get_pm25_map(days, layer_choice, india)
            
    except Exception as e:
        print(f"Error in get_forecast_map: {e}")
        traceback.print_exc()
        # Return a basic map if there's an error
        m = geemap.Map(center=[22.59, 78.96], zoom=5)
        m.addLayer(india, {'color': 'red'}, 'India Boundary')
        return m

def get_no2_map(days, layer_choice, india):
    """Create NO2 forecast map"""
    try:
        # NO₂ data from Sentinel-5P
        collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
            .select('NO2_column_number_density') \
            .filterBounds(india) \
            .filterDate('2019-06-01', '2020-06-01')
        
        # Add time band for trend analysis
        start = ee.Date('2019-06-01')
        
        def add_time(img):
            days_since = img.date().difference(start, 'day')
            return img.addBands(ee.Image.constant(days_since).rename('time').toFloat())
        
        no2_with_time = collection.map(add_time)
        
        # Calculate linear trend
        trend = no2_with_time.select(['time', 'NO2_column_number_density']) \
            .reduce(ee.Reducer.linearFit())
        
        # Generate forecast
        forecast = trend.select('scale').multiply(days).add(trend.select('offset')).clip(india)
        
        # Calculate risk zones
        risk = forecast.expression(
            "b(0) > 0.00015 ? 3 : (b(0) > 0.0001 ? 2 : 1)"
        )
        
        # Create map
        m = geemap.Map(center=[22.59, 78.96], zoom=5)
        
        if layer_choice == "Forecast":
            viz_params = {'min': 0, 'max': 0.0002, 'palette': ['blue', 'green', 'yellow', 'orange', 'red']}
            m.addLayer(forecast, viz_params, f'NO₂ Forecast ({days} days)')
        else:
            viz_params = {'min': 1, 'max': 3, 'palette': ['green', 'yellow', 'red']}
            m.addLayer(risk, viz_params, 'NO₂ Risk Zones')
        
        # Add India boundary
        m.addLayer(india, {'color': 'black', 'fillColor': '00000000'}, 'India Boundary')
        
        return m
        
    except Exception as e:
        print(f"Error in NO2 processing: {e}")
        return create_fallback_map(india)

def get_pm25_map(days, layer_choice, india):
    """Create PM2.5 forecast map using aerosol data"""
    try:
        # Use MODIS Aerosol Optical Depth as proxy for PM2.5
        collection = ee.ImageCollection('MODIS/006/MCD19A2_GRANULES') \
            .select('Optical_Depth_047') \
            .filterBounds(india) \
            .filterDate('2019-01-01', '2020-01-01')
        
        # Add time band
        start = ee.Date('2019-01-01')
        
        def add_time(img):
            days_since = img.date().difference(start, 'day')
            return img.addBands(ee.Image.constant(days_since).rename('time').toFloat())
        
        aod_with_time = collection.map(add_time)
        
        # Calculate trend
        trend = aod_with_time.select(['time', 'Optical_Depth_047']) \
            .reduce(ee.Reducer.linearFit())
        
        # Generate forecast (convert AOD to approximate PM2.5)
        forecast = trend.select('scale').multiply(days).add(trend.select('offset')).multiply(50).clip(india)
        
        # Calculate risk zones for PM2.5 (WHO guidelines: >15 μg/m³ unhealthy)
        risk = forecast.expression(
            "b(0) > 35 ? 3 : (b(0) > 15 ? 2 : 1)"
        )
        
        # Create map
        m = geemap.Map(center=[22.59, 78.96], zoom=5)
        
        if layer_choice == "Forecast":
            viz_params = {'min': 0, 'max': 50, 'palette': ['blue', 'green', 'yellow', 'orange', 'red']}
            m.addLayer(forecast, viz_params, f'PM2.5 Forecast ({days} days)')
        else:
            viz_params = {'min': 1, 'max': 3, 'palette': ['green', 'yellow', 'red']}
            m.addLayer(risk, viz_params, 'PM2.5 Risk Zones')
        
        # Add India boundary
        m.addLayer(india, {'color': 'black', 'fillColor': '00000000'}, 'India Boundary')
        
        return m
        
    except Exception as e:
        print(f"Error in PM2.5 processing: {e}")
        return create_fallback_map(india)

def create_fallback_map(india):
    """Create a basic map when data processing fails"""
    m = geemap.Map(center=[22.59, 78.96], zoom=5)
    m.addLayer(india, {'color': 'blue', 'fillColor': '0000FF30'}, 'India')
    
    # Add some sample data
    sample_data = ee.Image.random().multiply(100).clip(india)
    m.addLayer(sample_data, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'red']}, 'Sample Data')
    
    return m

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        # Get form parameters with defaults
        days = int(request.form.get("days", 30))
        layer_choice = request.form.get("layer", "Forecast")
        pollutant_type = request.form.get("pollutant", "NO2")
        
        # Validate inputs
        days = max(1, min(days, 365))  # Clamp between 1 and 365
        
        print(f"Processing request: {pollutant_type}, {days} days, {layer_choice}")
        
        # Generate map
        m = get_forecast_map(days, layer_choice, pollutant_type)
        map_html = m.to_html()
        
        return render_template("index.html", 
                             map_html=map_html, 
                             days=days, 
                             layer_choice=layer_choice,
                             pollutant_type=pollutant_type)
    
    except Exception as e:
        print(f"Error in main route: {e}")
        traceback.print_exc()
        
        # Return error page with basic map
        m = geemap.Map(center=[22.59, 78.96], zoom=5)
        map_html = m.to_html()
        
        return render_template("index.html", 
                             map_html=map_html, 
                             days=30, 
                             layer_choice="Forecast",
                             pollutant_type="NO2",
                             error_message="An error occurred while processing your request.")

if __name__ == "__main__":
    print("Starting Flask application...")
    print("Make sure you have:")
    print("1. Google Earth Engine account set up")
    print("2. earthengine-api installed: pip install earthengine-api")
    print("3. geemap installed: pip install geemap")
    print("4. Authenticated: earthengine authenticate")
    print("\nStarting server on http://localhost:9000")
    
    app.run(debug=True, port=9000)
