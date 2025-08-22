from flask import Flask, render_template, request
import ee
import geemap.foliumap as geemap


ee.Initialize(project="rishishahubuggers")


app = Flask(__name__)

def get_forecast_map(days, layer_choice):
    countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    india = countries.filter(ee.Filter.eq('country_na', 'India'))

    collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
        .select('NO2_column_number_density') \
        .filterDate('2019-06-01', '2020-06-06')

    start = ee.Date('2019-06-01')

    def add_time(img):
        days_since = img.date().difference(start, 'day')
        return img.addBands(ee.Image.constant(days_since).rename('time').toFloat())
    no2_with_time = collection.map(add_time)


    trend = no2_with_time.select(['time', 'NO2_column_number_density']) \
        .reduce(ee.Reducer.linearFit())

    forecast = trend.select('scale').multiply(days).add(trend.select('offset')).clip(india)


    risk = forecast.expression(
        "b(0) > 0.00015 ? 3 : (b(0) > 0.0001 ? 2 : 1)"
    )

    
    m = geemap.Map(center=[22.59, 78.96], zoom=5)
    if layer_choice == "Forecasted NO₂":
        viz = {'min': 0, 'max': 0.0002, 'palette': ['blue', 'orange', 'red']}
        m.addLayer(forecast, viz, 'Forecasted NO₂')
    else:
        viz_risk = {'min': 1, 'max': 3, 'palette': ['green', 'yellow', 'red']}
        m.addLayer(risk, viz_risk, 'Risk Zones')

    return m

@app.route("/", methods=["GET", "POST"])
def index():
    days = int(request.form.get("days", 30))
    layer_choice = request.form.get("layer", "Forecasted NO₂")

    m = get_forecast_map(days, layer_choice)
    map_html = m.to_html()

    return render_template("index.html", map_html=map_html, days=days, layer_choice=layer_choice)

if __name__ == "__main__":
    app.run(debug=True)
