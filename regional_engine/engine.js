// 1. Load Sentinel-5P NO2 dataset
var no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
  .select('NO2_column_number_density')
  .filterDate('2019-06-01', '2020-06-06');

// 2. India boundary
var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');
var india = countries.filter(ee.Filter.eq('country_na', 'India'));

// 3. Add a "time" band (in days since start date) to each image
var startDate = ee.Date('2019-06-01');
var no2WithTime = no2.map(function(img) {
  var days = img.date().difference(startDate, 'day');
  return img.addBands(ee.Image.constant(days).rename('time').toFloat());
});

var trend = no2WithTime.select(['time', 'NO2_column_number_density'])
  .reduce(ee.Reducer.linearFit());



var forecastNO2 = trend.select('scale').multiply(30).add(trend.select('offset'));

// 6. Risk Zone Classification (based on forecasted NO2 values)
var riskZones = forecastNO2.expression(
  "b(0) > 0.00015 ? 3 : (b(0) > 0.0001 ? 2 : 1)"
);

// 1 = Low, 2 = Moderate, 3 = High
var palette = ['green', 'yellow', 'red'];

// 7. Visualization
Map.centerObject(india, 5);
Map.addLayer(forecastNO2.clip(india),
  {min:0, max:0.0002, palette:['blue','orange','red']},
  "Forecasted NO2 (30 days)");
Map.addLayer(riskZones.clip(india),
  {min:1, max:3, palette:palette},
  "Risk Zones");

// 8. Add Label
var label = ui.Label({
  value: 'Forecasted NO₂ Risk Zones (30-day projection)',
  style: {
    position: 'bottom-left',
    padding: '8px',
    fontSize: '14px',
    backgroundColor: 'rgba(255, 255, 255, 0.6)'
  }
});
Map.add(label);
