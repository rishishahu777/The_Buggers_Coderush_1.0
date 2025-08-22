# The Buggers Coderush 1.0

##  Project Overview

This project integrates **IoT sensors, machine learning forecasting, and a regional risk engine** to monitor and predict air pollution levels (PM2.5, NO2, CO2, AQI). It combines **Arduino hardware**, **Python data processing**, and a **web interface** for visualization.

The system collects real-time pollution data from IoT sensors, stores and processes it, forecasts future pollution trends using ML, and displays regional risk zones via a web application.

---

##  Features

- **IoT Sensor Integration**: Arduino-based CO2 and air quality monitoring.
- **Data Logging**: Python scripts to collect and store sensor data.
- **Historical Datasets**: AQI data for Nagpur (2021–2025).
- **ML Forecasting Model**: Predicts pollution levels based on historical trends.
- **Regional Risk Engine**: JavaScript + Python engine to calculate and visualize pollution risk zones.
- **Web Interface**: HTML/CSS frontend for displaying pollution levels and risk predictions.

---

## 📂 Project Structure

```
The_Buggers_Coderush_1.0-main/
│── app.py                        # Main Python script (Arduino data logging)
│── ardiuno_C02_.ino               # Arduino sketch for CO2/air sensors
│── index new.html                 # Webpage frontend
│── style new.css                  # Styling for the webpage
│── the buggers.pdf                # Project report/documentation
│
├── IOT sensors Circuit/           # Circuit diagrams & images
│
├── data/                          # Historical AQI datasets
│   ├── AQI_nagpur_2021.csv
│   ├── AQI_nagpur_2022.csv
│   ├── AQI_nagpur_2023.csv
│   ├── AQI_nagpur_2024.csv
│   ├── AQI_nagpur_2025.csv
│
├── ml_forecasting_model/          # Machine learning forecasting module
│   ├── app.py                     # Forecasting script
│
├── regional_engine/               # Regional risk zone calculator
│   ├── engine.js                  # Risk analysis logic
│   ├── web_app/
│       ├── app.py                 # Flask web app backend
│       ├── templates/index.html   # Web app frontend
```

---

##  Setup & Installation

### 1️⃣ Arduino Sensor Setup

- Upload `ardiuno_C02_.ino` to Arduino.
- Connect CO2 / air sensors as per circuit diagrams in `IOT sensors Circuit/`.
- Ensure Arduino is connected to your system via COM port.

### 2️⃣ Python Environment Setup

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

*(If **********`requirements.txt`********** is missing, install **********`flask, pandas, numpy, matplotlib, scikit-learn, pyserial, xarray, netCDF4`********** manually)*

### 3️⃣ Running Sensor Data Logger

```bash
python app.py
```

This logs Arduino sensor readings into CSV format.

### 4️⃣ Running ML Forecasting

```bash
cd ml_forecasting_model
python app.py
```

This forecasts PM2.5/NO2 trends based on historical AQI data.

### 5️⃣ Running Web Application

```bash
cd regional_engine/web_app
python app.py
```

Now open `http://127.0.0.1:5000/` in your browser to view pollution data & risk zones.

---

## 📊 Data Sources

- **IoT Sensors (Arduino)** – Live CO2 & air quality readings.
- **Historical AQI Data** – Pre-collected datasets for Nagpur (2021–2025).
- **ML Model** – Trained on AQI datasets to predict future pollution.

---

##  Future Improvements

- Integration with **Sentinel-5P satellite APIs** for real-time regional NO2 & PM2.5 data.
- GIS dashboard to display pollution heatmaps.
- More sensor support (SO2, O3, VOCs).
- Cloud-based data pipeline (AWS/GCP/Azure).

---

##  Team

Developed as part of **Coderush 1.0 Hackathon** by *The Buggers* team.

