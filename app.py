from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="Patient Load Forecast API (with Disease Risks)")

# ----------------------------
# Load trained models
# ----------------------------
model_bundle = joblib.load("final_patient_disease_forecast_model.joblib")
load_model = model_bundle["load_model"]
disease_model = model_bundle["disease_model"]
features = model_bundle["features"]

# ----------------------------
# Request Schema
# ----------------------------
class ForecastRequest(BaseModel):
    pincode: int
    aqi_index: float
    temperature_mean_c: float
    relative_humidity_mean: float
    rain_mm: float
    uv_index_mean: float

# ----------------------------
# Helper Functions
# ----------------------------
def make_base_input(req: ForecastRequest):
    """Builds a minimal input DataFrame for the model"""
    base = {
        "city": [0],
        "pincode": [req.pincode],
        "population_density": [2000],
        "no_of_hospitals": [5],
        "avg_capacity": [150],
        "pm2_5": [req.aqi_index * 0.6],
        "pm10": [req.aqi_index * 0.4],
        "temperature_mean_c": [req.temperature_mean_c],
        "relative_humidity_mean": [req.relative_humidity_mean],
        "uv_index_mean": [req.uv_index_mean],
        "rain_mm": [req.rain_mm],
        "is_weekend": [0],
        "is_festival": [0],
        "school_open": [1],
        "month": [datetime.now().month],
        "respiratory_risk": [np.random.uniform(0.1, 0.5)],
        "flu_risk": [np.random.uniform(0.1, 0.5)],
        "vector_risk": [np.random.uniform(0.1, 0.5)],
        "gastro_risk": [np.random.uniform(0.1, 0.5)],
    }

    # Add lag placeholders
    for lag in [1, 2, 3, 7, 14]:
        for col in [
            "patient_load", "respiratory_cases", "flu_cases",
            "vector_cases", "gastro_cases", "other_cases"
        ]:
            base[f"{col}_lag{lag}"] = [0.0]
    return pd.DataFrame(base)

def get_intensity_from_aqi(aqi_value: float) -> str:
    """Returns intensity category based on AQI levels"""
    if 0 <= aqi_value <= 50:
        return "Good"
    elif 51 <= aqi_value <= 100:
        return "Moderate"
    elif 101 <= aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi_value <= 200:
        return "Unhealthy"
    elif 201 <= aqi_value <= 300:
        return "Very Unhealthy"
    elif 301 <= aqi_value <= 500:
        return "Hazardous"
    else:
        return "Invalid AQI"

# ----------------------------
# Main Endpoint
# ----------------------------
@app.post("/predict_forecast")
def predict_forecast(req: ForecastRequest):
    df_input = make_base_input(req)
    preds = []
    base_date = datetime.now()

    for i in range(3):
        day = base_date + timedelta(days=i + 1)

        # --- Compute disease risks dynamically ---
        respiratory_risk = round(min(1.0, 0.15 + 0.006 * (req.aqi_index - 50) + 0.008 * max(0, 22 - req.temperature_mean_c)), 2)
        flu_risk = round(min(1.0, 0.1 + 0.05 * (1 if req.temperature_mean_c < 22 else 0) + 0.03), 2)
        vector_risk = round(min(1.0, 0.08 + 0.004 * req.relative_humidity_mean + 0.02 * (1 if req.rain_mm > 3 else 0)), 2)
        gastro_risk = round(min(1.0, 0.07 + 0.01 * max(0, req.temperature_mean_c - 32)), 2)

        risk_dict = {
            "respiratory_risk": respiratory_risk,
            "flu_risk": flu_risk,
            "vector_risk": vector_risk,
            "gastro_risk": gastro_risk
        }

        # --- Compute dominant risk (max risk value) ---
        dominant_risk = max(risk_dict, key=risk_dict.get)

        # --- Model predictions ---
        load_pred = load_model.predict(df_input)[0]
        disease_pred = disease_model.predict(df_input)[0]

        # 🔸 Optional: override predicted disease with dominant risk
        # disease_pred = dominant_risk

        intensity_pred = get_intensity_from_aqi(req.aqi_index)

        preds.append({
            "date": day.strftime("%Y-%m-%d"),
            "predicted_patient_load": float(load_pred),
            "predicted_disease_spike": str(disease_pred),
            "predicted_intensity": intensity_pred,
            "dominant_risk": dominant_risk,
            "disease_risks": risk_dict
        })

        # --- Update lag placeholders ---
        for lag in [1, 2, 3, 7, 14]:
            if f"patient_load_lag{lag}" in df_input.columns:
                df_input[f"patient_load_lag{lag}"] = load_pred

    forecast_df = pd.DataFrame(preds)

    # --- Create chart ---
    plt.figure(figsize=(7, 4))
    plt.plot(
        forecast_df["date"],
        forecast_df["predicted_patient_load"],
        marker="o",
        linewidth=2,
        color="blue"
    )
    plt.title(f"Predicted Patient Load (Pincode: {req.pincode})")
    plt.xlabel("Date")
    plt.ylabel("Predicted Patient Load")
    plt.grid(True)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    coordinates = [
        {"x": row["date"], "y": row["predicted_patient_load"]}
        for _, row in forecast_df.iterrows()
    ]

    return {
        "pincode": req.pincode,
        "forecast_days": len(forecast_df),
        "aqi_index": req.aqi_index,
        "aqi_intensity": get_intensity_from_aqi(req.aqi_index),
        "predictions": preds,
        "chart_coordinates": coordinates,
        "chart_base64": f"data:image/png;base64,{img_base64}"
    }

# ----------------------------
# User Data Endpoint
# ----------------------------
@app.get("/user_data")
def get_user_data():
    """
    Returns sample current user environmental data for testing.
    In production, this would fetch actual user location/sensor data.
    """
    return {
        "pincode": 500001,
        "city": "Hyderabad",
        "aqi_index": 145.0,
        "temperature_mean_c": 28.5,
        "relative_humidity_mean": 68.0,
        "rain_mm": 3.5,
        "uv_index_mean": 7.2,
        "timestamp": datetime.now().isoformat(),
        "location": {
            "latitude": 17.385044,
            "longitude": 78.486671
        }
    }
