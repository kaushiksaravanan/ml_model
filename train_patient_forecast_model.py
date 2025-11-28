"""
LightGBM forecasting model — predicts next 72 hours (3 days) of:
1. `patient_load`
2. `which disease will spike`
3. `how intense the spike will be`

Trains using 1 year of historical city-level data from `patient_load_daily_1y_multicity.csv`
Saves final model as `final_patient_disease_forecast_model.joblib` in the same folder.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

# ----------------------
# Load and preprocess
# ----------------------
print("📂 Loading CSV data...")
df = pd.read_csv('patient_load_daily_1y_multicity.csv')

# Normalize columns
df.columns = [c.strip().lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date'])

# Encode categorical columns
cat_cols = ['city', 'pincode', 'is_weekend', 'is_festival', 'school_open', 'month']
for col in cat_cols:
    if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Sort by date for forecasting
df = df.sort_values(['city', 'pincode', 'date']).reset_index(drop=True)

# ----------------------
# Feature Engineering
# ----------------------
def add_lag_features(df, lag_days=[1,2,3,7,14]):
    for lag in lag_days:
        for col in ['patient_load', 'respiratory_cases', 'flu_cases', 'vector_cases', 'gastro_cases', 'other_cases']:
            df[f'{col}_lag{lag}'] = df.groupby(['city','pincode'])[col].shift(lag)
    return df

print("🧠 Creating lag features...")
df = add_lag_features(df)

# Drop rows with missing lag values
df = df.dropna().reset_index(drop=True)

# Define features
features = [
    'city', 'pincode', 'population_density', 'no_of_hospitals', 'avg_capacity',
    'pm2_5', 'pm10', 'temperature_mean_c', 'relative_humidity_mean', 'uv_index_mean', 'rain_mm',
    'is_weekend', 'is_festival', 'school_open', 'month',
    'respiratory_risk', 'flu_risk', 'vector_risk', 'gastro_risk'
]

# Add lag features dynamically
features += [c for c in df.columns if c.endswith(tuple([f'_lag{i}' for i in [1,2,3,7,14]]))]

# ----------------------
# Targets
# ----------------------
y_load = df['patient_load']
risk_cols = ['respiratory_risk', 'flu_risk', 'vector_risk', 'gastro_risk']
df['max_risk'] = df[risk_cols].idxmax(axis=1)
y_disease = df['max_risk']

# Define intensity based on change in patient load
load_diff = df.groupby(['city','pincode'])['patient_load'].diff().fillna(0)
df['spike_intensity'] = pd.cut(
    load_diff,
    bins=[-np.inf, 10, 50, 150, np.inf],
    labels=['Low', 'Moderate', 'High', 'Severe']
)
y_intensity = df['spike_intensity']

# ----------------------
# Train models
# ----------------------
print("🚀 Training patient load forecaster...")
model_load = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1500,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_load.fit(df[features], y_load)

print("🚀 Training disease spike classifier...")
model_disease = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=800,
    learning_rate=0.03,
    random_state=42
)
model_disease.fit(df[features], y_disease)

print("🚀 Training spike intensity classifier...")
model_intensity = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=800,
    learning_rate=0.03,
    random_state=42
)
model_intensity.fit(df[features], y_intensity)

# ----------------------
# Save final model bundle
# ----------------------
joblib.dump({
    'load_model': model_load,
    'disease_model': model_disease,
    'intensity_model': model_intensity,
    'features': features
}, 'final_patient_disease_forecast_model.joblib')

print("\n✅ Final 72-hour forecast model saved as final_patient_disease_forecast_model.joblib")

# ----------------------
# Prediction function (72-hour forecast)
# ----------------------
def forecast_next_3_days(history_df: pd.DataFrame):
    models = joblib.load('final_patient_disease_forecast_model.joblib')
    load_model = models['load_model']
    disease_model = models['disease_model']
    intensity_model = models['intensity_model']
    feats = models['features']

    forecast_rows = []

    hist = history_df.copy()
    hist = hist.sort_values(['city', 'pincode', 'date']).reset_index(drop=True)

    for day in range(1, 4):  # next 3 days
        next_date = hist['date'].max() + timedelta(days=day)
        new_entry = hist.iloc[-1:].copy()
        new_entry['date'] = next_date

        # recompute lags dynamically
        for lag in [1,2,3,7,14]:
            for col in ['patient_load', 'respiratory_cases', 'flu_cases', 'vector_cases', 'gastro_cases', 'other_cases']:
                new_entry[f'{col}_lag{lag}'] = hist[col].iloc[-lag]

        X_new = new_entry[feats]
        load_pred = load_model.predict(X_new)[0]
        disease_pred = disease_model.predict(X_new)[0]
        intensity_pred = intensity_model.predict(X_new)[0]

        new_entry['predicted_patient_load'] = load_pred
        new_entry['predicted_disease_spike'] = disease_pred
        new_entry['predicted_intensity'] = intensity_pred

        forecast_rows.append(new_entry[['city','pincode','date','predicted_patient_load','predicted_disease_spike','predicted_intensity']])
        
        # Append prediction to history for next-step forecasting
        new_entry['patient_load'] = load_pred
        hist = pd.concat([hist, new_entry]).reset_index(drop=True)

    forecast_df = pd.concat(forecast_rows).reset_index(drop=True)
    return forecast_df


# ----------------------
# Run forecast automatically after training
# ----------------------
print("\n🔮 Generating 72-hour forecast (based on last available data)...")
forecast_result = forecast_next_3_days(df)
print("\n=== 72-HOUR FORECAST RESULTS ===")
print(forecast_result.head(10))

# Save the forecast output
forecast_result.to_csv("next_3_days_forecast.csv", index=False)
print("\n📁 Forecast saved as next_3_days_forecast.csv")
