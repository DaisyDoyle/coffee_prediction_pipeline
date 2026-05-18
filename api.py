from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import h3

app = FastAPI(title="Coffee Chain Predictor")

MODEL_DATA = joblib.load("models/coffee_chain_model.pkl")
model = MODEL_DATA["model"]
FEATURES = MODEL_DATA["features"]


GLOBAL_DF = None  

class LocationInput(BaseModel):
    lat: float
    lon: float


def build_features(lat, lon, df=None):
    h3_cell = h3.latlng_to_cell(lat, lon, 9)

    features = {
        "nearest_cafe_distance": 200.0,   
        "count_nearby_cafes": 5,
        "hex_total_cafes": 12,
        "hex_chain_count": 3,
        "hex_chain_ratio": 0.25,
        "ring_cafe_count": 18
    }

    return features



@app.get("/")
def root():
    return {"status": "ok", "message": "Coffee Chain API running"}



@app.post("/predict")
def predict(data: LocationInput):

    features = build_features(data.lat, data.lon)

    df = pd.DataFrame([features])[FEATURES]

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability_chain": float(prob),
        "h3_cell": h3.latlng_to_cell(data.lat, data.lon, 9)
    }