from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import h3
# from explainer import explain_hex
# import shap

app = FastAPI(title="Coffee Chain Predictor")

# Mock modely
class MockModel:
    def predict_proba(self, df):
        return [[0.3, 0.7]]

model = MockModel()
FEATURES = ["nearest_cafe_distance", "count_nearby_cafes", "hex_total_cafes", "hex_chain_count", "hex_chain_ratio", "ring_cafe_count"]

class LocationInput(BaseModel):
    lat: float
    lon: float


def build_features(lat, lon, df=None):
    # h3_cell = h3.latlng_to_cell(lat, lon, 9)

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


# @app.post("/predict")
# def predict(request: PredictionRequest):
#     features = request.dict()
#     feature_df = pd.DataFrame([features])[FEATURE_COLS]

#     score = float(pipeline.predict_proba(feature_df)[0, 1])

#     # Compute SHAP for this single prediction
#     explainer = shap.Explainer(pipeline.named_steps['model'],
#                                pd.DataFrame([features])[FEATURE_COLS])
#     shap_vals  = explainer(feature_df)
#     shap_dict  = dict(zip(FEATURE_COLS, shap_vals.values[0].tolist()))

#     explanation = explain_hex(
#         hex_id=request.h3_cell,
#         features=features,
#         shap_values=shap_dict,
#         score=score,
#     )

#     return {
#         "h3_cell":           request.h3_cell,
#         "chain_likelihood":  round(score, 4),
#         "explanation":       explanation,
#         "top_shap_drivers":  dict(sorted(shap_dict.items(),
#                                          key=lambda x: abs(x[1]),
#                                          reverse=True)[:3]),
#     }