import h3
import numpy as np
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMap

Path("reports").mkdir(exist_ok=True)



lat_min, lat_max = 53.30, 53.45
lon_min, lon_max = -2.99, -2.85
resolution = 9

def generate_h3_grid(lat_min, lat_max, lon_min, lon_max, res=9):
    lats = np.linspace(lat_min, lat_max, 50)
    lons = np.linspace(lon_min, lon_max, 50)

    cells = set()

    for lat in lats:
        for lon in lons:
            cells.add(h3.latlng_to_cell(lat, lon, res))

    return list(cells)


def build_cell_features(cell):
    lat, lon = h3.cell_to_latlng(cell)

    return {
        "nearest_cafe_distance": 200,
        "count_nearby_cafes": 5,
        "hex_total_cafes": 10,
        "hex_chain_count": 2,
        "hex_chain_ratio": 0.2,
        "ring_cafe_count": 15
    }





print("🔥 heatmap.py is running")

model_data = joblib.load("models/coffee_chain_model.pkl")
model = model_data["model"]
features = model_data["features"]

print("✅ model loaded")

cells = generate_h3_grid(lat_min, lat_max, lon_min, lon_max)
print("✅ grid created:", len(cells))

results = []

for cell in cells:
    
    feats = build_cell_features(cell)

    df = pd.DataFrame([feats])

    # ensure all required features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0

    df = df[features]

    prob = model.predict_proba(df)[0][1]

    lat, lon = h3.cell_to_latlng(cell)

    results.append({
        "h3": cell,
        "lat": lat,
        "lon": lon,
        "prob": prob
    })

heatmap_df = pd.DataFrame(results)




plt.figure(figsize=(10, 8))

plt.scatter(
    heatmap_df["lon"],
    heatmap_df["lat"],
    c=heatmap_df["prob"],
    cmap="Reds",
    s=100
)

plt.colorbar(label="Chain Probability")
plt.title("Coffee Chain Probability Heatmap (Liverpool)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig("reports/heatmap_debug.png")
plt.close()





m = folium.Map(location=[53.4084, -2.9916], zoom_start=12)


heat_data = [
    [row["lat"], row["lon"], row["prob"]]
    for _, row in heatmap_df.iterrows()
]


HeatMap(
    heat_data,
    radius=15,
    blur=10,
    max_zoom=13
).add_to(m)


print("💾 about to save HTML")
output_path = Path("reports/coffee_heatmap.html").resolve()
print("Saving to:", output_path)

m.save(output_path)