import requests
import time
import pandas as pd
import re
import numpy as np
import h3



def validate_data(cafe):
    name = cafe.get("name")
    lat = cafe.get("lat")
    lon = cafe.get("lon")

    if not name:
        return False

    if lat is None or lon is None:
        return False

    if not (-90 <= lat <= 90):
        return False

    if not (-180 <= lon <= 180):
        return False

    return True



def fetch_coffee_shops(city):
    query = f"""
    [out:json][timeout:25];

    rel["name"="{city}"]["boundary"="administrative"];
    map_to_area->.searchArea;

    node["amenity"="cafe"](area.searchArea);

    out body;
    """

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query,
        headers={"User-Agent": "coffee-ml-pipeline/1.0"}
    )

    if response.status_code != 200:
        raise Exception(f"Overpass error: {response.status_code}")

    if not response.text.strip():
        raise Exception("Empty response from Overpass API")

    try:
        return response.json()
    except Exception:
        raise Exception(f"Invalid JSON response: {response.text[:200]}")


def extract_features(data):
    elements = data.get("elements", [])

    cafes = []

    for c in elements:
        tags = c.get("tags", {})

        name = tags.get("name")
        lat = c.get("lat")
        lon = c.get("lon")

        cafe = {
            "name": name,
            "lat": lat,
            "lon": lon
        }

        if validate_data(cafe):
            cafes.append(cafe)

    return cafes


def remove_duplicates(data):
    seen = set()
    unique_cafes = []

    for cafe in data:
        name = cafe.get("name")
        lat = cafe.get("lat")
        lon = cafe.get("lon")

        if name is None or lat is None or lon is None:
            continue

        identifier = (name.strip().lower(), round(lat, 6), round(lon, 6))

        if identifier not in seen:
            seen.add(identifier)
            unique_cafes.append(cafe)

    return unique_cafes