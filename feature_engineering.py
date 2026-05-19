import h3
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import osmnx as ox

def assign_h3(df, lat_col='lat', lon_col='lon', resolution=9):
    """
    Resolution guide for Liverpool:
    - 8  → ~460m hexagons (neighbourhood level)
    - 9  → ~170m hexagons (street cluster level)  ← recommended
    - 10 → ~65m  hexagons (individual block level)
    """
    df['h3_cell'] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], resolution),
        axis=1
    )
    
    df['h3_lat'], df['h3_lon'] = zip(*df['h3_cell'].map(h3.cell_to_latlng))
    return df


def build_features(df, radius_m=500):
    coords_rad = np.radians(df[['lat', 'lon']].values)
    tree = BallTree(coords_rad, metric='haversine')
    
    radius_rad = radius_m / 6_371_000  # Earth radius in metres

    # 1. nearest_cafe_distance (metres)
    dist, _ = tree.query(coords_rad, k=2)  # k=2: skip self
    df['nearest_cafe_distance'] = dist[:, 1] * 6_371_000

    # 2. count_nearby_cafes
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    df['count_nearby_cafes'] = counts - 1  # subtract self

    # 3. is_chain_cafe
    chains = ["starbucks", "costa", "nero", "pret"]
    df['is_chain_cafe'] = df['name'].str.lower().apply(
        lambda name: int(any(chain in name for chain in chains))
    )

    return df


def hex_aggregations(df):
    df_lookup = df.copy()

    # Hex-level aggregations (group by h3_cell)
    hex_stats = df.groupby('h3_cell').agg(
        hex_total_cafes=('name', 'count'),
        hex_chain_count=('is_chain_cafe', 'sum'),
        hex_chain_ratio=('is_chain_cafe', 'mean')
    ).reset_index()

    df = df.merge(hex_stats, on='h3_cell')

    # H3 ring features — cafes in neighbouring hexagons
    def count_in_rings(h3_cell, df_lookup, k=2):
        neighbours = h3.grid_disk(h3_cell, k)  
        return df_lookup[df_lookup['h3_cell'].isin(neighbours)].shape[0]

    df["ring_cafe_count"] = df["h3_cell"].apply(
        lambda cell: count_in_rings(cell, df_lookup, k=2)
    )

    return df


# def get_osm_features(lat, lon, radius=300):
#     point = (lat, lon)
#     tags = {
#         'amenity': ['restaurant', 'pub', 'fast_food', 'bank', 'pharmacy'],
#         'shop': ['supermarket', 'clothes', 'convenience'],
#         'public_transport': True,
#         'office': True
#     }
#     try:
#         gdf = ox.features_from_point(point, tags=tags, dist=radius)
#         return len(gdf)
#     except:
#         return 0


# def build_osm_features_bulk(df):
#     unique_hexes = df[['h3_cell', 'h3_lat', 'h3_lon']].drop_duplicates()

#     hex_features = []

#     for _, hex_row in unique_hexes.iterrows():

#         count = get_osm_features(hex_row.h3_lat, hex_row.h3_lon)

#         hex_features.append({
#             "h3_cell": hex_row.h3_cell,
#             "osm_poi_count": count
#         })

#     hex_df = pd.DataFrame(hex_features)

#     hex_df.to_parquet("data/processed/osm_hex_features.parquet")

#     return df.merge(hex_df, on="h3_cell", how="left")