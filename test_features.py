# tests/test_features.py
import pandas as pd
import numpy as np
from model_training import spatial_split, FEATURE_COLS

def make_dummy_df(n=100):
    df = pd.DataFrame({
        'h3_lat':                np.linspace(53.35, 53.50, n),
        'h3_lon':                np.full(n, -2.99),
        'nearest_cafe_distance': np.random.uniform(50, 500, n),
        'count_nearby_cafes':    np.random.randint(0, 10, n),
        'hex_total_cafes':       np.random.randint(1, 20, n),
        'hex_chain_count':       np.random.randint(0, 5, n),
        'hex_chain_ratio':       np.random.uniform(0, 1, n),
        'ring_cafe_count':       np.random.randint(0, 30, n),
        'is_chain_cafe':         np.random.randint(0, 2, n),
        'h3_cell':               [f'cell_{i}' for i in range(n)],
    })
    return df

def test_spatial_split_no_leakage():
    df = make_dummy_df(100)
    X_train, y_train, X_val, y_val = spatial_split(df)
    # Val set should be southern rows only — no overlap
    assert len(X_train) + len(X_val) == len(df)
    assert X_train.index.intersection(X_val.index).empty

def test_spatial_split_has_both_classes():
    df = make_dummy_df(200)
    _, y_train, _, y_val = spatial_split(df)
    assert y_train.nunique() == 2, "Train set missing a class"
    assert y_val.nunique() == 2,   "Val set missing a class"

def test_feature_cols_present():
    df = make_dummy_df()
    X_train, *_ = spatial_split(df)
    assert list(X_train.columns) == FEATURE_COLS