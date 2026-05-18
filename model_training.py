import joblib
from pathlib import Path
import joblib
Path("models").mkdir(parents=True, exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


FEATURE_COLS = [
    'nearest_cafe_distance',
    'count_nearby_cafes',
    'hex_total_cafes',
    'hex_chain_count',
    'hex_chain_ratio',
    'ring_cafe_count'
]

TARGET_COL = 'is_chain_cafe'


def spatial_split(df):

    lat_threshold = df['h3_lat'].quantile(0.25)

    val_mask = df['h3_lat'] < lat_threshold

    train_df = df[~val_mask]
    val_df = df[val_mask]

    return (
        train_df[FEATURE_COLS],
        train_df[TARGET_COL],
        val_df[FEATURE_COLS],
        val_df[TARGET_COL]
    )


def train(df):

    X_train, y_train, X_val, y_val = spatial_split(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)

    probs = pipeline.predict_proba(X_val)[:, 1]

    print("\nClassification Report\n")
    print(classification_report(y_val, preds))

    print("\nROC AUC")
    print(roc_auc_score(y_val, probs))

    joblib.dump(
        {
            "model": pipeline,
            "features": FEATURE_COLS
        },
        "models/coffee_chain_model.pkl"
    )

    print("\nModel saved.")

    return {
        "model": pipeline,
        "features": FEATURE_COLS
    }