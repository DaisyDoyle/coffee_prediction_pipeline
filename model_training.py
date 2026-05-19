import joblib
from pathlib import Path
import joblib
Path("models").mkdir(parents=True, exist_ok=True)
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("coffee-location-model")


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

    with mlflow.start_run() as run:

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]

        roc_auc = roc_auc_score(y_val, probs)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name = "coffee-chain-model"
        )

        run_id = run.info.run_id

    return run_id

    # with mlflow.start_run() as run:
    #     run_id = run.info.run_id

        

    #     pipeline.fit(X_train, y_train)

    #     preds = pipeline.predict(X_val)

    #     probs = pipeline.predict_proba(X_val)[:, 1]

    #     print("\nClassification Report\n")
    #     classification = classification_report(y_val, preds)
    #     print(classification_report(y_val, preds))

    #     print("\nROC AUC")
    #     roc_auc =roc_auc_score(y_val, probs)
    #     print(roc_auc_score(y_val, probs))


    #     mlflow.log_metric("roc_auc", roc_auc)

    #     # log params
    #     mlflow.log_param("model_type", "LogisticRegression")

    #     # log model
    #     mlflow.sklearn.log_model(
    #         pipeline,
    #         artifact_path="model"
    #     )

    #     mlflow.register_model(
    #         f"runs:/{run_id}/model",
    #         "coffee-chain-model"
    #     )



    #     # joblib.dump(
    #     #     {
    #     #         "model": pipeline,
    #     #         "features": FEATURE_COLS
    #     #     },
    #     #     "models/coffee_chain_model.pkl"
    #     # )

    #     print("\nModel saved.")

    #     return {
    #         "model": pipeline,
    #         "features": FEATURE_COLS
    #     }

