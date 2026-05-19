import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

Path("reports").mkdir(exist_ok=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("coffee-location-model")

FEATURE_COLS = [
    'nearest_cafe_distance',
    'count_nearby_cafes',
    'hex_total_cafes',
    'hex_chain_count',
    'hex_chain_ratio',
    'ring_cafe_count',
]
TARGET_COL = 'is_chain_cafe'


def spatial_split(df):
    lat_threshold = df['h3_lat'].quantile(0.25)
    val_mask = df['h3_lat'] < lat_threshold
    train_df, val_df = df[~val_mask], df[val_mask]
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")
    print(f"Train chain rate: {train_df[TARGET_COL].mean():.2%} | "
          f"Val chain rate: {val_df[TARGET_COL].mean():.2%}")
    return (
        train_df[FEATURE_COLS], train_df[TARGET_COL],
        val_df[FEATURE_COLS],   val_df[TARGET_COL],
    )


def log_eval_artifacts(pipeline, X_val, y_val, probs):
    """Save curves and SHAP plot as MLflow artifacts."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_val, probs, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_val, probs, ax=axes[1])
    # Mark the no-skill baseline on PR curve
    axes[1].axhline(y=y_val.mean(), color='gray', linestyle='--',
                    label=f'Baseline ({y_val.mean():.2%})')
    axes[1].legend()
    plt.tight_layout()
    path = "reports/eval_curves.png"
    fig.savefig(path, dpi=120)
    plt.close()
    mlflow.log_artifact(path)

    # SHAP — use the underlying model, not the pipeline
    model_step = pipeline.named_steps['model']
    X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=FEATURE_COLS)
    explainer = shap.Explainer(model_step, X_val_scaled_df)
    shap_values = explainer(X_val_scaled_df)
    shap.plots.beeswarm(shap_values, show=False)
    shap_path = "reports/shap_beeswarm.png"
    plt.savefig(shap_path, dpi=120, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(shap_path)


def train(df):
    X_train, y_train, X_val, y_val = spatial_split(df)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"Class imbalance — neg: {neg}, pos: {pos}, ratio: {neg/pos:.1f}x")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, probs)
        pr_auc  = average_precision_score(y_val, probs)

        mlflow.log_params({
            "model":         "GradientBoostingClassifier",
            "n_estimators":  200,
            "max_depth":     4,
            "learning_rate": 0.05,
            "split":         "spatial_south_25pct",
        })
        mlflow.log_metrics({"roc_auc": roc_auc, "pr_auc": pr_auc})

        print(classification_report(y_val, preds, target_names=['independent', 'chain']))
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        log_eval_artifacts(pipeline, X_val, y_val, probs)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="coffee-chain-model",
            input_example=X_val.iloc[:1],
        )

        return run.info.run_id