import logging
import json
import pandas as pd

import data_ingestion
import constants
import feature_engineering
import model_training
import api
import mlflow


logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
    )


def access_ingestion():
    logging.info("Starting data ingestion")
    try:
        data = data_ingestion.fetch_coffee_shops(constants.LOCATION)
        logging.info(f"Data ingestion completed successfully. Fetched {len(data['elements'])} records.")

        extracted_data = data_ingestion.extract_features(data)
        logging.info(f"Feature extraction completed. Extracted {len(extracted_data)} valid records")

        unique_data = data_ingestion.remove_duplicates(extracted_data)
        logging.info(f"Duplicate removal completed. {len(unique_data)} unique records remain.")

        raw_df = pd.DataFrame(unique_data)

        logging.info(unique_data[:5])  
        logging.info("Data ingestion pipeline completed successfully.")

        return raw_df

    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        raise


def call_feature_engineering(raw_df):
    logging.info("Starting feature engineering")
    
    h3_df = feature_engineering.assign_h3(raw_df, resolution=9)
    logging.info(f"H3 assignment completed successfully. {len(h3_df)} records with H3 cells assigned.")
    feat_df = feature_engineering.build_features(h3_df, radius_m=500)
    logging.info(f"Feature building completed successfully. {len(feat_df)} records with features built.")
    hex_df = feature_engineering.hex_aggregations(feat_df)
    logging.info(f"Hex feature aggregation completed successfully. {len(hex_df)} hex-level records created.")
    
    logging.info(f"Sample of engineered features:\n{hex_df.info()}")
    logging.info("Feature engineering completed successfully.")
    return hex_df


def train_model(processed_df):
    logging.info("Starting model training")
    target = model_training.get_city_hexes()

    
    logging.info("Model training completed successfully.")


raw_df = access_ingestion()
processed_df = call_feature_engineering(raw_df)

run_id = model_training.train(processed_df)

mlflow.register_model(
    f"runs:/{run_id}/model",
    "coffee-chain-model"
)


