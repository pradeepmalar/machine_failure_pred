# src/pipeline/training_pipeline.py

import os
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.components.data_preprocessing import load_and_clean_data
from src.components.feature_engineering import apply_feature_engineering
from src.components.train_model import train_xgb_model, evaluate_model
from src.utils import save_object

def main():
    try:
        logging.info("🚀 Starting training pipeline...")

        # STEP 1: Load and clean data
        data_path = "data/machine failure.csv"
        logging.info(f"Loading data from: {data_path}")
        df = load_and_clean_data(data_path)

        # STEP 2: Feature engineering
        logging.info("Applying feature engineering...")
        X, y, scaler = apply_feature_engineering(df)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # STEP 3: Train model
        logging.info("Training XGBoost model...")
        model = train_xgb_model(X_train, y_train)

        # STEP 4: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        test_accuracy = metrics["accuracy"]
        logging.info(f"✅ Training complete. Test accuracy: {test_accuracy:.4f}")

        # STEP 5: Save artifacts
        logging.info("Saving model and scaler...")
        save_object(file_path="artifacts/model.pkl", obj=model)
        save_object(file_path="artifacts/scaler.pkl", obj=scaler)
        logging.info("✅ Artifacts saved successfully.")

        print(f"\n🎯 Final model test accuracy: {test_accuracy:.4f}")

    except Exception as e:
        logging.error("❌ Training pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
