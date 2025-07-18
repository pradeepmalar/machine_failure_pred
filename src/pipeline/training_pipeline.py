import sys
import joblib
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_preprocessing import load_and_clean_data
from src.components.feature_engineering import apply_feature_engineering
from src.components.train_model import train_xgb_model, evaluate_model
from src.utils import save_object
from src.exception import CustomException

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

        # STEP 3: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # STEP 4: Train model
        logging.info("Training XGBoost model...")
        model = train_xgb_model(X_train, y_train)
        feature_columns = X_train.columns.tolist()
        joblib.dump(feature_columns, "artifacts/feature_columns.pkl")
        logging.info("Feature column order saved to artifacts/feature_columns.pkl")

        # STEP 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        logging.info(f"✅ Training complete. Test accuracy: {metrics['accuracy']:.4f}")

        # STEP 6: Save model and scaler artifacts
        logging.info("Saving model and scaler...")
        save_object("artifacts/model.pkl", model)
        save_object("artifacts/scaler.pkl", scaler)
        logging.info("✅ Artifacts saved successfully.")
        
        print(f"Final test accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        logging.error("❌ Training pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
