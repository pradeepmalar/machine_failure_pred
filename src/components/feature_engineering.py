import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

from src.logger import logging
from src.exception import CustomException

def apply_feature_engineering(df):
    """
    Performs feature engineering:
    - Separates input features (X) and label (y)
    - One-hot encodes the 'Type' column
    - Standardizes numerical features using StandardScaler
    - Returns: X, y, and scaler object
    """
    try:
        logging.info("Feature engineering started.")

        # Step 1: Feature and label selection
        features = [
            'Air_temperature_K', 
            'Process_temperature_K', 
            'Rotational_speed_rpm',
            'Torque_Nm', 
            'Tool_wear_min', 
            'Type'
        ]
        target = 'Machine failure'

        X = df[features]
        y = df[target]
        logging.info(f"Selected features: {features}")
        logging.info(f"Target column: '{target}'")

        # Step 2: One-hot encode 'Type'
        X = pd.get_dummies(X, columns=['Type'], prefix='Type')
        logging.info(f"One-hot encoded 'Type' column. Result columns: {list(X.columns)}")

        # Ensure all three possible types are represented (L, M, H)
        for col in ['Type_L', 'Type_M', 'Type_H']:
            if col not in X.columns:
                X[col] = 0
                logging.info(f"Added missing column: {col} with default 0s")

        # Step 3: Scale numeric features
        numerical_features = [
            'Air_temperature_K', 
            'Process_temperature_K',
            'Rotational_speed_rpm', 
            'Torque_Nm', 
            'Tool_wear_min'
        ]

        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        logging.info("Standard scaling applied to numerical features.")

        logging.info("Feature engineering completed successfully.")

        return X, y, scaler

    except Exception as e:
        logging.error("Exception occurred during feature engineering.")
        raise CustomException(e, sys)
