import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging

def get_prediction(model, scaler, params, X_columns=None, threshold=0.7):
    """
    Given a trained model, fitted scaler, user parameters as a dict, and the training columns,
    return a prediction ("Failure"/"No Failure") and failure probability.
    """
    try:
        logging.info("Starting prediction step.")

        # Convert input parameters to DataFrame
        input_df = pd.DataFrame([params])

        # One-hot encode 'Type' (ensure alignment)
        input_df = pd.get_dummies(input_df, columns=['Type'], prefix='Type')

        # Guarantee all type columns are present
        for col in ['Type_L', 'Type_M', 'Type_H']:
            if col not in input_df.columns:
                input_df[col] = 0

        # Respect the original feature order
        feature_order = X_columns if X_columns is not None else [
            'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
            'Torque_Nm', 'Tool_wear_min', 'Type_L', 'Type_M', 'Type_H'
        ]
        input_df = input_df.reindex(columns=feature_order, fill_value=0)

        # Scale numeric features
        numeric_cols = [
            'Air_temperature_K', 'Process_temperature_K',
            'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'
        ]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict probability of failure
        failure_proba = model.predict_proba(input_df)[:, 1][0]
        prediction = "Failure" if failure_proba >= threshold else "No Failure"

        logging.info(f"Prediction complete. Result: {prediction} | Probability: {failure_proba:.3f}")

        return prediction, failure_proba

    except Exception as e:
        logging.error("Error during prediction.")
        raise CustomException(e, sys)
