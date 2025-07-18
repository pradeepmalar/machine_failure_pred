# src/predict.py

import pandas as pd
import numpy as np

def get_prediction(model, scaler, params, X_columns=None, threshold=0.7):
    """
    Accepts a trained model, scaler, input parameters (dict), and original X columns.
    Returns the prediction label ("Failure" or "No Failure") and the failure probability.
    
    - model: trained classifier with predict_proba method
    - scaler: fitted StandardScaler (from feature_engineering.py)
    - params: dict with keys matching original features, including 'Type'
    - X_columns: predicted set of features as used in training (for order, dummies)
    - threshold: probability cutoff for positive prediction
    """

    # Create DataFrame from user input
    df_input = pd.DataFrame([params])

    # One-hot encoding for 'Type', same as training
    df_input = pd.get_dummies(df_input, columns=['Type'], prefix='Type')
    # Ensure all dummy columns present
    for col in ['Type_L', 'Type_M', 'Type_H']:
        if col not in df_input.columns:
            df_input[col] = 0  # Add missing as zero

    # Order columns as in training data
    if X_columns is not None:
        # For batch or new data where columns may vary
        df_input = df_input.reindex(columns=X_columns, fill_value=0)
    else:
        # Use standard training order
        feature_order = [
            'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
            'Torque_Nm', 'Tool_wear_min', 'Type_L', 'Type_M', 'Type_H'
        ]
        df_input = df_input.reindex(columns=feature_order, fill_value=0)

    # Scale numeric columns
    numeric_cols = [
        'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
        'Torque_Nm', 'Tool_wear_min'
    ]
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # Predict probability
    failure_proba = model.predict_proba(df_input)[:, 1][0]
    prediction = "Failure" if failure_proba >= threshold else "No Failure"

    return prediction, failure_proba
