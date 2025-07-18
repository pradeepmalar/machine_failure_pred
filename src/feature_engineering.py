# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def apply_feature_engineering(df):
    """
    Takes a cleaned DataFrame, applies feature selection, encoding, and scaling.
    Returns feature matrix X, target vector y, and the scaler object.
    """
    # Define which features to use
    features = [
        'Air_temperature_K',
        'Process_temperature_K',
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min',
        'Type'
    ]

    # Select features and define target
    X = df[features]
    y = df['Machine failure']

    # One-hot encode the categorical "Type" feature
    X = pd.get_dummies(X, columns=['Type'], prefix='Type')

    # List numerical features to be standardized
    numerical_features = [
        'Air_temperature_K',
        'Process_temperature_K',
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min'
    ]

    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y, scaler
