import os
import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
import joblib

class PredictPipeline:
    """
    Pipeline for making predictions using saved model and scaler artifacts,
    and strictly matching input feature order to training.
    """
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            # Load serialized model, scaler, and feature column order
            model = load_object("artifacts/model.pkl")
            scaler = load_object("artifacts/scaler.pkl")
            feature_columns = joblib.load("artifacts/feature_columns.pkl")

            # Ensure one-hot encoding columns present, then match exact order
            features = features.reindex(columns=feature_columns, fill_value=0)

            # Scale numeric columns only (get them from feature_columns or hardcoded)
            numeric_cols = [
                'Air_temperature_K', 'Process_temperature_K',
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'
            ]
            features[numeric_cols] = scaler.transform(features[numeric_cols])

            # Predict
            preds = model.predict(features)
            proba = model.predict_proba(features)[:, 1]
            return preds, proba

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Constructs a properly formatted DataFrame from user input for prediction.
    """

    def __init__(self, 
                 air_temperature_K: float,
                 process_temperature_K: float,
                 rotational_speed_rpm: float,
                 torque_Nm: float,
                 tool_wear_min: float,
                 M_type: str):  # Should be "L", "M", or "H"
        self.air_temperature_K = air_temperature_K
        self.process_temperature_K = process_temperature_K
        self.rotational_speed_rpm = rotational_speed_rpm
        self.torque_Nm = torque_Nm
        self.tool_wear_min = tool_wear_min
        self.M_type = M_type

    def get_data_as_dataframe(self):
        try:
            # Create a base dict/DataFrame for the input row
            df = pd.DataFrame([{
                "Air_temperature_K": self.air_temperature_K,
                "Process_temperature_K": self.process_temperature_K,
                "Rotational_speed_rpm": self.rotational_speed_rpm,
                "Torque_Nm": self.torque_Nm,
                "Tool_wear_min": self.tool_wear_min,
                "Type": self.M_type
            }])
            # One-hot encode the 'Type' column
            df = pd.get_dummies(df, columns=['Type'], prefix='Type')
            for col in ['Type_L', 'Type_M', 'Type_H']:
                if col not in df.columns:
                    df[col] = 0
            return df
        except Exception as e:
            raise CustomException(e, sys)
