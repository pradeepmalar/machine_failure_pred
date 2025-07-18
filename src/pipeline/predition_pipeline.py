# src/pipeline/prediction_pipeline.py

import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            scaler_path = os.path.join("artifacts", "scaler.pkl")

            # Load serialized model and scaler
            model = load_object(model_path)
            scaler = load_object(scaler_path)

            # Re-order columns just in case
            feature_order = [
                'Air_temperature_K',
                'Process_temperature_K',
                'Rotational_speed_rpm',
                'Torque_Nm',
                'Tool_wear_min',
                'Type_L', 'Type_M', 'Type_H'
            ]
            features = features.reindex(columns=feature_order, fill_value=0)

            # Scale numerical features
            numeric_cols = [
                'Air_temperature_K',
                'Process_temperature_K',
                'Rotational_speed_rpm',
                'Torque_Nm',
                'Tool_wear_min'
            ]
            features[numeric_cols] = scaler.transform(features[numeric_cols])

            preds = model.predict(features)
            proba = model.predict_proba(features)[:, 1]

            return preds, proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Accepts raw input (e.g., from webapp) and converts it into the correct DataFrame format.
    """
    def __init__(self, 
                 air_temperature_K: float,
                 process_temperature_K: float,
                 rotational_speed_rpm: float,
                 torque_Nm: float,
                 tool_wear_min: float,
                 M_type: str):
        self.air_temperature_K = air_temperature_K
        self.process_temperature_K = process_temperature_K
        self.rotational_speed_rpm = rotational_speed_rpm
        self.torque_Nm = torque_Nm
        self.tool_wear_min = tool_wear_min
        self.M_type = M_type  # expects values: 'L', 'M', or 'H'

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([{
                "Air_temperature_K": self.air_temperature_K,
                "Process_temperature_K": self.process_temperature_K,
                "Rotational_speed_rpm": self.rotational_speed_rpm,
                "Torque_Nm": self.torque_Nm,
                "Tool_wear_min": self.tool_wear_min,
                "Type": self.M_type
            }])

            # One-hot encode "Type"
            df = pd.get_dummies(df, columns=['Type'], prefix='Type')

            # Ensure all expected one-hot columns exist
            for col in ['Type_L', 'Type_M', 'Type_H']:
                if col not in df.columns:
                    df[col] = 0

            return df

        except Exception as e:
            raise CustomException(e, sys)
