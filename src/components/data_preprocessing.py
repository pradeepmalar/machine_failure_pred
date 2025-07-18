import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomException

def load_and_clean_data(filepath):
    """
    Loads machine failure dataset, cleans data, and returns DataFrame.
    Steps:
    - Loads CSV file
    - Standardizes column names
    - Drops duplicates
    - Handles missing values (impute or drop)
    - Logs every step
    """
    try:
        logging.info(f"Reading data from {filepath}")
        df = pd.read_csv(filepath)

        logging.info(f"Initial data shape: {df.shape}")

        # Standardize column names (specific to your dataset)
        rename_dict = {
            'Air temperature [K]': 'Air_temperature_K',
            'Process temperature [K]': 'Process_temperature_K',
            'Rotational speed [rpm]': 'Rotational_speed_rpm',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_wear_min'
        }
        df.rename(columns=rename_dict, inplace=True)
        logging.info(f"Renamed columns as: {rename_dict}")

        # Remove duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            df = df.drop_duplicates()
            logging.info(f"Dropped {n_duplicates} duplicate rows.")
        else:
            logging.info("No duplicate rows found.")

        # Handling missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            logging.info(f"Missing values found: {df.isnull().sum()}")
            # Impute numerical columns with mean
            num_cols = df.select_dtypes(include='number').columns
            for col in num_cols:
                if df[col].isnull().any():
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                    logging.info(f"Imputed missing values in {col} with mean: {mean_value}")
            # For categorical, impute with mode (if any)
            cat_cols = df.select_dtypes(include='object').columns
            for col in cat_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                    logging.info(f"Imputed missing values in {col} with mode: {mode_value}")
        else:
            logging.info("No missing values to handle.")

        # Log final shape
        logging.info(f"Data cleaning complete. Final shape: {df.shape}")

        return df

    except Exception as e:
        logging.error("Exception occurred during data preprocessing")
        raise CustomException(e, sys)
