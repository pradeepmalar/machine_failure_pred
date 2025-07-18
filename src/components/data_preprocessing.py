# src/data_preprocessing.py

import pandas as pd

def load_and_clean_data(filepath):
    """
    Loads the machine failure dataset, renames columns for consistency,
    and returns a cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min'
    })
    # If you have any additional cleaning steps (missing value handling, etc), add here.
    return df
