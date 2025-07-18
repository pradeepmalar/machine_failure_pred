import os
import sys
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save any Python object using dill serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a serialized Python object (previously saved with dill).
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Optional: add other utility functions as needed for your ML workflow
