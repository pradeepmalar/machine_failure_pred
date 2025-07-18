# src/utils.py

import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV

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
    Load a serialized Python object (previously saved with 'dill').
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV, returning a dict with test accuracy scores.
    Prints detailed training and testing metrics for each model.
    """
    try:
        report = {}
        for model_name, model in models.items():
            try:
                print(f"\n🔍 Training model: {model_name}")
                hyperparams = param.get(model_name, {})

                gs = GridSearchCV(
                    model,
                    hyperparams,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1,
                    error_score='raise',
                    refit=True
                )

                # LightGBM compatibility (optional)
                if "LGBM" in model_name:
                    if not isinstance(X_train, pd.DataFrame):
                        X_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(X_train.shape[1])])
                    if not isinstance(X_test, pd.DataFrame):
                        X_test = pd.DataFrame(X_test, columns=X_train.columns)

                gs.fit(X_train, y_train)

                # Ensure estimator is fitted
                try:
                    _ = gs.best_estimator_.predict(X_train[:5])
                    best_model = gs.best_estimator_
                except Exception as pred_err:
                    raise Exception(f"{model_name} is not fitted. Error: {pred_err}")

                # Predict train/test
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Train metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_cm = confusion_matrix(y_train, y_train_pred)

                # Test metrics
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_cm = confusion_matrix(y_test, y_test_pred)

                print(f"\n📊 {model_name} → Training Metrics:")
                print(f"Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}, MSE: {train_mse:.4f}")
                print("Confusion Matrix:\n", train_cm)

                print(f"\n📊 {model_name} → Testing Metrics:")
                print(f"Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, MSE: {test_mse:.4f}")
                print("Confusion Matrix:\n", test_cm)

                report[model_name] = test_acc

            except Exception as model_err:
                print(f"⚠️ Skipping model '{model_name}' due to error:\n{model_err}\n")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
