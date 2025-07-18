from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

from src.exception import CustomException
from src.logger import logging

def train_xgb_model(X_train, y_train, random_state=42):
    """
    Trains an XGBoost classifier and returns the trained model.
    """
    try:
        logging.info("Starting training of XGBoost model.")
        model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        logging.info("XGBoost model training complete.")
        return model
    except Exception as e:
        logging.error("Error occurred during model training.")
        raise CustomException(e, sys)

def evaluate_model(model, X_test, y_test, threshold=0.7):
    """
    Evaluates the model and returns a dictionary of key performance metrics.
    """
    try:
        logging.info("Evaluating model performance on test set.")
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
        }
        logging.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error("Error occurred during model evaluation.")
        raise CustomException(e, sys)
