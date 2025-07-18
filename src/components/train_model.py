# src/train_model.py

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_xgb_model(X_train, y_train, random_state=42):
    """
    Trains an XGBoost classifier on the provided training data.
    Returns the trained model.
    """
    model = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.7):
    """
    Evaluates the given model and returns a dictionary of performance metrics.
    The threshold determines the classification cutoff for positive class.
    """
    # Predicted probabilities for positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }
    return metrics
