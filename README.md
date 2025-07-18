Machine Failure Prediction (End-to-End ML Pipeline)
Overview
This project predicts machine failures using operational sensor data and the XGBoost classification algorithm. The pipeline is fully modularized for maintainability, reproducibility, and deployment readiness. Business impact and model interpretability are included.

Project Structure:

machine_failure_pred/
├── artifacts/            # Saved models, scalers, outputs
├── data/                 # Raw/source data (not tracked by git)
├── notebook/
│   └── machine_failure_analysis.ipynb  # Main E2E notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── tra     in_model.py
│   ├── predict.py
│   └── dashboard_utils.py
├── requirements.txt
├── README.md
└── .gitignore



Setup & Installation

1. Clone this repository
    git clone https://github.com/pradeepmalar/machine_failure_pred.git
    cd machine_failure_pred

2. Set up a virtual environment
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

4. Place data file
    Make sure your dataset (e.g., machine failure.csv) is in the data/ directory.


How to Run: 
Option 1: Run the End-to-End Notebook

1. Open notebook/machine_failure_analysis.ipynb in Jupyter or VS Code.
2. Run all cells to:
    Load and clean data
    Apply feature engineering
    Train and evaluate the model
    Visualize results and business impact
    Perform interactive predictions

Option 2: Use Individual Python Modules
    Import and use functions in your own scripts for batch or real-time inference.
    
    Example:
    
    from src.data_preprocessing import load_and_clean_data
    from src.feature_engineering import apply_feature_engineering
    from src.train_model import train_xgb_model, evaluate_model
    from src.predict import get_prediction

    df = load_and_clean_data("data/machine failure.csv")
    X, y, scaler = apply_feature_engineering(df)
    # ...train, split, evaluate as in the notebook


Pipeline Components
1. Data Preprocessing: Cleans input data, renames columns, handles missing values
2. Feature Engineering: One-hot encoding for Type, standardizes numeric features
3. Model Training: XGBoost classifier fit on train data
4. Evaluation: Reports Accuracy, Precision, Recall, F1-score, and AUC
5. Prediction: Clean, encode, and scale new samples in a single call
6. Dashboard: Visualization and business impact summary


Example Prediction (In Code):

params = {
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 308.6,
    "Rotational_speed_rpm": 1551,
    "Torque_Nm": 42.8,
    "Tool_wear_min": 0,
    "Type": "L"
}
prediction, probability = get_prediction(model, scaler, params, X_columns=X.columns)
print(f"Prediction: {prediction}, Probability: {probability:.2%}")


Business Impact:
1. Prevented Failures: $2,050,000
2. Maintenance Costs: $275,000
3. False Alarm Costs: $14,000
4. Missed Failure Costs: $1,000,000
5. Net Savings: $761,000
6. Annual Savings: $3,805,000
7. ROI: 7510.0%


Contributing
Pull requests and enhancements are welcome!
If submitting major changes, please open an issue first to discuss your suggestions.


Credits
Developed by Pradeep M.
Inspired by open-source end-to-end ML pipelines.