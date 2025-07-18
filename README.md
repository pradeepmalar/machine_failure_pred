Machine Failure Prediction (End-to-End ML Pipeline & Dashboard)
Predict machine failures using sensor and process data with an XGBoost model. This repository features a modular Python codebase, a Streamlit web dashboard, and a Jupyter notebook for exploratory analysis.
Project Structure
machine_failure_pred/
├── app.py                        # Streamlit web dashboard
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── dashboard_utils.py
├── notebook/
│   └── machine_failure_analysis.ipynb
├── data/
│   └── machine failure.csv
├── artifacts/                    # (optional: for saved models, etc)
├── requirements.txt
├── README.md
└── .gitignore

Setup Instructions
1.	Clone this repository
git clone https://github.com/pradeepmalar/machine_failure_pred.git
cd machine_failure_pred

2.	(Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3.	Install dependencies
pip install -r requirements.txt

4.	Ensure data is available
o	Place machine failure.csv in the data/ directory.
How to Run the Project
A. Interactive Notebook Workflow
Navigate to the notebook/ directory and open the notebook:
jupyter notebook notebook/machine_failure_analysis.ipynb

•	Run all cells for end-to-end data cleaning, model training, evaluation, and interactive prediction.
•	All code uses modularized functions from the src/ folder.
B. Web Dashboard (Streamlit)
You can interact with the model and make live predictions in your browser.
Start the dashboard:
streamlit run app.py

•	Enter sensor data in the sidebar to get instant predictions.
•	View model performance and business impact directly on the dashboard.
To stop the dashboard:
•	Focus the terminal and press Ctrl + C.
Key Modules
•	src/data_preprocessing.py: Load and clean datasets
•	src/feature_engineering.py: Feature engineering and scaling
•	src/train_model.py: Model training and evaluation
•	src/predict.py: Inference for new samples
•	src/dashboard_utils.py: Visualization and business metrics
Example Prediction in Python
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

Business Impact
Example financial outcomes by using this predictive pipeline:
Item	Value
Prevented Failures	$2,050,000
Maintenance Costs	$275,000
False Alarm Costs	$14,000
Missed Failure Cost	$1,000,000
Annual Savings	$3,805,000
ROI	7510.0%

Contributing
•	Pull requests, ideas, and improvements are welcome!
•	Please open an issue for major changes.


Quick Start for Dashboard:
streamlit run app.py
