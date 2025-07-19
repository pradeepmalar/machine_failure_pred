💥 Machine Failure Prediction

An end-to-end modular ML pipeline with Streamlit dashboard for real-time machine failure detection.

📁 Project Structure

machine_failure_pred/

├── app.py                             # Streamlit web dashboard

├── src/                               # Core code modules

│   ├── components/                    # Feature engineering & training components

│   │   ├── data_preprocessing.py

│   │   ├── feature_engineering.py

│   │   ├── train_model.py

│   │   ├── predict.py

│   │   └── dashboard_utils.py

│   ├── pipeline/                      # Training and prediction orchestration

│   │   ├── training_pipeline.py

│   │   └── prediction_pipeline.py

│   ├── utils.py                       # Generic save/load functions

│   ├── exception.py                   # Custom exception handling

│   ├── logger.py                      # Logging setup

│   └── __init__.py

├── data/

│   └── machine failure.csv            # Input dataset

├── artifacts/                         # Saved model, scaler, and feature_columns

│   ├── model.pkl

│   ├── scaler.pkl

│   └── feature_columns.pkl

├── logs/                              # Logged pipeline/debug info

├── notebook/

│   └── machine_failure_analysis.ipynb # EDA & experimentation notebook

├── requirements.txt

├── setup.py

└── README.md

🚀 Features
•	✅ Modular training and prediction pipeline (src/pipeline/)
•	✅ Streamlit dashboard for real-time predictions (app.py)
•	✅ Custom exceptions, centralized logging
•	✅ Artifacts saved for reproducibility (artifacts/)
•	✅ Visual metric dashboard + business impact summaries
•	✅ Handles one-hot and scaler alignment under the hood
📦 Setup Instructions
1. Clone the Repo
git clone https://github.com/YOUR_USERNAME/machine_failure_pred.git
cd machine_failure_pred

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Prepare the Data
Make sure your dataset is present at:
data/machine failure.csv

🛠️ Run the Training Pipeline
This trains your model, evaluates it, and saves:
•	model.pkl
•	scaler.pkl
•	feature_columns.pkl
python src/pipeline/training_pipeline.py

Artifacts are saved in the artifacts/ directory.
💻 Run the Streamlit Dashboard
streamlit run app.py

Then visit: http://localhost:8501/
Use the interactive fields to predict machine failures in real-time!
✅ Handles:
•	Feature scaling
•	Type encoding alignment
•	Failure probability prediction
📊 Sample Dashboard Output
•	Inputs: Sensor & process variables
•	Output: Category (❌ Failure | ✅ No Failure), probability, and performance metrics
🧪 Example Notebook Workflow
Run freely in notebook/machine_failure_analysis.ipynb:
•	Load & clean data
•	Train model with updated features
•	Test single prediction interactively
🏁 Sample Usage (Python)
from src.utils import load_object
from src.components.predict import get_prediction

model = load_object("artifacts/model.pkl")
scaler = load_object("artifacts/scaler.pkl")

params = {
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 308.6,
    "Rotational_speed_rpm": 1551,
    "Torque_Nm": 42.8,
    "Tool_wear_min": 0,
    "Type": "L"
}

prediction, probability = get_prediction(model, scaler, params)

📈 Business Impact (Sample)
Metric	Value
Prevented Failures	$2,050,000
Maintenance Cost	$275,000
False Alarm Cost	$14,000
Missed Failure Cost	$1,000,000
Annual Net Savings	$3,805,000
ROI	7510%

✅ Requirements
•	Python 3.7+
•	pandas, numpy, scikit-learn
•	xgboost
•	streamlit
•	dill
•	joblib
📌 All listed in requirements.txt
🤝 Contributing
Pull requests and issues are welcome — this project follows modular clean-code principles and is built for maintainability and educational growth.

