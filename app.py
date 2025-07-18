import streamlit as st
import pandas as pd
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import apply_feature_engineering
from src.train_model import train_xgb_model, evaluate_model
from src.predict import get_prediction

st.title("Machine Failure Prediction Dashboard")

# Step 1: Data loading
@st.cache_data
def load_data():
    df = load_and_clean_data("data/machine failure.csv")
    return df

df = load_data()
st.subheader("Sample Data")
st.write(df.head())

# Step 2: Feature engineering and train model
X, y, scaler = apply_feature_engineering(df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_xgb_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

st.subheader("Model Performance")
st.write(metrics)

# Step 3: Interactive Prediction
st.subheader("Try a Prediction")
air_temp = st.number_input("Air temperature (K)", value=298.1)
proc_temp = st.number_input("Process temperature (K)", value=308.6)
rpm = st.number_input("Rotational speed (rpm)", value=1551)
torque = st.number_input("Torque (Nm)", value=42.8)
tool_wear = st.number_input("Tool wear (min)", value=0)
type_input = st.selectbox("Type", options=["L", "M", "H"])

if st.button("Predict Failure"):
    params = {
        "Air_temperature_K": air_temp,
        "Process_temperature_K": proc_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear,
        "Type": type_input
    }
    prediction, probability = get_prediction(model, scaler, params, X_columns=X.columns)
    st.write(f"Prediction: **{prediction}**")
    st.write(f"Failure Probability: **{probability:.2%}**")
