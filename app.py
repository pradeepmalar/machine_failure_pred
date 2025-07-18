# app.py

import sys
import os
import pandas as pd
import streamlit as st

# Ensure src/ is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Machine Failure Prediction", layout="centered")

st.title("⚙️ Machine Failure Prediction Dashboard")
st.markdown("---")

st.subheader("🔧 Enter Input Parameters")

# Input fields
air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=298.0)
proc_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=500.0, value=308.6)
rpm = st.number_input("Rotational speed [rpm]", min_value=100.0, max_value=3000.0, value=1551.0)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=42.8)
tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=250.0, value=0.0)

type_input = st.selectbox("Product Type", options=["L", "M", "H"])

st.markdown("---")

# When Predict button is clicked
if st.button("🔍 Predict Failure"):
    try:
        # Capture input
        user_input = CustomData(
            air_temperature_K=air_temp,
            process_temperature_K=proc_temp,
            rotational_speed_rpm=rpm,
            torque_Nm=torque,
            tool_wear_min=tool_wear,
            M_type=type_input
        )

        # Convert to DataFrame
        input_df = user_input.get_data_as_dataframe()

        # Predict
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(input_df)

        # Display result
        st.subheader("📈 Prediction Result")
        result = "⚠️ Machine Failure" if prediction[0] == 1 else "✅ No Failure Detected"
        st.markdown(f"**Result:** {result}")
        st.markdown(f"**Failure Probability:** `{probability[0]:.2%}`")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
