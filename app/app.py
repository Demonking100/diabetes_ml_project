import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Load model
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
# Title
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient details below to predict diabetes risk.")

# Sidebar inputs (better UX)
st.sidebar.header("Patient Input")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 100)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 60.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

# Show inputs summary
st.subheader("Input Summary")
st.write({
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BP": bp,
    "Skin Thickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DPF": dpf,
    "Age": age
})

# Prediction
if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]

    st.subheader("Result")

    if prob > 0.6:
        st.error(f"⚠️ High Risk ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({(1-prob)*100:.2f}%)")

    st.caption("⚠️ This is only a prediction, not a medical diagnosis.")