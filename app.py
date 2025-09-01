import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')

# App configuration
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: red;'>Heart Attack Risk Predictor ❤️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter your health details below to predict the risk of heart disease.</p>", unsafe_allow_html=True)
st.write("---")

# User Input Section
st.subheader("🩺 Personal & Health Information")

age = st.number_input("Age", min_value=18, max_value=90, value=30)
sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0,1])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=200, value=150)
exang = st.selectbox("Exercise Induced Angina?", [0,1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of ST segment (0-2)", [0,1,2])
ca = st.number_input("Number of major vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia (3,6,7)", [3,6,7])

# Feature columns
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                              thalach, exang, oldpeak, slope, ca, thal]],
                            columns=feature_cols)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100  # probability for "Heart Disease"
    
    # Display result with colors
    if prediction == 0:
        st.success("Prediction: Low Risk ✅ No Heart Disease")
        st.info("Maintain a healthy lifestyle and regular checkups.")
    else:
        if prob < 50:
            st.warning(f"Prediction: Medium Risk ⚠️ Probability: {prob:.1f}%")
            st.info("Consult a doctor for preventive measures.")
        else:
            st.error(f"Prediction: High Risk ❌ Probability: {prob:.1f}%")
            st.info("Please seek medical advice immediately!")

# Footer
st.write("---")
st.markdown("<p style='text-align: center; color: grey;'>This is a predictive demo model. Not a medical diagnosis.</p>", unsafe_allow_html=True)
