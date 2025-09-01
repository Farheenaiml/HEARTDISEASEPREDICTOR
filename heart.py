import streamlit as st
import pandas as pd
import joblib
import time

# Load model and scaler
model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️", layout="wide")

# Custom CSS for gradient background, glassmorphism, and animation
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #222;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(130deg, #f5f7fa 0%, #a7c7dc 100%);
}
.glass-card {
    background: rgba(255,255,255,0.22);
    border-radius: 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09);
    backdrop-filter: blur(14px);
    border: 1.2px solid rgba(255,255,255,0.23);
    padding: 30px;
    margin-bottom: 36px;
}
h1, h2, h3 {
    color: #4f9a94 !important;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-weight: 600;
}
.stButton > button {
    background: linear-gradient(90deg, #ffdde1 0%, #a2d5f2 100%);
    color: #333;
    border-radius: 16px;
    font-weight: bold;
    height: 52px;
    box-shadow: 0 2px 8px rgba(160,160,160,0.1);
    border: none;
    transition: background 0.4s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #a2d5f2 0%, #ffdde1 100%);
    color: #006d77;
}
.heart {
  width: 56px;
  height: 56px;
  background: radial-gradient(circle at 60% 60%, #ff78b3 60%, #ffdde1 100%);
  position: relative;
  transform: rotate(-45deg);
  animation: beat 1.1s infinite;
}
.heart::before, .heart::after {
  content: "";
  width: 56px;
  height: 56px;
  background: radial-gradient(circle at 60% 60%, #ff78b3 60%, #ffdde1 100%);
  border-radius: 50%;
  position: absolute;
}
.heart::before { top: -28px; left: 0; }
.heart::after { left: 28px; top: 0; }
@keyframes beat {
  0%, 100% { transform: scale(1) rotate(-45deg); }
  50% { transform: scale(1.15) rotate(-45deg); }
}
::-webkit-scrollbar {
  width: 9px;
  background: #e3e3e3;
}
::-webkit-scrollbar-thumb {
  background: #b8d9fa;
  border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='glass-card' style='text-align: center; margin-top:25px;'>
    <h1>❤️ Heart Attack Risk Predictor ❤️</h1>
    <h3 style='color:#3ea6a2;'>💓 Stay Heart-Smart! Check Your Risk 💓</h3>
</div>
""", unsafe_allow_html=True)

# All User Inputs in a single vertical glass-card
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🩺 Personal & Health Information")

age = st.number_input("Age", min_value=18, max_value=90, value=30)
sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0,1])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 200, 150)
exang = st.selectbox("Exercise Induced Angina?", [0,1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment (0-2)", [0,1,2])
ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thalassemia (3,6,7)", [3,6,7])

st.markdown("</div>", unsafe_allow_html=True)

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Predict button and result in glass-card
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
if st.button("Predict"):
    st.write("Analyzing heart data...")
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.008)
        progress_bar.progress(percent + 1)

    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]],
                            columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("📝 Your Inputs")
    st.table(input_df.T.rename(columns={0: "Value"}))

    st.subheader("📊 Prediction Result")
    st.markdown("<div class='heart'></div>", unsafe_allow_html=True)
    if prediction == 0:
        st.success(f"✅ Low Risk — {prob:.1f}% probability of Heart Disease")
        st.info("Keep exercising & maintain a healthy lifestyle.")
    else:
        if prob < 50:
            st.warning(f"⚠️ Medium Risk — {prob:.1f}% probability")
            st.info("Consult a doctor for preventive measures.")
        else:
            st.error(f"❌ High Risk — {prob:.1f}% probability")
            st.info("Seek medical advice immediately!")
st.markdown("</div>", unsafe_allow_html=True)

# Heart health tips in glass-card
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
with st.expander("💡 Heart Health Tips"):
    st.write("""
    - Exercise regularly 🏃‍♂️
    - Eat balanced meals 🥗
    - Monitor blood pressure & cholesterol 🩸
    - Avoid smoking 🚭
    - Reduce stress 🧘
    - Regular health checkups 🏥
    """)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("""
<p style='text-align: center; color: #7d91a7; font-size: 17px;'>
    This is a demo model. Not a medical diagnosis.
</p>
""", unsafe_allow_html=True)
