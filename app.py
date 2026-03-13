import streamlit as st
import joblib
import pandas as pd

# تحميل الموديل والميزان
model = joblib.load('body_performance_model_compressed.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Body Performance AI", layout="wide")
st.title("🏋️‍♂️ AI Body Performance Predictor")
st.markdown("---")

# تصميم الواجهة بشكل احترافي
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 10, 80, 25)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    height = st.number_input("Height (cm)", 120.0, 220.0, 175.0)
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
with col2:
    fat = st.number_input("Body Fat %", 5.0, 50.0, 15.0)
    sit_ups = st.number_input("Sit-ups Count", 0, 100, 40)
    bend = st.number_input("Sit and Bend (cm)", -10.0, 40.0, 15.0)
    jump = st.number_input("Broad Jump (cm)", 0.0, 350.0, 200.0)

if st.button("Analyze My Fitness", use_container_width=True):
    bmi = weight / ((height / 100) ** 2)
    # ترتيب الأعمدة الـ 12 كما في التدريب
    input_data = pd.DataFrame([[age, gender, height, weight, fat, 80, 120, 45, bend, sit_ups, jump, bmi]],
                              columns=['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
                                       'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm',
                                       'sit-ups counts', 'broad jump_cm', 'BMI'])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    classes = ['A (Excellent)', 'B (Good)', 'C (Average)', 'D (Poor)']

    st.balloons()
    st.markdown(f"### Result: **{classes[prediction[0]]}**")
