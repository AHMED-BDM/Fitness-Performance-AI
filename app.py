import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Fitness AI Pro", page_icon="💪", layout="wide")

# 2. إضافة CSS مخصص لتغيير الخلفية والخطوط والألوان
st.markdown("""
    <style>
    /* تغيير خلفية التطبيق */
    .stApp {
        background: linear-gradient(to right, #1e1e2f, #2d2d44);
        color: white;
    }
    /* تنسيق كروت المدخلات */
    .stNumberInput, .stSlider {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    /* تنسيق زر التحليل */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff2b2b;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    model = joblib.load('body_performance_model_compressed.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# العنوان الرئيسي
st.title("🏋️‍♂️ AI Body Performance Analyzer")
st.markdown("---")

# 3. تنظيم المدخلات في أعمدة (Layout)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    age = st.number_input("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 120, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)

with col2:
    st.subheader("📊 Measurements")
    body_fat = st.slider("Body Fat %", 5.0, 50.0, 20.0)
    diastolic = st.number_input("Diastolic BP", 40, 120, 80)
    systolic = st.number_input("Systolic BP", 80, 200, 120)
    grip_force = st.number_input("Grip Force", 0.0, 100.0, 40.0)

with col3:
    st.subheader("🏃 Performance")
    sit_ups = st.number_input("Sit-ups Count", 0, 100, 30)
    broad_jump = st.number_input("Broad Jump (cm)", 0, 300, 180)
    sit_bend = st.number_input("Sit & Bend forward (cm)", -20.0, 50.0, 15.0)
    running_400m = st.number_input("400m Run (sec)", 30.0, 200.0, 70.0)

st.markdown("---")

# 4. المعالجة والتوقع
gender_val = 0 if gender == "Male" else 1

if st.button("🚀 Analyze My Performance"):
    # تجهيز البيانات بنفس ترتيب التدريب
    features = np.array([[age, gender_val, height, weight, body_fat, diastolic, 
                          systolic, grip_force, sit_bend, sit_ups, broad_jump]])
    
    # التحجيم
    features_scaled = scaler.transform(features)
    
    # التوقع
    prediction = model.predict(features_scaled)[0]
    
    # عرض النتيجة بشكل مبهر
    st.markdown(f"<h2 style='text-align: center;'>Your Fitness Class is:</h2>", unsafe_allow_html=True)
    
    # تلوين النتيجة حسب الفئة
    colors = {"A": "#28a745", "B": "#ffc107", "C": "#fd7e14", "D": "#dc3545"}
    res_color = colors.get(prediction, "#ffffff")
    
    st.markdown(f"<h1 style='text-align: center; color: {res_color}; font-size: 80px;'>{prediction}</h1>", unsafe_allow_html=True)
    
    if prediction == "A":
        st.balloons()
        st.success("Excellent! You are in peak physical condition.")
    elif prediction == "B":
        st.info("Great job! You are above average.")
