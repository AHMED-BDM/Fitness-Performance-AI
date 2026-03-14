import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="BIO-TECH AI | Fitness Pro", page_icon="🤖", layout="wide")

# 2. تصميم CSS المتقدم (Cyberpunk Style)
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle, #0d1117 0%, #010409 100%);
        color: #e6edf3;
    }
    div[data-baseweb="input"], div[data-baseweb="select"], .stSlider {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    h1 {
        color: #58a6ff;
        text-shadow: 0 0 15px rgba(88, 166, 255, 0.5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
        transition: 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(46, 160, 67, 0.6);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    model = joblib.load('body_performance_model_compressed.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

# 4. واجهة المستخدم (Input Form)
st.markdown("<h1 style='text-align: center;'>🧬 NEURAL FITNESS ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>AI-Powered Body Performance Classification</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 👤 Bio-Data")
    age = st.number_input("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 120.0, 220.0, 175.0)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)

with col2:
    st.markdown("### 🩺 Vitals")
    body_fat = st.slider("Body Fat %", 5.0, 50.0, 18.0)
    dia_bp = st.number_input("Diastolic BP", 40, 120, 80)
    sys_bp = st.number_input("Systolic BP", 80, 200, 120)
    grip = st.number_input("Grip Force", 0.0, 100.0, 45.0)

with col3:
    st.markdown("### ⚡ Performance")
    sit_bend = st.number_input("Sit & Bend (cm)", -20.0, 50.0, 15.0)
    sit_ups = st.number_input("Sit-ups Count", 0, 100, 40)
    jump = st.number_input("Broad Jump (cm)", 0, 300, 210)

st.markdown("<br>", unsafe_allow_html=True)

# 5. التوقع (Prediction Logic)
if st.button("EXECUTE ANALYSIS"):
    # تحويل الجنس لرقم
    g_val = 0 if gender == "Male" else 1
    
    # حساب العمود الـ 12 (BMI) - تأكد أنه هو العمود المطلوب في كولاب
    bmi = weight / ((height / 100) ** 2)
    
    # تجميع البيانات بنفس ترتيب كولاب (12 عمود)
    # ملاحظة: الترتيب هنا حيوي جداً لنجاح التوقع
    input_data = np.array([[
        age, g_val, height, weight, body_fat, 
        dia_bp, sys_bp, grip, sit_bend, sit_ups, jump, bmi
    ]])
    
    # تنفيذ التحجيم والتوقع
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # عرض النتيجة
        st.markdown("---")
        colors = {"A": "#238636", "B": "#1f6feb", "C": "#d29922", "D": "#f85149"}
        res_color = colors.get(prediction, "#ffffff")
        
        st.markdown(f"""
            <div style="border: 1px solid {res_color}; padding: 30px; border-radius: 15px; text-align: center; background: rgba(0,0,0,0.2);">
                <h2 style="color: #8b949e; margin-bottom: 5px;">CLASSIFICATION RESULT</h2>
                <h1 style="font-size: 100px; color: {res_color}; margin: 0; padding: 0;">{prediction}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if prediction == "A":
            st.balloons()
            st.success("Performance Status: ELITE")
        elif prediction == "D":
            st.warning("Performance Status: NEEDS IMPROVEMENT")
            
    except ValueError as e:
        st.error(f"Input dimension mismatch: {e}")
        st.info("Check if your model expects 12 features including BMI.")
