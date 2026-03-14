import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Bio-Tech AI | Fitness", page_icon="🤖", layout="wide")

# 2. تصميم CSS متقدم (Neon & High-Tech Look)
st.markdown("""
    <style>
    /* خلفية داكنة بتقنية الـ Glassmorphism */
    .stApp {
        background: radial-gradient(circle, #1a1a2e 0%, #0f0f1a 100%);
        color: #e0e0e0;
    }
    
    /* تنسيق كروت المدخلات لتشبه شاشات المختبر */
    div[data-baseweb="input"], div[data-baseweb="select"], .stSlider {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* تأثير النيون على العناوين */
    h1, h2 {
        color: #00d4ff;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
        font-family: 'Courier New', Courier, monospace;
    }

    /* زر التحليل بتصميم "Cyber" */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #00d4ff, #005f73);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        letter-spacing: 2px;
        box-shadow: 0 0 15px #00d4ff;
        width: 100%;
        transition: 0.5s;
    }

    div.stButton > button:first-child:hover {
        box-shadow: 0 0 30px #00d4ff;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# تحميل الملفات
@st.cache_resource
def load_assets():
    model = joblib.load('body_performance_model_compressed.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ فشل في تحميل الموديل. تأكد من أسماء الملفات على GitHub.")

# --- الواجهة ---
st.markdown("<h1 style='text-align: center;'>🧬 BIO-METRIC AI ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Advanced Machine Learning for Athletic Classification</p>", unsafe_allow_html=True)

# تقسيم الشاشة لـ 3 أقسام تشبه لوحة التحكم (Dashboard)
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("### 🛠 Core Metrics")
    age = st.number_input("Biological Age", 10, 80, 25)
    gender = st.selectbox("Gender Orientation", ["Male", "Female"])
    height = st.number_input("Stature (cm)", 120.0, 220.0, 175.0)
    weight = st.number_input("Mass (kg)", 30.0, 200.0, 75.0)

with col2:
    st.markdown("### 📊 Vital Signs")
    body_fat = st.slider("Adipose Index (Fat %)", 5.0, 50.0, 18.0)
    sys_bp = st.number_input("Systolic Pressure", 80, 200, 120)
    dia_bp = st.number_input("Diastolic Pressure", 40, 120, 80)
    grip = st.number_input("Neural Grip Force", 0.0, 100.0, 45.0)

with col3:
    st.markdown("### ⚡ Kinetic Tests")
    sit_ups = st.number_input("Core Stability (Sit-ups)", 0, 100, 40)
    jump = st.number_input("Explosive Power (Jump cm)", 0, 300, 210)
    bend = st.number_input("Flexibility Index", -20.0, 50.0, 15.0)

st.markdown("<br>", unsafe_allow_html=True)

# تفعيل التحليل
if st.button("INITIATE NEURAL ANALYSIS"):
    # تجهيز البيانات
    g_val = 0 if gender == "Male" else 1
    input_data = np.array([[age, g_val, height, weight, body_fat, dia_bp, sys_bp, grip, bend, sit_ups, jump]])
    
    # المعالجة
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # عرض النتيجة بتصميم تقني
    st.markdown("---")
    res_colors = {"A": "#00ffcc", "B": "#00d4ff", "C": "#ffcc00", "D": "#ff4b4b"}
    final_color = res_colors.get(prediction, "#ffffff")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
            <div style="border: 2px solid {final_color}; padding: 20px; border-radius: 15px; text-align: center; background: rgba(0,0,0,0.3);">
                <h2 style="color: white; text-shadow: none;">RESULT CLASSIFICATION</h2>
                <h1 style="font-size: 100px; color: {final_color}; margin: 0;">{prediction}</h1>
            </div>
        """, unsafe_allow_html=True)
        if prediction == "A": st.balloons()
