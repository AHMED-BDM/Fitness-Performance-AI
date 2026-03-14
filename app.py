import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Bio-Tech AI | Fitness", page_icon="🤖", layout="wide")

# 2. تصميم CSS "المستقبلي" القوي
st.markdown("""
    <style>
    /* تغيير الخلفية لكل طبقات التطبيق */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    /* تحسين النصوص وجعلها نيون */
    h1 {
        color: #38bdf8 !important;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
    
    h3 {
        color: #94a3b8 !important;
        border-bottom: 2px solid #38bdf8;
        padding-bottom: 10px;
    }

    /* تنسيق كروت المدخلات لتشبه أجهزة القياس */
    .stNumberInput, .stSelectbox, div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* تنسيق زر التحليل (Cyber Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-size: 22px !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 10px 20px rgba(14, 165, 233, 0.3) !important;
        transition: 0.4s all ease-in-out !important;
    }

    div.stButton > button:hover {
        box-shadow: 0 0 40px rgba(34, 197, 94, 0.5) !important;
        transform: scale(1.02) !important;
    }

    /* إخفاء القوائم غير الضرورية لشكل أنظف */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# تحميل الموديل والميزان
@st.cache_resource
def load_assets():
    model = joblib.load('body_performance_model_compressed.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- الواجهة ---
st.markdown("<h1 style='text-align: center;'>🧬 NEURAL BODY ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem;'>Deep Intelligence for Human Biometrics</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# تقسيم الواجهة (Dashboard Look)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 👤 BIO-DATA")
    age = st.number_input("Biological Age", 10, 80, 25)
    gender = st.selectbox("Gender Orientation", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    height = st.number_input("Stature (cm)", 120.0, 220.0, 175.0)

with col2:
    st.markdown("### 📊 MASS & FAT")
    weight = st.number_input("Body Mass (kg)", 30.0, 150.0, 70.0)
    fat = st.number_input("Adipose Index (Fat %)", 5.0, 50.0, 15.0)
    st.info("💡 AI: Vital signs are set to optimized defaults.")

with col3:
    st.markdown("### ⚡ CAPACITY")
    sit_ups = st.number_input("Core Stability (Sit-ups)", 0, 100, 40)
    bend = st.number_input("Flexibility (cm)", -10.0, 40.0, 15.0)
    jump = st.number_input("Explosive Power (cm)", 0.0, 350.0, 200.0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("RUN NEURAL INFERENCE"):
    # الحسابات
    bmi = weight / ((height / 100) ** 2)
    
    # الترتيب الصحيح للأعمدة الـ 12
    input_data = pd.DataFrame([[age, gender, height, weight, fat, 80, 120, 45, bend, sit_ups, jump, bmi]],
                              columns=['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 
                                       'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm', 
                                       'sit-ups counts', 'broad jump_cm', 'BMI'])

    # المعالجة والتوقع
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    # النتائج
    classes = ['A (Elite Performance)', 'B (Good Condition)', 'C (Average)', 'D (Needs Improvement)']
    colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444'] 
    
    res_text = classes[prediction[0]]
    res_color = colors[prediction[0]]

    st.balloons()
    
    st.markdown(f"""
        <div style="border: 2px solid {res_color}; padding: 40px; border-radius: 20px; text-align: center; background: rgba(15, 23, 42, 0.8); box-shadow: 0 0 50px {res_color}33;">
            <h2 style="color: #94a3b8; font-size: 20px;">SYSTEM CLASSIFICATION</h2>
            <h1 style="color: {res_color}; font-size: 60px; margin: 10px 0;">{res_text}</h1>
            <p style="color: #64748b;">Recommendation: Continue AI-guided physical optimization.</p>
        </div>
    """, unsafe_allow_html=True)
