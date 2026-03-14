import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Bio-Tech AI | Fitness", page_icon="🤖", layout="wide")

# 2. تصميم CSS متطور (AI & Fitness Fusion)
st.markdown("""
    <style>
    /* خلفية داكنة بتقنية الـ Glassmorphism */
    .stApp {
        background: radial-gradient(circle, #0d1117 0%, #010409 100%);
        color: #e6edf3;
    }
    
    /* تنسيق كروت المدخلات */
    div[data-baseweb="input"], div[data-baseweb="select"], .stNumberInput {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }

    /* تأثير النيون على العناوين */
    h1 {
        color: #58a6ff;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.5);
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 2px;
    }

    /* زر التحليل بتصميم Cyber */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #1f6feb, #238636);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(31, 111, 235, 0.4);
        width: 100%;
        transition: 0.5s;
    }

    div.stButton > button:first-child:hover {
        box-shadow: 0 0 30px rgba(46, 160, 67, 0.6);
        transform: scale(1.01);
    }
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
st.markdown("<p style='text-align: center; color: #8b949e;'>AI Inference Engine for Human Performance</p>", unsafe_allow_html=True)
st.markdown("---")

# تقسيم الواجهة لثلاثة أعمدة (Dashboard Look)
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("### 👤 Bio-Metrics")
    age = st.number_input("Biological Age", 10, 80, 25)
    gender = st.selectbox("Gender Orientation", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    height = st.number_input("Stature (cm)", 120.0, 220.0, 175.0)

with col2:
    st.markdown("### 📊 Composition")
    weight = st.number_input("Body Mass (kg)", 30.0, 150.0, 70.0)
    fat = st.number_input("Adipose Index (Fat %)", 5.0, 50.0, 15.0)
    # قيم افتراضية للضغط وقوة القبضة لضمان اكتمال الـ 12 عمود
    st.caption("Standard vital signs will be used for analysis.")

with col3:
    st.markdown("### ⚡ Kinetic Tests")
    sit_ups = st.number_input("Core Stability (Sit-ups)", 0, 100, 40)
    bend = st.number_input("Flexibility (cm)", -10.0, 40.0, 15.0)
    jump = st.number_input("Explosive Power (cm)", 0.0, 350.0, 200.0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 INITIATE NEURAL ANALYSIS"):
    # العمليات الحسابية
    bmi = weight / ((height / 100) ** 2)
    
    # بناء الـ DataFrame بنفس الترتيب المطلوب (12 عمود)
    input_data = pd.DataFrame([[age, gender, height, weight, fat, 80, 120, 45, bend, sit_ups, jump, bmi]],
                              columns=['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 
                                       'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm', 
                                       'sit-ups counts', 'broad jump_cm', 'BMI'])

    # التحجيم والتوقع
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    # النتائج والتنسيق الملون
    classes = ['A (Elite Performance)', 'B (Good Condition)', 'C (Average)', 'D (Needs Improvement)']
    colors = ['#238636', '#1f6feb', '#d29922', '#f85149'] # أخضر، أزرق، أصفر، أحمر
    
    res_text = classes[prediction[0]]
    res_color = colors[prediction[0]]

    st.balloons()
    
    # عرض النتيجة بشكل مبهر
    st.markdown(f"""
        <div style="border: 1px solid {res_color}; padding: 30px; border-radius: 15px; text-align: center; background: rgba(0,0,0,0.3); box-shadow: 0 0 20px {res_color}44;">
            <h2 style="color: #8b949e; letter-spacing: 2px;">FINAL CLASSIFICATION</h2>
            <h1 style="color: {res_color}; font-size: 50px; text-shadow: 0 0 10px {res_color};">{res_text}</h1>
        </div>
    """, unsafe_allow_html=True)
