import streamlit as st
import joblib
import pandas as pd

# 1. إعدادات الصفحة (تغيير الأيقونة والعنوان في التاب)
st.set_page_config(page_title="Fitness AI Pro", page_icon="💪", layout="centered")

# 2. إضافة CSS مخصص لجعل الأزرار والخلفية أجمل
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 3. العنوان مع صورة أو أيقونة
st.title("🏋️‍♂️ AI Body Performance Predictor")
st.write("ادخل بياناتك البدنية وسيقوم الذكاء الاصطناعي بتحليل مستواك فوراً.")

# 4. تقسيم المدخلات إلى أعمدة (Layout) بدلاً من قائمة طويلة
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("العمر (Age)", 10, 80, 25)
    height = st.number_input("الطول (cm)", 120, 220, 170)
    weight = st.number_input("الوزن (kg)", 30, 200, 70)
    body_fat = st.slider("نسبة الدهون (%)", 5.0, 50.0, 20.0)

with col2:
    grip_force = st.number_input("قوة القبضة (Grip Force)", 0.0, 100.0, 40.0)
    sit_ups = st.number_input("عدات البطن (Sit-ups)", 0, 100, 30)
    broad_jump = st.number_input("الوثب الطويل (cm)", 0, 300, 180)
    # أضف بقية المدخلات هنا...

# 5. زر التحليل بتصميم بارز
if st.button("Analyze My Fitness"):
    # كود التوقع (نفس الكود القديم)
    # ...
    # عرض النتيجة بشكل جذاب
    st.success(f"### Your Fitness Category is: {prediction}")
    st.balloons() # إضافة تأثير احتفالي عند ظهور النتيجة
