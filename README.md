# 🏋️‍♂️ AI Body Performance Predictor

This project is an End-to-End Machine Learning application that predicts a person's fitness category (A, B, C, or D) based on physical metrics. It uses an **Ensemble Learning** approach to ensure high accuracy and is deployed as an interactive web app.

## 🚀 Live Demo
[Insert your Streamlit Cloud Link Here after deployment]

## 📋 Project Overview
The goal of this project is to classify body performance using various physical tests like sit-ups, broad jumps, and body fat percentage.

### Key Features:
* **Advanced Modeling:** Utilized a **Voting Classifier** (Random Forest + XGBoost + LightGBM) to achieve the best performance.
* **Data Processing:** Automated scaling and feature engineering (BMI calculation).
* **Interactive UI:** Built with **Streamlit** for real-time user predictions.
* **Deployment:** Hosted on **Streamlit Cloud** for 24/7 availability.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Joblib, XGBoost, LightGBM.
* **UI/UX:** Streamlit.
* **Version Control:** Git & GitHub.

## 🗂️ File Structure
* `app.py`: The main script for the Streamlit web application.
* `body_performance_model_compressed.pkl`: The trained and compressed machine learning model.
* `scaler.pkl`: The StandardScaler object for data normalization.
* `requirements.txt`: List of dependencies for the environment.

## 🏃‍♂️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Fitness-Performance-AI.git](https://github.com/YOUR_USERNAME/Fitness-Performance-AI.git)
