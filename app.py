import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("Bank_churn_model7.pkl")   # trained XGBoost model
scaler = joblib.load("StandardScaler7.pkl")     # trained StandardScaler

# -------------------------------
# FEATURE ORDER (FROM TRAINING)
# -------------------------------
feature_order = [
    'Geography_France',
    'Geography_Germany',
    'Geography_Spain',
    'CreditScore',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

# Numerical columns
num_cols = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]

# -------------------------------
# APP UI
# -------------------------------
st.title("Bank Churn Prediction App")
st.subheader("Predict whether a customer will exit or stay")

# User name
name = st.text_input("Enter your name")

if name:
    st.write(f"Welcome, {name} 👋")

# -------------------------------
# INPUTS
# -------------------------------
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products", min_value=1, max_value=4)
estimated_salary = st.number_input("Estimated Salary")

has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# -------------------------------
# ENCODING
# -------------------------------
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender = 1 if gender == "Male" else 0
has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# -------------------------------
# CREATE DATAFRAME
# -------------------------------
input_data = {
    'Geography_France': geo_france,
    'Geography_Germany': geo_germany,
    'Geography_Spain': geo_spain,
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])

# -------------------------------
# ENSURE CORRECT ORDER
# -------------------------------
input_df = input_df[feature_order]

# -------------------------------
# SCALING
# -------------------------------
input_df[num_cols] = scaler.transform(input_df[num_cols])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Result
    if prediction == 1:
        st.error("⚠️ Customer is likely to EXIT (Churn)")
    else:
        st.success("✅ Customer is likely to STAY")

    # Probability (ALWAYS shown correctly)
    st.write(f"### 🔍 Churn Probability: {probability * 100:.2f}%")