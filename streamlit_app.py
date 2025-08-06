# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Bank Product Subscription Predictor", layout="centered")
st.title("💼 Precision Banking Predictor")
st.subheader("Will a client subscribe to a bank product? 🤖")

# Feature input form
with st.form(key="client_form"):
    age = st.slider("Age", 18, 100, 35)
    job = st.selectbox("Job (encoded)", options=range(0, 12))
    marital = st.selectbox("Marital Status (encoded)", options=range(0, 3))
    education = st.selectbox("Education (encoded)", options=range(0, 4))
    default = st.selectbox("Has Credit in Default?", options=[0, 1])
    balance = st.number_input("Account Balance (€)", value=0)
    housing = st.selectbox("Has Housing Loan?", options=[0, 1])
    loan = st.selectbox("Has Personal Loan?", options=[0, 1])
    contact = st.selectbox("Contact Type (encoded)", options=range(0, 2))
    day = st.slider("Day of Month Contacted", 1, 31, 15)
    month = st.selectbox("Month (encoded)", options=range(0, 12))
    campaign = st.slider("No. of Contacts During Campaign", 1, 50, 1)
    pdays = st.slider("Days Since Last Contact", -1, 999, -1)
    previous = st.slider("Previous Contacts Before Campaign", 0, 50, 0)
    poutcome = st.selectbox("Previous Outcome (encoded)", options=range(0, 3))
    duration_days = st.number_input("Call Duration (days)", value=0.01)

    submit = st.form_submit_button("Predict")

# Run prediction
if submit:
    input_data = pd.DataFrame([[
        age, job, marital, education, default, balance,
        housing, loan, contact, day, month,
        campaign, pdays, previous, poutcome, duration_days
    ]], columns=[
        "age", "job", "marital", "education", "default", "balance", "housing",
        "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome", "duration_days"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.success(f"✅ The client is **likely to subscribe** (Confidence: {round(probability * 100, 2)}%)")
    else:
        st.warning(f"❌ The client is **unlikely to subscribe** (Confidence: {round((1 - probability) * 100, 2)}%)")
