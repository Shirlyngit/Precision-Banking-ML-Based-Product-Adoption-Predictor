import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Bank Product Subscription Predictor", layout="centered")
st.title("💼 Precision Banking Predictor")
st.subheader("Will a client subscribe to a bank product? 🤖")

# ---- Category encoding maps ----
job_map = {
    'admin': 0, 'blue collar': 1, 'entrepreneur': 2, 'housemaid': 3,
    'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
    'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11
}
marital_map = {'divorced': 0, 'married': 1, 'single': 2}
education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
default_map = {'no': 0, 'yes': 1}
housing_map = {'no': 0, 'yes': 1}
loan_map = {'no': 0, 'yes': 1}
contact_map = {'cellular': 0, 'telephone': 1, 'unknown': 2}
month_map = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
poutcome_map = {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3}

# ---- Feature input form ----
with st.form(key="client_form"):
    age = st.slider("Age", 18, 100, 35)
    job = st.selectbox("Job", options=list(job_map.keys()))
    marital = st.selectbox("Marital Status", options=list(marital_map.keys()))
    education = st.selectbox("Education Level", options=list(education_map.keys()))
    default = st.selectbox("Has Credit in Default?", options=list(default_map.keys()))
    balance = st.number_input("Account Balance (€)", value=0)
    housing = st.selectbox("Has Housing Loan?", options=list(housing_map.keys()))
    loan = st.selectbox("Has Personal Loan?", options=list(loan_map.keys()))
    contact = st.selectbox("Contact Type", options=list(contact_map.keys()))
    day = st.slider("Day of Month Contacted", 1, 31, 15)
    month = st.selectbox("Month Contacted", options=list(month_map.keys()))
    duration = st.number_input("Call Duration (in seconds)", value=30)
    campaign = st.slider("No. of Contacts During Campaign", 1, 50, 1)
    pdays = st.slider("Days Since Last Contact", -1, 999, -1)
    previous = st.slider("Previous Contacts Before Campaign", 0, 50, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", options=list(poutcome_map.keys()))

    submit = st.form_submit_button("Predict")

# ---- Prediction ----
if submit:
    input_data = pd.DataFrame([[
        age,
        job_map[job],
        marital_map[marital],
        education_map[education],
        default_map[default],
        balance,
        housing_map[housing],
        loan_map[loan],
        contact_map[contact],
        day,
        month_map[month],
        campaign,
        pdays,
        previous,
        poutcome_map[poutcome],
        duration
    ]], columns=[
        "age", "job", "marital", "education", "default", "balance", "housing",
        "loan", "contact", "day", "month", "campaign", "pdays", "previous",
        "poutcome", "duration"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.success(f"✅ The client is **likely to subscribe** (Confidence: {round(probability * 100, 2)}%)")
    else:
        st.warning(f"❌ The client is **unlikely to subscribe** (Confidence: {round((1 - probability) * 100, 2)}%)")
