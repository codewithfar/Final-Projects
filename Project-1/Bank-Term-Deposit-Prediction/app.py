# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def engineer_features(data):
    data = data.copy()

    education_mapping = {
        "unknown": 0,
        "primary": 1,
        "secondary": 2,
        "tertiary": 3
    }

    data["education_level"] = data["education"].map(
        education_mapping
    )

    data["age_balance"] = data["age"] * data["balance"]
    data["age_duration"] = data["age"] * data["duration"]
    data["balance_duration"] = data["balance"] * data["duration"]
    data["age_education"] = data["age"] * data["education_level"]
    data["balance_education"] = (
        data["balance"] * data["education_level"]
    )

    data = data.drop("education", axis=1)

    return data
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/final_model.pkl")


model = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 Bank Term Deposit Subscription Predictor")

st.write(
    "Enter the customer's details below to predict "
    "whether they are likely to subscribe to a term deposit."
)


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["divorced", "married", "single"]
    )

    education = st.selectbox(
        "Education",
        ["unknown", "primary", "secondary", "tertiary"]
    )


with col2:
    default = st.selectbox(
        "Credit in Default?",
        ["no", "yes"]
    )

    balance = st.number_input(
        "Account Balance",
        value=1000
    )

    housing = st.selectbox(
        "Housing Loan?",
        ["no", "yes"]
    )

    loan = st.selectbox(
        "Personal Loan?",
        ["no", "yes"]
    )


with col3:
    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=300
    )


# --------------------------------------------------
# Campaign Information
# --------------------------------------------------

st.header("Campaign Information")

col4, col5, col6 = st.columns(3)

with col4:
    campaign = st.number_input(
        "Number of Contacts During Campaign",
        min_value=1,
        value=1
    )

with col5:
    pdays = st.number_input(
        "Days Since Previous Campaign Contact",
        value=-1
    )

with col6:
    previous = st.number_input(
        "Previous Campaign Contacts",
        min_value=0,
        value=0
    )

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Subscription",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "✅ Customer is likely to subscribe to a term deposit."
        )
    else:
        st.warning(
            "❌ Customer is unlikely to subscribe to a term deposit."
        )

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "Prediction generated using the trained machine-learning model."
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.divider()

st.subheader("About This Model")

st.write(
    "The application uses a machine-learning pipeline containing "
    "feature engineering, preprocessing, and the selected final "
    "classification model."
)