import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="centered"
)

# ---------------------------------
# Custom Dark UI Styling
# ---------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
    background-color: #262730;
    color: white;
    border-radius: 8px;
    padding: 0.6rem;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #3a3b45;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load trained model
# ---------------------------------
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------------------------
# App Title
# ---------------------------------
st.markdown(
    "<h1 style='text-align:center;'>Fraud Detection Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Enter transaction details and click Predict</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------
# Input Form (Centered)
# ---------------------------------
with st.form("fraud_form"):

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN"]
    )

    amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)

    oldbalance_org = st.number_input(
        "Old Balance (Sender)", min_value=0.0, value=10000.0, step=500.0
    )

    newbalance_org = st.number_input(
        "New Balance (Sender)", min_value=0.0, value=9000.0, step=500.0
    )

    oldbalance_dest = st.number_input(
        "Old Balance (Receiver)", min_value=0.0, value=0.0, step=500.0
    )

    newbalance_dest = st.number_input(
        "New Balance (Receiver)", min_value=0.0, value=0.0, step=500.0
    )

    submit = st.form_submit_button("Predict")

# ---------------------------------
# Prediction Logic
# ---------------------------------
if submit:

    type_map = {
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "DEPOSIT": 3,
        "CASH_IN": 4
    }

    input_data = pd.DataFrame([[
        type_map[transaction_type],
        amount,
        oldbalance_org,
        newbalance_org,
        oldbalance_dest,
        newbalance_dest
    ]], columns=[
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] * 100

    st.divider()

    if prediction[0] == 1:
        st.error(
            f"🚨 Fraudulent Transaction Detected\n\n"
            f"Confidence: **{probability:.2f}%**"
        )
    else:
        st.success(
            f"✅ Legitimate Transaction\n\n"
            f"Confidence: **{100 - probability:.2f}%**"
        )

   
