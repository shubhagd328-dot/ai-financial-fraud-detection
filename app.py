import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# Configure the dashboard
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Dashboard heading
st.title("🛡️ AI Financial Fraud Detection System")
st.caption("Machine-learning prototype for transaction risk assessment")

st.info(
    "Demonstration project using synthetic transaction data. "
    "Predictions are not suitable for real banking decisions."
)

# Sidebar inputs
st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=2000.0,
    step=100.0
)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=14
)

previous_failed_transactions = st.sidebar.number_input(
    "Previous Failed Transactions",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

international_transaction = st.sidebar.selectbox(
    "International Transaction",
    options=["No", "Yes"]
)

international_value = 1 if international_transaction == "Yes" else 0

# Prepare transaction input in the training feature order
transaction = pd.DataFrame({
    "amount": [amount],
    "transaction_hour": [transaction_hour],
    "previous_failed_transactions": [previous_failed_transactions],
    "international_transaction": [international_value]
})

# Predict transaction risk
if st.button("Analyze Transaction", type="primary"):
    fraud_probability = model.predict_proba(transaction)[0][1]

    st.subheader("Transaction Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Transaction Amount",
            f"{amount:,.2f}"
        )

    # Demonstration threshold
    threshold = 0.20

    if fraud_probability >= threshold:
        st.error("⚠️ HIGHER RISK: Flag for further review")
    else:
        st.success("✅ LOWER RISK: No alert from this prototype")

    st.caption(
        "The 20% threshold is a demonstration setting, "
        "not a validated banking threshold."
    )

    # Display input details
    st.subheader("Transaction Summary")
    st.dataframe(transaction, use_container_width=True)

    # Probability visualization
    chart_data = pd.DataFrame({
        "Category": ["Estimated fraud", "Estimated non-fraud"],
        "Probability": [
            fraud_probability * 100,
            (1 - fraud_probability) * 100
        ]
    })

    figure = px.bar(
        chart_data,
        x="Category",
        y="Probability",
        title="Model-Estimated Probabilities",
        text_auto=".2f"
    )

    figure.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(figure, use_container_width=True)