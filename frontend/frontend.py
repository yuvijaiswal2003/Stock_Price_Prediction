import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("📈 Stock Price Prediction App")

st.write(
    "Predict next-day stock closing price "
    "using GRU/LSTM deep learning model."
)

# -----------------------------
# User Inputs
# -----------------------------
ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

end_date = st.date_input(
    "Select Historical Date"
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):

    with st.spinner("Predicting..."):

        try:

            response = requests.post(
                "https://stock-price-prediction-rrxz.onrender.com/predict",
                json={
                    "ticker": ticker,
                    "end_date": str(end_date)
                },
                timeout=120
            )

            result = response.json()

            # -----------------------------
            # Display Results
            # -----------------------------
            st.success("Prediction Complete ✅")

            st.subheader("Prediction Results")

            st.write(
                f"Ticker: {result['ticker']}"
            )

            st.write(
                f"Prediction Based On Date: "
                f"{result['prediction_based_on_date']}"
            )

            st.write(
                f"Predicted Next Close: "
                f"{result['predicted_next_close']}"
            )

            st.write(
                f"Actual Next Close: "
                f"{result['actual_next_close']}"
            )

            st.write(
                f"Absolute Error: "
                f"{result['absolute_error']}"
            )

            st.write(
                f"Percentage Error: "
                f"{result['percentage_error']}%"
            )

            # -----------------------------
            # Create Graph Data
            # -----------------------------
            graph_df = pd.DataFrame({
                "Type": [
                    "Predicted",
                    "Actual"
                ],
                "Price": [
                    result['predicted_next_close'],
                    result['actual_next_close']
                ]
            })

            # -----------------------------
            # Plot Graph
            # -----------------------------
            fig, ax = plt.subplots()

            ax.bar(
                graph_df["Type"],
                graph_df["Price"]
            )

            ax.set_xlabel("Price Type")

            ax.set_ylabel("Stock Price")

            ax.set_title(
                f"{ticker} Predicted vs Actual Price"
            )

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Error: {e}")