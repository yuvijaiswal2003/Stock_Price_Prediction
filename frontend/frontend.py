import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="centered"
)

# -----------------------------
# Backend URL
# -----------------------------
BACKEND_URL = "https://stock-price-prediction-rrxz.onrender.com"

# -----------------------------
# Wake up Render backend (free tier spins down)
# -----------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes
def wake_up_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=60)
        return r.status_code == 200
    except Exception:
        return False

# -----------------------------
# Title
# -----------------------------
st.title("📈 Stock Price Prediction App")

st.write(
    "Predict next-day stock closing price "
    "using GRU/LSTM deep learning model."
)

# -----------------------------
# Wake backend on load
# -----------------------------
with st.spinner("🔌 Connecting to backend (may take ~30s on first load)..."):
    is_alive = wake_up_backend()

msg = st.empty()

if is_alive:
    msg.success("✅ Backend connected!")
else:
    msg.warning("⚠️ Backend may be slow to respond. Please wait.")

time.sleep(2)      # show for 2 seconds
msg.empty()        # then disappear

# -----------------------------
# User Inputs
# -----------------------------
ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="e.g. AAPL, TSLA, GOOGL, INFY.NS"
)

end_date = st.date_input(
    "Select Historical Date",
    help="Prediction is made based on 60 days before this date"
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict"):

    if not ticker.strip():
        st.error("Please enter a valid stock ticker.")
    else:
        with st.spinner("⏳ Fetching data and predicting... (may take up to 60s)"):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={
                        "ticker": ticker.strip().upper(),
                        "end_date": str(end_date)
                    },
                    timeout=120
                )

                # Check HTTP status
                if response.status_code != 200:
                    st.error(f"Backend error: HTTP {response.status_code}")
                    st.stop()

                result = response.json()

                # -----------------------------
                # Check for API-level errors
                # -----------------------------
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    st.stop()

                # -----------------------------
                # Display Results
                # -----------------------------
                st.success("Prediction Complete ✅")

                st.subheader("📊 Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Predicted Next Close",
                        value=f"${result['predicted_next_close']}"
                    )

                with col2:
                    st.metric(
                        label="Actual Next Close",
                        value=f"${result['actual_next_close']}"
                    )

                col3, col4 = st.columns(2)

                with col3:
                    st.metric(
                        label="Absolute Error",
                        value=f"${result['absolute_error']}"
                    )

                with col4:
                    st.metric(
                        label="Percentage Error",
                        value=f"{result['percentage_error']}%"
                    )

                st.caption(
                    f"Ticker: **{result['ticker']}** | "
                    f"Prediction based on data up to: **{result['prediction_based_on_date']}**"
                )

                # -----------------------------
                # Plot Graph
                # -----------------------------
                st.subheader("📉 Predicted vs Actual Price")

                graph_df = pd.DataFrame({
                    "Type": ["Predicted", "Actual"],
                    "Price": [
                        result['predicted_next_close'],
                        result['actual_next_close']
                    ]
                })

                fig, ax = plt.subplots(figsize=(5, 4))

                bars = ax.bar(
                    graph_df["Type"],
                    graph_df["Price"],
                    color=["#4C72B0", "#DD8452"],
                    width=0.4
                )

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"${height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11
                    )

                ax.set_xlabel("Price Type")
                ax.set_ylabel("Stock Price (USD)")
                ax.set_title(f"{ticker.upper()} — Predicted vs Actual")
                ax.set_ylim(0, max(graph_df["Price"]) * 1.2)

                st.pyplot(fig)

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out. The backend may be cold-starting. "
                    "Please wait 30 seconds and try again."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Could not connect to backend. "
                    "Check if Render service is running."
                )

            except Exception as e:
                st.error(f"Unexpected error: {e}")