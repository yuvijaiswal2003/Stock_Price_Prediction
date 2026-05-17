from fastapi import FastAPI
from pydantic import BaseModel

import torch
import pickle
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import logging

from model import RNNModel

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# -----------------------------
# Device
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load Scalers
# -----------------------------
scaler = pickle.load(open("scaler.pkl", "rb"))
close_scaler = pickle.load(open("close_scaler.pkl", "rb"))

# -----------------------------
# Sequence Length
# -----------------------------
SEQ_LEN = 60

# -----------------------------
# Twelve Data API Key
# -----------------------------
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "")

# -----------------------------
# Load Model
# -----------------------------
model = RNNModel(
    input_size=5,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    rnn_type="GRU"
).to(device)

model.load_state_dict(torch.load("stock_model.pth", map_location=device))
model.eval()

# -----------------------------
# Input Schema
# -----------------------------
class StockInput(BaseModel):
    ticker: str
    end_date: str

# -----------------------------
# Fetch stock data from Twelve Data
# -----------------------------
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol":     ticker,
        "interval":   "1day",
        "start_date": start_date,
        "end_date":   end_date,
        "outputsize": 5000,          # max rows
        "order":      "ASC",         # oldest first
        "apikey":     TWELVE_DATA_KEY
    }

    logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    # -----------------------------
    # Error handling
    # -----------------------------
    if data.get("status") == "error":
        msg = data.get("message", "Unknown error")
        raise ValueError(f"Twelve Data error: {msg}")

    if "values" not in data:
        logger.error(f"Unexpected response: {data}")
        raise ValueError(f"No data returned for ticker '{ticker}'")

    values = data["values"]

    if len(values) == 0:
        raise ValueError(f"Empty data for ticker '{ticker}'")

    # -----------------------------
    # Parse into DataFrame
    # -----------------------------
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index(ascending=True)

    df.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume"
    }, inplace=True)

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    logger.info(f"Fetched {len(df)} rows for {ticker}")

    return df

# -----------------------------
# Home / Health Check Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Stock Prediction API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: StockInput):

    ticker = data.ticker
    logger.info(f"Predicting for ticker: {ticker}, date: {data.end_date}")

    # -----------------------------
    # Validate date
    # -----------------------------
    try:
        end_dt = datetime.strptime(data.end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    start_dt     = end_dt - timedelta(days=200)
    future_end_dt = end_dt + timedelta(days=10)

    # -----------------------------
    # Fetch stock data
    # -----------------------------
    try:
        full_df = fetch_stock_data(
            ticker,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=future_end_dt.strftime("%Y-%m-%d")
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        return {"error": f"Failed to fetch stock data: {str(e)}"}

    if full_df.empty:
        return {"error": f"No data found for ticker '{ticker}'"}

    # -----------------------------
    # Split into historical and future
    # -----------------------------
    hist_df   = full_df[full_df.index < pd.Timestamp(end_dt)]
    future_df = full_df[full_df.index >= pd.Timestamp(end_dt)]

    logger.info(f"hist_df: {len(hist_df)} rows, future_df: {len(future_df)} rows")

    if len(hist_df) < SEQ_LEN:
        return {"error": f"Not enough historical data. Got {len(hist_df)}, need {SEQ_LEN}"}

    if len(future_df) < 1:
        return {"error": "Could not fetch future trading data after the given date"}

    # -----------------------------
    # Prepare features
    # -----------------------------
    features = hist_df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(SEQ_LEN)

    try:
        scaled_data = scaler.transform(features)
    except Exception as e:
        return {"error": f"Scaler transform failed: {str(e)}"}

    sequence = scaled_data[-SEQ_LEN:]

    X = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # -----------------------------
    # Prediction
    # -----------------------------
    with torch.no_grad():
        prediction = model(X)

    predicted_price = close_scaler.inverse_transform(
        [[prediction.item()]]
    )[0][0]

    # -----------------------------
    # Actual Next Trading Day Price
    # -----------------------------
    actual_price = future_df['Close'].iloc[0].item()

    # -----------------------------
    # Error Calculation
    # -----------------------------
    absolute_error   = abs(actual_price - predicted_price)
    percentage_error = (absolute_error / actual_price) * 100

    # -----------------------------
    # Final Response
    # -----------------------------
    return {
        "ticker": ticker,
        "prediction_based_on_date": data.end_date,
        "predicted_next_close":     round(float(predicted_price), 2),
        "actual_next_close":        round(float(actual_price), 2),
        "absolute_error":           round(float(absolute_error), 2),
        "percentage_error":         round(float(percentage_error), 2)
    }