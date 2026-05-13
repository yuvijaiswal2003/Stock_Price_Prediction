from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

import torch
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model import RNNModel

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
# Load Model
# -----------------------------
model = RNNModel(
    input_size=5,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    rnn_type="GRU"
).to(device)

model.load_state_dict(
    torch.load("stock_model.pth", map_location=device)
)
model.eval()

# -----------------------------
# Input Schema
# -----------------------------
class StockInput(BaseModel):
    ticker: str
    end_date: str

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

    try:
        end_dt = datetime.strptime(data.end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Fetch enough history before end_date (single call)
    start_dt = end_dt - timedelta(days=200)
    future_end_dt = end_dt + timedelta(days=10)

    try:
        # -----------------------------
        # Single yfinance call for all data
        # -----------------------------
        full_df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=future_end_dt.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )
    except Exception as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}

    if full_df.empty:
        return {"error": f"No data found for ticker '{ticker}'"}

    # Flatten MultiIndex columns if present
    if isinstance(full_df.columns, pd.MultiIndex):
        full_df.columns = full_df.columns.get_level_values(0)

    # -----------------------------
    # Split into historical and future
    # -----------------------------
    hist_df = full_df[full_df.index < pd.Timestamp(end_dt)]
    future_df = full_df[full_df.index >= pd.Timestamp(end_dt)]

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
    absolute_error = abs(actual_price - predicted_price)
    percentage_error = (absolute_error / actual_price) * 100

    # -----------------------------
    # Final Response
    # -----------------------------
    return {
        "ticker": ticker,
        "prediction_based_on_date": data.end_date,
        "predicted_next_close": round(float(predicted_price), 2),
        "actual_next_close": round(float(actual_price), 2),
        "absolute_error": round(float(absolute_error), 2),
        "percentage_error": round(float(percentage_error), 2)
    }