from fastapi import FastAPI
from pydantic import BaseModel

import torch
import pickle
import yfinance as yf
import pandas as pd
import numpy as np

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
scaler = pickle.load(
    open("scaler.pkl", "rb")
)

close_scaler = pickle.load(
    open("close_scaler.pkl", "rb")
)

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
    torch.load(
        "stock_model.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------
# Input Schema
# -----------------------------
class StockInput(BaseModel):
    ticker: str
    end_date:str

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():

    return {
        "message":
        "Stock Prediction API Running"
    }

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: StockInput):

    ticker = data.ticker

    # Download latest stock data
    df = yf.download(
        ticker,
        end=data.end_date,
        period="120d"
    )

    # Validate data
    if len(df) < SEQ_LEN:

        return {
            "error":
            "Not enough stock data"
        }

    # OHLCV Features
    features = df[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ]

    # Scale features
    scaled_data = scaler.transform(features)

    # Last 60-day sequence
    sequence = scaled_data[-SEQ_LEN:]

    # Tensor conversion
    X = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        prediction = model(X)

    # Convert back to actual price
    predicted_price = close_scaler.inverse_transform(
        [[prediction.item()]]
    )[0][0]

    # Actual Next Trading Day Price
    # -----------------------------
    future_df = yf.download(
        ticker,
        start=data.end_date,
        period="5d"
    )

    # Validate future data
    if len(future_df) < 2:

        return {
            "error":
            "Could not fetch future trading data"
        }

    actual_price = future_df['Close'].iloc[1].item()

    # -----------------------------
    # Error Calculation
    # -----------------------------
    absolute_error = abs(
        actual_price - predicted_price
    )

    percentage_error = (
        absolute_error / actual_price
    ) * 100

    # -----------------------------
    # Final Response
    # -----------------------------
    return {

        "ticker": ticker,

        "prediction_based_on_date":
        data.end_date,

        "predicted_next_close":
        round(float(predicted_price), 2),

        "actual_next_close":
        round(float(actual_price), 2),

        "absolute_error":
        round(float(absolute_error), 2),

        "percentage_error":
        round(float(percentage_error), 2)
    }