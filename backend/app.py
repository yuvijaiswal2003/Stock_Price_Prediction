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
# Alpha Vantage API Key
# -----------------------------
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")

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
# Fetch stock data from Alpha Vantage
# -----------------------------
def fetch_stock_data(ticker: str) -> pd.DataFrame:

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",       # get full history
        "datatype": "json",
        "apikey": ALPHA_VANTAGE_KEY
    }

    logger.info(f"Fetching data for {ticker} from Alpha Vantage")

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    # Check for errors
    if "Error Message" in data:
        raise ValueError(f"Invalid ticker: {ticker}")

    if "Note" in data:
        raise ValueError("Alpha Vantage API rate limit reached. Try again in a minute.")

    if "Information" in data:
        raise ValueError("Alpha Vantage API limit reached. Please check your API key.")

    if "Time Series (Daily)" not in data:
        logger.error(f"Unexpected response: {data}")
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Parse into DataFrame
    ts = data["Time Series (Daily)"]

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    df.rename(columns={
        "1. open":   "Open",
        "2. high":   "High",
        "3. low":    "Low",
        "4. close":  "Close",
        "5. volume": "Volume"
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

    try:
        end_dt = datetime.strptime(data.end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # -----------------------------
    # Fetch stock data
    # -----------------------------
    try:
        full_df = fetch_stock_data(ticker)
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
    hist_df = full_df[full_df.index < pd.Timestamp(end_dt)]
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