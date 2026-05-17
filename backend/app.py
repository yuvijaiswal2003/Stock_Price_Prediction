from fastapi import FastAPI
from pydantic import BaseModel

import torch
import pickle
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta

import yfinance as yf
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
# Build robust session for yfinance
# -----------------------------
def get_session():
    session = Session()

    # Retry on failure
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Browser-like headers
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    })

    return session

# -----------------------------
# Fetch stock data with retries
# -----------------------------
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:

    session = get_session()

    # Try up to 3 times with delay
    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt+1}: Fetching {ticker} from {start} to {end}")

            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                session=session
            )

            if not df.empty:
                logger.info(f"Success: {len(df)} rows fetched")
                return df

            logger.warning(f"Attempt {attempt+1}: Empty dataframe")
            time.sleep(2)

        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(2)

    return pd.DataFrame()

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

    start_dt      = end_dt - timedelta(days=200)
    future_end_dt = end_dt + timedelta(days=10)

    # -----------------------------
    # Fetch all data in one call
    # -----------------------------
    full_df = fetch_stock_data(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=future_end_dt.strftime("%Y-%m-%d")
    )

    if full_df.empty:
        return {"error": f"Could not fetch data for '{ticker}'. Yahoo Finance may be blocking requests. Try again in a few minutes."}

    # Flatten MultiIndex columns if present
    if isinstance(full_df.columns, pd.MultiIndex):
        full_df.columns = full_df.columns.get_level_values(0)

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