import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple

FEATURES = ['high', 'low', 'open', 'close', 'volume']


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for stock data."""
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def _drop_repeated_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the first row if it is a repeated header-like row, e.g.
    FCX,FCX,FCX,FCX,FCX
    """
    if df.empty:
        return df

    first_row = df.iloc[0].astype(str).str.strip()
    if len(set(first_row)) == 1:
        value = first_row.iloc[0]
        # The bogus row is usually made of the ticker repeated in every column.
        # We only drop it if it is clearly non-numeric.
        try:
            float(value)
            return df
        except ValueError:
            return df.iloc[1:].reset_index(drop=True)

    return df

def load_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data for a given ticker from Yahoo Finance.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        time.sleep(1.2)
        return _normalize_columns(df)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the data by normalizing and splitting into features and targets.
    """
    df = _normalize_columns(df)
    df = _drop_repeated_header_row(df)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data_filtered = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    data_filtered = data_filtered.dropna().reset_index(drop=True)

    if data_filtered.empty:
        raise ValueError("No numeric rows available after cleaning the dataset.")

    X = data_filtered.values
    y = data_filtered['close'].values
    
    X = X[:-1]  # Removing the last row from features
    y = y[1:]   # Removing the first element from targets
    
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X_norm = (X - x_min) / (x_max - x_min)
    
    y_max = np.max(y)
    y_min = np.min(y)
    y_norm = (y - y_min) / (y_max - y_min)
    
    X_seq = []
    y_seq = []
    for i in range(sequence_length, len(X_norm)):
        X_seq.append(X_norm[i-sequence_length:i])
        y_seq.append(y_norm[i])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, y_min, y_max

def preprocess_transformer_data(df: pd.DataFrame, sequence_length: int, train_split: float = 0.8):
    """
    Target = future percentual return
    """

    df = _normalize_columns(df)
    df = _drop_repeated_header_row(df)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data_filtered = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    data_filtered = data_filtered.dropna().reset_index(drop=True)

    if data_filtered.empty:
        raise ValueError("No numeric rows available.")

    X_raw = data_filtered.values

    close_prices = data_filtered["close"].values

    # percentual return
    returns = (
        close_prices[1:] - close_prices[:-1]
    ) / close_prices[:-1]

    X_raw = X_raw[:-1]

    split_index = int(len(X_raw) * train_split)

    X_train_raw = X_raw[:split_index]
    X_test_raw = X_raw[split_index:]

    y_train_raw = returns[:split_index]
    y_test_raw = returns[split_index:]

    x_min = np.min(X_train_raw, axis=0)
    x_max = np.max(X_train_raw, axis=0)

    X_train_norm = (X_train_raw - x_min) / (x_max - x_min + 1e-8)
    X_test_norm = (X_test_raw - x_min) / (x_max - x_min + 1e-8)

    # target normalization
    y_min = np.min(y_train_raw)
    y_max = np.max(y_train_raw)

    y_train_norm = (y_train_raw - y_min) / (y_max - y_min + 1e-8)
    y_test_norm = (y_test_raw - y_min) / (y_max - y_min + 1e-8)


    def build_sequences(X, y):
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = build_sequences(X_train_norm, y_train_norm)
    X_test_seq, y_test_seq = build_sequences(X_test_norm, y_test_norm)

    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_min, y_max, x_min, x_max)

def save_data_to_csv(tickers: list, start_date: str, end_date: str, data_folder: str = "data"):
    """
    Save data for multiple tickers to CSV.
    """
    stock_data_folder = os.path.join(data_folder, 'stock_data')
    if not os.path.exists(stock_data_folder):
        os.makedirs(stock_data_folder)
        print(f"Created directory: {stock_data_folder}")

    for ticker in tickers:
        csv_path = os.path.join(stock_data_folder, f"{ticker}_data.csv")
        if os.path.exists(csv_path):
            print(f"Data for {ticker} already exists at {csv_path}. Skipping fetch.")
            continue

        print(f"Fetching and saving data for {ticker}")
        df = load_ticker_data(ticker, start_date, end_date)
        if not df.empty:
            df = _normalize_columns(df)
            df.to_csv(csv_path, index=False)
            print(f"Data for {ticker} saved at {csv_path}")
        else:
            print(f"Failed to fetch data for {ticker}")

def load_data_from_csv(ticker: str, data_folder="../data") -> pd.DataFrame:
    """
    Load data for a specific ticker from CSV.
    """
    csv_path = os.path.join(data_folder, 'stock_data', f"{ticker}_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)
        return _drop_repeated_header_row(df)
    else:
        print(f"No data found for {ticker} in {csv_path}")
        return pd.DataFrame()
