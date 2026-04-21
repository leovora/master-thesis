import os
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
from typing import Tuple

FEATURES = ['high', 'low', 'open', 'close', 'volume']

def load_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data for a given ticker from Yahoo Finance.
    """
    try:
        df = si.get_data(ticker, start_date=start_date, end_date=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the data by normalizing and splitting into features and targets.
    """
    data_filtered = df[FEATURES]
    X = data_filtered.values
    y = df['close'].values
    
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
        return df
    else:
        print(f"No data found for {ticker} in {csv_path}")
        return pd.DataFrame()
