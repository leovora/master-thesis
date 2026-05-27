import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import date
import keras
from typing import Callable, Optional, Tuple
from keras.models import load_model
from scr.trade.backtest import simulate_trades_with_allocation
from scr.trade.trading_strategy import moving_average_strategy


FEATURES = ['high', 'low', 'open', 'close', 'volume']
tickers = ['GOOGL', 'AMZN', 'AAPL', 'PEP', 'JNJ', 'PFE', 'MRK', 'ABBV', 'PG', 'KO',
           'WMT', 'JPM', 'BAC', 'GS', 'V', 'XOM', 'CVX', 'COP', 'BP', 'BA',
           'MMM', 'HON', 'GE', 'T', 'VZ', 'TMUS', 'HSY', 'DUK', 'SO', 'EXC', 'AEP',
           'AMT', 'PLD', 'SPG', 'BHP', 'RIO', 'VALE', 'FCX']
start_date = '2021-01-01'
end_date = '2024-12-31'
sequence_length = 50
window_sizes = [50, 40, 30]

if not os.path.exists("models"):
    os.makedirs("models")

def partition_dataset(sequence_length: int, data: np.ndarray) -> np.ndarray:
    sequences = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
    return np.array(sequences)

def preprocess_data(
    ticker,
    start_date,
    end_date,
    sequence_length,
    source="yfinance",
    data_folder="../data",
    use_partition_dataset = True,
    download_start=None,
):

    if source == "yfinance":
        if download_start is None and not use_partition_dataset:
            # Pull a small warm-up window so the first prediction still lands
            # on the requested evaluation start date.
            download_start = str(
                (pd.Timestamp(start_date) - pd.offsets.BDay(sequence_length + 1)).date()
            )
        download_start = download_start or start_date
        end_date_inclusive = str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date())
        df = safe_download(ticker, download_start, end_date_inclusive, retries=3)
    elif source == "csv":
        csv_path = os.path.join(data_folder, "stock_data", f"{ticker}_data.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] No CSV data for {ticker}: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    if df is None or df.empty:
        print(f"[SKIP] No data for {ticker}")
        return None
    
    # Normalize yfinance download to yahoo_fin format
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()

    required_cols = ['open', 'high', 'low', 'close', 'volume']

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    if df.empty:
        raise ValueError("No numeric rows available after cleaning the dataset.")

    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    target_dates = pd.to_datetime(df.index).to_numpy()[1:] if has_datetime_index else None
    
    #print("DF shape:", df.shape)
    #print(df.head())

    data_filtered = df[FEATURES]
    X = data_filtered.values
    y = data_filtered['close'].values
    
    X = X[:-1]
    y = y[1:]
    
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X_norm = (X - x_min) / (x_max-x_min)
    
    y_max = np.max(y)
    y_min = np.min(y)
    y_norm = (y - y_min) / (y_max-y_min)
    
    X_seq = partition_dataset(sequence_length, X_norm)
    y_seq = y_norm[sequence_length:]
    seq_dates = target_dates[sequence_length:] if target_dates is not None else None

    feature_dim = X_norm.shape[1]
    empty_X = np.empty((0, sequence_length, feature_dim))
    empty_y = np.empty((0,), dtype=y_seq.dtype if y_seq.size else float)

    if X_seq.size == 0 or y_seq.size == 0:
        return empty_X, empty_y, empty_X.copy(), empty_y.copy(), y_min, y_max, x_min, x_max

    if use_partition_dataset:
        split = int(X_seq.shape[0] * 0.8)
        X_train = X_seq[:split]
        y_train = y_seq[:split]
        X_test = X_seq[split:]
        y_test = y_seq[split:]
    else:
        # Use the requested evaluation interval as the test set, while
        # allowing callers to download extra history for lookback context.
        if seq_dates is not None:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            eval_mask = (seq_dates >= start_ts.to_datetime64()) & (seq_dates <= end_ts.to_datetime64())
            X_seq = X_seq[eval_mask]
            y_seq = y_seq[eval_mask]
        X_train = empty_X
        y_train = empty_y
        X_test = X_seq
        y_test = y_seq

    return X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max


def preprocess_data_transformer(ticker, start_date, end_date, sequence_length, source="yfinance", data_folder="../data"):

    df = safe_download(ticker, start_date, end_date, retries=3)

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols].apply(pd.to_numeric, errors='coerce').dropna()

    df = df.reset_index(drop=True)

    X = df[FEATURES].values.astype(np.float32)

    # TARGET = RETURN (% change)
    close_prices = df['close'].values.astype(np.float32)

    returns = (close_prices[1:] / close_prices[:-1]) - 1

    # align X with returns
    X = X[:-1]

    split_idx = int(len(X) * 0.8)

    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = returns[:split_idx], returns[split_idx:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)

    y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    def make_seq(X, y):
        Xs, ys = [], []
        for i in range(sequence_length, len(X)):
            Xs.append(X[i-sequence_length:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = make_seq(X_train_scaled, y_train_scaled)
    X_test_seq, y_test_seq = make_seq(X_test_scaled, y_test_scaled)

    # Keep the original close series so predicted returns can be mapped back
    # to the correct base day without introducing a one-step shift.
    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_scaler, close_prices)
    
def safe_download(ticker, start, end, retries=3):
    for i in range(retries):
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            return df
        time.sleep(1)
    return None


def build_quarterly_windows(simulation_start, simulation_end, months=3):
    start = pd.Timestamp(simulation_start).normalize()
    end_exclusive = pd.Timestamp(simulation_end).normalize() + pd.Timedelta(days=1)

    windows = []
    cursor = start
    while cursor < end_exclusive:
        next_cursor = cursor + pd.DateOffset(months=months)
        window_end = min(next_cursor, end_exclusive)
        windows.append((cursor, window_end))
        cursor = window_end

    return windows


def _normalize_stock_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex) and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    return df.sort_index()


def _resolve_window_model_path(model_folder, ticker, window_start, window_end):
    root = Path(model_folder)
    window_start_str = pd.Timestamp(window_start).strftime("%Y%m%d")
    window_end_str = pd.Timestamp(window_end).strftime("%Y%m%d")

    def _candidate_files(base: Path):
        if not base.exists():
            return []
        return list(base.glob(f"{ticker}_*.h5")) + list(base.glob(f"{ticker}_*.keras"))

    candidates = []

    # 1) Preferred layout: model_folder/ticker/*.h5
    ticker_dir = root / ticker
    candidates.extend(_candidate_files(ticker_dir))

    # 2) Recursive search under the whole model tree
    if not candidates and root.exists():
        candidates = list(root.rglob(f"{ticker}_*.h5")) + list(root.rglob(f"{ticker}_*.keras"))

    if not candidates:
        return None

    # Prefer the exact window match if available.
    exact_matches = [
        p for p in candidates
        if window_start_str in p.name and window_end_str in p.name
    ]
    if exact_matches:
        candidates = exact_matches

    candidates = sorted(set(candidates))
    return candidates[-1]


def _prepare_window_sequences(df, sequence_length, train_start, train_end, test_start, test_end):
    df = _normalize_stock_frame(df)
    if df.empty:
        raise ValueError("Empty dataframe after normalization.")

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("No numeric rows available after cleaning the dataset.")

    close_prices = data["close"].to_numpy(dtype=float)
    target_dates = pd.to_datetime(data.index).to_numpy()[1:]

    x_raw = data.to_numpy(dtype=float)[:-1]
    y_raw = close_prices[1:]

    train_mask = (target_dates >= np.datetime64(pd.Timestamp(train_start))) & (target_dates < np.datetime64(pd.Timestamp(train_end)))
    test_mask = (target_dates >= np.datetime64(pd.Timestamp(test_start))) & (target_dates < np.datetime64(pd.Timestamp(test_end)))

    if train_mask.sum() <= sequence_length:
        raise ValueError("Not enough training rows for the selected window.")
    if test_mask.sum() == 0:
        raise ValueError("No test rows found for the selected window.")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(x_raw[train_mask])
    y_scaler.fit(y_raw[train_mask].reshape(-1, 1))

    x_scaled = x_scaler.transform(x_raw)
    y_scaled = y_scaler.transform(y_raw.reshape(-1, 1)).flatten()

    x_seq = []
    y_seq = []
    seq_dates = []

    for i in range(sequence_length, len(x_scaled)):
        x_seq.append(x_scaled[i - sequence_length:i])
        y_seq.append(float(y_scaled[i]))
        seq_dates.append(pd.Timestamp(target_dates[i]))

    x_seq = np.asarray(x_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    seq_dates = np.asarray(seq_dates, dtype="datetime64[ns]")

    train_seq_mask = (seq_dates >= np.datetime64(pd.Timestamp(train_start))) & (seq_dates < np.datetime64(pd.Timestamp(train_end)))
    test_seq_mask = (seq_dates >= np.datetime64(pd.Timestamp(test_start))) & (seq_dates < np.datetime64(pd.Timestamp(test_end)))

    X_train = x_seq[train_seq_mask]
    y_train = y_seq[train_seq_mask]
    X_test = x_seq[test_seq_mask]
    y_test = y_seq[test_seq_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Window preprocessing produced empty train/test sequences.")

    return X_train, y_train, X_test, y_test, y_scaler

def get_predictions(tickers, start_date, end_date, sequence_length, folder_path="models", use_partition_dataset=True, download_start=None):
    predictions = {}

    actuals = {}
    
    for ticker in tickers:
        print(f"Processing {ticker}...")

        model_path = os.path.join(folder_path, f"{ticker}_model.h5")
        if not os.path.exists(model_path):
            print(f"No model found for {ticker}. Skipping.")
            continue

        model = load_model(model_path)

        result = preprocess_data(
            ticker,
            start_date,
            end_date,
            sequence_length,
            use_partition_dataset=use_partition_dataset,
            download_start=download_start,
        )
        
        if result is None:
            continue
        
        X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = result

        y_pred = model.predict(X_test)
        y_pred_rescaled = y_pred * (y_max - y_min) + y_min
        y_actual = y_test * (y_max - y_min) + y_min

        predictions[ticker] = y_pred_rescaled.flatten()
        actuals[ticker] = y_actual.flatten()

    return predictions, actuals

def get_predictions_transformer(tickers, start_date, end_date, sequence_length, folder_path="models"):

    predictions = {}
    actuals = {}

    for ticker in tickers:

        print(f"Processing {ticker}...")

        model_path = os.path.join(folder_path, f"{ticker}_model.h5")

        if not os.path.exists(model_path):
            continue

        model = load_model(model_path, safe_mode=False)

        result = preprocess_data_transformer(ticker, start_date, end_date, sequence_length)

        if result is None:
            continue

        X_train, y_train, X_test, y_test, y_scaler, close_prices = result

        pred_scaled = model.predict(X_test, verbose=0)

        pred_returns = y_scaler.inverse_transform(pred_scaled).flatten()
        actual_returns = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        split_idx = int(len(close_prices) * 0.8)
        test_start = split_idx + sequence_length

        # Each predicted return at index i refers to the move from close[t]
        # to close[t+1]. The base price must therefore be close[t], not close[t+1].
        base_prices = close_prices[test_start : test_start + len(pred_returns)]

        # safety alignment
        min_len = min(len(base_prices), len(pred_returns), len(actual_returns))

        base_prices = base_prices[:min_len]
        pred_returns = pred_returns[:min_len]
        actual_returns = actual_returns[:min_len]

        pred_prices = base_prices * (1 + pred_returns)
        actual_prices = base_prices * (1 + actual_returns)

        predictions[ticker] = pred_prices
        actuals[ticker] = actual_prices

        print(f"{ticker} OK -> {pred_prices.shape}")

    return predictions, actuals

def get_predictions_quarterly_models(
    tickers,
    simulation_start,
    simulation_end,
    sequence_length,
    model_folder="models",
    data_start=None,
    months=3,
    train_mode="cumulative",
    train_window_months=3,
    source="yfinance",
    data_folder="../data",
):
    """
    Load versioned quarterly LSTM models and build concatenated predictions/actuals.

    train_mode:
    - "cumulative": train on all data before each test window
    - "rolling": train on the last `train_window_months` before each test window
    """
    predictions = {}
    actuals = {}

    simulation_start_ts = pd.Timestamp(simulation_start).normalize()
    simulation_end_ts = pd.Timestamp(simulation_end).normalize()
    data_start_ts = pd.Timestamp(data_start).normalize() if data_start is not None else simulation_start_ts
    windows = build_quarterly_windows(simulation_start_ts, simulation_end_ts, months=months)

    if train_mode not in {"cumulative", "rolling"}:
        raise ValueError("train_mode must be either 'cumulative' or 'rolling'.")

    if source == "yfinance":
        def load_data_fn(ticker):
            return safe_download(ticker, str(data_start_ts.date()), str((simulation_end_ts + pd.Timedelta(days=1)).date()))
    elif source == "csv":
        def load_data_fn(ticker):
            csv_path = os.path.join(data_folder, "stock_data", f"{ticker}_data.csv")
            if not os.path.exists(csv_path):
                return pd.DataFrame()
            return pd.read_csv(csv_path)
    else:
        raise ValueError(f"Unknown data source: {source}")

    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = _normalize_stock_frame(load_data_fn(ticker))
        if df.empty:
            print(f"[SKIP] No data for {ticker}")
            continue

        ticker_predictions = []
        ticker_actuals = []

        for window_start, window_end in windows:
            model_path = _resolve_window_model_path(model_folder, ticker, window_start, window_end)
            if model_path is None:
                print(f"[SKIP] No model found for {ticker} in window {window_start.date()} -> {window_end.date()}")
                continue

            if train_mode == "cumulative":
                train_start = data_start_ts
            else:
                train_start = max(data_start_ts, window_start - pd.DateOffset(months=train_window_months))

            try:
                X_train, y_train, X_test, y_test, y_scaler = _prepare_window_sequences(
                    df=df,
                    sequence_length=sequence_length,
                    train_start=train_start,
                    train_end=window_start,
                    test_start=window_start,
                    test_end=window_end,
                )

                model = load_model(model_path, compile=False)
                pred_scaled = model.predict(X_test, verbose=0)
                pred_values = y_scaler.inverse_transform(pred_scaled).flatten()
                actual_values = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                min_len = min(len(pred_values), len(actual_values))
                ticker_predictions.append(pred_values[:min_len])
                ticker_actuals.append(actual_values[:min_len])
            except Exception as exc:
                print(f"[SKIP] {ticker} {window_start.date()} -> {window_end.date()}: {exc}")

        if not ticker_predictions:
            continue

        predictions[ticker] = np.concatenate(ticker_predictions)
        actuals[ticker] = np.concatenate(ticker_actuals)

    if not predictions:
        return {}, {}

    min_len = min(len(values) for values in predictions.values())
    predictions = {ticker: values[:min_len] for ticker, values in predictions.items()}
    actuals = {ticker: values[:min_len] for ticker, values in actuals.items()}

    return predictions, actuals

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))

def daily_sharpe_ratio(returns, risk_free_rate_annual=0.0505, trading_days=252):
    """Calculate daily Sharpe Ratio."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return np.nan

    risk_free_rate_daily = risk_free_rate_annual / trading_days
    excess_returns = returns - risk_free_rate_daily
    excess_std = np.std(excess_returns, ddof=1)

    if not np.isfinite(excess_std) or excess_std == 0:
        return np.nan

    return np.mean(excess_returns) / excess_std * np.sqrt(trading_days)


def _set_auto_xlim(ax, series_length):
    """Set a sensible x-axis limit based on the plotted series length."""
    if series_length <= 0:
        return
    ax.set_xlim(0, max(series_length - 1, 1))

def plot_dynamic_sharpe_ratio(returns, risk_free_rate=0.0505, trading_days=252, folder='plots/baseline', min_periods=20):
    rolling_sharpe = []
    returns = np.asarray(returns, dtype=float)
    min_periods = max(2, min(min_periods, len(returns)))

    for i in range(1, len(returns) + 1):
        if i < min_periods:
            rolling_sharpe.append(np.nan)
            continue

        temp_returns = returns[:i]
        temp_sharpe = daily_sharpe_ratio(temp_returns, risk_free_rate, trading_days)
        rolling_sharpe.append(temp_sharpe)

    plt.figure(figsize=(14, 8))
    rolling_sharpe = np.asarray(rolling_sharpe, dtype=float)
    valid_idx = np.isfinite(rolling_sharpe)
    plt.plot(np.flatnonzero(valid_idx), rolling_sharpe[valid_idx], label='Sharpe Ratio (baseline)', color='blue')  
    plt.ylabel('Sharpe Ratio', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    _set_auto_xlim(plt.gca(), len(returns))
    plt.grid(True) 
    plt.tight_layout()  
    
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.legend(fontsize=50)  
    filename = os.path.join(folder, "sharpe_ratio_baseline.pdf")
    plt.savefig(filename, dpi=300) 
    plt.show()

    
def plot_daily_returns(portfolio_returns, folder='plots/baseline'):
   
    returns_series = pd.Series(portfolio_returns)

    plt.figure(figsize=(14, 8))  
    plt.plot(returns_series, label='Daily Returns', color='blue')
    plt.xlabel('Days', fontsize=40) 
    plt.ylabel('Returns', fontsize=40)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20) 
    plt.legend(fontsize=50)  
    _set_auto_xlim(plt.gca(), len(returns_series))
    plt.grid(True)  
    plt.tight_layout()  
    filename = os.path.join(folder, "daily_returns.pdf")
    plt.savefig(filename, dpi=300)  
    plt.show()

    
cap_threshold = 0.02

def calculate_cumulative_returns(returns):
    # Cap the returns and compute the cumulative product
    capped_returns = np.clip(returns, -cap_threshold, cap_threshold)
    adjusted_returns = capped_returns + 1
    cumulative_returns = np.cumprod(adjusted_returns) - 1
    return cumulative_returns

def plot_cumulative_returns_baseline(portfolio_returns, folder='plots/baseline'):
    plt.figure(figsize=(14, 8))
    
    baseline_cumulative_returns = calculate_cumulative_returns(portfolio_returns)
    plt.plot(baseline_cumulative_returns, label='Cumulative Returns', color='blue', linewidth=2)
    
    plt.ylabel('Cumulative Returns (%)', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    _set_auto_xlim(plt.gca(), len(baseline_cumulative_returns))
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    filename = os.path.join(folder, "cumulative_returns_baseline.pdf")
    plt.savefig(filename, dpi=300)
    plt.show()   
    

import matplotlib.pyplot as plt
import numpy as np

def plot_average_predictions(predictions: dict, actuals: dict, folder='plots/baseline'):
    """
    Plots the average predictions and actual values for all stocks and saves the plot.

    Parameters:
    - predictions: A dictionary containing predicted values for each stock.
    - actuals: A dictionary containing actual values for each stock.
    """
    # Ensure all tickers have the same length
    first_ticker = next(iter(predictions))
    valid_length = len(predictions[first_ticker])

    valid_tickers = [ticker for ticker in predictions 
                     if ticker in actuals and len(predictions[ticker]) == valid_length and len(actuals[ticker]) == valid_length]

    if not valid_tickers:
        raise ValueError("No valid tickers with matching lengths found.")
        
    min_length = valid_length

    avg_predictions = np.zeros(min_length)
    avg_actuals = np.zeros(min_length)

    n_stocks = len(valid_tickers)

    # Compute sums
    for ticker in valid_tickers:
        avg_predictions += predictions[ticker][:min_length]
        avg_actuals += actuals[ticker][:min_length]

    # Compute averages
    avg_predictions /= n_stocks
    avg_actuals /= n_stocks

    plt.figure(figsize=(14, 8))

    plt.plot(avg_predictions, color='blue', label='Predicted')
    plt.plot(avg_actuals, color='red', label='Actual')

    plt.xlabel("Days", fontsize=40)
    plt.ylabel("Average Price", fontsize=40)
    plt.legend(fontsize=50)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    filename = os.path.join(folder, 'average_predictions.pdf')
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length, folder_path='models')

    
def perform_ephemeral_attack(ticker, start_date, end_date, sequence_length, days_to_attack, window_size=30, folder_path="models", use_partition_dataset=True):
    model_path = os.path.join(folder_path, f"{ticker}_model.h5")
    model = load_model(model_path)

    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length, use_partition_dataset=use_partition_dataset)

    np.random.seed(0)
    attack_indices = np.random.choice(X_test.shape[0], size=days_to_attack, replace=False)

    attacked_X_test = X_test.copy()
    for i in attack_indices:
        window = max(0, sequence_length-window_size)
        stdev = np.std(attacked_X_test[i, window:, 3])
        attacked_X_test[i, -1, 3] += 2 * stdev

    y_pred = model.predict(attacked_X_test)
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min

    predictions = {ticker: y_pred_rescaled.flatten()}
    actuals = {ticker: y_actual.flatten()}

    return predictions, actuals, attack_indices

def ephemeral_attacks(tickers, start_date, end_date, sequence_length, attack_days, window_size=30):
    # Get predictions and actuals
    #predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length)

    # Generate trading signals
    predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length, folder_path='models')

# Trading strategy
    trading_signals = moving_average_strategy(predictions)

# Backtest the strategy
    portfolio_returns = simulate_trades_with_allocation(predictions, actuals, trading_signals)


    returns_after_attacks = [portfolio_returns]
    
    # Apply the attack to Google's model and get the attacked predictions
    for day in attack_days:
        attacked_predictions, _, _ = perform_ephemeral_attack('GOOGL', start_date, end_date, sequence_length, day, window_size=window_size)
        predictions['GOOGL'] = attacked_predictions['GOOGL']  # Update Google's predictions with the attacked ones
        
        # Generate trading signals
        trading_signals = moving_average_strategy(predictions)
        
        # Calculate the portfolio returns after the attack
        portfolio_returns_after_attack = simulate_trades_with_allocation(predictions, actuals, trading_signals)
        
        returns_after_attacks.append(portfolio_returns_after_attack)

    return returns_after_attacks


def perform_ephemeral_attack_with_window(model, ticker, start_date, end_date, sequence_length, attack_index, window_size):
    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    attacked_X_test = X_test.copy()
    window_start = max(0, sequence_length - window_size)
    stdev = np.std(attacked_X_test[attack_index, window_start:, 3])
    attacked_X_test[attack_index, -1, 3] += 2 * stdev

    y_pred_attack = model.predict(attacked_X_test)
    y_pred_rescaled_attack = y_pred_attack * (y_max - y_min) + y_min

    return y_pred_rescaled_attack

def plot_and_save_attack_model(ticker, start_date, end_date, sequence_length, attack_day, model_path, plot_dir):
    if not os.path.exists(model_path):
        print(f"No model found for {ticker}. Skipping.")
        return

    model = load_model(model_path)
    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    y_pred = model.predict(X_test)
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min

    windows = [50, 40, 30]
    attacked_predictions = {}
    for w in windows:
        attacked_predictions[w] = perform_ephemeral_attack_with_window(model, ticker, start_date, end_date, sequence_length, attack_day, w)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(y_actual, color='green', label='Real Value', linewidth=2)
    ax.plot(y_pred_rescaled, color='blue', label='Predicted Value', linewidth=2)
    colors = ['red', 'purple', 'orange']

    for idx, w in enumerate(windows):
        ax.plot(range(attack_day - 1, attack_day + 2), attacked_predictions[w][attack_day - 1:attack_day + 2], color=colors[idx], linestyle='--', label=f'Attack $\omega$={w}', linewidth=1.5)

    # Draw a vertical line representing the attack day
    ax.axvspan(attack_day - 1, attack_day + 1, color='black', linestyle=':', alpha=0.5)

    # Create a zoomed inset
    axins = ax.inset_axes([0.5, 0.2, 0.3, 0.3])
    axins.plot(y_actual, color='green')
    axins.plot(y_pred_rescaled, color='blue')
    for idx, w in enumerate(windows):
        axins.plot(range(attack_day - 1, attack_day + 2), attacked_predictions[w][attack_day - 1:attack_day + 2], color=colors[idx], linestyle='--')

    x1, x2, y1, y2 = attack_day - 10, attack_day + 10, min(y_actual[attack_day - 3:attack_day + 4].min(), y_pred_rescaled[attack_day - 3:attack_day + 4].min(), attacked_predictions[30][attack_day], attacked_predictions[40][attack_day], attacked_predictions[50][attack_day]) - 5, max(y_actual[attack_day - 3:attack_day + 4].max(), y_pred_rescaled[attack_day - 3:attack_day + 4].max(), attacked_predictions[30][attack_day], attacked_predictions[40][attack_day], attacked_predictions[50][attack_day]) + 5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlabel('Days', fontsize=40)
    ax.set_ylabel(f'{ticker} Closing Price', fontsize=40)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper left', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    plot_dir_path = os.path.join(plot_dir, "attack_plots")
    os.makedirs(plot_dir_path, exist_ok=True)
    
    plot_path = os.path.join(plot_dir_path, f"{ticker.lower()}_stock_prediction_attack_day_{attack_day}.pdf")
    plt.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    


def plot_cumulative_returns_after_attack(before_returns, after_returns_list, window_sizes, attack_day, folder='plots/attack_plots'):
    plt.figure(figsize=(14, 8))
    
    # Loop through each window size and calculate/plot cumulative returns after the attack
    colors = ['red', 'green', 'orange', 'purple', 'cyan']  # Extend this list if more windows are needed
    for idx, returns_after in enumerate(after_returns_list):
        after_cumulative_returns = calculate_cumulative_returns(returns_after)
        plt.plot(after_cumulative_returns, label=f'After Attack $\omega$={window_sizes[idx]}', color=colors[idx], linewidth=1.5)
    
    # Plot the baseline cumulative returns from the beginning last
    baseline_cumulative_returns = calculate_cumulative_returns(before_returns)
    plt.plot(baseline_cumulative_returns, label='Baseline', color='blue', linewidth=2)
    
    # Add a vertical line at the attack day
    plt.axvline(x=attack_day, color='gray', linestyle='--', alpha=0.7, label=f'Attack Day {attack_day}')
    
    plt.ylabel('Cumulative Returns', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    max_series_length = max(
        [len(before_returns)] +
        [len(returns_after) for returns_after in after_returns_list]
    ) if after_returns_list else len(before_returns)
    _set_auto_xlim(plt.gca(), max_series_length)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    plot_dir = folder
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = os.path.join(plot_dir, f"cumulative_returns_after_attack_day_{attack_day}.pdf")
    plt.savefig(filename, dpi=300)
    plt.close()  

def plot_cumulative_returns_after_attack_day(tickers, start_date, end_date, sequence_length, window_sizes, attack_day):
    # Get pre-attack returns
    pre_attack_results = ephemeral_attacks(tickers, start_date, end_date, sequence_length, [])
    pre_attack_returns = pre_attack_results[0]

    after_returns_for_all_windows = []

    for w_size in window_sizes:
        # Perform the attack and get the returns
        attack_results = ephemeral_attacks(tickers, start_date, end_date, sequence_length, [attack_day], window_size=w_size)
        returns_after_attack = attack_results[1]

        # Ensure returns are the same up to the attack day
        combined_returns = np.concatenate((pre_attack_returns[:attack_day], returns_after_attack[attack_day:]))
        
        after_returns_for_all_windows.append(combined_returns)

    # Plot the cumulative returns for all windows
    plot_cumulative_returns_after_attack(pre_attack_returns, after_returns_for_all_windows, window_sizes, attack_day)
