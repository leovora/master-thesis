from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler

from scr.data.load_data import (
    FEATURES,
    _drop_repeated_header_row,
    _normalize_columns,
    load_ticker_data,
)


@dataclass
class QuarterlyModelSaveResult:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    ticker: str
    model_path: str
    saved: bool
    reused: bool


def _clean_stock_frame(df):
    """Normalize, clean and sort a price dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df = _drop_repeated_header_row(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        # Keep a copy so callers can pass raw dataframes if needed.
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

    df = df.sort_index()
    return df


def _build_cumulative_lstm_model(sequence_length, feature_dim):
    model = Sequential()
    n_neurons = sequence_length * feature_dim
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(sequence_length, feature_dim)))
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _train_cumulative_lstm_model(X, y, sequence_length):
    """
    Train the LSTM on a cumulative history using a chronological validation split.
    """
    if len(X) < 2:
        raise ValueError("Need at least two samples to train the model.")

    val_size = max(1, int(len(X) * 0.2))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    model = _build_cumulative_lstm_model(sequence_length, X.shape[2])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=50,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1,
    )

    return model


def build_cumulative_windows(simulation_start, simulation_end, months = 3):
    """
    Build fixed-length, non-overlapping windows of the simulation period.

    The end of each window is treated as exclusive.
    """
    start = pd.Timestamp(simulation_start).normalize()
    end_exclusive = pd.Timestamp(simulation_end).normalize() + pd.Timedelta(days=1)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end_exclusive:
        next_cursor = cursor + pd.DateOffset(months=months)
        window_end = min(next_cursor, end_exclusive)
        windows.append((cursor, window_end))
        cursor = window_end

    return windows


def _prepare_cumulative_sequences(df, sequence_length, train_end, test_end):
    """
    Prepare train/test sequences for one simulation window.

    Training uses all samples whose target date is strictly before `train_end`.
    Testing uses all samples whose target date is in [train_end, test_end).
    """
    df = _clean_stock_frame(df)
    if df.empty:
        raise ValueError("Empty dataframe after cleaning.")

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("No numeric rows available after cleaning the dataset.")

    close_prices = data["close"].to_numpy(dtype=float)
    target_dates = pd.to_datetime(data.index).to_numpy()[1:]

    # X and y are shifted by one step to model next-day close prediction.
    X_raw = data.to_numpy(dtype=float)[:-1]
    y_raw = close_prices[1:]

    train_mask = target_dates < train_end
    test_mask = (target_dates >= train_end) & (target_dates < test_end)

    if train_mask.sum() <= sequence_length:
        raise ValueError(
            f"Not enough training rows before {train_end.date()} for sequence_length={sequence_length}."
        )
    if test_mask.sum() == 0:
        raise ValueError(f"No test rows found in [{train_end.date()}, {test_end.date()}).")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(X_raw[train_mask])
    y_scaler.fit(y_raw[train_mask].reshape(-1, 1))

    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw.reshape(-1, 1)).flatten()

    X_seq = []
    y_seq = []
    seq_dates = []

    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i - sequence_length : i])
        y_seq.append(float(y_scaled[i]))
        seq_dates.append(pd.Timestamp(target_dates[i]))

    X_seq = np.asarray(X_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    seq_dates = np.asarray(seq_dates, dtype="datetime64[ns]")

    train_seq_mask = seq_dates < np.datetime64(train_end)
    test_seq_mask = (seq_dates >= np.datetime64(train_end)) & (seq_dates < np.datetime64(test_end))

    X_train = X_seq[train_seq_mask]
    y_train = y_seq[train_seq_mask]
    X_test = X_seq[test_seq_mask]
    y_test = y_seq[test_seq_mask]

    if len(X_train) == 0:
        raise ValueError(f"No train sequences generated before {train_end.date()}.")
    if len(X_test) == 0:
        raise ValueError(f"No test sequences generated in [{train_end.date()}, {test_end.date()}).")

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def prepare_cumulative_quarter_data(df, sequence_length, window_start, window_end):
    """
    Build a train/test split for one cumulative quarterly window.

    Returns train/test arrays plus the fitted scalers.
    """
    train_end = pd.Timestamp(window_start).normalize()
    test_end = pd.Timestamp(window_end).normalize()
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = _prepare_cumulative_sequences(
        df=df,
        sequence_length=sequence_length,
        train_end=train_end,
        test_end=test_end,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def train_cumulative_lstm_for_window(df, sequence_length, window_start, window_end):
    """
    Train a fresh LSTM on the cumulative history available before `window_start`.
    """
    prepared = prepare_cumulative_quarter_data(df, sequence_length, window_start, window_end)
    model = _train_cumulative_lstm_model(prepared["X_train"], prepared["y_train"], sequence_length)
    return model, prepared


def train_and_save_cumulative_quarterly_models(
    tickers, 
    simulation_start,
    simulation_end,
    sequence_length,
    data_start = None,
    data_end = None,
    months = 3,
    model_folder = "models/cumulative_lstm",
    save_models = True,
    force_retrain = False):
    """
    Train and save one quarterly cumulative LSTM model per ticker and window.

    For each window:
    - train one fresh LSTM per ticker using all data before the window;
    - save the versioned model on disk for later prediction loading.
    """
    sim_start = pd.Timestamp(simulation_start).normalize()
    sim_end = pd.Timestamp(simulation_end).normalize()
    data_start_ts = pd.Timestamp(data_start).normalize() if data_start is not None else sim_start
    data_end_ts = pd.Timestamp(data_end).normalize() if data_end is not None else sim_end

    windows = build_cumulative_windows(sim_start, sim_end, months=months)
    model_root = Path(model_folder)
    model_root.mkdir(parents=True, exist_ok=True)

    history_cache = {}
    results = []

    for ticker in tickers:
        df = load_ticker_data(ticker, start_date=str(data_start_ts.date()), end_date=str((data_end_ts + pd.Timedelta(days=1)).date()))
        history_cache[ticker] = _clean_stock_frame(df)

    for window_start, window_end in windows:
        window_end_inclusive = window_end - pd.Timedelta(days=1)
        print(f"Training cumulative window {window_start.date()} -> {window_end_inclusive.date()}")

        for ticker in tickers:
            df = history_cache.get(ticker, pd.DataFrame())
            if df.empty:
                continue

            try:
                ticker_folder = model_root / ticker
                ticker_folder.mkdir(parents=True, exist_ok=True)
                model_name = (
                    f"{ticker}_{window_start.strftime('%Y%m%d')}_{window_end_inclusive.strftime('%Y%m%d')}.h5"
                )
                model_path = ticker_folder / model_name

                prepared = prepare_cumulative_quarter_data(
                    df=df,
                    sequence_length=sequence_length,
                    window_start=window_start,
                    window_end=window_end,
                )
                if model_path.exists() and not force_retrain:
                    reused = True
                    saved = False
                    print(
                        f"Reusing existing model for {ticker} in window "
                        f"{window_start.date()} -> {window_end_inclusive.date()}"
                    )
                else:
                    model = _train_cumulative_lstm_model(
                        prepared["X_train"],
                        prepared["y_train"],
                        sequence_length,
                    )
                    reused = False
                    saved = False
                    if save_models:
                        model.save(model_path)
                        saved = True
                        print(
                            f"Saved model for {ticker} in window "
                            f"{window_start.date()} -> {window_end_inclusive.date()}"
                        )

                results.append(
                    QuarterlyModelSaveResult(
                        window_start=window_start,
                        window_end=window_end_inclusive,
                        ticker=ticker,
                        model_path=str(model_path),
                        saved=saved,
                        reused=reused,
                    )
                )
            except Exception as exc:
                print(f"[SKIP] {ticker} in window {window_start.date()} -> {window_end.date()}: {exc}")

    if not results:
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "ticker",
                "model_path",
                "saved",
                "reused",
            ]
        )

    return pd.DataFrame([result.__dict__ for result in results])
