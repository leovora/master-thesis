from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

from scr.data.load_data import (
    FEATURES,
    _drop_repeated_header_row,
    _normalize_columns,
    load_ticker_data,
)
from scr.trade.trading_strategy import (
    moving_average_strategy,
    rate_of_change_strategy,
    rolling_std_deviation_strategy,
)


@dataclass
class RollingModelSaveResult:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    ticker: str
    model_path: str
    saved: bool
    reused: bool


def _clean_stock_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize, clean and sort a price dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df = _drop_repeated_header_row(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

    df = df.sort_index()
    return df


def _build_rolling_lstm_model(sequence_length: int, feature_dim: int) -> Sequential:
    """Build the LSTM used for rolling-window retraining."""
    model = Sequential()
    n_neurons = sequence_length * feature_dim
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(sequence_length, feature_dim)))
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _train_rolling_lstm_model(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Sequential:
    """
    Train the LSTM on a fixed rolling history using a chronological validation split.
    """
    if len(X) < 2:
        raise ValueError("Need at least two samples to train the model.")

    val_size = max(1, int(len(X) * 0.2))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    model = _build_rolling_lstm_model(sequence_length, X.shape[2])

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


def build_rolling_windows(
    simulation_start: str | pd.Timestamp,
    simulation_end: str | pd.Timestamp,
    months: int = 3,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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


def _prepare_rolling_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Prepare train/test sequences for one rolling window.

    Training uses only samples whose full sequence and target lie inside
    [train_start, train_end). Testing uses samples whose full sequence starts
    inside the rolling history and whose target lies in [train_end, test_end).
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

    data = data.sort_index()
    close_prices = data["close"].to_numpy(dtype=float)
    row_dates = pd.to_datetime(data.index).to_numpy()

    if len(data) <= sequence_length:
        raise ValueError(
            f"Not enough rows for sequence_length={sequence_length} inside the available data."
        )

    train_data = data[(data.index >= train_start) & (data.index < train_end)]
    if train_data.empty:
        raise ValueError(
            f"No training rows found in rolling window [{train_start.date()}, {train_end.date()})."
        )

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(train_data.to_numpy(dtype=float))
    y_scaler.fit(train_data["close"].to_numpy(dtype=float).reshape(-1, 1))

    X_scaled = x_scaler.transform(data.to_numpy(dtype=float))
    y_scaled = y_scaler.transform(close_prices.reshape(-1, 1)).flatten()

    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    seq_start_dates: List[pd.Timestamp] = []
    seq_target_dates: List[pd.Timestamp] = []

    for i in range(sequence_length, len(X_scaled)):
        seq_start_dates.append(pd.Timestamp(row_dates[i - sequence_length]))
        seq_target_dates.append(pd.Timestamp(row_dates[i]))
        X_seq.append(X_scaled[i - sequence_length : i])
        y_seq.append(float(y_scaled[i]))

    X_seq = np.asarray(X_seq, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    seq_start_dates = np.asarray(seq_start_dates, dtype="datetime64[ns]")
    seq_target_dates = np.asarray(seq_target_dates, dtype="datetime64[ns]")

    train_seq_mask = (
        (seq_start_dates >= np.datetime64(train_start))
        & (seq_target_dates < np.datetime64(train_end))
        & (seq_target_dates >= np.datetime64(train_start))
    )
    test_seq_mask = (
        (seq_start_dates >= np.datetime64(train_start))
        & (seq_target_dates >= np.datetime64(train_end))
        & (seq_target_dates < np.datetime64(test_end))
    )

    X_train = X_seq[train_seq_mask]
    y_train = y_seq[train_seq_mask]
    X_test = X_seq[test_seq_mask]
    y_test = y_seq[test_seq_mask]

    if len(X_train) == 0:
        raise ValueError(
            f"No train sequences generated in rolling window [{train_start.date()}, {train_end.date()})."
        )
    if len(X_test) == 0:
        raise ValueError(f"No test sequences generated in [{train_end.date()}, {test_end.date()}).")

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def prepare_rolling_window_data(
    df: pd.DataFrame,
    sequence_length: int,
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    lookback_years: int = 3,
) -> Dict[str, np.ndarray | MinMaxScaler | pd.Timestamp]:
    """
    Build a train/test split for one rolling quarterly window.

    The model trains only on the fixed historical window ending at `window_start`
    and spanning `lookback_years` years backwards.
    """
    test_start = pd.Timestamp(window_start).normalize()
    test_end = pd.Timestamp(window_end).normalize()
    train_start = test_start - pd.DateOffset(years=lookback_years)
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = _prepare_rolling_sequences(
        df=df,
        sequence_length=sequence_length,
        train_start=train_start,
        train_end=test_start,
        test_end=test_end,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_start": train_start,
        "train_end": test_start,
    }


def train_rolling_lstm_for_window(
    df: pd.DataFrame,
    sequence_length: int,
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    lookback_years: int = 3,
):
    """
    Train a fresh LSTM on the fixed rolling history available before `window_start`.
    """
    prepared = prepare_rolling_window_data(
        df,
        sequence_length,
        window_start,
        window_end,
        lookback_years=lookback_years,
    )
    model = _train_rolling_lstm_model(prepared["X_train"], prepared["y_train"], sequence_length)
    return model, prepared


def _align_series_to_min_length(
    predictions: Dict[str, np.ndarray],
    actuals: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    if not predictions:
        return {}, {}, 0

    min_len = min(len(values) for values in predictions.values())
    aligned_predictions = {ticker: values[:min_len] for ticker, values in predictions.items()}
    aligned_actuals = {ticker: values[:min_len] for ticker, values in actuals.items()}
    return aligned_predictions, aligned_actuals, min_len


def _resolve_strategy_name(strategy: str | Callable) -> str:
    if callable(strategy):
        return getattr(strategy, "__name__", "custom_strategy")
    return str(strategy)


def _zero_signal_dict_like(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {ticker: np.zeros(len(values), dtype=int) for ticker, values in predictions.items()}


def _generate_trading_signals(
    strategy: str | Callable,
    predictions: Dict[str, np.ndarray],
    actuals: Dict[str, np.ndarray],
    strategy_kwargs: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Resolve a strategy name or callable into a signal dictionary.

    Supported strategy names:
    - "ma" / "moving_average"
    - "roc" / "rate_of_change"
    - "bb" / "rolling_std"
    """
    strategy_kwargs = strategy_kwargs or {}

    if callable(strategy):
        try:
            return strategy(predictions, actuals, **strategy_kwargs)
        except TypeError:
            return strategy(predictions, **strategy_kwargs)

    key = str(strategy).strip().lower()
    if key in {"ma", "moving_average"}:
        long_window = int(strategy_kwargs.get("long_window", 20))
        if any(len(pred) < long_window for pred in predictions.values()):
            return _zero_signal_dict_like(predictions)
        return moving_average_strategy(predictions, **strategy_kwargs)
    if key in {"roc", "rate_of_change"}:
        window = int(strategy_kwargs.get("window", 14))
        if any(len(pred) <= window for pred in predictions.values()):
            return _zero_signal_dict_like(predictions)
        return rate_of_change_strategy(predictions, **strategy_kwargs)
    if key in {"bb", "bollinger", "rolling_std", "rolling_std_deviation"}:
        window = int(strategy_kwargs.get("window", 20))
        if any(len(pred) < window or len(actuals.get(ticker, [])) < window for ticker, pred in predictions.items()):
            return _zero_signal_dict_like(predictions)
        return rolling_std_deviation_strategy(predictions, actuals, **strategy_kwargs)

    raise ValueError(
        f"Unknown trading strategy '{strategy}'. "
        "Use 'ma', 'roc', 'bb' or pass a callable."
    )


def train_and_save_rolling_quarterly_models(
    tickers: Sequence[str],
    simulation_start: str | pd.Timestamp,
    simulation_end: str | pd.Timestamp,
    sequence_length: int,
    data_start: Optional[str | pd.Timestamp] = None,
    data_end: Optional[str | pd.Timestamp] = None,
    months: int = 3,
    lookback_years: int = 3,
    model_folder: str = "models/LSTM_rolling",
    save_models: bool = True,
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Train and save one quarterly rolling LSTM model per ticker and window.

    For each window:
    - train one fresh LSTM per ticker using only the last `lookback_years` years;
    - save the versioned model on disk for later prediction loading.
    """
    if lookback_years <= 0:
        raise ValueError("lookback_years must be greater than zero.")

    sim_start = pd.Timestamp(simulation_start).normalize()
    sim_end = pd.Timestamp(simulation_end).normalize()
    data_start_ts = pd.Timestamp(data_start).normalize() if data_start is not None else sim_start
    data_end_ts = pd.Timestamp(data_end).normalize() if data_end is not None else sim_end

    windows = build_rolling_windows(sim_start, sim_end, months=months)
    model_root = Path(model_folder)
    model_root.mkdir(parents=True, exist_ok=True)

    history_cache: Dict[str, pd.DataFrame] = {}
    results: List[RollingModelSaveResult] = []

    fetch_start = min(data_start_ts, sim_start - pd.DateOffset(years=lookback_years))

    for ticker in tickers:
        df = load_ticker_data(
            ticker,
            start_date=str(fetch_start.date()),
            end_date=str((data_end_ts + pd.Timedelta(days=1)).date()),
        )
        history_cache[ticker] = _clean_stock_frame(df)

    for window_start, window_end in windows:
        window_end_inclusive = window_end - pd.Timedelta(days=1)
        train_start = window_start - pd.DateOffset(years=lookback_years)
        print(
            "Running rolling window "
            f"{window_start.date()} -> {window_end_inclusive.date()} "
            f"with lookback from {train_start.date()}"
        )

        for ticker in tickers:
            df = history_cache.get(ticker, pd.DataFrame())
            if df.empty:
                continue

            try:
                ticker_folder = model_root / ticker
                ticker_folder.mkdir(parents=True, exist_ok=True)
                model_name = (
                    f"{ticker}_{window_start.strftime('%Y%m%d')}_{window_end_inclusive.strftime('%Y%m%d')}"
                    f"_lookback{lookback_years}y.h5"
                )
                model_path = ticker_folder / model_name

                prepared = prepare_rolling_window_data(
                    df=df,
                    sequence_length=sequence_length,
                    window_start=window_start,
                    window_end=window_end,
                    lookback_years=lookback_years,
                )
                if model_path.exists() and not force_retrain:
                    reused = True
                    saved = False
                    print(
                        f"Reusing existing model for {ticker} in window "
                        f"{window_start.date()} -> {window_end_inclusive.date()}"
                    )
                else:
                    model = _train_rolling_lstm_model(
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
                    RollingModelSaveResult(
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


# Backward-compatible alias: keep the old name available for existing notebooks.
run_rolling_quarterly_experiment = train_and_save_rolling_quarterly_models
