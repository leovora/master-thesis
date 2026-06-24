from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
DEFAULT_EVENT_LOG_PATH = Path("data/attack_dataset/attack_dataset.csv")
DEFAULT_PRICE_DATA_FOLDER = Path("data/validation_time")

NON_FEATURE_COLUMNS = {
    "attack_day",
    "attack_date",
    "target",
    "target_label",
    "target_signal",
    "baseline_signal",
    "attacked_signal",
    "success",
    "already_target",
    "valid",
    "assets",
    "baseline_label",
    "attacked_label",
    "min_delta_norm",
    "min_abs_delta_norm",
    "prediction_shift",
    "delta_final_cr",
    "delta_cumulative_return_in_period",
}

TECHNICAL_FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "quarter",
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_20d",
    "rsi_14",
    "atr_14",
    "bollinger_width_20",
    "macd",
    "trend_strength",
    "volume_ratio",
]

ENGINEERED_FEATURE_NAMES = [
    # Volatility regime
    "vol_ratio_5_20", "vol_spread", "bb_atr_ratio",
    # Trend ambiguity
    "rsi_neutrality", "trend_strength_sq", "macd_sign", "return_reversal",
    # Interactions
    "vol_x_weak_trend", "vol_x_low_volume", "rsi_neutral_x_atr", "bb_breakout",
    # Lag-derived
    "lag_mean_return", "lag_return_std", "lag_return_autocorr",
    "trend_break", "close_zscore", "volume_zscore", "volume_trend",
    "doji_score", "hl_range_mean",
    # Calendar
    "is_monday", "is_friday", "is_quarter_end_month",
    # Composite
    "vulnerability_score",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def _ensure_datetime_index(df: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Ensure the frame has a DatetimeIndex.

    The local CSV often do not preserve the date
    column. In that case we fall back to a business-day range starting from start_date.
    """
    df = _normalize_columns(df)
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        if start_date is None:
            raise ValueError(
                "Price data has no datetime index. Provide start_date or use a source with dates."
            )
        df = df.copy()
        df.index = pd.bdate_range(start=pd.Timestamp(start_date).normalize(), periods=len(df))

    return df.sort_index()


def _clean_price_frame(df: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    df = _ensure_datetime_index(df, start_date=start_date)
    if df.empty:
        return df

    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cleaned = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if cleaned.empty:
        raise ValueError("No numeric rows available after cleaning price data.")
    return cleaned.sort_index()


def _download_price_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Download OHLCV data from Yahoo Finance if available."""
    try:
        import yfinance as yf
    except Exception:
        return None

    try:
        end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        df = yf.download(
            ticker,
            start=str(pd.Timestamp(start_date).normalize().date()),
            end=str(end_ts.date()),
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        return _normalize_columns(df)
    except Exception:
        return None


def _history_covers_range(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return False

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    start_ok = df.index.min() <= (start_ts + pd.Timedelta(days=7))
    end_ok = df.index.max() >= end_ts
    return start_ok and end_ok


@lru_cache(maxsize=64)
def _load_price_history_cached(
    ticker: str,
    price_data_folder: str,
    start_date: str,
    end_date: str,
    prefer_download: bool,
) -> pd.DataFrame:
    """Load and cache OHLCV data for one ticker."""
    df = None
    if prefer_download:
        df = _download_price_history(ticker, start_date, end_date)

    if df is None or df.empty:
        csv_path = Path(price_data_folder) / "stock_data" / f"{ticker}_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing price data for {ticker}: {csv_path}")
        df = pd.read_csv(csv_path)

    cleaned = _clean_price_frame(df, start_date=start_date)
    if not _history_covers_range(cleaned, start_date, end_date):
        downloaded = _download_price_history(ticker, start_date, end_date)
        if downloaded is not None and not downloaded.empty:
            cleaned = _clean_price_frame(downloaded, start_date=start_date)
        if not _history_covers_range(cleaned, start_date, end_date):
            raise ValueError(
                f"Price data for {ticker} does not cover the requested range "
                f"[{start_date}, {end_date}] and download is unavailable or incomplete."
            )

    return cleaned


def load_price_history(
    ticker: str,
    price_data_folder: str | Path = DEFAULT_PRICE_DATA_FOLDER,
    start_date: str | None = None,
    end_date: str | None = None,
    prefer_download: bool = True,
) -> pd.DataFrame:
    """Load a standardized OHLCV frame for one ticker."""
    start = str(pd.Timestamp(start_date).normalize().date()) if start_date is not None else "1900-01-01"
    end = str(pd.Timestamp(end_date).normalize().date()) if end_date is not None else "2100-01-01"
    return _load_price_history_cached(
        ticker=ticker,
        price_data_folder=str(Path(price_data_folder)),
        start_date=start,
        end_date=end,
        prefer_download=prefer_download,
    ).copy()


def load_attack_event_log(
    csv_path: str | Path = DEFAULT_EVENT_LOG_PATH,
    *,
    drop_invalid: bool = True,
    drop_already_target: bool = True,
) -> pd.DataFrame:
    """Load the attack event log and apply the default filters."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing attack dataset: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    if drop_invalid and "valid" in df.columns:
        df = df[df["valid"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()

    if drop_already_target and "already_target" in df.columns:
        mask = ~df["already_target"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[mask].copy()

    if "success" in df.columns:
        success_mask = df["success"].astype(str).str.lower().isin({"true", "1", "yes"})
        failed_mask = ~success_mask

        zero_fill_cols = [
            "min_delta_norm",
            "min_abs_delta_norm",
            "prediction_shift",
            "delta_final_cr",
            "delta_cumulative_return_in_period",
        ]
        for col in zero_fill_cols:
            if col in df.columns:
                df.loc[failed_mask, col] = pd.to_numeric(df.loc[failed_mask, col], errors="coerce").fillna(0.0)

        if "baseline_signal" in df.columns and "attacked_signal" in df.columns:
            df.loc[failed_mask, "attacked_signal"] = df.loc[failed_mask, "baseline_signal"]
        if "baseline_label" in df.columns and "attacked_label" in df.columns:
            df.loc[failed_mask, "attacked_label"] = df.loc[failed_mask, "baseline_label"]

    return df.reset_index(drop=True)


def _infer_attack_date(
    price_df: pd.DataFrame,
    attack_day: int,
    sequence_length: int,
    simulation_start: str,
    simulation_end: str,
) -> pd.Timestamp:
    """Map the relative attack_day index to the actual trading date."""
    data = _clean_price_frame(price_df, start_date=simulation_start)
    close_prices = data["close"].to_numpy(dtype=float)
    target_dates = pd.to_datetime(data.index).to_numpy()[1:]

    if len(close_prices) <= sequence_length + 1:
        raise ValueError("Not enough price history to infer the attack date.")

    seq_dates = target_dates[sequence_length:]
    sim_start = np.datetime64(pd.Timestamp(simulation_start).normalize())
    sim_end = np.datetime64(pd.Timestamp(simulation_end).normalize())
    mask = (seq_dates >= sim_start) & (seq_dates <= sim_end)
    eligible_dates = seq_dates[mask]

    idx = int(attack_day)
    if idx < 0 or idx >= len(eligible_dates):
        raise IndexError(
            f"attack_day={attack_day} is out of bounds for the available period "
            f"(0..{len(eligible_dates) - 1})."
        )

    return pd.Timestamp(eligible_dates[idx]).normalize()


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < window:
        return float("nan")

    gains = delta.clip(lower=0).tail(window)
    losses = (-delta.clip(upper=0)).tail(window)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 0.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _atr(history: pd.DataFrame, window: int = 14) -> float:
    if len(history) < window + 1:
        return float("nan")

    high = history["high"]
    low = history["low"]
    close = history["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(window).mean())


def _bollinger_width(close: pd.Series, window: int = 20, num_std: int = 2) -> float:
    if len(close) < window:
        return float("nan")

    rolling = close.tail(window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    if mean == 0:
        return float("nan")

    upper = mean + num_std * std
    lower = mean - num_std * std
    return float((upper - lower) / mean)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> float:
    if len(close) < slow:
        return float("nan")
    ema_fast = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=slow, adjust=False).mean().iloc[-1]
    return float(ema_fast - ema_slow)


def _trend_strength(close: pd.Series, window: int = 20) -> float:
    if len(close) < window:
        return float("nan")

    window_close = close.tail(window).to_numpy(dtype=float)
    x = np.arange(len(window_close), dtype=float)
    slope = np.polyfit(x, window_close, 1)[0]
    mean_price = np.mean(window_close)
    if mean_price == 0:
        return float("nan")
    return float(slope / mean_price)


def _volume_ratio(volume: pd.Series, window: int = 20) -> float:
    if len(volume) < window:
        return float("nan")
    baseline = volume.tail(window).mean()
    if baseline == 0:
        return float("nan")
    return float(volume.iloc[-1] / baseline)


def _attack_target_from_delta(delta_cumulative_return_in_period: object) -> tuple[int, str]:
    """Map the cumulative return delta to the binary and textual target."""
    delta = pd.to_numeric(pd.Series([delta_cumulative_return_in_period]), errors="coerce").iloc[0]
    if pd.isna(delta):
        return 0, "non attack"

    if float(delta) < 0:
        return 1, "attack"
    return 0, "non attack"


def _build_ohlcv_window_features(
    history: pd.DataFrame,
    window: int,
) -> dict[str, float]:
    """Return a flattened, scale-normalized OHLCV history window.

    Each lag is reported as a ratio relative to the last close, which keeps the
    feature space usable across many tickers with different price levels.
    """
    history = history.tail(window).copy()
    if len(history) < window:
        raise ValueError(f"Not enough history for an OHLCV window of size {window}.")

    current_close = float(history["close"].iloc[-1])
    volume_baseline = float(history["volume"].mean())
    if current_close == 0:
        raise ValueError("Current close is zero; cannot normalize OHLC window.")
    if volume_baseline == 0:
        volume_baseline = 1.0

    features: dict[str, float] = {}
    reversed_history = history.iloc[::-1].reset_index(drop=True)
    for lag, (_, row) in enumerate(reversed_history.iterrows()):
        features[f"ohlcv_lag_{lag}_open_rel"] = float(row["open"] / current_close)
        features[f"ohlcv_lag_{lag}_high_rel"] = float(row["high"] / current_close)
        features[f"ohlcv_lag_{lag}_low_rel"] = float(row["low"] / current_close)
        features[f"ohlcv_lag_{lag}_close_rel"] = float(row["close"] / current_close)
        features[f"ohlcv_lag_{lag}_volume_rel"] = float(row["volume"] / volume_baseline)
    return features


def _build_technical_features(history: pd.DataFrame) -> dict[str, float]:
    """Compute the technical indicators requested for the ML model."""
    close = history["close"]
    volume = history["volume"]

    feats = {
        "return_1d": float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else float("nan"),
        "return_5d": float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else float("nan"),
        "return_10d": float(close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else float("nan"),
        "volatility_5d": float(_daily_returns(close).tail(5).std(ddof=0)) if len(close) >= 6 else float("nan"),
        "volatility_20d": float(_daily_returns(close).tail(20).std(ddof=0)) if len(close) >= 21 else float("nan"),
        "rsi_14": _rsi(close, window=14),
        "atr_14": _atr(history, window=14),
        "bollinger_width_20": _bollinger_width(close, window=20, num_std=2),
        "macd": _macd(close, fast=12, slow=26),
        "trend_strength": _trend_strength(close, window=20),
        "volume_ratio": _volume_ratio(volume, window=20),
    }
    return feats


def _model_feature_columns(ohlcv_window: int) -> list[str]:
    """Return the exact model-input column order requested by the pipeline."""
    columns = ["day_of_week", "month", "quarter"]
    for lag in range(ohlcv_window):
        columns.extend(
            [
                f"ohlcv_lag_{lag}_open_rel",
                f"ohlcv_lag_{lag}_high_rel",
                f"ohlcv_lag_{lag}_low_rel",
                f"ohlcv_lag_{lag}_close_rel",
                f"ohlcv_lag_{lag}_volume_rel",
            ]
        )
    columns.extend(TECHNICAL_FEATURE_COLUMNS)
    return columns


def build_attack_day_feature_row(
    attack_row: pd.Series,
    *,
    price_data_folder: str | Path = DEFAULT_PRICE_DATA_FOLDER,
    simulation_start: str,
    simulation_end: str,
    data_start: str | None = None,
    sequence_length: int = 50,
    ohlcv_window: int = 20,
    prefer_download: bool = True,
) -> dict[str, object]:
    """Build a single feature row for one attack event."""
    ticker = str(attack_row["attacked_ticker"])
    price_start = data_start or simulation_start
    price_history = load_price_history(
        ticker,
        price_data_folder=price_data_folder,
        start_date=price_start,
        end_date=simulation_end,
        prefer_download=prefer_download,
    )
    attack_date = _infer_attack_date(
        price_history,
        attack_day=int(attack_row["attack_day"]),
        sequence_length=sequence_length,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
    )

    history = price_history.loc[:attack_date].copy()
    min_history = max(ohlcv_window, 26, 20, 14, 10) + 1
    if len(history) < min_history:
        raise ValueError(
            f"Not enough history for {ticker} on {attack_date.date()}: "
            f"need at least {min_history} rows, got {len(history)}."
        )

    features: dict[str, object] = {
        **attack_row.to_dict(),
        "attack_date": attack_date,
        "day_of_week": int(attack_date.dayofweek),
        "month": int(attack_date.month),
        "quarter": int(attack_date.quarter),
    }
    features.update(_build_ohlcv_window_features(history, window=ohlcv_window))
    features.update(_build_technical_features(history))
    target, target_label = _attack_target_from_delta(
        attack_row.get("delta_cumulative_return_in_period")
    )
    features["target"] = target
    features["target_label"] = target_label
    return features


def build_attack_day_feature_frame(
    attack_log: pd.DataFrame,
    *,
    price_data_folder: str | Path = DEFAULT_PRICE_DATA_FOLDER,
    simulation_start: str,
    simulation_end: str,
    data_start: str | None = None,
    sequence_length: int = 50,
    ohlcv_window: int = 20,
    prefer_download: bool = True,
    drop_invalid: bool = True,
    drop_already_target: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Enrich the attack event log with calendar, OHLCV and indicator features."""
    df = attack_log.copy()
    if drop_invalid and "valid" in df.columns:
        df = df[df["valid"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    if drop_already_target and "already_target" in df.columns:
        mask = ~df["already_target"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[mask].copy()

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            rows.append(
                build_attack_day_feature_row(
                    row,
                    price_data_folder=price_data_folder,
                    simulation_start=simulation_start,
                    simulation_end=simulation_end,
                    data_start=data_start,
                    sequence_length=sequence_length,
                    ohlcv_window=ohlcv_window,
                    prefer_download=prefer_download,
                )
            )
        except Exception as exc:
            skipped += 1
            if verbose:
                print(
                    f"[SKIP] {row.get('setup', 'NA')} / {row.get('attacked_ticker', 'NA')} / "
                    f"day={row.get('attack_day', 'NA')}: {exc}"
                )

    feature_df = pd.DataFrame(rows)
    if verbose:
        print(f"Built {len(feature_df)} feature rows. Skipped {skipped}.")
    return feature_df


def encode_attack_day_features(
    feature_df: pd.DataFrame,
    *,
    categorical_cols: Sequence[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns for tree-based ML models."""
    if feature_df.empty:
        return feature_df.copy(), []

    categorical_cols = list(
        categorical_cols
        or ["setup", "model", "strategy", "attacked_ticker", "objective", "timing_policy"]
    )
    present = [col for col in categorical_cols if col in feature_df.columns]
    encoded = pd.get_dummies(feature_df, columns=present, drop_first=drop_first, dtype=int)

    feature_columns = [
        col
        for col in encoded.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(encoded[col])
    ]
    return encoded, feature_columns


def prepare_attack_day_dataset(
    csv_path: str | Path = DEFAULT_EVENT_LOG_PATH,
    *,
    price_data_folder: str | Path = DEFAULT_PRICE_DATA_FOLDER,
    simulation_start: str,
    simulation_end: str,
    data_start: str | None = None,
    sequence_length: int = 50,
    ohlcv_window: int = 20,
    target_col: str = "target",
    include_categoricals: bool = True,
    drop_invalid: bool = True,
    drop_already_target: bool = True,
    prefer_download: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """End-to-end preprocessing for the attack-day classification dataset.

    Returns a dictionary with:
    - ``raw``: filtered event log
    - ``features``: engineered feature frame
    - ``X``: model-ready feature matrix
    - ``y``: target series
    - ``feature_columns``: selected predictor columns
    """
    attack_log = load_attack_event_log(
        csv_path,
        drop_invalid=drop_invalid,
        drop_already_target=drop_already_target,
    )

    feature_df = build_attack_day_feature_frame(
        attack_log,
        price_data_folder=price_data_folder,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        data_start=data_start,
        sequence_length=sequence_length,
        ohlcv_window=ohlcv_window,
        prefer_download=prefer_download,
        drop_invalid=False,
        drop_already_target=False,
        verbose=verbose,
    )

    if feature_df.empty:
        raise ValueError("Feature engineering produced an empty dataset.")

    if target_col not in feature_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    if include_categoricals:
        encoded_df, feature_columns = encode_attack_day_features(feature_df)
        feature_columns = [col for col in feature_columns if col != target_col]
    else:
        encoded_df = feature_df.copy()
        feature_columns = [
            col
            for col in encoded_df.columns
            if col not in NON_FEATURE_COLUMNS
            and col != target_col
            and pd.api.types.is_numeric_dtype(encoded_df[col])
        ]

    if target_col not in encoded_df.columns:
        raise ValueError(f"Target column '{target_col}' was dropped unexpectedly.")

    y = encoded_df[target_col].astype(int)
    X = encoded_df[feature_columns].copy()

    return {
        "raw": attack_log,
        "features": encoded_df,
        "X": X,
        "y": y,
        "feature_columns": feature_columns,
        "target_column": target_col,
    }


def build_model_input_dataframe(
    csv_path: str | Path = DEFAULT_EVENT_LOG_PATH,
    *,
    price_data_folder: str | Path = DEFAULT_PRICE_DATA_FOLDER,
    start_date: str,
    end_date: str,
    data_start: str | None = None,
    sequence_length: int = 50,
    ohlcv_window: int = 20,
    drop_invalid: bool = True,
    drop_already_target: bool = True,
    prefer_download: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the model-ready dataframe from the attack event log.

    The dataframe includes:
    - calendar features: day_of_week, month, quarter
    - OHLCV history of the last ``ohlcv_window`` days
    - technical indicators requested by the user

    Parameters
    ----------
    start_date, end_date:
        Date range used to map ``attack_day`` to the underlying trading dates
        and to select the history window.
    """
    result = prepare_attack_day_dataset(
        csv_path=csv_path,
        price_data_folder=price_data_folder,
        simulation_start=start_date,
        simulation_end=end_date,
        data_start=data_start,
        sequence_length=sequence_length,
        ohlcv_window=ohlcv_window,
        target_col="target",
        include_categoricals=False,
        drop_invalid=drop_invalid,
        drop_already_target=drop_already_target,
        prefer_download=prefer_download,
        verbose=verbose,
    )

    feature_df = result["features"].copy()
    model_columns = _model_feature_columns(ohlcv_window)
    missing = [col for col in model_columns if col not in feature_df.columns]
    if missing:
        raise ValueError(
            "The engineered dataset is missing required model features: "
            f"{missing}"
        )

    return feature_df[model_columns + ["target", "target_label"]].copy()


def chronological_train_test_split(
    feature_df: pd.DataFrame,
    *,
    split_column: str = "attack_day",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split helper for the enriched feature table."""
    if feature_df.empty:
        return feature_df.copy(), feature_df.copy()

    if split_column not in feature_df.columns:
        raise ValueError(f"Missing split column: {split_column}")

    ordered = feature_df.sort_values(split_column).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(ordered) - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()

def feature_correlation_analysis(
    feature_df: pd.DataFrame,
    *,
    target_col: str = "target",
    top_k: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> pd.Series:
    """
    Compute a simple feature analysis based on Pearson correlation with the target.
    """

    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "feature_correlation_analysis requires seaborn. "
        )

    if target_col not in feature_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_col}' must be numeric."
        )

    corr_with_target = (
        numeric_df.corr(method="pearson")[target_col]
        .drop(target_col)
        .sort_values(key=np.abs, ascending=False)
    )

    if corr_with_target.empty:
        raise ValueError("No numeric features available.")

    top_features = corr_with_target.head(top_k).index.tolist()

    heatmap_cols = top_features + [target_col]
    corr_matrix = numeric_df[heatmap_cols].corr(method="pearson")

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=False,
    )
    plt.title(
        f"Top {min(top_k, len(top_features))} Feature Correlations with '{target_col}'"
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print("\nTop features correlated with target:\n")
    print(corr_with_target.head(top_k))

    return corr_with_target

# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Build additional features from the existing feature matrix.

    The goal is to capture market conditions that make an adversarial attack
    more likely to succeed.
      - Anomaly / z-score features: how unusual is today vs recent history?
      - Volatility regime: is the market in a noisy, hard-to-predict state?
      - Interaction features: conditions that jointly favour attack success
      - Lag-derived features: autocorrelation, trend breaks, candle patterns
        extracted from the 20 OHLCV lags already present in X
    """
    df = X.copy()

    # 1. VOLATILITY REGIME FEATURES

    # Ratio between short-term and long-term volatility.
    # > 1  →  volatility expanding (regime change, good for attack)
    # < 1  →  volatility contracting (calm market, harder to fool)
    df["vol_ratio_5_20"] = df["volatility_5d"] / (df["volatility_20d"] + 1e-8)

    # Absolute level: how far is 5d vol from the "normal" 20d vol?
    df["vol_spread"] = df["volatility_5d"] - df["volatility_20d"]

    # Bollinger width normalised by ATR — captures relative band expansion
    df["bb_atr_ratio"] = df["bollinger_width_20"] / (df["atr_14"] + 1e-8)

    # 2. TREND AMBIGUITY FEATURES

    # RSI distance from neutrality (50): close to 50 = no momentum = ambiguous
    df["rsi_neutrality"] = 1.0 - (df["rsi_14"] - 50).abs() / 50.0

    # Trend strength squared: penalises weak trends more
    df["trend_strength_sq"] = df["trend_strength"] ** 2

    # MACD sign: positive = uptrend, negative = downtrend
    df["macd_sign"] = np.sign(df["macd"])

    # Return reversal: short return opposing long return = unstable trend
    df["return_reversal"] = -np.sign(df["return_1d"]) * np.sign(df["return_10d"])

    # 3. INTERACTION FEATURES

    # High volatility + weak trend = ideal attack conditions
    df["vol_x_weak_trend"] = df["volatility_5d"] * (1.0 - df["trend_strength"].abs())

    # High vol + low volume = thin, noisy market
    df["vol_x_low_volume"] = df["volatility_5d"] * (1.0 / (df["volume_ratio"] + 1e-8))

    # RSI neutrality + high ATR = uncertain direction + large moves
    df["rsi_neutral_x_atr"] = df["rsi_neutrality"] * df["atr_14"]

    # Bollinger squeeze broken: high bb_width + high vol = breakout regime
    df["bb_breakout"] = df["bollinger_width_20"] * df["volatility_5d"]

    # 4. OHLCV-LAG DERIVED FEATURES

    close_cols  = [f"ohlcv_lag_{i}_close_rel"  for i in range(20)]
    volume_cols = [f"ohlcv_lag_{i}_volume_rel" for i in range(20)]
    high_cols   = [f"ohlcv_lag_{i}_high_rel"   for i in range(20)]
    low_cols    = [f"ohlcv_lag_{i}_low_rel"    for i in range(20)]

    available_close  = [c for c in close_cols  if c in df.columns]
    available_volume = [c for c in volume_cols if c in df.columns]
    available_high   = [c for c in high_cols   if c in df.columns]
    available_low    = [c for c in low_cols    if c in df.columns]

    if available_close:
        close_series = df[available_close]

        # Average daily return over the window (lag 0 = today, lag 1 = yesterday)
        # close_rel values are already relative to current close, so
        # differences approximate daily returns.
        close_diffs = close_series.diff(axis=1).iloc[:, 1:]
        df["lag_mean_return"]    = close_diffs.mean(axis=1)
        df["lag_return_std"]     = close_diffs.std(axis=1)   # realised vol in window

        # Autocorrelation of returns (lag-1): positive = momentum, negative = reversal
        def _autocorr_row(row):
            r = row.dropna().values
            if len(r) < 4:
                return 0.0
            return float(pd.Series(r).autocorr(lag=1) or 0.0)

        df["lag_return_autocorr"] = close_diffs.apply(_autocorr_row, axis=1)

        # Trend break: is the most recent 5-day return opposing the prior 15-day return?
        if len(available_close) >= 16:
            recent_ret   = df[available_close[0]]  - df[available_close[4]]   # last 5 days
            prior_ret    = df[available_close[5]]  - df[available_close[15]]  # days 5-15
            df["trend_break"] = (-np.sign(recent_ret) * np.sign(prior_ret)).astype(float)
        else:
            df["trend_break"] = 0.0

        # Z-score of today's close relative to the 20-day window
        rolling_mean = close_series.mean(axis=1)
        rolling_std  = close_series.std(axis=1)
        df["close_zscore"] = (
            (df[available_close[0]] - rolling_mean) / (rolling_std + 1e-8)
        )

    if available_volume:
        volume_series = df[available_volume]
        # Volume z-score: unusually high/low volume signals anomalous activity
        vol_mean = volume_series.mean(axis=1)
        vol_std  = volume_series.std(axis=1)
        df["volume_zscore"] = (
            (df[available_volume[0]] - vol_mean) / (vol_std + 1e-8)
        )
        # Volume trend: is volume increasing or decreasing over the window?
        if len(available_volume) >= 5:
            df["volume_trend"] = (
                df[available_volume[:5]].mean(axis=1)
                - df[available_volume[5:10]].mean(axis=1)
                if len(available_volume) >= 10
                else df[available_volume[0]] - df[available_volume[-1]]
            )

    if available_high and available_low and available_close:
        # Average candle body size relative to range (doji detection)
        # Small body + large range = indecision candle = ambiguous market
        open_cols = [f"ohlcv_lag_{i}_open_rel" for i in range(20)]
        available_open = [c for c in open_cols if c in df.columns]
        if available_open:
            body   = (df[available_close[0]] - df[available_open[0]]).abs()
            candle_range = df[available_high[0]] - df[available_low[0]] + 1e-8
            df["doji_score"] = 1.0 - (body / candle_range)  # 1 = full doji, 0 = marubozu

        # High-low range normalised: large range = high intraday uncertainty
        df["hl_range_mean"] = (
            df[available_high].values - df[available_low].values
        ).mean(axis=1)

    # 5. CALENDAR INTERACTION FEATURES

    # Monday (0) and Friday (4) often have anomalous return patterns
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # End of quarter: window dressing increases volatility
    df["is_quarter_end_month"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    # 6. COMPOSITE VULNERABILITY SCORE
    
    score_components = []
    if "vol_ratio_5_20"   in df.columns: score_components.append(_minmax(df["vol_ratio_5_20"]))
    if "rsi_neutrality"   in df.columns: score_components.append(df["rsi_neutrality"])
    if "bb_atr_ratio"     in df.columns: score_components.append(_minmax(df["bb_atr_ratio"]))
    if "trend_break"      in df.columns: score_components.append((df["trend_break"] + 1) / 2)

    if score_components:
        df["vulnerability_score"] = sum(score_components) / len(score_components)

    return df


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1]; returns 0.5 if constant."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)
