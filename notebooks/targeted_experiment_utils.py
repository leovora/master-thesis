from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.append(str(Path(__file__).resolve().parent / "scr"))

from trade.trading_strategy import (
    moving_average_strategy,
    rate_of_change_strategy,
    rolling_std_deviation_strategy,
)
from trade.backtest import simulate_trades_with_allocation
from utils import calculate_cumulative_returns, preprocess_data

FEATURES = ["high", "low", "open", "close", "volume"]
CLOSE_FEATURE_INDEX = FEATURES.index("close")

SIGNAL_LABELS = {-1: "SELL", 0: "HOLD", 1: "BUY"}
TARGET_SIGNALS = {"force_buy": 1, "force_sell": -1}

STRATEGIES = {
    "MA": lambda predictions, actuals: moving_average_strategy(
        predictions, short_window=5, long_window=20, threshold=0.01
    ),
    "BB": lambda predictions, actuals: rolling_std_deviation_strategy(
        predictions, actuals, window=20, num_std_dev=2
    ),
    "ROC": lambda predictions, actuals: rate_of_change_strategy(
        predictions, window=14, threshold=0.01
    ),
}

# ---------------------------------------------------------------------------
# ATSSetup
# ---------------------------------------------------------------------------

@dataclass
class ATSSetup:
    '''Combines tickers, strategy and ML model to test'''

    name: str
    tickers: Tuple[str, ...]
    strategy_name: str
    model_name: str = "LSTM"
    attacked_ticker: str = "GOOGL"

    @property
    def strategy(self):
        return STRATEGIES[self.strategy_name]

    def model_folder(self, model_folders):
        return model_folders[self.model_name]


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

_DATA_CACHE = {}
_MODEL_CACHE = {}
_PREDICTION_CACHE = {}


def preprocess_ticker_from_csv(ticker, sequence_length, project_root):
    result = preprocess_data(ticker, start_date=None, end_date=None, sequence_length=sequence_length, source="csv", data_folder=str(project_root / "data"))
    if result is None:
        raise FileNotFoundError(f"Missing CSV for {ticker}")

    _, _, X_test, y_test, y_min, y_max, _, _ = result
    y_range = y_max - y_min if y_max != y_min else 1.0

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_actual": y_test * y_range + y_min,
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_range,
    }

def get_ticker_data(ticker, sequence_length, project_root):
    key = (ticker, sequence_length)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = preprocess_ticker_from_csv(ticker, sequence_length, project_root)
    return _DATA_CACHE[key]


def get_model(ticker, model_folder):
    model_path = Path(model_folder) / f"{ticker}_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {ticker}: {model_path}")

    key = (str(model_path.resolve()), ticker)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(model_path, compile=False)
    return _MODEL_CACHE[key]


def predict_ticker(ticker, model_folder, sequence_length, project_root, X_override=None):
    data = get_ticker_data(ticker, sequence_length, project_root)
    X = data["X_test"] if X_override is None else X_override
    model = get_model(ticker, model_folder)
    pred_norm = model.predict(X, verbose=0).reshape(-1)
    return pred_norm * data["y_range"] + data["y_min"]


def get_predictions_and_actuals(tickers, model_folder, sequence_length, project_root):  
    tickers = tuple(tickers)
    key = (tickers, str(Path(model_folder).resolve()), sequence_length)
    if key in _PREDICTION_CACHE:
        return _PREDICTION_CACHE[key]

    predictions, actuals = {}, {}
    for ticker in tickers:
        predictions[ticker] = predict_ticker(ticker, model_folder, sequence_length, project_root)
        actuals[ticker] = get_ticker_data(ticker, sequence_length, project_root)["y_actual"]

    _PREDICTION_CACHE[key] = (predictions, actuals)
    return predictions, actuals


def final_cumulative_return(returns):
    if len(returns) == 0:
        return float("nan")
    return float(calculate_cumulative_returns(returns)[-1])


# ---------------------------------------------------------------------------
# Attack execution
# ---------------------------------------------------------------------------

def make_delta_grid(max_abs_delta = 0.35, steps = 35):
    '''Build perturbations in increasing absolute size, testing both directions'''
    magnitudes = np.linspace(0, max_abs_delta, steps + 1)
    deltas = [0.0]
    for m in magnitudes[1:]:
        deltas.extend([float(m), float(-m)])
    return deltas


def signal_at(signals, ticker, attack_day):
    '''Return the trading signal for one ticker/day, or None if unavailable'''
    ticker_signals = signals.get(ticker)
    if ticker_signals is None or attack_day < 0 or attack_day >= len(ticker_signals):
        return None
    return int(ticker_signals[attack_day])


def evaluate_targeted_attack(setup, attack_day, target_signal, model_folders, project_root, sequence_length, max_abs_delta = 0.35, steps = 35):
    ''' Evaluate one targeted attack by searching for a perturbation that forces the target signal'''
    mf = setup.model_folder(model_folders)

    # Build the clean ATS baseline
    predictions_base, actuals = get_predictions_and_actuals(setup.tickers, mf, sequence_length, project_root)
    signals_base = setup.strategy(predictions_base, actuals)
    baseline_signal = signal_at(signals_base, setup.attacked_ticker, attack_day)

    _base_row = {
        "setup": setup.name,
        "model": setup.model_name,
        "strategy": setup.strategy_name,
        "assets": ",".join(setup.tickers),
        "attacked_ticker": setup.attacked_ticker,
        "attack_day": attack_day,
        "target_signal": target_signal,
        "target_label": SIGNAL_LABELS[target_signal],
    }

    # Skip invalid attack days
    if baseline_signal is None:
        return {
            **_base_row,
            "baseline_signal": float("nan"),
            "baseline_label": "NA",
            "attacked_signal": float("nan"),
            "attacked_label": "NA",
            "success": False,
            "already_target": False,
            "min_delta_norm": float("nan"),
            "min_abs_delta_norm": float("nan"),
            "prediction_shift": float("nan"),
            "delta_final_cr": float("nan"),
            "valid": False,
        }

    baseline_returns = simulate_trades_with_allocation(predictions_base, actuals, signals_base)
    baseline_final_cr = final_cumulative_return(baseline_returns)

    # Get original data from attacked ticker
    data = get_ticker_data(setup.attacked_ticker, sequence_length, project_root)
    X_original = data["X_test"]
    baseline_attacked_pred = predictions_base[setup.attacked_ticker]

    best = None
    # Search the smallest perturbation that reaches the target signal
    for delta in make_delta_grid(max_abs_delta=max_abs_delta, steps=steps):
        attacked_X = X_original.copy()
        # Perturb only the last "close" value of the attacked input window
        attacked_X[attack_day, -1, CLOSE_FEATURE_INDEX] = np.clip(
            attacked_X[attack_day, -1, CLOSE_FEATURE_INDEX] + delta, 0.0, 1.0
        )

        attacked_pred = predict_ticker(setup.attacked_ticker, mf, sequence_length, project_root, X_override=attacked_X)

        # Replace the target ticker's baseline predictions with adversarial ones
        predictions_attack = {t: v.copy() for t, v in predictions_base.items()}
        predictions_attack[setup.attacked_ticker] = attacked_pred[: len(baseline_attacked_pred)]

        # Recompute the trading decision after the attacked prediction
        signals_attack = setup.strategy(predictions_attack, actuals)
        attacked_signal = signal_at(signals_attack, setup.attacked_ticker, attack_day)

        if attacked_signal == target_signal:
            # Store the first successful delta
            attacked_returns = simulate_trades_with_allocation(predictions_attack, actuals, signals_attack)
            best = {
                "attacked_signal": attacked_signal,
                "min_delta_norm": delta,
                "min_abs_delta_norm": abs(delta),
                "prediction_shift": float(
                    predictions_attack[setup.attacked_ticker][attack_day]
                    - baseline_attacked_pred[attack_day]
                ),
                "delta_final_cr": final_cumulative_return(attacked_returns) - baseline_final_cr,
            }
            break

    if best is None:
        # No tested perturbation reached the requested target
        best = {
            "attacked_signal": float("nan"),
            "min_delta_norm": float("nan"),
            "min_abs_delta_norm": float("nan"),
            "prediction_shift": float("nan"),
            "delta_final_cr": float("nan"),
        }

    attacked_signal = best["attacked_signal"]
    return {
        **_base_row,
        "baseline_signal": baseline_signal,
        "baseline_label": SIGNAL_LABELS.get(baseline_signal, "NA"),
        "attacked_signal": attacked_signal,
        "attacked_label": SIGNAL_LABELS.get(attacked_signal, "NA"),
        "success": attacked_signal == target_signal,
        "already_target": baseline_signal == target_signal,
        "valid": True,
        **best,
    }

def run_targeted_attack_experiment(
    setups,
    attack_days,
    model_folders,
    project_root,
    target_signals = TARGET_SIGNALS,
    sequence_length = 50,
    max_abs_delta = 0.35,
    steps = 35,
    timing_policy = "manual",
    verbose = True
):
    rows = []
    attack_days = list(attack_days)
    total = len(setups) * len(attack_days) * len(target_signals)
    done = 0

    for setup in setups:
        for attack_day in attack_days:
            for objective_name, target_signal in target_signals.items():
                row = evaluate_targeted_attack(
                    setup=setup,
                    attack_day=int(attack_day),
                    target_signal=target_signal,
                    model_folders=model_folders,
                    project_root=project_root,
                    sequence_length=sequence_length,
                    max_abs_delta=max_abs_delta,
                    steps=steps,
                )
                row["objective"] = objective_name
                row["timing_policy"] = timing_policy
                rows.append(row)
                done += 1
                if verbose and (done % 5 == 0 or done == total):
                    print(f"  {done}/{total} completed trials")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attack timing policies
# ---------------------------------------------------------------------------

def _rolling_std(values, window):
    return pd.Series(values).rolling(window=window, min_periods=window).std().to_numpy()


def _rate_of_change(values, window):
    roc = np.full(len(values), np.nan)
    if len(values) > window:
        denom = np.where(values[:-window] == 0, np.nan, values[:-window])
        roc[window:] = (values[window:] - values[:-window]) / denom
    return roc


def _ma_decision_boundary_distance(predictions, short_window = 5, long_window = 20, threshold = 0.01):
    """Distance from the closest MA decision threshold for each signal index"""
    distance = np.full(len(predictions), np.nan)
    if len(predictions) < long_window:
        return distance
    short_mavg = np.convolve(predictions, np.ones(short_window), "valid") / short_window
    long_mavg = np.convolve(predictions, np.ones(long_window), "valid") / long_window
    mavg_diff = short_mavg[long_window - short_window :] - long_mavg
    shifted = np.minimum(np.abs(mavg_diff - threshold), np.abs(mavg_diff + threshold))
    shifted = np.pad(shifted, (1, 0), "constant")[:-1] # Shift the signal by 1 to avoid lookahead bias
    distance[: len(shifted)] = shifted
    return distance


def _roc_decision_boundary_distance(predictions, window = 20, threshold = 0.01):
    '''Distance from the closest ROC decision threshold for each signal index'''
    roc = _rate_of_change(predictions, window)
    valid = ~np.isnan(roc)
    distance = np.full(len(predictions), np.nan)
    shifted = np.minimum(np.abs(roc - threshold), np.abs(roc + threshold))
    distance[valid] = shifted[valid]
    return distance


def _bollinger_decision_boundary_distance(predictions, window = 20, num_std = 2):
    """Distance from the closest BB decision threshold for each signal index"""
    series = pd.Series(predictions)
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return np.minimum(np.abs(series - upper), np.abs(series - lower)).to_numpy()


def candidate_attack_days(setup, model_folders, project_root, sequence_length = 50, min_day = 30, max_day = None):
    '''Return the valid day indices that can be considered for launching an attack'''
    predictions, actuals = get_predictions_and_actuals(setup.tickers, setup.model_folder(model_folders), sequence_length, project_root)
    signals = setup.strategy(predictions, actuals)
    upper = min(len(signals[setup.attacked_ticker]), len(predictions[setup.attacked_ticker]))
    if max_day is not None:
        upper = min(upper, max_day)
    return np.arange(min_day, upper)


def timing_scores_for_setup(setup, model_folders, project_root, sequence_length = 50, volatility_window = 20, trend_window = 20, min_day = 30, max_day = None):
    '''Compute timing-related metrics used to rank candidate attack days for one setup'''
    predictions, actuals = get_predictions_and_actuals(setup.tickers, setup.model_folder(model_folders), sequence_length, project_root)
    ticker = setup.attacked_ticker
    days = candidate_attack_days(setup, model_folders, project_root, sequence_length, min_day, max_day)

    actual = actuals[ticker]
    pred = predictions[ticker]

    scores = pd.DataFrame({"attack_day": days})
    scores["actual_volatility"] = _rolling_std(actual, volatility_window)[days]
    scores["prediction_volatility"] = _rolling_std(pred, volatility_window)[days]
    scores["abs_trend"] = np.abs(_rate_of_change(actual, trend_window)[days])
    scores["baseline_signal"] = setup.strategy(predictions, actuals)[ticker][days]

    if setup.strategy_name == "MA":
        scores["strategy_boundary_distance"] = _ma_decision_boundary_distance(pred)[days]

    elif setup.strategy_name == "ROC":
        scores["strategy_boundary_distance"] = _roc_decision_boundary_distance(pred)[days]

    elif setup.strategy_name == "BB":
        scores["strategy_boundary_distance"] = _bollinger_decision_boundary_distance(pred)[days]

    return scores


def select_attack_days_by_policy(setup, policy, model_folders, project_root, n_days = 12, sequence_length = 50, min_day = 30, max_day = None):
    '''Select attack days according to the requested timing policy and ranking criterion'''
    scores = timing_scores_for_setup(setup, model_folders, project_root, sequence_length,min_day=min_day, max_day=max_day)

    if policy == "uniform":
        candidates = scores["attack_day"].to_numpy()
        positions = np.linspace(0, len(candidates) - 1, n_days).round().astype(int)
        return candidates[positions]

    score_columns = {
        "high_volatility": ("actual_volatility", False),
        "high_pred_volatility": ("prediction_volatility", False),
        "strong_trend": ("abs_trend", False),
        "weak_trend": ("abs_trend", True),
        "near_strategy_boundary": ("strategy_boundary_distance", True),
    }
    if policy not in score_columns:
        raise ValueError(f"Unknown timing policy: {policy}")

    column, ascending = score_columns[policy]
    selected = (
        scores.dropna(subset=[column])
        .sort_values(column, ascending=ascending)
        .head(n_days)
        .sort_values("attack_day")
    )
    return selected["attack_day"].to_numpy(dtype=int)


def build_attack_day_plan(setups, policies, model_folders, project_root, n_days, sequence_length, min_day, max_day = None):
    '''Build a mapping from each setup-policy pair to its selected attack days'''
    plan = {}
    for setup in setups:
        for policy in policies:
            plan[(setup.name, policy)] = select_attack_days_by_policy(
                setup, policy, model_folders, project_root,
                n_days=n_days, sequence_length=sequence_length,
                min_day=min_day, max_day=max_day,
            )
    return plan


# ---------------------------------------------------------------------------
# Analize results
# ---------------------------------------------------------------------------

def compute_summary_by_setup(results):
    valid = results[results["valid"]].copy()
    valid["success"] = valid["success"].astype(bool)
    valid["already_target"] = valid["already_target"].astype(bool)
    valid["nontrivial_success"] = valid["success"] & ~valid["already_target"]

    summary = (
        valid.groupby(
            ["setup", "model", "strategy", "timing_policy", "objective"], as_index=False
        ).agg(
            trials=("success", "size"),
            attack_success_rate=("success", "mean"),
            baseline_target_rate=("already_target", "mean"),
            nontrivial_trials=("already_target", lambda x: int((~x).sum())),
            nontrivial_successes=("nontrivial_success", "sum"),
            mean_min_abs_delta_norm=("min_abs_delta_norm", "mean"),
            median_min_abs_delta_norm=("min_abs_delta_norm", "median"),
            mean_delta_final_cr=("delta_final_cr", "mean"),
        )
    )
    summary["nontrivial_attack_success_rate"] = np.where(
        summary["nontrivial_trials"] > 0,
        summary["nontrivial_successes"] / summary["nontrivial_trials"],
        np.nan,
    )
    return summary.sort_values(
        ["objective", "nontrivial_attack_success_rate", "attack_success_rate"],
        ascending=[True, False, False],
    )


def compute_timing_summaries(results):
    valid = results[results["valid"]].copy()
    valid["success"] = valid["success"].astype(bool)
    valid["already_target"] = valid["already_target"].astype(bool)
    valid["nontrivial_success"] = valid["success"] & ~valid["already_target"]

    timing_summary = (
        valid.groupby(["timing_policy", "attack_day"], as_index=False).agg(
            attack_success_rate=("success", "mean"),
            nontrivial_attack_success_rate=("nontrivial_success", "mean"),
            baseline_target_rate=("already_target", "mean"),
            mean_min_abs_delta_norm=("min_abs_delta_norm", "mean"),
            mean_abs_prediction_shift=("prediction_shift", lambda x: np.nanmean(np.abs(x))),
            trials=("success", "size"),
        )
    )

    policy_summary = (
        valid.groupby("timing_policy", as_index=False).agg(
            trials=("success", "size"),
            attack_success_rate=("success", "mean"),
            baseline_target_rate=("already_target", "mean"),
            nontrivial_trials=("already_target", lambda x: int((~x).sum())),
            nontrivial_successes=("nontrivial_success", "sum"),
            mean_min_abs_delta_norm=("min_abs_delta_norm", "mean"),
        )
    )
    policy_summary["nontrivial_attack_success_rate"] = np.where(
        policy_summary["nontrivial_trials"] > 0,
        policy_summary["nontrivial_successes"] / policy_summary["nontrivial_trials"],
        np.nan,
    )
    return (
        timing_summary,
        policy_summary.sort_values("nontrivial_attack_success_rate", ascending=False),
    )
