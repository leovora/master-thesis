import numpy as np

def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window

def moving_average_strategy(predictions, short_window=5, long_window=20, threshold=0.01):
    trading_signals = {}
    for ticker, pred in predictions.items():
        short_mavg = moving_average(pred, short_window)
        long_mavg = moving_average(pred, long_window)
        mavg_diff = short_mavg[long_window - short_window:] - long_mavg

        signal = np.where(mavg_diff > threshold, 1, np.where(mavg_diff < -threshold, -1, 0))
        signal = np.pad(signal, (1, 0), 'constant')[:-1]  # Shift the signal by 1 to avoid lookahead bias

        trading_signals[ticker] = signal

    return trading_signals


def rolling_std_deviation(data, window):
    """Compute rolling standard deviation using NumPy's sliding window."""
    if len(data) < window:
        return np.array([])  # Return empty array if not enough data
    rolling_windows = np.lib.stride_tricks.sliding_window_view(data, window)
    return np.std(rolling_windows, axis=1)

def rolling_std_deviation_strategy(predictions, actuals, window=20, num_std_dev=2):
    trading_signals = {}

    for ticker, pred in predictions.items():
        if len(actuals[ticker]) < window:
            # If not enough data for rolling calculations, return zeros
            trading_signals[ticker] = np.zeros(len(pred))
            continue

        rolling_mean = moving_average(actuals[ticker], window)
        rolling_std = rolling_std_deviation(actuals[ticker], window)

        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)

        # Trim the predictions to match the size of rolling calculations
        trimmed_pred = pred[window - 1:]

        # Generate buy and sell signals
        signal = np.where(trimmed_pred > upper_band, -1, np.where(trimmed_pred < lower_band, 1, 0))
        
        # Shift the signal by 1 to avoid lookahead bias
        signal = np.pad(signal, (1, 0), 'constant')[:-1]

        trading_signals[ticker] = signal

    return trading_signals

def rate_of_change_strategy(predictions, window=14, threshold=0.01):
    trading_signals = {}

    for ticker, pred in predictions.items():
        roc = rate_of_change(pred, window)

        # Generate buy, sell, or hold signals based on RoC values and the threshold
        signal = np.where(roc > threshold, 1, np.where(roc < -threshold, -1, 0))
        
        # Shift the signal by 1 to avoid lookahead bias
        signal = np.pad(signal, (1, 0), 'constant')[:-1]

        trading_signals[ticker] = signal

    return trading_signals

def rate_of_change(prices, window):
    """
    Compute the rate of change (RoC) for a given price series.
    """
    # Type check
    if not isinstance(window, int):
        raise TypeError(f"Window must be an integer, but got {type(window).__name__}")

    # Length check
    if len(prices) < window:
        return np.array([])  # Return empty array if not enough data

    # Compute RoC
    return (prices[window:] - prices[:-window]) / prices[:-window]

