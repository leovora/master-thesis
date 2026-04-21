import os
import numpy as np
from keras.models import load_model
from utils import preprocess_data, get_predictions, generate_trading_signals
from trade.backtest import simulate_trades_with_allocation


def perform_ephemeral_attack(ticker, start_date, end_date, sequence_length, days_to_attack, window_size=30, folder_path="models"):
    model_path = os.path.join(folder_path, f"{ticker}_model.h5")
    model = load_model(model_path)

     X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    attacked_X_test = X_test.copy()
    if attack_day < len(attacked_X_test):
        window = max(0, sequence_length - window_size)
        stdev = np.std(attacked_X_test[attack_day, window:, 3])
        attacked_X_test[attack_day, -1, 3] += 2 * stdev

    y_pred = model.predict(attacked_X_test)
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min

    predictions = {ticker: y_pred_rescaled.flatten()}
    actuals = {ticker: y_actual.flatten()}
    return predictions, actuals, attack_indices