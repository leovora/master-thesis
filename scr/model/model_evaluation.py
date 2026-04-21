import os
import numpy as np
from keras.models import load_model
from data.load_data import load_data_from_csv, preprocess_data


def get_predictions(tickers, start_date, end_date, sequence_length, folder_path="../models"):
    predictions = {}

    actuals = {}
    
    for ticker in tickers:
        print(f"Processing {ticker}...")

        model_path = os.path.join(folder_path, f"{ticker}_model.h5")
        if not os.path.exists(model_path):
            print(f"No model found for {ticker}. Skipping.")
            continue

        model = load_model(model_path)

        X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

        y_pred = model.predict(X_test)
        y_pred_rescaled = y_pred * (y_max - y_min) + y_min
        y_actual = y_test * (y_max - y_min) + y_min

        predictions[ticker] = y_pred_rescaled.flatten()
        actuals[ticker] = y_actual.flatten()

    return predictions, actuals




# def get_predictions(tickers, start_date, end_date, sequence_length, folder_path="../models"):
#     predictions = {}
#     actuals = {}
    
#     for ticker in tickers:
#         print(f"Processing {ticker}...")

#         model_path = os.path.join(folder_path, f"{ticker}_model.h5")
#         if not os.path.exists(model_path):
#             print(f"No model found for {ticker}. Skipping.")
#             continue

#         model = load_model(model_path)

#         df = load_data_from_csv(ticker)
#         if df.empty:
#             print(f"No data found for {ticker}. Skipping.")
#             continue

#         X, y, y_min, y_max = preprocess_data(df, sequence_length)

#         y_pred = model.predict(X)
#         y_pred_rescaled = y_pred * (y_max - y_min) + y_min
#         y_actual = y * (y_max - y_min) + y_min

#         predictions[ticker] = y_pred_rescaled.flatten()
#         actuals[ticker] = y_actual.flatten()

#     return predictions, actuals
