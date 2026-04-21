from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si


def train_and_predict_arima(tickers, start_date, end_date, models_folder='arima_models', predictions_folder='arima_predictions'):
    """
    Train and predict stock prices using ARIMA models for each ticker in the tickers list.
    Models are saved in 'models_folder' and predictions are saved in 'predictions_folder'.

    Args:
        tickers (list): List of stock tickers to train models on.
        start_date (str): The start date for fetching historical stock data (format: 'YYYY-MM-DD').
        end_date (str): The end date for fetching historical stock data (format: 'YYYY-MM-DD').
        models_folder (str): Folder to save trained ARIMA models.
        predictions_folder (str): Folder to save ARIMA model predictions.
    """
    
    # Create folders if they don't exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    arima_results = {}  # Dictionary to store RMSE and predictions for each ticker

    # Define a separate date for testing predictions
    test_start_date = '2021-02-02'  # Replace with your own test start date if needed

    for ticker in tickers:
        try:
            print(f"Processing {ticker} data...")
            
            model_path = os.path.join(models_folder, f"{ticker}_arima.pkl")
            predictions_path = os.path.join(predictions_folder, f"{ticker}_predictions.csv")
            
            if os.path.exists(predictions_path):
                print(f"Predictions for {ticker} already exist. Skipping...")
                continue

            # Fetch historical stock data from Yahoo Finance
            df = si.get_data(ticker, start_date=start_date, end_date=end_date)
            df.dropna(inplace=True)
            close_prices = df['close']

            # Find the index of the test start date
            test_start_idx = df.index.get_loc(test_start_date)
            train, test = close_prices[:test_start_idx], close_prices[test_start_idx:]

            history = [x for x in train]  # Historical data for walk-forward validation
            predictions = []  # List to store predictions

            # Load existing model if available, otherwise train a new model
            if os.path.exists(model_path):
                print(f"Loading existing model for {ticker}...")
                auto_model = joblib.load(model_path)
                order = auto_model.order
            else:
                # Train a new ARIMA model using auto_arima
                try:
                    print(f"Finding best ARIMA model parameters for {ticker}...")
                    auto_model = auto_arima(train, seasonal=False, trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)
                    print(auto_model.summary())
                    joblib.dump(auto_model, model_path)
                    order = auto_model.order
                except Exception as e:
                    print(f"Failed to find best ARIMA model parameters for {ticker}: {e}")
                    print("Falling back to predefined ARIMA model.")
                    order = (5,1,0)  # Default ARIMA order

            # Walk-forward validation to make predictions
            for t in range(len(test)):
                model_arima = ARIMA(history, order=order)
                model_fit = model_arima.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            # Calculate RMSE (Root Mean Squared Error)
            rmse = sqrt(mean_squared_error(test, predictions))

            # Store the results for this ticker
            arima_results[ticker] = {
                'train': train,
                'test': test,
                'predictions': predictions,
                'rmse': rmse
            }

            # Create DataFrame to save predictions
            predictions_df = pd.DataFrame({'Actual': test, 'Predicted': predictions}, index=test.index)
            print(f"DataFrame to be saved for {ticker}:")
            print(predictions_df.head())

            # Save the predictions DataFrame as CSV
            print(f"Saving predictions to {predictions_path}")
            predictions_df.to_csv(predictions_path)
            print(f"Predictions for {ticker} saved at {predictions_path}")
            print(f'Test RMSE for {ticker}: {rmse:.3f}')
        
        except Exception as e:
            print(f"An error occurred for {ticker}: {e}")

    return arima_results


if __name__ == '__main__':
    tickers = [
        'GOOGL', 'AMZN', 'AAPL', 'PEP', 'JNJ', 'PFE', 'MRK', 'ABBV', 'PG', 'KO',
        'WMT', 'JPM', 'BAC', 'GS', 'V', 'XOM', 'CVX', 'COP', 'BP', 'BA',
        'MMM', 'HON', 'GE', 'T', 'VZ', 'TMUS', 'HSY', 'DUK', 'SO', 'EXC', 'AEP',
        'AMT', 'PLD', 'SPG', 'BHP', 'RIO', 'VALE', 'FCX'
    ]
    start_date = '2010-01-01'
    end_date = '2023-10-23'

    train_and_predict_arima(tickers, start_date, end_date)