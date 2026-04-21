# ARIMA Model for Stock Price Prediction

## Overview

This project uses ARIMA (AutoRegressive Integrated Moving Average) models to predict stock prices for various tickers. We leverage pmdarima to find the best model parameters and statsmodels to perform forecasting.

## Folder Structure

- **arima_models**: This folder contains trained ARIMA models saved in `.pkl` format for each ticker.
- **arima_predictions**: This folder stores prediction results in `.csv` format for each ticker, including actual vs. predicted values.

## How to Run

### Install Dependencies:

Ensure you have all necessary Python packages installed by running:


bash
pip install pmdarima statsmodels yahoo_fin joblib scikit-learn matplotlib pandas numpy

 Train and Predict:

Modify the list of tickers, `start_date`, and `end_date` in the `train_and_predict_arima.py` file.

Run the script to train the ARIMA models and generate predictions:


python train_and_predict_arima.py

Output:

- Trained models will be saved in the **arima_models** folder.
- Predictions (actual vs. predicted) will be saved as `.csv` files in the **arima_predictions** folder.

File Descriptions

- **train_and_predict_arima.py**: The script that handles fetching stock data, training ARIMA models, and generating predictions.
- **README.md**: This file explaining the project setup.
