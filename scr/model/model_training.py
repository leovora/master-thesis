from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from data.load_data import preprocess_data, load_data_from_csv

def train_model(X, y, sequence_length):
    model = Sequential()
    n_neurons = X.shape[1] * X.shape[2]
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(sequence_length, X.shape[2])))
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    epochs = 50
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = model.fit(X, y,
                        batch_size=16,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stop])

    return model

if not os.path.exists("models"):
    os.makedirs("models")

def train_and_save_models(tickers, sequence_length):
    models = {}
    for ticker in tickers:
        model_path = os.path.join("models", f"{ticker}_model.h5")  # Path where the model will be saved

        # Check if the model already exists
        if os.path.exists(model_path):
            print(f"Model for {ticker} already exists. Skipping training.")
            continue  # Skip to the next ticker

        print(f"Processing {ticker}...")

        try:
            # Load data from CSV and preprocess
            df = load_data_from_csv(ticker)
            if df.empty:
                print(f"No data found for {ticker}. Skipping...")
                continue
            X, y, _, _ = preprocess_data(df, sequence_length)

            # Check if data is available for training
            if len(X) == 0:
                print(f"No data found for {ticker}. Skipping...")
                continue

            # Train the model
            model = train_model(X, y, sequence_length)
            models[ticker] = model

            # Save the model
            model.save(model_path)
            print(f"Model for {ticker} saved.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return models
