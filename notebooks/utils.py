import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import keras
from typing import Tuple
from keras.models import load_model
from trade.backtest import simulate_trades_with_allocation
from trade.trading_strategy import moving_average_strategy


FEATURES = ['high', 'low', 'open', 'close', 'volume']
tickers = ['GOOGL', 'AMZN', 'AAPL', 'PEP', 'JNJ', 'PFE', 'MRK', 'ABBV', 'PG', 'KO',
           'WMT', 'JPM', 'BAC', 'GS', 'V', 'XOM', 'CVX', 'COP', 'BP', 'BA',
           'MMM', 'HON', 'GE', 'T', 'VZ', 'TMUS', 'HSY', 'DUK', 'SO', 'EXC', 'AEP',
           'AMT', 'PLD', 'SPG', 'BHP', 'RIO', 'VALE', 'FCX']
start_date = '2010-01-01'
end_date = '2023-10-23'
sequence_length = 50
window_sizes = [50, 40, 30]

if not os.path.exists("models"):
    os.makedirs("models")

def partition_dataset(sequence_length: int, data: np.ndarray) -> np.ndarray:
    sequences = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
    return np.array(sequences)

def preprocess_data(ticker, start_date, end_date, sequence_length):
    df = si.get_data(ticker, start_date=start_date, end_date=end_date)

    data_filtered = df[FEATURES]
    X = data_filtered.values
    y = data_filtered['close'].values
    
    X = X[:-1]
    y = y[1:]
    
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X_norm = (X - x_min) / (x_max-x_min)
    
    y_max = np.max(y)
    y_min = np.min(y)
    y_norm = (y - y_min) / (y_max-y_min)
    
    X_seq = partition_dataset(sequence_length, X_norm)
    y_seq = y_norm[sequence_length:]

    split = int(X_seq.shape[0] * 0.8)
    X_train = X_seq[:split]
    y_train = y_seq[:split]
    X_test = X_seq[split:]
    y_test = y_seq[split:]

    return X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max

def get_predictions(tickers, start_date, end_date, sequence_length, folder_path="models"):
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



def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))

def daily_sharpe_ratio(returns, risk_free_rate_annual=0.0505, trading_days=252):
    """Calculate daily Sharpe Ratio."""
    risk_free_rate_daily = risk_free_rate_annual / trading_days
    excess_returns = returns - risk_free_rate_daily
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days)

def plot_dynamic_sharpe_ratio(returns, risk_free_rate=0.0505, trading_days=252):
    rolling_sharpe = []
    for i in range(1, len(returns) + 1):
        temp_returns = returns[:i]
        temp_sharpe = daily_sharpe_ratio(temp_returns, risk_free_rate, trading_days)
        rolling_sharpe.append(temp_sharpe)

    plt.figure(figsize=(14, 8))
    plt.plot(rolling_sharpe, label='Sharpe Ratio (baseline)', color='blue')  
    plt.ylabel('Sharpe Ratio', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    plt.xlim(0, 700)  
    plt.grid(True) 
    plt.tight_layout()  
    
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.legend(fontsize=50)  
    filename = "sharpe_ratio_baseline.pdf"
    plt.savefig('plots/baseline/sharpe_ratio_baseline.pdf', dpi=300) 
    plt.show()

    
def plot_daily_returns(portfolio_returns):
   
    returns_series = pd.Series(portfolio_returns)

    plt.figure(figsize=(14, 8))  
    plt.plot(returns_series, label='Daily Returns', color='blue')
    plt.xlabel('Days', fontsize=40) 
    plt.ylabel('Returns', fontsize=40)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20) 
    plt.legend(fontsize=50)  
    plt.xlim(0, 700) 
    plt.grid(True)  
    plt.tight_layout()  
    filename = "daily_returns.pdf"
    plt.savefig('plots/baseline/daily_returns.pdf', dpi=300)  
    plt.show()

    
cap_threshold = 0.02

def calculate_cumulative_returns(returns):
    # Cap the returns and compute the cumulative product
    capped_returns = np.clip(returns, -cap_threshold, cap_threshold)
    adjusted_returns = capped_returns + 1
    cumulative_returns = np.cumprod(adjusted_returns) - 1
    return cumulative_returns

def plot_cumulative_returns_baseline(portfolio_returns):
    plt.figure(figsize=(14, 8))
    
    baseline_cumulative_returns = calculate_cumulative_returns(portfolio_returns)
    plt.plot(baseline_cumulative_returns, label='Cumulative Returns', color='blue', linewidth=2)
    
    plt.ylabel('Cumulative Returns (%)', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(0, 700)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    filename = "cumulative_returns_baseline.pdf"
    plt.savefig('plots/baseline/cumulative_returns_baseline.pdf', dpi=300)
    plt.show()   
    

import matplotlib.pyplot as plt
import numpy as np

def plot_average_predictions(predictions: dict, actuals: dict):
    """
    Plots the average predictions and actual values for all stocks and saves the plot.

    Parameters:
    - predictions: A dictionary containing predicted values for each stock.
    - actuals: A dictionary containing actual values for each stock.
    """
    # Ensure all tickers have the same length
    first_ticker = next(iter(predictions))
    valid_length = len(predictions[first_ticker])

    valid_tickers = [ticker for ticker in predictions 
                     if ticker in actuals and len(predictions[ticker]) == valid_length and len(actuals[ticker]) == valid_length]

    if not valid_tickers:
        raise ValueError("No valid tickers with matching lengths found.")
        
    min_length = valid_length

    avg_predictions = np.zeros(min_length)
    avg_actuals = np.zeros(min_length)

    n_stocks = len(valid_tickers)

    # Compute sums
    for ticker in valid_tickers:
        avg_predictions += predictions[ticker][:min_length]
        avg_actuals += actuals[ticker][:min_length]

    # Compute averages
    avg_predictions /= n_stocks
    avg_actuals /= n_stocks

    plt.figure(figsize=(14, 8))

    plt.plot(avg_predictions, color='blue', label='Predicted')
    plt.plot(avg_actuals, color='red', label='Actual')

    plt.xlabel("Days", fontsize=40)
    plt.ylabel("Average Price", fontsize=40)
    plt.legend(fontsize=50)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig('plots/baseline/average_predictions.pdf', dpi=300)
    plt.show()

predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length, folder_path='models')

    
def perform_ephemeral_attack(ticker, start_date, end_date, sequence_length, days_to_attack, window_size=30, folder_path="models"):
    model_path = os.path.join(folder_path, f"{ticker}_model.h5")
    model = load_model(model_path)

    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    np.random.seed(0)
    attack_indices = np.random.choice(X_test.shape[0], size=days_to_attack, replace=False)

    attacked_X_test = X_test.copy()
    for i in attack_indices:
        window = max(0, sequence_length-window_size)
        stdev = np.std(attacked_X_test[i, window:, 3])
        attacked_X_test[i, -1, 3] += 2 * stdev

    y_pred = model.predict(attacked_X_test)
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min

    predictions = {ticker: y_pred_rescaled.flatten()}
    actuals = {ticker: y_actual.flatten()}

    return predictions, actuals, attack_indices

def ephemeral_attacks(tickers, start_date, end_date, sequence_length, attack_days, window_size=30):
    # Get predictions and actuals
    predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length)

    # Generate trading signals
    predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length, folder_path='models')

# Trading strategy
    trading_signals = moving_average_strategy(predictions)

# Backtest the strategy
    portfolio_returns = simulate_trades_with_allocation(predictions, actuals, trading_signals)


    returns_after_attacks = [portfolio_returns]
    
    # Apply the attack to Google's model and get the attacked predictions
    for day in attack_days:
        attacked_predictions, _, _ = perform_ephemeral_attack('GOOGL', start_date, end_date, sequence_length, day, window_size=window_size)
        predictions['GOOGL'] = attacked_predictions['GOOGL']  # Update Google's predictions with the attacked ones
        
        # Generate trading signals
        trading_signals = moving_average_strategy(predictions)
        
        # Calculate the portfolio returns after the attack
        portfolio_returns_after_attack = simulate_trades_with_allocation(predictions, actuals, trading_signals)
        
        returns_after_attacks.append(portfolio_returns_after_attack)

    return returns_after_attacks


def perform_ephemeral_attack_with_window(model, ticker, start_date, end_date, sequence_length, attack_index, window_size):
    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    attacked_X_test = X_test.copy()
    window_start = max(0, sequence_length - window_size)
    stdev = np.std(attacked_X_test[attack_index, window_start:, 3])
    attacked_X_test[attack_index, -1, 3] += 2 * stdev

    y_pred_attack = model.predict(attacked_X_test)
    y_pred_rescaled_attack = y_pred_attack * (y_max - y_min) + y_min

    return y_pred_rescaled_attack

def plot_and_save_attack_model(ticker, start_date, end_date, sequence_length, attack_day, model_path, plot_dir):
    if not os.path.exists(model_path):
        print(f"No model found for {ticker}. Skipping.")
        return

    model = load_model(model_path)
    X_train, y_train, X_test, y_test, y_min, y_max, x_min, x_max = preprocess_data(ticker, start_date, end_date, sequence_length)

    y_pred = model.predict(X_test)
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min

    windows = [50, 40, 30]
    attacked_predictions = {}
    for w in windows:
        attacked_predictions[w] = perform_ephemeral_attack_with_window(model, ticker, start_date, end_date, sequence_length, attack_day, w)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(y_actual, color='green', label='Real Value', linewidth=2)
    ax.plot(y_pred_rescaled, color='blue', label='Predicted Value', linewidth=2)
    colors = ['red', 'purple', 'orange']

    for idx, w in enumerate(windows):
        ax.plot(range(attack_day - 1, attack_day + 2), attacked_predictions[w][attack_day - 1:attack_day + 2], color=colors[idx], linestyle='--', label=f'Attack $\omega$={w}', linewidth=1.5)

    # Draw a vertical line representing the attack day
    ax.axvspan(attack_day - 1, attack_day + 1, color='black', linestyle=':', alpha=0.5)

    # Create a zoomed inset
    axins = ax.inset_axes([0.5, 0.2, 0.3, 0.3])
    axins.plot(y_actual, color='green')
    axins.plot(y_pred_rescaled, color='blue')
    for idx, w in enumerate(windows):
        axins.plot(range(attack_day - 1, attack_day + 2), attacked_predictions[w][attack_day - 1:attack_day + 2], color=colors[idx], linestyle='--')

    x1, x2, y1, y2 = attack_day - 10, attack_day + 10, min(y_actual[attack_day - 3:attack_day + 4].min(), y_pred_rescaled[attack_day - 3:attack_day + 4].min(), attacked_predictions[30][attack_day], attacked_predictions[40][attack_day], attacked_predictions[50][attack_day]) - 5, max(y_actual[attack_day - 3:attack_day + 4].max(), y_pred_rescaled[attack_day - 3:attack_day + 4].max(), attacked_predictions[30][attack_day], attacked_predictions[40][attack_day], attacked_predictions[50][attack_day]) + 5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlabel('Days', fontsize=40)
    ax.set_ylabel(f'{ticker} Closing Price', fontsize=40)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper left', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    plot_dir_path = os.path.join(plot_dir, "attack_plots")
    os.makedirs(plot_dir_path, exist_ok=True)
    
    plot_path = os.path.join(plot_dir_path, f"{ticker.lower()}_stock_prediction_attack_day_{attack_day}.pdf")
    plt.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    


def plot_cumulative_returns_after_attack(before_returns, after_returns_list, window_sizes, attack_day):
    plt.figure(figsize=(14, 8))
    
    # Loop through each window size and calculate/plot cumulative returns after the attack
    colors = ['red', 'green', 'orange', 'purple', 'cyan']  # Extend this list if more windows are needed
    for idx, returns_after in enumerate(after_returns_list):
        after_cumulative_returns = calculate_cumulative_returns(returns_after)
        plt.plot(after_cumulative_returns, label=f'After Attack $\omega$={window_sizes[idx]}', color=colors[idx], linewidth=1.5)
    
    # Plot the baseline cumulative returns from the beginning last
    baseline_cumulative_returns = calculate_cumulative_returns(before_returns)
    plt.plot(baseline_cumulative_returns, label='Baseline', color='blue', linewidth=2)
    
    # Add a vertical line at the attack day
    plt.axvline(x=attack_day, color='gray', linestyle='--', alpha=0.7, label=f'Attack Day {attack_day}')
    
    plt.ylabel('Cumulative Returns', fontsize=40)
    plt.xlabel('Days', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(0, 700)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    plot_dir = "plots/attack_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = os.path.join(plot_dir, f"cumulative_returns_after_attack_day_{attack_day}.pdf")
    plt.savefig(filename, dpi=300)
    plt.close()  

def plot_cumulative_returns_after_attack_day(tickers, start_date, end_date, sequence_length, window_sizes, attack_day):
    # Get pre-attack returns
    pre_attack_results = ephemeral_attacks(tickers, start_date, end_date, sequence_length, [])
    pre_attack_returns = pre_attack_results[0]

    after_returns_for_all_windows = []

    for w_size in window_sizes:
        # Perform the attack and get the returns
        attack_results = ephemeral_attacks(tickers, start_date, end_date, sequence_length, [attack_day], window_size=w_size)
        returns_after_attack = attack_results[1]

        # Ensure returns are the same up to the attack day
        combined_returns = np.concatenate((pre_attack_returns[:attack_day], returns_after_attack[attack_day:]))
        
        after_returns_for_all_windows.append(combined_returns)

    # Plot the cumulative returns for all windows
    plot_cumulative_returns_after_attack(pre_attack_returns, after_returns_for_all_windows, window_sizes, attack_day)



