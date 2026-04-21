import os
import numpy as np
from keras.models import load_model
from utils import preprocess_data, get_predictions, generate_trading_signals
from trade.backtest import simulate_trades_with_allocation
from perform_ephemeral_attack import *

def ephemeral_attacks(tickers, start_date, end_date, sequence_length, attack_days, window_size=30):
    # Get predictions and actuals
    predictions, actuals = get_predictions(tickers, start_date, end_date, sequence_length)

    # Generate trading signals
    trading_signals = generate_trading_signals(predictions)

    # Get portfolio returns without any attack
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