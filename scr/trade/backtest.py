import numpy as np
import pandas as pd


def simulate_trades_with_allocation(predictions, actuals, signals, initial_capital=100000, allocation_percentage=0.1, slippage_rate=0.02, transaction_cost=0.005):
    portfolio_value = initial_capital
    portfolio_returns = []
    positions = {ticker: 0 for ticker in predictions.keys()}

    for i in range(len(next(iter(signals.values())))):
        for ticker, signal in signals.items():
            if i >= len(signal):
                continue

            if signal[i] == 1:  # Buy signal
                allocated_capital = portfolio_value * allocation_percentage
                num_shares_to_buy = int(allocated_capital / actuals[ticker][i])
                cost_of_buy = num_shares_to_buy * actuals[ticker][i] * (1 + slippage_rate) + transaction_cost
                if portfolio_value > cost_of_buy:
                    positions[ticker] += num_shares_to_buy
                    portfolio_value -= cost_of_buy

            elif signal[i] == -1 and positions[ticker] > 0:  # Sell signal
                num_shares_to_sell = positions[ticker]
                positions[ticker] = 0
                portfolio_value += num_shares_to_sell * actuals[ticker][i] * (1 - slippage_rate) - transaction_cost

        daily_return = sum(positions[ticker] * (actuals[ticker][i + 1] - actuals[ticker][i]) for ticker in positions if i + 1 < len(actuals[ticker]))
        portfolio_returns.append(daily_return / portfolio_value)
        portfolio_value += daily_return

    return np.array(portfolio_returns)
