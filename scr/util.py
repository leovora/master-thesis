import numpy as np

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))

def daily_sharpe_ratio(returns, risk_free_rate_annual=0.0505, trading_days=252):
    """Calculate daily Sharpe Ratio."""
    risk_free_rate_daily = risk_free_rate_annual / trading_days
    excess_returns = returns - risk_free_rate_daily
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days)
