"""
A simple paper backtest: no broker, no real trades, just a hypothetical
starting balance simulated from historical prices.

Rule: each day, if the prediction says the close will be higher than the
previous actual close, hold the stock that day; otherwise hold cash.
A transaction cost is charged every time the position flips between stock
and cash, so a signal that trades constantly has to earn its costs.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_signal_backtest(dates, prev_actual, actual, predicted,
                           starting_cash=10_000.0, cost_bps=5.0):
    """
    Returns a DataFrame with daily equity for the strategy and for
    buy-and-hold over the same period, plus the daily position.
    cost_bps: cost per position change in basis points (5 = 0.05%).
    """
    prev_actual = np.asarray(prev_actual, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    cost = cost_bps / 10_000

    daily_return = actual / prev_actual - 1
    position = (predicted > prev_actual).astype(float)
    trades = np.abs(np.diff(position, prepend=0.0))

    strategy_return = position * daily_return - trades * cost
    buyhold_return = daily_return.copy()
    buyhold_return[0] -= cost

    return pd.DataFrame({
        "Date": dates,
        "strategy_return": strategy_return,
        "buyhold_return": buyhold_return,
        "strategy_equity": starting_cash * np.cumprod(1 + strategy_return),
        "buyhold_equity": starting_cash * np.cumprod(1 + buyhold_return),
        "position": position,
    })


def performance_stats(daily_returns):
    """Total return, annualized Sharpe (risk-free rate 0), and max drawdown."""
    r = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1 + r)
    sd = r.std()
    drawdown = equity / np.maximum.accumulate(equity) - 1
    return {
        "total_return_pct": float((equity[-1] - 1) * 100),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100),
    }
