"""
A deliberately simple backtest: no real trading, no broker, no fees --
just a hypothetical starting balance simulated in code from historical
prices, so the "returns" are a paper simulation you can compute and plot,
never an instruction to actually place trades.

Rule: each day, if the model predicted the close would be higher than the
previous actual close, hold the stock that day (capture its return);
otherwise hold cash (0% that day). Ignores transaction costs and slippage
on purpose to keep the comparison legible -- a real strategy would need to
account for both.
"""
import numpy as np
import pandas as pd


def simple_signal_backtest(dates, prev_actual, actual, predicted, starting_cash=10_000.0):
    """
    dates: array-like of the test-period dates (for plotting)
    prev_actual: previous day's actual close (the decision point)
    actual: actual close on the predicted day
    predicted: model's predicted close for that day
    Returns a DataFrame with daily equity for the strategy and for
    buy-and-hold over the same period.
    """
    prev_actual = np.asarray(prev_actual, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    daily_return = (actual - prev_actual) / prev_actual
    signal = predicted > prev_actual  # True = go long that day

    strategy_daily_return = np.where(signal, daily_return, 0.0)

    strategy_equity = starting_cash * np.cumprod(1 + strategy_daily_return)
    buyhold_equity = starting_cash * np.cumprod(1 + daily_return)

    return pd.DataFrame({
        "Date": dates,
        "strategy_equity": strategy_equity,
        "buyhold_equity": buyhold_equity,
        "signal_long": signal,
    })
