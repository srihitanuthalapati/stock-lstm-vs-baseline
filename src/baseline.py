"""Naive baselines: 'tomorrow = today' and a simple moving average."""
import numpy as np
import pandas as pd


def naive_predictions(close: pd.Series) -> pd.Series:
    """Predict each day's close as the previous day's close."""
    return close.shift(1)


def moving_average_predictions(close: pd.Series, window: int = 5) -> pd.Series:
    """Predict each day's close as the trailing N-day average (up to and
    including the prior day, so it can't see the day it's predicting)."""
    return close.shift(1).rolling(window=window).mean()
