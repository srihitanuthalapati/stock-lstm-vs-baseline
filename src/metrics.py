"""Shared evaluation metrics for comparing prediction series."""
import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def directional_accuracy(prev_actual, y_true, y_pred):
    """
    Fraction of days the prediction got the direction of movement right
    (relative to the previous actual close), which is what matters for a
    trading decision -- not just how close the predicted price level is.
    """
    prev_actual = np.asarray(prev_actual, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_dir = np.sign(y_true - prev_actual)
    pred_dir = np.sign(y_pred - prev_actual)
    # treat a predicted "no change" as a miss (can't act on it either way)
    correct = (true_dir == pred_dir) & (true_dir != 0)
    return float(np.mean(correct))


def lag_similarity(y_pred, naive_pred):
    """
    How closely a model's predictions track the naive 'tomorrow = today'
    baseline. Values close to 1.0 mean the model is mostly just reproducing
    yesterday's price rather than adding real predictive signal -- the
    classic LSTM-on-price-level failure mode.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)
    corr = np.corrcoef(y_pred, naive_pred)[0, 1]
    return float(corr)
