"""Shared evaluation metrics for comparing prediction series."""
import numpy as np
from scipy import stats


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def direction_hits(prev_actual, y_true, y_pred):
    """
    Boolean array: did the prediction get the direction of the move right
    (relative to the previous actual close)? A predicted "no change" counts
    as a miss, since you can't act on it either way.
    """
    prev_actual = np.asarray(prev_actual, dtype=float)
    true_dir = np.sign(np.asarray(y_true, dtype=float) - prev_actual)
    pred_dir = np.sign(np.asarray(y_pred, dtype=float) - prev_actual)
    return (true_dir == pred_dir) & (true_dir != 0)


def directional_accuracy(prev_actual, y_true, y_pred):
    return float(np.mean(direction_hits(prev_actual, y_true, y_pred)))


def direction_p_value(prev_actual, y_true, y_pred, null_rate=0.5):
    """
    One-sided binomial test: if the model had no skill beyond `null_rate`,
    how likely is a hit rate at least this high? A small p-value (< 0.05)
    is evidence of real skill; anything else is indistinguishable from luck.

    The fair null is NOT always 50%. In a rising market most days are up
    days, so a model that just says "up" every day beats 50% with zero
    skill. main.py passes the always-up hit rate as the null for that reason.
    """
    hits = direction_hits(prev_actual, y_true, y_pred)
    return float(stats.binomtest(int(hits.sum()), len(hits), null_rate, alternative="greater").pvalue)


def information_coefficient(pred_returns, actual_returns):
    """
    Rank correlation between predicted and actual returns, the standard
    quant measure of signal quality. 0 means no signal; even 0.02-0.05 is
    considered meaningful in practice if it holds up out of sample.
    """
    pred_returns = np.asarray(pred_returns, dtype=float)
    if np.std(pred_returns) == 0:
        return 0.0
    return float(stats.spearmanr(pred_returns, actual_returns)[0])


def lag_similarity(y_pred, naive_pred):
    """
    Correlation between a model's price predictions and the naive
    'tomorrow = today' baseline. Values near 1.0 mean the model is mostly
    reproducing yesterday's price (the failure mode found in version 1).
    """
    return float(np.corrcoef(np.asarray(y_pred, dtype=float), np.asarray(naive_pred, dtype=float))[0, 1])
