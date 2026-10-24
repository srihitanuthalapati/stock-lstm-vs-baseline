"""
LSTM that predicts next-day log returns rather than raw prices.

Why returns: a price-level model can score a low error just by echoing
yesterday's price (version 1 of this project showed exactly that). Returns
remove the trend and the level, so the model has to predict the actual
move, which is the part that matters for a trading decision.
"""
from __future__ import annotations

import numpy as np


def make_sequences(values: np.ndarray, lookback: int):
    """
    Turn a 1D array into (X, y) windows:
    X[i] = values[i - lookback : i], y[i] = values[i]
    Also returns the index (into `values`) of each target.
    """
    X, y, target_idx = [], [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
        target_idx.append(i)
    return np.array(X).reshape(-1, lookback, 1), np.array(y), np.array(target_idx)


def build_lstm(lookback: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm_and_predict(log_returns: np.ndarray, train_end: int, test_end: int,
                           lookback: int = 60, epochs: int = 40, batch_size: int = 32,
                           seed: int = 42):
    """
    Train on targets before `train_end`, predict targets in [train_end, test_end).

    log_returns: chronological array of daily log returns.
    Standardization uses training-period mean and std only, so nothing about
    the test period leaks into training. Test inputs are past realized
    returns, the same information a trader would have that morning.

    Returns the predicted log returns and the indices they correspond to.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(seed)

    train_vals = log_returns[:train_end]
    mu, sd = train_vals.mean(), train_vals.std()
    z = (log_returns[:test_end] - mu) / sd

    X, y, idx = make_sequences(z, lookback)
    train_mask = idx < train_end
    test_mask = idx >= train_end

    model = build_lstm(lookback)
    # validation_split takes the LAST 10% of the training windows (before
    # shuffling), so early stopping is judged on the most recent unseen data.
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X[train_mask], y[train_mask], epochs=epochs, batch_size=batch_size,
              validation_split=0.1, shuffle=True, verbose=0, callbacks=[es])

    pred_z = model.predict(X[test_mask], verbose=0).flatten()
    return pred_z * sd + mu, idx[test_mask]
