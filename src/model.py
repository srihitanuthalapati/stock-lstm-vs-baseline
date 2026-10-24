"""LSTM model for next-day closing price prediction."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def make_sequences(values: np.ndarray, lookback: int):
    """
    Turn a 1D array of scaled prices into (X, y) windows:
    X[i] = values[i : i+lookback], y[i] = values[i+lookback]
    Returns X of shape (n, lookback, 1) and y of shape (n,), plus the
    index (into `values`) of each target so callers can map back to dates.
    """
    X, y, target_idx = [], [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
        target_idx.append(i)
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y, np.array(target_idx)


def build_lstm(lookback: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm_and_predict(close_prices: np.ndarray, n_train: int, lookback: int = 60,
                            epochs: int = 40, batch_size: int = 32, seed: int = 42):
    """
    close_prices: full chronological array of closing prices (train+test).
    n_train: number of rows (from the start) that belong to the train split.
    Returns: predicted prices (aligned to test targets), the target indices
    (into close_prices) they correspond to, and the fitted model/scaler.
    """
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_prices = close_prices[:n_train].reshape(-1, 1)
    scaler.fit(train_prices)

    scaled_all = scaler.transform(close_prices.reshape(-1, 1)).flatten()

    X_all, y_all, target_idx = make_sequences(scaled_all, lookback)

    # a target belongs to train if its index is < n_train, else test
    train_mask = target_idx < n_train
    test_mask = ~train_mask

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test = X_all[test_mask]
    test_target_idx = target_idx[test_mask]

    model = build_lstm(lookback)
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=[es], shuffle=True)

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    return pred_prices, test_target_idx, model, scaler
