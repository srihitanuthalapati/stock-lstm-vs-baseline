"""
End-to-end run: fetch data -> naive & moving-average baselines -> LSTM ->
compare metrics -> backtest -> save plots + a results/metrics.json with the
real numbers.

Usage: python3 main.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fetch_data import fetch_prices
from baseline import naive_predictions, moving_average_predictions
from model import train_lstm_and_predict
from metrics import rmse, mae, mape, directional_accuracy, lag_similarity
from backtest import simple_signal_backtest

TICKER = "AAPL"
START = "2015-01-01"
END = "2026-09-01"
LOOKBACK = 60
TEST_FRACTION = 0.2

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = fetch_prices(TICKER, START, END)
    close = df["Close"].astype(float)
    dates = df.index
    n = len(close)
    n_train = int(n * (1 - TEST_FRACTION))
    print(f"Loaded {n} rows for {TICKER}. Train rows: {n_train}, Test rows: {n - n_train}")
    print(f"Test period: {dates[n_train]} to {dates[-1]}")

    # ---- Baselines (computed over the whole series, we'll slice test rows) ----
    naive_pred_full = naive_predictions(close)
    ma_pred_full = moving_average_predictions(close, window=5)

    # ---- LSTM ----
    close_arr = close.values
    lstm_pred_prices, test_target_idx, model, scaler = train_lstm_and_predict(
        close_arr, n_train=n_train, lookback=LOOKBACK, epochs=40
    )

    # Align everything to the LSTM's test target indices (these all fall
    # within the chronological test period since target_idx >= n_train there,
    # and >= lookback which is well before n_train).
    test_dates = dates[test_target_idx]
    actual_test = close_arr[test_target_idx]
    prev_actual_test = close_arr[test_target_idx - 1]

    naive_pred_test = naive_pred_full.values[test_target_idx]
    ma_pred_test = ma_pred_full.values[test_target_idx]

    # ---- Metrics ----
    results = {}
    for name, pred in [
        ("naive_lag1", naive_pred_test),
        ("moving_average_5d", ma_pred_test),
        ("lstm", lstm_pred_prices),
    ]:
        results[name] = {
            "rmse": rmse(actual_test, pred),
            "mae": mae(actual_test, pred),
            "mape_pct": mape(actual_test, pred),
            "directional_accuracy_pct": directional_accuracy(prev_actual_test, actual_test, pred) * 100,
        }

    results["lstm"]["lag_similarity_to_naive"] = lag_similarity(lstm_pred_prices, naive_pred_test)

    print("\n=== Test-set metrics (lower RMSE/MAE/MAPE is better, higher directional accuracy is better) ===")
    for name, m in results.items():
        print(f"{name:20s} RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  MAPE={m['mape_pct']:.2f}%  "
              f"DirAcc={m['directional_accuracy_pct']:.1f}%")
    print(f"\nLSTM predictions' correlation with the naive 'yesterday' baseline: "
          f"{results['lstm']['lag_similarity_to_naive']:.4f} (1.0 = indistinguishable from lag)")

    # ---- Backtest: LSTM signal vs buy-and-hold ----
    bt = simple_signal_backtest(test_dates, prev_actual_test, actual_test, lstm_pred_prices)
    final_strategy = bt["strategy_equity"].iloc[-1]
    final_buyhold = bt["buyhold_equity"].iloc[-1]
    starting_cash = 10_000.0
    results["backtest"] = {
        "starting_cash": starting_cash,
        "lstm_signal_final_equity": float(final_strategy),
        "buy_and_hold_final_equity": float(final_buyhold),
        "lstm_signal_return_pct": float((final_strategy / starting_cash - 1) * 100),
        "buy_and_hold_return_pct": float((final_buyhold / starting_cash - 1) * 100),
        "test_period_start": str(test_dates[0].date()),
        "test_period_end": str(test_dates[-1].date()),
    }

    print(f"\n=== Backtest (${starting_cash:,.0f} start, {test_dates[0].date()} to {test_dates[-1].date()}) ===")
    print(f"LSTM-signal strategy final equity: ${final_strategy:,.2f} "
          f"({results['backtest']['lstm_signal_return_pct']:+.2f}%)")
    print(f"Buy-and-hold final equity:         ${final_buyhold:,.2f} "
          f"({results['backtest']['buy_and_hold_return_pct']:+.2f}%)")

    also_naive_bt = simple_signal_backtest(test_dates, prev_actual_test, actual_test, naive_pred_test)
    results["backtest"]["naive_signal_final_equity"] = float(also_naive_bt["strategy_equity"].iloc[-1])
    results["backtest"]["naive_signal_return_pct"] = float(
        (also_naive_bt["strategy_equity"].iloc[-1] / starting_cash - 1) * 100
    )

    # ---- Save metrics.json ----
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.join(RESULTS_DIR, 'metrics.json')}")

    # ---- Plots ----
    plt.figure(figsize=(11, 5))
    plt.plot(test_dates, actual_test, label="Actual close", linewidth=1.5)
    plt.plot(test_dates, naive_pred_test, label="Naive (yesterday's price)", linestyle="--", alpha=0.8)
    plt.plot(test_dates, lstm_pred_prices, label="LSTM prediction", alpha=0.9)
    plt.title(f"{TICKER}: actual vs. predicted close, test period")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predictions_vs_actual.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(bt["Date"], bt["buyhold_equity"], label="Buy & hold")
    plt.plot(bt["Date"], bt["strategy_equity"], label="LSTM-signal strategy")
    plt.title(f"{TICKER}: hypothetical ${starting_cash:,.0f} backtest (no real trading)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "backtest_equity_curve.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {PLOTS_DIR}")
    return results


if __name__ == "__main__":
    main()
