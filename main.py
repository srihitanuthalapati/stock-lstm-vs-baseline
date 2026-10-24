"""
PricePredictor v2: does an LSTM have real predictive skill on daily stock moves?

For each ticker, runs a walk-forward evaluation: train on everything before a
test year, predict that year one day at a time, then roll forward and repeat.
Every model is scored on the same out-of-sample days, with a significance
test on directional accuracy and a backtest that charges transaction costs.

Usage:
    python3 main.py                      # full run (all tickers, 4 folds)
    python3 main.py --tickers AAPL --folds 2 --epochs 10   # quick check
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import performance_stats, simple_signal_backtest
from src.baseline import moving_average_predictions, naive_predictions
from src.fetch_data import fetch_prices
from src.metrics import (direction_p_value, directional_accuracy, information_coefficient,
                         mae, mape, rmse)
from src.model import train_lstm_and_predict

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "SPY"]
START = "2012-01-01"
END = None  # None = up to today
FOLD_DAYS = 252  # one trading year per test fold
LOOKBACK = 60
COST_BPS = 5.0

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def walk_forward(ticker: str, n_folds: int, epochs: int) -> dict | None:
    """Run every model over n_folds consecutive out-of-sample years for one ticker."""
    try:
        close = fetch_prices(ticker, START, END)["Close"].dropna()
    except RuntimeError as e:
        print(f"[{ticker}] skipped: {e}")
        return None

    c = close.to_numpy(dtype=float)
    r = np.log(c[1:] / c[:-1])  # r[j] is the return that ends on close index j + 1
    first_test = len(r) - n_folds * FOLD_DAYS
    if first_test < LOOKBACK + 500:
        print(f"[{ticker}] skipped: not enough history for {n_folds} folds")
        return None

    naive = naive_predictions(close).to_numpy()
    ma5 = moving_average_predictions(close).to_numpy()

    rows = []
    for k in range(n_folds):
        start = first_test + k * FOLD_DAYS
        end = start + FOLD_DAYS
        print(f"[{ticker}] fold {k + 1}/{n_folds}: training on {start} days")
        pred_r, idx = train_lstm_and_predict(r, start, end, lookback=LOOKBACK, epochs=epochs)
        ci = idx + 1  # close index each prediction is for
        rows.append(pd.DataFrame({
            "fold": k + 1,
            "date": close.index[ci],
            "prev": c[ci - 1],
            "actual": c[ci],
            "lstm": c[ci - 1] * np.exp(pred_r),
            "lstm_ret": pred_r,
            "naive": naive[ci],
            "ma5": ma5[ci],
            "always_up": c[ci - 1] * 1.001,  # zero-skill "the stock goes up" guess
        }))
    return {"ticker": ticker, "oos": pd.concat(rows, ignore_index=True)}


def score(oos: pd.DataFrame) -> dict:
    actual_ret = np.log(oos["actual"] / oos["prev"])
    # The fair bar for directional skill is the share of up days, not 50%.
    up_rate = directional_accuracy(oos["prev"], oos["actual"], oos["always_up"])
    out = {"up_day_rate_pct": up_rate * 100}
    for name in ["naive", "ma5", "always_up", "lstm"]:
        p = oos[name]
        out[name] = {
            "rmse": rmse(oos["actual"], p),
            "mae": mae(oos["actual"], p),
            "mape_pct": mape(oos["actual"], p),
            "directional_accuracy_pct": directional_accuracy(oos["prev"], oos["actual"], p) * 100,
            "p_value_vs_coin_flip": direction_p_value(oos["prev"], oos["actual"], p, 0.5),
            "p_value_vs_always_up": direction_p_value(oos["prev"], oos["actual"], p, up_rate),
        }
    out["lstm"]["information_coefficient"] = information_coefficient(oos["lstm_ret"], actual_ret)
    out["lstm"]["directional_accuracy_by_fold_pct"] = [
        directional_accuracy(g["prev"], g["actual"], g["lstm"]) * 100 for _, g in oos.groupby("fold")
    ]

    bt = {}
    for name in ["lstm", "ma5"]:
        res = simple_signal_backtest(oos["date"], oos["prev"], oos["actual"], oos[name], cost_bps=COST_BPS)
        bt[f"{name}_signal"] = {**performance_stats(res["strategy_return"]),
                                "pct_days_in_market": float(res["position"].mean() * 100)}
        bt["buy_and_hold"] = performance_stats(res["buyhold_return"])
    out["backtest"] = bt
    out["test_period"] = [str(oos["date"].iloc[0].date()), str(oos["date"].iloc[-1].date())]
    out["test_days"] = len(oos)
    return out


def plot_equity(results: list[dict]):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.2 * len(results)), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        oos = res["oos"]
        for i, (name, label) in enumerate([("lstm", "LSTM signal"), ("ma5", "5-day MA signal")]):
            bt = simple_signal_backtest(oos["date"], oos["prev"], oos["actual"], oos[name], cost_bps=COST_BPS)
            if i == 0:
                ax.plot(bt["Date"], bt["buyhold_equity"], label="Buy and hold", color="gray",
                        linewidth=2.5, alpha=0.5)
            ax.plot(bt["Date"], bt["strategy_equity"], label=label, linewidth=1.2)
        ax.set_title(f"{res['ticker']}: $10,000 hypothetical, {COST_BPS:.0f} bps per trade")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "backtest_equity_curves.png"), dpi=130)
    plt.close(fig)


def plot_fold_accuracy(scores: dict):
    tickers = list(scores)
    n_folds = len(scores[tickers[0]]["lstm"]["directional_accuracy_by_fold_pct"])
    width = 0.8 / n_folds
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(n_folds):
        vals = [scores[t]["lstm"]["directional_accuracy_by_fold_pct"][k] for t in tickers]
        ax.bar(np.arange(len(tickers)) + k * width, vals, width, label=f"Fold {k + 1}")
    ax.axhline(50, color="black", linestyle="--", linewidth=1, label="Coin flip")
    ax.set_xticks(np.arange(len(tickers)) + width * (n_folds - 1) / 2)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("LSTM directional accuracy (%)")
    ax.set_ylim(30, 70)
    ax.legend(fontsize=8, ncol=n_folds + 1)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "lstm_accuracy_by_fold.png"), dpi=130)
    plt.close(fig)


def write_report(scores: dict, n_folds: int):
    lines = [
        "# Results",
        "",
        f"Generated by `main.py`. Walk-forward evaluation, {n_folds} folds of {FOLD_DAYS} trading days each, "
        f"{COST_BPS:.0f} bps transaction cost per position change.",
        "",
        "## Prediction quality (out of sample)",
        "",
        "Directional accuracy is tested two ways: against a coin flip (50%), and against the fairer bar of "
        "always guessing \"up,\" whose hit rate equals the share of up days in the test period.",
        "",
        "| Ticker | Test period | Model | RMSE ($) | Dir. accuracy | p vs. 50% | p vs. always up | IC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {"naive": "Naive", "ma5": "5-day MA", "always_up": "Always up", "lstm": "LSTM"}
    for t, s in scores.items():
        period = f"{s['test_period'][0]} to {s['test_period'][1]}"
        for m in ["naive", "ma5", "always_up", "lstm"]:
            ic = f"{s['lstm']['information_coefficient']:.3f}" if m == "lstm" else ""
            lines.append(f"| {t} | {period} | {labels[m]} | {s[m]['rmse']:.2f} | "
                         f"{s[m]['directional_accuracy_pct']:.1f}% | {s[m]['p_value_vs_coin_flip']:.3f} | "
                         f"{s[m]['p_value_vs_always_up']:.3f} | {ic} |")
    lines += [
        "",
        "## Backtest (out of sample, after costs)",
        "",
        "| Ticker | Strategy | Total return | Sharpe | Max drawdown | Days in market |",
        "|---|---|---|---|---|---|",
    ]
    names = {"lstm_signal": "LSTM signal", "ma5_signal": "5-day MA signal", "buy_and_hold": "Buy and hold"}
    for t, s in scores.items():
        for k, label in names.items():
            b = s["backtest"][k]
            days = f"{b['pct_days_in_market']:.0f}%" if "pct_days_in_market" in b else "100%"
            lines.append(f"| {t} | {label} | {b['total_return_pct']:+.1f}% | {b['sharpe']:.2f} | "
                         f"{b['max_drawdown_pct']:.1f}% | {days} |")
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = [res for t in args.tickers if (res := walk_forward(t, args.folds, args.epochs))]
    if not results:
        raise SystemExit("No tickers had data. Check your internet connection.")

    scores = {res["ticker"]: score(res["oos"]) for res in results}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(scores, f, indent=2)
    plot_equity(results)
    plot_fold_accuracy(scores)
    write_report(scores, args.folds)
    print(open(os.path.join(RESULTS_DIR, "RESULTS.md")).read())


if __name__ == "__main__":
    main()
