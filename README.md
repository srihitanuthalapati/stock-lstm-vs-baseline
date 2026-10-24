# Stock Price Prediction: LSTM vs. Naive Baselines

An LSTM neural network trained to predict next-day AAPL closing prices,
benchmarked against two intentionally dumb baselines — and a backtest that
checks whether the model's predictions would actually be worth trading on.

**The finding:** the LSTM's price predictions look reasonable on a chart,
but they're 98.9% correlated with simply guessing "tomorrow = today," its
error is *worse* than that trivial guess (RMSE 6.38 vs. 1.72), and its
directional accuracy (49.5%) is statistically a coin flip. A backtest of a
simple trading rule built on its signal lost money over the test period
while buy-and-hold gained +19%. This is the classic failure mode of
applying price-level LSTMs to markets, and this repo measures it directly
instead of asserting it.

## Results (real, computed — not illustrative)

Test set: 461 trading days, AAPL, 2014-05-01 to 2016-02-29 (last 20% of a
2,305-day chronological split; the LSTM never sees test-period data during
training or scaling).

| Model | RMSE ($) | MAE ($) | MAPE | Directional accuracy |
|---|---|---|---|---|
| Naive ("tomorrow = today") | 1.72 | 1.27 | 1.16% | 0.0%* |
| 5-day moving average | 2.40 | 1.85 | 1.71% | 50.8% |
| LSTM | 6.38 | 5.79 | 5.21% | 49.5% |

\* Structural, not a flaw: the naive baseline predicts *zero* change every
day, so it can never be credited with correctly calling a direction. It
still gets the lowest price error, because on any given day the stock
rarely moves far from where it closed yesterday — which is exactly the
trap: low price error does not mean useful predictions.

**LSTM predictions vs. the naive baseline:** correlation = 0.989. The
model isn't finding independent signal — it's producing a smoothed, lagged
version of "yesterday's price," just with more error than copying it
directly would have.

**Backtest** ($10,000 hypothetical starting balance, same test period, no
fees/slippage modeled — see [Notes and limitations](#notes-and-limitations)):

| Strategy | Final equity | Return |
|---|---|---|
| Buy and hold | $11,904.24 | +19.04% |
| LSTM-signal (long when model predicts up) | $9,931.59 | −0.68% |

Plots: [`results/plots/predictions_vs_actual.png`](results/plots/predictions_vs_actual.png),
[`results/plots/backtest_equity_curve.png`](results/plots/backtest_equity_curve.png).
Full numbers: [`results/metrics.json`](results/metrics.json).

## How it works

1. **Data** (`src/fetch_data.py`) — daily AAPL closes. Tries a live
   `yfinance` download first; if that's unavailable it falls back to a
   bundled historical CSV (`data/AAPL_2007-01-03_2016-02-29.csv`, sourced
   from Yahoo Finance via the public `plotly/datasets` repo) so the
   pipeline always runs.
2. **Baselines** (`src/baseline.py`) — naive lag-1 and a 5-day trailing
   moving average, both computed without looking ahead.
3. **LSTM** (`src/model.py`) — a 2-layer LSTM (50 units each, dropout 0.2)
   over 60-day lookback windows of MinMax-scaled prices. The scaler is fit
   on the training split only, so there's no leakage from test-period
   prices.
4. **Metrics** (`src/metrics.py`) — RMSE, MAE, MAPE, directional accuracy,
   and the lag-similarity check, all computed identically for every model
   over the same test-period targets.
5. **Backtest** (`src/backtest.py`) — a simple daily long/cash rule driven
   by each model's signal, compared to buy-and-hold.

Run it yourself:

```bash
pip install -r requirements.txt
python3 main.py
```

Outputs land in `results/` (`metrics.json` and `plots/`).

## Notes and limitations

- The backtest is a hypothetical simulation only — no real account, no
  real trades, no fees or slippage. It exists to answer "would this
  signal have been worth anything," not as a trading system.
- The bundled fallback dataset ends in Feb 2016. Swap in a live
  `yfinance` range (edit `START`/`END` in `main.py`) for a more recent
  window when running somewhere with normal internet access — the
  pipeline and conclusions don't depend on this specific date range.
- One ticker, one lookback window, one architecture. The point of this
  project isn't to find a config where the LSTM wins — it's to show a
  rigorous way to check whether it actually did.

## Why this project

Stock-price LSTM tutorials are common, and most stop at "look, the
predicted line tracks the actual line." This repo adds the comparison
that tutorial usually skips: a fair baseline, a lag-similarity check, and
a backtest — the three questions you'd actually ask before trusting a
model's predictions with money.
