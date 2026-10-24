# PricePredictor: Does an LSTM Actually Predict Stock Moves?

Most stock-prediction LSTM projects stop at a chart where the predicted line hugs the actual line. This project asks the harder question: does the model have real, repeatable skill on data it has never seen, and would that skill survive trading costs?

**Full results:** [`results/RESULTS.md`](results/RESULTS.md) (generated automatically by `main.py`, so the numbers always match the code), with full detail in [`results/metrics.json`](results/metrics.json).

## The finding

Across five tickers and four out-of-sample years (October 2022 to October 2026), the LSTM shows no real predictive skill once you compare it to the right baseline.

At first glance it looks good. On JPM and SPY it called the direction correctly 55.6% and 54.2% of the time, which is statistically significant against a coin flip (p < 0.01). But those years were a bull market, and simply guessing "up" every day was right 56.9% and 55.4% of the time. **The LSTM's directional accuracy was below the always-up baseline on all five tickers**, and none of its results beat that baseline with any significance (p ≥ 0.68). Its information coefficient ranged from −0.04 to 0.04, meaning its predicted returns barely relate to actual returns.

| Ticker | LSTM accuracy | Always-up accuracy | IC |
|---|---|---|---|
| AAPL | 50.8% | 54.2% | −0.037 |
| MSFT | 51.2% | 53.1% | 0.008 |
| JPM | 55.6% | 56.9% | −0.003 |
| XOM | 52.8% | 53.5% | 0.040 |
| SPY | 54.2% | 55.4% | 0.040 |

The backtest tells the same story. After costs, the LSTM signal trailed buy-and-hold on total return for four of five tickers (for example, +57% vs. +156% on AAPL). It beat buy-and-hold only on XOM (+128% vs. +105%), and since its XOM predictions were still no better than always guessing up, one win out of five is consistent with luck. The LSTM did usually have smaller drawdowns, but only because it sat in cash 7% to 27% of the time.

The takeaway: a model can look significantly better than random while adding nothing beyond the market's upward drift. Testing against the right baseline is what separates the two.

## Version 1: the price-level trap

The first version trained an LSTM on raw AAPL closing prices and tested it on April 2024 to August 2026. Its predictions were 98.8% correlated with simply guessing "tomorrow = today," its directional accuracy was 43.3% (worse than a coin flip), and a trading rule built on its signal lost 6.2% while buy-and-hold gained 89.2%. A model trained on price levels mostly learns to echo yesterday's price, so it looks accurate on a chart while having no useful signal.

## Version 2: a fairer, harder test

Version 2 fixes the ways version 1 could flatter or mislead:

- **Predicts returns, not prices.** The LSTM forecasts the next day's log return from the previous 60 days of returns, so it can't score well by copying yesterday's price.
- **Walk-forward validation.** Instead of one train/test split, each ticker is tested over four consecutive out-of-sample years. For each year the model is retrained on all data before it, then predicts that year one day at a time, the way it would be used in real life.
- **Multiple tickers.** AAPL, MSFT, JPM, XOM and SPY, so a result can't come from one lucky stock.
- **Statistical significance against the right bar.** Directional accuracy gets a one-sided binomial test against a coin flip and against an "always up" baseline. The second test matters: in a rising market more than half of days are up days, so a model that leans bullish beats 50% with no skill at all.
- **Information coefficient.** The rank correlation between predicted and actual returns, the standard quant measure of signal quality.
- **Realistic backtest.** 5 basis points of cost every time the strategy switches between stock and cash, with Sharpe ratio and max drawdown reported alongside total return.

Every model (naive "tomorrow = today," 5-day moving average, LSTM) is scored on exactly the same out-of-sample days.

## How it works

1. **Data** (`src/fetch_data.py`): daily adjusted closes from `yfinance`, with a bundled AAPL CSV (2007 to 2016) as an offline fallback.
2. **Baselines** (`src/baseline.py`): naive lag-1 and a 5-day trailing average, neither of which can see the day it predicts.
3. **LSTM** (`src/model.py`): two LSTM layers (32 units, 0.2 dropout) over 60-day windows of standardized returns. Standardization uses training-period statistics only, and early stopping is judged on the most recent 10% of the training data.
4. **Metrics** (`src/metrics.py`): RMSE, MAE, MAPE, directional accuracy with p-value, and information coefficient.
5. **Backtest** (`src/backtest.py`): long the stock on days the model predicts up, cash otherwise, net of costs.
6. **Pipeline** (`main.py`): runs the walk-forward loop and writes `results/metrics.json`, `results/RESULTS.md`, and the plots in `results/plots/`.

## Run it

```bash
pip install -r requirements.txt
python3 main.py
```

The full run trains 20 models and takes a while on a laptop CPU. For a quick check:

```bash
python3 main.py --tickers AAPL --folds 2 --epochs 10
```

## Notes and limitations

- The backtest is a paper simulation with no real account or trades. It answers "would this signal have been worth anything," not "how to trade."
- An LSTM that finds no real pattern tends to predict the average daily return, which is slightly positive, so it ends up long almost every day and its backtest nearly matches buy-and-hold. The "days in market" column makes that visible.
- One architecture and one feature (past returns). The goal isn't to find a setup where the LSTM wins; it's to measure honestly whether it did.
