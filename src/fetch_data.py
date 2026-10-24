"""
Load historical daily closing prices for a ticker.

Tries a live download via yfinance first. If that's unavailable (e.g. no
network access, rate limiting), falls back to a bundled historical CSV so
the rest of the pipeline still runs. The bundled fallback currently only
covers AAPL, 2007-01-03 to 2016-02-29 (sourced from Yahoo Finance via
plotly/datasets' public stockdata2.csv).
"""
import os
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
FALLBACK_FILE = os.path.join(DATA_DIR, "AAPL_2007-01-03_2016-02-29.csv")


def _fetch_live(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].copy()
    df.index.name = "Date"
    return df


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame indexed by Date with a single 'Close' column."""
    try:
        df = _fetch_live(ticker, start, end)
        print(f"[fetch_data] Loaded live data for {ticker} via yfinance.")
        return df
    except Exception as e:
        print(f"[fetch_data] Live fetch failed ({e}); falling back to bundled CSV.")
        if ticker.upper() != "AAPL" or not os.path.exists(FALLBACK_FILE):
            raise RuntimeError(
                f"No live data and no bundled fallback available for {ticker}."
            )
        df = pd.read_csv(FALLBACK_FILE, parse_dates=["Date"], index_col="Date")
        return df


if __name__ == "__main__":
    df = fetch_prices("AAPL", "2007-01-01", "2016-03-01")
    print(df.shape)
    print(df.head())
    print(df.tail())
