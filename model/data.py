"""Loads the bundled datasets once and keeps them in memory."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASETS = [
    {"id": "D1", "name": "S&P 500 constituents, daily prices", "source": "Kaggle all_stocks_5yr (plotly/datasets mirror)",
     "span": "Feb 2013 – Feb 2018", "used_for": "Clustering stocks"},
    {"id": "D2", "name": "GICS sector of each company", "source": "datasets/s-and-p-500-companies",
     "span": "current", "used_for": "External validation of clusters"},
    {"id": "D3", "name": "S&P 500 monthly index (Shiller)", "source": "datasets/s-and-p-500",
     "span": "Jan 1871 – latest", "used_for": "HMM regimes and backtest"},
    {"id": "D4", "name": "CBOE VIX fear index, daily", "source": "datasets/finance-vix",
     "span": "Jan 1990 – latest", "used_for": "Independent HMM cross-check"},
]


@lru_cache(maxsize=1)
def stock_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "stock_features.csv", index_col="ticker")
    # tickers missing from today's constituents list left the index (or were renamed) after 2018
    df["sector"] = df["sector"].replace("Unknown", "Left index since 2018")
    return df


@lru_cache(maxsize=1)
def daily_returns() -> pd.DataFrame:
    z = np.load(DATA_DIR / "returns_daily.npz", allow_pickle=False)
    df = pd.DataFrame(z["returns"].astype(np.float64),
                      index=pd.to_datetime(z["dates"].astype(str)),
                      columns=z["tickers"].astype(str))
    return df[stock_features().index]


@lru_cache(maxsize=1)
def market_daily() -> pd.Series:
    """Equal-weight market index of the 477 stocks (daily log-return)."""
    return daily_returns().mean(axis=1)


@lru_cache(maxsize=1)
def sp500_price() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "sp500_monthly.csv", parse_dates=["date"])
    return df.set_index("date")["price"].astype(float)


@lru_cache(maxsize=1)
def sp500_returns() -> pd.Series:
    return np.log(sp500_price()).diff().dropna()


@lru_cache(maxsize=1)
def vix_close() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "vix_daily.csv", parse_dates=["date"])
    return df.set_index("date")["close"].astype(float).dropna()
