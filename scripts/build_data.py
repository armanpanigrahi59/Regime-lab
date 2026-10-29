"""Build the compact data files the app ships with, from the original public datasets.

Usage
-----
    python scripts/build_data.py              # downloads the raw files from GitHub
    python scripts/build_data.py --raw DIR    # uses raw CSVs already in DIR

Raw sources (all public):
  D1  plotly/datasets            all_stocks_5yr.csv     S&P 500 daily OHLCV, Feb 2013 - Feb 2018
  D2  datasets/s-and-p-500-companies  constituents.csv  GICS sector per ticker
  D3  datasets/s-and-p-500       data.csv               Shiller S&P 500 monthly index, 1871 - today
  D4  datasets/finance-vix       vix-daily.csv          CBOE VIX daily, 1990 - today

Outputs (in ./data):
  stock_features.csv   477 tickers x 6 behavioural features + sector
  returns_daily.npz    daily log-returns matrix (1258 days x 477 tickers)
  sp500_monthly.csv    monthly S&P 500 price
  vix_daily.csv        daily VIX close
"""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/"
SOURCES = {
    "all_stocks_5yr.csv": BASE + "plotly/datasets/master/all_stocks_5yr.csv",
    "constituents.csv": BASE + "datasets/s-and-p-500-companies/main/data/constituents.csv",
    "sp500_monthly.csv": BASE + "datasets/s-and-p-500/main/data/data.csv",
    "vix.csv": BASE + "datasets/finance-vix/main/data/vix-daily.csv",
}
OUT = Path(__file__).resolve().parent.parent / "data"


def load(name, raw_dir):
    if raw_dir:
        return pd.read_csv(Path(raw_dir) / name)
    print("downloading", SOURCES[name])
    with urllib.request.urlopen(SOURCES[name]) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="folder that already holds the raw CSVs")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # ---- D1 + D2: stock features ------------------------------------------
    raw = load("all_stocks_5yr.csv", args.raw)
    raw["date"] = pd.to_datetime(raw["date"])
    gics = load("constituents.csv", args.raw)
    close = raw.pivot(index="date", columns="Name", values="close").sort_index()
    coverage = close.notna().mean()
    close = close[coverage[coverage >= 0.95].index].ffill().bfill()
    logret = np.log(close).diff().dropna()
    market = logret.mean(axis=1)
    mv = market.var()

    F = pd.DataFrame({
        "ann_return": logret.mean() * 252,
        "ann_vol": logret.std() * np.sqrt(252),
        "beta": logret.apply(lambda r: np.cov(r, market)[0, 1] / mv),
        "max_drawdown": close.apply(lambda p: (p / p.cummax() - 1).min()),
        "skew": logret.skew(),
    })
    F["idio_vol"] = np.sqrt(np.maximum(F.ann_vol ** 2 - (F.beta ** 2) * mv * 252, 0))
    F = F[["ann_return", "ann_vol", "beta", "idio_vol", "max_drawdown", "skew"]]
    F["sector"] = F.index.map(gics.set_index("Symbol")["GICS Sector"]).fillna("Unknown")
    F.index.name = "ticker"
    F.round(6).to_csv(OUT / "stock_features.csv")

    np.savez_compressed(
        OUT / "returns_daily.npz",
        dates=np.array(logret.index.strftime("%Y-%m-%d"), dtype="U10"),
        tickers=np.array([str(t) for t in logret.columns], dtype="U8"),
        returns=logret.values.astype(np.float32),
    )

    # ---- D3: S&P 500 monthly -------------------------------------------------
    spm = load("sp500_monthly.csv", args.raw)
    spm = spm.rename(columns={"Date": "date", "SP500": "price"})[["date", "price"]].dropna()
    spm = spm[spm.price > 0]
    spm.to_csv(OUT / "sp500_monthly.csv", index=False)

    # ---- D4: VIX daily ---------------------------------------------------------
    vix = load("vix.csv", args.raw)
    vix = vix.rename(columns={"DATE": "date", "CLOSE": "close"})[["date", "close"]].dropna()
    vix.to_csv(OUT / "vix_daily.csv", index=False)

    print(f"stocks {len(F)} | daily {logret.shape} | monthly {len(spm)} | vix {len(vix)}")


if __name__ == "__main__":
    main()
