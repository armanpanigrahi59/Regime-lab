# Regime Lab

**Market Regimes & Stock Personalities: Clustering + Hidden Markov Models (PBL-2)**

A website with a Python model behind it.

- **Website:** plain `index.html`, `style.css` and `script.js` in `public/`.
- **Model:** Python in `app.py` and `model/`. It clusters 477 S&P 500 stocks (K-Means written from scratch, plus GMM, Agglomerative and DBSCAN) and fits a Hidden Markov Model to 155 years of S&P 500 returns. Every chart on the site is computed live by this model.

```
app.py              the Python API (FastAPI): every /api/... route
model/
  clustering.py     K-Means from scratch, GMM, Agglomerative, DBSCAN, metrics, stock explorer
  regimes.py        HMM: Baum-Welch fit, our own forward algorithm + Viterbi, model selection, VIX check
  strategy.py       out-of-sample backtest, clusters x regimes fusion
  data.py, utils.py
public/
  index.html        the website
  style.css
  script.js
data/               the four public datasets (2.4 MB)
requirements.txt    Python packages
vercel.json         gives the model up to 60 s per request
notebook/           the Google Colab notebook
```

## Deploy on Vercel

1. Put this folder in a GitHub repository. On github.com, choose **New repository**, then **uploading an existing file**, and drag in everything inside this folder.
2. On **vercel.com**, choose **Add New → Project**, import the repository, and click **Deploy**. There is nothing to configure: Vercel finds `app.py` and `requirements.txt` and serves `public/` as the website.
3. The first build takes about 3–4 minutes because it installs scikit-learn and the other packages.

**Before you present:** open the site and click through all 7 sections once.
- The first visit to each section runs the model, which takes up to about 15 seconds.
- Vercel then caches every result, so later visits load instantly for everyone.
- Changing a slider to a new value runs the model again, live.

## Run on your laptop

Needs Python 3.10–3.13.

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open **http://localhost:8000**. The API reference is at **/docs**.

## API

| Endpoint | What it computes |
|---|---|
| `/api/overview` | datasets, headline numbers, current regime and next-month forecast |
| `/api/cluster?algo=kmeans&k=4` | clustering (`kmeans`, `gmm`, `agglomerative`, `dbscan`) |
| `/api/cluster/select-k` | elbow, silhouette, Davies-Bouldin, GMM BIC for k = 2..10 |
| `/api/cluster/compare` | all four algorithms with bootstrap stability |
| `/api/lab/kmeans?k=4&dataset=stocks` | every step of the from-scratch K-Means |
| `/api/stocks/KO` | one stock's fingerprint, cluster and look-alikes |
| `/api/hmm?n_states=3&start=1871&end=2026` | fit and decode the HMM |
| `/api/hmm/select` | log-likelihood, AIC, BIC for 1–5 states |
| `/api/hmm/viterbi?start=2007-06&length=18` | Viterbi trellis (δ, ψ, path) |
| `/api/hmm/compare` | HMM vs K-Means vs GMM on monthly returns |
| `/api/vix` | independent VIX HMM cross-check |
| `/api/backtest?train_end=1989&rule=bull&threshold=0.5` | out-of-sample backtest |
| `/api/fusion?algo=kmeans&k=4` | cluster returns in calm vs stressed markets |

## Data (all public)

- **D1:** S&P 500 daily prices 2013–2018 (Kaggle *all_stocks_5yr*)
- **D2:** GICS sectors
- **D3:** Shiller S&P 500 monthly index, 1871–Sep 2026
- **D4:** CBOE VIX daily, 1990–Sep 2026

`scripts/build_data.py` rebuilds `data/` from the original CSVs.
