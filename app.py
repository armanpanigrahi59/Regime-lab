"""Regime Lab: the Python model behind the website.

Vercel finds this file (app.py) and its FastAPI `app` automatically. The website itself
(public/index.html, style.css, script.js) is served straight from Vercel's CDN.

Run locally:  uvicorn app:app --reload     then open http://localhost:8000
API docs:     /docs
"""
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

# Prevent loky subprocess core-detection failure and warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", message=".*physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*loky.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*joblib.*")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import clustering, regimes, strategy
from model.data import DATASETS, daily_returns, sp500_returns, stock_features, vix_close
from model.utils import clean

PUBLIC = Path(__file__).resolve().parent / "public"

app = FastAPI(
    title="Regime Lab API",
    version="1.0.0",
    description="Clustering of 477 S&P 500 stocks and a Hidden Markov Model of 155 years of market regimes. "
                "Every endpoint computes its result live from the bundled datasets.",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

STARTED = time.time()


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Compute-Ms"] = f"{(time.perf_counter() - t) * 1000:.1f}"
    resp.headers["Access-Control-Expose-Headers"] = "X-Compute-Ms"
    # Results are deterministic for given parameters, so let Vercel's CDN cache them:
    # the first visitor computes a result, everyone after gets it instantly.
    if request.url.path.startswith("/api/") and request.url.path != "/api/health" and resp.status_code == 200:
        resp.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800, stale-while-revalidate=86400"
    return resp


def ok(payload):
    return JSONResponse(clean(payload))


def guard(fn, *a, **kw):
    try:
        return ok(fn(*a, **kw))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------ meta
@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED, 1), "on_vercel": bool(os.environ.get("VERCEL"))}


@lru_cache(maxsize=1)
def _overview_payload():
    F, R, r, v = stock_features(), daily_returns(), sp500_returns(), vix_close()
    reg = regimes.regimes(3, 1871, 2026)
    cl = clustering.run_clustering("kmeans", 4)
    bt = strategy.backtest(1989, "bull", 0.5, 0.0, 0.0)
    sizes = {"D1": f"{len(F)} stocks × {len(R)} days", "D2": f"{F.sector.nunique()} sectors",
             "D3": f"{len(r):,} monthly returns", "D4": f"{len(v):,} daily closes"}
    spans = {"D1": f"{R.index[0]:%b %Y} – {R.index[-1]:%b %Y}", "D3": f"{r.index[0]:%b %Y} – {r.index[-1]:%b %Y}",
             "D4": f"{v.index[0]:%b %Y} – {v.index[-1]:%b %Y}"}
    ds = [{**d, "size": sizes[d["id"]], "span": spans.get(d["id"], d["span"])} for d in DATASETS]
    return clean({
        "datasets": ds,
        "now": {**reg["now"], "names": reg["names"], "colors": reg["colors"]},
        "headline": {
            "stocks": len(F), "months": len(r), "years": int(r.index[-1].year - r.index[0].year),
            "clusters": [{"name": c["name"], "color": c["color"], "n": c["n"]} for c in cl["clusters"]],
            "regimes": [{"name": s["name"], "color": s["color"], "share": s["share"]} for s in reg["states"]],
            "drawdown_bh": bt["stats"]["buy_hold"]["max_drawdown"], "drawdown_hmm": bt["stats"]["strategy"]["max_drawdown"],
            "sharpe_bh": bt["stats"]["buy_hold"]["sharpe"], "sharpe_hmm": bt["stats"]["strategy"]["sharpe"],
        },
        "spark": {"dates": reg["series"]["dates"][-240:], "price": reg["series"]["price"][-240:],
                  "state": reg["series"]["state"][-240:]},
    })


@app.get("/api/overview", tags=["meta"], summary="Datasets, headline numbers and today's regime")
def overview():
    return JSONResponse(_overview_payload())


# ------------------------------------------------------------------ clustering
@app.get("/api/cluster", tags=["clustering"], summary="Run one clustering algorithm on the 477 stocks")
def cluster(algo: Literal["kmeans", "gmm", "agglomerative", "dbscan"] = "kmeans",
            k: int = Query(4, ge=2, le=10),
            eps: Optional[float] = Query(None, gt=0.1, le=6, description="DBSCAN radius; empty = auto from k-distance knee"),
            min_samples: int = Query(12, ge=3, le=40),
            linkage: Literal["ward", "complete", "average"] = "ward",
            covariance: Literal["full", "diag", "tied", "spherical"] = "full"):
    if algo != "dbscan":
        eps, min_samples = None, 12
    if algo != "agglomerative":
        linkage = "ward"
    if algo != "gmm":
        covariance = "full"
    if algo == "dbscan":
        k = 0
    return guard(clustering.run_clustering, algo, k, round(eps, 3) if eps else None, min_samples, linkage, covariance)


@app.get("/api/cluster/select-k", tags=["clustering"], summary="Elbow, silhouette, Davies-Bouldin and GMM BIC for k = 2..10")
def cluster_select_k():
    return ok(clustering.select_k())


@app.get("/api/cluster/compare", tags=["comparison"], summary="All four algorithms with bootstrap stability")
def cluster_compare():
    return ok(clustering.compare_algorithms())


@app.get("/api/lab/kmeans", tags=["algorithm lab"], summary="Every step of scratch K-Means, for animation")
def lab_kmeans(k: int = Query(4, ge=2, le=8), dataset: Literal["stocks", "blobs", "moons"] = "stocks",
               seed: int = Query(1, ge=0, le=999)):
    return ok(clustering.kmeans_animation(k, dataset, seed))


@lru_cache(maxsize=1)
def _stocks_payload():
    F = stock_features()
    res = clustering.run_clustering("kmeans", 4)
    return clean([{"ticker": t, "sector": s, "cluster": c} for t, s, c in zip(F.index, F.sector, res["points"]["cluster"])])


@app.get("/api/stocks", tags=["stocks"], summary="Ticker list for search")
def stocks():
    return JSONResponse(_stocks_payload())


@app.get("/api/stocks/{ticker}", tags=["stocks"], summary="One stock's fingerprint, cluster and nearest look-alikes")
def stock(ticker: str):
    d = clustering.stock_detail(ticker.upper())
    if d is None:
        raise HTTPException(404, f"{ticker.upper()} is not in the 477-stock universe")
    return ok(d)


# ------------------------------------------------------------------ HMM
@app.get("/api/hmm", tags=["hmm"], summary="Fit a Gaussian HMM to monthly S&P 500 returns")
def hmm(n_states: int = Query(3, ge=2, le=5), start: int = Query(1871, ge=1871, le=2000),
        end: int = Query(2026, ge=1950, le=2026)):
    if end - start < 40:
        raise HTTPException(400, "Pick a window of at least 40 years so every regime has enough months")
    return guard(regimes.regimes, n_states, start, end)


@app.get("/api/hmm/select", tags=["hmm"], summary="Log-likelihood, AIC and BIC for 1..5 hidden states")
def hmm_select():
    return ok(regimes.select_states())


@app.get("/api/hmm/viterbi", tags=["algorithm lab"], summary="Viterbi trellis (delta, back-pointers, path) for a window")
def hmm_viterbi(start: str = Query("2007-06", pattern=r"^\d{4}-\d{2}$"), length: int = Query(18, ge=6, le=36)):
    return guard(regimes.trellis, start, length)


@app.get("/api/hmm/compare", tags=["comparison"], summary="HMM vs K-Means vs GMM on the same monthly returns")
def hmm_compare():
    return ok(regimes.compare_regime_models())


@app.get("/api/vix", tags=["comparison"], summary="Independent 3-state HMM on the VIX, cross-checked against S&P regimes")
def vix():
    return ok(regimes.vix_regimes())


# ------------------------------------------------------------------ strategy
@app.get("/api/backtest", tags=["strategy"], summary="Out-of-sample regime-switching backtest")
def backtest(train_end: int = Query(1989, ge=1930, le=2010), rule: Literal["bull", "avoid_crisis"] = "bull",
             threshold: float = Query(0.5, ge=0.05, le=0.95), cost_bps: float = Query(0, ge=0, le=100),
             cash_rate: float = Query(0, ge=0, le=0.1)):
    return guard(strategy.backtest, train_end, rule, round(threshold, 2), round(cost_bps, 1), round(cash_rate, 4))


@app.get("/api/fusion", tags=["strategy"], summary="How each stock cluster behaves in calm vs stressed markets")
def fusion(algo: Literal["kmeans", "gmm", "agglomerative"] = "kmeans", k: int = Query(4, ge=2, le=8)):
    return guard(strategy.fusion, algo, k)


# ------------------------------------------------------------------ website
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PUBLIC / "index.html")



# On Vercel the public/ folder is served by the CDN, so nothing is mounted there.
# Locally (uvicorn app:app) FastAPI serves the same files itself.
if not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="site")
