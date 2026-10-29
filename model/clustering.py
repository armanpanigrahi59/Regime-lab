"""Stock clustering: K-Means (written from scratch in NumPy), GMM, Agglomerative and DBSCAN."""
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import stock_features

FEATURES = ["ann_return", "ann_vol", "beta", "idio_vol", "max_drawdown", "skew"]
FEATURE_LABELS = {
    "ann_return": "Annual return", "ann_vol": "Annual volatility", "beta": "Market beta",
    "idio_vol": "Stock-specific vol", "max_drawdown": "Max drawdown", "skew": "Skewness",
}
NAME_COLORS = {
    "Defensive": "#3d8bfd", "Growth Leaders": "#ff7a45", "High-β Cyclical": "#20c997",
    "Event-Shock": "#f5b301", "Core Market": "#a78bfa", "Laggards": "#f06595",
    "Momentum": "#63e6be", "Speculative": "#ffa94d", "Outliers": "#8a94a6",
}
EXTRA_COLORS = ["#74c0fc", "#d0bfff", "#ffd8a8", "#b2f2bb"]
ALGOS = {"kmeans": "K-Means (from scratch)", "gmm": "Gaussian Mixture",
         "agglomerative": "Agglomerative (hierarchical)", "dbscan": "DBSCAN"}
SEED = 42


# ---------------------------------------------------------------- data prep
@lru_cache(maxsize=1)
def matrices():
    """Winsorised features, standardised matrix X, and a 2-D PCA view oriented so that
    PC1 grows with risk and PC2 grows with return."""
    F = stock_features()
    raw = F[FEATURES]
    Fw = raw.clip(raw.quantile(0.01), raw.quantile(0.99), axis=1)
    X = StandardScaler().fit_transform(Fw)
    pca = PCA(2, random_state=SEED).fit(X)
    P = pca.transform(X)
    sign = np.array([
        1.0 if np.corrcoef(P[:, 0], Fw.ann_vol)[0, 1] >= 0 else -1.0,
        1.0 if np.corrcoef(P[:, 1], Fw.ann_return)[0, 1] >= 0 else -1.0,
    ])
    return F, Fw, X, P * sign, pca, sign


def project(points6d):
    _, _, _, _, pca, sign = matrices()
    return pca.transform(np.atleast_2d(points6d)) * sign


# ---------------------------------------------------------------- K-Means from scratch
def kmeans_pp_init(X, k, rng):
    centroids = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centroids)[None]) ** 2).sum(-1), axis=1)
        centroids.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centroids)


def kmeans_scratch(X, k, seed=0, max_iter=300, tol=1e-8, record=False):
    """Lloyd's algorithm: assign every point to its nearest centroid, move each centroid
    to the mean of its points, repeat until no centroid moves."""
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, k, rng)
    frames, history = [], []
    labels = np.full(len(X), -1)
    if record:
        frames.append({"phase": "init", "centroids": C.copy(), "labels": labels.copy(), "inertia": None, "moved": None})
    for it in range(max_iter):
        dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)          # assignment step
        new_labels = dist.argmin(1)
        moved = int((new_labels != labels).sum())
        labels = new_labels
        inertia = float(dist[np.arange(len(X)), labels].sum())
        history.append(inertia)
        if record:
            frames.append({"phase": "assign", "centroids": C.copy(), "labels": labels.copy(), "inertia": inertia, "moved": moved})
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])  # update step
        shift = float(np.abs(newC - C).max())
        C = newC
        if record:
            frames.append({"phase": "update", "centroids": C.copy(), "labels": labels.copy(), "inertia": inertia, "moved": moved})
        if shift < tol:
            break
    dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    labels = dist.argmin(1)
    inertia = float(dist[np.arange(len(X)), labels].sum())
    return labels, C, inertia, history, frames


def kmeans_best(X, k, n_init=10):
    best = None
    for s in range(n_init):
        lab, C, inertia, hist, _ = kmeans_scratch(X, k, seed=s)
        if best is None or inertia < best[2]:
            best = (lab, C, inertia, hist)
    return best


# ---------------------------------------------------------------- naming
def name_clusters(labels, Fw):
    """Give each cluster a human name from its fingerprint (works for any algorithm)."""
    ids = sorted(set(int(l) for l in labels) - {-1})
    prof = Fw.groupby(labels).mean()
    left, names = list(ids), {}

    def take(col, how, name, cond=None):
        if not left:
            return
        s = prof.loc[left, col]
        c = int(s.idxmin() if how == "min" else s.idxmax())
        if cond is None or cond(prof.loc[c]):
            names[c] = name
            left.remove(c)

    take("ann_vol", "min", "Defensive")
    take("skew", "min", "Event-Shock", cond=lambda p: p["skew"] < -3)
    take("ann_return", "max", "Growth Leaders")
    take("beta", "max", "High-β Cyclical")
    take("ann_return", "min", "Laggards")
    for extra in ["Core Market", "Momentum", "Speculative"]:
        take("ann_vol", "min", extra)
    for i, c in enumerate(list(left)):
        names[c] = f"Cluster {i + 1}"
    names[-1] = "Outliers"
    return names


def color_for(name, i=0):
    return NAME_COLORS.get(name, EXTRA_COLORS[i % len(EXTRA_COLORS)])


# ---------------------------------------------------------------- metrics
def internal_metrics(X, labels):
    m = labels != -1
    k = len(set(labels[m]))
    if k < 2 or m.sum() <= k:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": float(silhouette_score(X[m], labels[m])),
        "davies_bouldin": float(davies_bouldin_score(X[m], labels[m])),
        "calinski_harabasz": float(calinski_harabasz_score(X[m], labels[m])),
    }


def dbscan_auto_eps(X, min_samples):
    kd = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1])
    return float(np.quantile(kd, 0.90)), kd


def fit_labels(algo, k, eps=None, min_samples=12, linkage="ward", covariance="full", X=None):
    """Return integer labels for one algorithm (used by the API and by the stability test)."""
    X = matrices()[2] if X is None else X
    if algo == "kmeans":
        return kmeans_best(X, k)[0]
    if algo == "gmm":
        return GaussianMixture(k, covariance_type=covariance, n_init=5, random_state=SEED).fit(X).predict(X)
    if algo == "agglomerative":
        return AgglomerativeClustering(k, linkage=linkage).fit_predict(X)
    if algo == "dbscan":
        e = eps if eps else dbscan_auto_eps(X, min_samples)[0]
        return DBSCAN(eps=e, min_samples=min_samples).fit_predict(X)
    raise ValueError(algo)


# ---------------------------------------------------------------- main entry
@lru_cache(maxsize=64)
def run_clustering(algo="kmeans", k=4, eps=None, min_samples=12, linkage="ward", covariance="full"):
    F, Fw, X, P, _, _ = matrices()
    extras = {}
    t0 = time.perf_counter()
    if algo == "kmeans":
        labels, C, inertia, hist = kmeans_best(X, k)
        ms = (time.perf_counter() - t0) * 1000
        sk = KMeans(k, n_init=10, random_state=SEED).fit(X)
        extras["verification"] = {
            "inertia_scratch": inertia, "inertia_sklearn": float(sk.inertia_),
            "ari_vs_sklearn": float(adjusted_rand_score(labels, sk.labels_)),
            "iterations": len(hist), "inertia_history": hist,
        }
        extras["centroids_2d"] = project(C).tolist()
    elif algo == "dbscan":
        auto, kd = dbscan_auto_eps(X, min_samples)
        e = eps if eps else auto
        labels = DBSCAN(eps=e, min_samples=min_samples).fit_predict(X)
        ms = (time.perf_counter() - t0) * 1000
        extras["eps_used"], extras["eps_auto"] = e, auto
        extras["k_distance"] = kd[:: max(1, len(kd) // 160)].tolist()
    else:
        labels = fit_labels(algo, k, linkage=linkage, covariance=covariance)
        ms = (time.perf_counter() - t0) * 1000
        if algo == "gmm":
            g = GaussianMixture(k, covariance_type=covariance, n_init=5, random_state=SEED).fit(X)
            extras["bic"], extras["aic"] = float(g.bic(X)), float(g.aic(X))
            extras["mean_confidence"] = float(g.predict_proba(X).max(1).mean())
    labels = np.asarray(labels, dtype=int)

    names = name_clusters(labels, Fw)
    order = [names[c] for c in sorted(set(labels)) if c != -1]
    order = sorted(order, key=lambda n: list(NAME_COLORS).index(n) if n in NAME_COLORS else 99)
    if -1 in set(labels):
        order.append("Outliers")
    colors = {n: color_for(n, i) for i, n in enumerate(order)}
    lab_names = np.array([names[int(l)] for l in labels])

    mu, sd = Fw.mean(), Fw.std()
    clusters = []
    for n in order:
        m = lab_names == n
        means = F.loc[m, FEATURES].mean()
        sub = F.loc[m]
        clusters.append({
            "name": n, "color": colors[n], "n": int(m.sum()),
            "means": {f: float(means[f]) for f in FEATURES},
            "z": {f: float((Fw.loc[m, f].mean() - mu[f]) / sd[f]) for f in FEATURES},
            "examples": sub.sort_values("ann_vol").index[:8].tolist(),
            "top_sectors": sub.sector[sub.sector != "Left index since 2018"].value_counts().head(3).to_dict(),
        })

    sectors = sorted(F.sector.unique())
    ct = pd.crosstab(F.sector, lab_names, normalize="index").reindex(columns=order, fill_value=0)
    metrics = internal_metrics(X, labels)
    metrics.update({
        "ari_vs_sector": float(adjusted_rand_score(F.sector, labels)),
        "n_clusters": int(len(set(labels) - {-1})), "n_noise": int((labels == -1).sum()),
        "time_ms": round(ms, 1),
    })
    points = {
        "ticker": F.index.tolist(), "x": np.round(P[:, 0], 4).tolist(), "y": np.round(P[:, 1], 4).tolist(),
        "cluster": lab_names.tolist(), "sector": F.sector.tolist(),
        "ret": np.round(F.ann_return.values, 4).tolist(), "vol": np.round(F.ann_vol.values, 4).tolist(),
        "beta": np.round(F.beta.values, 3).tolist(),
    }
    return {
        "algo": algo, "algo_label": ALGOS[algo],
        "params": {"k": k, "eps": eps, "min_samples": min_samples, "linkage": linkage, "covariance": covariance},
        "features": FEATURES, "feature_labels": FEATURE_LABELS,
        "points": points, "clusters": clusters, "order": order, "colors": colors,
        "sector_mix": {"sectors": sectors, "shares": {n: ct[n].reindex(sectors).round(4).tolist() for n in order}},
        "metrics": metrics, "extras": extras,
        "pca_variance": matrices()[4].explained_variance_ratio_.round(4).tolist(),
    }


@lru_cache(maxsize=1)
def select_k():
    X = matrices()[2]
    rows = []
    for k in range(2, 11):
        lab, _, inertia, _ = kmeans_best(X, k)
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": inertia, "silhouette": float(silhouette_score(X, lab)),
                     "davies_bouldin": float(davies_bouldin_score(X, lab)),
                     "calinski_harabasz": float(calinski_harabasz_score(X, lab)), "gmm_bic": float(g.bic(X))})
    return {"rows": rows, "chosen": 4,
            "reason": "Elbow bends near 4, GMM BIC flattens after 4, and k = 4 has the best silhouette among k ≥ 3. "
                      "k = 2 scores highest on silhouette but only splits risky vs safe."}


@lru_cache(maxsize=1)
def compare_algorithms(B=10):
    F, _, X, _, _, _ = matrices()
    rng = np.random.default_rng(0)
    boots = [rng.choice(len(X), len(X), replace=True) for _ in range(B)]
    eps = dbscan_auto_eps(X, 12)[0]
    specs = {"kmeans": dict(k=4), "gmm": dict(k=4), "agglomerative": dict(k=4), "dbscan": dict(k=0, eps=eps)}
    rows = []
    for algo, kw in specs.items():
        res = run_clustering(algo, kw["k"], kw.get("eps"))
        base = np.array([res["order"].index(c) for c in res["points"]["cluster"]])
        stab = []
        for idx in boots:
            if algo == "kmeans":
                lab = kmeans_best(X[idx], 4, n_init=3)[0]
            elif algo == "dbscan":
                lab = DBSCAN(eps=eps, min_samples=12).fit_predict(X[idx])
            else:
                lab = fit_labels(algo, 4, X=X[idx])
            stab.append(adjusted_rand_score(base[idx], lab))
        m = res["metrics"]
        rows.append({"algo": algo, "label": ALGOS[algo], "clusters": m["n_clusters"], "noise": m["n_noise"],
                     "silhouette": m["silhouette"], "davies_bouldin": m["davies_bouldin"],
                     "calinski_harabasz": m["calinski_harabasz"], "stability": float(np.mean(stab)),
                     "ari_vs_sector": m["ari_vs_sector"], "time_ms": m["time_ms"]})
    return {"rows": rows, "bootstrap_runs": B,
            "verdict": "K-Means wins every internal metric and is the most stable real segmenter. GMM over-fits "
                       "six-dimensional covariances. DBSCAN finds one dense core plus outliers, so it works best "
                       "as an outlier detector."}


# ---------------------------------------------------------------- algorithm lab
def _synthetic(dataset, seed=7):
    rng = np.random.default_rng(seed)
    if dataset == "blobs":
        centers = [(-3, -2), (2.5, -1.5), (0, 2.5), (3.5, 2.5)]
        return np.vstack([rng.normal(c, 0.7, size=(70, 2)) for c in centers])
    a, b = np.pi * rng.random(150), np.pi * rng.random(150)
    m1 = np.c_[np.cos(a) * 3 - 1.2, np.sin(a) * 3 - 0.6]
    m2 = np.c_[1.8 - np.cos(b) * 3, -np.sin(b) * 3 + 1.2]
    return np.vstack([m1, m2]) + rng.normal(0, 0.18, size=(300, 2))


@lru_cache(maxsize=64)
def kmeans_animation(k=4, dataset="stocks", seed=1):
    """Every assignment and update step of scratch K-Means, ready to animate.
    On real stocks the algorithm runs in all 6 dimensions; centroids are projected to the PCA view."""
    if dataset == "stocks":
        _, _, X, P, _, _ = matrices()
        labels, C, inertia, hist, frames = kmeans_scratch(X, k, seed=seed, record=True)
        pts = P
        proj = project
    else:
        X = _synthetic(dataset)
        labels, C, inertia, hist, frames = kmeans_scratch(X, k, seed=seed, record=True)
        pts = X
        proj = lambda c: np.asarray(c)
    out = [{"phase": f["phase"], "centroids": np.round(proj(f["centroids"]), 4).tolist(),
            "labels": f["labels"].tolist(), "inertia": f["inertia"], "moved": f["moved"]} for f in frames]
    return {"dataset": dataset, "k": k, "seed": seed, "points": np.round(pts, 4).tolist(),
            "tickers": stock_features().index.tolist() if dataset == "stocks" else None,
            "frames": out, "final_inertia": inertia, "iterations": len(hist),
            "space": "6-D features, shown on 2 PCA axes" if dataset == "stocks" else "2-D"}


# ---------------------------------------------------------------- stock explorer
def stock_detail(ticker, returns_df):
    F, Fw, X, P, _, _ = matrices()
    if ticker not in F.index:
        return None
    i = F.index.get_loc(ticker)
    res = run_clustering("kmeans", 4)
    d = np.sqrt(((X - X[i]) ** 2).sum(1))
    nn = np.argsort(d)[1:7]
    pct = {f: float((F[f] < F[f].iloc[i]).mean()) for f in FEATURES}
    px = np.exp(returns_df[ticker].cumsum())
    weekly = px.resample("W").last()
    mkt = np.exp(returns_df.mean(axis=1).cumsum()).resample("W").last()
    return {
        "ticker": ticker, "sector": F.sector.iloc[i], "cluster": res["points"]["cluster"][i],
        "color": res["colors"][res["points"]["cluster"][i]],
        "features": {f: float(F[f].iloc[i]) for f in FEATURES}, "percentiles": pct,
        "neighbours": [{"ticker": F.index[j], "sector": F.sector.iloc[j], "cluster": res["points"]["cluster"][j],
                        "distance": float(d[j])} for j in nn],
        "price": {"dates": weekly.index.strftime("%Y-%m-%d").tolist(), "value": np.round(weekly.values, 4).tolist(),
                  "market": np.round(mkt.values, 4).tolist()},
    }
