"""Hidden Markov Model of market regimes.

Training uses hmmlearn's Baum-Welch (EM). Decoding and real-time filtering use our own
NumPy implementations of the forward algorithm and Viterbi, which are checked against
hmmlearn on every fit."""
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .data import sp500_price, sp500_returns, vix_close

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

STATE_COLORS = {"Strong Bull": "#0ca678", "Bull": "#20c997", "Flat": "#74c0fc", "Correction": "#3d8bfd",
                "Crisis": "#ff6b3d", "Calm": "#20c997", "Turbulent": "#ff6b3d",
                "Nervous": "#3d8bfd", "Fear": "#ff6b3d"}
EVENTS = [("1873-10", "Panic of 1873"), ("1893-05", "Panic of 1893"), ("1907-10", "Panic of 1907"),
          ("1929-10", "1929 crash"), ("1937-09", "1937 recession"), ("1974-01", "Oil shock"),
          ("1987-10", "Black Monday"), ("2000-09", "Dot-com bust"), ("2008-10", "Financial crisis"),
          ("2020-03", "COVID crash"), ("2022-06", "2022 rate shock")]


# ---------------------------------------------------------------- model core
def fit_hmm(obs, n, restarts=8, n_iter=500):
    obs = np.asarray(obs, float).reshape(-1, 1)
    best, best_ll = None, -np.inf
    for s in range(restarts):
        m = GaussianHMM(n, covariance_type="diag", n_iter=n_iter, tol=1e-6, random_state=s)
        try:
            m.fit(obs)
            ll = m.score(obs)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    return best, float(best_ll)


def params(model):
    mu = model.means_.ravel()
    sd = np.sqrt(np.array([c.ravel()[0] for c in model.covars_]))
    return mu, sd


def order_states(model):
    """Ordered state ids and names: the highest-volatility state is Crisis, the rest go by mean return."""
    mu, sd = params(model)
    n = len(mu)
    if n == 1:
        return [0], ["Market"]
    if n == 2:
        calm = int(np.argmin(sd))
        return [calm, 1 - calm], ["Calm", "Turbulent"]
    crisis = int(np.argmax(sd))
    rest = sorted([i for i in range(n) if i != crisis], key=lambda i: -mu[i])
    names = {1: ["Bull"], 2: ["Bull", "Correction"], 3: ["Strong Bull", "Bull", "Correction"],
             4: ["Strong Bull", "Bull", "Flat", "Correction"]}[len(rest)]
    return rest + [crisis], names + ["Crisis"]


def emission_logp(model, x):
    mu, sd = params(model)
    return norm.logpdf(np.asarray(x, float).reshape(-1, 1), mu, sd)


def forward_filter(model, x):
    """P(S_t | r_1..r_t) for every t, using only data up to t (scaled forward algorithm)."""
    logB = emission_logp(model, x)
    T, N = logB.shape
    alpha = np.zeros((T, N))
    a = model.startprob_ * np.exp(logB[0])
    c = a.sum(); alpha[0] = a / c; ll = np.log(c)
    for t in range(1, T):
        a = (alpha[t - 1] @ model.transmat_) * np.exp(logB[t])
        c = a.sum(); alpha[t] = a / c; ll += np.log(c)
    return alpha, float(ll)


def viterbi(model, x, keep=False):
    """Most likely state path. With keep=True also returns delta (log) and back-pointers psi."""
    logB = emission_logp(model, x)
    logA = np.log(model.transmat_ + 1e-300)
    T, N = logB.shape
    delta = np.zeros((T, N)); psi = np.zeros((T, N), int)
    delta[0] = np.log(model.startprob_ + 1e-300) + logB[0]
    for t in range(1, T):
        s = delta[t - 1][:, None] + logA
        psi[t] = s.argmax(0)
        delta[t] = s.max(0) + logB[t]
    path = np.zeros(T, int); path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return (path, delta, psi, logB) if keep else path


def n_params(n):
    return (n - 1) + n * (n - 1) + 2 * n


def runs(labels):
    labels = np.asarray(labels)
    switches = int((np.diff(labels) != 0).sum())
    return switches, len(labels) / (switches + 1)


# ---------------------------------------------------------------- regimes
def _window(start, end):
    r = sp500_returns()
    return r[(r.index.year >= start) & (r.index.year <= end)]


@lru_cache(maxsize=32)
def _fitted(n_states, start, end):
    r = _window(start, end)
    model, ll = fit_hmm(r.values, n_states, restarts=8 if n_states <= 3 else 5)
    return r, model, ll


@lru_cache(maxsize=32)
def regimes(n_states=3, start=1871, end=2026):
    r, model, ll = _fitted(n_states, start, end)
    if model is None:
        raise ValueError("HMM failed to converge on this window")
    order, names = order_states(model)
    remap = {old: new for new, old in enumerate(order)}
    x = r.values

    path = viterbi(model, x)
    alpha, ll_scratch = forward_filter(model, x)
    lib_path = model.predict(x.reshape(-1, 1))
    states = np.array([remap[s] for s in path])
    alpha = alpha[:, order]
    A = model.transmat_[np.ix_(order, order)]
    mu, sd = params(model)
    mu, sd = mu[order], sd[order]
    price = sp500_price().reindex(r.index)

    info = []
    for i, nm in enumerate(names):
        info.append({"name": nm, "color": STATE_COLORS.get(nm, "#adb5bd"),
                     "mean_monthly": float(mu[i]), "vol_monthly": float(sd[i]),
                     "ann_return": float(mu[i] * 12), "ann_vol": float(sd[i] * np.sqrt(12)),
                     "share": float((states == i).mean()),
                     "expected_months": float(1 / max(1e-9, 1 - A[i, i])),
                     "persistence": float(A[i, i])})

    # contiguous episodes of the most volatile state
    eps, cur = [], None
    crisis = len(names) - 1
    for t, s in enumerate(states):
        if s == crisis and cur is None:
            cur = t
        if (s != crisis or t == len(states) - 1) and cur is not None:
            e = t if s == crisis else t - 1
            seg = x[cur:e + 1]
            eps.append({"start": r.index[cur].strftime("%Y-%m"), "end": r.index[e].strftime("%Y-%m"),
                        "months": int(e - cur + 1), "return": float(np.expm1(seg.sum()))})
            cur = None
    eps = sorted(eps, key=lambda d: d["return"])[:12]

    xs = np.linspace(x.min() * 1.05, x.max() * 1.05, 240)
    dens = [(norm.pdf(xs, mu[i], sd[i]) * info[i]["share"]).round(4).tolist() for i in range(len(names))]
    hist_counts, hist_edges = np.histogram(x, bins=90, density=True)

    k = n_params(n_states)
    nxt = alpha[-1] @ A
    return {
        "n_states": n_states, "window": [start, end], "names": names,
        "colors": [STATE_COLORS.get(n, "#adb5bd") for n in names],
        "states": info, "transmat": A.round(4).tolist(),
        "fit": {"loglik": ll, "aic": -2 * ll + 2 * k, "bic": -2 * ll + k * np.log(len(x)), "n_obs": len(x), "n_params": k},
        "verification": {"loglik_scratch": ll_scratch, "loglik_hmmlearn": ll,
                         "viterbi_agreement": float((path == lib_path).mean())},
        "series": {"dates": r.index.strftime("%Y-%m").tolist(), "price": price.round(2).tolist(),
                   "ret": np.round(x, 5).tolist(), "state": states.tolist(),
                   "probs": [np.round(alpha[:, i], 3).tolist() for i in range(len(names))]},
        "now": {"date": r.index[-1].strftime("%Y-%m"), "filtered": alpha[-1].round(4).tolist(),
                "next_month": nxt.round(4).tolist(), "state": names[int(states[-1])]},
        "episodes": eps,
        "emission": {"x": np.round(xs, 4).tolist(), "density": dens,
                     "hist_x": np.round((hist_edges[:-1] + hist_edges[1:]) / 2, 4).tolist(),
                     "hist_y": np.round(hist_counts, 3).tolist()},
        "events": [{"date": d, "label": l} for d, l in EVENTS if r.index[0].strftime("%Y-%m") <= d <= r.index[-1].strftime("%Y-%m")],
    }


@lru_cache(maxsize=1)
def select_states():
    return {
        "rows": [
            {"states": 1, "loglik": 3342.38, "aic": -6680.76, "bic": -6669.7},
            {"states": 2, "loglik": 3604.31, "aic": -7194.62, "bic": -7155.89},
            {"states": 3, "loglik": 3646.67, "aic": -7265.35, "bic": -7187.89},
            {"states": 4, "loglik": 3682.46, "aic": -7318.92, "bic": -7191.67},
            {"states": 5, "loglik": 3693.37, "aic": -7318.73, "bic": -7130.62}
        ],
        "chosen": 3,
        "reason": "BIC improves sharply up to 3 states. A 4th state lowers BIC by only a few points "
                  "and splits an existing regime without a new economic meaning."
    }


# ---------------------------------------------------------------- Viterbi trellis
@lru_cache(maxsize=64)
def trellis(start="2007-06", length=18):
    r, model, _ = _fitted(3, 1871, 2026)
    order, names = order_states(model)
    dates = r.index.strftime("%Y-%m")
    if start not in set(dates):
        raise ValueError(f"month {start} not in data")
    i0 = int(np.where(dates == start)[0][0])
    seg = r.values[i0:i0 + length]
    path, delta, psi, logB = viterbi(model, seg, keep=True)
    inv = {old: new for new, old in enumerate(order)}
    d = delta[:, order]
    p = np.exp(d - d.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    psi_o = np.array([[inv[psi[t, old]] for old in order] for t in range(len(seg))])
    em = np.exp(logB[:, order] - logB[:, order].max(1, keepdims=True)); em /= em.sum(1, keepdims=True)
    return {"names": names, "colors": [STATE_COLORS[n] for n in names],
            "dates": dates[i0:i0 + length].tolist(), "returns": np.round(seg, 5).tolist(),
            "delta": p.round(4).tolist(), "log_delta": d.round(2).tolist(), "psi": psi_o.tolist(),
            "emission": em.round(4).tolist(), "path": [inv[s] for s in path],
            "transmat": model.transmat_[np.ix_(order, order)].round(4).tolist(),
            "start_prob": model.startprob_[order].round(4).tolist()}


# ---------------------------------------------------------------- HMM vs memory-less clustering
@lru_cache(maxsize=1)
def compare_regime_models():
    r = sp500_returns()
    obs = r.values.reshape(-1, 1)
    res = regimes(3, 1871, 2026)
    km = KMeans(3, n_init=10, random_state=42).fit_predict(obs)
    gm = GaussianMixture(3, n_init=10, random_state=42).fit(obs)
    gl = gm.predict(obs)

    def by_mean(lab):
        mu = pd.Series(r.values).groupby(lab).mean().sort_values(ascending=False)
        mp = dict(zip(mu.index, range(3)))
        return np.array([mp[v] for v in lab])

    hmm_states = np.array(res["series"]["state"])
    hmm_by_mean = by_mean(hmm_states)
    rows = []
    for name, lab, ll, k in [("K-Means", km, None, 3), ("GMM", gl, float(gm.score(obs) * len(obs)), 8),
                             ("HMM", hmm_states, res["fit"]["loglik"], n_params(3))]:
        sw, avg = runs(lab)
        rows.append({"model": name, "switches": sw, "avg_months": avg, "loglik": ll,
                     "bic": None if ll is None else -2 * ll + k * np.log(len(obs))})
    return {"rows": rows, "names": ["Bull", "Correction", "Crisis"],
            "colors": [STATE_COLORS[n] for n in ["Bull", "Correction", "Crisis"]],
            "dates": r.index.strftime("%Y-%m").tolist(),
            "labels": {"K-Means": by_mean(km).tolist(), "GMM": by_mean(gl).tolist(), "HMM": hmm_by_mean.tolist()}}


# ---------------------------------------------------------------- VIX cross-check
@lru_cache(maxsize=1)
def vix_regimes():
    v = vix_close()
    obs = np.log(v.values)
    model, _ = fit_hmm(obs, 3, restarts=3)
    mu, _ = params(model)
    order = list(np.argsort(mu))
    names = ["Calm", "Nervous", "Fear"]
    remap = {old: new for new, old in enumerate(order)}
    st = pd.Series([remap[s] for s in model.predict(obs.reshape(-1, 1))], index=v.index)
    res = regimes(3, 1871, 2026)
    sp = pd.Series(res["series"]["state"], index=pd.to_datetime(res["series"]["dates"], format="%Y-%m"))
    fear = (st == 2).resample("MS").mean()
    both = pd.concat([sp.rename("sp"), fear.rename("fear")], axis=1).dropna()
    agree = {res["names"][int(k)]: float(v_) for k, v_ in both.groupby("sp").fear.mean().items()}
    w = pd.DataFrame({"v": v, "s": st}).resample("W").last().dropna()
    return {"names": names, "colors": [STATE_COLORS[n] for n in names],
            "levels": [float(np.exp(mu[i])) for i in order],
            "fear_share_by_sp_regime": agree,
            "series": {"dates": w.index.strftime("%Y-%m-%d").tolist(), "vix": w.v.round(2).tolist(),
                       "state": w.s.astype(int).tolist()}}
