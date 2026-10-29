"""Out-of-sample regime backtest, and the clustering x HMM fusion analysis."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .clustering import run_clustering
from .data import daily_returns, market_daily, sp500_returns
from .regimes import fit_hmm, forward_filter, order_states, params


@lru_cache(maxsize=32)
def _train(train_end):
    r = sp500_returns()
    train = r[r.index.year <= train_end]
    model, _ = fit_hmm(train.values, 3)
    order, names = order_states(model)
    return model, order, names


def _stats(x, pos=None):
    eq = np.exp(x.cumsum())
    yrs = len(x) / 12
    dd = eq / eq.cummax() - 1
    sd = x.std()
    out = {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(sd * np.sqrt(12)),
           "sharpe": float(x.mean() / sd * np.sqrt(12)) if sd > 0 else None,
           "max_drawdown": float(dd.min()), "final": float(eq.iloc[-1]),
           "worst_month": float(np.expm1(x.min())), "best_month": float(np.expm1(x.max()))}
    if pos is not None:
        out["time_invested"] = float(pos.mean())
        out["trades"] = int((pos.diff().abs() > 0).sum())
    return out


@lru_cache(maxsize=256)
def backtest(train_end=1989, rule="bull", threshold=0.5, cost_bps=0.0, cash_rate=0.0):
    r = sp500_returns()
    model, order, names = _train(train_end)
    alpha, _ = forward_filter(model, r.values)          # causal: alpha[t] uses r[0..t] only
    alpha = pd.DataFrame(alpha[:, order], index=r.index, columns=names)
    test = r[r.index.year > train_end]

    if rule == "bull":
        signal = (alpha["Bull"] >= threshold).astype(float)
        rule_text = f"Hold the S&P 500 while P(Bull) ≥ {threshold:.2f}, otherwise cash"
    else:
        signal = (alpha["Crisis"] < threshold).astype(float)
        rule_text = f"Hold the S&P 500 unless P(Crisis) ≥ {threshold:.2f}, then cash"
    pos = signal.shift(1).fillna(1.0).reindex(test.index)        # decide at t, earn at t+1
    cash = np.log1p(cash_rate / 12)
    turnover = pos.diff().abs().fillna(0)
    strat = pos * test + (1 - pos) * cash - turnover * cost_bps / 1e4

    eq_bh, eq_st = np.exp(test.cumsum()), np.exp(strat.cumsum())
    mu, sd = params(model)
    return {
        "train_end": train_end, "rule": rule, "rule_text": rule_text, "threshold": threshold,
        "cost_bps": cost_bps, "cash_rate": cash_rate,
        "train_window": [int(r.index.year.min()), train_end], "test_window": [train_end + 1, int(r.index.year.max())],
        "train_states": [{"name": n, "ann_return": float(mu[i] * 12), "ann_vol": float(sd[i] * np.sqrt(12))}
                         for n, i in zip(names, order)],
        "stats": {"buy_hold": _stats(test), "strategy": _stats(strat, pos)},
        "series": {"dates": test.index.strftime("%Y-%m").tolist(),
                   "buy_hold": eq_bh.round(4).tolist(), "strategy": eq_st.round(4).tolist(),
                   "dd_buy_hold": (eq_bh / eq_bh.cummax() - 1).round(4).tolist(),
                   "dd_strategy": (eq_st / eq_st.cummax() - 1).round(4).tolist(),
                   "position": pos.round(0).astype(int).tolist(),
                   "p_bull": alpha["Bull"].reindex(test.index).round(3).tolist(),
                   "p_crisis": alpha["Crisis"].reindex(test.index).round(3).tolist()},
    }


@lru_cache(maxsize=1)
def _daily_regime():
    m = market_daily()
    model, _ = fit_hmm(m.values, 2, restarts=5)
    _, sd = params(model)
    calm = int(np.argmin(sd))
    st = pd.Series(np.where(model.predict(m.values.reshape(-1, 1)) == calm, "Calm", "Stressed"), index=m.index)
    return st


@lru_cache(maxsize=32)
def fusion(algo="kmeans", k=4):
    res = run_clustering(algo, k)
    rets = daily_returns()
    reg = _daily_regime()
    lab = pd.Series(res["points"]["cluster"], index=res["points"]["ticker"])
    cl = rets.T.groupby(lab).mean().T
    rows = []
    for name in res["order"]:
        if name not in cl:
            continue
        row = {"cluster": name, "color": res["colors"][name], "n": int((lab == name).sum())}
        for g in ["Calm", "Stressed"]:
            x = cl.loc[reg == g, name]
            row[g.lower()] = {"ann_return": float(x.mean() * 252), "ann_vol": float(x.std() * np.sqrt(252))}
        rows.append(row)
    best_stress = max(rows, key=lambda r_: r_["stressed"]["ann_return"])["cluster"]
    best_calm = max(rows, key=lambda r_: r_["calm"]["ann_return"])["cluster"]
    mkt = np.exp(market_daily().cumsum())
    return {"algo": algo, "k": k, "rows": rows, "stressed_share": float((reg == "Stressed").mean()),
            "insight": f"In calm markets {best_calm} leads; when the HMM flags stress, {best_stress} loses least.",
            "market": {"dates": mkt.index.strftime("%Y-%m-%d").tolist(), "index": mkt.round(4).tolist(),
                       "stressed": (reg == "Stressed").astype(int).tolist()}}
