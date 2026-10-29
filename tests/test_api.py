"""End-to-end checks: every endpoint answers, and the from-scratch algorithms match the libraries.

Run with:  pytest -q
"""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def get(path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, (path, r.text[:200])
    return r.json()


def test_health_and_frontend():
    assert get("/api/health")["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/script.js").status_code == 200


def test_kmeans_scratch_matches_sklearn():
    d = get("/api/cluster", algo="kmeans", k=4)
    v = d["extras"]["verification"]
    assert v["ari_vs_sklearn"] > 0.999
    assert abs(v["inertia_scratch"] - v["inertia_sklearn"]) < 1e-3
    assert d["order"] == ["Defensive", "Growth Leaders", "High-β Cyclical", "Event-Shock"]
    assert sum(c["n"] for c in d["clusters"]) == 477


def test_other_algorithms():
    for algo in ["gmm", "agglomerative", "dbscan"]:
        d = get("/api/cluster", algo=algo, k=4)
        assert len(d["points"]["ticker"]) == 477


def test_hmm_scratch_matches_hmmlearn():
    d = get("/api/hmm", n_states=3)
    v = d["verification"]
    assert abs(v["loglik_scratch"] - v["loglik_hmmlearn"]) < 1e-6
    assert v["viterbi_agreement"] == 1.0
    assert d["names"] == ["Bull", "Correction", "Crisis"]


def test_lab_and_viterbi():
    assert len(get("/api/lab/kmeans", k=3, dataset="blobs")["frames"]) > 2
    t = get("/api/hmm/viterbi", start="2007-06", length=18)
    assert len(t["path"]) == 18 and t["path"][-1] == 2   # ends in Crisis (Oct-Nov 2008)


def test_backtest_and_fusion():
    b = get("/api/backtest", train_end=1989, rule="bull", threshold=0.5)
    assert b["stats"]["strategy"]["max_drawdown"] > b["stats"]["buy_hold"]["max_drawdown"]
    f = get("/api/fusion", algo="kmeans", k=4)
    assert len(f["rows"]) == 4


def test_stocks_and_errors():
    assert get("/api/stocks/KO")["cluster"] == "Defensive"
    assert client.get("/api/stocks/NOPE").status_code == 404
    assert client.get("/api/hmm", params={"start": 2000, "end": 2026}).status_code == 400
