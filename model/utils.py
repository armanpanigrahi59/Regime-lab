"""Small helpers shared by the model modules."""
import math
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def clean(obj):
    """Make any nested result JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def r(x, nd=4):
    """Round a float (or array) for compact payloads."""
    if isinstance(x, (np.ndarray, list)):
        return np.round(np.asarray(x, dtype=float), nd).tolist()
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


@contextmanager
def timer(out: dict, key: str = "ms"):
    t = time.perf_counter()
    yield
    out[key] = round((time.perf_counter() - t) * 1000, 1)
