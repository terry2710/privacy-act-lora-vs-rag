from math import comb

import numpy as np


def bootstrap(stat, arrays: list[np.ndarray], resamples: int, seed: int, paired: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    estimate = float(stat(*arrays))
    draws = np.empty(resamples)
    for i in range(resamples):
        if paired:
            idx = rng.integers(0, len(arrays[0]), len(arrays[0]))
            draws[i] = stat(*(a[idx] for a in arrays))
        else:
            draws[i] = stat(*(a[rng.integers(0, len(a), len(a))] for a in arrays))
    low, high = np.percentile(draws, [2.5, 97.5])
    below = (np.sum(draws <= 0) + 1) / (resamples + 1)
    above = (np.sum(draws >= 0) + 1) / (resamples + 1)
    return {
        "estimate": round(estimate, 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "p_two_sided": round(float(min(1.0, 2 * min(below, above))), 5),
    }


def mean_diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(x) - np.mean(y))


def median_diff(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(x) - np.median(y))


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"x_only": b, "y_only": c, "p_two_sided": float(p)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted
