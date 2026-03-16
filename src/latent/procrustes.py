from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_o3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    c = A.T @ B
    u, _, vt = np.linalg.svd(c)
    r = u @ vt
    return r.astype(np.float64)


def stabilize_invertible(M: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    eye = np.eye(M.shape[0], dtype=np.float64)
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best = None
    for lam in lambdas:
        cand = M + lam * eye
        det = float(np.linalg.det(cand))
        cond = float(np.linalg.cond(cand))
        if np.isfinite(det) and np.isfinite(cond) and abs(det) > 1e-8 and cond < 1e8:
            return cand.astype(np.float64), float(lam), det, cond
        if best is None:
            best = (cand, float(lam), det, cond)
    assert best is not None
    cand, lam, det, cond = best
    return cand.astype(np.float64), float(lam), float(det), float(cond)
