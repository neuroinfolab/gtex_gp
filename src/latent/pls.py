from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0, ddof=0)
    return np.where(s < 1e-8, 1.0, s)


def canonicalize_signs(T: np.ndarray, U: np.ndarray, P: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = T.copy()
    u = U.copy()
    p = P.copy()
    c = C.copy()
    nc = min(t.shape[1], u.shape[1], p.shape[1], c.shape[1])
    for k in range(nc):
        if np.std(t[:, k]) > 1e-12 and np.std(u[:, k]) > 1e-12:
            rr = np.corrcoef(t[:, k], u[:, k])[0, 1]
            if np.isfinite(rr) and rr < 0:
                t[:, k] *= -1
                u[:, k] *= -1
                p[:, k] *= -1
                c[:, k] *= -1
    return t, u, p, c


def fit_subject_pls(X_obs_h: np.ndarray, Y_obs: np.ndarray, n_comp_target: int = 3, adaptive: bool = True) -> Dict[str, np.ndarray]:
    x_mean = np.nanmean(X_obs_h, axis=0)
    x_scale = safe_std(X_obs_h)
    Xs = (X_obs_h - x_mean) / x_scale
    nc = int(min(n_comp_target, Xs.shape[0] - 1, Xs.shape[1], Y_obs.shape[1]))
    if adaptive:
        rank_x = int(np.linalg.matrix_rank(Xs))
        rank_y = int(np.linalg.matrix_rank(Y_obs))
        nc = int(min(nc, rank_x, rank_y))
    nc = max(nc, 1)

    pls = PLSRegression(n_components=nc, scale=False)
    pls.fit(Xs, Y_obs)
    T = pls.x_scores_.astype(np.float64)
    U = pls.y_scores_.astype(np.float64)
    P = pls.x_loadings_.astype(np.float64)
    C = pls.y_loadings_.astype(np.float64)
    T, U, P, C = canonicalize_signs(T, U, P, C)
    return {
        "T": T,
        "U": U,
        "P": P,
        "C": C,
        "x_mean": x_mean.astype(np.float64),
        "x_scale": x_scale.astype(np.float64),
        "n_comp": np.int32(nc),
    }
