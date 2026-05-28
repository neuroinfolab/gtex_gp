from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0, ddof=0)
    return np.where(s < 1e-8, 1.0, s)


def canonical_signs(T: np.ndarray, U: np.ndarray) -> np.ndarray:
    nc = min(T.shape[1], U.shape[1])
    signs = np.ones(nc, dtype=np.float64)
    for k in range(nc):
        if np.std(T[:, k]) > 1e-12 and np.std(U[:, k]) > 1e-12:
            rr = np.corrcoef(T[:, k], U[:, k])[0, 1]
            if np.isfinite(rr) and rr < 0:
                signs[k] = -1.0
    return signs


def canonicalize_signs(T: np.ndarray, U: np.ndarray, P: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = T.copy()
    u = U.copy()
    p = P.copy()
    c = C.copy()
    signs = canonical_signs(t, u)
    nc = min(len(signs), p.shape[1], c.shape[1])
    t[:, :nc] *= signs[None, :nc]
    u[:, :nc] *= signs[None, :nc]
    p[:, :nc] *= signs[None, :nc]
    c[:, :nc] *= signs[None, :nc]
    return t, u, p, c


def _pls_y_mean(pls: PLSRegression, Y_obs: np.ndarray) -> np.ndarray:
    return np.asarray(getattr(pls, "_y_mean", np.nanmean(Y_obs, axis=0)), dtype=np.float64)


def _pls_y_scale(pls: PLSRegression, Y_obs: np.ndarray) -> np.ndarray:
    scale = np.asarray(getattr(pls, "_y_std", np.ones(Y_obs.shape[1], dtype=np.float64)), dtype=np.float64)
    return np.where(np.abs(scale) < 1e-12, 1.0, scale)


def _pls_y_rotations(pls: PLSRegression) -> np.ndarray:
    return np.asarray(getattr(pls, "y_rotations_", pls.y_weights_), dtype=np.float64)


def fit_pls_basis(
    X_obs_h: np.ndarray,
    Y_obs: np.ndarray,
    n_comp_target: int = 3,
    adaptive: bool = True,
    x_scale: np.ndarray | None = None,
) -> Dict[str, object]:
    """Fit a two-block PLS and return a *reusable projector*.

    Same fit as :func:`fit_subject_pls` (sign-canonicalized two-block PLS, X=genes,
    Y=spatial) but additionally retains the fitted estimator and the per-component
    sign flips, so :func:`project_pls_scores` can map new gene matrices onto the
    identical (frozen) latent axes — the AHBA-basis projection used for the
    "PLS gradient" analysis.

    ``x_scale`` optionally overrides the per-gene standardization (default
    ``safe_std``). Supplying a more stable scale (e.g. floored by an atlas std)
    keeps out-of-sample projections from exploding on low-variance genes; the same
    scale is stored and used for the inverse, so encode/decode stay matched.
    """
    x_mean = np.nanmean(X_obs_h, axis=0)
    x_scale = safe_std(X_obs_h) if x_scale is None else np.asarray(x_scale, dtype=np.float64)
    Xs = (X_obs_h - x_mean) / x_scale
    nc = int(min(n_comp_target, Xs.shape[0] - 1, Xs.shape[1], Y_obs.shape[1]))
    if adaptive:
        nc = int(min(nc, int(np.linalg.matrix_rank(Xs)), int(np.linalg.matrix_rank(Y_obs))))
    nc = max(nc, 1)

    pls = PLSRegression(n_components=nc, scale=False)
    pls.fit(Xs, Y_obs)
    T_raw = pls.x_scores_.astype(np.float64)
    U_raw = pls.y_scores_.astype(np.float64)
    P = pls.x_loadings_.astype(np.float64)
    C = pls.y_loadings_.astype(np.float64)

    # Per-component sign flips matching canonicalize_signs (corr(T, U) >= 0).
    signs = canonical_signs(T_raw, U_raw)
    return {
        "T": T_raw * signs,
        "U": U_raw * signs,
        "P": P * signs,
        "C": C * signs,
        "x_mean": x_mean.astype(np.float64),
        "x_scale": x_scale.astype(np.float64),
        "y_mean": _pls_y_mean(pls, Y_obs),
        "y_scale": _pls_y_scale(pls, Y_obs),
        "y_rotations": _pls_y_rotations(pls) * signs[None, :],
        "signs": signs,
        "n_comp": np.int32(nc),
        "_pls": pls,   # retained for exact forward transform (pickles fine)
    }


def project_pls_scores(basis: Dict[str, object], X_new: np.ndarray) -> np.ndarray:
    """Project new gene matrix ``(n_samples, n_genes)`` onto a fitted PLS basis.

    Returns scores ``(n_samples, n_comp)`` on the same sign-canonicalized latent
    axes as ``basis``. NaN rows in ``X_new`` propagate to NaN score rows.
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    Xs = (X_new - basis["x_mean"]) / basis["x_scale"]
    T = np.asarray(basis["_pls"].transform(Xs), dtype=np.float64)
    return T * basis["signs"][None, :]


def project_pls_y_scores(basis: Dict[str, object], Y_new: np.ndarray) -> np.ndarray:
    """Project spatial covariates onto the fitted PLS Y-score axes.

    This mirrors sklearn's Y-side transform semantics and applies the same
    component sign canonicalization used for ``basis["U"]``.
    """
    Y_new = np.asarray(Y_new, dtype=np.float64)
    Ys = (Y_new - basis["y_mean"]) / basis["y_scale"]
    return Ys @ basis["y_rotations"]


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
    signs = canonical_signs(T, U)
    T, U, P, C = canonicalize_signs(T, U, P, C)
    return {
        "T": T,
        "U": U,
        "P": P,
        "C": C,
        "x_mean": x_mean.astype(np.float64),
        "x_scale": x_scale.astype(np.float64),
        "y_mean": _pls_y_mean(pls, Y_obs),
        "y_scale": _pls_y_scale(pls, Y_obs),
        "y_rotations": _pls_y_rotations(pls) * signs[None, :],
        "n_comp": np.int32(nc),
    }
