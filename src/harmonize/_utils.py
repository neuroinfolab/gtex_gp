from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0, ddof=0)
    return np.where(s < 1e-8, 1.0, s)


def safe_mad_scale(x: np.ndarray, med: np.ndarray) -> np.ndarray:
    mad = np.nanmedian(np.abs(x - med[None, :]), axis=0)
    scale = 1.4826 * mad
    return np.where(scale < 1e-8, 1.0, scale)


def fit_genewise_affine_on_overlap(
    ahba_base: np.ndarray,
    gtex_base: np.ndarray,
    ahba_idx: np.ndarray,
    gtex_idx: np.ndarray,
    n_parcels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_genes = ahba_base.shape[1]
    ap = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    gp = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    for r in range(n_parcels):
        ma = ahba_idx == r
        mg = gtex_idx == r
        if np.any(ma):
            ap[r, :] = np.nanmean(ahba_base[ma, :], axis=0)
        if np.any(mg):
            gp[r, :] = np.nanmean(gtex_base[mg, :], axis=0)

    slope = np.ones(n_genes, dtype=np.float64)
    intercept = np.zeros(n_genes, dtype=np.float64)
    for g in range(n_genes):
        x = gp[:, g]
        y = ap[:, g]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2 and np.nanstd(x[m]) > 1e-8:
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
            slope[g] = np.clip(float(b1), -5.0, 5.0)
            intercept[g] = float(b0)
        elif m.sum() >= 1:
            slope[g] = 1.0
            intercept[g] = float(np.nanmean(y[m] - x[m]))
    slope = np.where(np.abs(slope) < 1e-6, 1e-6, slope)
    return slope, intercept


def quantile_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    sorted_vals = np.sort(x, axis=0)
    probs = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    zgrid = stats.norm.ppf(np.clip(probs, 1e-8, 1.0 - 1e-8))
    return sorted_vals.astype(np.float64), zgrid.astype(np.float64)


def quantile_forward(x: np.ndarray, sorted_vals: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for g in range(x.shape[1]):
        out[:, g] = np.interp(x[:, g], sorted_vals[:, g], zgrid, left=zgrid[0], right=zgrid[-1])
    return out


def quantile_inverse(z: np.ndarray, sorted_vals: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(z, dtype=np.float64)
    for g in range(z.shape[1]):
        out[:, g] = np.interp(z[:, g], zgrid, sorted_vals[:, g], left=sorted_vals[0, g], right=sorted_vals[-1, g])
    return out


def fit_zca(x: np.ndarray, eps: float) -> dict:
    mu = np.nanmean(x, axis=0)
    xc = x - mu[None, :]
    n, g = xc.shape
    denom = max(n - 1, 1)
    sigma = (xc.T @ xc) / float(denom)
    tr = float(np.trace(sigma))
    shrink = float(eps * (tr / g if g > 0 else 1.0))
    sigma_reg = sigma + shrink * np.eye(g, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(sigma_reg)
    eigvals = np.clip(eigvals.astype(np.float64), 1e-12, None)
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    sqrt = np.diag(np.sqrt(eigvals))
    W = eigvecs @ inv_sqrt @ eigvecs.T
    Winv = eigvecs @ sqrt @ eigvecs.T
    return {
        "mu": mu.astype(np.float64),
        "W": W.astype(np.float64),
        "Winv": Winv.astype(np.float64),
        "eigvals": eigvals.astype(np.float64),
        "cond_sigma_reg": float(np.linalg.cond(sigma_reg)),
        "shrink": shrink,
    }
