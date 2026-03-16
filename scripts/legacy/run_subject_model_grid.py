#!/usr/bin/env python3
"""
Subject-wise GTEx missing-parcel prediction benchmark across a model grid:
- Basis models: o3, affine_gl3, o3_diagscale
- Harmonization: zscore_affine, robustz_affine, quantile_affine, whiten_zca_affine

Outputs under out/o3_subject_model_grid*/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge

import run_o3_subject_level_prediction as base


VALID_BASIS_MODELS = {"o3", "affine_gl3", "o3_diagscale"}
VALID_HARMONIZERS = {"zscore_affine", "robustz_affine", "quantile_affine", "whiten_zca_affine"}
COMPARE_BASIS_POLICY = {"fixed_affine_gl3", "best_per_harmonizer", "both"}
SCATTER_TYPES = [
    "scatter_region_color_combined_raw.png",
    "scatter_gene_color_combined_raw.png",
    "scatter_region_corrcolor_combined_raw.png",
    "scatter_gene_corrcolor_combined_raw.png",
    "scatter_region_color_combined_harmonized.png",
    "scatter_gene_color_combined_harmonized.png",
    "scatter_region_corrcolor_combined_harmonized.png",
    "scatter_gene_corrcolor_combined_harmonized.png",
]


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/o3_subject_model_grid"
    min_observed_parcels: int = 5
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    seed: int = 123
    basis_models: List[str] = None
    harmonization_models: List[str] = None
    top_k_all_genes: int = 2
    all_genes_chunk_size: int = 500
    run_all_genes: bool = True
    n_jobs: int = 1
    percentile_clip_low: float = 2.0
    percentile_clip_high: float = 98.0
    whiten_eps: float = 1e-4
    comparison_prior_harmonizers: List[str] = None
    comparison_basis_policy: str = "both"
    comparison_fixed_basis: str = "affine_gl3"


@dataclass
class BasisMap:
    model: str
    M: np.ndarray
    b: np.ndarray
    invM: np.ndarray
    diagnostics: Dict[str, float]


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run subject-level model-grid benchmark for GTEx missing-parcel prediction.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--basis-models", default="o3,affine_gl3,o3_diagscale")
    p.add_argument("--harmonization-models", default="zscore_affine,robustz_affine,quantile_affine")
    p.add_argument("--top-k-all-genes", type=int, default=Config.top_k_all_genes)
    p.add_argument("--all-genes-chunk-size", type=int, default=Config.all_genes_chunk_size)
    p.add_argument(
        "--run-all-genes",
        type=lambda s: str(s).strip().lower() in {"1", "true", "yes", "y"},
        default=Config.run_all_genes,
    )
    p.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--comparison-prior-harmonizers", default="zscore_affine,robustz_affine")
    p.add_argument("--comparison-basis-policy", default="both")
    p.add_argument("--comparison-fixed-basis", default="affine_gl3")
    p.add_argument("--mitigation-strategies", default="")

    a = p.parse_args()
    basis = _parse_csv_list(a.basis_models)
    harms = _parse_csv_list(a.harmonization_models)
    prior_harms = _parse_csv_list(a.comparison_prior_harmonizers)

    bad_basis = [b for b in basis if b not in VALID_BASIS_MODELS]
    bad_h = [h for h in harms if h not in VALID_HARMONIZERS]
    bad_prior = [h for h in prior_harms if h not in VALID_HARMONIZERS]
    if bad_basis:
        raise RuntimeError(f"Unsupported basis models: {bad_basis}. Valid={sorted(VALID_BASIS_MODELS)}")
    if bad_h:
        raise RuntimeError(f"Unsupported harmonization models: {bad_h}. Valid={sorted(VALID_HARMONIZERS)}")
    if bad_prior:
        raise RuntimeError(f"Unsupported comparison prior harmonizers: {bad_prior}. Valid={sorted(VALID_HARMONIZERS)}")
    if len(basis) == 0 or len(harms) == 0:
        raise RuntimeError("At least one basis model and one harmonization model are required.")
    if str(a.comparison_basis_policy) not in COMPARE_BASIS_POLICY:
        raise RuntimeError(f"Unsupported comparison basis policy: {a.comparison_basis_policy}")
    if str(a.comparison_fixed_basis) not in VALID_BASIS_MODELS:
        raise RuntimeError(f"Unsupported comparison fixed basis: {a.comparison_fixed_basis}")
    if str(a.mitigation_strategies).strip():
        print("[note] --mitigation-strategies is accepted for compatibility; use run_misfit_distance_diagnostics.py for mitigation runs.")

    return Config(
        csv_path=a.csv_path,
        hvg_path=a.hvg_path,
        out_root=a.out_root,
        min_observed_parcels=a.min_observed_parcels,
        n_comp_target=a.n_comp_target,
        ridge_alpha_bridge=a.ridge_alpha_bridge,
        rbf_smoothing=a.rbf_smoothing,
        seed=a.seed,
        basis_models=basis,
        harmonization_models=harms,
        top_k_all_genes=max(1, int(a.top_k_all_genes)),
        all_genes_chunk_size=max(1, int(a.all_genes_chunk_size)),
        run_all_genes=bool(a.run_all_genes),
        n_jobs=max(1, int(a.n_jobs)),
        whiten_eps=max(1e-12, float(a.whiten_eps)),
        comparison_prior_harmonizers=prior_harms,
        comparison_basis_policy=str(a.comparison_basis_policy),
        comparison_fixed_basis=str(a.comparison_fixed_basis),
    )


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0, ddof=0)
    return np.where(s < 1e-8, 1.0, s)


def _safe_mad_scale(x: np.ndarray, med: np.ndarray) -> np.ndarray:
    mad = np.nanmedian(np.abs(x - med[None, :]), axis=0)
    scale = 1.4826 * mad
    return np.where(scale < 1e-8, 1.0, scale)


def _fit_genewise_affine_on_overlap(
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


def _quantile_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    sorted_vals = np.sort(x, axis=0)
    probs = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    zgrid = stats.norm.ppf(np.clip(probs, 1e-8, 1.0 - 1e-8))
    return sorted_vals.astype(np.float64), zgrid.astype(np.float64)


def _quantile_forward(x: np.ndarray, sorted_vals: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for g in range(x.shape[1]):
        out[:, g] = np.interp(x[:, g], sorted_vals[:, g], zgrid, left=zgrid[0], right=zgrid[-1])
    return out


def _quantile_inverse(z: np.ndarray, sorted_vals: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(z, dtype=np.float64)
    for g in range(z.shape[1]):
        out[:, g] = np.interp(z[:, g], zgrid, sorted_vals[:, g], left=sorted_vals[0, g], right=sorted_vals[-1, g])
    return out


def _fit_zca(x: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
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
    cond = float(np.linalg.cond(sigma_reg))
    return {
        "mu": mu.astype(np.float64),
        "W": W.astype(np.float64),
        "Winv": Winv.astype(np.float64),
        "eigvals": eigvals.astype(np.float64),
        "cond_sigma_reg": cond,
        "shrink": shrink,
    }


def _apply_zca(x: np.ndarray, zca: Dict[str, np.ndarray]) -> np.ndarray:
    xc = x - zca["mu"][None, :]
    return xc @ zca["W"]


def _inverse_zca(z: np.ndarray, zca: Dict[str, np.ndarray]) -> np.ndarray:
    return z @ zca["Winv"] + zca["mu"][None, :]


def fit_harmonizer(
    ahba_raw: pd.DataFrame,
    gtex_raw: pd.DataFrame,
    gene_cols: List[str],
    n_parcels: int,
    model: str,
    whiten_eps: float,
) -> Dict[str, np.ndarray]:
    xa = ahba_raw[gene_cols].to_numpy(dtype=np.float64)
    xg = gtex_raw[gene_cols].to_numpy(dtype=np.float64)

    cal: Dict[str, np.ndarray] = {"model": model, "genes": np.asarray(gene_cols, dtype=object)}

    if model == "zscore_affine":
        a_loc = np.nanmean(xa, axis=0)
        a_scale = _safe_std(xa)
        g_loc = np.nanmean(xg, axis=0)
        g_scale = _safe_std(xg)
        ahba_base = (xa - a_loc) / a_scale
        gtex_base = (xg - g_loc) / g_scale
        cal.update(
            {
                "ahba_loc": a_loc,
                "ahba_scale": a_scale,
                "gtex_loc": g_loc,
                "gtex_scale": g_scale,
            }
        )
    elif model == "robustz_affine":
        a_loc = np.nanmedian(xa, axis=0)
        a_scale = _safe_mad_scale(xa, a_loc)
        g_loc = np.nanmedian(xg, axis=0)
        g_scale = _safe_mad_scale(xg, g_loc)
        ahba_base = (xa - a_loc) / a_scale
        gtex_base = (xg - g_loc) / g_scale
        cal.update(
            {
                "ahba_loc": a_loc,
                "ahba_scale": a_scale,
                "gtex_loc": g_loc,
                "gtex_scale": g_scale,
            }
        )
    elif model == "quantile_affine":
        a_sorted, a_zgrid = _quantile_fit(xa)
        g_sorted, g_zgrid = _quantile_fit(xg)
        ahba_base = _quantile_forward(xa, a_sorted, a_zgrid)
        gtex_base = _quantile_forward(xg, g_sorted, g_zgrid)
        cal.update(
            {
                "ahba_sorted": a_sorted,
                "ahba_zgrid": a_zgrid,
                "gtex_sorted": g_sorted,
                "gtex_zgrid": g_zgrid,
            }
        )
    elif model == "whiten_zca_affine":
        az = _fit_zca(xa, eps=whiten_eps)
        gz = _fit_zca(xg, eps=whiten_eps)
        ahba_base = _apply_zca(xa, az)
        gtex_base = _apply_zca(xg, gz)
        cal.update(
            {
                "ahba_zca_mu": az["mu"],
                "ahba_zca_W": az["W"],
                "ahba_zca_Winv": az["Winv"],
                "ahba_zca_eigvals": az["eigvals"],
                "ahba_zca_cond": np.asarray([az["cond_sigma_reg"]], dtype=np.float64),
                "ahba_zca_shrink": np.asarray([az["shrink"]], dtype=np.float64),
                "gtex_zca_mu": gz["mu"],
                "gtex_zca_W": gz["W"],
                "gtex_zca_Winv": gz["Winv"],
                "gtex_zca_eigvals": gz["eigvals"],
                "gtex_zca_cond": np.asarray([gz["cond_sigma_reg"]], dtype=np.float64),
                "gtex_zca_shrink": np.asarray([gz["shrink"]], dtype=np.float64),
                "whiten_eps": np.asarray([whiten_eps], dtype=np.float64),
            }
        )
    else:
        raise RuntimeError(f"Unknown harmonization model: {model}")

    ah_idx = ahba_raw["parcel_idx"].to_numpy(dtype=np.int32)
    gt_idx = gtex_raw["parcel_idx"].to_numpy(dtype=np.int32)
    slope, intercept = _fit_genewise_affine_on_overlap(ahba_base, gtex_base, ah_idx, gt_idx, n_parcels=n_parcels)
    cal["slope"] = slope
    cal["intercept"] = intercept
    return cal


def harmonize_matrix(x_raw: np.ndarray, cal: Dict[str, np.ndarray], dataset_name: str) -> np.ndarray:
    ds = dataset_name.upper()
    if cal["model"] in {"zscore_affine", "robustz_affine"}:
        if ds == "AHBA":
            base_x = (x_raw - cal["ahba_loc"]) / cal["ahba_scale"]
            return base_x
        base_x = (x_raw - cal["gtex_loc"]) / cal["gtex_scale"]
        return base_x * cal["slope"] + cal["intercept"]

    if cal["model"] == "quantile_affine":
        if ds == "AHBA":
            return _quantile_forward(x_raw, cal["ahba_sorted"], cal["ahba_zgrid"])
        base_x = _quantile_forward(x_raw, cal["gtex_sorted"], cal["gtex_zgrid"])
        return base_x * cal["slope"] + cal["intercept"]

    # whiten_zca_affine
    if ds == "AHBA":
        xc = x_raw - cal["ahba_zca_mu"][None, :]
        return xc @ cal["ahba_zca_W"]
    xc = x_raw - cal["gtex_zca_mu"][None, :]
    base_x = xc @ cal["gtex_zca_W"]
    return base_x * cal["slope"] + cal["intercept"]


def inverse_harmonize_gtex_matrix(x_h: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    z = (x_h - cal["intercept"]) / cal["slope"]
    if cal["model"] in {"zscore_affine", "robustz_affine"}:
        return z * cal["gtex_scale"] + cal["gtex_loc"]
    if cal["model"] == "quantile_affine":
        return _quantile_inverse(z, cal["gtex_sorted"], cal["gtex_zgrid"])
    return z @ cal["gtex_zca_Winv"] + cal["gtex_zca_mu"][None, :]


def apply_harmonization_df(df_raw: pd.DataFrame, gene_cols: List[str], cal: Dict[str, np.ndarray], dataset_name: str) -> pd.DataFrame:
    x = df_raw[gene_cols].to_numpy(dtype=np.float64)
    h = harmonize_matrix(x, cal, dataset_name)
    out = df_raw.copy()
    out_genes = pd.DataFrame(h.astype(np.float32), columns=gene_cols, index=out.index)
    out = pd.concat([out.drop(columns=gene_cols), out_genes], axis=1)
    return out


def _stabilize_invertible(M: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    eye = np.eye(M.shape[0], dtype=np.float64)
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best = None
    for lam in lambdas:
        cand = M + lam * eye
        det = float(np.linalg.det(cand))
        cond = float(np.linalg.cond(cand))
        if np.isfinite(det) and np.isfinite(cond) and abs(det) > 1e-8 and cond < 1e8:
            return cand, float(lam), det, cond
        if best is None:
            best = (cand, float(lam), det, cond)
    assert best is not None
    return best


def fit_basis_map(T_sub: np.ndarray, T_ref: np.ndarray, model: str) -> BasisMap:
    k = T_sub.shape[1]
    z = np.zeros(k, dtype=np.float64)

    if model == "o3":
        R = base.fit_subject_o3(T_sub, T_ref).astype(np.float64)
        M = R
        b = z
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(np.linalg.det(M)),
            "orthogonality_error": float(np.linalg.norm(M.T @ M - np.eye(k), ord="fro")),
            "condition_number": float(np.linalg.cond(M)),
            "regularization_lambda": 0.0,
            "basis_rank": float(np.linalg.matrix_rank(M)),
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    if model == "affine_gl3":
        X = np.c_[T_sub, np.ones(T_sub.shape[0], dtype=np.float64)]
        beta, *_ = np.linalg.lstsq(X, T_ref, rcond=None)
        M0 = beta[:k, :].astype(np.float64)
        b = beta[k, :].astype(np.float64)
        M, lam, det, cond = _stabilize_invertible(M0)
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(det),
            "orthogonality_error": float(np.linalg.norm(M.T @ M - np.eye(k), ord="fro")),
            "condition_number": float(cond),
            "regularization_lambda": float(lam),
            "basis_rank": float(np.linalg.matrix_rank(M)),
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    if model == "o3_diagscale":
        R = base.fit_subject_o3(T_sub, T_ref).astype(np.float64)
        Z = T_sub @ R
        d = np.ones(k, dtype=np.float64)
        b = np.zeros(k, dtype=np.float64)
        for j in range(k):
            x = Z[:, j]
            y = T_ref[:, j]
            if np.nanstd(x) > 1e-8:
                s, c = np.polyfit(x, y, deg=1)
                d[j] = max(abs(float(s)), 1e-6)
                b[j] = float(c)
            else:
                d[j] = 1.0
                b[j] = float(np.nanmean(y - x))
        M0 = R @ np.diag(d)
        M, lam, det, cond = _stabilize_invertible(M0)
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(det),
            "orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(k), ord="fro")),
            "condition_number": float(cond),
            "regularization_lambda": float(lam),
            "basis_rank": float(np.linalg.matrix_rank(M)),
            "diag_scale_1": float(d[0]) if k >= 1 else np.nan,
            "diag_scale_2": float(d[1]) if k >= 2 else np.nan,
            "diag_scale_3": float(d[2]) if k >= 3 else np.nan,
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    raise RuntimeError(f"Unknown basis model: {model}")


def apply_basis_scores(T: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return T @ basis_map.M + basis_map.b[None, :]


def apply_basis_linear(U: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return U @ basis_map.M


def invert_basis_scores(Tp: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return (Tp - basis_map.b[None, :]) @ basis_map.invM


def combo_name(harmonization: str, basis_model: str) -> str:
    return f"{harmonization}__{basis_model}"


def split_combo(combo: str) -> Tuple[str, str]:
    parts = str(combo).split("__", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Invalid combo name: {combo}")
    return parts[0], parts[1]


def predict_subject_full(
    coords_obs: np.ndarray,
    X_obs_h: np.ndarray,
    X_obs_raw: np.ndarray,
    obs_idx: np.ndarray,
    coords_full: np.ndarray,
    pls_bundle: Dict[str, np.ndarray],
    basis_map: BasisMap,
    calibration: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    T_prime_obs = apply_basis_scores(pls_bundle["T"], basis_map)
    U_prime_obs = apply_basis_linear(pls_bundle["U"], basis_map)

    spatial_model = base.fit_spatial_field(coords_obs, U_prime_obs, cfg)
    bridge = base.fit_u_to_t_bridge(U_prime_obs, T_prime_obs, alpha=cfg.ridge_alpha_bridge)

    U_full_prime = base._predict_spatial_field(spatial_model, coords_full)
    T_full_prime = U_full_prime @ bridge["M"] + bridge["b"]
    T_full_native = invert_basis_scores(T_full_prime, basis_map)

    X_std_full = T_full_native @ pls_bundle["P"].T
    X_full_h = X_std_full * pls_bundle["x_scale"] + pls_bundle["x_mean"]
    X_full_raw = inverse_harmonize_gtex_matrix(X_full_h, calibration)

    X_full_h = X_full_h.astype(np.float64)
    X_full_raw = X_full_raw.astype(np.float64)
    X_full_h[obs_idx, :] = X_obs_h
    X_full_raw[obs_idx, :] = X_obs_raw

    diag = {
        "T_obs_prime": T_prime_obs.astype(np.float64),
        "U_obs_prime": U_prime_obs.astype(np.float64),
        "T_full_prime": T_full_prime.astype(np.float64),
        "U_full_prime": U_full_prime.astype(np.float64),
        "bridge_M": bridge["M"].astype(np.float64),
        "bridge_b": bridge["b"].astype(np.float64),
    }
    return X_full_h, X_full_raw, diag


def evaluate_subject_loro(
    subject: str,
    obs_idx: np.ndarray,
    X_obs_h: np.ndarray,
    coords_full: np.ndarray,
    ahba_T_ref: np.ndarray,
    ahba_h_full: np.ndarray,
    basis_model: str,
    harmonization: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    fold_rows: List[Dict[str, float]] = []
    true_all = []
    pred_all = []
    base_all = []
    skipped = 0

    idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}
    for hold in obs_idx.tolist():
        hold = int(hold)
        train_idx = np.asarray([p for p in obs_idx.tolist() if int(p) != hold], dtype=np.int32)
        if len(train_idx) < 3:
            skipped += 1
            continue

        train_pos = [idx_to_pos[int(p)] for p in train_idx.tolist()]
        hold_pos = idx_to_pos[hold]
        X_train_h = X_obs_h[train_pos, :]
        Y_train = np.c_[coords_full[train_idx, 1], coords_full[train_idx, 2], np.abs(coords_full[train_idx, 0])]

        try:
            pls_train = base.fit_subject_pls(X_train_h, Y_train, n_comp_target=cfg.n_comp_target, adaptive=True)
            k = int(pls_train["n_comp"])
            T_ref_train = ahba_T_ref[train_idx, :k]
            bmap = fit_basis_map(pls_train["T"], T_ref_train, model=basis_model)

            T_train_prime = apply_basis_scores(pls_train["T"], bmap)
            U_train_prime = apply_basis_linear(pls_train["U"], bmap)
            spatial = base.fit_spatial_field(coords_full[train_idx, :], U_train_prime, cfg)
            bridge = base.fit_u_to_t_bridge(U_train_prime, T_train_prime, alpha=cfg.ridge_alpha_bridge)

            U_hold = base._predict_spatial_field(spatial, coords_full[[hold], :])
            T_hold_prime = U_hold @ bridge["M"] + bridge["b"]
            T_hold_native = invert_basis_scores(T_hold_prime, bmap)
            X_std_hold = T_hold_native @ pls_train["P"].T
            X_hold_h = X_std_hold * pls_train["x_scale"] + pls_train["x_mean"]

            x_true = X_obs_h[hold_pos, :].astype(np.float64)
            x_pred = X_hold_h[0, :].astype(np.float64)
            x_base = ahba_h_full[hold, :].astype(np.float64)
            met = base._metrics_from_vectors(x_true, x_pred)
            fold_rows.append(
                {
                    "subject": subject,
                    "model_combo": combo_name(harmonization, basis_model),
                    "harmonization": harmonization,
                    "basis_model": basis_model,
                    "split": "loro_fold",
                    "fold_holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
                    "n_comp": int(k),
                    **met,
                    "baseline_rmse": float(np.sqrt(np.mean((x_true - x_base) ** 2))),
                    "status": "ok",
                }
            )
            true_all.append(x_true)
            pred_all.append(x_pred)
            base_all.append(x_base)
        except Exception as e:
            skipped += 1
            fold_rows.append(
                {
                    "subject": subject,
                    "model_combo": combo_name(harmonization, basis_model),
                    "harmonization": harmonization,
                    "basis_model": basis_model,
                    "split": "loro_fold",
                    "fold_holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
                    "n_comp": np.nan,
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "medae": np.nan,
                    "baseline_rmse": np.nan,
                    "status": f"failed:{str(e)[:120]}",
                }
            )

    folds_df = pd.DataFrame(fold_rows)
    if len(true_all) == 0:
        summary = {
            "subject": subject,
            "model_combo": combo_name(harmonization, basis_model),
            "harmonization": harmonization,
            "basis_model": basis_model,
            "split": "loro_summary",
            "n_obs_parcels": int(len(obs_idx)),
            "n_folds": 0,
            "n_skipped_folds": int(skipped),
            "n_points": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
            "baseline_rmse": np.nan,
            "better_than_baseline_rmse": False,
        }
        return folds_df, summary

    yt = np.concatenate(true_all)
    yp = np.concatenate(pred_all)
    yb = np.concatenate(base_all)
    met = base._metrics_from_vectors(yt, yp)
    baseline_rmse = float(np.sqrt(np.mean((yt - yb) ** 2)))
    summary = {
        "subject": subject,
        "model_combo": combo_name(harmonization, basis_model),
        "harmonization": harmonization,
        "basis_model": basis_model,
        "split": "loro_summary",
        "n_obs_parcels": int(len(obs_idx)),
        "n_folds": int(len(true_all)),
        "n_skipped_folds": int(skipped),
        "n_points": int(len(yt)),
        **met,
        "baseline_rmse": baseline_rmse,
        "better_than_baseline_rmse": bool(np.isfinite(met["rmse"]) and met["rmse"] < baseline_rmse),
    }
    return folds_df, summary


def build_subject_eligibility(gtex_raw: pd.DataFrame, min_observed_parcels: int) -> pd.DataFrame:
    rows = []
    subjects = sorted(gtex_raw["subject"].dropna().astype(str).unique().tolist())
    for sid in subjects:
        sub = gtex_raw[gtex_raw["subject"] == sid].copy()
        n_samples = int(len(sub))
        n_obs = int(sub["parcel_idx"].nunique())
        eligible = bool(n_obs >= min_observed_parcels)
        rows.append(
            {
                "subject": sid,
                "n_samples": n_samples,
                "n_obs_parcels": n_obs,
                "eligible": eligible,
                "reason": "eligible" if eligible else f"insufficient_observed_parcels<{min_observed_parcels}",
            }
        )
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def make_model_grid_figures(summary_df: pd.DataFrame, ranking_df: pd.DataFrame, fig_dir: Path, incumbent_combo: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if len(summary_df) == 0:
        return

    stat_df = ranking_df.copy()
    if len(stat_df) > 0:
        piv_p = stat_df.pivot(index="harmonization", columns="basis_model", values="mean_pearson")
        piv_r = stat_df.pivot(index="harmonization", columns="basis_model", values="mean_rmse")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        sns.heatmap(piv_p, annot=True, fmt=".3f", cmap="viridis", ax=axes[0])
        axes[0].set_title("Mean LORO Pearson")
        sns.heatmap(piv_r, annot=True, fmt=".3f", cmap="magma_r", ax=axes[1])
        axes[1].set_title("Mean LORO RMSE")
        fig.savefig(fig_dir / "model_grid_metric_heatmap.png", dpi=220)
        plt.close(fig)

    plot_df = summary_df[np.isfinite(summary_df["pearson_r"])].copy()
    if len(plot_df) > 0:
        order = ranking_df["model_combo"].tolist() if len(ranking_df) else sorted(plot_df["model_combo"].unique().tolist())
        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        sns.violinplot(data=plot_df, x="model_combo", y="pearson_r", order=order, ax=ax, inner="quartile", cut=0)
        ax.set_title("Subject-level LORO Pearson by model")
        ax.set_xlabel("Model combo")
        ax.set_ylabel("Pearson")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(fig_dir / "model_grid_pearson_violin.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        sns.violinplot(data=plot_df, x="model_combo", y="rmse", order=order, ax=ax, inner="quartile", cut=0)
        ax.set_title("Subject-level LORO RMSE by model")
        ax.set_xlabel("Model combo")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(fig_dir / "model_grid_rmse_violin.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        sns.scatterplot(data=plot_df, x="n_obs_parcels", y="pearson_r", hue="model_combo", alpha=0.65, s=28, ax=ax)
        ax.set_title("Coverage vs LORO Pearson by model")
        ax.set_xlabel("Observed parcels per subject")
        ax.set_ylabel("LORO Pearson")
        ax.grid(True, alpha=0.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
        fig.savefig(fig_dir / "coverage_vs_performance_by_model.png", dpi=220)
        plt.close(fig)

        inc = plot_df[plot_df["model_combo"] == incumbent_combo][["subject", "pearson_r", "rmse"]].copy()
        inc = inc.rename(columns={"pearson_r": "inc_pearson", "rmse": "inc_rmse"})
        delta = plot_df[plot_df["model_combo"] != incumbent_combo].merge(inc, on="subject", how="inner")
        if len(delta) > 0:
            delta["delta_pearson"] = delta["pearson_r"] - delta["inc_pearson"]
            delta["delta_rmse"] = delta["rmse"] - delta["inc_rmse"]
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            sns.scatterplot(data=delta, x="delta_rmse", y="delta_pearson", hue="model_combo", s=30, alpha=0.7, ax=ax)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.set_title("Delta vs incumbent (zscore_affine__o3)")
            ax.set_xlabel("RMSE delta (model - incumbent)")
            ax.set_ylabel("Pearson delta (model - incumbent)")
            ax.grid(True, alpha=0.2)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
            fig.savefig(fig_dir / "delta_vs_incumbent_scatter.png", dpi=220)
            plt.close(fig)


def make_threeway_comparison_figures(
    run_root: Path,
    selection_df: pd.DataFrame,
    policy_name: str,
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    req_harms = ["zscore_affine", "robustz_affine", "whiten_zca_affine"]

    pick = selection_df[selection_df["policy"] == policy_name].copy()
    pick = pick[pick["harmonization"].isin(req_harms)].copy()
    if len(pick) < 3:
        return out_paths

    order = [h for h in req_harms if h in set(pick["harmonization"].tolist())]
    if len(order) < 3:
        return out_paths
    combos = [pick[pick["harmonization"] == h].iloc[0]["model_combo"] for h in order]

    for st in SCATTER_TYPES:
        img_paths = []
        for cmb in combos:
            p = run_root / "figures" / "aggregate" / str(cmb) / "scatter_panels" / st
            if not p.exists():
                img_paths = []
                break
            img_paths.append(p)
        if len(img_paths) != 3:
            continue

        ims = [mpimg.imread(p) for p in img_paths]
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.imshow(ims[i])
            ax.axis("off")
            ax.set_title(str(combos[i]))
        out_path = out_dir / f"compare_3way__{st}"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        out_paths.append(str(out_path))

    return out_paths


def run_combo_all_genes(
    combo: str,
    harmonization: str,
    subject_models: Dict[str, Dict[str, np.ndarray]],
    ahba_all: pd.DataFrame,
    gtex_all: pd.DataFrame,
    gtex_all_raw_agg: pd.DataFrame,
    genes_all: List[str],
    target: pd.DataFrame,
    cfg: Config,
    table_dir: Path,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    if len(subject_models) == 0:
        return {"status": "skipped_no_subject_models", "model_combo": combo}, pd.DataFrame()

    agg_meta = gtex_all_raw_agg[["subject", "parcel_idx"]].copy().reset_index(drop=True)
    subject_row_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, idxs in agg_meta.groupby("subject").indices.items():
        rows = np.asarray(list(idxs), dtype=np.int32)
        parcels = agg_meta.iloc[rows]["parcel_idx"].to_numpy(dtype=np.int32)
        order = np.argsort(parcels)
        subject_row_map[str(sid)] = (rows[order], parcels[order])

    eligible_subjects = sorted(subject_models.keys())
    n_sub = len(eligible_subjects)
    n_parcels = len(target)
    n_genes = len(genes_all)
    chunk = max(1, int(cfg.all_genes_chunk_size))

    agg_mean_h = np.zeros((n_parcels, n_genes), dtype=np.float32)
    agg_median_h = np.zeros((n_parcels, n_genes), dtype=np.float32)
    agg_mean_raw = np.zeros((n_parcels, n_genes), dtype=np.float32)
    agg_median_raw = np.zeros((n_parcels, n_genes), dtype=np.float32)

    diag_acc = {
        sid: {
            "n_points": 0,
            "sum_true": 0.0,
            "sum_pred": 0.0,
            "sum_true2": 0.0,
            "sum_pred2": 0.0,
            "sum_true_pred": 0.0,
            "sse": 0.0,
            "sae": 0.0,
        }
        for sid in eligible_subjects
    }

    for start in range(0, n_genes, chunk):
        end = min(n_genes, start + chunk)
        chunk_genes = genes_all[start:end]
        csize = end - start
        cal_chunk = fit_harmonizer(ahba_all, gtex_all, chunk_genes, n_parcels=len(target), model=harmonization, whiten_eps=cfg.whiten_eps)
        chunk_raw = gtex_all_raw_agg[chunk_genes].to_numpy(dtype=np.float64)
        chunk_h = harmonize_matrix(chunk_raw, cal_chunk, "GTEX")

        stack_h = np.full((n_sub, n_parcels, csize), np.nan, dtype=np.float32)
        stack_raw = np.full((n_sub, n_parcels, csize), np.nan, dtype=np.float32)

        for si, sid in enumerate(eligible_subjects):
            model = subject_models[sid]
            if sid not in subject_row_map:
                continue
            row_idx_sorted, parcel_sorted = subject_row_map[sid]
            sub_raw_sorted = chunk_raw[row_idx_sorted, :]
            sub_h_sorted = chunk_h[row_idx_sorted, :]
            parcel_pos = {int(p): i for i, p in enumerate(parcel_sorted.tolist())}
            obs_idx = model["obs_idx"].astype(np.int32)
            if not all(int(p) in parcel_pos for p in obs_idx.tolist()):
                continue

            pos = np.asarray([parcel_pos[int(p)] for p in obs_idx.tolist()], dtype=np.int32)
            true_obs_h = sub_h_sorted[pos, :]
            true_obs_raw = sub_raw_sorted[pos, :]

            T_obs_prime = model["T_obs_prime"]
            T_full_prime = model["T_full_prime"]
            dec = Ridge(alpha=float(cfg.ridge_alpha_bridge), fit_intercept=True)
            dec.fit(T_obs_prime, true_obs_h)
            pred_full_h = dec.predict(T_full_prime)
            pred_obs_h = pred_full_h[obs_idx, :]

            acc = diag_acc[sid]
            t = true_obs_h.ravel()
            p = pred_obs_h.ravel()
            acc["n_points"] += int(t.size)
            acc["sum_true"] += float(np.sum(t))
            acc["sum_pred"] += float(np.sum(p))
            acc["sum_true2"] += float(np.sum(t * t))
            acc["sum_pred2"] += float(np.sum(p * p))
            acc["sum_true_pred"] += float(np.sum(t * p))
            acc["sse"] += float(np.sum((t - p) ** 2))
            acc["sae"] += float(np.sum(np.abs(t - p)))

            pred_full_h[obs_idx, :] = true_obs_h
            pred_full_raw = inverse_harmonize_gtex_matrix(pred_full_h, cal_chunk)
            pred_full_raw[obs_idx, :] = true_obs_raw

            stack_h[si, :, :] = pred_full_h.astype(np.float32)
            stack_raw[si, :, :] = pred_full_raw.astype(np.float32)

        agg_mean_h[:, start:end] = np.nanmean(stack_h, axis=0).astype(np.float32)
        agg_median_h[:, start:end] = np.nanmedian(stack_h, axis=0).astype(np.float32)
        agg_mean_raw[:, start:end] = np.nanmean(stack_raw, axis=0).astype(np.float32)
        agg_median_raw[:, start:end] = np.nanmedian(stack_raw, axis=0).astype(np.float32)

    diag_rows = []
    for sid in eligible_subjects:
        a = diag_acc[sid]
        n = max(int(a["n_points"]), 1)
        mean_t = a["sum_true"] / n
        mean_p = a["sum_pred"] / n
        var_t = (a["sum_true2"] / n) - mean_t**2
        var_p = (a["sum_pred2"] / n) - mean_p**2
        cov = (a["sum_true_pred"] / n) - (mean_t * mean_p)
        pear = np.nan
        if var_t > 1e-12 and var_p > 1e-12:
            pear = float(cov / np.sqrt(var_t * var_p))
        rmse = float(np.sqrt(a["sse"] / n))
        mae = float(a["sae"] / n)
        diag_rows.append(
            {
                "model_combo": combo,
                "subject": sid,
                "n_points": int(a["n_points"]),
                "pearson_r": pear,
                "rmse": rmse,
                "mae": mae,
            }
        )

    parcel_idx = target["parcel_idx"].to_numpy(dtype=np.int32)
    parcel_name = target["tissue_or_parcel"].astype(str).to_numpy()
    genes_arr = np.asarray(genes_all, dtype=object)
    np.savez_compressed(
        table_dir / f"aggregate_allgenes_{combo}_mean_raw.npz",
        matrix=agg_mean_raw,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / f"aggregate_allgenes_{combo}_mean_harmonized.npz",
        matrix=agg_mean_h,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / f"aggregate_allgenes_{combo}_median_raw.npz",
        matrix=agg_median_raw,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / f"aggregate_allgenes_{combo}_median_harmonized.npz",
        matrix=agg_median_h,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )

    return {
        "status": "ok",
        "model_combo": combo,
        "n_subjects": int(n_sub),
        "n_genes": int(n_genes),
        "chunk_size": int(chunk),
    }, pd.DataFrame(diag_rows)


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    if cfg.n_jobs != 1:
        print(f"[note] n_jobs={cfg.n_jobs} requested; current implementation runs serially (n_jobs=1).")

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header = base.load_gene_header_and_hvg(csv_path, hvg_path)
    genes_hvg = header["genes_hvg"]
    genes_all = header["genes_all"]
    if len(genes_hvg) == 0:
        raise RuntimeError("No HVG overlap found; cannot run pipeline.")

    print("[stage] Loading HVG subset and building shared objects")
    df_hvg = base.read_expression_subset(csv_path, genes_hvg)
    ahba_raw = df_hvg[df_hvg["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df_hvg[df_hvg["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEx split is empty.")

    target = base.build_target_parcels(ahba_raw)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = base.map_gtex_to_target(gtex_raw, target)

    coords_full = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    eligibility_df = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    eligibility_df.to_csv(table_dir / "subject_eligibility.csv", index=False)
    eligible_subjects = eligibility_df[eligibility_df["eligible"]]["subject"].astype(str).tolist()
    if len(eligible_subjects) == 0:
        raise RuntimeError("No eligible subjects passed min_observed_parcels gate.")

    _, global_obs_mask = base.build_region_matrix(gtex_raw, genes_hvg, target, agg="mean")
    meta_df = pd.DataFrame(
        {
            "parcel_idx": target["parcel_idx"].to_numpy(dtype=np.int32),
            "parcel_name": target["tissue_or_parcel"].astype(str).to_numpy(),
            "coord_x": target["coord_x"].to_numpy(dtype=np.float64),
            "coord_y": target["coord_y"].to_numpy(dtype=np.float64),
            "coord_z": target["coord_z"].to_numpy(dtype=np.float64),
            "is_observed_gtex": global_obs_mask.astype(bool),
        }
    )
    aligned_index_path = table_dir / "aligned_row_index.csv"
    meta_df.to_csv(aligned_index_path, index=False)

    transform_rows: List[Dict[str, float]] = []
    harm_diag_rows: List[Dict[str, float]] = []
    whitening_diag_rows: List[Dict[str, float]] = []
    fold_rows_all: List[pd.DataFrame] = []
    summary_rows_all: List[Dict[str, float]] = []

    combo_subject_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    print(
        f"[stage] Running model grid: {len(cfg.harmonization_models)} harmonizers x {len(cfg.basis_models)} basis models"
    )

    for hm in cfg.harmonization_models:
        print(f"  [harm] {hm}")
        cal = fit_harmonizer(
            ahba_raw,
            gtex_raw,
            genes_hvg,
            n_parcels=len(target),
            model=hm,
            whiten_eps=cfg.whiten_eps,
        )
        ahba_h = apply_harmonization_df(ahba_raw, genes_hvg, cal, "AHBA")
        gtex_h = apply_harmonization_df(gtex_raw, genes_hvg, cal, "GTEX")

        gtex_x_raw = gtex_raw[genes_hvg].to_numpy(dtype=np.float64)
        gtex_x_h = harmonize_matrix(gtex_x_raw, cal, "GTEX")
        gtex_x_raw_roundtrip = inverse_harmonize_gtex_matrix(gtex_x_h, cal)
        roundtrip_rmse = float(np.sqrt(np.mean((gtex_x_raw - gtex_x_raw_roundtrip) ** 2)))
        roundtrip_mae = float(np.mean(np.abs(gtex_x_raw - gtex_x_raw_roundtrip)))
        harm_diag_rows.append(
            {
                "harmonization": hm,
                "n_genes": int(len(genes_hvg)),
                "roundtrip_rmse_gtex_hvg": roundtrip_rmse,
                "roundtrip_mae_gtex_hvg": roundtrip_mae,
                "slope_mean": float(np.nanmean(cal["slope"])),
                "slope_std": float(np.nanstd(cal["slope"])),
                "intercept_mean": float(np.nanmean(cal["intercept"])),
                "intercept_std": float(np.nanstd(cal["intercept"])),
            }
        )

        if hm == "whiten_zca_affine":
            w = cal["gtex_zca_W"]
            winv = cal["gtex_zca_Winv"]
            ident_err = float(np.linalg.norm(w @ winv - np.eye(w.shape[0]), ord="fro"))
            whitening_diag_rows.append(
                {
                    "harmonization": hm,
                    "whiten_eps": float(cal.get("whiten_eps", np.asarray([cfg.whiten_eps]))[0]),
                    "gtex_cond_sigma_reg": float(cal["gtex_zca_cond"][0]),
                    "ahba_cond_sigma_reg": float(cal["ahba_zca_cond"][0]),
                    "gtex_min_eig": float(np.min(cal["gtex_zca_eigvals"])),
                    "gtex_max_eig": float(np.max(cal["gtex_zca_eigvals"])),
                    "ahba_min_eig": float(np.min(cal["ahba_zca_eigvals"])),
                    "ahba_max_eig": float(np.max(cal["ahba_zca_eigvals"])),
                    "gtex_shrink": float(cal["gtex_zca_shrink"][0]),
                    "ahba_shrink": float(cal["ahba_zca_shrink"][0]),
                    "W_Winv_identity_error": ident_err,
                    "roundtrip_rmse_gtex_hvg": roundtrip_rmse,
                    "roundtrip_mae_gtex_hvg": roundtrip_mae,
                }
            )

        ahba_h_full, _ = base.build_region_matrix(ahba_h, genes_hvg, target, agg="mean")
        ahba_raw_full, _ = base.build_region_matrix(ahba_raw, genes_hvg, target, agg="mean")
        ahba_ref_pls = base.fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
        ahba_ref_T = ahba_ref_pls["T"]

        for bm in cfg.basis_models:
            cmb = combo_name(hm, bm)
            print(f"    [combo] {cmb}")
            pred_h_list = []
            pred_raw_list = []
            pred_subject_ids = []
            subject_models: Dict[str, Dict[str, np.ndarray]] = {}
            obs_data_by_subject: Dict[str, Dict[str, np.ndarray]] = {}

            for i, sid in enumerate(eligible_subjects):
                subj_h = gtex_h[gtex_h["subject"] == sid].copy()
                subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
                obs_idx, x_obs_h, x_obs_raw = base.build_subject_observed_matrices(subj_h, subj_raw, genes_hvg)
                n_obs = int(len(obs_idx))
                if n_obs < cfg.min_observed_parcels:
                    continue

                try:
                    y_obs = np.c_[coords_full[obs_idx, 1], coords_full[obs_idx, 2], np.abs(coords_full[obs_idx, 0])]
                    pls_sub = base.fit_subject_pls(x_obs_h, y_obs, n_comp_target=cfg.n_comp_target, adaptive=True)
                    k = int(pls_sub["n_comp"])
                    t_ref_obs = ahba_ref_T[obs_idx, :k]
                    bmap = fit_basis_map(pls_sub["T"], t_ref_obs, model=bm)

                    x_full_h, x_full_raw, pred_diag = predict_subject_full(
                        coords_obs=coords_full[obs_idx, :],
                        X_obs_h=x_obs_h,
                        X_obs_raw=x_obs_raw,
                        obs_idx=obs_idx,
                        coords_full=coords_full,
                        pls_bundle=pls_sub,
                        basis_map=bmap,
                        calibration=cal,
                        cfg=cfg,
                    )

                    pre_m = base.evaluate_alignment_matrix(t_ref_obs, pls_sub["T"])
                    post_m = base.evaluate_alignment_matrix(t_ref_obs, pred_diag["T_obs_prime"])

                    trow = {
                        "subject": sid,
                        "model_combo": cmb,
                        "harmonization": hm,
                        "basis_model": bm,
                        "n_obs_parcels": n_obs,
                        "n_comp": k,
                        "pre_global_r2": pre_m.get("global_r2", np.nan),
                        "pre_mean_pearson": pre_m.get("mean_pearson", np.nan),
                        "pre_rmse": pre_m.get("rmse", np.nan),
                        "post_global_r2": post_m.get("global_r2", np.nan),
                        "post_mean_pearson": post_m.get("mean_pearson", np.nan),
                        "post_rmse": post_m.get("rmse", np.nan),
                        "status": "ok",
                    }
                    trow.update(bmap.diagnostics)
                    transform_rows.append(trow)

                    subject_models[sid] = {
                        "obs_idx": obs_idx.astype(np.int32),
                        "T_obs_prime": pred_diag["T_obs_prime"].astype(np.float64),
                        "T_full_prime": pred_diag["T_full_prime"].astype(np.float64),
                        "n_comp": np.int32(k),
                    }
                    obs_data_by_subject[sid] = {
                        "obs_idx": obs_idx.astype(np.int32),
                        "X_obs_h": x_obs_h.astype(np.float64),
                    }

                    pred_subject_ids.append(sid)
                    pred_h_list.append(x_full_h.astype(np.float32))
                    pred_raw_list.append(x_full_raw.astype(np.float32))
                except Exception as e:
                    transform_rows.append(
                        {
                            "subject": sid,
                            "model_combo": cmb,
                            "harmonization": hm,
                            "basis_model": bm,
                            "n_obs_parcels": n_obs,
                            "n_comp": np.nan,
                            "pre_global_r2": np.nan,
                            "pre_mean_pearson": np.nan,
                            "pre_rmse": np.nan,
                            "post_global_r2": np.nan,
                            "post_mean_pearson": np.nan,
                            "post_rmse": np.nan,
                            "det_linear": np.nan,
                            "orthogonality_error": np.nan,
                            "condition_number": np.nan,
                            "regularization_lambda": np.nan,
                            "basis_rank": np.nan,
                            "status": f"failed:{str(e)[:120]}",
                        }
                    )

                if (i + 1) % 50 == 0:
                    print(f"      processed {i+1}/{len(eligible_subjects)} subjects")

            if len(pred_subject_ids) == 0:
                print(f"      no successful subjects for {cmb}, skipping")
                continue

            pred_h_tensor = np.stack(pred_h_list, axis=0).astype(np.float32)
            pred_raw_tensor = np.stack(pred_raw_list, axis=0).astype(np.float32)
            combo_subject_models[cmb] = subject_models

            for sid in pred_subject_ids:
                obs_d = obs_data_by_subject[sid]
                folds_df, summ = evaluate_subject_loro(
                    subject=sid,
                    obs_idx=obs_d["obs_idx"],
                    X_obs_h=obs_d["X_obs_h"],
                    coords_full=coords_full,
                    ahba_T_ref=ahba_ref_T,
                    ahba_h_full=ahba_h_full,
                    basis_model=bm,
                    harmonization=hm,
                    cfg=cfg,
                )
                if len(folds_df) > 0:
                    fold_rows_all.append(folds_df)
                summary_rows_all.append(summ)

            agg_mean_h = np.nanmean(pred_h_tensor, axis=0).astype(np.float32)
            agg_median_h = np.nanmedian(pred_h_tensor, axis=0).astype(np.float32)
            agg_mean_raw = np.nanmean(pred_raw_tensor, axis=0).astype(np.float32)
            agg_median_raw = np.nanmedian(pred_raw_tensor, axis=0).astype(np.float32)

            base.matrix_to_aligned_table(agg_mean_raw, genes_hvg, meta_df).to_csv(
                table_dir / f"aggregate_hvg_{cmb}_mean_raw.csv", index=False
            )
            base.matrix_to_aligned_table(agg_mean_h, genes_hvg, meta_df).to_csv(
                table_dir / f"aggregate_hvg_{cmb}_mean_harmonized.csv", index=False
            )
            base.matrix_to_aligned_table(agg_median_raw, genes_hvg, meta_df).to_csv(
                table_dir / f"aggregate_hvg_{cmb}_median_raw.csv", index=False
            )
            base.matrix_to_aligned_table(agg_median_h, genes_hvg, meta_df).to_csv(
                table_dir / f"aggregate_hvg_{cmb}_median_harmonized.csv", index=False
            )

            combo_fig_dir = fig_dir / "aggregate" / cmb
            combo_tab_dir = table_dir / "aggregate" / cmb
            combo_fig_dir.mkdir(parents=True, exist_ok=True)
            combo_tab_dir.mkdir(parents=True, exist_ok=True)

            base.make_side_by_side_heatmap(
                left_mat=agg_mean_raw,
                right_mat=ahba_raw_full,
                title_left=f"GTEx predicted mean ({cmb}, raw)",
                title_right="AHBA reference (raw)",
                suptitle=f"GTEx vs AHBA side-by-side heatmap [{cmb}]",
                out_path=combo_fig_dir / "side_by_side_heatmap_raw.png",
                cfg=cfg,
            )
            base.make_side_by_side_heatmap(
                left_mat=agg_mean_h,
                right_mat=ahba_h_full,
                title_left=f"GTEx predicted mean ({cmb}, harmonized)",
                title_right="AHBA reference (harmonized)",
                suptitle=f"GTEx vs AHBA side-by-side heatmap [{cmb}]",
                out_path=combo_fig_dir / "side_by_side_heatmap_harmonized.png",
                cfg=cfg,
            )

            g_raw_path = combo_tab_dir / "gtex_predicted_allen_rows_raw.csv"
            g_h_path = combo_tab_dir / "gtex_predicted_allen_rows_harmonized.csv"
            a_raw_path = combo_tab_dir / "allen_allen_rows_raw.csv"
            a_h_path = combo_tab_dir / "allen_allen_rows_harmonized.csv"
            idx_path = combo_tab_dir / "aligned_row_index.csv"
            base.matrix_to_aligned_table(agg_mean_raw, genes_hvg, meta_df).to_csv(g_raw_path, index=False)
            base.matrix_to_aligned_table(agg_mean_h, genes_hvg, meta_df).to_csv(g_h_path, index=False)
            base.matrix_to_aligned_table(ahba_raw_full, genes_hvg, meta_df).to_csv(a_raw_path, index=False)
            base.matrix_to_aligned_table(ahba_h_full, genes_hvg, meta_df).to_csv(a_h_path, index=False)
            meta_df.to_csv(idx_path, index=False)

            base.run_scatter_suite(
                root=root,
                gtex_raw_path=g_raw_path,
                allen_raw_path=a_raw_path,
                gtex_harm_path=g_h_path,
                allen_harm_path=a_h_path,
                aligned_index_path=idx_path,
                out_fig_dir=combo_fig_dir / "scatter_panels",
                out_tab_dir=combo_tab_dir / "scatter_panels",
            )

    transform_df = pd.DataFrame(transform_rows)
    harm_diag_df = pd.DataFrame(harm_diag_rows)
    whitening_diag_df = pd.DataFrame(whitening_diag_rows)
    folds_all_df = pd.concat(fold_rows_all, ignore_index=True) if len(fold_rows_all) else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows_all)

    transform_df.to_csv(table_dir / "transform_diagnostics.csv", index=False)
    harm_diag_df.to_csv(table_dir / "harmonization_diagnostics.csv", index=False)
    whitening_diag_df.to_csv(table_dir / "whitening_diagnostics.csv", index=False)
    folds_all_df.to_csv(table_dir / "subject_loro_folds_hvg.csv", index=False)
    summary_df.to_csv(table_dir / "subject_loro_summary_hvg.csv", index=False)

    if len(summary_df) == 0:
        raise RuntimeError("No LORO summary rows were generated.")

    grp_cols = ["model_combo", "harmonization", "basis_model"]
    msum = (
        summary_df.groupby(grp_cols, as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_pearson=("pearson_r", "mean"),
            median_pearson=("pearson_r", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_baseline_rmse=("baseline_rmse", "mean"),
            frac_better_naive_rmse=("better_than_baseline_rmse", "mean"),
        )
        .sort_values("mean_pearson", ascending=False)
        .reset_index(drop=True)
    )
    msum.to_csv(table_dir / "model_grid_summary_hvg.csv", index=False)

    incumbent_combo = combo_name("zscore_affine", "o3")
    inc = summary_df[summary_df["model_combo"] == incumbent_combo][["subject", "pearson_r", "rmse"]].copy()
    inc = inc.rename(columns={"pearson_r": "inc_pearson", "rmse": "inc_rmse"})

    b_rows = []
    for _, row in msum.iterrows():
        cmb = str(row["model_combo"])
        sdf = summary_df[summary_df["model_combo"] == cmb].copy()
        m = sdf.merge(inc, on="subject", how="inner")
        frac_better_inc_rmse = float(np.mean((m["rmse"] < m["inc_rmse"]).to_numpy(dtype=bool))) if len(m) else np.nan
        dpear = float(np.nanmean(m["pearson_r"] - m["inc_pearson"])) if len(m) else np.nan
        drmse = float(np.nanmean(m["rmse"] - m["inc_rmse"])) if len(m) else np.nan
        b_rows.append(
            {
                "model_combo": cmb,
                "harmonization": str(row["harmonization"]),
                "basis_model": str(row["basis_model"]),
                "n_subjects": int(row["n_subjects"]),
                "mean_pearson": float(row["mean_pearson"]),
                "mean_rmse": float(row["mean_rmse"]),
                "mean_baseline_rmse": float(row["mean_baseline_rmse"]),
                "frac_better_naive_rmse": float(row["frac_better_naive_rmse"]),
                "frac_better_incumbent_rmse": frac_better_inc_rmse,
                "delta_mean_pearson_vs_incumbent": dpear,
                "delta_mean_rmse_vs_incumbent": drmse,
            }
        )
    model_vs_base_df = pd.DataFrame(b_rows)
    model_vs_base_df.to_csv(table_dir / "model_vs_baseline.csv", index=False)

    rank_df = model_vs_base_df.sort_values(
        ["mean_pearson", "mean_rmse", "frac_better_naive_rmse"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=np.int32)
    rank_df.to_csv(table_dir / "model_ranking.csv", index=False)

    make_model_grid_figures(summary_df, rank_df, fig_dir, incumbent_combo=incumbent_combo)

    sel = rank_df.head(min(cfg.top_k_all_genes, len(rank_df))).copy()
    sel.to_csv(table_dir / "selected_models_for_all_genes.csv", index=False)

    allgenes_rows = []
    allgenes_diag_rows = []
    if cfg.run_all_genes and len(sel) > 0:
        print("[stage] Running all-genes aggregate transport for selected models")
        df_all = base.read_expression_subset(csv_path, genes_all)
        ahba_all = df_all[df_all["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
        gtex_all = df_all[df_all["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
        ahba_all["parcel_idx"] = ahba_all["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
        gtex_all = base.map_gtex_to_target(gtex_all, target)
        gtex_all_raw_agg = gtex_all.groupby(["subject", "parcel_idx"], sort=False)[genes_all].mean().reset_index()
        gtex_all_raw_agg["subject"] = gtex_all_raw_agg["subject"].astype(str)

        for _, srow in sel.iterrows():
            cmb = str(srow["model_combo"])
            hm = str(srow["harmonization"])
            if cmb not in combo_subject_models:
                allgenes_rows.append({"model_combo": cmb, "status": "skipped_missing_subject_models"})
                continue
            print(f"  [all-genes] {cmb}")
            res, diag_df = run_combo_all_genes(
                combo=cmb,
                harmonization=hm,
                subject_models=combo_subject_models[cmb],
                ahba_all=ahba_all,
                gtex_all=gtex_all,
                gtex_all_raw_agg=gtex_all_raw_agg,
                genes_all=genes_all,
                target=target,
                cfg=cfg,
                table_dir=table_dir,
            )
            allgenes_rows.append(res)
            if len(diag_df):
                allgenes_diag_rows.append(diag_df)

    allgenes_diag_df = pd.concat(allgenes_diag_rows, ignore_index=True) if len(allgenes_diag_rows) else pd.DataFrame()
    allgenes_diag_df.to_csv(table_dir / "allgenes_subject_diagnostics.csv", index=False)

    # Build comparison selections and figures (whitening vs prior two harmonizers).
    comp_rows = []
    required_harms = list(dict.fromkeys(cfg.comparison_prior_harmonizers + ["whiten_zca_affine"]))

    if cfg.comparison_basis_policy in {"fixed_affine_gl3", "both"}:
        for h in required_harms:
            cmb = combo_name(h, cfg.comparison_fixed_basis)
            rr = rank_df[rank_df["model_combo"] == cmb]
            if len(rr) == 0:
                continue
            r = rr.iloc[0]
            comp_rows.append(
                {
                    "policy": "fixed_affine_gl3",
                    "harmonization": h,
                    "basis_model": cfg.comparison_fixed_basis,
                    "model_combo": cmb,
                    "rank": int(r["rank"]),
                    "mean_pearson": float(r["mean_pearson"]),
                    "mean_rmse": float(r["mean_rmse"]),
                }
            )

    if cfg.comparison_basis_policy in {"best_per_harmonizer", "both"}:
        for h in required_harms:
            sub = rank_df[rank_df["harmonization"] == h].sort_values("rank")
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            comp_rows.append(
                {
                    "policy": "best_per_harmonizer",
                    "harmonization": h,
                    "basis_model": str(r["basis_model"]),
                    "model_combo": str(r["model_combo"]),
                    "rank": int(r["rank"]),
                    "mean_pearson": float(r["mean_pearson"]),
                    "mean_rmse": float(r["mean_rmse"]),
                }
            )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(table_dir / "comparison_harmonizer_selection.csv", index=False)

    compare_fig_paths = []
    compare_root = fig_dir / "comparisons" / "whitening_vs_prev"
    if len(comp_df) > 0:
        if cfg.comparison_basis_policy in {"fixed_affine_gl3", "both"}:
            compare_fig_paths.extend(
                make_threeway_comparison_figures(
                    run_root=out_root,
                    selection_df=comp_df,
                    policy_name="fixed_affine_gl3",
                    out_dir=compare_root / "fixed_affine_gl3",
                )
            )
        if cfg.comparison_basis_policy in {"best_per_harmonizer", "both"}:
            compare_fig_paths.extend(
                make_threeway_comparison_figures(
                    run_root=out_root,
                    selection_df=comp_df,
                    policy_name="best_per_harmonizer",
                    out_dir=compare_root / "best_per_harmonizer",
                )
            )

    overall = {
        "n_subjects_total": int(eligibility_df["subject"].nunique()),
        "n_subjects_eligible": int(eligibility_df["eligible"].sum()),
        "n_model_combos_requested": int(len(cfg.harmonization_models) * len(cfg.basis_models)),
        "n_model_combos_completed": int(summary_df["model_combo"].nunique()),
        "incumbent_combo": incumbent_combo,
        "best_combo": str(rank_df.iloc[0]["model_combo"]) if len(rank_df) else None,
        "best_mean_pearson": float(rank_df.iloc[0]["mean_pearson"]) if len(rank_df) else np.nan,
        "best_mean_rmse": float(rank_df.iloc[0]["mean_rmse"]) if len(rank_df) else np.nan,
    }

    summary = {
        "config": asdict(cfg),
        "data_summary": {
            "ahba_rows_hvg": int(len(ahba_raw)),
            "gtex_rows_hvg": int(len(gtex_raw)),
            "n_target_parcels": int(len(target)),
            "n_hvg_requested": int(len(header["hvg_requested"])),
            "n_hvg_matched": int(len(genes_hvg)),
            "n_hvg_missing": int(len(header["missing_hvg"])),
        },
        "overall": overall,
        "all_genes_stage": allgenes_rows,
        "comparison": {
            "harmonizers": required_harms,
            "basis_policy": cfg.comparison_basis_policy,
            "fixed_basis": cfg.comparison_fixed_basis,
            "n_comparison_rows": int(len(comp_df)),
            "n_comparison_figures": int(len(compare_fig_paths)),
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = f"""# Subject-Level Model Grid Report

## Configuration
- Basis models: {', '.join(cfg.basis_models)}
- Harmonization models: {', '.join(cfg.harmonization_models)}
- Eligible-subject gate: >= {cfg.min_observed_parcels} observed mapped parcels
- Seed: {cfg.seed}
- Whitening epsilon: {cfg.whiten_eps}

## Coverage
- GTEx subjects total: {overall['n_subjects_total']}
- Eligible GTEx subjects: {overall['n_subjects_eligible']}
- Target parcels: {summary['data_summary']['n_target_parcels']}
- Matched HVGs: {summary['data_summary']['n_hvg_matched']}

## Model Grid Outcome
- Requested combos: {overall['n_model_combos_requested']}
- Completed combos: {overall['n_model_combos_completed']}
- Incumbent combo: {overall['incumbent_combo']}
- Best combo: {overall['best_combo']}
- Best mean Pearson: {overall['best_mean_pearson']:.4f}
- Best mean RMSE: {overall['best_mean_rmse']:.4f}

## Whitening vs Prior-2 Comparison
- Prior harmonizers: {', '.join(cfg.comparison_prior_harmonizers)}
- Added harmonizer: whiten_zca_affine
- Comparison policy: {cfg.comparison_basis_policy}
- Fixed basis (if used): {cfg.comparison_fixed_basis}
- Selection table: `tables/comparison_harmonizer_selection.csv`
- Comparison figures root: `figures/comparisons/whitening_vs_prev/`

## Key Tables
- `tables/model_grid_summary_hvg.csv`
- `tables/model_vs_baseline.csv`
- `tables/model_ranking.csv`
- `tables/harmonization_diagnostics.csv`
- `tables/whitening_diagnostics.csv`

## Figures
- `figures/model_grid_metric_heatmap.png`
- `figures/model_grid_pearson_violin.png`
- `figures/model_grid_rmse_violin.png`
- `figures/coverage_vs_performance_by_model.png`
- `figures/delta_vs_incumbent_scatter.png`

## All-Genes Stage
- Selected top-K: {cfg.top_k_all_genes}
- Run flag: {cfg.run_all_genes}
- Diagnostics table: `tables/allgenes_subject_diagnostics.csv`
"""
    (out_root / "report.md").write_text(report)

    print("Completed subject-level model-grid benchmark.")
    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
