#!/usr/bin/env python3
"""
Subject-level GTEx O(3)-aligned extrapolation with aggregate atlas outputs,
individual representative visualizations, and per-subject LORO evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import RBFInterpolator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")
META_COLS = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/o3_subject_level"
    min_observed_parcels: int = 5
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    seed: int = 123
    representative_count: int = 9
    all_genes_chunk_size: int = 500
    run_all_genes: bool = True
    percentile_clip_low: float = 2.0
    percentile_clip_high: float = 98.0


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run subject-level GTEx O(3)-aligned extrapolation.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--representative-count", type=int, default=Config.representative_count)
    p.add_argument("--all-genes-chunk-size", type=int, default=Config.all_genes_chunk_size)
    p.add_argument(
        "--run-all-genes",
        type=lambda s: str(s).strip().lower() in {"1", "true", "yes", "y"},
        default=Config.run_all_genes,
    )
    a = p.parse_args()
    return Config(
        csv_path=a.csv_path,
        hvg_path=a.hvg_path,
        out_root=a.out_root,
        min_observed_parcels=a.min_observed_parcels,
        n_comp_target=a.n_comp_target,
        ridge_alpha_bridge=a.ridge_alpha_bridge,
        rbf_smoothing=a.rbf_smoothing,
        seed=a.seed,
        representative_count=a.representative_count,
        all_genes_chunk_size=a.all_genes_chunk_size,
        run_all_genes=a.run_all_genes,
    )


def normalize_gene_name(name: str) -> str:
    return str(name).upper().replace("-", "_").replace(".", "_")


def parse_coordinate_centroid(coord_text: str) -> Tuple[float, float, float]:
    toks = COORD_PATTERN.findall(str(coord_text))
    if not toks:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray([[float(a), float(b), float(c)] for a, b, c in toks], dtype=np.float64)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0, ddof=0)
    return np.where(s < 1e-8, 1.0, s)


def load_gene_header_and_hvg(csv_path: Path, hvg_path: Path) -> Dict[str, List[str]]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    genes_all = header[6:]
    hvg_requested = [ln.strip() for ln in hvg_path.read_text().splitlines() if ln.strip()]
    norm_map = {normalize_gene_name(g): g for g in genes_all}
    genes_hvg = [norm_map[normalize_gene_name(g)] for g in hvg_requested if normalize_gene_name(g) in norm_map]
    missing_hvg = [g for g in hvg_requested if normalize_gene_name(g) not in norm_map]
    return {
        "genes_all": genes_all,
        "hvg_requested": hvg_requested,
        "genes_hvg": genes_hvg,
        "missing_hvg": missing_hvg,
    }


def read_expression_subset(csv_path: Path, gene_cols: List[str]) -> pd.DataFrame:
    usecols = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"] + list(gene_cols)
    dtype_map = {
        "subject": "string",
        "age": "string",
        "sex": "string",
        "dataset": "string",
        "tissue_or_parcel": "string",
        "coordinates": "string",
    }
    dtype_map.update({g: np.float32 for g in gene_cols})
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, low_memory=False)
    xyz = np.vstack([parse_coordinate_centroid(c) for c in df["coordinates"]])
    df["coord_x"] = xyz[:, 0]
    df["coord_y"] = xyz[:, 1]
    df["coord_z"] = xyz[:, 2]
    df["coord_abs_x"] = np.abs(df["coord_x"])
    df = df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    df["subject"] = df["subject"].astype(str)
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    return df


def build_target_parcels(ahba_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        ahba_df.groupby("tissue_or_parcel")[["coord_x", "coord_y", "coord_z"]]
        .mean()
        .reset_index()
        .sort_values("tissue_or_parcel")
        .reset_index(drop=True)
    )
    grp["parcel_idx"] = np.arange(len(grp), dtype=np.int32)
    return grp[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]]


def map_gtex_to_target(gtex_df: pd.DataFrame, target_parcels: pd.DataFrame) -> pd.DataFrame:
    out = gtex_df.copy()
    txyz = target_parcels[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = out[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((gxyz[:, None, :] - txyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    out["parcel_idx"] = idx.astype(np.int32)
    out["mapped_parcel"] = target_parcels.loc[idx, "tissue_or_parcel"].to_numpy()
    out["mapping_distance"] = np.sqrt(d2[np.arange(len(out)), idx]).astype(np.float32)
    return out


def fit_domain_calibration_fast(
    ahba_raw: pd.DataFrame,
    gtex_raw: pd.DataFrame,
    gene_cols: List[str],
    n_parcels: int,
) -> Dict[str, np.ndarray]:
    xa = ahba_raw[gene_cols].to_numpy(dtype=np.float64)
    xg = gtex_raw[gene_cols].to_numpy(dtype=np.float64)
    am = np.nanmean(xa, axis=0)
    asd = _safe_std(xa)
    gm = np.nanmean(xg, axis=0)
    gsd = _safe_std(xg)

    ahz = (xa - am) / asd
    gtz = (xg - gm) / gsd

    ah_idx = ahba_raw["parcel_idx"].to_numpy(dtype=np.int32)
    gt_idx = gtex_raw["parcel_idx"].to_numpy(dtype=np.int32)
    ap = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    gp = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    for r in range(n_parcels):
        ma = ah_idx == r
        mg = gt_idx == r
        if np.any(ma):
            ap[r, :] = np.nanmean(ahz[ma, :], axis=0)
        if np.any(mg):
            gp[r, :] = np.nanmean(gtz[mg, :], axis=0)

    slope = np.ones(len(gene_cols), dtype=np.float64)
    intercept = np.zeros(len(gene_cols), dtype=np.float64)
    for g in range(len(gene_cols)):
        x = gp[:, g]
        y = ap[:, g]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2 and np.nanstd(x[m]) > 1e-8:
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
            slope[g] = np.clip(b1, -5.0, 5.0)
            intercept[g] = b0
        elif m.sum() >= 1:
            slope[g] = 1.0
            intercept[g] = float(np.nanmean(y[m] - x[m]))
    slope = np.where(np.abs(slope) < 1e-6, 1e-6, slope)

    return {
        "genes": np.asarray(gene_cols, dtype=object),
        "ahba_mean": am,
        "ahba_std": asd,
        "gtex_mean": gm,
        "gtex_std": gsd,
        "slope": slope,
        "intercept": intercept,
    }


def apply_harmonization(df_raw: pd.DataFrame, gene_cols: List[str], cal: Dict[str, np.ndarray], dataset_name: str) -> pd.DataFrame:
    x = df_raw[gene_cols].to_numpy(dtype=np.float64)
    if dataset_name.upper() == "AHBA":
        h = (x - cal["ahba_mean"]) / cal["ahba_std"]
    else:
        z = (x - cal["gtex_mean"]) / cal["gtex_std"]
        h = z * cal["slope"] + cal["intercept"]
    out = df_raw.copy()
    out_genes = pd.DataFrame(h.astype(np.float32), columns=gene_cols, index=out.index)
    out = pd.concat([out.drop(columns=gene_cols), out_genes], axis=1)
    return out


def inverse_harmonize_gtex(xh: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    z = (xh - cal["intercept"]) / cal["slope"]
    return z * cal["gtex_std"] + cal["gtex_mean"]


def inverse_harmonize_gtex_chunk(
    xh_chunk: np.ndarray,
    gm: np.ndarray,
    gsd: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    z = (xh_chunk - intercept) / slope
    return z * gsd + gm


def build_region_matrix(
    df: pd.DataFrame,
    gene_cols: List[str],
    target_parcels: pd.DataFrame,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    r = len(target_parcels)
    out = np.full((r, len(gene_cols)), np.nan, dtype=np.float64)
    obs = np.zeros(r, dtype=bool)
    for i in range(r):
        sub = df[df["parcel_idx"] == i]
        if len(sub) == 0:
            continue
        x = sub[gene_cols].to_numpy(dtype=np.float64)
        out[i, :] = np.nanmean(x, axis=0) if agg == "mean" else np.nanmedian(x, axis=0)
        obs[i] = True
    return out, obs


def build_subject_observed_matrices(
    subj_h: pd.DataFrame,
    subj_raw: pd.DataFrame,
    gene_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grp_h = subj_h.groupby("parcel_idx")[gene_cols].mean()
    grp_raw = subj_raw.groupby("parcel_idx")[gene_cols].mean()
    common = sorted(set(grp_h.index.tolist()) & set(grp_raw.index.tolist()))
    if len(common) == 0:
        return np.array([], dtype=np.int32), np.zeros((0, len(gene_cols))), np.zeros((0, len(gene_cols)))
    obs_idx = np.asarray(common, dtype=np.int32)
    xh = grp_h.loc[common, gene_cols].to_numpy(dtype=np.float64)
    xr = grp_raw.loc[common, gene_cols].to_numpy(dtype=np.float64)
    return obs_idx, xh, xr


def _canonicalize_pls_signs(
    T: np.ndarray,
    U: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def fit_subject_pls(
    X_obs_h: np.ndarray,
    Y_obs: np.ndarray,
    n_comp_target: int,
    adaptive: bool = True,
) -> Dict[str, np.ndarray]:
    x_mean = np.nanmean(X_obs_h, axis=0)
    x_scale = _safe_std(X_obs_h)
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
    T, U, P, C = _canonicalize_pls_signs(T, U, P, C)

    return {
        "T": T,
        "U": U,
        "P": P,
        "C": C,
        "x_mean": x_mean.astype(np.float64),
        "x_scale": x_scale.astype(np.float64),
        "n_comp": np.int32(nc),
    }


def fit_subject_o3(T_sub_obs: np.ndarray, T_ahba_obs: np.ndarray) -> np.ndarray:
    c = T_sub_obs.T @ T_ahba_obs
    u, _, vt = np.linalg.svd(c)
    r = u @ vt
    return r


def apply_pls_consistent_transform(pls_bundle: Dict[str, np.ndarray], R: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(pls_bundle)
    out["T_prime"] = pls_bundle["T"] @ R
    out["U_prime"] = pls_bundle["U"] @ R
    out["P_prime"] = pls_bundle["P"] @ R
    out["C_prime"] = pls_bundle["C"] @ R
    out["R"] = R
    return out


def fit_spatial_field(coords_obs: np.ndarray, U_prime_obs: np.ndarray, cfg: Config) -> Dict[str, object]:
    models = []
    for k in range(U_prime_obs.shape[1]):
        try:
            rbf = RBFInterpolator(
                coords_obs,
                U_prime_obs[:, k],
                kernel="thin_plate_spline",
                smoothing=float(cfg.rbf_smoothing),
            )
            models.append(rbf)
        except Exception:
            models.append(None)
    return {"coords_obs": coords_obs.copy(), "U_obs": U_prime_obs.copy(), "models": models}


def _predict_spatial_field(spatial_model: Dict[str, object], coords_new: np.ndarray) -> np.ndarray:
    coords_obs = spatial_model["coords_obs"]
    U_obs = spatial_model["U_obs"]
    models = spatial_model["models"]
    nc = len(models)
    out = np.zeros((coords_new.shape[0], nc), dtype=np.float64)
    for k, mdl in enumerate(models):
        if mdl is not None:
            out[:, k] = mdl(coords_new)
        else:
            d2 = ((coords_new[:, None, :] - coords_obs[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d2, axis=1)
            out[:, k] = U_obs[nn, k]
    return out


def fit_u_to_t_bridge(U_prime_obs: np.ndarray, T_prime_obs: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    ridge.fit(U_prime_obs, T_prime_obs)
    m = ridge.coef_.T.astype(np.float64)
    b = ridge.intercept_.astype(np.float64)
    return {"M": m, "b": b}


def predict_subject_full(
    coords_obs: np.ndarray,
    X_obs_h: np.ndarray,
    X_obs_raw: np.ndarray,
    obs_idx: np.ndarray,
    coords_full: np.ndarray,
    pls_prime: Dict[str, np.ndarray],
    calibration: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    spatial_model = fit_spatial_field(coords_obs, pls_prime["U_prime"], cfg)
    bridge = fit_u_to_t_bridge(pls_prime["U_prime"], pls_prime["T_prime"], alpha=cfg.ridge_alpha_bridge)

    U_full_prime = _predict_spatial_field(spatial_model, coords_full)
    T_full_prime = U_full_prime @ bridge["M"] + bridge["b"]
    X_std_full = T_full_prime @ pls_prime["P_prime"].T
    X_full_h = X_std_full * pls_prime["x_scale"] + pls_prime["x_mean"]
    X_full_raw = inverse_harmonize_gtex(X_full_h, calibration)

    X_full_h = X_full_h.astype(np.float64)
    X_full_raw = X_full_raw.astype(np.float64)
    X_full_h[obs_idx, :] = X_obs_h
    X_full_raw[obs_idx, :] = X_obs_raw

    diag = {
        "U_full_prime": U_full_prime.astype(np.float64),
        "T_full_prime": T_full_prime.astype(np.float64),
        "bridge_M": bridge["M"].astype(np.float64),
        "bridge_b": bridge["b"].astype(np.float64),
    }
    return X_full_h, X_full_raw, diag


def evaluate_alignment_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    out = {"global_r2": r2, "rmse": rmse, "mae": mae}
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
            out[f"pearson_c{c+1}"] = float(stats.pearsonr(yt, yp).statistic)
        else:
            out[f"pearson_c{c+1}"] = np.nan
    out["mean_pearson"] = float(np.nanmean([out.get(f"pearson_c{i+1}", np.nan) for i in range(y_true.shape[1])]))
    return out


def _metrics_from_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"pearson_r": np.nan, "spearman_rho": np.nan, "rmse": np.nan, "mae": np.nan, "medae": np.nan}
    pear = np.nan
    if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pear = float(stats.pearsonr(y_true, y_pred).statistic)
    spear = float(stats.spearmanr(y_true, y_pred).statistic)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    medae = float(np.median(np.abs(y_true - y_pred)))
    return {"pearson_r": pear, "spearman_rho": spear, "rmse": rmse, "mae": mae, "medae": medae}


def evaluate_subject_loro(
    subject: str,
    obs_idx: np.ndarray,
    X_obs_h: np.ndarray,
    coords_full: np.ndarray,
    ahba_T_ref: np.ndarray,
    ahba_h_full: np.ndarray,
    calibration: Dict[str, np.ndarray],
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
            pls_train = fit_subject_pls(X_train_h, Y_train, n_comp_target=cfg.n_comp_target, adaptive=True)
            k = int(pls_train["n_comp"])
            T_ref_train = ahba_T_ref[train_idx, :k]
            R = fit_subject_o3(pls_train["T"], T_ref_train)
            prime = apply_pls_consistent_transform(pls_train, R)

            spatial = fit_spatial_field(coords_full[train_idx, :], prime["U_prime"], cfg)
            bridge = fit_u_to_t_bridge(prime["U_prime"], prime["T_prime"], alpha=cfg.ridge_alpha_bridge)
            U_hold = _predict_spatial_field(spatial, coords_full[[hold], :])
            T_hold = U_hold @ bridge["M"] + bridge["b"]
            X_std_hold = T_hold @ prime["P_prime"].T
            X_hold_h = X_std_hold * prime["x_scale"] + prime["x_mean"]
            x_true = X_obs_h[hold_pos, :].astype(np.float64)
            x_pred = X_hold_h[0, :].astype(np.float64)
            x_base = ahba_h_full[hold, :].astype(np.float64)
            met = _metrics_from_vectors(x_true, x_pred)
            fold_rows.append(
                {
                    "subject": subject,
                    "split": "loro_fold",
                    "fold_holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
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
                    "split": "loro_fold",
                    "fold_holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
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
    met = _metrics_from_vectors(yt, yp)
    baseline_rmse = float(np.sqrt(np.mean((yt - yb) ** 2)))
    summary = {
        "subject": subject,
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


def make_side_by_side_heatmap(
    left_mat: np.ndarray,
    right_mat: np.ndarray,
    title_left: str,
    title_right: str,
    suptitle: str,
    out_path: Path,
    cfg: Config,
) -> Dict[str, float]:
    vals = np.concatenate([left_mat[np.isfinite(left_mat)], right_mat[np.isfinite(right_mat)]])
    if vals.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = np.percentile(vals, [cfg.percentile_clip_low, cfg.percentile_clip_high]).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    sns.heatmap(
        left_mat,
        ax=axes[0],
        cmap="coolwarm",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    axes[0].set_title(title_left)
    axes[0].set_xlabel("Genes")
    axes[0].set_ylabel("Allen parcels")

    sns.heatmap(
        right_mat,
        ax=axes[1],
        cmap="coolwarm",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_kws={"label": "expression (shared scale)"},
    )
    axes[1].set_title(title_right)
    axes[1].set_xlabel("Genes")
    axes[1].set_ylabel("Allen parcels")
    fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {"vmin": float(vmin), "vmax": float(vmax)}


def matrix_to_aligned_table(
    matrix: np.ndarray,
    gene_cols: List[str],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    out = meta_df.copy()
    out_genes = pd.DataFrame(matrix.astype(np.float32), columns=gene_cols)
    out = pd.concat([out.reset_index(drop=True), out_genes], axis=1)
    return out


def run_scatter_suite(
    root: Path,
    gtex_raw_path: Path,
    allen_raw_path: Path,
    gtex_harm_path: Path,
    allen_harm_path: Path,
    aligned_index_path: Path,
    out_fig_dir: Path,
    out_tab_dir: Path,
) -> None:
    scatter_script = Path(__file__).resolve().parent / "run_o3_scatter_panels.py"
    cmd = [
        sys.executable,
        str(scatter_script),
        "--gtex-raw-path",
        str(gtex_raw_path),
        "--allen-raw-path",
        str(allen_raw_path),
        "--gtex-harm-path",
        str(gtex_harm_path),
        "--allen-harm-path",
        str(allen_harm_path),
        "--aligned-index-path",
        str(aligned_index_path),
        "--out-fig-dir",
        str(out_fig_dir),
        "--out-tab-dir",
        str(out_tab_dir),
        "--shift-frac",
        "0.0",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)


def select_representative_subjects(ranking_df: pd.DataFrame, total_count: int) -> pd.DataFrame:
    if len(ranking_df) == 0 or total_count <= 0:
        return pd.DataFrame(columns=["subject", "bucket", "rank", "pearson_r"])
    work = ranking_df.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    n_each = max(1, total_count // 3)
    top = work.head(n_each).copy()
    top["bucket"] = "top"

    bottom = work.tail(n_each).copy()
    bottom["bucket"] = "bottom"

    remaining = work[~work["subject"].isin(pd.concat([top["subject"], bottom["subject"]], axis=0))].copy()
    if len(remaining) == 0:
        middle = pd.DataFrame(columns=work.columns.tolist() + ["bucket"])
    else:
        mid_take = min(n_each, len(remaining))
        center = len(remaining) // 2
        start = max(0, center - (mid_take // 2))
        end = min(len(remaining), start + mid_take)
        middle = remaining.iloc[start:end].copy()
        middle["bucket"] = "middle"

    reps = pd.concat([top, middle, bottom], ignore_index=True)
    reps = reps.drop_duplicates(subset=["subject"]).reset_index(drop=True)
    reps["rank"] = reps["subject"].map({s: i + 1 for i, s in enumerate(work["subject"].tolist())})
    reps = reps.sort_values(["bucket", "rank"]).reset_index(drop=True)
    return reps[["subject", "bucket", "rank", "pearson_r"]]


def make_loro_summary_figures(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf = summary_df.copy()
    sdf = sdf[np.isfinite(sdf["pearson_r"])].copy()
    if len(sdf) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.histplot(sdf["pearson_r"], bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Subject-level LORO Pearson distribution")
    ax.set_xlabel("LORO mean Pearson (harmonized)")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "loro_subject_distribution_pearson.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.histplot(sdf["rmse"], bins=30, kde=True, ax=ax, color="#d62728")
    ax.set_title("Subject-level LORO RMSE distribution")
    ax.set_xlabel("LORO RMSE (harmonized)")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "loro_subject_distribution_rmse.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(sdf["n_obs_parcels"], sdf["pearson_r"], s=26, alpha=0.75)
    axes[0].set_xlabel("Observed parcels per subject")
    axes[0].set_ylabel("LORO Pearson")
    axes[0].set_title("Coverage vs Pearson")
    axes[0].grid(True, alpha=0.2)
    axes[1].scatter(sdf["n_obs_parcels"], sdf["rmse"], s=26, alpha=0.75)
    axes[1].set_xlabel("Observed parcels per subject")
    axes[1].set_ylabel("LORO RMSE")
    axes[1].set_title("Coverage vs RMSE")
    axes[1].grid(True, alpha=0.2)
    fig.savefig(out_dir / "loro_coverage_vs_performance.png", dpi=220)
    plt.close(fig)

    rank_df = sdf.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax.bar(np.arange(len(rank_df)), rank_df["pearson_r"].to_numpy(), width=0.9, color="#2ca02c")
    ax.set_xlabel("Subject rank (sorted by Pearson)")
    ax.set_ylabel("LORO Pearson")
    ax.set_title("Subject ranking by LORO Pearson")
    ax.grid(True, axis="y", alpha=0.2)
    if len(rank_df) > 0:
        for ridx in [0, min(1, len(rank_df) - 1), max(len(rank_df) - 2, 0), len(rank_df) - 1]:
            row = rank_df.iloc[ridx]
            ax.text(
                ridx,
                row["pearson_r"] + 0.01,
                str(row["subject"]),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.savefig(out_dir / "loro_subject_ranking.png", dpi=220)
    plt.close(fig)


def run_all_genes_transport_streaming(
    root: Path,
    cfg: Config,
    csv_path: Path,
    genes_all: List[str],
    target: pd.DataFrame,
    subject_models: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, object]:
    out_root = root / cfg.out_root
    table_dir = out_root / "tables"
    if len(subject_models) == 0:
        return {"status": "skipped_no_subject_models"}

    df_all = read_expression_subset(csv_path, genes_all)
    ahba_all = df_all[df_all["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_all = df_all[df_all["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_all["parcel_idx"] = ahba_all["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_all = map_gtex_to_target(gtex_all, target)
    cal_all = fit_domain_calibration_fast(ahba_all, gtex_all, genes_all, n_parcels=len(target))

    gtex_all_raw_agg = (
        gtex_all.groupby(["subject", "parcel_idx"], sort=False)[genes_all]
        .mean()
        .reset_index()
    )
    gtex_all_raw_agg["subject"] = gtex_all_raw_agg["subject"].astype(str)
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

    print(f"[all-genes] subjects={n_sub}, genes={n_genes}, chunk_size={chunk}")
    for start in range(0, n_genes, chunk):
        end = min(n_genes, start + chunk)
        chunk_genes = genes_all[start:end]
        csize = end - start
        chunk_raw = gtex_all_raw_agg[chunk_genes].to_numpy(dtype=np.float64)
        gm = cal_all["gtex_mean"][start:end]
        gsd = cal_all["gtex_std"][start:end]
        slope = cal_all["slope"][start:end]
        intercept = cal_all["intercept"][start:end]
        chunk_h = ((chunk_raw - gm) / gsd) * slope + intercept

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
            pred_full_raw = inverse_harmonize_gtex_chunk(pred_full_h, gm=gm, gsd=gsd, slope=slope, intercept=intercept)
            pred_full_raw[obs_idx, :] = true_obs_raw

            stack_h[si, :, :] = pred_full_h.astype(np.float32)
            stack_raw[si, :, :] = pred_full_raw.astype(np.float32)

        agg_mean_h[:, start:end] = np.nanmean(stack_h, axis=0).astype(np.float32)
        agg_median_h[:, start:end] = np.nanmedian(stack_h, axis=0).astype(np.float32)
        agg_mean_raw[:, start:end] = np.nanmean(stack_raw, axis=0).astype(np.float32)
        agg_median_raw[:, start:end] = np.nanmedian(stack_raw, axis=0).astype(np.float32)
        print(f"[all-genes] processed chunk {start}:{end}")

    allgenes_diag_rows = []
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
        allgenes_diag_rows.append(
            {
                "subject": sid,
                "n_points": int(a["n_points"]),
                "pearson_r": pear,
                "rmse": rmse,
                "mae": mae,
            }
        )
    allgenes_diag_df = pd.DataFrame(allgenes_diag_rows).sort_values("pearson_r", ascending=False, na_position="last")
    allgenes_diag_df.to_csv(table_dir / "allgenes_subject_diagnostics.csv", index=False)

    parcel_idx = target["parcel_idx"].to_numpy(dtype=np.int32)
    parcel_name = target["tissue_or_parcel"].astype(str).to_numpy()
    genes_arr = np.asarray(genes_all, dtype=object)
    np.savez_compressed(
        table_dir / "aggregate_allgenes_mean_raw.npz",
        matrix=agg_mean_raw,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / "aggregate_allgenes_mean_harmonized.npz",
        matrix=agg_mean_h,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / "aggregate_allgenes_median_raw.npz",
        matrix=agg_median_raw,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    np.savez_compressed(
        table_dir / "aggregate_allgenes_median_harmonized.npz",
        matrix=agg_median_h,
        genes=genes_arr,
        parcel_idx=parcel_idx,
        parcel_name=parcel_name,
    )
    return {
        "status": "ok",
        "n_subjects": int(n_sub),
        "n_genes": int(n_genes),
        "chunk_size": int(chunk),
        "diagnostics_rows": int(len(allgenes_diag_df)),
    }


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header = load_gene_header_and_hvg(csv_path, hvg_path)
    genes_hvg = header["genes_hvg"]
    genes_all = header["genes_all"]
    if len(genes_hvg) == 0:
        raise RuntimeError("No HVG overlap found; cannot run subject-level pipeline.")

    print("[stage] Loading HVG subset and building shared objects")
    df_hvg = read_expression_subset(csv_path, genes_hvg)
    ahba_raw = df_hvg[df_hvg["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df_hvg[df_hvg["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEx split is empty.")

    target = build_target_parcels(ahba_raw)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)

    coords_full = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    cal_hvg = fit_domain_calibration_fast(ahba_raw, gtex_raw, genes_hvg, n_parcels=len(target))
    ahba_h = apply_harmonization(ahba_raw, genes_hvg, cal_hvg, "AHBA")
    gtex_h = apply_harmonization(gtex_raw, genes_hvg, cal_hvg, "GTEX")

    ahba_h_full, _ = build_region_matrix(ahba_h, genes_hvg, target, agg="mean")
    ahba_raw_full, _ = build_region_matrix(ahba_raw, genes_hvg, target, agg="mean")
    _, global_obs_mask = build_region_matrix(gtex_raw, genes_hvg, target, agg="mean")

    ahba_ref_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
    ahba_ref_T = ahba_ref_pls["T"]

    subjects = sorted(gtex_h["subject"].dropna().astype(str).unique().tolist())
    print(f"[stage] Subject-level HVG modeling across {len(subjects)} GTEx subjects")
    eligibility_rows = []
    o3_diag_rows = []
    subject_models: Dict[str, Dict[str, np.ndarray]] = {}
    pred_h_list = []
    pred_raw_list = []
    pred_subject_ids = []
    obs_data_by_subject: Dict[str, Dict[str, np.ndarray]] = {}

    for i, sid in enumerate(subjects):
        subj_h = gtex_h[gtex_h["subject"] == sid].copy()
        subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
        n_samples = int(len(subj_h))
        obs_idx, x_obs_h, x_obs_raw = build_subject_observed_matrices(subj_h, subj_raw, genes_hvg)
        n_obs = int(len(obs_idx))
        row = {
            "subject": sid,
            "n_samples": n_samples,
            "n_obs_parcels": n_obs,
            "eligible": False,
            "reason": "",
        }
        if n_obs < cfg.min_observed_parcels:
            row["reason"] = f"insufficient_observed_parcels<{cfg.min_observed_parcels}"
            eligibility_rows.append(row)
            continue
        try:
            y_obs = np.c_[coords_full[obs_idx, 1], coords_full[obs_idx, 2], np.abs(coords_full[obs_idx, 0])]
            pls_sub = fit_subject_pls(x_obs_h, y_obs, n_comp_target=cfg.n_comp_target, adaptive=True)
            k = int(pls_sub["n_comp"])
            t_ref_obs = ahba_ref_T[obs_idx, :k]
            R = fit_subject_o3(pls_sub["T"], t_ref_obs)
            prime = apply_pls_consistent_transform(pls_sub, R)

            x_full_h, x_full_raw, pred_diag = predict_subject_full(
                coords_obs=coords_full[obs_idx, :],
                X_obs_h=x_obs_h,
                X_obs_raw=x_obs_raw,
                obs_idx=obs_idx,
                coords_full=coords_full,
                pls_prime=prime,
                calibration=cal_hvg,
                cfg=cfg,
            )

            pre_m = evaluate_alignment_matrix(t_ref_obs, pls_sub["T"])
            post_m = evaluate_alignment_matrix(t_ref_obs, prime["T_prime"])
            ortho = float(np.linalg.norm(R.T @ R - np.eye(R.shape[0]), ord="fro"))
            det = float(np.linalg.det(R))

            o3_diag_rows.append(
                {
                    "subject": sid,
                    "n_obs_parcels": n_obs,
                    "n_comp": k,
                    "det_o3": det,
                    "orthogonality_error": ortho,
                    "pre_global_r2": pre_m["global_r2"],
                    "pre_mean_pearson": pre_m["mean_pearson"],
                    "post_global_r2": post_m["global_r2"],
                    "post_mean_pearson": post_m["mean_pearson"],
                    "pre_rmse": pre_m["rmse"],
                    "post_rmse": post_m["rmse"],
                }
            )

            subject_models[sid] = {
                "obs_idx": obs_idx.astype(np.int32),
                "T_obs_prime": prime["T_prime"].astype(np.float64),
                "T_full_prime": pred_diag["T_full_prime"].astype(np.float64),
                "n_comp": np.int32(k),
            }
            obs_data_by_subject[sid] = {
                "obs_idx": obs_idx.astype(np.int32),
                "X_obs_h": x_obs_h.astype(np.float64),
                "X_obs_raw": x_obs_raw.astype(np.float64),
            }
            pred_subject_ids.append(sid)
            pred_h_list.append(x_full_h.astype(np.float32))
            pred_raw_list.append(x_full_raw.astype(np.float32))
            row["eligible"] = True
            row["reason"] = "eligible"
        except Exception as e:
            row["reason"] = f"modeling_error:{str(e)[:140]}"
        eligibility_rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(subjects)} subjects")

    eligibility_df = pd.DataFrame(eligibility_rows).sort_values("subject").reset_index(drop=True)
    o3_diag_df = pd.DataFrame(o3_diag_rows).sort_values("subject").reset_index(drop=True)
    eligibility_df.to_csv(table_dir / "subject_eligibility.csv", index=False)
    o3_diag_df.to_csv(table_dir / "subject_o3_diagnostics.csv", index=False)

    eligible_subjects = [s for s in pred_subject_ids if s in set(eligibility_df[eligibility_df["eligible"]]["subject"].tolist())]
    if len(eligible_subjects) == 0:
        raise RuntimeError("No eligible subjects passed subject-level modeling.")
    print(f"[stage] Eligible subjects: {len(eligible_subjects)}")

    pred_h_tensor = np.stack(pred_h_list, axis=0).astype(np.float32)
    pred_raw_tensor = np.stack(pred_raw_list, axis=0).astype(np.float32)
    np.savez_compressed(
        table_dir / "subject_predictions_hvg_harmonized.npz",
        matrix=pred_h_tensor,
        subjects=np.asarray(pred_subject_ids, dtype=object),
        genes=np.asarray(genes_hvg, dtype=object),
        parcel_idx=target["parcel_idx"].to_numpy(dtype=np.int32),
        parcel_name=target["tissue_or_parcel"].astype(str).to_numpy(),
    )
    np.savez_compressed(
        table_dir / "subject_predictions_hvg_raw.npz",
        matrix=pred_raw_tensor,
        subjects=np.asarray(pred_subject_ids, dtype=object),
        genes=np.asarray(genes_hvg, dtype=object),
        parcel_idx=target["parcel_idx"].to_numpy(dtype=np.int32),
        parcel_name=target["tissue_or_parcel"].astype(str).to_numpy(),
    )

    print("[stage] Running per-subject HVG LORO")
    fold_rows = []
    summary_rows = []
    for sid in pred_subject_ids:
        obs_d = obs_data_by_subject[sid]
        folds_df, summary = evaluate_subject_loro(
            subject=sid,
            obs_idx=obs_d["obs_idx"],
            X_obs_h=obs_d["X_obs_h"],
            coords_full=coords_full,
            ahba_T_ref=ahba_ref_T,
            ahba_h_full=ahba_h_full,
            calibration=cal_hvg,
            cfg=cfg,
        )
        if len(folds_df) > 0:
            fold_rows.append(folds_df)
        summary_rows.append(summary)
    folds_all_df = pd.concat(fold_rows, ignore_index=True) if len(fold_rows) else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values("pearson_r", ascending=False, na_position="last")
    folds_all_df.to_csv(table_dir / "subject_loro_folds_hvg.csv", index=False)
    summary_df.to_csv(table_dir / "subject_loro_summary_hvg.csv", index=False)

    ranking_df = summary_df.sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)
    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1, dtype=np.int32)
    ranking_df.to_csv(table_dir / "subject_ranking_hvg.csv", index=False)

    reps_df = select_representative_subjects(ranking_df, total_count=cfg.representative_count)
    reps_df.to_csv(table_dir / "representative_subjects.csv", index=False)
    make_loro_summary_figures(summary_df, fig_dir)

    print("[stage] Building aggregate HVG atlases (mean + median)")
    agg_mean_h = np.nanmean(pred_h_tensor, axis=0).astype(np.float32)
    agg_median_h = np.nanmedian(pred_h_tensor, axis=0).astype(np.float32)
    agg_mean_raw = np.nanmean(pred_raw_tensor, axis=0).astype(np.float32)
    agg_median_raw = np.nanmedian(pred_raw_tensor, axis=0).astype(np.float32)

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

    allen_raw_table = matrix_to_aligned_table(ahba_raw_full, genes_hvg, meta_df)
    allen_h_table = matrix_to_aligned_table(ahba_h_full, genes_hvg, meta_df)
    allen_raw_table.to_csv(table_dir / "allen_hvg_reference_raw.csv", index=False)
    allen_h_table.to_csv(table_dir / "allen_hvg_reference_harmonized.csv", index=False)

    mean_raw_table = matrix_to_aligned_table(agg_mean_raw, genes_hvg, meta_df)
    mean_h_table = matrix_to_aligned_table(agg_mean_h, genes_hvg, meta_df)
    med_raw_table = matrix_to_aligned_table(agg_median_raw, genes_hvg, meta_df)
    med_h_table = matrix_to_aligned_table(agg_median_h, genes_hvg, meta_df)
    mean_raw_table.to_csv(table_dir / "aggregate_hvg_mean_raw.csv", index=False)
    mean_h_table.to_csv(table_dir / "aggregate_hvg_mean_harmonized.csv", index=False)
    med_raw_table.to_csv(table_dir / "aggregate_hvg_median_raw.csv", index=False)
    med_h_table.to_csv(table_dir / "aggregate_hvg_median_harmonized.csv", index=False)

    for mode, g_raw, g_h in [
        ("aggregate_mean", agg_mean_raw, agg_mean_h),
        ("aggregate_median", agg_median_raw, agg_median_h),
    ]:
        mode_fig_dir = fig_dir / mode
        mode_tab_dir = table_dir / mode
        mode_fig_dir.mkdir(parents=True, exist_ok=True)
        mode_tab_dir.mkdir(parents=True, exist_ok=True)
        make_side_by_side_heatmap(
            left_mat=g_raw,
            right_mat=ahba_raw_full,
            title_left=f"GTEx {mode.replace('_', ' ')} predicted (raw)",
            title_right="AHBA reference (raw)",
            suptitle=f"GTEx vs AHBA side-by-side heatmap [{mode}]",
            out_path=mode_fig_dir / "side_by_side_heatmap_raw.png",
            cfg=cfg,
        )
        make_side_by_side_heatmap(
            left_mat=g_h,
            right_mat=ahba_h_full,
            title_left=f"GTEx {mode.replace('_', ' ')} predicted (harmonized)",
            title_right="AHBA reference (harmonized)",
            suptitle=f"GTEx vs AHBA side-by-side heatmap [{mode}]",
            out_path=mode_fig_dir / "side_by_side_heatmap_harmonized.png",
            cfg=cfg,
        )
        mode_g_raw_path = mode_tab_dir / "gtex_predicted_allen_rows_raw.csv"
        mode_g_h_path = mode_tab_dir / "gtex_predicted_allen_rows_harmonized.csv"
        mode_a_raw_path = mode_tab_dir / "allen_allen_rows_raw.csv"
        mode_a_h_path = mode_tab_dir / "allen_allen_rows_harmonized.csv"
        mode_idx_path = mode_tab_dir / "aligned_row_index.csv"
        matrix_to_aligned_table(g_raw, genes_hvg, meta_df).to_csv(mode_g_raw_path, index=False)
        matrix_to_aligned_table(g_h, genes_hvg, meta_df).to_csv(mode_g_h_path, index=False)
        matrix_to_aligned_table(ahba_raw_full, genes_hvg, meta_df).to_csv(mode_a_raw_path, index=False)
        matrix_to_aligned_table(ahba_h_full, genes_hvg, meta_df).to_csv(mode_a_h_path, index=False)
        meta_df.to_csv(mode_idx_path, index=False)
        run_scatter_suite(
            root=root,
            gtex_raw_path=mode_g_raw_path,
            allen_raw_path=mode_a_raw_path,
            gtex_harm_path=mode_g_h_path,
            allen_harm_path=mode_a_h_path,
            aligned_index_path=mode_idx_path,
            out_fig_dir=mode_fig_dir / "scatter_panels",
            out_tab_dir=mode_tab_dir / "scatter_panels",
        )

    print("[stage] Building representative individual visual suites")
    pred_idx_map = {sid: i for i, sid in enumerate(pred_subject_ids)}
    for _, rep in reps_df.iterrows():
        sid = str(rep["subject"])
        if sid not in pred_idx_map:
            continue
        si = pred_idx_map[sid]
        subj_raw = pred_raw_tensor[si, :, :].astype(np.float64)
        subj_h = pred_h_tensor[si, :, :].astype(np.float64)
        subj_fig_dir = fig_dir / "subjects" / sid
        subj_tab_dir = table_dir / "subjects" / sid
        subj_fig_dir.mkdir(parents=True, exist_ok=True)
        subj_tab_dir.mkdir(parents=True, exist_ok=True)

        make_side_by_side_heatmap(
            left_mat=subj_raw,
            right_mat=ahba_raw_full,
            title_left=f"{sid} GTEx predicted (raw)",
            title_right="AHBA reference (raw)",
            suptitle=f"Subject-level GTEx vs AHBA [{sid}]",
            out_path=subj_fig_dir / "side_by_side_heatmap_raw.png",
            cfg=cfg,
        )
        make_side_by_side_heatmap(
            left_mat=subj_h,
            right_mat=ahba_h_full,
            title_left=f"{sid} GTEx predicted (harmonized)",
            title_right="AHBA reference (harmonized)",
            suptitle=f"Subject-level GTEx vs AHBA [{sid}]",
            out_path=subj_fig_dir / "side_by_side_heatmap_harmonized.png",
            cfg=cfg,
        )

        g_raw_path = subj_tab_dir / "gtex_predicted_allen_rows_raw.csv"
        g_h_path = subj_tab_dir / "gtex_predicted_allen_rows_harmonized.csv"
        a_raw_path = subj_tab_dir / "allen_allen_rows_raw.csv"
        a_h_path = subj_tab_dir / "allen_allen_rows_harmonized.csv"
        idx_path = subj_tab_dir / "aligned_row_index.csv"
        matrix_to_aligned_table(subj_raw, genes_hvg, meta_df).to_csv(g_raw_path, index=False)
        matrix_to_aligned_table(subj_h, genes_hvg, meta_df).to_csv(g_h_path, index=False)
        matrix_to_aligned_table(ahba_raw_full, genes_hvg, meta_df).to_csv(a_raw_path, index=False)
        matrix_to_aligned_table(ahba_h_full, genes_hvg, meta_df).to_csv(a_h_path, index=False)
        meta_df.to_csv(idx_path, index=False)

        run_scatter_suite(
            root=root,
            gtex_raw_path=g_raw_path,
            allen_raw_path=a_raw_path,
            gtex_harm_path=g_h_path,
            allen_harm_path=a_h_path,
            aligned_index_path=idx_path,
            out_fig_dir=subj_fig_dir / "scatter_panels",
            out_tab_dir=subj_tab_dir / "scatter_panels",
        )

        rep_note = {
            "subject": sid,
            "bucket": str(rep["bucket"]),
            "rank": int(rep["rank"]),
            "pearson_r": float(rep["pearson_r"]),
        }
        with open(subj_tab_dir / "subject_note.json", "w") as f:
            json.dump(rep_note, f, indent=2)

    allgenes_result = {"status": "skipped_by_flag"}
    if cfg.run_all_genes:
        print("[stage] Running all-genes aggregate-only transport")
        allgenes_result = run_all_genes_transport_streaming(
            root=root,
            cfg=cfg,
            csv_path=csv_path,
            genes_all=genes_all,
            target=target,
            subject_models=subject_models,
        )

    overall_loro = {
        "n_subjects_with_loro": int(len(summary_df)),
        "mean_pearson": float(np.nanmean(summary_df["pearson_r"].to_numpy(dtype=np.float64))),
        "median_pearson": float(np.nanmedian(summary_df["pearson_r"].to_numpy(dtype=np.float64))),
        "mean_rmse": float(np.nanmean(summary_df["rmse"].to_numpy(dtype=np.float64))),
        "median_rmse": float(np.nanmedian(summary_df["rmse"].to_numpy(dtype=np.float64))),
        "mean_baseline_rmse": float(np.nanmean(summary_df["baseline_rmse"].to_numpy(dtype=np.float64))),
        "fraction_better_than_baseline": float(np.mean(summary_df["better_than_baseline_rmse"].astype(bool).to_numpy())),
    }
    recommendation = bool(
        np.isfinite(overall_loro["mean_pearson"])
        and overall_loro["mean_pearson"] >= 0.50
        and np.isfinite(overall_loro["mean_rmse"])
        and np.isfinite(overall_loro["mean_baseline_rmse"])
        and overall_loro["mean_rmse"] < overall_loro["mean_baseline_rmse"]
    )

    summary = {
        "config": asdict(cfg),
        "data_summary": {
            "ahba_rows": int(len(ahba_raw)),
            "gtex_rows": int(len(gtex_raw)),
            "n_gtex_subjects_total": int(len(subjects)),
            "n_target_parcels": int(len(target)),
            "n_hvg_requested": int(len(header["hvg_requested"])),
            "n_hvg_matched": int(len(genes_hvg)),
            "n_hvg_missing": int(len(header["missing_hvg"])),
        },
        "subject_summary": {
            "n_eligible_subjects": int(len(pred_subject_ids)),
            "n_ineligible_subjects": int(len(subjects) - len(pred_subject_ids)),
            "min_observed_parcels_gate": int(cfg.min_observed_parcels),
        },
        "loro_summary_hvg": overall_loro,
        "representative_subjects": reps_df.to_dict(orient="records"),
        "all_genes_stage": allgenes_result,
        "recommendation": {
            "subject_level_pipeline_success": recommendation,
            "rule": "mean_pearson>=0.50 and mean_rmse < mean_baseline_rmse",
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = f"""# Subject-Level GTEx O(3) Extrapolation Report

## Coverage
- GTEx subjects total: {len(subjects)}
- Eligible subjects (>= {cfg.min_observed_parcels} parcels): {len(pred_subject_ids)}
- Target parcels: {len(target)}
- HVG matched: {len(genes_hvg)} / requested {len(header["hvg_requested"])}

## Subject-Level LORO (HVG)
- Mean Pearson: {overall_loro['mean_pearson']:.4f}
- Median Pearson: {overall_loro['median_pearson']:.4f}
- Mean RMSE: {overall_loro['mean_rmse']:.4f}
- Mean baseline RMSE: {overall_loro['mean_baseline_rmse']:.4f}
- Fraction subjects better than baseline RMSE: {overall_loro['fraction_better_than_baseline']:.4f}

## Representative Subjects
Selected subjects (top/middle/bottom by Pearson):

```text
{reps_df.to_string(index=False)}
```

## All-Genes Stage
Status: {allgenes_result.get('status')}

## Recommendation
- Subject-level pipeline success: {recommendation}
"""
    (out_root / "report.md").write_text(report)

    print("Completed subject-level O(3) pipeline.")
    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
