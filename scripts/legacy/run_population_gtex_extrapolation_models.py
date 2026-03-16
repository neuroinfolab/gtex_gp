#!/usr/bin/env python3
"""
Population-level GTEx missing-region extrapolation with 7 models:
- Classical: method1..method4
- Basis transport: basis_so3, basis_o3, basis_affine

Outputs under out/population_extrapolation_basis/:
- summary.json
- report.md
- tables/*.csv
- figures/*.png
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/population_extrapolation_basis"

    random_seed: int = 123
    n_components: int = 3
    n_pca_axes: int = 100

    method2_latent_dim: int = 12
    method2_rbf_smoothing: float = 0.1
    method3_ridge: float = 0.1
    method4_rank: int = 10
    method4_ridge: float = 1e-2

    n_two_mask_repeats: int = 200
    dist_bins: int = 20

    gate_min_pearson: float = 0.60
    gate_rmse_improvement: float = 0.10
    gate_max_jsd: float = 0.15
    gate_var_ratio_lo: float = 0.5
    gate_var_ratio_hi: float = 2.0


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


GTEX_TO_DIV = {
    "brain - cortex": "Frontal",
    "brain - frontal cortex (ba9)": "Frontal",
    "brain - cerebellum": "Cerebellum",
    "brain - cerebellar hemisphere": "Cerebellum",
    "brain - caudate (basal ganglia)": "Caudate",
    "brain - nucleus accumbens (basal ganglia)": "Accumbens",
    "brain - putamen (basal ganglia)": "Putamen",
    "brain - hypothalamus": "Hypothalamus",
    "brain - hippocampus": "Hippocampus",
    "brain - anterior cingulate cortex (ba24)": "AntCing",
    "brain - substantia nigra": "Nigra",
    "brain - amygdala": "Amygdala",
}


def normalize_gene_name(name: str) -> str:
    return str(name).upper().replace("-", "_").replace(".", "_")


def parse_coordinate_centroid(coord_text: str) -> Tuple[float, float, float]:
    toks = COORD_PATTERN.findall(str(coord_text))
    if not toks:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray([[float(a), float(b), float(c)] for a, b, c in toks], dtype=np.float64)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def _safe_std(vec: np.ndarray) -> np.ndarray:
    sd = np.nanstd(vec, axis=0, ddof=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return sd


def load_gene_header_and_hvg(csv_path: Path, hvg_path: Path) -> Dict[str, List[str]]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    genes_all = header[6:]
    hvg = [ln.strip() for ln in hvg_path.read_text().splitlines() if ln.strip()]

    norm_map = {normalize_gene_name(g): g for g in genes_all}
    genes_hvg = [norm_map[normalize_gene_name(g)] for g in hvg if normalize_gene_name(g) in norm_map]
    missing_hvg = [g for g in hvg if normalize_gene_name(g) not in norm_map]
    return {
        "genes_all": genes_all,
        "hvg_requested": hvg,
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

    ds = df["dataset"].astype(str).str.upper().str.strip()
    df["dataset_upper"] = ds
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


def _parcel_gene_means(df: pd.DataFrame, gene_cols: List[str], n_parcels: int) -> np.ndarray:
    out = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    for r in range(n_parcels):
        sub = df[df["parcel_idx"] == r]
        if len(sub) == 0:
            continue
        out[r, :] = sub[gene_cols].to_numpy(dtype=np.float64).mean(axis=0)
    return out


def fit_domain_calibration(
    ahba_raw: pd.DataFrame,
    gtex_raw: pd.DataFrame,
    gene_cols: List[str],
    n_parcels: int,
) -> Dict[str, np.ndarray]:
    xa = ahba_raw[gene_cols].to_numpy(dtype=np.float64)
    xg = gtex_raw[gene_cols].to_numpy(dtype=np.float64)

    ahba_mean = np.nanmean(xa, axis=0)
    ahba_std = _safe_std(xa)
    gtex_mean = np.nanmean(xg, axis=0)
    gtex_std = _safe_std(xg)

    ahba_z = (xa - ahba_mean) / ahba_std
    gtex_z = (xg - gtex_mean) / gtex_std

    ah = ahba_raw.copy()
    gt = gtex_raw.copy()
    ah.loc[:, gene_cols] = pd.DataFrame(ahba_z.astype(np.float32), columns=gene_cols, index=ah.index)
    gt.loc[:, gene_cols] = pd.DataFrame(gtex_z.astype(np.float32), columns=gene_cols, index=gt.index)

    a_par = _parcel_gene_means(ah, gene_cols, n_parcels)
    g_par = _parcel_gene_means(gt, gene_cols, n_parcels)

    slope = np.ones(len(gene_cols), dtype=np.float64)
    intercept = np.zeros(len(gene_cols), dtype=np.float64)
    for g in range(len(gene_cols)):
        x = g_par[:, g]
        y = a_par[:, g]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2 and np.nanstd(x[m]) > 1e-8:
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
            slope[g] = np.clip(b1, -5.0, 5.0)
            intercept[g] = b0
        elif m.sum() >= 1:
            slope[g] = 1.0
            intercept[g] = np.nanmean(y[m] - x[m])

    slope = np.where(np.abs(slope) < 1e-6, np.sign(slope) * 1e-6 + (slope == 0) * 1e-6, slope)

    return {
        "genes": np.asarray(gene_cols, dtype=object),
        "ahba_mean": ahba_mean,
        "ahba_std": ahba_std,
        "gtex_mean": gtex_mean,
        "gtex_std": gtex_std,
        "slope": slope,
        "intercept": intercept,
    }


def apply_harmonization(df_raw: pd.DataFrame, gene_cols: List[str], cal: Dict[str, np.ndarray], dataset_name: str) -> pd.DataFrame:
    x = df_raw[gene_cols].to_numpy(dtype=np.float64)
    if dataset_name.upper() == "AHBA":
        xh = (x - cal["ahba_mean"]) / cal["ahba_std"]
    else:
        zg = (x - cal["gtex_mean"]) / cal["gtex_std"]
        xh = zg * cal["slope"] + cal["intercept"]
    out = df_raw.copy()
    out.loc[:, gene_cols] = pd.DataFrame(xh.astype(np.float32), columns=gene_cols, index=out.index)
    return out


def inverse_harmonize_gtex(xh: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    zg = (xh - cal["intercept"]) / cal["slope"]
    xr = zg * cal["gtex_std"] + cal["gtex_mean"]
    return xr


def build_region_matrix(
    df: pd.DataFrame,
    gene_cols: List[str],
    target_parcels: pd.DataFrame,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_parcels = len(target_parcels)
    mat = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    obs = np.zeros(n_parcels, dtype=bool)

    for r in range(n_parcels):
        sub = df[df["parcel_idx"] == r]
        if len(sub) == 0:
            continue
        x = sub[gene_cols].to_numpy(dtype=np.float64)
        if agg == "mean":
            mat[r, :] = np.nanmean(x, axis=0)
        elif agg == "median":
            mat[r, :] = np.nanmedian(x, axis=0)
        else:
            raise ValueError(agg)
        obs[r] = True
    return mat, obs


def zscore_cols(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd, mu, sd


def fit_so3(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    c = x.T @ y
    u, s, vt = np.linalg.svd(c)
    q = u @ vt
    if np.linalg.det(q) < 0:
        u[:, -1] *= -1.0
        q = u @ vt
    return {"matrix": q, "intercept": np.zeros(3), "singular_values": s}


def fit_o3(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    c = x.T @ y
    u, s, vt = np.linalg.svd(c)
    q = u @ vt
    return {"matrix": q, "intercept": np.zeros(3), "singular_values": s}


def fit_affine(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x1 = np.c_[x, np.ones(len(x))]
    b, *_ = np.linalg.lstsq(x1, y, rcond=None)
    m = b[:3, :]
    i = b[3, :]
    s = np.linalg.svd(x.T @ y, compute_uv=False)
    return {"matrix": m, "intercept": i, "singular_values": s}


def compute_pls_axes(df: pd.DataFrame, gene_cols: List[str], cfg: Config) -> pd.DataFrame:
    x = df[gene_cols].to_numpy(dtype=np.float64)
    y = df[["coord_y", "coord_z", "coord_abs_x"]].to_numpy(dtype=np.float64)

    xs = StandardScaler().fit_transform(x)
    n_pca = int(min(cfg.n_pca_axes, xs.shape[0] - 1, xs.shape[1]))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed)
    xp = pca.fit_transform(xs)

    pls = PLSRegression(n_components=cfg.n_components)
    pls.fit(xp, y)
    tx = pls.x_scores_.copy()
    ty = pls.y_scores_.copy()

    for c in range(cfg.n_components):
        rr = np.corrcoef(tx[:, c], ty[:, c])[0, 1]
        if np.isfinite(rr) and rr < 0:
            tx[:, c] *= -1.0
            ty[:, c] *= -1.0

    out = df[["tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].copy()
    for c in range(3):
        out[f"X_C{c+1}"] = tx[:, c]
        out[f"Y_C{c+1}"] = ty[:, c]
    return out


def region_mean_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["X_C1", "X_C2", "X_C3", "Y_C1", "Y_C2", "Y_C3", "coord_x", "coord_y", "coord_z"]
    return score_df.groupby("tissue_or_parcel", as_index=False)[cols].mean()


def build_pairing_nearest(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ah = region_mean_scores(ahba_scores)
    gt = region_mean_scores(gtex_scores)

    axyz = ah[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = gt[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((gxyz[:, None, :] - axyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(gt)), idx])

    pair = pd.DataFrame(
        {
            "pairing": "nearest_parcel",
            "gtex_tissue": gt["tissue_or_parcel"].values,
            "ahba_region": ah.loc[idx, "tissue_or_parcel"].values,
            "distance": dist,
        }
    )
    for c in [1, 2, 3]:
        pair[f"G_X_C{c}"] = gt[f"X_C{c}"].values
        pair[f"A_X_C{c}"] = ah.loc[idx, f"X_C{c}"].values
    pair["gtex_division"] = pair["gtex_tissue"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))
    return pair.sort_values("gtex_tissue").reset_index(drop=True)


def assign_ahba_divisions_from_nearest_gtex(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ah = region_mean_scores(ahba_scores)
    gt = region_mean_scores(gtex_scores).copy()
    gt["division"] = gt["tissue_or_parcel"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))

    axyz = ah[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = gt[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((axyz[:, None, :] - gxyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    ah["division"] = gt.loc[idx, "division"].values
    return ah


def build_pairing_coarse(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ahd = assign_ahba_divisions_from_nearest_gtex(ahba_scores, gtex_scores)
    gtr = region_mean_scores(gtex_scores).copy()
    gtr["division"] = gtr["tissue_or_parcel"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))

    ac = ahd.groupby("division", as_index=False)[[f"X_C{i}" for i in [1, 2, 3]]].mean()
    gc = gtr.groupby("division", as_index=False)[[f"X_C{i}" for i in [1, 2, 3]]].mean()
    shared = sorted(set(ac["division"]) & set(gc["division"]))
    ac = ac[ac["division"].isin(shared)].sort_values("division").reset_index(drop=True)
    gc = gc[gc["division"].isin(shared)].sort_values("division").reset_index(drop=True)

    out = pd.DataFrame({"pairing": "coarse_division", "division": shared})
    for c in [1, 2, 3]:
        out[f"G_X_C{c}"] = gc[f"X_C{c}"].values
        out[f"A_X_C{c}"] = ac[f"X_C{c}"].values
    return out


def fit_basis_transforms_from_sample_scores(
    ahba_h: pd.DataFrame,
    gtex_h: pd.DataFrame,
    gene_cols: List[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    ah_scores = compute_pls_axes(ahba_h, gene_cols, cfg)
    gt_scores = compute_pls_axes(gtex_h, gene_cols, cfg)

    pair_near = build_pairing_nearest(ah_scores, gt_scores)
    pair_coarse = build_pairing_coarse(ah_scores, gt_scores)

    rows: List[Dict[str, float]] = []
    transforms: Dict[str, Dict[str, np.ndarray]] = {}
    for pname, ptab in [("nearest_parcel", pair_near), ("coarse_division", pair_coarse)]:
        gx = ptab[["G_X_C1", "G_X_C2", "G_X_C3"]].to_numpy(dtype=np.float64)
        ax = ptab[["A_X_C1", "A_X_C2", "A_X_C3"]].to_numpy(dtype=np.float64)
        gz, _, _ = zscore_cols(gx)
        az, _, _ = zscore_cols(ax)

        so3 = fit_so3(gz, az)
        o3 = fit_o3(gz, az)
        aff = fit_affine(gz, az)

        for mname, mm in [("basis_so3", so3), ("basis_o3", o3), ("basis_affine", aff)]:
            m = mm["matrix"]
            b = mm["intercept"]
            rows.append(
                {
                    "pairing": pname,
                    "model": mname,
                    "det": float(np.linalg.det(m)),
                    "orthogonality_error": float(np.linalg.norm(m.T @ m - np.eye(3), ord="fro")),
                    "condition_number": float(np.linalg.cond(m)),
                    "intercept_1": float(b[0]),
                    "intercept_2": float(b[1]),
                    "intercept_3": float(b[2]),
                    "m_11": float(m[0, 0]),
                    "m_12": float(m[0, 1]),
                    "m_13": float(m[0, 2]),
                    "m_21": float(m[1, 0]),
                    "m_22": float(m[1, 1]),
                    "m_23": float(m[1, 2]),
                    "m_31": float(m[2, 0]),
                    "m_32": float(m[2, 1]),
                    "m_33": float(m[2, 2]),
                }
            )
        transforms[pname] = {
            "basis_so3_matrix": so3["matrix"],
            "basis_so3_intercept": so3["intercept"],
            "basis_o3_matrix": o3["matrix"],
            "basis_o3_intercept": o3["intercept"],
            "basis_affine_matrix": aff["matrix"],
            "basis_affine_intercept": aff["intercept"],
        }

    diag = pd.DataFrame(rows)
    return pair_near, pair_coarse, {"diag": diag, "transforms": transforms}


def fit_pls_basis_for_gtex_obs(
    x_obs_h: np.ndarray,
    y_obs: np.ndarray,
    n_components: int,
) -> Dict[str, np.ndarray]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_obs_h)
    nc = int(min(n_components, xs.shape[0] - 1, xs.shape[1]))
    nc = max(nc, 1)
    pls = PLSRegression(n_components=nc, scale=False)
    pls.fit(xs, y_obs)
    t = pls.x_scores_.astype(np.float64)
    p = pls.x_loadings_.astype(np.float64)
    return {
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "scores_obs": t,
        "loadings": p,
    }


def _interp_scores(coords_src: np.ndarray, vals_src: np.ndarray, coords_tgt: np.ndarray, smoothing: float) -> np.ndarray:
    n_tgt = coords_tgt.shape[0]
    n_comp = vals_src.shape[1]
    out = np.zeros((n_tgt, n_comp), dtype=np.float64)
    for c in range(n_comp):
        try:
            rbf = RBFInterpolator(
                coords_src,
                vals_src[:, c],
                smoothing=smoothing,
                kernel="thin_plate_spline",
            )
            out[:, c] = rbf(coords_tgt)
        except Exception:
            d2 = ((coords_tgt[:, None, :] - coords_src[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d2, axis=1)
            out[:, c] = vals_src[nn, c]
    return out


def predict_basis_transport(
    model_name: str,
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    coords_full: np.ndarray,
    basis_mat: np.ndarray,
    basis_intercept: np.ndarray,
    cfg: Config,
    enforce_observed: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    y_obs = np.c_[coords_full[obs_idx, 1], coords_full[obs_idx, 2], np.abs(coords_full[obs_idx, 0])]
    pls_obj = fit_pls_basis_for_gtex_obs(x_obs_h, y_obs, cfg.n_components)
    t_obs = pls_obj["scores_obs"]
    p = pls_obj["loadings"]

    if model_name == "basis_affine":
        t_a = t_obs @ basis_mat + basis_intercept
    else:
        t_a = t_obs @ basis_mat

    t_full = _interp_scores(coords_full[obs_idx], t_a, coords_full, cfg.method2_rbf_smoothing)
    x_full_std = t_full @ p.T
    x_full = x_full_std * pls_obj["scaler_scale"] + pls_obj["scaler_mean"]
    if enforce_observed:
        x_full[obs_idx, :] = x_obs_h

    diag = {
        "scores_obs": t_obs,
        "scores_transformed_obs": t_a,
        "scores_transformed_full": t_full,
    }
    return x_full.astype(np.float64), diag


def predict_method1(
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    ahba_h: np.ndarray,
    coords_full: np.ndarray,
) -> np.ndarray:
    pred = ahba_h.copy()
    pred[obs_idx, :] = x_obs_h
    miss = np.array([r for r in range(len(ahba_h)) if r not in set(obs_idx.tolist())], dtype=int)
    if len(miss) == 0:
        return pred
    for r in miss:
        d2 = ((coords_full[obs_idx] - coords_full[r]) ** 2).sum(axis=1)
        o = obs_idx[np.argmin(d2)]
        pred[r, :] = x_obs_h[np.where(obs_idx == o)[0][0], :] + (ahba_h[r, :] - ahba_h[o, :])
    return pred


def predict_method2(
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    ahba_h: np.ndarray,
    coords_full: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    scaler = StandardScaler()
    ah_s = scaler.fit_transform(ahba_h)
    latent_dim = int(min(cfg.method2_latent_dim, ah_s.shape[0] - 1, ah_s.shape[1], 20))
    latent_dim = max(latent_dim, 1)
    pca = PCA(n_components=latent_dim, random_state=cfg.random_seed)
    z_ah = pca.fit_transform(ah_s)
    dec = Ridge(alpha=1e-2, random_state=cfg.random_seed)
    dec.fit(z_ah, ah_s)

    if len(obs_idx) < 2:
        pred = ahba_h.copy()
        pred[obs_idx] = x_obs_h
        return pred

    x_obs_s = scaler.transform(x_obs_h)
    z_obs = pca.transform(x_obs_s)
    z_full = _interp_scores(coords_full[obs_idx], z_obs, coords_full, cfg.method2_rbf_smoothing)
    pred_s = dec.predict(z_full)
    pred = scaler.inverse_transform(pred_s)
    pred[obs_idx] = x_obs_h
    return pred


def predict_method3(
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    ahba_h: np.ndarray,
    ridge: float,
) -> np.ndarray:
    r = ahba_h.shape[0]
    miss = np.array([i for i in range(r) if i not in set(obs_idx.tolist())], dtype=int)
    pred = ahba_h.copy()
    pred[obs_idx, :] = x_obs_h
    if len(miss) == 0:
        return pred

    sigma = np.cov(ahba_h, rowvar=True, bias=True) + ridge * np.eye(r)
    soo = sigma[np.ix_(obs_idx, obs_idx)] + ridge * np.eye(len(obs_idx))
    smo = sigma[np.ix_(miss, obs_idx)]
    k = smo @ np.linalg.pinv(soo)

    d_obs = x_obs_h - ahba_h[obs_idx, :]
    d_miss = k @ d_obs
    pred[miss, :] = ahba_h[miss, :] + d_miss
    return pred


def predict_method4(
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    ahba_h: np.ndarray,
    rank: int,
    ridge: float,
) -> np.ndarray:
    r, g = ahba_h.shape
    gene_mu = ahba_h.mean(axis=0, keepdims=True)
    a0 = ahba_h - gene_mu
    u, s, vt = np.linalg.svd(a0, full_matrices=False)
    k = int(min(rank, max(1, len(obs_idx) - 1), len(s), r, g))
    uk = u[:, :k]
    uo = uk[obs_idx, :]
    xo = x_obs_h - gene_mu
    m = np.linalg.solve(uo.T @ uo + ridge * np.eye(k), uo.T @ xo)
    pred = uk @ m + gene_mu
    pred[obs_idx, :] = x_obs_h
    return pred


def run_model_prediction(
    model_name: str,
    obs_idx: np.ndarray,
    x_obs_h: np.ndarray,
    ahba_h: np.ndarray,
    coords_full: np.ndarray,
    cfg: Config,
    basis_params: Optional[Dict[str, np.ndarray]] = None,
    enforce_observed: bool = True,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    if model_name == "method1":
        return predict_method1(obs_idx, x_obs_h, ahba_h, coords_full), None
    if model_name == "method2":
        return predict_method2(obs_idx, x_obs_h, ahba_h, coords_full, cfg), None
    if model_name == "method3":
        return predict_method3(obs_idx, x_obs_h, ahba_h, cfg.method3_ridge), None
    if model_name == "method4":
        return predict_method4(obs_idx, x_obs_h, ahba_h, cfg.method4_rank, cfg.method4_ridge), None
    if model_name in {"basis_so3", "basis_o3", "basis_affine"}:
        if basis_params is None:
            raise ValueError("basis_params required for basis models")
        pred, diag = predict_basis_transport(
            model_name=model_name,
            obs_idx=obs_idx,
            x_obs_h=x_obs_h,
            coords_full=coords_full,
            basis_mat=basis_params[f"{model_name}_matrix"],
            basis_intercept=basis_params[f"{model_name}_intercept"],
            cfg=cfg,
            enforce_observed=enforce_observed,
        )
        return pred, diag
    raise ValueError(model_name)


def eval_vector(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) < 2:
        return {
            "n_points": int(len(yt)),
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
        }
    pr = stats.pearsonr(yt, yp).statistic if np.std(yt) > 0 and np.std(yp) > 0 else np.nan
    sr = stats.spearmanr(yt, yp).statistic
    return {
        "n_points": int(len(yt)),
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mae": float(np.mean(np.abs(yt - yp))),
        "medae": float(np.median(np.abs(yt - yp))),
    }


def jsd_1d(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return np.nan
    h1, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    h2, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    p = h1.astype(np.float64) + 1e-12
    q = h2.astype(np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def distribution_metrics(
    pred_full: np.ndarray,
    gtex_full: np.ndarray,
    obs_idx: np.ndarray,
    bins: int,
) -> Dict[str, float]:
    miss_idx = np.array([i for i in range(len(gtex_full)) if i not in set(obs_idx.tolist())], dtype=int)
    if len(miss_idx) == 0:
        return {
            "jsd_median": np.nan,
            "wasserstein_median": np.nan,
            "ks_median": np.nan,
            "var_ratio_median": np.nan,
            "mean_shift_z_median": np.nan,
        }

    obs = gtex_full[obs_idx, :]
    pred_miss = pred_full[miss_idx, :]
    jsd_vals = []
    wd_vals = []
    ks_vals = []
    vr_vals = []
    ms_vals = []
    for g in range(obs.shape[1]):
        o = obs[:, g]
        p = pred_miss[:, g]
        m = np.isfinite(o)
        n = np.isfinite(p)
        if m.sum() < 2 or n.sum() < 2:
            continue
        oo = o[m]
        pp = p[n]
        jsd_vals.append(jsd_1d(oo, pp, bins=bins))
        wd_vals.append(float(wasserstein_distance(oo, pp)))
        ks_vals.append(float(ks_2samp(oo, pp).statistic))
        vo = float(np.var(oo, ddof=0))
        vp = float(np.var(pp, ddof=0))
        if vo > 1e-12:
            vr_vals.append(vp / vo)
            ms_vals.append(abs(np.mean(pp) - np.mean(oo)) / np.sqrt(vo))

    return {
        "jsd_median": float(np.nanmedian(jsd_vals)) if len(jsd_vals) else np.nan,
        "wasserstein_median": float(np.nanmedian(wd_vals)) if len(wd_vals) else np.nan,
        "ks_median": float(np.nanmedian(ks_vals)) if len(ks_vals) else np.nan,
        "var_ratio_median": float(np.nanmedian(vr_vals)) if len(vr_vals) else np.nan,
        "mean_shift_z_median": float(np.nanmedian(ms_vals)) if len(ms_vals) else np.nan,
    }


def compute_biology_ordering(
    ahba_h: np.ndarray,
    coords_full: np.ndarray,
    gene_cols: List[str],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
    scaler = StandardScaler()
    xs = scaler.fit_transform(ahba_h)
    pls = PLSRegression(n_components=min(cfg.n_components, ahba_h.shape[0] - 1, ahba_h.shape[1]), scale=False)
    pls.fit(xs, y)
    c1_scores = pls.x_scores_[:, 0]
    c1_load = pls.x_loadings_[:, 0]

    region_order = np.argsort(c1_scores)
    gene_order = np.lexsort((-np.abs(c1_load), -c1_load))
    return region_order.astype(int), gene_order.astype(int), c1_scores, c1_load


def heatmap_matrix(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    vmax_clip: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    plot_mat = mat.copy()
    if vmax_clip is None:
        lo, hi = np.nanpercentile(plot_mat, [2, 98])
    else:
        lo, hi = vmax_clip
    sns.heatmap(
        plot_mat,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=lo,
        vmax=hi,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "expression"},
    )
    ax.set_title(title)
    ax.set_xlabel("genes")
    ax.set_ylabel("regions")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cv_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(yt, yp, s=8, alpha=0.35)
    lo = min(np.min(yt), np.min(yp))
    hi = max(np.max(yt), np.max(yp))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    rr = np.corrcoef(yt, yp)[0, 1] if np.std(yt) > 0 and np.std(yp) > 0 else np.nan
    ax.set_title(f"{title}\nr={rr:.3f}, RMSE={np.sqrt(np.mean((yt-yp)**2)):.3f}")
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_distribution_compare(
    obs_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.kdeplot(obs_vals, ax=ax, label="GTEx observed", fill=True, alpha=0.25)
    sns.kdeplot(pred_vals, ax=ax, label="Pred missing", fill=True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("expression")
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_score_space_prepost(
    score_diag: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    t_pre = score_diag["scores_obs"]
    t_post = score_diag["scores_transformed_obs"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for c in [0, 1, 2]:
        ax = axes[c]
        if c >= t_pre.shape[1] or c >= t_post.shape[1]:
            ax.axis("off")
            continue
        ax.scatter(t_pre[:, c], t_post[:, c], s=40, alpha=0.8)
        lo = min(np.min(t_pre[:, c]), np.min(t_post[:, c]))
        hi = max(np.max(t_pre[:, c]), np.max(t_post[:, c]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        rr = np.corrcoef(t_pre[:, c], t_post[:, c])[0, 1] if np.std(t_pre[:, c]) > 0 and np.std(t_post[:, c]) > 0 else np.nan
        ax.set_title(f"C{c+1} pre/post (r={rr:.2f})")
        ax.set_xlabel("pre")
        ax.set_ylabel("post")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.random_seed)
    sns.set_context("talk")
    sns.set_style("whitegrid")

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
    gene_cols = header["genes_hvg"]
    if len(gene_cols) == 0:
        raise RuntimeError("No HVG overlap found between ahba_100hvg.txt and CSV header.")

    df = read_expression_subset(csv_path, gene_cols)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEx subset empty after load/parse.")

    target = build_target_parcels(ahba_raw)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)

    cal = fit_domain_calibration(ahba_raw, gtex_raw, gene_cols, n_parcels=len(target))
    ahba_h = apply_harmonization(ahba_raw, gene_cols, cal, "AHBA")
    gtex_h = apply_harmonization(gtex_raw, gene_cols, cal, "GTEX")

    # Pairings and basis transform diagnostics from sample-level harmonized scores.
    pair_near, pair_coarse, basis_fit = fit_basis_transforms_from_sample_scores(ahba_h, gtex_h, gene_cols, cfg)
    pair_near.to_csv(table_dir / "pairing_map_nearest.csv", index=False)
    pair_coarse.to_csv(table_dir / "pairing_map_coarse.csv", index=False)

    basis_diag = basis_fit["diag"].copy()
    basis_diag.to_csv(table_dir / "basis_model_diagnostics.csv", index=False)
    basis_params = basis_fit["transforms"]["nearest_parcel"]

    coords_full = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    region_names = target["tissue_or_parcel"].tolist()

    all_models = ["method1", "method2", "method3", "method4", "basis_so3", "basis_o3", "basis_affine"]
    basis_models = {"basis_so3", "basis_o3", "basis_affine"}
    classical_models = {"method1", "method2", "method3", "method4"}

    cv_rows: List[Dict[str, object]] = []
    dist_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []

    per_model_predictions: Dict[Tuple[str, str], np.ndarray] = {}

    for agg in ["mean", "median"]:
        ahba_raw_mat, _ = build_region_matrix(ahba_raw, gene_cols, target, agg)
        gtex_raw_mat, obs_mask = build_region_matrix(gtex_raw, gene_cols, target, agg)
        ahba_h_mat, _ = build_region_matrix(ahba_h, gene_cols, target, agg)
        gtex_h_mat, obs_mask_h = build_region_matrix(gtex_h, gene_cols, target, agg)
        if not np.array_equal(obs_mask, obs_mask_h):
            raise RuntimeError("Observed masks differ between raw and harmonized matrices.")

        obs_idx = np.where(obs_mask)[0]
        if len(obs_idx) < 3:
            raise RuntimeError(f"Too few observed GTEx parcels ({len(obs_idx)}) for CV in agg={agg}.")
        miss_idx = np.array([i for i in range(len(target)) if i not in set(obs_idx.tolist())], dtype=int)

        # Ordering derived from AHBA harmonized to keep stable across plots.
        reg_order, gene_order, c1_scores, c1_load = compute_biology_ordering(ahba_h_mat, coords_full, gene_cols, cfg)

        ah_ord = ahba_raw_mat[np.ix_(reg_order, gene_order)]
        gt_obs_ord = gtex_raw_mat[np.ix_(obs_idx, gene_order)]
        obs_reg_order = np.argsort(c1_scores[obs_idx])
        gt_obs_ord = gt_obs_ord[obs_reg_order, :]
        gt_obs_labels = [region_names[obs_idx[i]] for i in obs_reg_order]

        heatmap_matrix(
            mat=ah_ord,
            row_labels=[region_names[i] for i in reg_order],
            col_labels=[gene_cols[i] for i in gene_order],
            title=f"AHBA region x HVG raw ({agg})",
            out_path=fig_dir / f"heatmap_ahba_raw_{agg}.png",
        )
        heatmap_matrix(
            mat=gt_obs_ord,
            row_labels=gt_obs_labels,
            col_labels=[gene_cols[i] for i in gene_order],
            title=f"GTEx observed region x HVG raw ({agg})",
            out_path=fig_dir / f"heatmap_gtex_observed_raw_{agg}.png",
        )

        # Naive baseline for gate.
        naive_pred = ahba_h_mat.copy()
        y_true_loro_naive = []
        y_pred_loro_naive = []
        for hold in obs_idx:
            y_true_loro_naive.append(gtex_h_mat[hold, :].copy())
            y_pred_loro_naive.append(naive_pred[hold, :].copy())
        y_true_loro_naive = np.asarray(y_true_loro_naive, dtype=np.float64)
        y_pred_loro_naive = np.asarray(y_pred_loro_naive, dtype=np.float64)
        naive_m = eval_vector(y_true_loro_naive, y_pred_loro_naive)
        cv_rows.append(
            {
                "aggregation": agg,
                "model": "naive_ahba_fill",
                "split": "loro",
                **naive_m,
            }
        )
        baseline_rmse = naive_m["rmse"]

        for model in all_models:
            x_obs_h = gtex_h_mat[obs_idx, :]
            pred_h_full, score_diag = run_model_prediction(
                model_name=model,
                obs_idx=obs_idx,
                x_obs_h=x_obs_h,
                ahba_h=ahba_h_mat,
                coords_full=coords_full,
                cfg=cfg,
                basis_params=basis_params,
                enforce_observed=True,
            )
            pred_raw_full = inverse_harmonize_gtex(pred_h_full, cal)
            per_model_predictions[(agg, model)] = pred_raw_full

            # Save matrix as CSV.
            pred_df = pd.DataFrame(pred_raw_full, columns=gene_cols)
            pred_df.insert(0, "parcel_name", region_names)
            pred_df.insert(0, "parcel_idx", np.arange(len(region_names), dtype=np.int32))
            pred_df.to_csv(table_dir / f"population_predictions_{model}_{agg}.csv", index=False)

            # Heatmaps.
            pred_raw_ord = pred_raw_full[np.ix_(reg_order, gene_order)]
            pred_h_ord = pred_h_full[np.ix_(reg_order, gene_order)]
            heatmap_matrix(
                mat=pred_raw_ord,
                row_labels=[region_names[i] for i in reg_order],
                col_labels=[gene_cols[i] for i in gene_order],
                title=f"GTEx observed + extrapolated raw ({model}, {agg})",
                out_path=fig_dir / f"heatmap_gtex_full_raw_{model}_{agg}.png",
            )
            heatmap_matrix(
                mat=pred_h_ord,
                row_labels=[region_names[i] for i in reg_order],
                col_labels=[gene_cols[i] for i in gene_order],
                title=f"GTEx observed + extrapolated harmonized ({model}, {agg})",
                out_path=fig_dir / f"heatmap_gtex_full_harmonized_{model}_{agg}.png",
            )
            if model in basis_models:
                heatmap_matrix(
                    mat=pred_raw_ord,
                    row_labels=[region_names[i] for i in reg_order],
                    col_labels=[gene_cols[i] for i in gene_order],
                    title=f"GTEx full raw rotated-basis ({model}, {agg})",
                    out_path=fig_dir / f"heatmap_gtex_full_rotated_basis_raw_{model}_{agg}.png",
                )

            # LORO CV.
            y_true = []
            y_pred = []
            for hold in obs_idx:
                train_idx = np.array([i for i in obs_idx if i != hold], dtype=int)
                train_x = gtex_h_mat[train_idx, :]
                pred_h_fold, _ = run_model_prediction(
                    model_name=model,
                    obs_idx=train_idx,
                    x_obs_h=train_x,
                    ahba_h=ahba_h_mat,
                    coords_full=coords_full,
                    cfg=cfg,
                    basis_params=basis_params,
                    enforce_observed=False,
                )
                y_true.append(gtex_h_mat[hold, :].copy())
                y_pred.append(pred_h_fold[hold, :].copy())

            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            met_loro = eval_vector(y_true, y_pred)
            cv_rows.append(
                {
                    "aggregation": agg,
                    "model": model,
                    "split": "loro",
                    **met_loro,
                }
            )

            plot_cv_scatter(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{model} LORO CV ({agg})",
                out_path=fig_dir / f"cv_scatter_{model}_{agg}.png",
            )

            # 2-region mask repeated CV.
            y_true2 = []
            y_pred2 = []
            if len(obs_idx) >= 4:
                for _ in range(cfg.n_two_mask_repeats):
                    hold2 = rng.choice(obs_idx, size=2, replace=False)
                    train_idx = np.array([i for i in obs_idx if i not in set(hold2.tolist())], dtype=int)
                    train_x = gtex_h_mat[train_idx, :]
                    pred_h_fold, _ = run_model_prediction(
                        model_name=model,
                        obs_idx=train_idx,
                        x_obs_h=train_x,
                        ahba_h=ahba_h_mat,
                        coords_full=coords_full,
                        cfg=cfg,
                        basis_params=basis_params,
                        enforce_observed=False,
                    )
                    y_true2.append(gtex_h_mat[hold2, :].copy())
                    y_pred2.append(pred_h_fold[hold2, :].copy())
                y_true2 = np.concatenate(y_true2, axis=0)
                y_pred2 = np.concatenate(y_pred2, axis=0)
                met_2mask = eval_vector(y_true2, y_pred2)
                cv_rows.append(
                    {
                        "aggregation": agg,
                        "model": model,
                        "split": "two_mask_repeat",
                        **met_2mask,
                    }
                )

            # Distribution diagnostics (harmonized for gate).
            dmet = distribution_metrics(pred_h_full, gtex_h_mat, obs_idx, bins=cfg.dist_bins)
            dmet.update({"aggregation": agg, "model": model, "scale": "harmonized"})
            dist_rows.append(dmet)

            # Raw distribution figure.
            obs_vals = gtex_raw_mat[obs_idx, :].reshape(-1)
            pred_vals = pred_raw_full[miss_idx, :].reshape(-1) if len(miss_idx) else pred_raw_full.reshape(-1)
            m = np.isfinite(obs_vals)
            n = np.isfinite(pred_vals)
            if m.sum() > 2 and n.sum() > 2:
                plot_distribution_compare(
                    obs_vals=obs_vals[m],
                    pred_vals=pred_vals[n],
                    title=f"Observed vs extrapolated missing ({model}, {agg})",
                    out_path=fig_dir / f"distribution_compare_{model}_{agg}.png",
                )

            if model in basis_models and score_diag is not None:
                plot_score_space_prepost(
                    score_diag=score_diag,
                    title=f"GTEx score-space pre/post transform ({model}, {agg})",
                    out_path=fig_dir / f"score_space_prepost_{model}_{agg}.png",
                )

            # Gate by primary split LORO + distribution.
            rmse_ok = bool(np.isfinite(met_loro["rmse"]) and np.isfinite(baseline_rmse) and (met_loro["rmse"] <= (1.0 - cfg.gate_rmse_improvement) * baseline_rmse))
            pear_ok = bool(np.isfinite(met_loro["pearson_r"]) and (met_loro["pearson_r"] >= cfg.gate_min_pearson))
            jsd_ok = bool(np.isfinite(dmet["jsd_median"]) and dmet["jsd_median"] <= cfg.gate_max_jsd)
            var_ok = bool(np.isfinite(dmet["var_ratio_median"]) and (cfg.gate_var_ratio_lo <= dmet["var_ratio_median"] <= cfg.gate_var_ratio_hi))
            gate_rows.append(
                {
                    "aggregation": agg,
                    "model": model,
                    "pass_cv_pearson": pear_ok,
                    "pass_cv_rmse": rmse_ok,
                    "pass_dist_jsd": jsd_ok,
                    "pass_dist_var_ratio": var_ok,
                    "pass_dual_gate": bool(pear_ok and rmse_ok and jsd_ok and var_ok),
                }
            )

        # Ranking for this aggregation.
        cv_df_tmp = pd.DataFrame(cv_rows)
        dist_df_tmp = pd.DataFrame(dist_rows)
        loro = cv_df_tmp[(cv_df_tmp["aggregation"] == agg) & (cv_df_tmp["split"] == "loro") & (cv_df_tmp["model"] != "naive_ahba_fill")].copy()
        dist_a = dist_df_tmp[(dist_df_tmp["aggregation"] == agg) & (dist_df_tmp["scale"] == "harmonized")].copy()
        rk = loro.merge(dist_a, on=["aggregation", "model"], how="left")
        rk["rank_pearson"] = rk["pearson_r"].rank(ascending=False, method="average")
        rk["rank_rmse"] = rk["rmse"].rank(ascending=True, method="average")
        rk["rank_jsd"] = rk["jsd_median"].rank(ascending=True, method="average")
        rk["rank_var"] = np.abs(np.log(np.clip(rk["var_ratio_median"], 1e-6, 1e6))).rank(ascending=True, method="average")
        rk["overall_rank_score"] = rk[["rank_pearson", "rank_rmse", "rank_jsd", "rank_var"]].mean(axis=1)
        rk = rk.sort_values("overall_rank_score").reset_index(drop=True)
        rank_rows.extend(rk.to_dict("records"))

    # Save tables.
    cv_df = pd.DataFrame(cv_rows)
    dist_df = pd.DataFrame(dist_rows)
    rank_df = pd.DataFrame(rank_rows)
    gate_df = pd.DataFrame(gate_rows)

    cv_df.to_csv(table_dir / "model_metrics_cv.csv", index=False)
    dist_df.to_csv(table_dir / "model_metrics_distribution.csv", index=False)
    rank_df.to_csv(table_dir / "model_ranking.csv", index=False)
    gate_df.to_csv(table_dir / "model_gate_results.csv", index=False)

    # Final gate verdict on primary aggregation (mean).
    gate_mean = gate_df[gate_df["aggregation"] == "mean"].copy()
    pass_basis = bool(gate_mean[(gate_mean["model"].isin(list(basis_models))) & (gate_mean["pass_dual_gate"])].shape[0] > 0)
    pass_classical = bool(gate_mean[(gate_mean["model"].isin(list(classical_models))) & (gate_mean["pass_dual_gate"])].shape[0] > 0)
    proceed_single_subject = bool(pass_basis and pass_classical)

    data_summary = {
        "ahba_rows": int(len(ahba_raw)),
        "gtex_rows": int(len(gtex_raw)),
        "ahba_regions": int(ahba_raw["tissue_or_parcel"].nunique()),
        "gtex_tissues": int(gtex_raw["tissue_or_parcel"].nunique()),
        "target_parcels": int(len(target)),
        "gtex_observed_target_parcels_mean": int(build_region_matrix(gtex_raw, gene_cols, target, "mean")[1].sum()),
        "n_hvg_requested": int(len(header["hvg_requested"])),
        "n_hvg_matched": int(len(gene_cols)),
        "n_hvg_missing": int(len(header["missing_hvg"])),
    }

    summary = {
        "config": asdict(cfg),
        "data_summary": data_summary,
        "pairings": {
            "nearest_pairs": int(len(pair_near)),
            "coarse_pairs": int(len(pair_coarse)),
        },
        "models": {
            "classical": sorted(list(classical_models)),
            "basis": sorted(list(basis_models)),
            "all": all_models,
        },
        "gate": {
            "min_loro_pearson": cfg.gate_min_pearson,
            "rmse_improvement_vs_baseline": cfg.gate_rmse_improvement,
            "max_jsd_median": cfg.gate_max_jsd,
            "var_ratio_range": [cfg.gate_var_ratio_lo, cfg.gate_var_ratio_hi],
            "pass_basis_model": pass_basis,
            "pass_classical_model": pass_classical,
            "proceed_single_subject": proceed_single_subject,
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Report.
    def _fmt_df(df: pd.DataFrame, n: int = 20) -> str:
        if df.empty:
            return "(empty)"
        return df.head(n).to_string(index=False)

    loro_mean = cv_df[(cv_df["aggregation"] == "mean") & (cv_df["split"] == "loro")].copy()
    dist_mean = dist_df[(dist_df["aggregation"] == "mean") & (dist_df["scale"] == "harmonized")].copy()
    rank_mean = rank_df[rank_df["aggregation"] == "mean"].copy() if "aggregation" in rank_df.columns else pd.DataFrame()

    report = f"""# Population GTEx Missing-Region Extrapolation (7 Models)

## Scope
Population-level region x HVG extrapolation over AHBA target parcels using:
- Classical: method1..method4
- Basis transport: basis_so3, basis_o3, basis_affine

## Data Summary
- AHBA rows: {data_summary['ahba_rows']}
- GTEx rows: {data_summary['gtex_rows']}
- AHBA unique regions: {data_summary['ahba_regions']}
- GTEx tissues: {data_summary['gtex_tissues']}
- Target parcels: {data_summary['target_parcels']}
- GTEx observed target parcels: {data_summary['gtex_observed_target_parcels_mean']}
- HVG matched: {data_summary['n_hvg_matched']} (requested {data_summary['n_hvg_requested']}, missing {data_summary['n_hvg_missing']})

## Primary Gate (mean aggregation)
- Pass at least one basis model: **{pass_basis}**
- Pass at least one classical model: **{pass_classical}**
- Proceed to single-subject stage: **{proceed_single_subject}**

## LORO CV (mean aggregation)
```text
{_fmt_df(loro_mean.sort_values('pearson_r', ascending=False), 30)}
```

## Distribution Diagnostics (mean aggregation, harmonized)
```text
{_fmt_df(dist_mean.sort_values('jsd_median', ascending=True), 30)}
```

## Ranking (mean aggregation)
```text
{_fmt_df(rank_mean.sort_values('overall_rank_score', ascending=True), 30)}
```

## Outputs
- Tables: `out/population_extrapolation_basis/tables/`
- Figures: `out/population_extrapolation_basis/figures/`
- Summary JSON: `out/population_extrapolation_basis/summary.json`
"""
    (out_root / "report.md").write_text(report)

    print("Completed population-level extrapolation analysis.")
    print("Outputs:", out_root)


if __name__ == "__main__":
    main()

