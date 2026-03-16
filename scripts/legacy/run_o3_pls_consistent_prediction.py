#!/usr/bin/env python3
"""
O(3)-aligned GTEx->AHBA latent space with PLS-consistent geometry
and new-location gene prediction.

Outputs under out/o3_pls_consistent/:
- tables/o3_matrix.csv
- tables/overlap_alignment_metrics.csv
- tables/internal_gtex_alignment_metrics.csv
- tables/new_location_cv_metrics.csv
- tables/predicted_expression_new_locations.csv
- figures/overlap_scatter_pre_vs_post_o3.png
- figures/gtex_internal_Tprime_vs_Uprime.png
- figures/gtex_internal_crosscov_heatmap.png
- figures/new_location_prediction_scatter_cv.png
- figures/new_location_expression_heatmap_raw.png
- summary.json
- report.md
"""

from __future__ import annotations

import csv
import json
import re
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
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/o3_pls_consistent"

    random_seed: int = 123
    n_comp: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


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
    missing = [g for g in hvg_requested if normalize_gene_name(g) not in norm_map]
    return {
        "genes_all": genes_all,
        "hvg_requested": hvg_requested,
        "genes_hvg": genes_hvg,
        "missing_hvg": missing,
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


def _parcel_gene_mean(df: pd.DataFrame, gene_cols: List[str], n_parcels: int) -> np.ndarray:
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

    am = np.nanmean(xa, axis=0)
    asd = _safe_std(xa)
    gm = np.nanmean(xg, axis=0)
    gsd = _safe_std(xg)

    ahz = (xa - am) / asd
    gtz = (xg - gm) / gsd

    ah_tmp = ahba_raw.copy()
    gt_tmp = gtex_raw.copy()
    ah_tmp_genes = pd.DataFrame(ahz.astype(np.float32), columns=gene_cols, index=ah_tmp.index)
    gt_tmp_genes = pd.DataFrame(gtz.astype(np.float32), columns=gene_cols, index=gt_tmp.index)
    ah_tmp = pd.concat([ah_tmp.drop(columns=gene_cols), ah_tmp_genes], axis=1)
    gt_tmp = pd.concat([gt_tmp.drop(columns=gene_cols), gt_tmp_genes], axis=1)

    ap = _parcel_gene_mean(ah_tmp, gene_cols, n_parcels)
    gp = _parcel_gene_mean(gt_tmp, gene_cols, n_parcels)

    slope = np.ones(len(gene_cols), dtype=np.float64)
    intercept = np.zeros(len(gene_cols), dtype=np.float64)
    for g in range(len(gene_cols)):
        x = gp[:, g]
        y = ap[:, g]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2 and np.nanstd(x[m]) > 1e-8:
            b1, b0 = np.polyfit(x[m], y[m], 1)
            slope[g] = np.clip(b1, -5.0, 5.0)
            intercept[g] = b0
        elif m.sum() >= 1:
            slope[g] = 1.0
            intercept[g] = np.nanmean(y[m] - x[m])
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
    out_h = pd.DataFrame(h.astype(np.float32), columns=gene_cols, index=out.index)
    out = pd.concat([out.drop(columns=gene_cols), out_h], axis=1)
    return out


def inverse_harmonize_gtex(xh: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    z = (xh - cal["intercept"]) / cal["slope"]
    return z * cal["gtex_std"] + cal["gtex_mean"]


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


def _canonicalize_pls_signs(T: np.ndarray, U: np.ndarray, P: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = T.copy()
    u = U.copy()
    p = P.copy()
    c = C.copy()
    nc = min(t.shape[1], u.shape[1], p.shape[1], c.shape[1])
    for k in range(nc):
        rr = np.corrcoef(t[:, k], u[:, k])[0, 1] if np.std(t[:, k]) > 0 and np.std(u[:, k]) > 0 else np.nan
        if np.isfinite(rr) and rr < 0:
            t[:, k] *= -1
            u[:, k] *= -1
            p[:, k] *= -1
            c[:, k] *= -1
    return t, u, p, c


def fit_gtex_pls(Xg_h: np.ndarray, Yg: np.ndarray, n_comp: int = 3) -> Dict[str, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xg_h)
    nc = int(min(n_comp, Xs.shape[0] - 1, Xs.shape[1], Yg.shape[1]))
    nc = max(nc, 1)
    pls = PLSRegression(n_components=nc, scale=False)
    pls.fit(Xs, Yg)

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
        "x_mean": scaler.mean_.astype(np.float64),
        "x_scale": scaler.scale_.astype(np.float64),
        "n_comp": np.int32(nc),
    }


def fit_o3_alignment(Tg_overlap: np.ndarray, Ta_overlap: np.ndarray) -> np.ndarray:
    c = Tg_overlap.T @ Ta_overlap
    u, _, vt = np.linalg.svd(c)
    r = u @ vt
    return r


def apply_pls_consistent_transform(pls_bundle: Dict[str, np.ndarray], R: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(pls_bundle)
    out["T_prime"] = pls_bundle["T"] @ R
    out["U_prime"] = pls_bundle["U"] @ R
    out["P_prime"] = pls_bundle["P"] @ R
    out["C_prime"] = pls_bundle["C"] @ R
    return out


def fit_spatial_field(coords_obs: np.ndarray, U_prime_obs: np.ndarray, config: Config) -> Dict[str, object]:
    models = []
    for k in range(U_prime_obs.shape[1]):
        try:
            rbf = RBFInterpolator(
                coords_obs,
                U_prime_obs[:, k],
                kernel="thin_plate_spline",
                smoothing=float(config.rbf_smoothing),
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
    m = ridge.coef_.T.astype(np.float64)  # (n_comp, n_comp)
    b = ridge.intercept_.astype(np.float64)  # (n_comp,)
    return {"M": m, "b": b}


def predict_at_locations(
    coords_new: np.ndarray,
    spatial_model: Dict[str, object],
    M: Dict[str, np.ndarray],
    pls_bundle_prime: Dict[str, np.ndarray],
    calibration: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    U_new_prime = _predict_spatial_field(spatial_model, coords_new)
    T_new_prime = U_new_prime @ M["M"] + M["b"]

    X_std = T_new_prime @ pls_bundle_prime["P_prime"].T
    X_h = X_std * pls_bundle_prime["x_scale"] + pls_bundle_prime["x_mean"]
    X_raw = inverse_harmonize_gtex(X_h, calibration)
    return X_h.astype(np.float64), X_raw.astype(np.float64)


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


def evaluate_internal_alignment(T_prime: np.ndarray, U_prime: np.ndarray) -> pd.DataFrame:
    bridge = fit_u_to_t_bridge(U_prime, T_prime, alpha=1e-2)
    T_hat = U_prime @ bridge["M"] + bridge["b"]

    sse = np.sum((T_prime - T_hat) ** 2)
    sst = np.sum((T_prime - T_prime.mean(axis=0)) ** 2)
    ridge_r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    try:
        nc = int(min(3, U_prime.shape[0] - 1, U_prime.shape[1], T_prime.shape[1]))
        cca = CCA(n_components=max(nc, 1))
        uc, tc = cca.fit_transform(U_prime, T_prime)
        can_corrs = []
        for i in range(uc.shape[1]):
            if np.std(uc[:, i]) > 1e-12 and np.std(tc[:, i]) > 1e-12:
                can_corrs.append(float(np.corrcoef(uc[:, i], tc[:, i])[0, 1]))
            else:
                can_corrs.append(np.nan)
        can_mean = float(np.nanmean(can_corrs))
    except Exception:
        can_corrs = [np.nan, np.nan, np.nan]
        can_mean = np.nan

    cross = np.cov(T_prime.T, U_prime.T, bias=True)[: T_prime.shape[1], T_prime.shape[1] :]
    resid = T_prime - T_hat
    cross_resid = np.cov(resid.T, U_prime.T, bias=True)[: resid.shape[1], resid.shape[1] :]
    offdiag = cross_resid - np.diag(np.diag(cross_resid))
    offdiag_fro = float(np.linalg.norm(offdiag, ord="fro"))

    row = {
        "ridge_r2_uprime_to_tprime": ridge_r2,
        "canonical_corr_c1": can_corrs[0] if len(can_corrs) > 0 else np.nan,
        "canonical_corr_c2": can_corrs[1] if len(can_corrs) > 1 else np.nan,
        "canonical_corr_c3": can_corrs[2] if len(can_corrs) > 2 else np.nan,
        "canonical_corr_mean": can_mean,
        "crosscov_fro": float(np.linalg.norm(cross, ord="fro")),
        "residual_crosscov_offdiag_fro": offdiag_fro,
    }
    return pd.DataFrame([row])


def evaluate_new_location_cv(
    Xg_h_full: np.ndarray,
    Xa_h_full: np.ndarray,
    obs_idx: np.ndarray,
    coords_full: np.ndarray,
    Ta_full: np.ndarray,
    calibration: Dict[str, np.ndarray],
    config: Config,
    anchor_override: bool = True,
) -> pd.DataFrame:
    fold_rows = []
    y_true_all = []
    y_pred_all = []
    y_base_all = []
    skipped = 0

    for hold in obs_idx:
        train_idx = np.array([i for i in obs_idx if i != hold], dtype=int)
        if len(train_idx) < 3:
            skipped += 1
            continue

        Xtrain = Xg_h_full[train_idx, :]
        Ytrain = np.c_[coords_full[train_idx, 1], coords_full[train_idx, 2], np.abs(coords_full[train_idx, 0])]
        pls_train = fit_gtex_pls(Xtrain, Ytrain, n_comp=config.n_comp)

        Tg_ov = pls_train["T"]
        Ta_ov = Ta_full[train_idx, : pls_train["T"].shape[1]]
        R = fit_o3_alignment(Tg_ov, Ta_ov)
        prime = apply_pls_consistent_transform(pls_train, R)

        spatial = fit_spatial_field(coords_full[train_idx], prime["U_prime"], config)
        bridge = fit_u_to_t_bridge(prime["U_prime"], prime["T_prime"], alpha=config.ridge_alpha_bridge)
        Xh_pred_hold, _ = predict_at_locations(
            coords_new=coords_full[[hold], :],
            spatial_model=spatial,
            M=bridge,
            pls_bundle_prime=prime,
            calibration=calibration,
        )

        x_true = Xg_h_full[hold, :].copy()
        x_pred = Xh_pred_hold[0, :].copy()
        x_base = Xa_h_full[hold, :].copy()

        y_true_all.append(x_true)
        y_pred_all.append(x_pred)
        y_base_all.append(x_base)

        fold_rows.append(
            {
                "split": "loro_fold",
                "fold_holdout_parcel_idx": int(hold),
                "pearson_r": float(stats.pearsonr(x_true, x_pred).statistic) if np.std(x_true) > 0 and np.std(x_pred) > 0 else np.nan,
                "spearman_rho": float(stats.spearmanr(x_true, x_pred).statistic),
                "rmse": float(np.sqrt(np.mean((x_true - x_pred) ** 2))),
                "mae": float(np.mean(np.abs(x_true - x_pred))),
                "medae": float(np.median(np.abs(x_true - x_pred))),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    if len(y_true_all) == 0:
        summary = pd.DataFrame(
            [
                {
                    "split": "loro_summary",
                    "n_folds": 0,
                    "n_skipped_folds": int(skipped),
                    "n_points": 0,
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "medae": np.nan,
                    "baseline_rmse": np.nan,
                }
            ]
        )
        return pd.concat([fold_df, summary], ignore_index=True)

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    yb = np.concatenate(y_base_all)

    summary = pd.DataFrame(
        [
            {
                "split": "loro_summary",
                "n_folds": int(len(y_true_all)),
                "n_skipped_folds": int(skipped),
                "n_points": int(len(yt)),
                "pearson_r": float(stats.pearsonr(yt, yp).statistic) if np.std(yt) > 0 and np.std(yp) > 0 else np.nan,
                "spearman_rho": float(stats.spearmanr(yt, yp).statistic),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "mae": float(np.mean(np.abs(yt - yp))),
                "medae": float(np.median(np.abs(yt - yp))),
                "baseline_rmse": float(np.sqrt(np.mean((yt - yb) ** 2))),
            }
        ]
    )
    return pd.concat([fold_df, summary], ignore_index=True)


def make_overlap_scatter(
    Ta: np.ndarray,
    Tg_pre: np.ndarray,
    Tg_post: np.ndarray,
    out_path: Path,
) -> None:
    nc = Ta.shape[1]
    fig, axes = plt.subplots(nc, 2, figsize=(10, 4 * nc), constrained_layout=True)
    if nc == 1:
        axes = np.array([axes])
    for i in range(nc):
        for j, (name, pred) in enumerate([("Pre", Tg_pre), ("Post O3", Tg_post)]):
            ax = axes[i, j]
            y = Ta[:, i]
            x = pred[:, i]
            ax.scatter(y, x, s=45, alpha=0.85)
            lo, hi = min(y.min(), x.min()), max(y.max(), x.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            r = np.corrcoef(y, x)[0, 1] if np.std(y) > 1e-12 and np.std(x) > 1e-12 else np.nan
            ax.set_title(f"C{i+1} {name} (r={r:.3f})")
            ax.set_xlabel("AHBA overlap score")
            ax.set_ylabel("GTEx overlap score")
    fig.suptitle("Overlap score alignment: pre vs post O(3)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_internal_scatter(Tp: np.ndarray, Up: np.ndarray, out_path: Path) -> None:
    bridge = fit_u_to_t_bridge(Up, Tp, alpha=1e-2)
    That = Up @ bridge["M"] + bridge["b"]
    nc = Tp.shape[1]
    fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 4), constrained_layout=True)
    if nc == 1:
        axes = [axes]
    for i in range(nc):
        ax = axes[i]
        y = Tp[:, i]
        x = That[:, i]
        ax.scatter(y, x, s=50, alpha=0.9)
        lo, hi = min(y.min(), x.min()), max(y.max(), x.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = np.corrcoef(y, x)[0, 1] if np.std(y) > 1e-12 and np.std(x) > 1e-12 else np.nan
        ax.set_title(f"T'_C{i+1} vs pred(U') (r={r:.3f})")
        ax.set_xlabel("T'_true")
        ax.set_ylabel("T'_pred")
    fig.suptitle("Within-GTEx transformed latent coherence")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_crosscov_heatmap(Tp: np.ndarray, Up: np.ndarray, out_path: Path) -> None:
    bridge = fit_u_to_t_bridge(Up, Tp, alpha=1e-2)
    That = Up @ bridge["M"] + bridge["b"]
    cross = np.cov(Tp.T, Up.T, bias=True)[: Tp.shape[1], Tp.shape[1] :]
    resid = Tp - That
    cross_res = np.cov(resid.T, Up.T, bias=True)[: resid.shape[1], resid.shape[1] :]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sns.heatmap(cross, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0], cbar=False, square=True)
    axes[0].set_title("Cov(T', U')")
    axes[0].set_xlabel("U' components")
    axes[0].set_ylabel("T' components")
    sns.heatmap(cross_res, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1], cbar=False, square=True)
    axes[1].set_title("Cov(T'-T_hat, U')")
    axes[1].set_xlabel("U' components")
    axes[1].set_ylabel("Residual components")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_cv_scatter(true_vec: np.ndarray, pred_vec: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(true_vec, pred_vec, s=8, alpha=0.35)
    lo, hi = min(true_vec.min(), pred_vec.min()), max(true_vec.max(), pred_vec.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    r = np.corrcoef(true_vec, pred_vec)[0, 1] if np.std(true_vec) > 1e-12 and np.std(pred_vec) > 1e-12 else np.nan
    rmse = np.sqrt(np.mean((true_vec - pred_vec) ** 2))
    ax.set_title(f"LORO new-location prediction (r={r:.3f}, RMSE={rmse:.3f})")
    ax.set_xlabel("True GTEx harmonized expression")
    ax.set_ylabel("Predicted harmonized expression")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_prediction_heatmap_raw(
    x_raw_full: np.ndarray,
    region_names: List[str],
    gene_names: List[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    lo, hi = np.nanpercentile(x_raw_full, [2, 98])
    sns.heatmap(
        x_raw_full,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=lo,
        vmax=hi,
        xticklabels=gene_names,
        yticklabels=region_names,
        cbar_kws={"label": "raw expression"},
    )
    ax.set_title("Predicted GTEx expression at all AHBA target parcels (raw, anchors enforced)")
    ax.set_xlabel("HVG genes")
    ax.set_ylabel("Target parcels")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = Config()
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

    hdr = load_gene_header_and_hvg(csv_path, hvg_path)
    genes = hdr["genes_hvg"]
    if len(genes) == 0:
        raise RuntimeError("No HVG overlap found.")

    df = read_expression_subset(csv_path, genes)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEx data empty.")

    target = build_target_parcels(ahba_raw)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    coords_full = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    region_names = target["tissue_or_parcel"].tolist()

    cal = fit_domain_calibration(ahba_raw, gtex_raw, genes, n_parcels=len(target))
    ahba_h = apply_harmonization(ahba_raw, genes, cal, "AHBA")
    gtex_h = apply_harmonization(gtex_raw, genes, cal, "GTEX")

    # Population region matrices.
    Xa_h_full, _ = build_region_matrix(ahba_h, genes, target, agg="mean")
    Xg_h_full, obs_mask = build_region_matrix(gtex_h, genes, target, agg="mean")
    Xa_raw_full, _ = build_region_matrix(ahba_raw, genes, target, agg="mean")
    Xg_raw_full, _ = build_region_matrix(gtex_raw, genes, target, agg="mean")
    obs_idx = np.where(obs_mask)[0]
    miss_idx = np.array([i for i in range(len(target)) if i not in set(obs_idx.tolist())], dtype=int)

    # 1) Fit latent models.
    Yg_obs = np.c_[coords_full[obs_idx, 1], coords_full[obs_idx, 2], np.abs(coords_full[obs_idx, 0])]
    Ya_all = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
    pls_g = fit_gtex_pls(Xg_h_full[obs_idx, :], Yg_obs, n_comp=cfg.n_comp)
    pls_a = fit_gtex_pls(Xa_h_full, Ya_all, n_comp=cfg.n_comp)

    Tg_overlap = pls_g["T"][:, : cfg.n_comp]
    Ta_overlap = pls_a["T"][obs_idx, : cfg.n_comp]
    R = fit_o3_alignment(Tg_overlap, Ta_overlap)

    # Save O3 matrix diagnostics.
    o3_row = {
        "det": float(np.linalg.det(R)),
        "orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(R.shape[0]), ord="fro")),
    }
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            o3_row[f"r_{i+1}{j+1}"] = float(R[i, j])
    pd.DataFrame([o3_row]).to_csv(table_dir / "o3_matrix.csv", index=False)

    # 2) Apply PLS-consistent transform.
    prime = apply_pls_consistent_transform(pls_g, R)
    Tg_prime = prime["T_prime"]
    Ug_prime = prime["U_prime"]

    # Decoder equivalence check.
    x_std_a = Tg_prime @ prime["P_prime"].T
    x_std_b = (Tg_prime @ R.T) @ pls_g["P"].T
    dec_equiv_rel = float(np.linalg.norm(x_std_a - x_std_b, ord="fro") / max(np.linalg.norm(x_std_a, ord="fro"), 1e-12))

    # Overlap alignment metrics.
    pre_metrics = evaluate_alignment_matrix(Ta_overlap, Tg_overlap)
    post_metrics = evaluate_alignment_matrix(Ta_overlap, Tg_prime)
    overlap_df = pd.DataFrame(
        [
            {"stage": "pre_o3", **pre_metrics},
            {"stage": "post_o3", **post_metrics},
        ]
    )
    overlap_df.to_csv(table_dir / "overlap_alignment_metrics.csv", index=False)

    # 3) Internal GTEx alignment diagnostics in transformed basis.
    internal_df = evaluate_internal_alignment(Tg_prime, Ug_prime)
    internal_df.insert(0, "decoder_equivalence_relative_error", dec_equiv_rel)
    internal_df.to_csv(table_dir / "internal_gtex_alignment_metrics.csv", index=False)

    # 4) Build predictor for missing locations.
    spatial_model = fit_spatial_field(coords_full[obs_idx], Ug_prime, cfg)
    bridge = fit_u_to_t_bridge(Ug_prime, Tg_prime, alpha=cfg.ridge_alpha_bridge)

    # Predict for missing and then compose full matrix with anchor overwrite.
    Xmiss_h, Xmiss_raw = predict_at_locations(
        coords_new=coords_full[miss_idx, :],
        spatial_model=spatial_model,
        M=bridge,
        pls_bundle_prime=prime,
        calibration=cal,
    )
    Xfull_h = np.full_like(Xa_h_full, np.nan, dtype=np.float64)
    Xfull_raw = np.full_like(Xa_h_full, np.nan, dtype=np.float64)
    Xfull_h[miss_idx, :] = Xmiss_h
    Xfull_raw[miss_idx, :] = Xmiss_raw
    # Anchor override at observed GTEx parcels.
    Xfull_h[obs_idx, :] = Xg_h_full[obs_idx, :]
    Xfull_raw[obs_idx, :] = Xg_raw_full[obs_idx, :]

    pred_df = pd.DataFrame(Xfull_raw, columns=genes)
    pred_df.insert(0, "is_observed_gtex", pd.Series([bool(i in set(obs_idx.tolist())) for i in range(len(target))]))
    pred_df.insert(0, "coord_z", target["coord_z"].to_numpy())
    pred_df.insert(0, "coord_y", target["coord_y"].to_numpy())
    pred_df.insert(0, "coord_x", target["coord_x"].to_numpy())
    pred_df.insert(0, "parcel_name", target["tissue_or_parcel"].to_numpy())
    pred_df.insert(0, "parcel_idx", target["parcel_idx"].to_numpy(dtype=np.int32))
    pred_df.to_csv(table_dir / "predicted_expression_new_locations.csv", index=False)

    # 5) New-location CV.
    cv_df = evaluate_new_location_cv(
        Xg_h_full=Xg_h_full,
        Xa_h_full=Xa_h_full,
        obs_idx=obs_idx,
        coords_full=coords_full,
        Ta_full=pls_a["T"],
        calibration=cal,
        config=cfg,
        anchor_override=True,
    )
    cv_df.to_csv(table_dir / "new_location_cv_metrics.csv", index=False)

    # Build vectors for CV scatter (from per-fold rows, regenerate for clean plot).
    true_vec = []
    pred_vec = []
    for hold in obs_idx:
        train_idx = np.array([i for i in obs_idx if i != hold], dtype=int)
        if len(train_idx) < 3:
            continue
        Xtrain = Xg_h_full[train_idx, :]
        Ytrain = np.c_[coords_full[train_idx, 1], coords_full[train_idx, 2], np.abs(coords_full[train_idx, 0])]
        pls_train = fit_gtex_pls(Xtrain, Ytrain, n_comp=cfg.n_comp)
        Rf = fit_o3_alignment(pls_train["T"], pls_a["T"][train_idx, : pls_train["T"].shape[1]])
        prime_f = apply_pls_consistent_transform(pls_train, Rf)
        spatial_f = fit_spatial_field(coords_full[train_idx], prime_f["U_prime"], cfg)
        bridge_f = fit_u_to_t_bridge(prime_f["U_prime"], prime_f["T_prime"], alpha=cfg.ridge_alpha_bridge)
        Xh_hold, _ = predict_at_locations(
            coords_new=coords_full[[hold], :],
            spatial_model=spatial_f,
            M=bridge_f,
            pls_bundle_prime=prime_f,
            calibration=cal,
        )
        true_vec.append(Xg_h_full[hold, :].copy())
        pred_vec.append(Xh_hold[0, :].copy())
    if len(true_vec):
        true_stack = np.concatenate(true_vec)
        pred_stack = np.concatenate(pred_vec)
    else:
        true_stack = np.array([0.0])
        pred_stack = np.array([0.0])

    # 6) Figures.
    make_overlap_scatter(
        Ta=Ta_overlap,
        Tg_pre=Tg_overlap,
        Tg_post=Tg_prime,
        out_path=fig_dir / "overlap_scatter_pre_vs_post_o3.png",
    )
    make_internal_scatter(
        Tp=Tg_prime,
        Up=Ug_prime,
        out_path=fig_dir / "gtex_internal_Tprime_vs_Uprime.png",
    )
    make_crosscov_heatmap(
        Tp=Tg_prime,
        Up=Ug_prime,
        out_path=fig_dir / "gtex_internal_crosscov_heatmap.png",
    )
    make_cv_scatter(
        true_vec=true_stack,
        pred_vec=pred_stack,
        out_path=fig_dir / "new_location_prediction_scatter_cv.png",
    )
    make_prediction_heatmap_raw(
        x_raw_full=Xfull_raw,
        region_names=region_names,
        gene_names=genes,
        out_path=fig_dir / "new_location_expression_heatmap_raw.png",
    )

    # 7) Summary + report with recommendation.
    cv_summary = cv_df[cv_df["split"] == "loro_summary"].copy()
    if len(cv_summary):
        cv_row = cv_summary.iloc[0]
        rmse = float(cv_row["rmse"])
        base_rmse = float(cv_row["baseline_rmse"])
        pear = float(cv_row["pearson_r"])
    else:
        rmse = np.nan
        base_rmse = np.nan
        pear = np.nan
    better_than_baseline = bool(np.isfinite(rmse) and np.isfinite(base_rmse) and rmse < base_rmse)
    recommend_subject_level = bool(np.isfinite(pear) and pear >= 0.50 and better_than_baseline)

    data_summary = {
        "ahba_rows": int(len(ahba_raw)),
        "gtex_rows": int(len(gtex_raw)),
        "target_parcels": int(len(target)),
        "gtex_observed_target_parcels": int(len(obs_idx)),
        "gtex_missing_target_parcels": int(len(miss_idx)),
        "n_hvg_requested": int(len(hdr["hvg_requested"])),
        "n_hvg_matched": int(len(genes)),
        "n_hvg_missing": int(len(hdr["missing_hvg"])),
    }
    summary = {
        "config": asdict(cfg),
        "data_summary": data_summary,
        "o3": o3_row,
        "decoder_equivalence_relative_error": dec_equiv_rel,
        "overlap_alignment": overlap_df.to_dict(orient="records"),
        "internal_alignment": internal_df.to_dict(orient="records"),
        "new_location_cv_summary": cv_summary.to_dict(orient="records"),
        "recommendation": {
            "better_than_baseline_rmse": better_than_baseline,
            "pearson_threshold": 0.50,
            "recommend_subject_level_rollout": recommend_subject_level,
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = f"""# O(3) PLS-Consistent GTEx->AHBA Prediction Report

## Data
- AHBA rows: {data_summary['ahba_rows']}
- GTEx rows: {data_summary['gtex_rows']}
- Target parcels: {data_summary['target_parcels']}
- Observed GTEx target parcels: {data_summary['gtex_observed_target_parcels']}
- Missing target parcels predicted: {data_summary['gtex_missing_target_parcels']}
- HVG matched: {data_summary['n_hvg_matched']} (requested {data_summary['n_hvg_requested']}, missing {data_summary['n_hvg_missing']})

## O(3) Matrix
- det(R): {o3_row['det']:.6f}
- orthogonality error ||R^TR-I||_F: {o3_row['orthogonality_error']:.3e}

## Overlap Alignment (GTEx genetic scores vs AHBA genetic scores)
```text
{overlap_df.to_string(index=False)}
```

## Internal GTEx Coherence in Transformed Basis
```text
{internal_df.to_string(index=False)}
```

## New-Location LORO CV
```text
{cv_df.to_string(index=False)}
```

## Decoder Equivalence Check
- Relative error: {dec_equiv_rel:.3e}
- Criterion target: < 1e-8

## Recommendation
- Better RMSE than AHBA-atlas baseline: {better_than_baseline}
- Subject-level rollout recommended: {recommend_subject_level}
"""
    (out_root / "report.md").write_text(report)

    print("Completed O(3) PLS-consistent prediction analysis.")
    print("Outputs:", out_root)


if __name__ == "__main__":
    main()
