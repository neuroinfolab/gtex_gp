#!/usr/bin/env python3
"""
Build side-by-side heatmap panels with shared color scale across models.

Outputs:
- out/population_extrapolation_basis/figures/heatmap_panel_raw_models_<agg>_sharedscale.png
- out/population_extrapolation_basis/figures/heatmap_panel_harmonized_models_<agg>_sharedscale.png
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    pred_dir: str = "out/population_extrapolation_basis/tables"
    fig_dir: str = "out/population_extrapolation_basis/figures"
    random_seed: int = 123
    n_components: int = 3


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
    sd = np.nanstd(x, axis=0, ddof=0)
    return np.where(sd < 1e-8, 1.0, sd)


def load_header_and_hvg(csv_path: Path, hvg_path: Path) -> List[str]:
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    genes_all = header[6:]
    hvg = [ln.strip() for ln in hvg_path.read_text().splitlines() if ln.strip()]
    norm_map = {normalize_gene_name(g): g for g in genes_all}
    genes = [norm_map[normalize_gene_name(g)] for g in hvg if normalize_gene_name(g) in norm_map]
    return genes


def read_expression_subset(csv_path: Path, gene_cols: List[str]) -> pd.DataFrame:
    usecols = ["subject", "dataset", "tissue_or_parcel", "coordinates"] + list(gene_cols)
    dtype_map = {"subject": "string", "dataset": "string", "tissue_or_parcel": "string", "coordinates": "string"}
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


def map_gtex_to_target(gtex_df: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    out = gtex_df.copy()
    txyz = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = out[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((gxyz[:, None, :] - txyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    out["parcel_idx"] = idx.astype(np.int32)
    return out


def fit_domain_calibration(ahba_raw: pd.DataFrame, gtex_raw: pd.DataFrame, gene_cols: List[str], n_parcels: int) -> Dict[str, np.ndarray]:
    xa = ahba_raw[gene_cols].to_numpy(dtype=np.float64)
    xg = gtex_raw[gene_cols].to_numpy(dtype=np.float64)
    am, asd = np.nanmean(xa, axis=0), _safe_std(xa)
    gm, gsd = np.nanmean(xg, axis=0), _safe_std(xg)

    ahz = (xa - am) / asd
    gtz = (xg - gm) / gsd
    ah = ahba_raw.copy()
    gt = gtex_raw.copy()
    ah.loc[:, gene_cols] = ahz
    gt.loc[:, gene_cols] = gtz

    def parcel_means(df: pd.DataFrame) -> np.ndarray:
        out = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
        for r in range(n_parcels):
            sub = df[df["parcel_idx"] == r]
            if len(sub):
                out[r, :] = sub[gene_cols].to_numpy(dtype=np.float64).mean(axis=0)
        return out

    ap = parcel_means(ah)
    gp = parcel_means(gt)
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
            intercept[g] = np.nanmean(y[m] - x[m])
    slope = np.where(np.abs(slope) < 1e-6, 1e-6, slope)
    return {"ahba_mean": am, "ahba_std": asd, "gtex_mean": gm, "gtex_std": gsd, "slope": slope, "intercept": intercept}


def harmonize(df: pd.DataFrame, gene_cols: List[str], cal: Dict[str, np.ndarray], dataset: str) -> pd.DataFrame:
    x = df[gene_cols].to_numpy(dtype=np.float64)
    if dataset == "AHBA":
        h = (x - cal["ahba_mean"]) / cal["ahba_std"]
    else:
        z = (x - cal["gtex_mean"]) / cal["gtex_std"]
        h = z * cal["slope"] + cal["intercept"]
    out = df.copy()
    out.loc[:, gene_cols] = h
    return out


def raw_to_harmonized_gtex(x_raw: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    z = (x_raw - cal["gtex_mean"]) / cal["gtex_std"]
    return z * cal["slope"] + cal["intercept"]


def build_region_matrix(df: pd.DataFrame, gene_cols: List[str], target: pd.DataFrame, agg: str) -> Tuple[np.ndarray, np.ndarray]:
    r = len(target)
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


def compute_biology_ordering(ahba_h: np.ndarray, coords: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.c_[coords[:, 1], coords[:, 2], np.abs(coords[:, 0])]
    xs = StandardScaler().fit_transform(ahba_h)
    nc = int(min(n_components, ahba_h.shape[0] - 1, ahba_h.shape[1]))
    pls = PLSRegression(n_components=max(1, nc), scale=False)
    pls.fit(xs, y)
    c1_scores = pls.x_scores_[:, 0]
    c1_load = pls.x_loadings_[:, 0]
    region_order = np.argsort(c1_scores)
    gene_order = np.lexsort((-np.abs(c1_load), -c1_load))
    return region_order.astype(int), gene_order.astype(int), c1_scores


def load_prediction_csv(path: Path, gene_cols: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    if "parcel_idx" in df.columns:
        df = df.sort_values("parcel_idx")
    return df[gene_cols].to_numpy(dtype=np.float64)


def _prepare_panel_mats(
    agg: str,
    region_order: np.ndarray,
    gene_order: np.ndarray,
    ahba_raw: np.ndarray,
    gtex_raw: np.ndarray,
    ahba_h: np.ndarray,
    gtex_h: np.ndarray,
    obs_idx: np.ndarray,
    pred_dir: Path,
    gene_cols: List[str],
    cal: Dict[str, np.ndarray],
) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    models = ["method1", "method2", "method3", "method4", "basis_so3", "basis_o3", "basis_affine"]

    obs_raw_full = np.full_like(ahba_raw, np.nan)
    obs_raw_full[obs_idx, :] = gtex_raw[obs_idx, :]
    obs_h_full = np.full_like(ahba_h, np.nan)
    obs_h_full[obs_idx, :] = gtex_h[obs_idx, :]

    titles_raw = ["AHBA raw", "GTEx observed raw"]
    mats_raw = [ahba_raw, obs_raw_full]
    titles_h = ["AHBA harmonized", "GTEx observed harmonized"]
    mats_h = [ahba_h, obs_h_full]

    for m in models:
        p = pred_dir / f"population_predictions_{m}_{agg}.csv"
        x_raw = load_prediction_csv(p, gene_cols)
        x_h = raw_to_harmonized_gtex(x_raw, cal)
        titles_raw.append(f"{m} raw")
        mats_raw.append(x_raw)
        titles_h.append(f"{m} harmonized")
        mats_h.append(x_h)

    mats_raw_ord = [m[np.ix_(region_order, gene_order)] for m in mats_raw]
    mats_h_ord = [m[np.ix_(region_order, gene_order)] for m in mats_h]
    return titles_raw, mats_raw_ord, titles_h, mats_h_ord


def _shared_limits(mats: List[np.ndarray]) -> Tuple[float, float]:
    vals = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
    lo, hi = np.nanpercentile(vals, [2, 98])
    return float(lo), float(hi)


def plot_panel(titles: List[str], mats: List[np.ndarray], out_path: Path, main_title: str) -> None:
    n = len(mats)
    cols = 3
    rows = int(np.ceil(n / cols))
    vmin, vmax = _shared_limits(mats)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 5.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        sns.heatmap(
            mats[i],
            ax=ax,
            cmap="coolwarm",
            center=0.0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
        )
        ax.set_title(titles[i], fontsize=11)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01)
    cbar.set_label("expression (shared scale)")
    fig.suptitle(main_title, fontsize=16)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    sns.set_context("talk")
    sns.set_style("whitegrid")

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    pred_dir = (root / cfg.pred_dir).resolve()
    fig_dir = (root / cfg.fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    genes = load_header_and_hvg(csv_path, hvg_path)
    df = read_expression_subset(csv_path, genes)
    ahba = df[df["dataset_upper"] == "AHBA"].copy()
    gtex = df[df["dataset_upper"] == "GTEX"].copy()
    target = build_target_parcels(ahba)
    map_idx = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba["parcel_idx"] = ahba["tissue_or_parcel"].map(map_idx).astype(np.int32)
    gtex = map_gtex_to_target(gtex, target)

    cal = fit_domain_calibration(ahba, gtex, genes, n_parcels=len(target))
    ahba_h = harmonize(ahba, genes, cal, "AHBA")
    gtex_h = harmonize(gtex, genes, cal, "GTEX")

    coords = target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)

    for agg in ["mean", "median"]:
        ahba_raw_mat, _ = build_region_matrix(ahba, genes, target, agg)
        gtex_raw_mat, obs_mask = build_region_matrix(gtex, genes, target, agg)
        ahba_h_mat, _ = build_region_matrix(ahba_h, genes, target, agg)
        gtex_h_mat, _ = build_region_matrix(gtex_h, genes, target, agg)
        obs_idx = np.where(obs_mask)[0]

        region_order, gene_order, _ = compute_biology_ordering(ahba_h_mat, coords, cfg.n_components)
        titles_raw, mats_raw, titles_h, mats_h = _prepare_panel_mats(
            agg=agg,
            region_order=region_order,
            gene_order=gene_order,
            ahba_raw=ahba_raw_mat,
            gtex_raw=gtex_raw_mat,
            ahba_h=ahba_h_mat,
            gtex_h=gtex_h_mat,
            obs_idx=obs_idx,
            pred_dir=pred_dir,
            gene_cols=genes,
            cal=cal,
        )

        plot_panel(
            titles=titles_raw,
            mats=mats_raw,
            out_path=fig_dir / f"heatmap_panel_raw_models_{agg}_sharedscale.png",
            main_title=f"Population Extrapolation Comparison (raw, {agg}, shared color scale)",
        )
        plot_panel(
            titles=titles_h,
            mats=mats_h,
            out_path=fig_dir / f"heatmap_panel_harmonized_models_{agg}_sharedscale.png",
            main_title=f"Population Extrapolation Comparison (harmonized, {agg}, shared color scale)",
        )

    print("Saved shared-scale comparison panels to", fig_dir)


if __name__ == "__main__":
    main()

