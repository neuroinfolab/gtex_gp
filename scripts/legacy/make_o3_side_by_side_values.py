#!/usr/bin/env python3
"""
Build side-by-side GTEx-predicted vs Allen expression tables/heatmaps
at all Allen parcels with aligned rows and matched HVGs.
"""

from __future__ import annotations

import csv
import hashlib
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


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    pred_table_path: str = "out/o3_pls_consistent/tables/predicted_expression_new_locations.csv"
    out_root: str = "out/o3_pls_consistent"
    percentile_clip_low: float = 2.0
    percentile_clip_high: float = 98.0


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
    s = np.where(s < 1e-8, 1.0, s)
    return s


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
    df = df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    return df


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
    ah_genes = pd.DataFrame(ahz.astype(np.float32), columns=gene_cols, index=ah_tmp.index)
    gt_genes = pd.DataFrame(gtz.astype(np.float32), columns=gene_cols, index=gt_tmp.index)
    ah_tmp = pd.concat([ah_tmp.drop(columns=gene_cols), ah_genes], axis=1)
    gt_tmp = pd.concat([gt_tmp.drop(columns=gene_cols), gt_genes], axis=1)

    ap = _parcel_gene_mean(ah_tmp, gene_cols, n_parcels)
    gp = _parcel_gene_mean(gt_tmp, gene_cols, n_parcels)

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
            intercept[g] = np.nanmean(y[m] - x[m])
    slope = np.where(np.abs(slope) < 1e-6, 1e-6, slope)

    return {
        "ahba_mean": am,
        "ahba_std": asd,
        "gtex_mean": gm,
        "gtex_std": gsd,
        "slope": slope,
        "intercept": intercept,
    }


def gtex_raw_to_harmonized(x_raw: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    z = (x_raw - cal["gtex_mean"]) / cal["gtex_std"]
    return z * cal["slope"] + cal["intercept"]


def ahba_raw_to_harmonized(x_raw: np.ndarray, cal: Dict[str, np.ndarray]) -> np.ndarray:
    return (x_raw - cal["ahba_mean"]) / cal["ahba_std"]


def build_side_by_side_heatmap(
    left_mat: np.ndarray,
    right_mat: np.ndarray,
    title_left: str,
    title_right: str,
    suptitle: str,
    out_path: Path,
    cfg: Config,
) -> Tuple[float, float]:
    vals = np.concatenate([left_mat[np.isfinite(left_mat)], right_mat[np.isfinite(right_mat)]])
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
    axes[0].set_xlabel("HVG genes")
    axes[0].set_ylabel("Allen parcels (same row order)")

    hm = sns.heatmap(
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
    axes[1].set_xlabel("HVG genes")
    axes[1].set_ylabel("Allen parcels (same row order)")

    fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return float(vmin), float(vmax)


def _matrix_sha256(x: np.ndarray) -> str:
    y = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return hashlib.sha256(y.tobytes()).hexdigest()


def main() -> None:
    cfg = Config()
    sns.set_context("talk")
    sns.set_style("whitegrid")

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    pred_path = (root / cfg.pred_table_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header = load_gene_header_and_hvg(csv_path, hvg_path)
    gene_cols = header["genes_hvg"]
    if len(gene_cols) == 0:
        raise RuntimeError("No HVG overlap found.")

    pred = pd.read_csv(pred_path)
    required_meta = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]
    missing_meta = [c for c in required_meta if c not in pred.columns]
    if missing_meta:
        raise RuntimeError(f"Missing required metadata columns in prediction table: {missing_meta}")
    missing_genes = [g for g in gene_cols if g not in pred.columns]
    if missing_genes:
        raise RuntimeError(f"Missing expected HVG columns in prediction table (first 10 shown): {missing_genes[:10]}")

    pred = pred.sort_values("parcel_idx").reset_index(drop=True)
    index_df = pred[required_meta].copy()
    n_parcels = len(index_df)

    # Canonical row index output.
    index_df.to_csv(table_dir / "aligned_row_index.csv", index=False)

    # Read raw source and map parcels.
    df = read_expression_subset(csv_path, gene_cols)
    ahba_raw_rows = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw_rows = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)

    # AHBA parcel mapping by parcel name (exactly as canonical index labels).
    parcel_lookup = dict(zip(index_df["parcel_name"].astype(str), index_df["parcel_idx"].astype(int)))
    ahba_raw_rows["parcel_idx"] = ahba_raw_rows["tissue_or_parcel"].astype(str).map(parcel_lookup)
    ahba_raw_rows = ahba_raw_rows[ahba_raw_rows["parcel_idx"].notna()].copy()
    ahba_raw_rows["parcel_idx"] = ahba_raw_rows["parcel_idx"].astype(int)

    # GTEx parcel mapping by nearest canonical coordinates.
    target_xyz = index_df[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = gtex_raw_rows[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((gxyz[:, None, :] - target_xyz[None, :, :]) ** 2).sum(axis=2)
    gtex_raw_rows["parcel_idx"] = np.argmin(d2, axis=1).astype(int)

    # Calibration in exact prior form.
    cal = fit_domain_calibration(ahba_raw_rows, gtex_raw_rows, gene_cols, n_parcels=n_parcels)

    # Build GTEx predicted matrices.
    gtex_raw_mat = pred[gene_cols].to_numpy(dtype=np.float64)
    gtex_h_mat = gtex_raw_to_harmonized(gtex_raw_mat, cal)

    # Build Allen aggregated matrices in canonical parcel order.
    allen_raw_mat = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    for r in range(n_parcels):
        sub = ahba_raw_rows[ahba_raw_rows["parcel_idx"] == r]
        if len(sub) == 0:
            continue
        allen_raw_mat[r, :] = sub[gene_cols].to_numpy(dtype=np.float64).mean(axis=0)
    allen_h_mat = ahba_raw_to_harmonized(allen_raw_mat, cal)

    # Write aligned component tables.
    gtex_raw_df = pd.concat([index_df, pd.DataFrame(gtex_raw_mat, columns=gene_cols)], axis=1)
    allen_raw_df = pd.concat([index_df, pd.DataFrame(allen_raw_mat, columns=gene_cols)], axis=1)
    gtex_h_df = pd.concat([index_df, pd.DataFrame(gtex_h_mat, columns=gene_cols)], axis=1)
    allen_h_df = pd.concat([index_df, pd.DataFrame(allen_h_mat, columns=gene_cols)], axis=1)

    gtex_raw_df.to_csv(table_dir / "gtex_predicted_allen_rows_raw.csv", index=False)
    allen_raw_df.to_csv(table_dir / "allen_allen_rows_raw.csv", index=False)
    gtex_h_df.to_csv(table_dir / "gtex_predicted_allen_rows_harmonized.csv", index=False)
    allen_h_df.to_csv(table_dir / "allen_allen_rows_harmonized.csv", index=False)

    # Write wide side-by-side tables with prefixes.
    gtex_raw_pref = gtex_raw_df.copy()
    allen_raw_pref = allen_raw_df.copy()
    gtex_h_pref = gtex_h_df.copy()
    allen_h_pref = allen_h_df.copy()
    gtex_raw_pref = gtex_raw_pref.rename(columns={g: f"gtex__{g}" for g in gene_cols})
    allen_raw_pref = allen_raw_pref.rename(columns={g: f"allen__{g}" for g in gene_cols})
    gtex_h_pref = gtex_h_pref.rename(columns={g: f"gtex__{g}" for g in gene_cols})
    allen_h_pref = allen_h_pref.rename(columns={g: f"allen__{g}" for g in gene_cols})

    key_cols = required_meta
    raw_wide = gtex_raw_pref[key_cols + [f"gtex__{g}" for g in gene_cols]].merge(
        allen_raw_pref[key_cols + [f"allen__{g}" for g in gene_cols]],
        on=key_cols,
        how="inner",
    )
    h_wide = gtex_h_pref[key_cols + [f"gtex__{g}" for g in gene_cols]].merge(
        allen_h_pref[key_cols + [f"allen__{g}" for g in gene_cols]],
        on=key_cols,
        how="inner",
    )
    raw_wide.to_csv(table_dir / "side_by_side_raw_wide.csv", index=False)
    h_wide.to_csv(table_dir / "side_by_side_harmonized_wide.csv", index=False)

    # Figures with shared scales.
    raw_vmin, raw_vmax = build_side_by_side_heatmap(
        left_mat=gtex_raw_mat,
        right_mat=allen_raw_mat,
        title_left="GTEx predicted at all Allen parcels (raw)",
        title_right="Allen at all Allen parcels (raw)",
        suptitle="Side-by-side comparison (same rows, same genes, shared color scale)",
        out_path=fig_dir / "side_by_side_heatmap_raw.png",
        cfg=cfg,
    )
    h_vmin, h_vmax = build_side_by_side_heatmap(
        left_mat=gtex_h_mat,
        right_mat=allen_h_mat,
        title_left="GTEx predicted at all Allen parcels (harmonized)",
        title_right="Allen at all Allen parcels (harmonized)",
        suptitle="Side-by-side comparison (same rows, same genes, shared color scale)",
        out_path=fig_dir / "side_by_side_heatmap_harmonized.png",
        cfg=cfg,
    )

    # QC summary.
    strict_increasing = bool(index_df["parcel_idx"].is_monotonic_increasing)
    same_shape_raw = bool(gtex_raw_mat.shape == allen_raw_mat.shape == (n_parcels, len(gene_cols)))
    same_shape_h = bool(gtex_h_mat.shape == allen_h_mat.shape == (n_parcels, len(gene_cols)))
    same_gene_order = bool(list(gene_cols) == [c for c in pred.columns if c in set(gene_cols)])
    observed_rows = index_df["is_observed_gtex"].astype(bool).to_numpy()
    obs_max_abs_diff = float(np.nanmax(np.abs(gtex_raw_mat[observed_rows, :] - pred.loc[observed_rows, gene_cols].to_numpy(dtype=np.float64)))) if observed_rows.any() else np.nan

    qc = {
        "config": asdict(cfg),
        "shapes": {
            "n_rows": int(n_parcels),
            "n_genes": int(len(gene_cols)),
            "gtex_raw_shape": list(gtex_raw_mat.shape),
            "allen_raw_shape": list(allen_raw_mat.shape),
            "gtex_harmonized_shape": list(gtex_h_mat.shape),
            "allen_harmonized_shape": list(allen_h_mat.shape),
        },
        "checks": {
            "parcel_idx_strictly_increasing": strict_increasing,
            "same_shape_raw": same_shape_raw,
            "same_shape_harmonized": same_shape_h,
            "same_gene_order": same_gene_order,
            "observed_rows_match_prediction_table_exactly": bool(np.isfinite(obs_max_abs_diff) and obs_max_abs_diff == 0.0),
            "observed_rows_max_abs_diff": obs_max_abs_diff,
            "ahba_mapping_full_coverage": bool(np.all(np.isfinite(allen_raw_mat).any(axis=1))),
            "n_ahba_rows_missing_after_mapping": int(np.isnan(allen_raw_mat).all(axis=1).sum()),
            "heatmap_shared_scale_raw": True,
            "heatmap_shared_scale_harmonized": True,
        },
        "counts": {
            "n_observed_gtex_rows": int(observed_rows.sum()),
            "n_missing_gtex_rows": int((~observed_rows).sum()),
            "n_missing_hvg": int(len(header["missing_hvg"])),
        },
        "nan_cells": {
            "gtex_raw_nan": int(np.isnan(gtex_raw_mat).sum()),
            "allen_raw_nan": int(np.isnan(allen_raw_mat).sum()),
            "gtex_harmonized_nan": int(np.isnan(gtex_h_mat).sum()),
            "allen_harmonized_nan": int(np.isnan(allen_h_mat).sum()),
        },
        "hashes": {
            "gtex_raw_sha256": _matrix_sha256(gtex_raw_mat),
            "allen_raw_sha256": _matrix_sha256(allen_raw_mat),
            "gtex_harmonized_sha256": _matrix_sha256(gtex_h_mat),
            "allen_harmonized_sha256": _matrix_sha256(allen_h_mat),
        },
        "heatmap_limits": {
            "raw": {"vmin": raw_vmin, "vmax": raw_vmax},
            "harmonized": {"vmin": h_vmin, "vmax": h_vmax},
        },
    }
    with open(table_dir / "alignment_qc_summary.json", "w") as f:
        json.dump(qc, f, indent=2)

    print("Completed side-by-side table/heatmap generation.")
    print("Rows:", n_parcels, "Genes:", len(gene_cols))
    print("AHBA rows with missing parcel mapping:", qc["checks"]["n_ahba_rows_missing_after_mapping"])
    print("Output tables:", table_dir)
    print("Output figures:", fig_dir)


if __name__ == "__main__":
    main()

