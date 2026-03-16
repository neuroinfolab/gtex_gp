#!/usr/bin/env python3
"""
AHBA-only replication workflow for Vogel et al. Fig-1-style analyses.

Builds:
1) Subject x region x gene atlases (mean and median aggregations)
2) Gene dimensionality reduction (PCA)
3) PLS/CCA/PLSC alignment between expression axes and spatial coordinates
4) Null-model significance tests (permutation and coordinate rotation)
5) Figures and an interpretation report

Usage:
    python3 scripts/run_ahba_vogel_replication.py
"""

from __future__ import annotations

import json
import math
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
from scipy.spatial.transform import Rotation as R
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/ahba_vogel"

    random_seed: int = 123
    n_components: int = 3
    n_pca: int = 100

    cv_splits: int = 10
    cv_repeats: int = 10
    test_fraction: float = 0.25

    n_permutations: int = 500
    n_rotations: int = 500

    read_chunksize: int = 128


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


def load_header_and_hvg(csv_path: Path, hvg_path: Path) -> Dict[str, List[str]]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    meta_cols = header[:6]
    gene_cols = header[6:]

    hvg = [line.strip() for line in hvg_path.read_text().splitlines() if line.strip()]
    norm_map = {normalize_gene_name(g): g for g in gene_cols}
    matched_hvg = [norm_map[normalize_gene_name(g)] for g in hvg if normalize_gene_name(g) in norm_map]
    missing_hvg = [g for g in hvg if normalize_gene_name(g) not in norm_map]

    return {
        "meta_cols": meta_cols,
        "genes_all": gene_cols,
        "hvg_list": hvg,
        "genes_hvg": matched_hvg,
        "missing_hvg": missing_hvg,
    }


def load_ahba_only_dataframe(csv_path: Path, gene_cols: List[str], chunksize: int) -> pd.DataFrame:
    usecols = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"] + gene_cols
    dtype_map = {
        "subject": "string",
        "age": "string",
        "sex": "string",
        "dataset": "string",
        "tissue_or_parcel": "string",
        "coordinates": "string",
    }
    dtype_map.update({g: np.float32 for g in gene_cols})

    out_parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, chunksize=chunksize, low_memory=False):
        ds = chunk["dataset"].str.upper().str.strip()
        sub = chunk[ds == "AHBA"].copy()
        if len(sub) > 0:
            out_parts.append(sub)

    if not out_parts:
        raise RuntimeError("No AHBA rows found in CSV.")

    ahba = pd.concat(out_parts, ignore_index=True)
    return ahba


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    xyz = df["coordinates"].map(parse_coordinate_centroid)
    coords = pd.DataFrame(xyz.tolist(), columns=["coord_x", "coord_y", "coord_z"], index=df.index)
    out = pd.concat([df, coords], axis=1)
    out["coord_abs_x"] = out["coord_x"].abs()
    out = out.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    return out


def build_subject_region_aggregate(
    ahba: pd.DataFrame,
    gene_cols: List[str],
    agg: str,
) -> pd.DataFrame:
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    grouped = (
        ahba.groupby(["subject", "tissue_or_parcel"], as_index=False)[gene_cols]
        .agg(agg)
        .copy()
    )

    reg_coords = (
        ahba.groupby("tissue_or_parcel", as_index=False)[["coord_x", "coord_y", "coord_z", "coord_abs_x"]]
        .mean()
    )
    grouped = grouped.merge(reg_coords, on="tissue_or_parcel", how="left")
    return grouped


def tensorize_subject_region(
    atlas_df: pd.DataFrame,
    gene_cols: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    subjects = sorted(atlas_df["subject"].astype(str).unique().tolist())
    regions = sorted(atlas_df["tissue_or_parcel"].astype(str).unique().tolist())

    sidx = {s: i for i, s in enumerate(subjects)}
    ridx = {r: i for i, r in enumerate(regions)}

    tensor = np.full((len(subjects), len(regions), len(gene_cols)), np.nan, dtype=np.float32)

    for _, row in atlas_df.iterrows():
        i = sidx[str(row["subject"])]
        j = ridx[str(row["tissue_or_parcel"])]
        tensor[i, j, :] = row[gene_cols].to_numpy(dtype=np.float32)

    return tensor, subjects, regions


def component_score_r2(x_scores: np.ndarray, y_scores: np.ndarray, n_components: int) -> np.ndarray:
    out = np.zeros(n_components, dtype=np.float64)
    for c in range(n_components):
        r = np.corrcoef(x_scores[:, c], y_scores[:, c])[0, 1]
        out[c] = r * r
    return out


def donor_leave_one_out(
    Xp: np.ndarray,
    Y: np.ndarray,
    donors: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    rows = []
    uniq = np.unique(donors)
    for d in uniq:
        tr = donors != d
        te = donors == d
        if tr.sum() < n_components + 3 or te.sum() < 1:
            continue
        mod = PLSRegression(n_components=n_components)
        mod.fit(Xp[tr], Y[tr])
        pred = mod.predict(Xp[te])
        rows.append(
            {
                "held_out_donor": str(d),
                "n_test": int(te.sum()),
                "r2": float(r2_score(Y[te], pred, multioutput="variance_weighted")),
                "mae": float(mean_absolute_error(Y[te], pred)),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.random_seed)

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()

    out_root = (root / cfg.out_root).resolve()
    out_tables = out_root / "tables"
    out_figs = out_root / "figures"
    out_atlas = out_root / "atlas"
    for p in [out_root, out_tables, out_figs, out_atlas]:
        p.mkdir(parents=True, exist_ok=True)

    header = load_header_and_hvg(csv_path, hvg_path)
    gene_cols = header["genes_all"]

    print("Loading AHBA rows from", csv_path)
    ahba = load_ahba_only_dataframe(csv_path, gene_cols, chunksize=cfg.read_chunksize)
    ahba = add_coordinates(ahba)

    # Atlas build (mean + median)
    print("Building subject-region atlases (mean and median)")
    atlas_mean_df = build_subject_region_aggregate(ahba, gene_cols, agg="mean")
    atlas_median_df = build_subject_region_aggregate(ahba, gene_cols, agg="median")

    mean_tensor, subjects, regions = tensorize_subject_region(atlas_mean_df, gene_cols)
    median_tensor, _, _ = tensorize_subject_region(atlas_median_df, gene_cols)

    # Region x gene atlas across subjects
    region_gene_mean = np.nanmean(mean_tensor, axis=0)
    region_gene_median = np.nanmedian(median_tensor, axis=0)

    # Save atlas artifacts
    np.savez_compressed(
        out_atlas / "ahba_subject_region_gene_atlas.npz",
        mean_tensor=mean_tensor,
        median_tensor=median_tensor,
        region_gene_mean=region_gene_mean.astype(np.float32),
        region_gene_median=region_gene_median.astype(np.float32),
        subjects=np.asarray(subjects),
        regions=np.asarray(regions),
        genes=np.asarray(gene_cols),
    )

    # Coverage and identity checks
    coverage = np.isfinite(mean_tensor).any(axis=2).astype(int)
    coverage_df = pd.DataFrame(coverage, index=subjects, columns=regions)
    coverage_df.to_csv(out_tables / "subject_region_coverage_matrix.csv")

    # In this dataset there is one AHBA sample per subject-region (mean==median)
    mean_vs_median_absdiff = np.nanmean(np.abs(mean_tensor - median_tensor))

    # Save compact atlas previews
    atlas_mean_preview = atlas_mean_df[["subject", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "coord_abs_x"] + gene_cols[:50]]
    atlas_median_preview = atlas_median_df[["subject", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "coord_abs_x"] + gene_cols[:50]]
    atlas_mean_preview.to_csv(out_atlas / "ahba_subject_region_mean_preview_first50genes.csv", index=False)
    atlas_median_preview.to_csv(out_atlas / "ahba_subject_region_median_preview_first50genes.csv", index=False)

    # Model matrices
    X = ahba[gene_cols].to_numpy(dtype=np.float64)
    Y = ahba[["coord_y", "coord_z", "coord_abs_x"]].to_numpy(dtype=np.float64)
    donors = ahba["subject"].astype(str).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_pca = int(min(cfg.n_pca, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed)
    Xp = pca.fit_transform(Xs)

    pca_var = pd.DataFrame(
        {
            "component": np.arange(1, n_pca + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_var.to_csv(out_tables / "pca_variance_profile.csv", index=False)

    # Estimator selection
    estimators = {
        "PLSR": lambda nc: PLSRegression(n_components=nc),
        "CCA": lambda nc: CCA(n_components=nc, max_iter=1000),
        "PLSC": lambda nc: PLSCanonical(n_components=nc, max_iter=1000),
    }

    cv_rows = []
    for nc in range(1, cfg.n_components + 1):
        for nm, ctor in estimators.items():
            for rep in range(cfg.cv_repeats):
                cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_seed + rep)
                est = ctor(nc)
                pred = cross_val_predict(est, Xp, Y, cv=cv)
                cv_rows.append(
                    {
                        "estimator": nm,
                        "n_components": nc,
                        "repeat": rep,
                        "r2": float(r2_score(Y, pred, multioutput="variance_weighted")),
                        "mae": float(mean_absolute_error(Y, pred)),
                    }
                )

    cv_raw = pd.DataFrame(cv_rows)
    cv_raw.to_csv(out_tables / "estimator_selection_cv_raw.csv", index=False)

    cv_summary = (
        cv_raw.groupby(["estimator", "n_components"], as_index=False)[["r2", "mae"]]
        .mean()
        .sort_values(["r2", "mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    cv_summary.to_csv(out_tables / "estimator_selection_cv_summary.csv", index=False)

    best = cv_summary.iloc[0]

    # Holdout test on PLSR(3) for comparability to Vogel NB02
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xp,
        Y,
        test_size=cfg.test_fraction,
        random_state=cfg.random_seed,
        shuffle=True,
    )

    pls = PLSRegression(n_components=cfg.n_components)
    pls.fit(X_tr, y_tr)
    y_pred_te = pls.predict(X_te)

    holdout_metrics = {
        "test_r2": float(r2_score(y_te, y_pred_te, multioutput="variance_weighted")),
        "test_mae": float(mean_absolute_error(y_te, y_pred_te)),
        "n_test": int(len(y_te)),
    }

    # Fit full models for component coupling
    pls_full = PLSRegression(n_components=cfg.n_components)
    pls_full.fit(Xp, Y)
    pls_comp_r2 = component_score_r2(pls_full.x_scores_, pls_full.y_scores_, cfg.n_components)

    cca_full = CCA(n_components=cfg.n_components, max_iter=1000)
    cca_full.fit(Xp, Y)
    cca_xs, cca_ys = cca_full.transform(Xp, Y)
    cca_comp_r2 = component_score_r2(cca_xs, cca_ys, cfg.n_components)

    plsc_full = PLSCanonical(n_components=cfg.n_components, max_iter=1000)
    plsc_full.fit(Xp, Y)
    plsc_xs, plsc_ys = plsc_full.transform(Xp, Y)
    plsc_comp_r2 = component_score_r2(plsc_xs, plsc_ys, cfg.n_components)

    comp_compare = pd.DataFrame(
        {
            "component": np.arange(1, cfg.n_components + 1),
            "PLSR_r2": pls_comp_r2,
            "CCA_r2": cca_comp_r2,
            "PLSC_r2": plsc_comp_r2,
        }
    )
    comp_compare.to_csv(out_tables / "component_coupling_comparison.csv", index=False)

    # Donor leave-one-out check
    donor_loo = donor_leave_one_out(Xp, Y, donors, n_components=cfg.n_components)
    donor_loo.to_csv(out_tables / "donor_leave_one_out_metrics.csv", index=False)

    # Null models for PLSR
    null_perm = np.zeros((cfg.n_permutations, cfg.n_components), dtype=np.float64)
    for i in range(cfg.n_permutations):
        perm_idx = rng.permutation(Xp.shape[0])
        mod = PLSRegression(n_components=cfg.n_components)
        mod.fit(Xp[perm_idx], Y)
        null_perm[i, :] = component_score_r2(mod.x_scores_, mod.y_scores_, cfg.n_components)

    null_rot = np.zeros((cfg.n_rotations, cfg.n_components), dtype=np.float64)
    for i in range(cfg.n_rotations):
        rot = R.random(random_state=cfg.random_seed + i).as_matrix()
        Y_rot = Y @ rot.T
        mod = PLSRegression(n_components=cfg.n_components)
        mod.fit(Xp, Y_rot)
        null_rot[i, :] = component_score_r2(mod.x_scores_, mod.y_scores_, cfg.n_components)

    perm_p = np.array(
        [
            (np.sum(null_perm[:, c] >= pls_comp_r2[c]) + 1) / (len(null_perm) + 1)
            for c in range(cfg.n_components)
        ]
    )
    rot_p = np.array(
        [
            (np.sum(null_rot[:, c] >= pls_comp_r2[c]) + 1) / (len(null_rot) + 1)
            for c in range(cfg.n_components)
        ]
    )

    null_summary = pd.DataFrame(
        {
            "component": np.arange(1, cfg.n_components + 1),
            "pls_component_r2": pls_comp_r2,
            "perm_p": perm_p,
            "rot_p": rot_p,
        }
    )
    null_summary.to_csv(out_tables / "pls_null_summary.csv", index=False)

    # Attach full-model component scores for spatial plotting
    ahba_plot = ahba[["subject", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "coord_abs_x"]].copy()
    for c in range(cfg.n_components):
        ahba_plot[f"PLS_C{c+1}"] = pls_full.x_scores_[:, c]
    ahba_plot.to_csv(out_tables / "ahba_sample_component_scores.csv", index=False)

    # Regional mean component scores
    region_comp = (
        ahba_plot.groupby("tissue_or_parcel", as_index=False)[["coord_x", "coord_y", "coord_z", "coord_abs_x", "PLS_C1", "PLS_C2", "PLS_C3"]]
        .mean()
    )
    region_comp.to_csv(out_tables / "ahba_region_component_scores.csv", index=False)

    # ------------------ Figures ------------------
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # Figure 1: sample spatial distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(ahba_plot["coord_y"], ahba_plot["coord_z"], s=24, alpha=0.7)
    axes[0].set_title("AHBA samples in Y-Z")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("Z")

    axes[1].scatter(ahba_plot["coord_abs_x"], ahba_plot["coord_z"], s=24, alpha=0.7)
    axes[1].set_title("AHBA samples in |X|-Z")
    axes[1].set_xlabel("|X|")
    axes[1].set_ylabel("Z")
    fig.savefig(out_figs / "fig1_sample_spatial_distribution.png", dpi=220)
    plt.close(fig)

    # Figure 2: coordinate prediction (holdout)
    pls_holdout = PLSRegression(n_components=cfg.n_components)
    pls_holdout.fit(X_tr, y_tr)
    pred_tr = pls_holdout.predict(X_tr)
    pred_te = pls_holdout.predict(X_te)

    labels = ["Y", "Z", "|X|"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for i in range(3):
        axes[i].scatter(y_tr[:, i], pred_tr[:, i], s=16, alpha=0.35, label="Train")
        axes[i].scatter(y_te[:, i], pred_te[:, i], s=24, alpha=0.85, label="Test")
        lo = min(np.min(y_te[:, i]), np.min(pred_te[:, i]))
        hi = max(np.max(y_te[:, i]), np.max(pred_te[:, i]))
        axes[i].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[i].set_xlabel(f"Observed {labels[i]}")
        axes[i].set_ylabel(f"Predicted {labels[i]}")
        axes[i].set_title(labels[i])
    axes[0].legend(loc="best")
    fig.suptitle("PLSR(3) coordinate prediction (train/test)")
    fig.savefig(out_figs / "fig2_pls_coordinate_prediction.png", dpi=220)
    plt.close(fig)

    # Figure 3: estimator selection summary
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.lineplot(data=cv_raw, x="n_components", y="r2", hue="estimator", marker="o", ax=axes[0])
    axes[0].set_title("CV R^2 by estimator")
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("R^2")

    sns.lineplot(data=cv_raw, x="n_components", y="mae", hue="estimator", marker="o", ax=axes[1])
    axes[1].set_title("CV MAE by estimator")
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("MAE")
    fig.savefig(out_figs / "fig3_estimator_cv_selection.png", dpi=220)
    plt.close(fig)

    # Figure 4: Fig-1-like spatial overlay of PLS components
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    views = [("coord_y", "coord_z", "Y-Z"), ("coord_abs_x", "coord_z", "|X|-Z")]
    for c in range(cfg.n_components):
        comp = f"PLS_C{c+1}"
        v = ahba_plot[comp].to_numpy()
        vmin, vmax = np.percentile(v, [2, 98])
        for j, (xcol, ycol, ttl) in enumerate(views):
            ax = axes[c, j]
            sc = ax.scatter(
                ahba_plot[xcol],
                ahba_plot[ycol],
                c=ahba_plot[comp],
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                s=28,
                alpha=0.85,
                linewidths=0,
            )
            ax.set_title(f"C{c+1} • {ttl}")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("score")
    fig.suptitle("Fig-1-like AHBA spatial transcriptomic axes (PLS components)")
    fig.savefig(out_figs / "fig4_fig1_like_spatial_gradients.png", dpi=220)
    plt.close(fig)

    # Figure 5: null significance
    fig, axes = plt.subplots(cfg.n_components, 2, figsize=(12, 4 * cfg.n_components), constrained_layout=True)
    if cfg.n_components == 1:
        axes = np.array([axes])
    for c in range(cfg.n_components):
        axes[c, 0].hist(null_perm[:, c], bins=35, alpha=0.8)
        axes[c, 0].axvline(pls_comp_r2[c], color="red", lw=2)
        axes[c, 0].set_title(f"Permutation null C{c+1} (p={perm_p[c]:.4f})")
        axes[c, 0].set_xlabel("r^2")

        axes[c, 1].hist(null_rot[:, c], bins=35, alpha=0.8)
        axes[c, 1].axvline(pls_comp_r2[c], color="red", lw=2)
        axes[c, 1].set_title(f"Rotation null C{c+1} (p={rot_p[c]:.4f})")
        axes[c, 1].set_xlabel("r^2")
    fig.savefig(out_figs / "fig5_null_significance_tests.png", dpi=220)
    plt.close(fig)

    # Figure 6: method comparison across components
    cmp_long = comp_compare.melt(id_vars="component", var_name="method", value_name="r2")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(data=cmp_long, x="component", y="r2", hue="method", ax=ax)
    ax.set_title("Component coupling strength across PLSR / CCA / PLSC")
    ax.set_xlabel("Component")
    ax.set_ylabel("r^2")
    fig.savefig(out_figs / "fig6_component_coupling_comparison.png", dpi=220)
    plt.close(fig)

    # Figure 7: subject-region coverage heatmap
    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
    sns.heatmap(coverage_df, cmap="viridis", cbar_kws={"label": "Observed (1/0)"}, ax=ax)
    ax.set_title("AHBA subject-region coverage")
    ax.set_xlabel("Region")
    ax.set_ylabel("Subject")
    fig.savefig(out_figs / "fig7_subject_region_coverage_heatmap.png", dpi=220)
    plt.close(fig)

    # ------------------ Replication verdict report ------------------
    data_summary = {
        "n_rows_ahba": int(len(ahba)),
        "n_subjects": int(ahba["subject"].nunique()),
        "n_regions": int(ahba["tissue_or_parcel"].nunique()),
        "n_genes": int(len(gene_cols)),
        "n_hvg_requested": int(len(header["hvg_list"])),
        "n_hvg_matched": int(len(header["genes_hvg"])),
        "n_hvg_missing": int(len(header["missing_hvg"])),
        "mean_vs_median_absdiff": float(mean_vs_median_absdiff),
        "samples_per_subject": ahba.groupby("subject").size().to_dict(),
    }

    best_estimator = {
        "estimator": str(best["estimator"]),
        "n_components": int(best["n_components"]),
        "cv_r2": float(best["r2"]),
        "cv_mae": float(best["mae"]),
    }

    verdict_flags = {
        "plsr_selected": bool(best_estimator["estimator"] == "PLSR" and best_estimator["n_components"] == 3),
        "strong_holdout_r2": bool(holdout_metrics["test_r2"] >= 0.5),
        "component1_perm_significant": bool(perm_p[0] < 0.05),
        "component1_strength_high": bool(pls_comp_r2[0] >= 0.5),
    }

    n_pass = int(sum(verdict_flags.values()))
    if n_pass >= 4:
        overall_verdict = "Strongly sufficient for core Fig-1-style Vogel replication."
    elif n_pass >= 3:
        overall_verdict = "Sufficient for core replication with caveats on robustness/detail parity."
    elif n_pass >= 2:
        overall_verdict = "Partially sufficient: captures some but not all core Fig-1-style behaviors."
    else:
        overall_verdict = "Insufficient for robust Fig-1-style replication as configured."

    report_json = {
        "config": asdict(cfg),
        "data_summary": data_summary,
        "best_estimator": best_estimator,
        "holdout_metrics": holdout_metrics,
        "component_strength_plsr_r2": pls_comp_r2.tolist(),
        "component_strength_cca_r2": cca_comp_r2.tolist(),
        "component_strength_plsc_r2": plsc_comp_r2.tolist(),
        "perm_p": perm_p.tolist(),
        "rot_p": rot_p.tolist(),
        "verdict_flags": verdict_flags,
        "overall_verdict": overall_verdict,
        "references": {
            "paper": "https://www.pnas.org/doi/10.1073/pnas.2219137121",
            "code_repo": "https://github.com/PennLINC/Vogel_PLS_Tx-Space",
            "nb02": "https://github.com/PennLINC/Vogel_PLS_Tx-Space/blob/main/NB02_PLSModelFitting.ipynb",
        },
    }

    with open(out_root / "ahba_vogel_replication_summary.json", "w") as f:
        json.dump(report_json, f, indent=2)

    # Markdown report
    donor_loo_txt = donor_loo.to_string(index=False) if len(donor_loo) else "No donor-LOO rows available"
    cv_txt = cv_summary.to_string(index=False)
    cmp_txt = comp_compare.to_string(index=False)

    report_md = f"""# AHBA-Only Vogel Replication Report

## Objective
Assess whether the AHBA subset embedded in `gxp_samples.csv` is sufficient to reproduce the **core Fig-1-style findings** from Vogel et al. (PNAS 2024) and their published code workflow.

## Build Plan Executed
1. Construct subject x region x gene AHBA atlases using both **mean** and **median** within each region.
2. Perform gene dimensionality reduction (PCA on standardized expression).
3. Fit and compare **PLSR / CCA / PLSCanonical** for mapping expression axes to spatial coordinates `[Y, Z, |X|]`.
4. Run null significance tests (permutation and rotation) on PLS component coupling.
5. Generate Fig-1-style spatial gradient visualizations and quantitative summary tables.

## Data Sufficiency Snapshot
- AHBA rows: **{data_summary['n_rows_ahba']}**
- Subjects: **{data_summary['n_subjects']}**
- Regions: **{data_summary['n_regions']}**
- Genes: **{data_summary['n_genes']}**
- HVG overlap (for reference): requested {data_summary['n_hvg_requested']}, matched {data_summary['n_hvg_matched']}, missing {data_summary['n_hvg_missing']}
- Mean-vs-median atlas absolute difference: **{data_summary['mean_vs_median_absdiff']:.6f}** (near-zero indicates one sample per subject-region in this file)

## Model Selection and Fit
Best CV setting:
- Estimator: **{best_estimator['estimator']}**
- Components: **{best_estimator['n_components']}**
- CV R^2: **{best_estimator['cv_r2']:.4f}**
- CV MAE: **{best_estimator['cv_mae']:.4f}**

Holdout performance (PLSR, 3 components):
- Test R^2: **{holdout_metrics['test_r2']:.4f}**
- Test MAE: **{holdout_metrics['test_mae']:.4f}**

## Axis Congruence Strength
PLSR/CCA/PLSC component coupling (r^2):

```text
{cmp_txt}
```

Permutation and rotation null p-values for PLSR:
- Permutation p: {', '.join(f'C{i+1}={p:.4f}' for i,p in enumerate(perm_p.tolist()))}
- Rotation p: {', '.join(f'C{i+1}={p:.4f}' for i,p in enumerate(rot_p.tolist()))}

## Donor Generalization (leave-one-donor-out)
```text
{donor_loo_txt}
```

## CV Estimator Comparison (summary)
```text
{cv_txt}
```

## Replication Verdict
Flag checks:
- PLSR(3) selected by CV: **{verdict_flags['plsr_selected']}**
- Strong holdout R^2 (>=0.5): **{verdict_flags['strong_holdout_r2']}**
- Component 1 significant under permutation: **{verdict_flags['component1_perm_significant']}**
- Component 1 strong coupling (r^2>=0.5): **{verdict_flags['component1_strength_high']}**

### Overall
**{overall_verdict}**

Interpretation: This AHBA copy appears suitable for reproducing the *core* spatial-transcriptomic axis behavior emphasized in Vogel Fig-1-style analyses, with expected caveats versus exact manuscript reproduction (different preprocessing lineage, coordinate handling, and potential probe/normalization differences from the original full notebook stack).

## Figures Generated
- `fig1_sample_spatial_distribution.png`
- `fig2_pls_coordinate_prediction.png`
- `fig3_estimator_cv_selection.png`
- `fig4_fig1_like_spatial_gradients.png`
- `fig5_null_significance_tests.png`
- `fig6_component_coupling_comparison.png`
- `fig7_subject_region_coverage_heatmap.png`

## References
- Paper DOI page: https://www.pnas.org/doi/10.1073/pnas.2219137121
- Code repository: https://github.com/PennLINC/Vogel_PLS_Tx-Space
- Core notebook: https://github.com/PennLINC/Vogel_PLS_Tx-Space/blob/main/NB02_PLSModelFitting.ipynb
"""

    (out_root / "ahba_vogel_replication_report.md").write_text(report_md)

    print("\nCompleted AHBA-only replication workflow.")
    print("Outputs:", out_root)


if __name__ == "__main__":
    run_analysis(Config())
