#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)


META_COLS = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]
DET_MODEL = "combat__affine_gl3__constrained_anchor__rbf"
UNI_MODEL = "unified__harm=combat__cal=hier_affine_map__robust=student_t__hetero=gene_var__uncshrink=false"
DET_NAME = "DLAM"
UNI_NAME = "PLAM"
NAIVE_NAME = "Naive fill"


@dataclass
class Config:
    csv_path: str = "/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv"
    hvg_path: str = "/Users/erdem/Documents/github/gtex_gp/ahba_100hvg.txt"
    deterministic_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment_allgenes"
    unified_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_unified"
    unified_phase2_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_phase2"
    unified_export_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_final"
    out_root: str = "/Users/erdem/Downloads/main_scientific_assets"
    combat_use_covariates: bool = True
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_rbf_length: float = 25.0
    c_min: int = 8
    top_n_parcels: int = 10
    top_n_genes: int = 12
    seed: int = 123


FONT = {
    "title": 9,
    "label": 8,
    "tick": 7,
    "legend": 7,
    "small": 6.5,
}

SCATTER_SIZES = {
    "background": 10,
    "highlight": 20,
    "legend_marker": 6,
}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build revised figures and manifests for the scientific manuscript.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--deterministic-root", default=Config.deterministic_root)
    p.add_argument("--unified-root", default=Config.unified_root)
    p.add_argument("--unified-phase2-root", default=Config.unified_phase2_root)
    p.add_argument("--unified-export-root", default=Config.unified_export_root)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.combat_use_covariates)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--gp-rbf-length", type=float, default=Config.gp_rbf_length)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--top-n-parcels", type=int, default=Config.top_n_parcels)
    p.add_argument("--top-n-genes", type=int, default=Config.top_n_genes)
    p.add_argument("--seed", type=int, default=Config.seed)
    a = p.parse_args()
    return Config(**vars(a))


def _parcel_gene_matrix(df: pd.DataFrame, gene_cols: List[str], n_parcels: int) -> np.ndarray:
    out = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    grp = df.groupby("parcel_idx")[gene_cols].mean()
    for pidx in grp.index.tolist():
        p = int(pidx)
        if 0 <= p < n_parcels:
            out[p, :] = grp.loc[pidx, gene_cols].to_numpy(dtype=np.float64)
    return out


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _masked_imshow(ax, mat: np.ndarray, cmap: str, vmin: float, vmax: float, title: str):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", interpolation="none", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FONT["title"])
    ax.set_xlabel("Gene index", fontsize=FONT["label"])
    ax.set_ylabel("Parcel index", fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    return im


def _panel_label(ax, tag: str) -> None:
    ax.text(
        0.01,
        0.99,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbbbbb", alpha=0.95),
    )


def _distinct_palette(n: int) -> List[str]:
    base = list(plt.get_cmap("tab20").colors)
    if n > len(base):
        raise ValueError(f"Requested {n} colors but only {len(base)} distinct colors are available.")
    return [mcolors.to_hex(base[i]) for i in range(n)]




def _apply_scatter_limits(ax, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return
    xx = x[m]
    yy = y[m]
    xlo, xhi = np.nanpercentile(xx, [0.5, 99.5])
    ylo, yhi = np.nanpercentile(yy, [0.5, 99.5])
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        xlo, xhi = float(np.nanmin(xx)), float(np.nanmax(xx))
    if not np.isfinite(ylo) or not np.isfinite(yhi) or yhi <= ylo:
        ylo, yhi = float(np.nanmin(yy)), float(np.nanmax(yy))
    xpad = 0.06 * max(xhi - xlo, 1e-3)
    ypad = 0.06 * max(yhi - ylo, 1e-3)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)

def _set_scatter_axes(ax, xlabel: str, ylabel: str, title: str):
    ax.set_title(title, fontsize=FONT["title"])
    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=FONT["tick"])


def _select_representative_subject(summary_df: pd.DataFrame, c_min: int) -> pd.DataFrame:
    df = summary_df[summary_df["n_obs_parcels"] >= c_min].copy()
    if len(df) == 0:
        raise RuntimeError("No eligible subjects for representative selection.")
    med_p = float(df["pearson_r"].median())
    med_r = float(df["rmse"].median())
    df["abs_delta_pearson"] = np.abs(df["pearson_r"] - med_p)
    df["abs_delta_rmse"] = np.abs(df["rmse"] - med_r)
    df = df.sort_values(["abs_delta_pearson", "abs_delta_rmse", "subject"], ascending=[True, True, True]).reset_index(drop=True)
    row = df.iloc[[0]].copy()
    row["selection_rule"] = "closest_to_median_unified_performance_among_n_obs_ge_cmin"
    return row[["subject", "n_obs_parcels", "pearson_r", "rmse", "selection_rule"]].rename(columns={"pearson_r": "pearson_unified", "rmse": "rmse_unified"})


def _build_long_from_matrices(allen_h: np.ndarray, gtex_h: np.ndarray, genes: List[str], target_meta: pd.DataFrame, obs_mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parcel_idx": np.repeat(target_meta["parcel_idx"].to_numpy(dtype=np.int32), len(genes)),
            "parcel_name": np.repeat(target_meta["tissue_or_parcel"].astype(str).to_numpy(), len(genes)),
            "gene": np.tile(np.asarray(genes), len(target_meta)),
            "allen_value": allen_h.reshape(-1),
            "gtex_value": gtex_h.reshape(-1),
            "is_observed_gtex": np.repeat(obs_mask.astype(bool), len(genes)),
        }
    )


def _select_top_parcels(ahba_h_mat: np.ndarray, target_meta: pd.DataFrame, top_n: int) -> pd.DataFrame:
    score = np.nanvar(ahba_h_mat, axis=1)
    df = target_meta[["parcel_idx", "tissue_or_parcel"]].copy()
    df["selection_score"] = score
    out = df.sort_values(["selection_score", "parcel_idx"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["color_hex"] = _distinct_palette(len(out))
    return out


def _select_top_genes(ahba_h_mat: np.ndarray, genes: List[str], top_n: int) -> pd.DataFrame:
    score = np.nanvar(ahba_h_mat, axis=0)
    df = pd.DataFrame({"gene": genes, "selection_score": score})
    out = df.sort_values(["selection_score", "gene"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out["color_hex"] = _distinct_palette(len(out))
    return out


def _shape_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor="#6f6f6f", markeredgecolor="#6f6f6f", label="Observed"),
        Line2D([0], [0], marker="^", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor="#9a9a9a", markeredgecolor="#9a9a9a", label="Imputed"),
    ]


def _plot_legend_top10_parcels(parcel_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(4.8, 3.8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    handles = _shape_handles()
    parcel_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=f"{int(row.parcel_idx):03d} {row.tissue_or_parcel}")
        for row in parcel_df.itertuples(index=False)
    ]
    leg1 = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=FONT["legend"], title="Markers", title_fontsize=FONT["legend"])
    ax.add_artist(leg1)
    ax.legend(handles=parcel_handles, loc="upper left", bbox_to_anchor=(0, 0.72), frameon=False, fontsize=FONT["legend"], title="Top parcels", title_fontsize=FONT["legend"])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_legend_top_genes(gene_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(3.8, 4.2))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=row.gene)
        for row in gene_df.itertuples(index=False)
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=FONT["legend"], title="Top genes", title_fontsize=FONT["legend"])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_parcel_identity(long_df: pd.DataFrame, parcel_df: pd.DataFrame, out_path: Path) -> None:
    parcel_colors = {int(r.parcel_idx): r.color_hex for r in parcel_df.itertuples(index=False)}
    fig = plt.figure(figsize=(8.2, 5.1))
    gs = GridSpec(1, 2, width_ratios=[4.8, 1.7], figure=fig, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    d = long_df.copy()
    top_set = set(parcel_colors.keys())
    bg_u = d[~d["is_observed_gtex"]]
    bg_o = d[d["is_observed_gtex"]]
    ax.scatter(bg_u["allen_value"], bg_u["gtex_value"], c="#9a9a9a", s=SCATTER_SIZES["background"], alpha=0.22, marker="^", linewidths=0)
    ax.scatter(bg_o["allen_value"], bg_o["gtex_value"], c="#808080", s=SCATTER_SIZES["background"], alpha=0.32, marker="o", linewidths=0)
    d = d[d["parcel_idx"].isin(top_set)].copy()
    d["color"] = d["parcel_idx"].map(parcel_colors)
    u = d[~d["is_observed_gtex"]]
    o = d[d["is_observed_gtex"]]
    ax.scatter(u["allen_value"], u["gtex_value"], c=u["color"], s=SCATTER_SIZES["highlight"], alpha=0.72, marker="^", linewidths=0)
    ax.scatter(o["allen_value"], o["gtex_value"], c=o["color"], s=SCATTER_SIZES["highlight"], alpha=0.86, marker="o", linewidths=0)
    _set_scatter_axes(ax, "AHBA harmonized value", "GTEx harmonized value", "Parcel identity")
    _apply_scatter_limits(ax, long_df["allen_value"].to_numpy(dtype=float), long_df["gtex_value"].to_numpy(dtype=float))
    _panel_label(ax, "A")

    shape_leg = ax_leg.legend(handles=_shape_handles(), loc="upper left", frameon=False, fontsize=FONT["legend"], title="Markers", title_fontsize=FONT["legend"])
    ax_leg.add_artist(shape_leg)
    parcel_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=f"{int(row.parcel_idx):03d} {row.tissue_or_parcel}")
        for row in parcel_df.itertuples(index=False)
    ]
    ax_leg.legend(handles=parcel_handles, loc="upper left", bbox_to_anchor=(0, 0.72), frameon=False, fontsize=FONT["legend"], title="Top parcels", title_fontsize=FONT["legend"])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_gene_identity(long_df: pd.DataFrame, gene_df: pd.DataFrame, out_path: Path) -> None:
    gene_colors = {str(r.gene): r.color_hex for r in gene_df.itertuples(index=False)}
    d_full = long_df.copy()
    d = long_df[long_df["gene"].isin(gene_colors.keys())].copy()
    fig = plt.figure(figsize=(8.2, 5.1))
    gs = GridSpec(1, 2, width_ratios=[4.8, 1.6], figure=fig, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    bg_u = d_full[~d_full["is_observed_gtex"]]
    bg_o = d_full[d_full["is_observed_gtex"]]
    ax.scatter(bg_u["allen_value"], bg_u["gtex_value"], c="#9a9a9a", s=SCATTER_SIZES["background"], alpha=0.18, marker="^", linewidths=0)
    ax.scatter(bg_o["allen_value"], bg_o["gtex_value"], c="#808080", s=SCATTER_SIZES["background"], alpha=0.26, marker="o", linewidths=0)
    u = d[~d["is_observed_gtex"]]
    o = d[d["is_observed_gtex"]]
    ax.scatter(u["allen_value"], u["gtex_value"], c=[gene_colors[g] for g in u["gene"]], s=SCATTER_SIZES["highlight"], alpha=0.78, marker="^", linewidths=0)
    ax.scatter(o["allen_value"], o["gtex_value"], c=[gene_colors[g] for g in o["gene"]], s=SCATTER_SIZES["highlight"], alpha=0.90, marker="o", linewidths=0)
    _set_scatter_axes(ax, "AHBA harmonized value", "GTEx harmonized value", "Gene identity")
    _apply_scatter_limits(ax, d_full["allen_value"].to_numpy(dtype=float), d_full["gtex_value"].to_numpy(dtype=float))
    _panel_label(ax, "B")

    shape_leg = ax_leg.legend(handles=_shape_handles(), loc="upper left", frameon=False, fontsize=FONT["legend"], title="Markers", title_fontsize=FONT["legend"])
    ax_leg.add_artist(shape_leg)
    gene_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=row.gene)
        for row in gene_df.itertuples(index=False)
    ]
    ax_leg.legend(handles=gene_handles, loc="upper left", bbox_to_anchor=(0, 0.72), frameon=False, fontsize=FONT["legend"], title="Top genes", title_fontsize=FONT["legend"])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _annotate_corr_summary(ax, labels: List[str], vals: np.ndarray, prefix: str) -> None:
    s = pd.DataFrame({"label": labels, "value": vals}).dropna()
    if len(s) == 0:
        return
    top = s.sort_values("value", ascending=False).head(3)
    bot = s.sort_values("value", ascending=True).head(3)
    txt = f"Top {prefix.lower()}s: " + ", ".join(top["label"].tolist()) + "\n" + f"Bottom {prefix.lower()}s: " + ", ".join(bot["label"].tolist())
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=FONT["small"], va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))


def _plot_scatter_parcel_corr(long_df: pd.DataFrame, target_meta: pd.DataFrame, out_path: Path) -> None:
    corr = []
    labels = []
    for row in target_meta.itertuples(index=False):
        d = long_df[long_df["parcel_idx"] == int(row.parcel_idx)]
        corr.append(_pearson_safe(d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float)))
        labels.append(str(row.tissue_or_parcel))
    corr = np.asarray(corr, dtype=float)
    d = long_df.merge(pd.DataFrame({"parcel_idx": target_meta["parcel_idx"], "corr": corr}), on="parcel_idx", how="left")

    fig = plt.figure(figsize=(8.2, 5.1))
    gs = GridSpec(1, 2, width_ratios=[4.8, 1.7], figure=fig, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.axis("off")
    u = d[(~d["is_observed_gtex"]) & np.isfinite(d["corr"])]
    o = d[(d["is_observed_gtex"]) & np.isfinite(d["corr"])]
    sc = ax.scatter(u["allen_value"], u["gtex_value"], c=u["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.28, marker="^", linewidths=0)
    ax.scatter(o["allen_value"], o["gtex_value"], c=o["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.40, marker="o", linewidths=0)
    _set_scatter_axes(ax, "AHBA harmonized value", "GTEx harmonized value", "Parcel-level correlation")
    _apply_scatter_limits(ax, d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float))
    _panel_label(ax, "C")
    cax = ax_side.inset_axes([0.02, 0.58, 0.18, 0.30])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("r", fontsize=FONT["legend"])
    cb.ax.tick_params(labelsize=FONT["tick"])
    _annotate_corr_summary(ax_side, labels, corr, "Parcel")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_gene_corr(long_df: pd.DataFrame, genes: List[str], out_path: Path) -> None:
    corr = []
    for g in genes:
        d = long_df[long_df["gene"] == g]
        corr.append(_pearson_safe(d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float)))
    corr = np.asarray(corr, dtype=float)
    d = long_df.merge(pd.DataFrame({"gene": genes, "corr": corr}), on="gene", how="left")

    fig = plt.figure(figsize=(8.2, 5.1))
    gs = GridSpec(1, 2, width_ratios=[4.8, 1.7], figure=fig, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.axis("off")
    u = d[(~d["is_observed_gtex"]) & np.isfinite(d["corr"])]
    o = d[(d["is_observed_gtex"]) & np.isfinite(d["corr"])]
    sc = ax.scatter(u["allen_value"], u["gtex_value"], c=u["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.28, marker="^", linewidths=0)
    ax.scatter(o["allen_value"], o["gtex_value"], c=o["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.40, marker="o", linewidths=0)
    _set_scatter_axes(ax, "AHBA harmonized value", "GTEx harmonized value", "Gene-level correlation")
    _apply_scatter_limits(ax, d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float))
    _panel_label(ax, "D")
    cax = ax_side.inset_axes([0.02, 0.58, 0.18, 0.30])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("r", fontsize=FONT["legend"])
    cb.ax.tick_params(labelsize=FONT["tick"])
    _annotate_corr_summary(ax_side, genes, corr, "Gene")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_scatter_suite(long_df: pd.DataFrame, target_meta: pd.DataFrame, parcel_df: pd.DataFrame, gene_df: pd.DataFrame, out_path: Path) -> None:
    parcel_colors = {int(r.parcel_idx): r.color_hex for r in parcel_df.itertuples(index=False)}
    gene_colors = {str(r.gene): r.color_hex for r in gene_df.itertuples(index=False)}

    region_corr = []
    for row in target_meta.itertuples(index=False):
        d = long_df[long_df["parcel_idx"] == int(row.parcel_idx)]
        region_corr.append(_pearson_safe(d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float)))
    region_corr = np.asarray(region_corr, dtype=float)
    gene_corr = []
    genes = gene_df["gene"].tolist()
    for g in genes:
        d = long_df[long_df["gene"] == g]
        gene_corr.append(_pearson_safe(d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float)))
    gene_corr = np.asarray(gene_corr, dtype=float)

    fig = plt.figure(figsize=(12.6, 8.8))
    gs = GridSpec(2, 3, width_ratios=[1.0, 1.0, 0.58], figure=fig, wspace=0.25, hspace=0.28)
    axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    ax_leg = fig.add_subplot(gs[:, 2])
    ax_leg.axis("off")

    d = long_df.copy()
    top_set = set(parcel_colors.keys())
    bg_u = d[~d["is_observed_gtex"]]
    bg_o = d[d["is_observed_gtex"]]
    axs[0, 0].scatter(bg_u["allen_value"], bg_u["gtex_value"], c="#9a9a9a", s=SCATTER_SIZES["background"], alpha=0.24, marker="^", linewidths=0)
    axs[0, 0].scatter(bg_o["allen_value"], bg_o["gtex_value"], c="#808080", s=SCATTER_SIZES["background"], alpha=0.34, marker="o", linewidths=0)
    d_top = d[d["parcel_idx"].isin(top_set)].copy()
    d_top["parcel_color"] = d_top["parcel_idx"].map(parcel_colors)
    u = d_top[~d_top["is_observed_gtex"]]
    o = d_top[d_top["is_observed_gtex"]]
    axs[0, 0].scatter(u["allen_value"], u["gtex_value"], c=u["parcel_color"], s=SCATTER_SIZES["highlight"], alpha=0.84, marker="^", linewidths=0)
    axs[0, 0].scatter(o["allen_value"], o["gtex_value"], c=o["parcel_color"], s=SCATTER_SIZES["highlight"], alpha=0.96, marker="o", linewidths=0)
    _set_scatter_axes(axs[0, 0], "AHBA harmonized value", "GTEx harmonized value", "Parcel identity")
    _apply_scatter_limits(axs[0, 0], d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float))
    _panel_label(axs[0, 0], "A")

    dg = d[d["gene"].isin(gene_colors.keys())].copy()
    axs[0, 1].scatter(bg_u["allen_value"], bg_u["gtex_value"], c="#9a9a9a", s=SCATTER_SIZES["background"], alpha=0.18, marker="^", linewidths=0)
    axs[0, 1].scatter(bg_o["allen_value"], bg_o["gtex_value"], c="#808080", s=SCATTER_SIZES["background"], alpha=0.26, marker="o", linewidths=0)
    ug = dg[~dg["is_observed_gtex"]]
    og = dg[dg["is_observed_gtex"]]
    axs[0, 1].scatter(ug["allen_value"], ug["gtex_value"], c=[gene_colors[g] for g in ug["gene"]], s=SCATTER_SIZES["highlight"], alpha=0.84, marker="^", linewidths=0)
    axs[0, 1].scatter(og["allen_value"], og["gtex_value"], c=[gene_colors[g] for g in og["gene"]], s=SCATTER_SIZES["highlight"], alpha=0.96, marker="o", linewidths=0)
    _set_scatter_axes(axs[0, 1], "AHBA harmonized value", "GTEx harmonized value", "Gene identity")
    _apply_scatter_limits(axs[0, 1], d["allen_value"].to_numpy(dtype=float), d["gtex_value"].to_numpy(dtype=float))
    _panel_label(axs[0, 1], "B")

    dr = d.merge(pd.DataFrame({"parcel_idx": target_meta["parcel_idx"], "corr": region_corr}), on="parcel_idx", how="left")
    ur = dr[(~dr["is_observed_gtex"]) & np.isfinite(dr["corr"])]
    orr = dr[(dr["is_observed_gtex"]) & np.isfinite(dr["corr"])]
    sc1 = axs[1, 0].scatter(ur["allen_value"], ur["gtex_value"], c=ur["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.26, marker="^", linewidths=0)
    axs[1, 0].scatter(orr["allen_value"], orr["gtex_value"], c=orr["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.40, marker="o", linewidths=0)
    _set_scatter_axes(axs[1, 0], "AHBA harmonized value", "GTEx harmonized value", "Parcel-level correlation")
    _apply_scatter_limits(axs[1, 0], dr["allen_value"].to_numpy(dtype=float), dr["gtex_value"].to_numpy(dtype=float))
    _panel_label(axs[1, 0], "C")
    cax1 = inset_axes(axs[1, 0], width="3%", height="45%", loc="upper right", borderpad=1.0)
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.set_label("r", fontsize=FONT["legend"])
    cb1.ax.tick_params(labelsize=FONT["tick"])

    dg2 = d[d["gene"].isin(genes)].merge(pd.DataFrame({"gene": genes, "corr": gene_corr}), on="gene", how="left")
    ug2 = dg2[(~dg2["is_observed_gtex"]) & np.isfinite(dg2["corr"])]
    og2 = dg2[(dg2["is_observed_gtex"]) & np.isfinite(dg2["corr"])]
    sc2 = axs[1, 1].scatter(ug2["allen_value"], ug2["gtex_value"], c=ug2["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.26, marker="^", linewidths=0)
    axs[1, 1].scatter(og2["allen_value"], og2["gtex_value"], c=og2["corr"], cmap="coolwarm", vmin=-1, vmax=1, s=SCATTER_SIZES["highlight"], alpha=0.40, marker="o", linewidths=0)
    _set_scatter_axes(axs[1, 1], "AHBA harmonized value", "GTEx harmonized value", "Gene-level correlation")
    _apply_scatter_limits(axs[1, 1], dg2["allen_value"].to_numpy(dtype=float), dg2["gtex_value"].to_numpy(dtype=float))
    _panel_label(axs[1, 1], "D")
    cax2 = inset_axes(axs[1, 1], width="3%", height="45%", loc="upper right", borderpad=1.0)
    cb2 = fig.colorbar(sc2, cax=cax2)
    cb2.set_label("r", fontsize=FONT["legend"])
    cb2.ax.tick_params(labelsize=FONT["tick"])

    shape_leg = ax_leg.legend(handles=_shape_handles(), loc="upper left", frameon=False, fontsize=FONT["legend"], title="Markers", title_fontsize=FONT["legend"])
    ax_leg.add_artist(shape_leg)
    parcel_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=f"{int(row.parcel_idx):03d} {row.tissue_or_parcel}")
        for row in parcel_df.itertuples(index=False)
    ]
    leg2 = ax_leg.legend(handles=parcel_handles, loc="upper left", bbox_to_anchor=(0.0, 0.72), frameon=False, fontsize=FONT["legend"], title="Top parcels", title_fontsize=FONT["legend"])
    ax_leg.add_artist(leg2)
    gene_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=SCATTER_SIZES["legend_marker"], markerfacecolor=row.color_hex, markeredgecolor=row.color_hex, label=row.gene)
        for row in gene_df.itertuples(index=False)
    ]
    ax_leg.legend(handles=gene_handles, loc="upper left", bbox_to_anchor=(0.0, 0.27), frameon=False, fontsize=FONT["legend"], title="Top genes", title_fontsize=FONT["legend"])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_demo(
    ahba_raw_mat: np.ndarray,
    sparse_raw: np.ndarray,
    sparse_h: np.ndarray,
    ahba_h_mat: np.ndarray,
    filled_h: np.ndarray,
    out_path: Path,
    filled_title: str = "Completed GTEx harmonized",
    residual_title: str = "Residual: GTEx - AHBA",
) -> None:
    raw_stack = np.concatenate([ahba_raw_mat[np.isfinite(ahba_raw_mat)], sparse_raw[np.isfinite(sparse_raw)]]) if np.isfinite(sparse_raw).any() else ahba_raw_mat[np.isfinite(ahba_raw_mat)]
    harm_stack = np.concatenate([ahba_h_mat[np.isfinite(ahba_h_mat)], sparse_h[np.isfinite(sparse_h)], filled_h[np.isfinite(filled_h)]])
    diff_h = filled_h - ahba_h_mat
    vmin_raw, vmax_raw = np.percentile(raw_stack, [2, 98]) if raw_stack.size else (-1, 1)
    vmin_h, vmax_h = np.percentile(harm_stack, [2, 98]) if harm_stack.size else (-1, 1)
    dabs = np.abs(diff_h[np.isfinite(diff_h)])
    lim_d = float(np.percentile(dabs, 98)) if dabs.size else 1.0

    fig = plt.figure(figsize=(16.5, 8.8))
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.06], figure=fig, wspace=0.18, hspace=0.24)
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])], [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]])
    cax1 = fig.add_subplot(gs[0, 3])
    cax2 = fig.add_subplot(gs[1, 3])
    cax3 = inset_axes(axes[1, 2], width="3%", height="60%", loc="center right", borderpad=1.0)

    im1 = _masked_imshow(axes[0, 0], ahba_raw_mat, "viridis", vmin_raw, vmax_raw, "AHBA raw")
    _masked_imshow(axes[0, 1], sparse_raw, "viridis", vmin_raw, vmax_raw, "Sparse GTEx raw")
    _masked_imshow(axes[0, 2], sparse_h, "viridis", vmin_h, vmax_h, "Sparse GTEx harmonized")
    im2 = _masked_imshow(axes[1, 0], ahba_h_mat, "viridis", vmin_h, vmax_h, "AHBA harmonized")
    _masked_imshow(axes[1, 1], filled_h, "viridis", vmin_h, vmax_h, filled_title)
    im3 = _masked_imshow(axes[1, 2], diff_h, "coolwarm", -lim_d, lim_d, residual_title)
    for tag, ax in zip(["A", "B", "C", "D", "E", "F"], axes.ravel()):
        _panel_label(ax, tag)

    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Raw", fontsize=FONT["legend"])
    cb1.ax.tick_params(labelsize=FONT["tick"])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Harmonized", fontsize=FONT["legend"])
    cb2.ax.tick_params(labelsize=FONT["tick"])
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label("Residual", fontsize=FONT["legend"])
    cb3.ax.tick_params(labelsize=FONT["tick"])

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x: float, y: float, w: float, h: float, txt: str, fc: str):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.0, edgecolor="#333333", facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, txt, ha="center", va="center", fontsize=9)


def _draw_arrow(ax, x0: float, y0: float, x1: float, y1: float):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#444444"))


def _plot_schematic_deterministic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (0.03, "Raw AHBA\n+ sparse GTEx", "#ddebf7"),
        (0.20, "Harmonize", "#fce5cd"),
        (0.36, "Local PLS", "#d9ead3"),
        (0.52, "Affine score\nalignment", "#f4cccc"),
        (0.68, "RBF spatial\ncompletion", "#e4d7f5"),
        (0.84, "Bridge +\ndecode", "#d9ead3"),
    ]
    for x, txt, fc in items:
        _draw_box(ax, x, 0.32, 0.11, 0.30, txt, fc)
    for i in range(len(items) - 1):
        _draw_arrow(ax, items[i][0] + 0.11, 0.47, items[i + 1][0], 0.47)
    ax.text(0.5, 0.86, "Deterministic latent alignment model (DLAM)", ha="center", va="center", fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_schematic_unified(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (0.03, "Raw AHBA\n+ sparse GTEx", "#ddebf7"),
        (0.20, "Global\nharmonization", "#fce5cd"),
        (0.36, "Atlas factor\nmodel", "#d9ead3"),
        (0.52, "Subject alignment\n+ calibration", "#f4cccc"),
        (0.68, "GP deviation\nconditioning", "#e4d7f5"),
        (0.84, "Decode +\noverwrite", "#d9ead3"),
    ]
    for x, txt, fc in items:
        _draw_box(ax, x, 0.32, 0.11, 0.30, txt, fc)
    for i in range(len(items) - 1):
        _draw_arrow(ax, items[i][0] + 0.11, 0.47, items[i + 1][0], 0.47)
    ax.text(0.5, 0.86, "Probabilistic latent alignment model (PLAM)", ha="center", va="center", fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_schematic_graphical_model(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.08, 0.10), 0.82, 0.76, fill=False, linewidth=1.2, edgecolor="#666666"))
    ax.text(0.88, 0.83, "subjects $s$", fontsize=9, ha="right")
    ax.add_patch(Rectangle((0.18, 0.18), 0.60, 0.50, fill=False, linewidth=1.0, edgecolor="#888888"))
    ax.text(0.76, 0.66, "parcels $r$", fontsize=9, ha="right")

    def node(x, y, text, fc):
        box = FancyBboxPatch((x - 0.055, y - 0.032), 0.11, 0.064, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=0.9, edgecolor="#333333", facecolor=fc)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        return (x, y)

    U = node(0.24, 0.76, "$U$", "#d9ead3")
    W = node(0.41, 0.76, "$W$", "#d9ead3")
    A = node(0.56, 0.76, "$\\alpha$", "#d9ead3")
    Q = node(0.20, 0.45, "$Q_s$", "#f4cccc")
    D = node(0.36, 0.45, "$\\Delta_s$", "#f4cccc")
    Z = node(0.52, 0.45, "$z_{s,r}$", "#fce5cd")
    C = node(0.68, 0.45, "$(a_s,b_s)$", "#f4cccc")
    XA = node(0.40, 0.90, "$X_A^{(h)}$", "#ddebf7")
    XG = node(0.83, 0.45, "$X_s^{(G,h)}$", "#ddebf7")
    for p0, p1 in [(U, XA), (W, XA), (A, XA), (U, Z), (D, Z), (Q, Z), (W, Z), (A, Z), (Z, XG), (C, XG)]:
        _draw_arrow(ax, p0[0], p0[1], p1[0], p1[1])
    ax.text(0.5, 0.97, "Graphical summary of PLAM", ha="center", va="center", fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _read_combined_harmonized(path: Path, genes_hvg: List[str]) -> pd.DataFrame:
    usecols = META_COLS + ["is_imputed"] + genes_hvg
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["subject"] = df["subject"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    df["is_imputed"] = df["is_imputed"].astype(int)
    return df


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    sns.set_theme(style="whitegrid", context="paper")

    out_root = Path(cfg.out_root).resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    header = io_utils.load_gene_header_and_hvg(Path(cfg.csv_path), Path(cfg.hvg_path))
    genes_hvg = header["genes_hvg"]
    df = io_utils.read_expression_subset(Path(cfg.csv_path), genes_hvg)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)

    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    n_parcels = int(len(target_meta))
    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    unified_summary = pd.read_csv(Path(cfg.unified_phase2_root) / "tables" / "allgenes_subject_loro_summary.csv")
    rep = _select_representative_subject(unified_summary, cfg.c_min)
    rep_subject = str(rep.iloc[0]["subject"])
    rep.to_csv(tab_dir / "representative_subject_manifest.csv", index=False)

    eligible_subjects = set(unified_summary["subject"].astype(str).tolist())
    gtex_raw_eligible = gtex_raw[gtex_raw["subject"].astype(str).isin(eligible_subjects)].copy()

    hm_cfg = SimpleNamespace(combat_use_covariates=cfg.combat_use_covariates)
    combat = fit_harmonizer(ahba_raw, gtex_raw_eligible, genes_hvg, method="combat", cfg=hm_cfg)
    ahba_h = combat.transform(ahba_raw, "AHBA")
    gtex_h = combat.transform(gtex_raw_eligible, "GTEX")

    ahba_raw_mat = _parcel_gene_matrix(ahba_raw, genes_hvg, n_parcels)
    ahba_h_mat = _parcel_gene_matrix(ahba_h, genes_hvg, n_parcels)
    gtex_sparse_raw = _parcel_gene_matrix(gtex_raw_eligible, genes_hvg, n_parcels)
    gtex_sparse_h = _parcel_gene_matrix(gtex_h, genes_hvg, n_parcels)
    global_obs_idx = np.sort(gtex_raw_eligible["parcel_idx"].astype(np.int32).unique())
    global_obs_mask = np.zeros(n_parcels, dtype=bool)
    global_obs_mask[global_obs_idx] = True
    gtex_sparse_raw[~global_obs_mask, :] = np.nan
    gtex_sparse_h[~global_obs_mask, :] = np.nan

    parcel_df = _select_top_parcels(ahba_h_mat, target_meta, cfg.top_n_parcels)
    parcel_df.to_csv(tab_dir / "parcel_legend_top10.csv", index=False)
    gene_df = _select_top_genes(ahba_h_mat, genes_hvg, cfg.top_n_genes)

    det_h = pd.read_csv(Path(cfg.deterministic_root) / "tables" / f"aggregate_allgenes_{DET_MODEL}_mean_harmonized.csv", usecols=["parcel_idx"] + genes_hvg).sort_values("parcel_idx")
    det_filled_h = det_h[genes_hvg].to_numpy(dtype=np.float64)

    combined = _read_combined_harmonized(Path(cfg.unified_export_root) / "gxp_completed_harmonized_combined.csv", genes_hvg)
    combined = combined[combined["subject"].astype(str).isin(eligible_subjects) | (combined["dataset"].str.upper() == "AHBA")].copy()
    combined["parcel_idx"] = combined["tissue_or_parcel"].map(lk).astype(np.int32)
    unified_gtex = combined[combined["dataset"].str.upper() == "GTEX"].copy()
    unified_filled_h = _parcel_gene_matrix(unified_gtex, genes_hvg, n_parcels)
    unified_sparse_h = _parcel_gene_matrix(unified_gtex[unified_gtex["is_imputed"] == 0].copy(), genes_hvg, n_parcels)
    unified_sparse_h[~global_obs_mask, :] = np.nan
    naive_filled_h = np.where(np.isfinite(gtex_sparse_h), gtex_sparse_h, ahba_h_mat)

    rep_sparse_raw = _parcel_gene_matrix(gtex_raw_eligible[gtex_raw_eligible["subject"] == rep_subject].copy(), genes_hvg, n_parcels)
    rep_sparse_h = _parcel_gene_matrix(gtex_h[gtex_h["subject"] == rep_subject].copy(), genes_hvg, n_parcels)
    rep_obs_idx = np.sort(gtex_raw_eligible[gtex_raw_eligible["subject"] == rep_subject]["parcel_idx"].astype(np.int32).unique())
    rep_obs_mask = np.zeros(n_parcels, dtype=bool)
    rep_obs_mask[rep_obs_idx] = True
    rep_sparse_raw[~rep_obs_mask, :] = np.nan
    rep_sparse_h[~rep_obs_mask, :] = np.nan

    ahba_h_full, _ = build_region_matrix(ahba_h, genes_hvg, target_meta, agg="mean")
    ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
    rep_subj_h = gtex_h[gtex_h["subject"] == rep_subject].copy()
    rep_subj_raw = gtex_raw_eligible[gtex_raw_eligible["subject"] == rep_subject].copy()
    rep_obs_idx2, rep_xh, rep_xr = build_subject_observed_matrices(rep_subj_h, rep_subj_raw, genes_hvg)
    det_pred, _ = run_subject(
        {
            "subject": rep_subject,
            "obs_idx": rep_obs_idx2,
            "X_obs_h": rep_xh,
            "X_obs_raw": rep_xr,
            "coords_full": coords_full,
            "target_meta": target_meta,
        },
        {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_pls["T"]},
        {
            "harmonizer": combat,
            "basis_model": "affine_gl3",
            "strategy": "constrained_anchor",
            "spatial_method": "rbf",
            "n_comp_target": cfg.n_comp_target,
            "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
            "rbf_smoothing": cfg.rbf_smoothing,
            "gp_rbf_length": cfg.gp_rbf_length,
            "seed": cfg.seed,
            "c_min": cfg.c_min,
            "distance_d0": 45.0,
            "distance_tau": 10.0,
            "uncertainty_shrink": False,
        },
        vars(cfg),
    )
    det_sub_filled_h = np.asarray(det_pred["X_full_h"], dtype=np.float64)

    unified_rep = unified_gtex[unified_gtex["subject"] == rep_subject].copy()
    unified_sub_filled_h = _parcel_gene_matrix(unified_rep, genes_hvg, n_parcels)
    unified_sub_sparse_h = _parcel_gene_matrix(unified_rep[unified_rep["is_imputed"] == 0].copy(), genes_hvg, n_parcels)
    unified_sub_sparse_h[~rep_obs_mask, :] = np.nan
    naive_sub_filled_h = np.where(np.isfinite(rep_sparse_h), rep_sparse_h, ahba_h_mat)

    _plot_legend_top10_parcels(parcel_df, fig_dir / "legend_top10_parcels.pdf")
    _plot_legend_top_genes(gene_df, fig_dir / "legend_top_genes.pdf")
    _plot_schematic_deterministic(fig_dir / "schematic_deterministic_pipeline.pdf")
    _plot_schematic_unified(fig_dir / "schematic_unified_pipeline.pdf")
    _plot_schematic_graphical_model(fig_dir / "schematic_unified_graphical_model.pdf")

    _plot_heatmap_demo(ahba_raw_mat, gtex_sparse_raw, gtex_sparse_h, ahba_h_mat, det_filled_h, fig_dir / "atlas_deterministic_heatmap_demo.pdf")
    _plot_heatmap_demo(ahba_raw_mat, gtex_sparse_raw, unified_sparse_h, ahba_h_mat, unified_filled_h, fig_dir / "atlas_unified_heatmap_demo.pdf")
    _plot_heatmap_demo(
        ahba_raw_mat,
        gtex_sparse_raw,
        gtex_sparse_h,
        ahba_h_mat,
        naive_filled_h,
        fig_dir / "atlas_naive_heatmap_demo.pdf",
        filled_title="Naive-fill GTEx harmonized",
        residual_title="Residual: naive fill - AHBA",
    )
    _plot_heatmap_demo(ahba_raw_mat, rep_sparse_raw, rep_sparse_h, ahba_h_mat, det_sub_filled_h, fig_dir / "subject_deterministic_heatmap_demo.pdf")
    _plot_heatmap_demo(ahba_raw_mat, rep_sparse_raw, unified_sub_sparse_h, ahba_h_mat, unified_sub_filled_h, fig_dir / "subject_unified_heatmap_demo.pdf")
    _plot_heatmap_demo(
        ahba_raw_mat,
        rep_sparse_raw,
        rep_sparse_h,
        ahba_h_mat,
        naive_sub_filled_h,
        fig_dir / "subject_naive_heatmap_demo.pdf",
        filled_title="Naive-fill GTEx harmonized",
        residual_title="Residual: naive fill - AHBA",
    )

    det_atlas_long = _build_long_from_matrices(ahba_h_mat, det_filled_h, genes_hvg, target_meta, global_obs_mask)
    uni_atlas_long = _build_long_from_matrices(ahba_h_mat, unified_filled_h, genes_hvg, target_meta, global_obs_mask)
    naive_atlas_long = _build_long_from_matrices(ahba_h_mat, naive_filled_h, genes_hvg, target_meta, global_obs_mask)
    det_subject_long = _build_long_from_matrices(ahba_h_mat, det_sub_filled_h, genes_hvg, target_meta, rep_obs_mask)
    uni_subject_long = _build_long_from_matrices(ahba_h_mat, unified_sub_filled_h, genes_hvg, target_meta, rep_obs_mask)
    naive_subject_long = _build_long_from_matrices(ahba_h_mat, naive_sub_filled_h, genes_hvg, target_meta, rep_obs_mask)

    _plot_scatter_parcel_identity(det_atlas_long, parcel_df, fig_dir / "atlas_deterministic_scatter_region_harmonized.pdf")
    _plot_scatter_gene_identity(det_atlas_long, gene_df, fig_dir / "atlas_deterministic_scatter_gene_harmonized.pdf")
    _plot_scatter_parcel_corr(det_atlas_long, target_meta, fig_dir / "atlas_deterministic_scatter_region_corr.pdf")
    _plot_scatter_gene_corr(det_atlas_long, genes_hvg, fig_dir / "atlas_deterministic_scatter_gene_corr.pdf")

    _plot_scatter_parcel_identity(uni_atlas_long, parcel_df, fig_dir / "atlas_unified_scatter_region_harmonized.pdf")
    _plot_scatter_gene_identity(uni_atlas_long, gene_df, fig_dir / "atlas_unified_scatter_gene_harmonized.pdf")
    _plot_scatter_parcel_corr(uni_atlas_long, target_meta, fig_dir / "atlas_unified_scatter_region_corr.pdf")
    _plot_scatter_gene_corr(uni_atlas_long, genes_hvg, fig_dir / "atlas_unified_scatter_gene_corr.pdf")
    _plot_scatter_parcel_identity(naive_atlas_long, parcel_df, fig_dir / "atlas_naive_scatter_region_harmonized.pdf")
    _plot_scatter_gene_identity(naive_atlas_long, gene_df, fig_dir / "atlas_naive_scatter_gene_harmonized.pdf")
    _plot_scatter_parcel_corr(naive_atlas_long, target_meta, fig_dir / "atlas_naive_scatter_region_corr.pdf")
    _plot_scatter_gene_corr(naive_atlas_long, genes_hvg, fig_dir / "atlas_naive_scatter_gene_corr.pdf")

    _plot_subject_scatter_suite(det_subject_long, target_meta, parcel_df, gene_df, fig_dir / "subject_deterministic_scatter_harmonized.pdf")
    _plot_subject_scatter_suite(uni_subject_long, target_meta, parcel_df, gene_df, fig_dir / "subject_unified_scatter_harmonized.pdf")
    _plot_subject_scatter_suite(naive_subject_long, target_meta, parcel_df, gene_df, fig_dir / "subject_naive_scatter_harmonized.pdf")

    qc = {
        "deterministic_model": DET_MODEL,
        "probabilistic_model": UNI_MODEL,
        "naive_model": "harmonized_ahba_fill",
        "representative_subject": rep_subject,
        "figure_files": sorted([p.name for p in fig_dir.glob("*.pdf")]),
        "legend_counts": {"top_parcels": int(len(parcel_df)), "top_genes": int(len(gene_df))},
        "top_parcels": parcel_df[["parcel_idx", "tissue_or_parcel"]].to_dict(orient="records"),
        "page_layout_notes": "GridSpec-based layouts with dedicated legend/colorbar panes; no figure suptitles.",
        "build_timestamp": io_utils.utc_timestamp(),
    }
    (tab_dir / "figure_qc_manifest.json").write_text(json.dumps(qc, indent=2))
    print(f"Built scientific manuscript assets under {out_root}")


if __name__ == "__main__":
    main()
