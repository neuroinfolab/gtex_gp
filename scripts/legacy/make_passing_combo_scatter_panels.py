#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.preprocess import add_sample_groups, add_target_meta, build_region_matrix, build_target_parcels, map_gtex_to_target

META_COLS = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]
META_CORE_COLS = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z"]


@dataclass
class Config:
    run_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment"
    csv_path: str = "/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv"
    hvg_path: str = "/Users/erdem/Documents/github/gtex_gp/ahba_100hvg.txt"
    out_fig_dir: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment/figures/passing_combo_scatter"
    out_tab_dir: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment/tables/passing_combo_scatter"
    top_n_gene_legend: int = 15
    highlight_k: int = 5


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate faceted Allen-vs-GTEx scatter panels for passing vNext combos.")
    p.add_argument("--run-root", default=Config.run_root)
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-fig-dir", default=Config.out_fig_dir)
    p.add_argument("--out-tab-dir", default=Config.out_tab_dir)
    p.add_argument("--top-n-gene-legend", type=int, default=Config.top_n_gene_legend)
    p.add_argument("--highlight-k", type=int, default=Config.highlight_k)
    a = p.parse_args()
    return Config(
        run_root=a.run_root,
        csv_path=a.csv_path,
        hvg_path=a.hvg_path,
        out_fig_dir=a.out_fig_dir,
        out_tab_dir=a.out_tab_dir,
        top_n_gene_legend=max(1, int(a.top_n_gene_legend)),
        highlight_k=max(1, int(a.highlight_k)),
    )


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _shape_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor="#999999", markeredgecolor="black", label="Observed GTEx parcel"),
        Line2D([0], [0], marker="^", linestyle="", markersize=6, markerfacecolor="#999999", markeredgecolor="black", label="Unobserved GTEx parcel"),
    ]


def _facet_grid_shape(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    return 2, 3


def _shared_limits(long_by_combo: Dict[str, pd.DataFrame]) -> Tuple[float, float, float, float]:
    x = np.concatenate([d["allen_value"].to_numpy(dtype=np.float64) for d in long_by_combo.values()])
    y = np.concatenate([d["gtex_value"].to_numpy(dtype=np.float64) for d in long_by_combo.values()])
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return -1.0, 1.0, -1.0, 1.0
    xx = x[m]
    yy = y[m]
    xlo, xhi = np.percentile(xx, [1, 99]).tolist()
    ylo, yhi = np.percentile(yy, [1, 99]).tolist()
    px = 0.03 * (xhi - xlo + 1e-6)
    py = 0.03 * (yhi - ylo + 1e-6)
    return xlo - px, xhi + px, ylo - py, yhi + py


def _build_legend_figure(items: List[Tuple[str, Tuple[float, float, float, float]]], title: str, out_path: Path, ncol: int = 4) -> None:
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=c, markeredgecolor="black", label=lab) for lab, c in items]
    fig = plt.figure(figsize=(4.2 * ncol, max(4.5, 0.30 * len(items))))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=7)
    plt.axis("off")
    plt.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _annotate_top_bottom(ax, corr_df: pd.DataFrame, key_col: str, val_col: str, k: int) -> None:
    s = corr_df[[key_col, val_col]].dropna().copy()
    if len(s) == 0:
        return
    s_top = s.sort_values(val_col, ascending=False).head(k)
    s_bot = s.sort_values(val_col, ascending=True).head(k)
    top_txt = ", ".join([str(x) for x in s_top[key_col].tolist()])
    bot_txt = ", ".join([str(x) for x in s_bot[key_col].tolist()])
    msg = f"Top: {top_txt}\nBottom: {bot_txt}"
    ax.text(
        0.02,
        0.02,
        msg,
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.75),
    )


def _load_passing_combos(run_root: Path) -> pd.DataFrame:
    gate_path = run_root / "tables" / "rollout_gate_summary_consolidated.csv"
    df = pd.read_csv(gate_path)
    need_cols = {"model_combo", "mean_pearson", "gate_pass"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"Gate table missing columns: {sorted(miss)}")
    out = df[df["gate_pass"] == True].copy().sort_values("mean_pearson", ascending=False).reset_index(drop=True)
    if len(out) != 6:
        raise RuntimeError(f"Expected exactly 6 passing combos, found {len(out)}")
    return out


def _prepare_data(cfg: Config):
    csv_path = Path(cfg.csv_path)
    hvg_path = Path(cfg.hvg_path)
    header = io_utils.load_gene_header_and_hvg(csv_path, hvg_path)
    genes = header["genes_hvg"]
    if len(genes) == 0:
        raise RuntimeError("No overlapping HVGs found")
    df = io_utils.read_expression_subset(csv_path, genes)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    return genes, ahba_raw, gtex_raw, target_meta


def _read_combo_gtex_tables(run_root: Path, combo: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tdir = run_root / "tables"
    raw = pd.read_csv(tdir / f"aggregate_hvg_{combo}_mean_raw.csv")
    harm = pd.read_csv(tdir / f"aggregate_hvg_{combo}_mean_harmonized.csv")
    return raw, harm


def _validate_meta_and_genes(df: pd.DataFrame, genes: List[str], tag: str) -> None:
    miss_m = [c for c in META_CORE_COLS if c not in df.columns]
    if miss_m:
        raise RuntimeError(f"{tag} missing meta cols: {miss_m}")
    miss_g = [g for g in genes if g not in df.columns]
    if miss_g:
        raise RuntimeError(f"{tag} missing gene cols count={len(miss_g)}")


def _align_to_index(df: pd.DataFrame, idx: pd.DataFrame, genes: List[str], tag: str) -> pd.DataFrame:
    cols = ["parcel_idx"] + [g for g in genes if g in df.columns]
    sub = df[cols].copy().sort_values("parcel_idx").drop_duplicates("parcel_idx", keep="last")
    x = idx[META_COLS].merge(sub, on="parcel_idx", how="left", validate="one_to_one")
    if x[genes].isna().any().any():
        raise RuntimeError(f"{tag} alignment generated NaNs")
    return x


def _wide_to_long(gtex: pd.DataFrame, allen: pd.DataFrame, genes: List[str], combo: str, scale: str) -> pd.DataFrame:
    g_long = gtex.melt(id_vars=META_COLS, value_vars=genes, var_name="gene", value_name="gtex_value")
    a_long = allen.melt(id_vars=META_COLS, value_vars=genes, var_name="gene", value_name="allen_value")
    long = g_long.merge(a_long[META_COLS + ["gene", "allen_value"]], on=META_COLS + ["gene"], how="inner", validate="one_to_one")
    long["combo"] = combo
    long["scale"] = scale
    return long


def _region_correlations(gtex: pd.DataFrame, allen: pd.DataFrame, genes: List[str], combo: str, scale: str) -> pd.DataFrame:
    rows = []
    for i in range(len(gtex)):
        gx = gtex.iloc[i][genes].to_numpy(dtype=np.float64)
        ax = allen.iloc[i][genes].to_numpy(dtype=np.float64)
        rows.append(
            {
                "combo": combo,
                "scale": scale,
                "parcel_idx": int(gtex.iloc[i]["parcel_idx"]),
                "parcel_name": str(gtex.iloc[i]["parcel_name"]),
                "is_observed_gtex": bool(gtex.iloc[i]["is_observed_gtex"]),
                "pearson_r": _pearson_safe(gx, ax),
            }
        )
    return pd.DataFrame(rows)


def _gene_correlations(gtex: pd.DataFrame, allen: pd.DataFrame, genes: List[str], combo: str, scale: str) -> pd.DataFrame:
    rows = []
    for gene in genes:
        gx = gtex[gene].to_numpy(dtype=np.float64)
        ax = allen[gene].to_numpy(dtype=np.float64)
        rows.append(
            {
                "combo": combo,
                "scale": scale,
                "gene": gene,
                "pearson_r": _pearson_safe(gx, ax),
                "allen_var": float(np.nanvar(ax)),
                "gtex_var": float(np.nanvar(gx)),
            }
        )
    return pd.DataFrame(rows)


def _plot_region_color_faceted(order: List[str], long_by_combo: Dict[str, pd.DataFrame], region_colors: Dict[int, Tuple[float, float, float, float]], out_path: Path, title: str) -> None:
    n = len(order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.7 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_combo)

    for i, combo in enumerate(order):
        ax = axes[i]
        d = long_by_combo[combo]
        un = d[d["is_observed_gtex"] == False]
        ob = d[d["is_observed_gtex"] == True]
        ax.scatter(
            un["allen_value"],
            un["gtex_value"],
            c=[region_colors[int(p)] for p in un["parcel_idx"].to_numpy(dtype=np.int32)],
            s=8,
            alpha=0.60,
            marker="^",
            edgecolor="none",
        )
        ax.scatter(
            ob["allen_value"],
            ob["gtex_value"],
            c=[region_colors[int(p)] for p in ob["parcel_idx"].to_numpy(dtype=np.int32)],
            s=10,
            alpha=0.85,
            marker="o",
            edgecolor="black",
            linewidth=0.12,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(combo, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(handles=_shape_handles(), loc="best", fontsize=7, frameon=True)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_gene_color_faceted(order: List[str], long_by_combo: Dict[str, pd.DataFrame], gene_colors: Dict[str, Tuple[float, float, float, float]], top_genes: List[str], out_path: Path, title: str) -> None:
    n = len(order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.7 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_combo)

    for i, combo in enumerate(order):
        ax = axes[i]
        d = long_by_combo[combo]
        un = d[d["is_observed_gtex"] == False]
        ob = d[d["is_observed_gtex"] == True]
        ax.scatter(
            un["allen_value"],
            un["gtex_value"],
            c=[gene_colors[str(g)] for g in un["gene"].astype(str).tolist()],
            s=8,
            alpha=0.60,
            marker="^",
            edgecolor="none",
        )
        ax.scatter(
            ob["allen_value"],
            ob["gtex_value"],
            c=[gene_colors[str(g)] for g in ob["gene"].astype(str).tolist()],
            s=10,
            alpha=0.85,
            marker="o",
            edgecolor="black",
            linewidth=0.12,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(combo, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    shape_h = _shape_handles()
    gene_h = [Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=gene_colors[g], markeredgecolor="black", label=g) for g in top_genes]
    leg1 = axes[0].legend(handles=shape_h, loc="upper left", fontsize=7, frameon=True, title="Markers", title_fontsize=7)
    axes[0].add_artist(leg1)
    axes[0].legend(handles=gene_h, loc="lower left", fontsize=6, frameon=True, title=f"Top {len(top_genes)} genes", title_fontsize=7)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_corr_faceted(
    order: List[str],
    long_by_combo: Dict[str, pd.DataFrame],
    corr_by_combo: Dict[str, pd.DataFrame],
    key_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    highlight_k: int,
) -> None:
    n = len(order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.7 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_combo)

    cmap = cm.get_cmap("coolwarm")
    norm = Normalize(vmin=-1.0, vmax=1.0)

    for i, combo in enumerate(order):
        ax = axes[i]
        d = long_by_combo[combo].copy()
        cdf = corr_by_combo[combo].copy()
        cmap_dict = dict(zip(cdf[key_col].astype(str), cdf["pearson_r"].astype(float)))
        d["corr_value"] = d[key_col].astype(str).map(cmap_dict).astype(float)

        un = d[d["is_observed_gtex"] == False]
        ob = d[d["is_observed_gtex"] == True]
        ax.scatter(
            un["allen_value"],
            un["gtex_value"],
            c=un["corr_value"],
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.60,
            marker="^",
            edgecolor="none",
        )
        sc = ax.scatter(
            ob["allen_value"],
            ob["gtex_value"],
            c=ob["corr_value"],
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.85,
            marker="o",
            edgecolor="black",
            linewidth=0.12,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(combo, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")
        _annotate_top_bottom(ax, cdf.rename(columns={label_col: "label"}), "label", "pearson_r", highlight_k)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(handles=_shape_handles(), loc="best", fontsize=7, frameon=True)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:n], shrink=0.9)
    cbar.set_label("Pearson r (Allen vs GTEx)")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()

    run_root = Path(cfg.run_root).resolve()
    out_fig = Path(cfg.out_fig_dir).resolve()
    out_tab = Path(cfg.out_tab_dir).resolve()
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    passing_df = _load_passing_combos(run_root)
    combos = passing_df["model_combo"].astype(str).tolist()
    passing_df.to_csv(out_tab / "approach_manifest.csv", index=False)

    genes, ahba_raw, gtex_raw, target_meta = _prepare_data(cfg)

    # Build per-harmonization Allen harmonized references (strict per-family scale compatibility).
    harmonizers_needed = sorted({c.split("__", 1)[0] for c in combos})
    family_to_allen_h: Dict[str, np.ndarray] = {}
    hcfg = SimpleNamespace(
        whiten_eps=1e-4,
        combat_use_covariates=True,
        hier_lambda_a=10.0,
        hier_lambda_b=10.0,
        hier_base_method="robustz_affine",
    )

    ahba_raw_mat, _ = build_region_matrix(ahba_raw, genes, target_meta, agg="mean")
    _, gtex_obs_mask = build_region_matrix(gtex_raw, genes, target_meta, agg="mean")

    for fam in harmonizers_needed:
        harm = fit_harmonizer(ahba_raw, gtex_raw, genes, method=fam, cfg=hcfg)
        ahba_h = harm.transform(ahba_raw, "AHBA")
        ahba_h_mat, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")
        family_to_allen_h[fam] = ahba_h_mat

    # canonical metadata from first passing combo file
    g0_raw, _ = _read_combo_gtex_tables(run_root, combos[0])
    _validate_meta_and_genes(g0_raw, genes, f"{combos[0]} raw")
    canonical_idx = g0_raw[META_CORE_COLS].sort_values("parcel_idx").reset_index(drop=True)

    # overwrite observed mask from current data mapping for consistency
    canonical_idx["is_observed_gtex"] = gtex_obs_mask[canonical_idx["parcel_idx"].to_numpy(dtype=np.int32)]

    allen_raw_df = pd.concat(
        [
            canonical_idx.copy(),
            pd.DataFrame(ahba_raw_mat[canonical_idx["parcel_idx"].to_numpy(dtype=np.int32), :], columns=genes),
        ],
        axis=1,
    )

    # color maps
    parcel_ids = canonical_idx["parcel_idx"].astype(int).tolist()
    region_palette = sns.color_palette("husl", n_colors=len(parcel_ids))
    region_colors = {int(pid): tuple(region_palette[i]) for i, pid in enumerate(parcel_ids)}

    gene_palette = sns.color_palette("husl", n_colors=len(genes))
    gene_colors = {str(g): tuple(gene_palette[i]) for i, g in enumerate(genes)}

    pd.DataFrame(
        {
            "parcel_idx": parcel_ids,
            "parcel_name": canonical_idx["parcel_name"].astype(str).tolist(),
            "r": [region_colors[p][0] for p in parcel_ids],
            "g": [region_colors[p][1] for p in parcel_ids],
            "b": [region_colors[p][2] for p in parcel_ids],
        }
    ).to_csv(out_tab / "region_color_map.csv", index=False)
    pd.DataFrame(
        {
            "gene": genes,
            "r": [gene_colors[str(g)][0] for g in genes],
            "g": [gene_colors[str(g)][1] for g in genes],
            "b": [gene_colors[str(g)][2] for g in genes],
        }
    ).to_csv(out_tab / "gene_color_map.csv", index=False)

    gene_var = np.nanvar(allen_raw_df[genes].to_numpy(dtype=np.float64), axis=0)
    top_gene_idx = np.argsort(-gene_var)[: min(cfg.top_n_gene_legend, len(genes))]
    top_genes = [genes[i] for i in top_gene_idx.tolist()]

    long_raw_by_combo: Dict[str, pd.DataFrame] = {}
    long_h_by_combo: Dict[str, pd.DataFrame] = {}
    rc_raw_by_combo: Dict[str, pd.DataFrame] = {}
    rc_h_by_combo: Dict[str, pd.DataFrame] = {}
    gc_raw_by_combo: Dict[str, pd.DataFrame] = {}
    gc_h_by_combo: Dict[str, pd.DataFrame] = {}

    for combo in combos:
        fam = combo.split("__", 1)[0]
        gtex_raw_df, gtex_h_df = _read_combo_gtex_tables(run_root, combo)
        _validate_meta_and_genes(gtex_raw_df, genes, f"{combo} raw")
        _validate_meta_and_genes(gtex_h_df, genes, f"{combo} harmonized")

        gtex_raw_al = _align_to_index(gtex_raw_df, canonical_idx, genes, f"{combo} raw")
        gtex_h_al = _align_to_index(gtex_h_df, canonical_idx, genes, f"{combo} harmonized")

        allen_h_df = pd.concat(
            [
                canonical_idx.copy(),
                pd.DataFrame(family_to_allen_h[fam][canonical_idx["parcel_idx"].to_numpy(dtype=np.int32), :], columns=genes),
            ],
            axis=1,
        )

        lraw = _wide_to_long(gtex_raw_al, allen_raw_df, genes, combo, "raw")
        lhar = _wide_to_long(gtex_h_al, allen_h_df, genes, combo, "harmonized")
        long_raw_by_combo[combo] = lraw
        long_h_by_combo[combo] = lhar

        scombo = _sanitize(combo)
        lraw.to_csv(out_tab / f"long_raw_{scombo}.csv", index=False)
        lhar.to_csv(out_tab / f"long_harmonized_{scombo}.csv", index=False)

        rc_raw = _region_correlations(gtex_raw_al, allen_raw_df, genes, combo, "raw")
        rc_h = _region_correlations(gtex_h_al, allen_h_df, genes, combo, "harmonized")
        gc_raw = _gene_correlations(gtex_raw_al, allen_raw_df, genes, combo, "raw")
        gc_h = _gene_correlations(gtex_h_al, allen_h_df, genes, combo, "harmonized")
        rc_raw_by_combo[combo] = rc_raw
        rc_h_by_combo[combo] = rc_h
        gc_raw_by_combo[combo] = gc_raw
        gc_h_by_combo[combo] = gc_h

        rc_raw.to_csv(out_tab / f"region_correlations_raw_{scombo}.csv", index=False)
        rc_h.to_csv(out_tab / f"region_correlations_harmonized_{scombo}.csv", index=False)
        gc_raw.to_csv(out_tab / f"gene_correlations_raw_{scombo}.csv", index=False)
        gc_h.to_csv(out_tab / f"gene_correlations_harmonized_{scombo}.csv", index=False)

    # plots: raw
    _plot_region_color_faceted(
        combos,
        long_raw_by_combo,
        region_colors,
        out_fig / "scatter_region_color_faceted_raw.png",
        "Allen vs GTEx (Raw) - Color by Brain Region",
    )
    _plot_gene_color_faceted(
        combos,
        long_raw_by_combo,
        gene_colors,
        top_genes,
        out_fig / "scatter_gene_color_faceted_raw.png",
        "Allen vs GTEx (Raw) - Color by Gene",
    )
    _plot_corr_faceted(
        combos,
        long_raw_by_combo,
        rc_raw_by_combo,
        key_col="parcel_idx",
        label_col="parcel_name",
        out_path=out_fig / "scatter_region_corrcolor_faceted_raw.png",
        title="Allen vs GTEx (Raw) - Color by Region Correlation",
        highlight_k=cfg.highlight_k,
    )
    _plot_corr_faceted(
        combos,
        long_raw_by_combo,
        gc_raw_by_combo,
        key_col="gene",
        label_col="gene",
        out_path=out_fig / "scatter_gene_corrcolor_faceted_raw.png",
        title="Allen vs GTEx (Raw) - Color by Gene Correlation",
        highlight_k=cfg.highlight_k,
    )

    # plots: harmonized
    _plot_region_color_faceted(
        combos,
        long_h_by_combo,
        region_colors,
        out_fig / "scatter_region_color_faceted_harmonized.png",
        "Allen vs GTEx (Harmonized) - Color by Brain Region",
    )
    _plot_gene_color_faceted(
        combos,
        long_h_by_combo,
        gene_colors,
        top_genes,
        out_fig / "scatter_gene_color_faceted_harmonized.png",
        "Allen vs GTEx (Harmonized) - Color by Gene",
    )
    _plot_corr_faceted(
        combos,
        long_h_by_combo,
        rc_h_by_combo,
        key_col="parcel_idx",
        label_col="parcel_name",
        out_path=out_fig / "scatter_region_corrcolor_faceted_harmonized.png",
        title="Allen vs GTEx (Harmonized) - Color by Region Correlation",
        highlight_k=cfg.highlight_k,
    )
    _plot_corr_faceted(
        combos,
        long_h_by_combo,
        gc_h_by_combo,
        key_col="gene",
        label_col="gene",
        out_path=out_fig / "scatter_gene_corrcolor_faceted_harmonized.png",
        title="Allen vs GTEx (Harmonized) - Color by Gene Correlation",
        highlight_k=cfg.highlight_k,
    )

    # legends (same mapping for raw and harmonized, emitted separately by request)
    region_items = [(f"{int(pid)}:{str(canonical_idx.loc[i, 'parcel_name'])}", region_colors[int(pid)]) for i, pid in enumerate(parcel_ids)]
    gene_items_top = [(str(g), gene_colors[str(g)]) for g in top_genes]
    _build_legend_figure(region_items, "Region color legend (raw)", out_fig / "legend_region_colors_raw.png", ncol=6)
    _build_legend_figure(region_items, "Region color legend (harmonized)", out_fig / "legend_region_colors_harmonized.png", ncol=6)
    _build_legend_figure(gene_items_top, f"Gene color legend (raw, top {len(top_genes)})", out_fig / "legend_gene_colors_raw.png", ncol=3)
    _build_legend_figure(gene_items_top, f"Gene color legend (harmonized, top {len(top_genes)})", out_fig / "legend_gene_colors_harmonized.png", ncol=3)

    qc = {
        "n_passing_combos": int(len(combos)),
        "combos": combos,
        "n_parcels": int(len(canonical_idx)),
        "n_genes": int(len(genes)),
        "expected_points_per_combo_scale": int(len(canonical_idx) * len(genes)),
        "observed_parcels": int(canonical_idx["is_observed_gtex"].sum()),
        "unobserved_parcels": int((~canonical_idx["is_observed_gtex"].astype(bool)).sum()),
        "figures": sorted([p.name for p in out_fig.glob("*.png")]),
        "tables": sorted([p.name for p in out_tab.glob("*.csv")]),
    }
    (out_tab / "plot_qc_summary.json").write_text(json.dumps(qc, indent=2))

    print("Generated passing combo scatter suite")
    print(f"Figures: {out_fig}")
    print(f"Tables: {out_tab}")


if __name__ == "__main__":
    main()
