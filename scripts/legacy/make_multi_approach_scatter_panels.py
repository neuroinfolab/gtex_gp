#!/usr/bin/env python3
"""
Build faceted multi-approach Allen vs GTEx scatter panels for top-6 approaches.

Outputs under out/approach_scatter_comparison/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns

META_COLS = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]


@dataclass
class Config:
    out_root: str = "out/approach_scatter_comparison"
    top_n_gene_legend: int = 15
    highlight_k: int = 5


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Create faceted multi-approach scatter comparison suite.")
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--top-n-gene-legend", type=int, default=Config.top_n_gene_legend)
    p.add_argument("--highlight-k", type=int, default=Config.highlight_k)
    a = p.parse_args()
    return Config(out_root=a.out_root, top_n_gene_legend=max(1, int(a.top_n_gene_legend)), highlight_k=max(1, int(a.highlight_k)))


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _shape_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#888888", markeredgecolor="black", label="Observed"),
        Line2D([0], [0], marker="^", linestyle="", markersize=7, markerfacecolor="#888888", markeredgecolor="black", label="Unobserved"),
    ]


def _load_manifest() -> pd.DataFrame:
    """Decision-complete file mapping from user specification."""
    root_mit = Path("/Users/erdem/Documents/github/gtex_gp/out/o3_subject_misfit_mitigation/tables")
    root_w = Path("/Users/erdem/Documents/github/gtex_gp/out/o3_subject_model_grid_whitening/tables")

    rows = []
    # mitigation strategies
    for app in ["baseline", "distance_shrink", "constrained_anchor", "piecewise_harmonization"]:
        rows.append(
            {
                "approach": app,
                "source": "mitigation",
                "gtex_raw": str(root_mit / f"aggregate_hvg_{app}_mean_raw.csv"),
                "gtex_harmonized": str(root_mit / f"aggregate_hvg_{app}_mean_harmonized.csv"),
            }
        )

    # prior approaches from whitening run
    for app in ["whiten_zca_affine__affine_gl3", "zscore_affine__affine_gl3"]:
        rows.append(
            {
                "approach": app,
                "source": "whitening",
                "gtex_raw": str(root_w / f"aggregate_hvg_{app}_mean_raw.csv"),
                "gtex_harmonized": str(root_w / f"aggregate_hvg_{app}_mean_harmonized.csv"),
            }
        )
    return pd.DataFrame(rows)


def _validate_and_align(
    gtex_df: pd.DataFrame,
    allen_df: pd.DataFrame,
    idx_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    for name, df in [("gtex", gtex_df), ("allen", allen_df), ("index", idx_df)]:
        miss = [c for c in META_COLS if c not in df.columns]
        if miss:
            raise RuntimeError(f"{name} missing metadata columns: {miss}")

    gene_g = [c for c in gtex_df.columns if c not in META_COLS]
    gene_a = [c for c in allen_df.columns if c not in META_COLS]
    if gene_g != gene_a:
        raise RuntimeError("Gene columns mismatch between GTEx and Allen tables.")

    idx = idx_df[META_COLS].sort_values("parcel_idx").reset_index(drop=True)
    g = idx.merge(gtex_df[META_COLS + gene_g], on=META_COLS, how="left", validate="one_to_one")
    a = idx.merge(allen_df[META_COLS + gene_g], on=META_COLS, how="left", validate="one_to_one")

    if g.isna().any().any() or a.isna().any().any():
        raise RuntimeError("Missing values after alignment merge.")
    if not np.array_equal(g["parcel_idx"].to_numpy(), a["parcel_idx"].to_numpy()):
        raise RuntimeError("parcel_idx mismatch after alignment.")

    return g, a, gene_g


def _wide_to_long(g: pd.DataFrame, a: pd.DataFrame, genes: List[str], scale: str, approach: str) -> pd.DataFrame:
    gl = g.melt(id_vars=META_COLS, value_vars=genes, var_name="gene", value_name="gtex_value")
    al = a.melt(id_vars=META_COLS, value_vars=genes, var_name="gene", value_name="allen_value")
    long = gl.merge(al[META_COLS + ["gene", "allen_value"]], on=META_COLS + ["gene"], how="inner", validate="one_to_one")
    long["scale"] = scale
    long["approach"] = approach
    return long


def _region_corr(g: pd.DataFrame, a: pd.DataFrame, genes: List[str], approach: str, scale: str) -> pd.DataFrame:
    rows = []
    for i in range(len(g)):
        gx = g.iloc[i][genes].to_numpy(dtype=np.float64)
        ax = a.iloc[i][genes].to_numpy(dtype=np.float64)
        r = _pearson_safe(gx, ax)
        rows.append(
            {
                "approach": approach,
                "scale": scale,
                "parcel_idx": int(g.iloc[i]["parcel_idx"]),
                "parcel_name": str(g.iloc[i]["parcel_name"]),
                "is_observed_gtex": bool(g.iloc[i]["is_observed_gtex"]),
                "pearson_r": r,
            }
        )
    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _gene_corr(g: pd.DataFrame, a: pd.DataFrame, genes: List[str], approach: str, scale: str) -> pd.DataFrame:
    rows = []
    for gene in genes:
        gx = g[gene].to_numpy(dtype=np.float64)
        ax = a[gene].to_numpy(dtype=np.float64)
        rows.append(
            {
                "approach": approach,
                "scale": scale,
                "gene": gene,
                "pearson_r": _pearson_safe(gx, ax),
                "allen_var": float(np.nanvar(ax)),
                "gtex_var": float(np.nanvar(gx)),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)


def _facet_grid_shape(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    return 2, 3


def _shared_limits(long_by_app: Dict[str, pd.DataFrame]) -> Tuple[float, float, float, float]:
    x = np.concatenate([df["allen_value"].to_numpy(dtype=np.float64) for df in long_by_app.values()])
    y = np.concatenate([df["gtex_value"].to_numpy(dtype=np.float64) for df in long_by_app.values()])
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return -1, 1, -1, 1
    xx = x[finite]
    yy = y[finite]
    xlo, xhi = np.percentile(xx, [1, 99]).tolist()
    ylo, yhi = np.percentile(yy, [1, 99]).tolist()
    pad_x = 0.03 * (xhi - xlo + 1e-6)
    pad_y = 0.03 * (yhi - ylo + 1e-6)
    return xlo - pad_x, xhi + pad_x, ylo - pad_y, yhi + pad_y


def _plot_faceted_region_color(
    approach_order: List[str],
    long_by_app: Dict[str, pd.DataFrame],
    region_cmap: Dict[int, Tuple[float, float, float, float]],
    out_path: Path,
) -> None:
    n = len(approach_order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_app)

    for i, app in enumerate(approach_order):
        ax = axes[i]
        d = long_by_app[app]
        obs = d[d["is_observed_gtex"] == True]
        un = d[d["is_observed_gtex"] == False]

        ax.scatter(
            un["allen_value"],
            un["gtex_value"],
            c=[region_cmap[int(p)] for p in un["parcel_idx"].to_numpy(dtype=np.int32)],
            s=10,
            alpha=0.65,
            marker="^",
            edgecolor="none",
        )
        ax.scatter(
            obs["allen_value"],
            obs["gtex_value"],
            c=[region_cmap[int(p)] for p in obs["parcel_idx"].to_numpy(dtype=np.int32)],
            s=14,
            alpha=0.85,
            marker="o",
            edgecolor="black",
            linewidth=0.15,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.2)
        ax.set_title(app)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(handles=_shape_handles(), loc="best", fontsize=8, frameon=True)
    fig.suptitle("Allen vs GTEx scatter (color by region)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_faceted_gene_color(
    approach_order: List[str],
    long_by_app: Dict[str, pd.DataFrame],
    gene_cmap: Dict[str, Tuple[float, float, float, float]],
    top_genes: List[str],
    out_path: Path,
) -> None:
    n = len(approach_order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_app)

    for i, app in enumerate(approach_order):
        ax = axes[i]
        d = long_by_app[app]
        obs = d[d["is_observed_gtex"] == True]
        un = d[d["is_observed_gtex"] == False]

        ax.scatter(
            un["allen_value"],
            un["gtex_value"],
            c=[gene_cmap[str(g)] for g in un["gene"]],
            s=10,
            alpha=0.65,
            marker="^",
            edgecolor="none",
        )
        ax.scatter(
            obs["allen_value"],
            obs["gtex_value"],
            c=[gene_cmap[str(g)] for g in obs["gene"]],
            s=14,
            alpha=0.85,
            marker="o",
            edgecolor="black",
            linewidth=0.15,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.2)
        ax.set_title(app)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    gene_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=gene_cmap[g], markeredgecolor="none", label=g)
        for g in top_genes
    ]
    axes[0].legend(handles=_shape_handles() + gene_handles, loc="best", fontsize=7, frameon=True, ncol=1)
    fig.suptitle("Allen vs GTEx scatter (color by gene)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _annotate_top_bottom_region(ax, d: pd.DataFrame, k: int) -> None:
    top = d.nlargest(k, "region_corr")["parcel_name"].astype(str).tolist()
    bot = d.nsmallest(k, "region_corr")["parcel_name"].astype(str).tolist()
    txt = "top: " + ", ".join(top[:3]) + "\n" + "bottom: " + ", ".join(bot[:3])
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6, bbox=dict(boxstyle="round", alpha=0.2))


def _annotate_top_bottom_gene(ax, d: pd.DataFrame, k: int) -> None:
    top = d.groupby("gene", as_index=False)["gene_corr"].mean().nlargest(k, "gene_corr")["gene"].tolist()
    bot = d.groupby("gene", as_index=False)["gene_corr"].mean().nsmallest(k, "gene_corr")["gene"].tolist()
    txt = "top: " + ", ".join(top[:3]) + "\n" + "bottom: " + ", ".join(bot[:3])
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6, bbox=dict(boxstyle="round", alpha=0.2))


def _plot_faceted_corr_color(
    approach_order: List[str],
    long_by_app: Dict[str, pd.DataFrame],
    kind: str,
    out_path: Path,
    highlight_k: int,
) -> None:
    n = len(approach_order)
    nrow, ncol = _facet_grid_shape(n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    xlo, xhi, ylo, yhi = _shared_limits(long_by_app)

    mappable = None
    for i, app in enumerate(approach_order):
        ax = axes[i]
        d = long_by_app[app]
        if kind == "region":
            cvals = d["region_corr"].to_numpy(dtype=np.float64)
        else:
            cvals = d["gene_corr"].to_numpy(dtype=np.float64)

        obs = d[d["is_observed_gtex"] == True]
        un = d[d["is_observed_gtex"] == False]

        if kind == "region":
            c_un = un["region_corr"].to_numpy(dtype=np.float64)
            c_obs = obs["region_corr"].to_numpy(dtype=np.float64)
        else:
            c_un = un["gene_corr"].to_numpy(dtype=np.float64)
            c_obs = obs["gene_corr"].to_numpy(dtype=np.float64)

        ax.scatter(un["allen_value"], un["gtex_value"], c=c_un, cmap="coolwarm", vmin=-1, vmax=1, s=10, alpha=0.7, marker="^", edgecolor="none")
        mappable = ax.scatter(obs["allen_value"], obs["gtex_value"], c=c_obs, cmap="coolwarm", vmin=-1, vmax=1, s=14, alpha=0.9, marker="o", edgecolor="black", linewidth=0.15)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.2)
        ax.set_title(app)
        ax.set_xlabel("Allen")
        ax.set_ylabel("GTEx")

        if kind == "region":
            _annotate_top_bottom_region(ax, d[["parcel_name", "region_corr"]].drop_duplicates(), highlight_k)
        else:
            _annotate_top_bottom_gene(ax, d[["gene", "gene_corr"]], highlight_k)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    axes[0].legend(handles=_shape_handles(), loc="best", fontsize=8, frameon=True)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.85)
        cbar.set_label("Pearson r (Allen vs GTEx)")
    title = "Allen vs GTEx scatter (color by region correlation)" if kind == "region" else "Allen vs GTEx scatter (color by gene correlation)"
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _legend_region(region_df: pd.DataFrame, region_cmap: Dict[int, Tuple[float, float, float, float]], out_path: Path) -> None:
    items = region_df.sort_values("parcel_idx")[ ["parcel_idx", "parcel_name"] ].to_dict("records")
    n = len(items)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, max(8, 0.28 * nrow * ncol)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, item in enumerate(items):
        ax = axes[i]
        ax.axis("off")
        pid = int(item["parcel_idx"])
        name = str(item["parcel_name"])
        ax.scatter([0.05], [0.5], s=70, c=[region_cmap[pid]])
        ax.text(0.12, 0.5, f"{pid}: {name}", va="center", ha="left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Region color legend")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _legend_gene(top_genes: List[str], gene_cmap: Dict[str, Tuple[float, float, float, float]], out_path: Path) -> None:
    n = len(top_genes)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * n)), constrained_layout=True)
    ax.axis("off")
    for i, g in enumerate(top_genes):
        y = 1.0 - (i + 1) / (n + 1)
        ax.scatter([0.08], [y], s=70, c=[gene_cmap[g]])
        ax.text(0.15, y, g, va="center", ha="left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Gene legend (top-variance 15)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    sns.set_style("whitegrid")
    sns.set_context("talk")

    out_root = Path(cfg.out_root).resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    manifest.to_csv(tab_dir / "approach_manifest.csv", index=False)

    idx_path = Path("/Users/erdem/Documents/github/gtex_gp/out/o3_subject_model_grid_whitening/tables/aligned_row_index.csv")
    allen_raw_path = Path("/Users/erdem/Documents/github/gtex_gp/out/o3_subject_model_grid_whitening/tables/aggregate/robustz_affine__affine_gl3/allen_allen_rows_raw.csv")
    allen_h_path = Path("/Users/erdem/Documents/github/gtex_gp/out/o3_subject_model_grid_whitening/tables/aggregate/robustz_affine__affine_gl3/allen_allen_rows_harmonized.csv")

    idx_df = pd.read_csv(idx_path)
    allen_raw = pd.read_csv(allen_raw_path)
    allen_h = pd.read_csv(allen_h_path)

    approach_order = manifest["approach"].tolist()
    qc = {
        "n_approaches": len(approach_order),
        "approaches": approach_order,
        "scales": ["raw", "harmonized"],
        "shape_checks": {},
        "files": {},
    }

    long_by_scale: Dict[str, Dict[str, pd.DataFrame]] = {"raw": {}, "harmonized": {}}
    region_corr_by_scale: Dict[str, Dict[str, pd.DataFrame]] = {"raw": {}, "harmonized": {}}
    gene_corr_by_scale: Dict[str, Dict[str, pd.DataFrame]] = {"raw": {}, "harmonized": {}}

    canonical_genes = None

    for _, row in manifest.iterrows():
        app = str(row["approach"])
        g_raw = pd.read_csv(Path(str(row["gtex_raw"])))
        g_h = pd.read_csv(Path(str(row["gtex_harmonized"])))

        g_raw_al, a_raw_al, genes_raw = _validate_and_align(g_raw, allen_raw, idx_df)
        g_h_al, a_h_al, genes_h = _validate_and_align(g_h, allen_h, idx_df)

        if genes_raw != genes_h:
            raise RuntimeError(f"Gene list mismatch between raw/harmonized for approach={app}")

        if canonical_genes is None:
            canonical_genes = genes_raw
        elif canonical_genes != genes_raw:
            raise RuntimeError(f"Gene order mismatch vs canonical list for approach={app}")

        qc["shape_checks"][app] = {
            "raw_shape": list(g_raw_al.shape),
            "harm_shape": list(g_h_al.shape),
            "n_genes": len(genes_raw),
        }

        # long + corrs raw
        long_r = _wide_to_long(g_raw_al, a_raw_al, canonical_genes, scale="raw", approach=app)
        rc_r = _region_corr(g_raw_al, a_raw_al, canonical_genes, approach=app, scale="raw")
        gc_r = _gene_corr(g_raw_al, a_raw_al, canonical_genes, approach=app, scale="raw")
        long_r = long_r.merge(rc_r[["parcel_idx", "pearson_r"]].rename(columns={"pearson_r": "region_corr"}), on="parcel_idx", how="left")
        long_r = long_r.merge(gc_r[["gene", "pearson_r"]].rename(columns={"pearson_r": "gene_corr"}), on="gene", how="left")

        # long + corrs harmonized
        long_h = _wide_to_long(g_h_al, a_h_al, canonical_genes, scale="harmonized", approach=app)
        rc_h = _region_corr(g_h_al, a_h_al, canonical_genes, approach=app, scale="harmonized")
        gc_h = _gene_corr(g_h_al, a_h_al, canonical_genes, approach=app, scale="harmonized")
        long_h = long_h.merge(rc_h[["parcel_idx", "pearson_r"]].rename(columns={"pearson_r": "region_corr"}), on="parcel_idx", how="left")
        long_h = long_h.merge(gc_h[["gene", "pearson_r"]].rename(columns={"pearson_r": "gene_corr"}), on="gene", how="left")

        long_by_scale["raw"][app] = long_r
        long_by_scale["harmonized"][app] = long_h
        region_corr_by_scale["raw"][app] = rc_r
        region_corr_by_scale["harmonized"][app] = rc_h
        gene_corr_by_scale["raw"][app] = gc_r
        gene_corr_by_scale["harmonized"][app] = gc_h

        # write per-approach tables
        long_r.to_csv(tab_dir / f"long_raw_{app}.csv", index=False)
        long_h.to_csv(tab_dir / f"long_harmonized_{app}.csv", index=False)
        rc_r.to_csv(tab_dir / f"region_correlations_raw_{app}.csv", index=False)
        rc_h.to_csv(tab_dir / f"region_correlations_harmonized_{app}.csv", index=False)
        gc_r.to_csv(tab_dir / f"gene_correlations_raw_{app}.csv", index=False)
        gc_h.to_csv(tab_dir / f"gene_correlations_harmonized_{app}.csv", index=False)

    # deterministic color maps from canonical ordering
    region_meta = idx_df[["parcel_idx", "parcel_name"]].drop_duplicates().sort_values("parcel_idx").reset_index(drop=True)
    parcel_ids = region_meta["parcel_idx"].to_numpy(dtype=np.int32)
    region_colors = cm.get_cmap("tab20", len(parcel_ids))
    region_cmap = {int(pid): region_colors(i) for i, pid in enumerate(parcel_ids.tolist())}

    genes = list(canonical_genes)
    gene_colors = cm.get_cmap("tab20b", len(genes))
    gene_cmap = {g: gene_colors(i) for i, g in enumerate(genes)}

    pd.DataFrame(
        [{"parcel_idx": int(pid), "parcel_name": str(region_meta.iloc[i]["parcel_name"]), "r": c[0], "g": c[1], "b": c[2], "a": c[3]} for i, (pid, c) in enumerate(region_cmap.items())]
    ).to_csv(tab_dir / "region_color_map.csv", index=False)
    pd.DataFrame(
        [{"gene": g, "r": c[0], "g_col": c[1], "b": c[2], "a": c[3]} for g, c in gene_cmap.items()]
    ).to_csv(tab_dir / "gene_color_map.csv", index=False)

    # top genes by variance across all approaches/scales (for compact legend)
    all_var = []
    for scale in ["raw", "harmonized"]:
        for app in approach_order:
            d = long_by_scale[scale][app]
            gv = d.groupby("gene", as_index=False)["allen_value"].var().rename(columns={"allen_value": "var"})
            gv["approach"] = app
            gv["scale"] = scale
            all_var.append(gv)
    var_df = pd.concat(all_var, ignore_index=True)
    top_genes = (
        var_df.groupby("gene", as_index=False)["var"]
        .mean()
        .sort_values("var", ascending=False)
        .head(cfg.top_n_gene_legend)["gene"]
        .tolist()
    )

    for scale in ["raw", "harmonized"]:
        long_by_app = long_by_scale[scale]
        _plot_faceted_region_color(
            approach_order=approach_order,
            long_by_app=long_by_app,
            region_cmap=region_cmap,
            out_path=fig_dir / f"scatter_region_color_faceted_{scale}.png",
        )
        _plot_faceted_gene_color(
            approach_order=approach_order,
            long_by_app=long_by_app,
            gene_cmap=gene_cmap,
            top_genes=top_genes,
            out_path=fig_dir / f"scatter_gene_color_faceted_{scale}.png",
        )
        _plot_faceted_corr_color(
            approach_order=approach_order,
            long_by_app=long_by_app,
            kind="region",
            out_path=fig_dir / f"scatter_region_corrcolor_faceted_{scale}.png",
            highlight_k=cfg.highlight_k,
        )
        _plot_faceted_corr_color(
            approach_order=approach_order,
            long_by_app=long_by_app,
            kind="gene",
            out_path=fig_dir / f"scatter_gene_corrcolor_faceted_{scale}.png",
            highlight_k=cfg.highlight_k,
        )

        _legend_region(region_meta, region_cmap, fig_dir / f"legend_region_colors_{scale}.png")
        _legend_gene(top_genes, gene_cmap, fig_dir / f"legend_gene_colors_{scale}.png")

    qc["files"]["figures"] = sorted([str(p) for p in fig_dir.glob("*.png")])
    qc["files"]["tables"] = sorted([str(p) for p in tab_dir.glob("*.csv")])
    qc["top_genes_legend"] = top_genes
    qc["n_points_per_approach_scale"] = {
        scale: {app: int(len(long_by_scale[scale][app])) for app in approach_order} for scale in ["raw", "harmonized"]
    }

    (tab_dir / "plot_qc_summary.json").write_text(json.dumps(qc, indent=2))

    print("Multi-approach faceted scatter suite complete.")
    print(f"Outputs: {out_root}")


if __name__ == "__main__":
    main()
