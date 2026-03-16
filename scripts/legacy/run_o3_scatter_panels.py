#!/usr/bin/env python3
"""
Build unified GTEx-vs-Allen HVG scatter panels (observed+unobserved)
with legends, categorical color variants, correlation color variants,
and top/bottom highlight annotations.

Outputs:
  out/o3_pls_consistent/figures/scatter_panels/*.png
  out/o3_pls_consistent/tables/scatter_panels/*.csv
  out/o3_pls_consistent/tables/scatter_panels/plot_qc_summary.json
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


META_COLS = [
    "parcel_idx",
    "parcel_name",
    "coord_x",
    "coord_y",
    "coord_z",
    "is_observed_gtex",
]


@dataclass
class Config:
    gtex_raw_path: str = "out/o3_pls_consistent/tables/gtex_predicted_allen_rows_raw.csv"
    allen_raw_path: str = "out/o3_pls_consistent/tables/allen_allen_rows_raw.csv"
    gtex_harm_path: str = "out/o3_pls_consistent/tables/gtex_predicted_allen_rows_harmonized.csv"
    allen_harm_path: str = "out/o3_pls_consistent/tables/allen_allen_rows_harmonized.csv"
    aligned_index_path: str = "out/o3_pls_consistent/tables/aligned_row_index.csv"
    out_fig_dir: str = "out/o3_pls_consistent/figures/scatter_panels"
    out_tab_dir: str = "out/o3_pls_consistent/tables/scatter_panels"
    shift_frac: float = 0.0
    top_n_gene_legend: int = 15
    top_n_region_legend: int = 15
    highlight_k: int = 5


def _validate_meta_columns(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing metadata columns: {missing}")


def _validate_gene_columns(gtex: pd.DataFrame, allen: pd.DataFrame, scale_name: str) -> List[str]:
    gtex_genes = [c for c in gtex.columns if c not in META_COLS]
    allen_genes = [c for c in allen.columns if c not in META_COLS]
    if len(gtex_genes) == 0:
        raise RuntimeError(f"{scale_name}: no gene columns found.")
    if gtex_genes != allen_genes:
        raise RuntimeError(f"{scale_name}: GTEx/Allen gene columns are not identical/in-order.")
    return gtex_genes


def _align_to_index(df: pd.DataFrame, idx: pd.DataFrame, table_name: str) -> pd.DataFrame:
    merged = idx[META_COLS].merge(
        df,
        on=META_COLS,
        how="left",
        validate="one_to_one",
    )
    if merged.isna().any().any():
        na_cols = [c for c in merged.columns if merged[c].isna().any()]
        raise RuntimeError(f"{table_name}: missing values after index alignment in columns: {na_cols[:10]}")
    return merged


def _read_and_validate_pair(
    gtex_path: Path,
    allen_path: Path,
    idx_path: Path,
    scale_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    gtex = pd.read_csv(gtex_path)
    allen = pd.read_csv(allen_path)
    idx = pd.read_csv(idx_path)
    _validate_meta_columns(gtex, f"GTEx {scale_name}")
    _validate_meta_columns(allen, f"Allen {scale_name}")
    _validate_meta_columns(idx, "Aligned index")

    idx = idx.sort_values("parcel_idx").reset_index(drop=True)
    gtex = gtex.sort_values("parcel_idx").reset_index(drop=True)
    allen = allen.sort_values("parcel_idx").reset_index(drop=True)

    gene_cols = _validate_gene_columns(gtex, allen, scale_name)
    gtex = _align_to_index(gtex[META_COLS + gene_cols], idx, f"GTEx {scale_name}")
    allen = _align_to_index(allen[META_COLS + gene_cols], idx, f"Allen {scale_name}")
    if not np.array_equal(gtex["parcel_idx"].to_numpy(), allen["parcel_idx"].to_numpy()):
        raise RuntimeError(f"{scale_name}: parcel_idx mismatch after alignment.")
    return gtex, allen, idx, gene_cols


def _wide_to_long(gtex: pd.DataFrame, allen: pd.DataFrame, gene_cols: List[str], scale: str) -> pd.DataFrame:
    id_vars = META_COLS
    g_long = gtex.melt(id_vars=id_vars, value_vars=gene_cols, var_name="gene", value_name="gtex_value")
    a_long = allen.melt(id_vars=id_vars, value_vars=gene_cols, var_name="gene", value_name="allen_value")
    keys = id_vars + ["gene"]
    long_df = g_long.merge(a_long[keys + ["allen_value"]], on=keys, how="inner", validate="one_to_one")
    long_df["scale"] = scale
    return long_df


def _iqr(x: np.ndarray) -> float:
    p25, p75 = np.percentile(x, [25.0, 75.0])
    return float(p75 - p25)


def _offset_map(groups: List[str], amp: float) -> Dict[str, float]:
    n = len(groups)
    if n <= 1:
        return {groups[0]: 0.0} if n == 1 else {}
    center = (n - 1) / 2.0
    denom = max(center, 1.0)
    return {g: amp * ((i - center) / denom) for i, g in enumerate(groups)}


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _compute_region_correlations(
    gtex: pd.DataFrame,
    allen: pd.DataFrame,
    gene_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for i in range(len(gtex)):
        gx = gtex.iloc[i][gene_cols].to_numpy(dtype=np.float64)
        ax = allen.iloc[i][gene_cols].to_numpy(dtype=np.float64)
        r = _pearson_safe(gx, ax)
        rows.append(
            {
                "parcel_idx": int(gtex.iloc[i]["parcel_idx"]),
                "parcel_name": str(gtex.iloc[i]["parcel_name"]),
                "coord_x": float(gtex.iloc[i]["coord_x"]),
                "coord_y": float(gtex.iloc[i]["coord_y"]),
                "coord_z": float(gtex.iloc[i]["coord_z"]),
                "is_observed_gtex": bool(gtex.iloc[i]["is_observed_gtex"]),
                "pearson_r": r,
            }
        )
    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)
    out["rank_desc"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def _compute_gene_correlations(
    gtex: pd.DataFrame,
    allen: pd.DataFrame,
    gene_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for g in gene_cols:
        gx = gtex[g].to_numpy(dtype=np.float64)
        ax = allen[g].to_numpy(dtype=np.float64)
        r = _pearson_safe(gx, ax)
        rows.append(
            {
                "gene": g,
                "pearson_r": r,
                "var_allen": float(np.nanvar(ax)),
                "var_gtex": float(np.nanvar(gx)),
            }
        )
    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)
    out["rank_desc"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def _make_color_map(keys: List[str], palette: str = "husl") -> Dict[str, Tuple[float, float, float]]:
    colors = sns.color_palette(palette, n_colors=len(keys))
    return {k: tuple(colors[i]) for i, k in enumerate(keys)}


def _shape_legend_handles() -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor="#777777",
            markeredgecolor="black",
            label="Observed GTEx parcel",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markersize=7,
            markerfacecolor="#777777",
            markeredgecolor="black",
            label="Unobserved GTEx parcel",
        ),
    ]


def _scatter_by_category(
    df: pd.DataFrame,
    category_col: str,
    color_map: Dict[str, Tuple[float, float, float]],
    title: str,
    out_path: Path,
    category_legend_title: str,
    category_legend_handles: List[Patch] | None = None,
    category_legend_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 9))
    for observed, marker in [(True, "o"), (False, "^")]:
        sub = df[df["is_observed_gtex"] == observed]
        if len(sub) == 0:
            continue
        cvals = [color_map[str(v)] for v in sub[category_col].astype(str).tolist()]
        ax.scatter(
            sub["x_plot"],
            sub["gtex_value"],
            c=cvals,
            marker=marker,
            s=16,
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xlabel("Allen expression (original values)")
    ax.set_ylabel("GTEx expression")
    ax.grid(True, alpha=0.2)

    shape_handles = _shape_legend_handles()
    shape_legend = ax.legend(handles=shape_handles, loc="upper left", frameon=True, title="Point shape")
    ax.add_artist(shape_legend)

    if category_legend_handles is not None and len(category_legend_handles) > 0:
        category_legend = ax.legend(
            handles=category_legend_handles,
            loc="upper right",
            frameon=True,
            title=category_legend_title,
            fontsize=8,
            title_fontsize=9,
            ncol=1,
        )
        ax.add_artist(category_legend)
        if category_legend_note:
            ax.text(
                0.995,
                0.02,
                category_legend_note,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=9,
            )
    else:
        if category_legend_note:
            ax.text(
                0.995,
                0.02,
                category_legend_note,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _scatter_by_correlation(
    df: pd.DataFrame,
    corr_col: str,
    title: str,
    out_path: Path,
    highlight_groups: pd.DataFrame,
    group_col: str,
    label_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 9))
    cmap = plt.get_cmap("coolwarm")
    mappable = None
    for observed, marker in [(True, "o"), (False, "^")]:
        sub = df[df["is_observed_gtex"] == observed]
        if len(sub) == 0:
            continue
        sc = ax.scatter(
            sub["x_plot"],
            sub["gtex_value"],
            c=sub[corr_col],
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            marker=marker,
            s=16,
            alpha=0.74,
            linewidths=0.0,
            rasterized=True,
        )
        mappable = sc

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Pearson r (Allen vs GTEx)")

    ax.set_title(title)
    ax.set_xlabel("Allen expression (original values)")
    ax.set_ylabel("GTEx expression")
    ax.grid(True, alpha=0.2)

    shape_legend = ax.legend(handles=_shape_legend_handles(), loc="upper left", frameon=True, title="Point shape")
    ax.add_artist(shape_legend)
    _annotate_highlights(ax=ax, points_df=df, highlight_df=highlight_groups, group_col=group_col, label_col=label_col)

    top_note = "; ".join(
        [f"{row[label_col]} ({row['pearson_r']:.2f})" for _, row in highlight_groups[highlight_groups["extreme"] == "top"].iterrows()]
    )
    bottom_note = "; ".join(
        [f"{row[label_col]} ({row['pearson_r']:.2f})" for _, row in highlight_groups[highlight_groups["extreme"] == "bottom"].iterrows()]
    )
    ax.text(0.01, 0.02, f"Top: {top_note}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    ax.text(0.01, 0.06, f"Bottom: {bottom_note}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _annotate_highlights(
    ax: plt.Axes,
    points_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    group_col: str,
    label_col: str,
) -> None:
    centers = (
        points_df.groupby(group_col)[["x_plot", "gtex_value"]]
        .median()
        .rename(columns={"x_plot": "cx", "gtex_value": "cy"})
        .reset_index()
    )
    merge_df = highlight_df.merge(centers, on=group_col, how="left")
    xspan = float(points_df["x_plot"].max() - points_df["x_plot"].min())
    yspan = float(points_df["gtex_value"].max() - points_df["gtex_value"].min())
    if xspan <= 0:
        xspan = 1.0
    if yspan <= 0:
        yspan = 1.0

    for i, (_, row) in enumerate(merge_df.iterrows()):
        if not np.isfinite(row["cx"]) or not np.isfinite(row["cy"]):
            continue
        direction = -1.0 if row["extreme"] == "bottom" else 1.0
        dx = 0.02 * xspan * ((i % 3) - 1)
        dy = direction * (0.02 * yspan + 0.015 * yspan * (i % 4))
        lbl = f"{row[label_col]} ({row['pearson_r']:.2f})"
        ax.annotate(
            lbl,
            xy=(row["cx"], row["cy"]),
            xytext=(row["cx"] + dx, row["cy"] + dy),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.6, "alpha": 0.7},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gray", "alpha": 0.75},
        )


def _pick_top_bottom(df: pd.DataFrame, value_col: str, label_col: str, k: int) -> pd.DataFrame:
    valid = df[np.isfinite(df[value_col])].copy()
    if len(valid) == 0:
        return pd.DataFrame(columns=list(df.columns) + ["extreme"])
    top = valid.nlargest(k, value_col).copy()
    top["extreme"] = "top"
    bottom = valid.nsmallest(k, value_col).copy()
    bottom["extreme"] = "bottom"
    out = pd.concat([top, bottom], ignore_index=True)
    out = out.drop_duplicates(subset=[label_col, value_col, "extreme"]).reset_index(drop=True)
    return out


def _legend_figure(
    color_map_df: pd.DataFrame,
    key_col: str,
    title: str,
    out_path: Path,
    max_items: int | None = None,
) -> None:
    df = color_map_df.copy()
    if max_items is not None:
        df = df.head(max_items).copy()
    handles = [
        Patch(facecolor=(r["r"], r["g"], r["b"]), edgecolor="none", label=str(r[key_col]))
        for _, r in df.iterrows()
    ]
    n = len(handles)
    ncol = 3 if n <= 30 else 4
    fig_h = max(4.0, min(16.0, 0.22 * n + 2.0))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    lg = ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        title=title,
        ncol=ncol,
        fontsize=8,
        title_fontsize=10,
    )
    ax.add_artist(lg)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _run_for_scale(
    scale: str,
    gtex: pd.DataFrame,
    allen: pd.DataFrame,
    idx: pd.DataFrame,
    gene_cols: List[str],
    cfg: Config,
    fig_dir: Path,
    tab_dir: Path,
) -> Dict[str, object]:
    long_df = _wide_to_long(gtex, allen, gene_cols, scale)
    long_out = tab_dir / f"long_{scale}.csv"
    long_df.to_csv(long_out, index=False)

    amp = max(0.0, cfg.shift_frac * _iqr(long_df["allen_value"].to_numpy(dtype=np.float64)))

    region_keys = [str(x) for x in idx.sort_values("parcel_idx")["parcel_idx"].astype(int).tolist()]
    gene_keys = [str(g) for g in gene_cols]
    region_offsets = _offset_map(region_keys, amp) if amp > 0 else {k: 0.0 for k in region_keys}
    gene_offsets = _offset_map(gene_keys, amp) if amp > 0 else {k: 0.0 for k in gene_keys}

    region_color_map = _make_color_map(region_keys, palette="husl")
    gene_color_map = _make_color_map(gene_keys, palette="husl")

    region_map_df = pd.DataFrame(
        [
            {
                "parcel_idx": int(k),
                "parcel_name": str(idx[idx["parcel_idx"] == int(k)]["parcel_name"].iloc[0]),
                "r": region_color_map[k][0],
                "g": region_color_map[k][1],
                "b": region_color_map[k][2],
            }
            for k in region_keys
        ]
    )
    gene_map_df = pd.DataFrame(
        [
            {
                "gene": k,
                "r": gene_color_map[k][0],
                "g": gene_color_map[k][1],
                "b": gene_color_map[k][2],
            }
            for k in gene_keys
        ]
    )
    region_map_df.to_csv(tab_dir / "region_color_map.csv", index=False)
    gene_map_df.to_csv(tab_dir / "gene_color_map.csv", index=False)

    region_plot_df = long_df.copy()
    region_plot_df["region_key"] = region_plot_df["parcel_idx"].astype(int).astype(str)
    region_plot_df["x_plot"] = region_plot_df["allen_value"] + region_plot_df["region_key"].map(region_offsets).astype(float)

    gene_plot_df = long_df.copy()
    gene_plot_df["gene_key"] = gene_plot_df["gene"].astype(str)
    gene_plot_df["x_plot"] = gene_plot_df["allen_value"] + gene_plot_df["gene_key"].map(gene_offsets).astype(float)

    region_corr_df = _compute_region_correlations(gtex, allen, gene_cols)
    gene_corr_df = _compute_gene_correlations(gtex, allen, gene_cols)
    region_corr_df.to_csv(tab_dir / f"region_correlations_{scale}.csv", index=False)
    gene_corr_df.to_csv(tab_dir / f"gene_correlations_{scale}.csv", index=False)

    region_var = np.nanvar(allen[gene_cols].to_numpy(dtype=np.float64), axis=1)
    region_var_df = pd.DataFrame(
        {
            "parcel_idx": allen["parcel_idx"].astype(int).to_numpy(),
            "parcel_name": allen["parcel_name"].astype(str).to_numpy(),
            "var_allen": region_var,
        }
    ).sort_values("var_allen", ascending=False)
    top_n_region = int(min(cfg.top_n_region_legend, len(region_var_df)))
    top_region_rows = region_var_df.head(top_n_region).copy()
    top_region_handles = [
        Patch(
            facecolor=region_color_map[str(int(r["parcel_idx"]))],
            edgecolor="none",
            label=str(r["parcel_name"]),
        )
        for _, r in top_region_rows.iterrows()
    ]

    _scatter_by_category(
        df=region_plot_df,
        category_col="region_key",
        color_map=region_color_map,
        title=f"GTEx vs Allen HVG scatter (color=region, observed+unobserved) [{scale}]",
        out_path=fig_dir / f"scatter_region_color_combined_{scale}.png",
        category_legend_title=f"Top {top_n_region} regions by Allen variance",
        category_legend_handles=top_region_handles,
        category_legend_note="Full region-color map saved in region_color_map.csv",
    )

    gene_var = gene_corr_df.sort_values("var_allen", ascending=False).reset_index(drop=True)
    top_n = int(min(cfg.top_n_gene_legend, len(gene_var)))
    top_genes = gene_var["gene"].head(top_n).tolist()
    top_handles = [
        Patch(facecolor=gene_color_map[g], edgecolor="none", label=g)
        for g in top_genes
    ]
    _scatter_by_category(
        df=gene_plot_df,
        category_col="gene_key",
        color_map=gene_color_map,
        title=f"GTEx vs Allen HVG scatter (color=gene, observed+unobserved) [{scale}]",
        out_path=fig_dir / f"scatter_gene_color_combined_{scale}.png",
        category_legend_title=f"Top {top_n} genes by Allen variance",
        category_legend_handles=top_handles,
        category_legend_note="Full gene-color map saved in gene_color_map.csv",
    )

    _legend_figure(
        color_map_df=region_map_df.sort_values("parcel_idx").copy(),
        key_col="parcel_name",
        title="Region color legend (all parcels)",
        out_path=fig_dir / f"legend_region_colors_{scale}.png",
        max_items=None,
    )
    _legend_figure(
        color_map_df=gene_map_df.copy(),
        key_col="gene",
        title=f"Gene color legend (top {top_n} shown in main plot; full mapping here)",
        out_path=fig_dir / f"legend_gene_colors_{scale}.png",
        max_items=None,
    )

    region_corr_map = dict(zip(region_corr_df["parcel_idx"].astype(int).astype(str), region_corr_df["pearson_r"]))
    gene_corr_map = dict(zip(gene_corr_df["gene"].astype(str), gene_corr_df["pearson_r"]))
    region_plot_df["pearson_r"] = region_plot_df["region_key"].map(region_corr_map).astype(float)
    gene_plot_df["pearson_r"] = gene_plot_df["gene_key"].map(gene_corr_map).astype(float)

    region_highlights = _pick_top_bottom(
        region_corr_df.rename(columns={"parcel_name": "label"}),
        value_col="pearson_r",
        label_col="label",
        k=cfg.highlight_k,
    ).rename(columns={"label": "parcel_name"})
    gene_highlights = _pick_top_bottom(
        gene_corr_df.rename(columns={"gene": "label"}),
        value_col="pearson_r",
        label_col="label",
        k=cfg.highlight_k,
    ).rename(columns={"label": "gene"})

    _scatter_by_correlation(
        df=region_plot_df,
        corr_col="pearson_r",
        title=f"GTEx vs Allen HVG scatter (color=region Pearson r) [{scale}]",
        out_path=fig_dir / f"scatter_region_corrcolor_combined_{scale}.png",
        highlight_groups=region_highlights[["parcel_idx", "parcel_name", "pearson_r", "extreme"]].copy(),
        group_col="parcel_idx",
        label_col="parcel_name",
    )
    _scatter_by_correlation(
        df=gene_plot_df,
        corr_col="pearson_r",
        title=f"GTEx vs Allen HVG scatter (color=gene Pearson r) [{scale}]",
        out_path=fig_dir / f"scatter_gene_corrcolor_combined_{scale}.png",
        highlight_groups=gene_highlights[["gene", "pearson_r", "extreme"]].copy(),
        group_col="gene",
        label_col="gene",
    )

    return {
        "scale": scale,
        "n_rows_long": int(len(long_df)),
        "n_regions": int(len(region_keys)),
        "n_genes": int(len(gene_cols)),
        "observed_parcels": int(idx["is_observed_gtex"].sum()),
        "unobserved_parcels": int((~idx["is_observed_gtex"].astype(bool)).sum()),
        "shift_amplitude": float(amp),
        "x_shift_applied": bool(amp > 0),
        "top_genes_in_plot_legend": top_genes,
        "top_regions_in_plot_legend": top_region_rows["parcel_name"].tolist(),
        "region_corr_min": float(np.nanmin(region_corr_df["pearson_r"].to_numpy(dtype=np.float64))),
        "region_corr_max": float(np.nanmax(region_corr_df["pearson_r"].to_numpy(dtype=np.float64))),
        "gene_corr_min": float(np.nanmin(gene_corr_df["pearson_r"].to_numpy(dtype=np.float64))),
        "gene_corr_max": float(np.nanmax(gene_corr_df["pearson_r"].to_numpy(dtype=np.float64))),
    }


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate GTEx-vs-Allen O(3) scatter panel suite.")
    p.add_argument("--gtex-raw-path", default=Config.gtex_raw_path)
    p.add_argument("--allen-raw-path", default=Config.allen_raw_path)
    p.add_argument("--gtex-harm-path", default=Config.gtex_harm_path)
    p.add_argument("--allen-harm-path", default=Config.allen_harm_path)
    p.add_argument("--aligned-index-path", default=Config.aligned_index_path)
    p.add_argument("--out-fig-dir", default=Config.out_fig_dir)
    p.add_argument("--out-tab-dir", default=Config.out_tab_dir)
    p.add_argument("--shift-frac", type=float, default=Config.shift_frac)
    p.add_argument("--top-n-gene-legend", type=int, default=Config.top_n_gene_legend)
    p.add_argument("--top-n-region-legend", type=int, default=Config.top_n_region_legend)
    p.add_argument("--highlight-k", type=int, default=Config.highlight_k)
    a = p.parse_args()
    return Config(
        gtex_raw_path=a.gtex_raw_path,
        allen_raw_path=a.allen_raw_path,
        gtex_harm_path=a.gtex_harm_path,
        allen_harm_path=a.allen_harm_path,
        aligned_index_path=a.aligned_index_path,
        out_fig_dir=a.out_fig_dir,
        out_tab_dir=a.out_tab_dir,
        shift_frac=a.shift_frac,
        top_n_gene_legend=a.top_n_gene_legend,
        top_n_region_legend=a.top_n_region_legend,
        highlight_k=a.highlight_k,
    )


def main() -> None:
    cfg = parse_args()
    sns.set_style("whitegrid")
    sns.set_context("talk")
    root = Path(".").resolve()

    fig_dir = (root / cfg.out_fig_dir).resolve()
    tab_dir = (root / cfg.out_tab_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    gtex_raw, allen_raw, idx, genes_raw = _read_and_validate_pair(
        gtex_path=(root / cfg.gtex_raw_path).resolve(),
        allen_path=(root / cfg.allen_raw_path).resolve(),
        idx_path=(root / cfg.aligned_index_path).resolve(),
        scale_name="raw",
    )
    gtex_h, allen_h, idx_h, genes_h = _read_and_validate_pair(
        gtex_path=(root / cfg.gtex_harm_path).resolve(),
        allen_path=(root / cfg.allen_harm_path).resolve(),
        idx_path=(root / cfg.aligned_index_path).resolve(),
        scale_name="harmonized",
    )
    if not np.array_equal(idx["parcel_idx"].to_numpy(), idx_h["parcel_idx"].to_numpy()):
        raise RuntimeError("Aligned index mismatch between raw and harmonized tables.")
    if genes_raw != genes_h:
        raise RuntimeError("Gene ordering mismatch between raw and harmonized tables.")

    qc = {
        "config": asdict(cfg),
        "input_shapes": {
            "raw": [int(gtex_raw.shape[0]), int(gtex_raw.shape[1])],
            "harmonized": [int(gtex_h.shape[0]), int(gtex_h.shape[1])],
        },
        "common": {
            "n_parcels": int(len(idx)),
            "n_genes": int(len(genes_raw)),
            "n_observed_parcels": int(idx["is_observed_gtex"].sum()),
            "n_unobserved_parcels": int((~idx["is_observed_gtex"].astype(bool)).sum()),
        },
    }

    raw_qc = _run_for_scale(
        scale="raw",
        gtex=gtex_raw,
        allen=allen_raw,
        idx=idx,
        gene_cols=genes_raw,
        cfg=cfg,
        fig_dir=fig_dir,
        tab_dir=tab_dir,
    )
    harm_qc = _run_for_scale(
        scale="harmonized",
        gtex=gtex_h,
        allen=allen_h,
        idx=idx,
        gene_cols=genes_h,
        cfg=cfg,
        fig_dir=fig_dir,
        tab_dir=tab_dir,
    )
    qc["scale_raw"] = raw_qc
    qc["scale_harmonized"] = harm_qc

    qc_path = tab_dir / "plot_qc_summary.json"
    with open(qc_path, "w") as f:
        json.dump(qc, f, indent=2)

    print("Scatter panel generation complete.")
    print(f"Figures: {fig_dir}")
    print(f"Tables: {tab_dir}")
    print(f"Rows long/raw: {raw_qc['n_rows_long']}; Rows long/harmonized: {harm_qc['n_rows_long']}")
    print(f"Parcels: {qc['common']['n_parcels']} (observed={qc['common']['n_observed_parcels']}, unobserved={qc['common']['n_unobserved_parcels']})")
    print(f"Genes: {qc['common']['n_genes']}")


if __name__ == "__main__":
    main()
