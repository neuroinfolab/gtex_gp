from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .eval_style import (
    REGION_SCATTER_GROUP_BASE,
    REGION_SCATTER_GROUP_ORDER,
    apply_tick_style,
    format_legend_label,
    font_size,
    model_label,
    ordered_models,
    ordered_region_scatter_values,
    region_scatter_palette,
)


def ahba_parcel_matrix(prepost: Mapping[str, object], genes: Sequence[str], agg: str = "mean") -> np.ndarray:
    """Return AHBA harmonized parcel x gene matrix aggregated over AHBA rows."""
    mode = str(agg).lower()
    if mode not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    ahba = prepost["ahba_h"]
    if not isinstance(ahba, pd.DataFrame):
        raise TypeError("prepost['ahba_h'] must be a DataFrame")
    gene_list = [str(g) for g in genes]
    n_parcels = int(len(prepost["target_meta"]))
    out = np.full((n_parcels, len(gene_list)), np.nan, dtype=np.float64)
    grouped = getattr(ahba.groupby("parcel_idx")[gene_list], mode)()
    for parcel_idx, row in grouped.iterrows():
        out[int(parcel_idx), :] = row[gene_list].to_numpy(dtype=np.float64)
    return out


def _color_values(meta_df: pd.DataFrame, genes: Sequence[str], n_rows: int, color_by: str) -> np.ndarray:
    key = str(color_by).lower()
    if key == "gene":
        return np.tile(np.asarray(list(genes), dtype=object), int(n_rows))
    if key == "region":
        key = "gtex_region"
    if key not in meta_df.columns:
        return np.repeat("all", int(n_rows) * len(genes))
    return np.repeat(meta_df[key].astype(str).to_numpy(), len(genes))


def _sample_frame(df: pd.DataFrame, max_points: int | None, seed: int) -> pd.DataFrame:
    if max_points is None or len(df) <= int(max_points):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_points), random_state=int(seed)).reset_index(drop=True)


def make_matched_distribution_frame(
    view: Mapping[str, object],
    ahba_mat: np.ndarray,
    *,
    color_by: str = "region_group",
    max_points_per_panel: int | None = 120_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build long AHBA-vs-truth/prediction rows for strict held-out LORO parcels."""
    truth = view["truth_df"].reset_index(drop=True)
    pred_dfs = view["pred_dfs"]
    models = list(view["models"])
    genes = [str(g) for g in view["genes"]]
    if not isinstance(truth, pd.DataFrame) or not isinstance(pred_dfs, Mapping):
        raise TypeError("view must come from make_prediction_eval_view")

    gene_idx = np.arange(len(genes), dtype=np.int64)
    parcels = truth["parcel_idx"].to_numpy(dtype=np.int64)
    x = ahba_mat[parcels[:, None], gene_idx[None, :]].ravel()
    meta_cols = ["subject", "parcel_idx", "gtex_region", "region_group", "sex", "age"]
    meta = truth[[c for c in meta_cols if c in truth.columns]].copy()
    colors = _color_values(meta, genes, len(truth), color_by)
    gene_long = np.tile(np.asarray(genes, dtype=object), len(truth))

    rows: list[pd.DataFrame] = []
    y_truth = truth[genes].to_numpy(dtype=np.float64).ravel()
    base = pd.DataFrame(
        {
            "x_ahba": x,
            "y": y_truth,
            "series": "GTEx truth",
            "model": "",
            "marker_kind": "matched truth",
            "color_value": colors,
            "gene": gene_long,
        }
    )
    rows.append(_sample_frame(base[np.isfinite(base["x_ahba"]) & np.isfinite(base["y"])], max_points_per_panel, seed))

    for i, model in enumerate(models):
        pred = pred_dfs[model]
        y_pred = pred[genes].to_numpy(dtype=np.float64).ravel()
        d = pd.DataFrame(
            {
                "x_ahba": x,
                "y": y_pred,
                "series": f"{model_label(model)} prediction",
                "model": model,
                "marker_kind": "matched prediction",
                "color_value": colors,
                "gene": gene_long,
            }
        )
        rows.append(_sample_frame(d[np.isfinite(d["x_ahba"]) & np.isfinite(d["y"])], max_points_per_panel, seed + i + 1))
    return pd.concat(rows, ignore_index=True)


def make_fullfit_imputed_distribution_frame(
    cfg: object,
    prepost: Mapping[str, object],
    ahba_mat: np.ndarray,
    genes: Sequence[str],
    models: Sequence[str],
    *,
    color_by: str = "region_group",
    max_points_per_model: int | None = 120_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build long AHBA-vs-fullfit rows for parcels outside each subject's LORO mask."""
    cache_root = Path(str(getattr(cfg, "cache_root")))
    gene_scope = str(getattr(cfg, "gene_scope")).lower()
    gene_list = [str(g) for g in genes]
    target_meta = prepost["target_meta"].copy()
    meta = target_meta[["parcel_idx", "tissue_or_parcel", "macro_system"]].rename(
        columns={"tissue_or_parcel": "gtex_region", "macro_system": "region_group"}
    )
    meta["parcel_idx"] = meta["parcel_idx"].astype(int)

    frames: list[pd.DataFrame] = []
    for m_i, model in enumerate(ordered_models(models)):
        model_dir = cache_root / gene_scope / model
        rows: list[pd.DataFrame] = []
        for path in sorted(model_dir.glob("*.npz")):
            with np.load(path, allow_pickle=True) as z:
                names = [str(g) for g in z["gene_names"].tolist()]
                keep_genes = [g for g in gene_list if g in names]
                if not keep_genes:
                    continue
                npz_gene_idx = [names.index(g) for g in keep_genes]
                panel_gene_idx = [gene_list.index(g) for g in keep_genes]
                vals = np.asarray(z["fullfit_subject_h"], dtype=np.float64)[:, npz_gene_idx]
                mask = np.asarray(z["imputed_mask"], dtype=bool)
                parcels = np.asarray(z["parcel_idx"], dtype=np.int64)[mask]
                vals = vals[mask, :]
                if len(parcels) == 0:
                    continue
                x = ahba_mat[parcels[:, None], np.asarray(panel_gene_idx, dtype=np.int64)[None, :]].ravel()
                parcel_meta = pd.DataFrame({"parcel_idx": parcels}).merge(meta, on="parcel_idx", how="left")
                colors = _color_values(parcel_meta, keep_genes, len(parcel_meta), color_by)
                rows.append(
                    pd.DataFrame(
                        {
                            "x_ahba": x,
                            "y": vals.ravel(),
                            "series": f"{model_label(model)} prediction",
                            "model": model,
                            "marker_kind": "fullfit imputed",
                            "color_value": colors,
                            "gene": np.tile(np.asarray(keep_genes, dtype=object), len(parcel_meta)),
                        }
                    )
                )
        if rows:
            d = pd.concat(rows, ignore_index=True)
            d = d[np.isfinite(d["x_ahba"]) & np.isfinite(d["y"])]
            frames.append(_sample_frame(d, max_points_per_model, seed + m_i))
    if not frames:
        return pd.DataFrame(columns=["x_ahba", "y", "series", "model", "marker_kind", "color_value", "gene"])
    return pd.concat(frames, ignore_index=True)


def _palette_for(df: pd.DataFrame, color_by: str) -> tuple[list[str], dict[str, str]]:
    vals = [str(v) for v in pd.unique(df["color_value"].dropna())]
    key = str(color_by).lower()
    if key == "region_group":
        order = [v for v in REGION_SCATTER_GROUP_ORDER if v in vals] + sorted(v for v in vals if v not in REGION_SCATTER_GROUP_ORDER)
        return order, {v: to_hex(REGION_SCATTER_GROUP_BASE.get(v, "#777777")) for v in order}
    if key in {"region", "gtex_region"}:
        lookup = {}
        if {"color_value", "region_group"}.issubset(df.columns):
            lookup = df.groupby("color_value")["region_group"].agg(lambda s: s.value_counts().index[0]).to_dict()
        order = ordered_region_scatter_values(vals, region_group_lookup=lookup if lookup else None)
        pal = region_scatter_palette(order, region_group_lookup=lookup if lookup else None)
        return order, {v: to_hex(pal[v]) for v in order}
    if key == "gene":
        order = [v for v in vals if v != "Other"]
        if "Other" in vals:
            order.append("Other")
    else:
        order = sorted(vals)
    cmap = plt.get_cmap("tab20")
    palette = {v: to_hex(cmap(i % 20)) for i, v in enumerate(order)}
    if "Other" in palette:
        palette["Other"] = "#9a9a9a"
    return order, palette


def _limit_color_categories(df: pd.DataFrame, color_by: str, top_n: int) -> pd.DataFrame:
    if str(color_by).lower() != "gene" or int(top_n) <= 0:
        return df
    ordered = [v for v in pd.unique(df["color_value"].astype(str)) if v != "Other"]
    keep = set(ordered[: int(top_n)])
    out = df.copy()
    out["color_value"] = np.where(out["color_value"].astype(str).isin(keep), out["color_value"].astype(str), "Other")
    return out


def plot_ahba_distribution_panels(
    df: pd.DataFrame,
    models: Sequence[str],
    *,
    color_by: str = "region_group",
    title: str,
    include_imputed: bool = False,
    top_n: int = 10,
    matched_point_size: float = 2.0,
    imputed_point_size: float = 5.0,
    matched_alpha: float = 0.16,
    imputed_alpha: float = 0.10,
    imputed_color: str = "#5f5f5f",
    max_legend_items: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot AHBA-vs-GTEx truth and AHBA-vs-prediction panels.

    Matched LORO rows render as circles. Optional fullfit imputed rows render as
    triangles in prediction panels.
    """
    df = _limit_color_categories(df, color_by=color_by, top_n=int(top_n))
    use_models = ordered_models(models)
    nrows = len(use_models)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.5, max(3.2, 2.85 * nrows)), squeeze=False, sharex=True, sharey=True)
    order, palette = _palette_for(df, color_by)
    finite = df[["x_ahba", "y"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) == 0:
        raise RuntimeError("No finite points to plot")
    lo = float(np.nanpercentile(finite.to_numpy().ravel(), 0.5))
    hi = float(np.nanpercentile(finite.to_numpy().ravel(), 99.5))
    pad = 0.04 * max(hi - lo, 1e-6)
    lim = (lo - pad, hi + pad)
    truth_df = df[df["series"] == "GTEx truth"]

    for r, model in enumerate(use_models):
        ax_t, ax_p = axes[r, 0], axes[r, 1]
        for val in order:
            color = palette[str(val)]
            t = truth_df[truth_df["color_value"].astype(str) == str(val)]
            if len(t):
                ax_t.scatter(t["x_ahba"], t["y"], s=matched_point_size, alpha=matched_alpha, color=color, linewidths=0, rasterized=True)
            p = df[(df["model"] == model) & (df["color_value"].astype(str) == str(val)) & (df["marker_kind"] == "matched prediction")]
            if len(p):
                ax_p.scatter(p["x_ahba"], p["y"], s=matched_point_size, alpha=matched_alpha, color=color, linewidths=0, rasterized=True)
        if include_imputed:
            imp = df[(df["model"] == model) & (df["marker_kind"] == "fullfit imputed")]
            if len(imp):
                ax_p.scatter(
                    imp["x_ahba"],
                    imp["y"],
                    s=imputed_point_size,
                    alpha=imputed_alpha,
                    marker="^",
                    color=imputed_color,
                    linewidths=0,
                    rasterized=True,
                )
        for ax in (ax_t, ax_p):
            ax.plot(lim, lim, color="#333333", linewidth=0.8, alpha=0.55)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.18)
            apply_tick_style(ax, label_fontsize=font_size("s"))
        ax_t.set_title(f"{model_label(model)}: AHBA vs held-out GTEx", fontsize=font_size("m"))
        ax_p.set_title(f"{model_label(model)}: AHBA vs prediction", fontsize=font_size("m"))
        ax_t.set_ylabel("GTEx harmonized value", fontsize=font_size("s"))

    axes[-1, 0].set_xlabel("AHBA harmonized value", fontsize=font_size("s"))
    axes[-1, 1].set_xlabel("AHBA harmonized value", fontsize=font_size("s"))
    legend_limit = int(max_legend_items) if max_legend_items is not None else int(top_n)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[v], markersize=5, label=format_legend_label(v))
        for v in order[:legend_limit]
    ]
    if len(order) > legend_limit:
        handles.append(Line2D([0], [0], marker="o", color="#9a9a9a", linestyle="none", markersize=5, label=f"Other ({len(order) - legend_limit} more)"))
    if include_imputed:
        handles.extend(
            [
                Line2D([0], [0], marker="o", color="#555555", linestyle="none", markersize=4, label="matched LORO"),
                Line2D([0], [0], marker="^", color=imputed_color, linestyle="none", markersize=5, label="fullfit imputed"),
            ]
        )
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title=format_legend_label(color_by), fontsize=font_size("s"), title_fontsize=font_size("s"))
    fig.suptitle(title, y=1.01, fontsize=font_size("l"))
    fig.tight_layout()
    return fig, axes
