#!/usr/bin/env python3
from __future__ import annotations

"""Flattened matrix diagnostics built from eval tensor views.

The matrix contract mirrors the Variantformer-style diagnostic view:
rows are observed subject-region samples and columns are genes. Paired truth
and reconstruction matrices use the same row ordering so both row-wise sample
correlations and column-wise gene correlations are well defined.
"""

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd

from .eval_samples import (
    GTEX_TENSOR_REGION_ORDER,
    TensorView,
    _format_axis_labels,
    _resolve_tick_step,
)
from .eval_style import apply_tick_style, font_size, pretty_gtex_label


GTEX_LORO_FLATTENED_GROUP_ORDER = [
    "brain - cortex",
    "brain - frontal cortex (ba9)",
    "brain - anterior cingulate cortex (ba24)",
    "brain - caudate (basal ganglia)",
    "brain - putamen (basal ganglia)",
    "brain - nucleus accumbens (basal ganglia)",
    "brain - hypothalamus",
    "brain - amygdala",
    "brain - hippocampus",
    "brain - cerebellum",
    "brain - cerebellar hemisphere",
    "brain - substantia nigra",
    "brain - pituitary",
    "brain - spinal cord (cervical c-1)",
]


@dataclass
class FlattenedTensorMatrix:
    X: np.ndarray
    row_metadata: pd.DataFrame
    genes: list[str]
    title: str
    dataset: str
    stage: str
    groupby: str
    group_slices: dict[str, slice]
    x_label: str = "Genes"
    y_label: str = "Observed subject-region samples"
    value_label: str = "Expression"


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()


def _native_gtex_region_lookup(view: TensorView) -> dict[str, str]:
    regions = [str(r) for r in view.regions]
    lookup = {r: r for r in regions}
    matching = view.matching or {}
    gtex_to_ahba = matching.get("gtex_to_ahba") or {}
    ahba_to_gtex = {str(v): str(k) for k, v in gtex_to_ahba.items()}
    for region in regions:
        if region in ahba_to_gtex:
            lookup[region] = ahba_to_gtex[region]
    return lookup


def _row_metadata(
    view: TensorView,
    row_subject_index: np.ndarray,
    row_region_index: np.ndarray,
    *,
    groupby: str,
    subject_label_mode: str,
) -> pd.DataFrame:
    groupby = str(groupby).strip().lower()
    regions = [str(r) for r in view.regions]
    native_lookup = _native_gtex_region_lookup(view)
    native_regions = [native_lookup[r] for r in regions]
    subject_labels = _format_axis_labels(
        view.subjects,
        semantic_name="subject",
        mode=subject_label_mode,
        dataset=view.dataset,
    )

    rows = []
    for row, (sidx, ridx) in enumerate(zip(row_subject_index, row_region_index)):
        region = regions[int(ridx)]
        native_region = native_regions[int(ridx)]
        if groupby in ("tissue", "gtex_region", "gtex_native_region", "native_region"):
            group = native_region
        elif groupby in ("region", "parcel", "target_parcel"):
            group = region
        elif groupby == "subject":
            group = str(view.subjects[int(sidx)])
        else:
            raise ValueError(
                "groupby must be one of 'tissue', 'gtex_native_region', "
                "'region', 'parcel', or 'subject'."
            )
        rows.append(
            {
                "row": row,
                "dataset": str(view.dataset),
                "subject": str(view.subjects[int(sidx)]),
                "subject_label": subject_labels[int(sidx)],
                "subject_index": int(sidx),
                "region": region,
                "gtex_native_region": native_region,
                "region_label": pretty_gtex_label(native_region),
                "region_index": int(ridx),
                "group": str(group),
                "group_label": pretty_gtex_label(group),
            }
        )
    return pd.DataFrame(rows)


def _ordered_groups(groups: Sequence[str], group_order: Sequence[str] | None) -> list[str]:
    present = list(dict.fromkeys(str(g) for g in groups))
    if group_order is None:
        group_order = GTEX_LORO_FLATTENED_GROUP_ORDER + GTEX_TENSOR_REGION_ORDER
    order_norm = {_normalize_label(g): i for i, g in enumerate(group_order)}
    return sorted(
        present,
        key=lambda g: (order_norm.get(_normalize_label(g), 10_000), _normalize_label(g)),
    )


def _paired_row_indices(
    truth_view: TensorView,
    recon_view: TensorView,
    *,
    require_all_genes_finite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if list(truth_view.subjects) != list(recon_view.subjects):
        raise ValueError("truth_view and recon_view must have identical subject order.")
    if list(truth_view.regions) != list(recon_view.regions):
        raise ValueError("truth_view and recon_view must have identical region order.")
    if list(truth_view.genes) != list(recon_view.genes):
        raise ValueError("truth_view and recon_view must have identical gene order.")
    truth_values = np.asarray(truth_view.values)
    recon_values = np.asarray(recon_view.values)
    if truth_values.shape != recon_values.shape:
        raise ValueError(f"shape mismatch: {truth_values.shape} vs {recon_values.shape}")
    row_mask = np.asarray(truth_view.observed_mask, dtype=bool) & np.asarray(
        recon_view.observed_mask,
        dtype=bool,
    )
    if require_all_genes_finite:
        row_mask &= np.isfinite(truth_values).all(axis=2)
        row_mask &= np.isfinite(recon_values).all(axis=2)
    else:
        row_mask &= np.isfinite(truth_values).any(axis=2)
        row_mask &= np.isfinite(recon_values).any(axis=2)
    return np.where(row_mask)


def build_paired_flattened_tensor_matrices(
    truth_view: TensorView,
    recon_view: TensorView,
    *,
    groupby: str = "tissue",
    group_order: Sequence[str] | None = None,
    subject_label_mode: str = "full",
    require_all_genes_finite: bool = True,
    truth_title: str = "True gene expression",
    recon_title: str = "Predicted gene expression",
    value_label: str = "Expression",
) -> tuple[FlattenedTensorMatrix, FlattenedTensorMatrix]:
    """Build aligned flattened matrices from paired truth/reconstruction views."""
    subj_i, region_i = _paired_row_indices(
        truth_view,
        recon_view,
        require_all_genes_finite=require_all_genes_finite,
    )
    meta = _row_metadata(
        truth_view,
        subj_i,
        region_i,
        groupby=groupby,
        subject_label_mode=subject_label_mode,
    )
    groups = _ordered_groups(meta["group"].tolist(), group_order)
    order = []
    group_slices: dict[str, slice] = {}
    cursor = 0
    for group in groups:
        idx = meta.index[meta["group"] == group].tolist()
        if not idx:
            continue
        order.extend(idx)
        group_slices[str(group)] = slice(cursor, cursor + len(idx))
        cursor += len(idx)
    order_arr = np.asarray(order, dtype=int)
    subj_i = subj_i[order_arr]
    region_i = region_i[order_arr]
    meta = meta.iloc[order_arr].reset_index(drop=True)
    meta["row"] = np.arange(len(meta), dtype=int)

    truth_X = np.asarray(truth_view.values, dtype=np.float64)[subj_i, region_i, :]
    recon_X = np.asarray(recon_view.values, dtype=np.float64)[subj_i, region_i, :]
    common = dict(
        row_metadata=meta,
        genes=[str(g) for g in truth_view.genes],
        dataset=str(truth_view.dataset),
        groupby=str(groupby),
        group_slices=group_slices,
        value_label=str(value_label),
    )
    truth = FlattenedTensorMatrix(
        X=truth_X,
        title=str(truth_title),
        stage=str(truth_view.pipeline_stage or "truth"),
        **common,
    )
    recon = FlattenedTensorMatrix(
        X=recon_X,
        title=str(recon_title),
        stage=str(recon_view.pipeline_stage or "recon"),
        **common,
    )
    return truth, recon


def _single_row_indices(
    view: TensorView,
    *,
    require_all_genes_finite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(view.values)
    row_mask = np.asarray(view.observed_mask, dtype=bool)
    if require_all_genes_finite:
        row_mask &= np.isfinite(values).all(axis=2)
    else:
        row_mask &= np.isfinite(values).any(axis=2)
    return np.where(row_mask)


def build_flattened_tensor_matrix(
    view: TensorView,
    *,
    groupby: str = "tissue",
    group_order: Sequence[str] | None = None,
    subject_label_mode: str = "full",
    require_all_genes_finite: bool = True,
    title: str = "Gene expression",
    value_label: str = "Expression",
) -> FlattenedTensorMatrix:
    """Flatten a single view to one (subject-region samples x genes) matrix.

    Standalone counterpart to `build_paired_flattened_tensor_matrices` — use it
    for views whose observed rows/subjects need not align with a paired view
    (e.g. raw pre-ComBat GTEx, whose subject set differs from the LORO caches).
    """
    subj_i, region_i = _single_row_indices(
        view, require_all_genes_finite=require_all_genes_finite
    )
    meta = _row_metadata(
        view, subj_i, region_i, groupby=groupby, subject_label_mode=subject_label_mode
    )
    groups = _ordered_groups(meta["group"].tolist(), group_order)
    order: list[int] = []
    group_slices: dict[str, slice] = {}
    cursor = 0
    for group in groups:
        idx = meta.index[meta["group"] == group].tolist()
        if not idx:
            continue
        order.extend(idx)
        group_slices[str(group)] = slice(cursor, cursor + len(idx))
        cursor += len(idx)
    order_arr = np.asarray(order, dtype=int)
    subj_i = subj_i[order_arr]
    region_i = region_i[order_arr]
    meta = meta.iloc[order_arr].reset_index(drop=True)
    meta["row"] = np.arange(len(meta), dtype=int)

    X = np.asarray(view.values, dtype=np.float64)[subj_i, region_i, :]
    return FlattenedTensorMatrix(
        X=X,
        row_metadata=meta,
        genes=[str(g) for g in view.genes],
        title=str(title),
        dataset=str(view.dataset),
        stage=str(view.pipeline_stage or "raw"),
        groupby=str(groupby),
        group_slices=group_slices,
        value_label=str(value_label),
    )


def _soften_cmap(cmap, amount: float = 0.35):
    """Lower-contrast version of `cmap`: blend its colors toward neutral grey by
    `amount` (0 = unchanged, 1 = flat grey). Easier on the eye for dense heatmaps."""
    amount = float(np.clip(amount, 0.0, 1.0))
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cols = base(np.linspace(0.0, 1.0, 256))
    grey = np.array([0.5, 0.5, 0.5, 1.0])
    cols = (1.0 - amount) * cols + amount * grey
    return ListedColormap(cols, name=f"{getattr(base, 'name', 'cmap')}_soft")


def build_donor_median_flattened_matrix(
    view: TensorView,
    *,
    groupby: str = "tissue",
    title: str = "Donor median",
    value_label: str = "Median expression",
) -> FlattenedTensorMatrix:
    """Collapse a view's subject axis to one (region x gene) median matrix.

    Rows are regions in the view's own axis order (e.g. region_matched_superset),
    one row per region; values are the across-donor median per region/gene
    (observed cells only). No group reordering — the view region order is kept.
    """
    vals = np.asarray(view.values, dtype=np.float64)            # (S, R, G)
    obs = np.asarray(view.observed_mask, dtype=bool)            # (S, R)
    masked = np.where(obs[:, :, None], vals, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(masked, axis=0)                     # (R, G)

    regions = [str(r) for r in view.regions]
    native_lookup = _native_gtex_region_lookup(view)
    gb = str(groupby).strip().lower()
    rows = []
    for r_i, region in enumerate(regions):
        native = native_lookup[region]
        if gb in ("tissue", "gtex_region", "gtex_native_region", "native_region"):
            group = native
        elif gb in ("region", "parcel", "target_parcel"):
            group = region
        else:
            raise ValueError("groupby must be 'tissue', 'gtex_native_region', 'region', or 'parcel'.")
        rows.append(
            {
                "row": r_i,
                "dataset": str(view.dataset),
                "subject": "donor_median",
                "subject_label": "donor median",
                "subject_index": 0,
                "region": region,
                "gtex_native_region": native,
                "region_label": pretty_gtex_label(native),
                "region_index": r_i,
                "group": str(group),
                "group_label": pretty_gtex_label(group),
            }
        )
    meta = pd.DataFrame(rows)

    # Contiguous group slices preserving region order (no reordering).
    group_slices: dict[str, slice] = {}
    seq = meta["group"].tolist()
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            group_slices[str(seq[start])] = slice(start, i)
            start = i

    return FlattenedTensorMatrix(
        X=med,
        row_metadata=meta,
        genes=[str(g) for g in view.genes],
        title=str(title),
        dataset=str(view.dataset),
        stage=str(view.pipeline_stage or "median"),
        groupby=str(groupby),
        group_slices=group_slices,
        value_label=str(value_label),
    )


def _robust_vrange(mats: Sequence[np.ndarray], q: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    values = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in mats])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(values, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _apply_matrix_ticks(
    ax: plt.Axes,
    matrix: FlattenedTensorMatrix,
    *,
    gene_tick_step: int | str | None,
    sample_tick_step: int | str | None,
    gene_label_mode: str,
    show_gene_ticklabels: bool,
    show_sample_ticklabels: bool,
) -> None:
    gene_step = _resolve_tick_step(gene_tick_step)
    row_step = _resolve_tick_step(sample_tick_step)
    gene_idx = np.arange(0, len(matrix.genes), gene_step)
    row_idx = np.arange(0, len(matrix.row_metadata), row_step)
    gene_labels = _format_axis_labels(
        matrix.genes,
        semantic_name="gene",
        mode=gene_label_mode,
        dataset=matrix.dataset,
    )
    ax.set_xticks(gene_idx)
    ax.set_xticklabels(
        [gene_labels[i] for i in gene_idx] if show_gene_ticklabels else [],
        rotation=90,
    )
    ax.set_yticks(row_idx)
    ax.set_yticklabels(
        matrix.row_metadata.loc[row_idx, "subject_label"].tolist()
        if show_sample_ticklabels
        else []
    )


def _draw_group_blocks(
    ax: plt.Axes,
    matrix: FlattenedTensorMatrix,
    *,
    show_group_labels: bool,
    show_group_separators: bool,
    show_region_separators: bool,
    region_separator_linewidth: float,
    region_separator_color: str,
    region_separator_alpha: float,
    group_separator_linewidth: float | None,
    group_separator_color: str,
    group_label_x: float,
    group_label_fontsize: int | float,
) -> None:
    if show_region_separators and "region" in matrix.row_metadata.columns:
        regions = matrix.row_metadata["region"].tolist()
        for row in range(1, len(regions)):
            if regions[row] != regions[row - 1]:
                ax.axhline(
                    row - 0.5,
                    color=region_separator_color,
                    linewidth=region_separator_linewidth,
                    alpha=region_separator_alpha,
                )
    for group, slc in matrix.group_slices.items():
        if show_group_separators and slc.start > 0:
            if group_separator_linewidth is None:
                # default subtle double line (white halo + thin grey)
                ax.axhline(slc.start - 0.5, color="white", linewidth=1.2)
                ax.axhline(slc.start - 0.5, color="#404040", linewidth=0.35, alpha=0.55)
            else:
                ax.axhline(slc.start - 0.5, color=group_separator_color,
                           linewidth=group_separator_linewidth)
        if show_group_labels:
            center = (slc.start + slc.stop - 1) / 2.0
            label = pretty_gtex_label(group)
            ax.text(
                group_label_x,
                center,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=group_label_fontsize,
                clip_on=False,
            )


def plot_flattened_tensor_matrix(
    matrix: FlattenedTensorMatrix,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    gene_tick_step: int | str | None = 10,
    sample_tick_step: int | str | None = 4,
    gene_label_mode: str = "full",
    show_gene_ticklabels: bool = False,
    show_sample_ticklabels: bool = True,
    show_group_labels: bool = True,
    show_group_separators: bool = True,
    show_region_separators: bool = False,
    region_separator_linewidth: float = 0.3,
    region_separator_color: str = "#808080",
    region_separator_alpha: float = 0.45,
    group_separator_linewidth: float | None = None,
    group_separator_color: str = "black",
    show_ylabel: bool = False,
    group_label_x: float = -0.08,
    group_label_fontsize: int | float = 8,
    font_scale: float = 1.0,
    cbar_font_scale: float = 1.0,
    soften: bool = False,
    soften_amount: float = 0.35,
    border_linewidth: float | None = None,
    border_color: str = "black",
    colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one flattened subject-region x gene matrix.

    Self-contained building block: accepts an external `ax`, every styling knob,
    and its own `soften` toggle, so it can be composed into custom figure layouts
    without going through `plot_flattened_tensor_matrices`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 7.0), constrained_layout=True)
    else:
        fig = ax.figure
    if soften:
        cmap = _soften_cmap(cmap, soften_amount)
    title_fs = font_size("l") * font_scale
    label_fs = font_size("m") * font_scale
    tick_fs = font_size("s") * font_scale
    cbar_label_fs = label_fs * cbar_font_scale
    cbar_tick_fs = tick_fs * cbar_font_scale
    if vmin is None or vmax is None:
        rvmin, rvmax = _robust_vrange([matrix.X])
        vmin = rvmin if vmin is None else vmin
        vmax = rvmax if vmax is None else vmax
    im = ax.imshow(
        matrix.X,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=Normalize(vmin=float(vmin), vmax=float(vmax)),
    )
    ax.set_title(matrix.title if title is None else title, fontsize=title_fs)
    ax.set_xlabel(matrix.x_label, fontsize=label_fs)
    ax.set_ylabel(matrix.y_label if show_ylabel else "", fontsize=label_fs)
    _apply_matrix_ticks(
        ax,
        matrix,
        gene_tick_step=gene_tick_step,
        sample_tick_step=sample_tick_step,
        gene_label_mode=gene_label_mode,
        show_gene_ticklabels=show_gene_ticklabels,
        show_sample_ticklabels=show_sample_ticklabels,
    )
    _draw_group_blocks(
        ax,
        matrix,
        show_group_labels=show_group_labels,
        show_group_separators=show_group_separators,
        show_region_separators=show_region_separators,
        region_separator_linewidth=region_separator_linewidth,
        region_separator_color=region_separator_color,
        region_separator_alpha=region_separator_alpha,
        group_separator_linewidth=group_separator_linewidth,
        group_separator_color=group_separator_color,
        group_label_x=group_label_x,
        group_label_fontsize=group_label_fontsize * font_scale,
    )
    apply_tick_style(ax, label_fontsize=tick_fs)
    ax.grid(False)  # imshow heatmap — suppress any style-inherited gridlines
    if border_linewidth is not None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(border_linewidth)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(matrix.value_label, fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
    return fig, ax


def plot_flattened_tensor_pair(
    truth: FlattenedTensorMatrix,
    recon: FlattenedTensorMatrix,
    *,
    figsize: tuple[float, float] = (15.0, 8.0),
    cmap: str = "viridis",
    shared_color_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    suptitle: str | None = None,
    gene_tick_step: int | str | None = 10,
    sample_tick_step: int | str | None = 4,
    gene_label_mode: str = "full",
    show_gene_ticklabels: bool = False,
    show_sample_ticklabels: bool = True,
    show_group_labels: bool = True,
    show_group_separators: bool = True,
    show_region_separators: bool = False,
    region_separator_linewidth: float = 0.3,
    region_separator_color: str = "#808080",
    region_separator_alpha: float = 0.45,
    group_separator_linewidth: float | None = None,
    group_separator_color: str = "black",
    soften: bool = False,
    soften_amount: float = 0.35,
    show_ylabel: bool = False,
    group_label_x: float = -0.08,
    group_label_fontsize: int | float = 8,
    font_scale: float = 1.0,
    cbar_font_scale: float = 1.0,
    border_linewidth: float | None = None,
    border_color: str = "black",
    dpi: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot aligned truth/reconstruction flattened matrices side by side."""
    if len(truth.row_metadata) != len(recon.row_metadata) or truth.genes != recon.genes:
        raise ValueError("truth and recon matrices must be aligned before plotting.")
    if shared_color_scale:
        rvmin, rvmax = _robust_vrange([truth.X, recon.X])
        vmin = rvmin if vmin is None else vmin
        vmax = rvmax if vmax is None else vmax
    cmap = _soften_cmap(cmap, soften_amount) if soften else cmap
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharey=True)
    if dpi is not None:
        fig.set_dpi(float(dpi))
    for col, (ax, matrix) in enumerate(zip(axes, (truth, recon))):
        plot_flattened_tensor_matrix(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            gene_tick_step=gene_tick_step,
            sample_tick_step=sample_tick_step,
            gene_label_mode=gene_label_mode,
            show_gene_ticklabels=show_gene_ticklabels,
            show_sample_ticklabels=show_sample_ticklabels,
            show_group_labels=show_group_labels,
            show_group_separators=show_group_separators,
            show_region_separators=show_region_separators,
            region_separator_linewidth=region_separator_linewidth,
            region_separator_color=region_separator_color,
            region_separator_alpha=region_separator_alpha,
            group_separator_linewidth=group_separator_linewidth,
            group_separator_color=group_separator_color,
            show_ylabel=show_ylabel and col == 0,
            group_label_x=group_label_x,
            group_label_fontsize=group_label_fontsize,
            font_scale=font_scale,
            cbar_font_scale=cbar_font_scale,
            border_linewidth=border_linewidth,
            border_color=border_color,
            colorbar=False,
        )
    if vmin is None or vmax is None:
        vmin, vmax = _robust_vrange([truth.X, recon.X])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=float(vmin), vmax=float(vmax)), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(truth.value_label, fontsize=font_size("m") * font_scale * cbar_font_scale)
    cbar.ax.tick_params(labelsize=font_size("s") * font_scale * cbar_font_scale)
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=font_size("l") * font_scale)
    return fig, axes


def plot_flattened_tensor_matrices(
    matrices: Sequence[FlattenedTensorMatrix],
    *,
    standalone_matrices: Sequence[FlattenedTensorMatrix] = (),
    standalone_first: bool = False,
    ncols: int = 4,
    figsize: tuple[float, float] = (18.0, 7.5),
    cmap: str = "viridis",
    shared_color_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    suptitle: str | None = None,
    gene_tick_step: int | str | None = 10,
    sample_tick_step: int | str | None = 4,
    gene_label_mode: str = "full",
    show_gene_ticklabels: bool = False,
    show_sample_ticklabels: bool = True,
    show_group_labels: bool = True,
    show_group_separators: bool = True,
    show_region_separators: bool = False,
    region_separator_linewidth: float = 0.3,
    region_separator_color: str = "#808080",
    region_separator_alpha: float = 0.45,
    group_separator_linewidth: float | None = None,
    group_separator_color: str = "black",
    soften: bool = False,
    soften_amount: float = 0.35,
    show_ylabel: bool = False,
    group_label_x: float = -0.08,
    group_label_fontsize: int | float = 8,
    font_scale: float = 1.0,
    cbar_font_scale: float = 1.0,
    border_linewidth: float | None = None,
    border_color: str = "black",
    dpi: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot aligned flattened matrices sharing one color scale, plus optional
    `standalone_matrices` appended as extra panels — each on its own scale with
    its own colorbar (use for views whose rows/scale don't align, e.g. raw
    pre-ComBat GTEx alongside harmonized truth/recon).
    """
    matrices = list(matrices)
    standalone = list(standalone_matrices)
    if not matrices:
        raise ValueError("matrices must contain at least one FlattenedTensorMatrix.")
    ref = matrices[0]
    for matrix in matrices[1:]:
        if matrix.X.shape != ref.X.shape or matrix.genes != ref.genes:
            raise ValueError("all aligned matrices must share shape and gene axis.")
    if shared_color_scale:
        rvmin, rvmax = _robust_vrange([m.X for m in matrices])
        vmin = rvmin if vmin is None else vmin
        vmax = rvmax if vmax is None else vmax

    cmap = _soften_cmap(cmap, soften_amount) if soften else cmap

    # (matrix, is_standalone) in render order; standalone panels can lead or trail.
    aligned_panels = [(m, False) for m in matrices]
    standalone_panels = [(m, True) for m in standalone]
    ordered = standalone_panels + aligned_panels if standalone_first else aligned_panels + standalone_panels
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(ordered) / ncols))
    # Standalone panels carry their own row count / scale, so don't share axes.
    share = not standalone
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=share,
        sharey=share,
    )
    if dpi is not None:
        fig.set_dpi(float(dpi))
    axes_arr = np.atleast_1d(axes).ravel()
    common = dict(
        cmap=cmap,
        gene_tick_step=gene_tick_step,
        sample_tick_step=sample_tick_step,
        gene_label_mode=gene_label_mode,
        show_gene_ticklabels=show_gene_ticklabels,
        show_sample_ticklabels=show_sample_ticklabels,
        show_group_labels=show_group_labels,
        show_group_separators=show_group_separators,
        show_region_separators=show_region_separators,
        region_separator_linewidth=region_separator_linewidth,
        region_separator_color=region_separator_color,
        region_separator_alpha=region_separator_alpha,
        group_separator_linewidth=group_separator_linewidth,
        group_separator_color=group_separator_color,
        group_label_x=group_label_x,
        group_label_fontsize=group_label_fontsize,
        font_scale=font_scale,
        cbar_font_scale=cbar_font_scale,
        border_linewidth=border_linewidth,
        border_color=border_color,
    )
    aligned_axes = []
    for col, (ax, (matrix, is_standalone)) in enumerate(zip(axes_arr, ordered)):
        if not is_standalone:
            aligned_axes.append(ax)
        plot_flattened_tensor_matrix(
            matrix,
            ax=ax,
            vmin=None if is_standalone else vmin,
            vmax=None if is_standalone else vmax,
            show_ylabel=show_ylabel and (col % ncols == 0),
            colorbar=is_standalone,  # standalone gets its own cbar; aligned share one
            **common,
        )
    for ax in axes_arr[len(ordered):]:
        ax.set_visible(False)
    if vmin is None or vmax is None:
        vmin, vmax = _robust_vrange([m.X for m in matrices])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=float(vmin), vmax=float(vmax)), cmap=cmap)
    cbar = fig.colorbar(sm, ax=aligned_axes, fraction=0.025, pad=0.02)
    cbar.set_label(ref.value_label, fontsize=font_size("m") * font_scale * cbar_font_scale)
    cbar.ax.tick_params(labelsize=font_size("s") * font_scale * cbar_font_scale)
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=font_size("l") * font_scale)
    return fig, axes_arr[: len(ordered)]


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    aa = np.asarray(a[mask], dtype=np.float64)
    bb = np.asarray(b[mask], dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(aa * bb) / denom)


def gene_correlation_table(
    truth: FlattenedTensorMatrix,
    recon: FlattenedTensorMatrix,
) -> pd.DataFrame:
    """Column-wise truth/reconstruction correlations across samples."""
    if truth.X.shape != recon.X.shape or truth.genes != recon.genes:
        raise ValueError("truth and recon matrices must have identical shapes and gene order.")
    rows = []
    for idx, gene in enumerate(truth.genes):
        rows.append(
            {
                "gene": gene,
                "gene_index": idx,
                "pearson_r": _pearson_1d(truth.X[:, idx], recon.X[:, idx]),
            }
        )
    return pd.DataFrame(rows)


def sample_correlation_table(
    truth: FlattenedTensorMatrix,
    recon: FlattenedTensorMatrix,
) -> pd.DataFrame:
    """Row-wise truth/reconstruction correlations across genes."""
    if truth.X.shape != recon.X.shape:
        raise ValueError("truth and recon matrices must have identical shapes.")
    meta = truth.row_metadata.copy()
    meta["pearson_r"] = [
        _pearson_1d(truth.X[i, :], recon.X[i, :]) for i in range(truth.X.shape[0])
    ]
    return meta


def reconstruction_summary(
    truth: FlattenedTensorMatrix,
    recon: FlattenedTensorMatrix,
) -> dict:
    """Mean truth/recon correlation in both directions, for figure annotation.

    - `across_gene_mean_r`   : mean per-sample r (each row correlated across genes;
      the schematic's horizontal "across gene recon" r̄).
    - `across_sample_mean_r` : mean per-gene r (each column correlated across
      samples; the "across sample recon" r̄).

    Also returns the underlying tables so callers can draw distributions/violins.
    """
    gene_tbl = gene_correlation_table(truth, recon)
    sample_tbl = sample_correlation_table(truth, recon)
    return {
        "across_gene_mean_r": float(np.nanmean(sample_tbl["pearson_r"].to_numpy())),
        "across_sample_mean_r": float(np.nanmean(gene_tbl["pearson_r"].to_numpy())),
        "gene_table": gene_tbl,
        "sample_table": sample_tbl,
    }
