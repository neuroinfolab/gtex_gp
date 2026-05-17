#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.samples import validate_gxp_samples
from src.samples.schema import META_COLS

from .eval_style import PARCEL_GROUP_COLORS, apply_tick_style, pretty_gtex_label


GTEX_TENSOR_REGION_ORDER = [
    "brain - frontal cortex (ba9)",
    "brain - anterior cingulate cortex (ba24)",
    "brain - cortex",
    "brain - nucleus accumbens (basal ganglia)",
    "brain - caudate (basal ganglia)",
    "brain - putamen (basal ganglia)",
    "brain - amygdala",
    "brain - hippocampus",
    "brain - hypothalamus",
    "brain - substantia nigra",
    "brain - cerebellar hemisphere",
    "brain - cerebellum",
]

GTEX_TENSOR_REGION_GROUP = {
    "brain - frontal cortex (ba9)": "cortical",
    "brain - anterior cingulate cortex (ba24)": "cortical",
    "brain - cortex": "cortical",
    "brain - nucleus accumbens (basal ganglia)": "basal_ganglia",
    "brain - caudate (basal ganglia)": "basal_ganglia",
    "brain - putamen (basal ganglia)": "basal_ganglia",
    "brain - amygdala": "limbic_midbrain",
    "brain - hippocampus": "limbic_midbrain",
    "brain - hypothalamus": "limbic_midbrain",
    "brain - substantia nigra": "limbic_midbrain",
    "brain - cerebellar hemisphere": "cerebellar",
    "brain - cerebellum": "cerebellar",
}

GTEX_TENSOR_GROUP_COLOR = {group: colors[0] for group, colors in PARCEL_GROUP_COLORS.items()}
TENSOR_AXIS_COMPONENTS = ("gene", "subject", "region")
DEFAULT_TENSOR_AXIS_ASSIGNMENT = ("gene", "region", "subject")


@dataclass
class TensorView:
    values: np.ndarray
    observed_mask: np.ndarray
    subjects: list[str]
    regions: list[str]
    genes: list[str]
    dataset: str
    region_axis_kind: str


def load_gxp_samples_table(csv_path: str | Path = "data/raw/gxp_samples.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_gxp_samples(df)
    return df


def load_gene_panel(path_or_name: str | Path) -> list[str]:
    raw = str(path_or_name).strip()
    path = Path(raw)
    if not path.exists():
        root = Path(__file__).resolve().parents[2]
        candidates = [
            (root / raw).resolve(),
            (root / "data" / "metadata" / "gene_lists" / raw).resolve(),
            (root / "data" / "metadata" / f"{raw}.txt").resolve(),
            (root / "data" / "metadata" / "gene_lists" / f"{raw}.txt").resolve(),
            (root / "data" / "raw" / "gene_lists" / raw).resolve(),
            (root / "data" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
            (root / "out" / "raw" / "gene_lists" / raw).resolve(),
            (root / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise FileNotFoundError(f"Could not resolve gene panel {path_or_name!r}")
    if path.suffix.lower() == ".csv":
        panel_df = pd.read_csv(path)
        if "label" in panel_df.columns and len(panel_df.columns) > 1:
            return [str(c) for c in panel_df.columns if str(c).strip().lower() != "label"]
        for col in ("gene", "gene_symbol", "symbol", "gene_name"):
            if col in panel_df.columns:
                return [str(x) for x in panel_df[col].dropna().tolist()]
        return [str(x) for x in panel_df.iloc[:, 0].dropna().tolist()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def sample_subjects(df: pd.DataFrame, n: int = 10, seed: int = 123) -> list[str]:
    subjects = sorted(df["subject"].astype(str).unique().tolist())
    if n >= len(subjects):
        return subjects
    rng = np.random.default_rng(seed)
    picked = rng.choice(subjects, size=n, replace=False).tolist()
    return sorted(str(x) for x in picked)


def sample_subjects_by_region_coverage(
    df: pd.DataFrame,
    *,
    n: int = 10,
    seed: int = 123,
    min_regions: int = 5,
    region_order: Sequence[str] = GTEX_TENSOR_REGION_ORDER,
) -> list[str]:
    subset = df.loc[df["dataset"].astype(str).str.upper().eq("GTEX")].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["tissue_or_parcel"] = subset["tissue_or_parcel"].astype(str)
    valid_regions = {str(r) for r in region_order}
    subset = subset[subset["tissue_or_parcel"].isin(valid_regions)]
    coverage = (
        subset.groupby("subject")["tissue_or_parcel"]
        .nunique()
        .rename("n_regions")
        .reset_index()
    )
    eligible = coverage.loc[coverage["n_regions"].ge(int(min_regions)), "subject"].astype(str).tolist()
    eligible = sorted(eligible)
    if not eligible:
        raise ValueError(f"No GTEx subjects have at least {min_regions} observed regions.")
    if n >= len(eligible):
        return eligible
    rng = np.random.default_rng(seed)
    picked = rng.choice(eligible, size=n, replace=False).tolist()
    return sorted(str(x) for x in picked)


def sample_genes(panel_genes: Sequence[str], available_genes: Sequence[str], n: int = 20, seed: int = 123) -> list[str]:
    available_upper = {str(g).upper(): str(g) for g in available_genes}
    genes = [available_upper[g.upper()] for g in panel_genes if str(g).upper() in available_upper]
    genes = list(dict.fromkeys(genes))
    if n >= len(genes):
        return genes
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.choice(genes, size=n, replace=False).tolist()]


def sample_regions(region_order: Sequence[str], n: int | None = None, seed: int = 123) -> list[str]:
    regions = [str(r) for r in region_order]
    if n is None or int(n) >= len(regions):
        return regions
    rng = np.random.default_rng(seed)
    sampled = {str(x) for x in rng.choice(regions, size=int(n), replace=False).tolist()}
    return [r for r in regions if r in sampled]


def build_gtex_tissue_tensor(
    df: pd.DataFrame,
    *,
    subjects: Sequence[str],
    genes: Sequence[str],
    region_order: Sequence[str] = GTEX_TENSOR_REGION_ORDER,
) -> TensorView:
    subset = df.loc[df["dataset"].astype(str).str.upper().eq("GTEX")].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["tissue_or_parcel"] = subset["tissue_or_parcel"].astype(str)
    dupes = subset.groupby(["subject", "tissue_or_parcel"]).size()
    dupes = dupes[dupes.gt(1)]
    if not dupes.empty:
        raise ValueError(
            "Expected one GTEx row per subject x tissue. "
            f"Found duplicates: {dupes.head().to_dict()}"
        )

    subjects = [str(s) for s in subjects]
    genes = [str(g) for g in genes]
    regions = [str(r) for r in region_order]
    values = np.full((len(subjects), len(regions), len(genes)), np.nan, dtype=np.float64)
    observed_mask = np.zeros((len(subjects), len(regions)), dtype=bool)
    subject_to_i = {s: i for i, s in enumerate(subjects)}
    region_to_j = {r: j for j, r in enumerate(regions)}

    subset = subset[subset["subject"].isin(subjects) & subset["tissue_or_parcel"].isin(regions)]
    for _, row in subset.iterrows():
        i = subject_to_i[row["subject"]]
        j = region_to_j[row["tissue_or_parcel"]]
        values[i, j, :] = row[genes].to_numpy(dtype=np.float64)
        observed_mask[i, j] = True

    return TensorView(
        values=values,
        observed_mask=observed_mask,
        subjects=subjects,
        regions=regions,
        genes=genes,
        dataset="GTEx",
        region_axis_kind="gtex_tissue",
    )


def build_sampled_gtex_tissue_tensor(
    samples: pd.DataFrame | str | Path,
    *,
    gene_panel: str | Path | Sequence[str] = "richiardi2015",
    n_subjects: int = 20,
    n_regions: int | None = None,
    n_genes: int = 50,
    min_regions_per_subject: int = 5,
    random_seed: int = 42,
    region_order: Sequence[str] = GTEX_TENSOR_REGION_ORDER,
) -> TensorView:
    df = load_gxp_samples_table(samples) if isinstance(samples, (str, Path)) else samples.copy()
    gtex_df = df.loc[df["dataset"].astype(str).str.upper().eq("GTEX")].copy()
    gene_cols = [c for c in gtex_df.columns if c not in META_COLS]
    panel_genes = load_gene_panel(gene_panel) if isinstance(gene_panel, (str, Path)) else [str(x) for x in gene_panel]
    selected_subjects = sample_subjects_by_region_coverage(
        gtex_df,
        n=n_subjects,
        seed=random_seed,
        min_regions=min_regions_per_subject,
        region_order=region_order,
    )
    selected_regions = sample_regions(region_order, n=n_regions, seed=random_seed)
    selected_genes = sample_genes(panel_genes, gene_cols, n=n_genes, seed=random_seed)
    return build_gtex_tissue_tensor(
        gtex_df,
        subjects=selected_subjects,
        genes=selected_genes,
        region_order=selected_regions,
    )


def summarize_tensor_selection(tensor_view: TensorView) -> pd.DataFrame:
    return pd.concat(
        [
            pd.Series(tensor_view.subjects, name="selected_subject"),
            pd.Series(tensor_view.regions, name="selected_region"),
            pd.Series(tensor_view.genes, name="selected_gene"),
        ],
        axis=1,
    )


def _robust_vrange(values: np.ndarray, *, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if int(finite.size) == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, lower))
    vmax = float(np.nanpercentile(finite, upper))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _resolve_box_aspect(
    values: np.ndarray,
    box_aspect: str | Sequence[float] | None,
) -> tuple[float, float, float]:
    if box_aspect is None or (isinstance(box_aspect, str) and box_aspect.lower() == "data"):
        return tuple(float(max(1, int(x))) for x in values.shape)
    if isinstance(box_aspect, str) and box_aspect.lower() == "compressed_data":
        return tuple(float(max(1.0, math.sqrt(max(1, int(x))))) for x in values.shape)
    if isinstance(box_aspect, str) and box_aspect.lower() == "equal":
        return (1.0, 1.0, 1.0)
    if len(box_aspect) != 3:
        raise ValueError("box_aspect must be None, 'data', 'compressed_data', 'equal', or a length-3 sequence.")
    return tuple(float(max(1e-6, float(x))) for x in box_aspect)


def _resolve_axis_assignment(axis_assignment: Sequence[str] | None) -> tuple[str, str, str]:
    if axis_assignment is None:
        return DEFAULT_TENSOR_AXIS_ASSIGNMENT
    axis_assignment = tuple(str(x).lower() for x in axis_assignment)
    if len(axis_assignment) != 3 or set(axis_assignment) != set(TENSOR_AXIS_COMPONENTS):
        raise ValueError(
            "axis_assignment must be a permutation of ('gene', 'subject', 'region')."
        )
    return axis_assignment


def _axis_payload(
    tensor_view: TensorView,
    axis_assignment: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]], tuple[str, str, str]]:
    axis_assignment = _resolve_axis_assignment(axis_assignment)
    base_values = np.transpose(tensor_view.values[:, ::-1, :], (2, 0, 1))
    base_observed = np.transpose(
        np.repeat(tensor_view.observed_mask[:, ::-1, None], len(tensor_view.genes), axis=2),
        (2, 0, 1),
    )
    semantic_to_base_axis = {"gene": 0, "subject": 1, "region": 2}
    perm = tuple(semantic_to_base_axis[name] for name in axis_assignment)
    values = np.transpose(base_values, perm)
    observed = np.transpose(base_observed, perm)
    labels = {
        "gene": list(tensor_view.genes),
        "subject": list(tensor_view.subjects),
        "region": list(tensor_view.regions[::-1]),
    }
    return values, observed, labels, axis_assignment


def _ticklabel_style_for_semantic(semantic_name: str, axis_name: str) -> dict[str, object]:
    if semantic_name == "subject":
        if axis_name == "y":
            return {"rotation": 0, "ha": "right", "va": "center"}
        return {"rotation": 0, "ha": "center", "va": "center"}
    if semantic_name == "region":
        if axis_name == "y":
            return {"rotation": 30, "ha": "right", "va": "center"}
        return {"rotation": 0, "ha": "right", "va": "center"}
    return {"rotation": 0, "ha": "center", "va": "center"}


def _resolve_tick_step(count: int, requested: int | str | None, *, max_labels: int) -> int:
    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        return max(1, int(math.ceil(max(1, count) / max(1, max_labels))))
    return max(1, int(requested))


def _format_axis_labels(
    labels: Sequence[str],
    *,
    semantic_name: str,
    mode: str,
) -> list[str]:
    mode = str(mode).lower()
    if semantic_name == "region":
        return [pretty_gtex_label(x) for x in labels]
    if semantic_name == "subject":
        if mode == "index":
            return [f"S{i+1}" for i in range(len(labels))]
        if mode == "suffix":
            return [str(x).split("-")[-1] for x in labels]
        return [str(x) for x in labels]
    if semantic_name == "gene":
        if mode == "index":
            return [f"G{i+1}" for i in range(len(labels))]
        return [str(x) for x in labels]
    return [str(x) for x in labels]


def plot_gtex_observation_mask(
    tensor_view: TensorView,
    *,
    figsize: tuple[float, float] = (8.5, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    mask = tensor_view.observed_mask.T.astype(float)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = ListedColormap(["#edf2f7", "#2b6cb0"])
    ax.imshow(mask, aspect="auto", interpolation="none", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("GTEx observation mask by subject and region")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Region")
    ax.set_xticks(np.arange(len(tensor_view.subjects)))
    ax.set_xticklabels(tensor_view.subjects, rotation=90)
    ax.set_yticks(np.arange(len(tensor_view.regions)))
    ax.set_yticklabels([pretty_gtex_label(x) for x in tensor_view.regions])
    for tick, region in zip(ax.get_yticklabels(), tensor_view.regions):
        tick.set_color(GTEX_TENSOR_GROUP_COLOR[GTEX_TENSOR_REGION_GROUP[region]])
    apply_tick_style(ax)
    legend_handles = [
        Patch(facecolor="#2b6cb0", edgecolor="none", label="Observed"),
        Patch(facecolor="#edf2f7", edgecolor="none", label="Missing"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cortical"], edgecolor="none", label="Cortical"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["basal_ganglia"], edgecolor="none", label="Basal ganglia"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["limbic_midbrain"], edgecolor="none", label="Limbic / midbrain"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cerebellar"], edgecolor="none", label="Cerebellar"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return fig, ax


def plot_gtex_tensor_voxels(
    tensor_view: TensorView,
    *,
    cmap: str = "viridis",
    missing_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0),
    edgecolor: str = "white",
    linewidth: float = 0.18,
    figsize: tuple[float, float] = (13.5, 9.5),
    axes_bbox: Sequence[float] = (0.03, 0.06, 0.78, 0.88),
    colorbar_bbox: Sequence[float] = (0.86, 0.16, 0.025, 0.68),
    box_aspect: str | Sequence[float] | None = "compressed_data",
    axis_assignment: Sequence[str] | None = None,
    elev: float = 20.0,
    azim: float = 24.0,
    subject_tick_step: int | str | None = "auto",
    region_tick_step: int | str | None = None,
    gene_tick_step: int | str | None = "auto",
    max_subject_ticklabels: int = 20,
    max_region_ticklabels: int = 12,
    max_gene_ticklabels: int = 10,
    region_tick_pad: float = 20.0,
    show_axis_labels: bool = False,
    show_x_axis_label: bool | None = True,
    show_y_axis_label: bool | None = None,
    show_z_axis_label: bool | None = None,
    show_subject_ticklabels: bool = True,
    show_region_ticklabels: bool = True,
    show_gene_ticklabels: bool = False,
    subject_label_mode: str = "full",
    gene_label_mode: str = "full",
    show_region_label_colors: bool = False,
    show_axis_lines: bool = True,
    show_tick_lines: bool = True,
    show_legend: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    values, observed, labels_by_semantic, axis_assignment = _axis_payload(tensor_view, axis_assignment)
    filled = np.ones(values.shape, dtype=bool)

    vmin, vmax = _robust_vrange(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    facecolors = np.empty(values.shape + (4,), dtype=np.float32)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if not observed[i, j, k] or not np.isfinite(values[i, j, k]):
                    facecolors[i, j, k] = missing_rgba
                else:
                    facecolors[i, j, k] = scalar_map.to_rgba(float(values[i, j, k]))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axes_bbox, projection="3d")
    ax.voxels(
        filled,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidth=linewidth,
        shade=False,
    )
    ax.set_box_aspect(_resolve_box_aspect(values, box_aspect))
    ax.view_init(elev=elev, azim=azim)
    axis_label_map = {"gene": "Genes", "subject": "Subject", "region": "Region"}
    ax.set_title(
        "GTEx input tensor "
        f"(x={axis_assignment[0]}, y={axis_assignment[1]}, z={axis_assignment[2]})"
    )
    show_x_axis_label = show_axis_labels if show_x_axis_label is None else show_x_axis_label
    show_y_axis_label = show_axis_labels if show_y_axis_label is None else show_y_axis_label
    show_z_axis_label = show_axis_labels if show_z_axis_label is None else show_z_axis_label
    ax.set_xlabel(axis_label_map[axis_assignment[0]] if show_x_axis_label else "")
    ax.set_ylabel(axis_label_map[axis_assignment[1]] if show_y_axis_label else "")
    ax.set_zlabel(axis_label_map[axis_assignment[2]] if show_z_axis_label else "")

    tick_step_map = {
        "gene": _resolve_tick_step(len(labels_by_semantic["gene"]), gene_tick_step, max_labels=max_gene_ticklabels),
        "subject": _resolve_tick_step(len(labels_by_semantic["subject"]), subject_tick_step, max_labels=max_subject_ticklabels),
        "region": _resolve_tick_step(len(labels_by_semantic["region"]), region_tick_step, max_labels=max_region_ticklabels),
    }
    show_ticklabel_map = {
        "gene": show_gene_ticklabels,
        "subject": show_subject_ticklabels,
        "region": show_region_ticklabels,
    }

    x_labels = labels_by_semantic[axis_assignment[0]]
    y_labels = labels_by_semantic[axis_assignment[1]]
    z_labels = labels_by_semantic[axis_assignment[2]]
    x_display = _format_axis_labels(x_labels, semantic_name=axis_assignment[0], mode=subject_label_mode if axis_assignment[0] == "subject" else gene_label_mode if axis_assignment[0] == "gene" else "full")
    y_display = _format_axis_labels(y_labels, semantic_name=axis_assignment[1], mode=subject_label_mode if axis_assignment[1] == "subject" else gene_label_mode if axis_assignment[1] == "gene" else "full")
    z_display = _format_axis_labels(z_labels, semantic_name=axis_assignment[2], mode=subject_label_mode if axis_assignment[2] == "subject" else gene_label_mode if axis_assignment[2] == "gene" else "full")
    x_style = _ticklabel_style_for_semantic(axis_assignment[0], "x")
    y_style = _ticklabel_style_for_semantic(axis_assignment[1], "y")
    z_style = _ticklabel_style_for_semantic(axis_assignment[2], "z")

    x_idx = np.arange(0, len(x_labels), tick_step_map[axis_assignment[0]])
    x_ticks = x_idx + 0.5
    y_idx = np.arange(0, len(y_labels), tick_step_map[axis_assignment[1]])
    y_ticks = y_idx + 0.5
    z_idx = np.arange(0, len(z_labels), tick_step_map[axis_assignment[2]])
    z_ticks = z_idx + 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [x_display[i] for i in x_idx.tolist()] if show_ticklabel_map[axis_assignment[0]] else [],
        rotation=x_style["rotation"],
        ha=x_style["ha"],
        fontsize=8,
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [y_display[i] for i in y_idx.tolist()] if show_ticklabel_map[axis_assignment[1]] else [],
        rotation=y_style["rotation"],
        ha=y_style["ha"],
        fontsize=8,
    )
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(
        [z_display[i] for i in z_idx.tolist()] if show_ticklabel_map[axis_assignment[2]] else [],
        rotation=z_style["rotation"],
        ha=z_style["ha"],
        fontsize=8,
    )
    tick_length = 3.5 if show_tick_lines else 0.0
    ax.tick_params(axis="x", length=tick_length)
    ax.tick_params(axis="y", length=tick_length)
    region_axis_name = {axis_assignment[0]: "x", axis_assignment[1]: "y", axis_assignment[2]: "z"}["region"]
    ax.tick_params(axis=region_axis_name, pad=region_tick_pad)
    ax.tick_params(axis="z", length=tick_length)
    if show_tick_lines:
        tick_factors = (0.2, 0.1)
    else:
        tick_factors = (0.0, 0.0)
    for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis), ("z", ax.zaxis)):
        try:
            axis._axinfo["tick"]["inward_factor"] = tick_factors[0]
            axis._axinfo["tick"]["outward_factor"] = tick_factors[1]
        except Exception:
            pass
    for tick, semantic_name, region_lookup in (
        (ax.get_xticklabels(), axis_assignment[0], [x_labels[i] for i in x_idx.tolist()]),
        (ax.get_yticklabels(), axis_assignment[1], [y_labels[i] for i in y_idx.tolist()]),
        (ax.get_zticklabels(), axis_assignment[2], [z_labels[i] for i in z_idx.tolist()]),
    ):
        if semantic_name == "region" and show_region_ticklabels:
            for label, region in zip(tick, region_lookup):
                label.set_horizontalalignment("right")
                label.set_verticalalignment("center")
                if show_region_label_colors:
                    label.set_color(GTEX_TENSOR_GROUP_COLOR[GTEX_TENSOR_REGION_GROUP[region]])
        elif semantic_name == "subject" and show_subject_ticklabels:
            for label in tick:
                label.set_horizontalalignment("right")
                label.set_verticalalignment("center")
        elif semantic_name == "gene" and show_gene_ticklabels:
            for label in tick:
                label.set_horizontalalignment("center")
                label.set_verticalalignment("center")

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
        if not show_axis_lines:
            try:
                axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass

    cax = fig.add_axes(colorbar_bbox)
    cbar = fig.colorbar(scalar_map, cax=cax)
    cbar.set_label("log1p(TPM)")

    if show_legend:
        legend_handles = [
            Patch(facecolor=missing_rgba[:3], edgecolor="none", alpha=missing_rgba[3], label="Missing region vector"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cortical"], edgecolor="none", label="Cortical"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["basal_ganglia"], edgecolor="none", label="Basal ganglia"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["limbic_midbrain"], edgecolor="none", label="Limbic / midbrain"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cerebellar"], edgecolor="none", label="Cerebellar"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return fig, ax


def plot_sampled_gtex_tensor(
    samples: pd.DataFrame | str | Path = "data/raw/gxp_samples.csv",
    *,
    gene_panel: str | Path | Sequence[str] = "richiardi2015",
    n_subjects: int = 20,
    n_regions: int | None = None,
    n_genes: int = 50,
    min_regions_per_subject: int = 5,
    random_seed: int = 42,
    region_order: Sequence[str] = GTEX_TENSOR_REGION_ORDER,
    cmap: str = "viridis",
    missing_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0),
    edgecolor: str = "white",
    linewidth: float = 0.18,
    figsize: tuple[float, float] = (13.5, 9.5),
    axes_bbox: Sequence[float] = (0.03, 0.06, 0.78, 0.88),
    colorbar_bbox: Sequence[float] = (0.86, 0.16, 0.025, 0.68),
    box_aspect: str | Sequence[float] | None = "compressed_data",
    axis_assignment: Sequence[str] | None = None,
    elev: float = 20.0,
    azim: float = 24.0,
    subject_tick_step: int | str | None = "auto",
    region_tick_step: int | str | None = None,
    gene_tick_step: int | str | None = "auto",
    max_subject_ticklabels: int = 20,
    max_region_ticklabels: int = 12,
    max_gene_ticklabels: int = 10,
    region_tick_pad: float = 20.0,
    show_axis_labels: bool = False,
    show_x_axis_label: bool | None = True,
    show_y_axis_label: bool | None = None,
    show_z_axis_label: bool | None = None,
    show_subject_ticklabels: bool = True,
    show_region_ticklabels: bool = True,
    show_gene_ticklabels: bool = False,
    subject_label_mode: str = "full",
    gene_label_mode: str = "full",
    show_region_label_colors: bool = False,
    show_axis_lines: bool = True,
    show_tick_lines: bool = True,
    show_legend: bool = False,
) -> tuple[plt.Figure, plt.Axes, TensorView, pd.DataFrame]:
    tensor_view = build_sampled_gtex_tissue_tensor(
        samples,
        gene_panel=gene_panel,
        n_subjects=n_subjects,
        n_regions=n_regions,
        n_genes=n_genes,
        min_regions_per_subject=min_regions_per_subject,
        random_seed=random_seed,
        region_order=region_order,
    )
    fig, ax = plot_gtex_tensor_voxels(
        tensor_view,
        cmap=cmap,
        missing_rgba=missing_rgba,
        edgecolor=edgecolor,
        linewidth=linewidth,
        figsize=figsize,
        axes_bbox=axes_bbox,
        colorbar_bbox=colorbar_bbox,
        box_aspect=box_aspect,
        axis_assignment=axis_assignment,
        elev=elev,
        azim=azim,
        subject_tick_step=subject_tick_step,
        region_tick_step=region_tick_step,
        gene_tick_step=gene_tick_step,
        max_subject_ticklabels=max_subject_ticklabels,
        max_region_ticklabels=max_region_ticklabels,
        max_gene_ticklabels=max_gene_ticklabels,
        region_tick_pad=region_tick_pad,
        show_axis_labels=show_axis_labels,
        show_x_axis_label=show_x_axis_label,
        show_y_axis_label=show_y_axis_label,
        show_z_axis_label=show_z_axis_label,
        show_subject_ticklabels=show_subject_ticklabels,
        show_region_ticklabels=show_region_ticklabels,
        show_gene_ticklabels=show_gene_ticklabels,
        subject_label_mode=subject_label_mode,
        gene_label_mode=gene_label_mode,
        show_region_label_colors=show_region_label_colors,
        show_axis_lines=show_axis_lines,
        show_tick_lines=show_tick_lines,
        show_legend=show_legend,
    )
    return fig, ax, tensor_view, summarize_tensor_selection(tensor_view)


__all__ = [
    "DEFAULT_TENSOR_AXIS_ASSIGNMENT",
    "GTEX_TENSOR_GROUP_COLOR",
    "GTEX_TENSOR_REGION_GROUP",
    "GTEX_TENSOR_REGION_ORDER",
    "TensorView",
    "build_gtex_tissue_tensor",
    "build_sampled_gtex_tissue_tensor",
    "load_gene_panel",
    "load_gxp_samples_table",
    "plot_gtex_observation_mask",
    "plot_sampled_gtex_tensor",
    "plot_gtex_tensor_voxels",
    "sample_regions",
    "sample_genes",
    "sample_subjects",
    "sample_subjects_by_region_coverage",
    "summarize_tensor_selection",
]
