#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_utils.eval_samples import plot_sampled_tensor


def _parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {text!r}")


def _parse_optional_bool(text: str | bool | None) -> bool | None:
    if text is None:
        return None
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"", "none", "null"}:
        return None
    return _parse_bool(value)


def _parse_optional_int(text: str | int | None) -> int | None:
    if text is None:
        return None
    if isinstance(text, int):
        return text
    value = str(text).strip().lower()
    if value in {"", "none", "null"}:
        return None
    return int(value)


def _parse_tick_step(text: str | int | None) -> int | None:
    return _parse_optional_int(text)


def _parse_auto_or_float(text: str | float) -> str | float:
    value = str(text).strip()
    if value.lower() == "auto":
        return "auto"
    return float(value)


def _parse_auto_or_tuple(text: str, *, length: int) -> str | tuple[float, ...]:
    value = str(text).strip()
    if value.lower() == "auto":
        return "auto"
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"Expected {length} comma-separated values or 'auto', got {text!r}")
    return tuple(parts)


def _parse_axis_assignment(text: str) -> tuple[str, str, str]:
    parts = tuple(x.strip().lower() for x in str(text).split(",") if x.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("axis-assignment must have three comma-separated entries.")
    return parts


def add_tensor_args(parser: argparse.ArgumentParser, *, dataset: str) -> argparse.ArgumentParser:
    default_subjects = 300 if dataset == "gtex" else 6
    default_regions = "none"
    default_subject_step = "10" if dataset == "gtex" else "1"
    default_region_step = "1" if dataset == "gtex" else "10"
    default_out = f"out/tensor_renders/{dataset}_gxp_tensor.png"

    parser.add_argument("--csv-path", default="data/raw/gxp_samples.csv")
    parser.add_argument("--active-gene-list", default="allgenes_stable_r0.2")
    parser.add_argument(
        "--region-ordering",
        default="dataset",
        choices=("dataset", "region_matched", "region_matched_superset"),
    )
    parser.add_argument("--matching-policy", default="centroids_and_volumes", choices=("centroids", "centroids_and_volumes"))
    parser.add_argument("--matching-policy-hemi-mode", default="default", choices=("default", "force_left"))
    parser.add_argument("--collapse-cerebellum", type=_parse_bool, default=False)
    parser.add_argument("--gtex-rep-mode", default="centroid")
    parser.add_argument("--gtex-hemi-mode", default="mirror_left")
    parser.add_argument("--n-subjects", type=int, default=default_subjects)
    parser.add_argument("--n-genes", type=int, default=400)
    parser.add_argument("--n-regions", type=_parse_optional_int, default=_parse_optional_int(default_regions))
    parser.add_argument("--min-regions-per-subject", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--missing-alpha", type=float, default=1.0)
    parser.add_argument("--figsize", type=lambda x: _parse_auto_or_tuple(x, length=2), default="auto")
    parser.add_argument("--axes-bbox", type=lambda x: _parse_auto_or_tuple(x, length=4), default="auto")
    parser.add_argument("--colorbar-bbox", type=lambda x: _parse_auto_or_tuple(x, length=4), default="auto")
    parser.add_argument("--box-aspect", default="compressed_data")
    parser.add_argument("--box-zoom", type=_parse_auto_or_float, default=1.0)
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument("--title-pad", type=float, default=0.0)
    parser.add_argument("--axis-assignment", type=_parse_axis_assignment, default=("gene", "region", "subject"))
    parser.add_argument("--show-grid", type=_parse_bool, default=False)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=24.0)
    parser.add_argument("--subject-tick-step", type=_parse_tick_step, default=_parse_tick_step(default_subject_step))
    parser.add_argument("--gene-tick-step", type=_parse_tick_step, default=10)
    parser.add_argument("--region-tick-step", type=_parse_tick_step, default=_parse_tick_step(default_region_step))
    parser.add_argument("--tick-label-pad", type=float, default=-3.0)
    parser.add_argument("--subject-label-mode", default="full", choices=("index", "suffix", "full"))
    parser.add_argument("--gene-label-mode", default="full", choices=("index", "full"))
    parser.add_argument("--show-x-axis-label", type=_parse_optional_bool, default=True)
    parser.add_argument("--show-y-axis-label", type=_parse_optional_bool, default=None)
    parser.add_argument("--show-z-axis-label", type=_parse_optional_bool, default=None)
    parser.add_argument("--show-axis-labels", type=_parse_bool, default=False)
    parser.add_argument("--show-subject-ticklabels", type=_parse_bool, default=True)
    parser.add_argument("--show-region-ticklabels", type=_parse_bool, default=True)
    parser.add_argument("--show-region-label-colors", type=_parse_bool, default=False)
    parser.add_argument("--show-gene-ticklabels", type=_parse_bool, default=False)
    parser.add_argument("--show-axis-lines", type=_parse_bool, default=True)
    parser.add_argument("--show-tick-lines", type=_parse_bool, default=True)
    parser.add_argument("--show-legend", type=_parse_bool, default=False)
    parser.add_argument("--out-path", default=default_out)
    return parser


def render_dataset_tensor(dataset: str, argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=f"Render a {dataset.upper()} GXP sample tensor.")
    add_tensor_args(parser, dataset=dataset)
    args = parser.parse_args(argv)

    missing_rgba = (0.78, 0.78, 0.78, float(args.missing_alpha))
    fig, _ax, tensor_view, selection_summary = plot_sampled_tensor(
        args.csv_path,
        dataset=dataset,
        region_ordering=args.region_ordering,
        gene_panel=args.active_gene_list,
        n_subjects=args.n_subjects,
        n_regions=args.n_regions,
        n_genes=args.n_genes,
        min_regions_per_subject=args.min_regions_per_subject,
        random_seed=args.random_seed,
        matching_policy=args.matching_policy,
        matching_policy_hemi_mode=args.matching_policy_hemi_mode,
        collapse_cerebellum=args.collapse_cerebellum,
        gtex_rep_mode=args.gtex_rep_mode,
        gtex_hemi_mode=args.gtex_hemi_mode,
        missing_rgba=missing_rgba,
        figsize=args.figsize,
        dpi=args.dpi,
        axes_bbox=args.axes_bbox,
        colorbar_bbox=args.colorbar_bbox,
        box_aspect=args.box_aspect,
        box_zoom=args.box_zoom,
        title_pad=args.title_pad,
        axis_assignment=args.axis_assignment,
        elev=args.elev,
        azim=args.azim,
        subject_tick_step=args.subject_tick_step,
        gene_tick_step=args.gene_tick_step,
        region_tick_step=args.region_tick_step,
        tick_label_pad=args.tick_label_pad,
        subject_label_mode=args.subject_label_mode,
        gene_label_mode=args.gene_label_mode,
        show_x_axis_label=args.show_x_axis_label,
        show_y_axis_label=args.show_y_axis_label,
        show_z_axis_label=args.show_z_axis_label,
        show_axis_labels=args.show_axis_labels,
        show_subject_ticklabels=args.show_subject_ticklabels,
        show_region_ticklabels=args.show_region_ticklabels,
        show_region_label_colors=args.show_region_label_colors,
        show_gene_ticklabels=args.show_gene_ticklabels,
        show_axis_lines=args.show_axis_lines,
        show_tick_lines=args.show_tick_lines,
        show_grid=args.show_grid,
        show_legend=args.show_legend,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    selection_path = out_path.with_suffix(".selection.csv")
    selection_summary.to_csv(selection_path, index=False)
    print(
        f"Rendered {tensor_view.dataset} tensor: "
        f"{len(tensor_view.genes)} genes x {len(tensor_view.regions)} regions x {len(tensor_view.subjects)} subjects"
    )
    print(f"Image: {out_path}")
    print(f"Selection CSV: {selection_path}")
    return out_path
