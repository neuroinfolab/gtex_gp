#!/usr/bin/env python3
from __future__ import annotations

"""DLAM latent-alignment visualization helpers.

This module is the notebook-facing plotting surface for
``eval_dlam_alignment.ipynb``. Model extraction stays in
``dlam_diagnostics.py``; this file handles subject selection, display
coordinates, and publication-style alignment plots.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
import numpy as np

from .dlam_diagnostics import DlamSubjectDiagnostics
from .eda_core import EDAConfig
from .eval_style import apply_tick_style, font_size, set_academic_style
from .results_eda import compute_subject_metrics_from_cache, select_subject_by_model


@dataclass(frozen=True)
class AlignmentScoreBundle:
    pre: np.ndarray
    post: np.ndarray
    ref_obs: np.ndarray
    ref_full: np.ndarray
    full_post: np.ndarray


_DEFAULT_FONTS = {
    "title": "xxl+8",
    "panel_title": "xxl+4",
    "label": "xl+5",
    "tick": "xl+2",
    "annotation": "xl+4",
    "cbar_label": "xl+6",
    "cbar_tick": "xl+2",
}

COMPONENT_CMAPS = ("cividis", "plasma", "viridis")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _fonts() -> dict[str, int]:
    return {k: font_size(v) for k, v in _DEFAULT_FONTS.items()}


def _component_cmap(component: int, cmap: str | None) -> str:
    if cmap is None or str(cmap).lower() in {"auto", "component"}:
        return COMPONENT_CMAPS[int(component) % len(COMPONENT_CMAPS)]
    return str(cmap)


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _cache_model_dirname(cfg: object, model: str) -> str:
    m = str(model).lower()
    if m == "dlam":
        return str(getattr(cfg, "dlam_cache_dirname", "dlam"))
    if m == "plam":
        return str(getattr(cfg, "plam_cache_dirname", "plam"))
    if m == "naive":
        return str(getattr(cfg, "naive_cache_dirname", "naive"))
    return m


def load_cache_run_config(
    cfg: object,
    *,
    model: str = "dlam",
    subject_id: str | None = None,
) -> dict[str, object]:
    """Read the run config embedded in a LORO subject-cache JSON file."""
    cache_root = _repo_path(getattr(cfg, "cache_root", cfg))
    gene_scope = str(getattr(cfg, "gene_scope", "allgenes")).lower()
    root = cache_root / gene_scope / _cache_model_dirname(cfg, model)
    if subject_id:
        candidates = [root / f"{subject_id}.json"]
    else:
        candidates = sorted(root.glob("*.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue
        config = meta.get("config", {})
        if isinstance(config, dict) and config:
            return dict(config)
    raise FileNotFoundError(f"No usable {model} cache config JSON found under {root}")


def cache_run_settings(
    cfg: object,
    *,
    model: str = "dlam",
    subject_id: str | None = None,
) -> dict[str, object]:
    """Return cache-derived settings that affect DLAM diagnostic refits."""
    config = load_cache_run_config(cfg, model=model, subject_id=subject_id)
    settings: dict[str, object] = {}
    direct_keys = [
        "csv_path",
        "hvg_path",
        "gene_scope",
        "min_observed_parcels",
        "combat_use_covariates",
        "drop_macro_system_covariate",
        "atlas_agg",
        "gtex_rep_mode",
        "gtex_hemi_mode",
        "matching_policy",
        "matching_policy_hemi_mode",
        "collapse_cerebellum",
        "n_comp_target",
        "ridge_alpha_bridge",
        "rbf_smoothing",
        "gp_length_scale",
        "seed",
        "c_min",
        "anchor_distance_shrink",
        "anchor_distance_d0",
        "anchor_distance_tau",
    ]
    for key in direct_keys:
        if key in config:
            settings[key] = config[key]
    if "dlam_strategy" in config:
        settings["strategy"] = config["dlam_strategy"]
    if "dlam_spatial_method" in config:
        settings["spatial_method"] = config["dlam_spatial_method"]
    return settings


def select_subject_for_alignment(
    cfg: object,
    *,
    model: str = "dlam",
    mode: str = "median",
    metric: str = "pearson_r",
    require_n_observed: int | None = 12,
) -> str:
    """Select a representative subject, optionally within a coverage stratum."""
    if require_n_observed is None:
        return select_subject_by_model(cfg, mode=mode, metric=metric, model=model)

    d = compute_subject_metrics_from_cache(cfg, model=model).dropna(subset=[metric]).copy()
    d = d[d["coverage"].astype(int) == int(require_n_observed)].copy()
    if len(d) == 0:
        raise RuntimeError(f"No {model} subjects found with coverage={require_n_observed}")

    if mode == "best":
        row = d.sort_values([metric, "subject"], ascending=[False if metric == "pearson_r" else True, True]).iloc[0]
    elif mode == "worst":
        row = d.sort_values([metric, "subject"], ascending=[True if metric == "pearson_r" else False, True]).iloc[0]
    else:
        med = float(d[metric].median())
        d["delta_med"] = np.abs(d[metric] - med)
        row = d.sort_values(["delta_med", "subject"], ascending=[True, True]).iloc[0]

    print(
        f"selected {row['subject']} from n={len(d)} subjects with "
        f"coverage={require_n_observed}; {metric}={float(row[metric]):.4f}"
    )
    return str(row["subject"])


def score_bundle(
    diag: DlamSubjectDiagnostics,
    *,
    component: int = 0,
    latent_space: str = "spatial",
) -> AlignmentScoreBundle:
    comp = int(component)
    if comp < 0 or comp >= int(diag.n_comp):
        raise ValueError(f"component must be in [0, {int(diag.n_comp) - 1}]")
    mode = str(latent_space).lower()
    if mode == "expression":
        pre = np.asarray(diag.u_obs[:, comp], dtype=np.float64)
        post = np.asarray(diag.u_prime_obs[:, comp], dtype=np.float64)
    elif mode == "spatial":
        pre = np.asarray(diag.t_obs[:, comp], dtype=np.float64)
        post = np.asarray(diag.t_prime_obs[:, comp], dtype=np.float64)
    else:
        raise ValueError("latent_space must be one of: spatial, expression")
    return AlignmentScoreBundle(
        pre=pre,
        post=post,
        ref_obs=np.asarray(diag.t_ref_obs[:, comp], dtype=np.float64),
        ref_full=np.asarray(diag.t_ref_full[:, comp], dtype=np.float64),
        full_post=np.asarray(diag.t_full_prime[:, comp], dtype=np.float64),
    )


def robust_limits(arrays: Sequence[np.ndarray], pct: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64)[np.isfinite(a)] for a in arrays if np.asarray(a).size])
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(vals, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)


def full_with_train_values(diag: DlamSubjectDiagnostics, values_train: np.ndarray) -> np.ndarray:
    out = np.full(len(diag.target_meta), np.nan, dtype=np.float64)
    out[np.asarray(diag.train_idx, dtype=int)] = np.asarray(values_train, dtype=np.float64)
    return out


def display_coords(
    diag: DlamSubjectDiagnostics,
    *,
    undo_force_left: bool = True,
    observed_from_gtex_labels: bool = False,
) -> np.ndarray:
    """Return coordinates for visualization, not modeling.

    The cache may use force-left matching so GTEx right-hemisphere observations
    share left-hemisphere target parcels. For brain views, this helper restores
    the target parcel display side using AHBA target hemisphere metadata. By
    default it does not move observed GTEx matches by their original GTEx labels:
    under force-left preprocessing those highlighted matches should remain on
    their left AHBA target parcels. The parcel index and model values are
    unchanged.
    """
    coords = np.asarray(diag.coords_full, dtype=np.float64).copy()
    if not undo_force_left:
        return coords

    hemi = diag.target_meta.get("hemisphere")
    if hemi is not None:
        h = hemi.astype(str).to_numpy()
        coords[h == "L", 0] = -np.abs(coords[h == "L", 0])
        coords[h == "R", 0] = np.abs(coords[h == "R", 0])

    if observed_from_gtex_labels:
        # Off by default for force-left displays. This branch is useful only for
        # non-force-left exploratory views; broad labels like "Cerebellar
        # hemisphere" can otherwise be falsely read as right-hemisphere labels.
        for parcel_idx, label in getattr(diag, "gtex_label_by_parcel", {}).items():
            lab = str(label).lower()
            idx = int(parcel_idx)
            if idx < 0 or idx >= coords.shape[0]:
                continue
            tokens = {tok.strip("()[]{}:;,.").lower() for tok in lab.replace("-", " ").replace("_", " ").split()}
            if "right" in tokens or "rh" in tokens:
                coords[idx, 0] = np.abs(coords[idx, 0])
            elif "left" in tokens or "lh" in tokens:
                coords[idx, 0] = -np.abs(coords[idx, 0])
    return coords


def _plot_nilearn_markers(ax, vals, coords, display_mode, cmap, vmin, vmax, node_size):
    from nilearn import plotting

    vals = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(vals)
    plotting.plot_markers(
        vals[finite],
        coords[finite],
        node_size=node_size,
        node_cmap=cmap,
        node_vmin=vmin,
        node_vmax=vmax,
        display_mode=display_mode,
        axes=ax,
        colorbar=False,
        annotate=False,
        title=None,
    )


def _highlight_matched_on_nilearn(ax, vals, coords, train_idx, display_mode, cmap, vmin, vmax, matched_node_size):
    train_idx = np.asarray(train_idx, dtype=int)
    vals = np.asarray(vals, dtype=np.float64)
    matched_vals = np.full(vals.shape, np.nan, dtype=np.float64)
    matched_vals[train_idx] = vals[train_idx]
    m = np.isfinite(matched_vals)
    _plot_nilearn_markers(
        ax,
        np.zeros(int(m.sum()), dtype=np.float64),
        coords[m],
        display_mode,
        ListedColormap(["black"]),
        0.0,
        1.0,
        matched_node_size,
    )
    _plot_nilearn_markers(
        ax,
        matched_vals[m],
        coords[m],
        display_mode,
        cmap,
        vmin,
        vmax,
        max(1, int(0.62 * matched_node_size)),
    )


def _plot_score_scatter(ax, ahba_vals, gtex_vals, title, ylabel, *, cmap: str, vmin: float, vmax: float):
    f = _fonts()
    xvals = np.asarray(ahba_vals, dtype=np.float64)
    yvals = np.asarray(gtex_vals, dtype=np.float64)
    ax.scatter(
        xvals,
        yvals,
        c=xvals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=150,
        alpha=0.92,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )
    lo, hi = robust_limits([xvals, yvals], pct=(0.0, 100.0))
    pad = 0.09 * max(hi - lo, 1e-6)
    finite = np.isfinite(xvals) & np.isfinite(yvals)
    if int(finite.sum()) >= 2:
        slope, intercept = np.polyfit(xvals[finite], yvals[finite], deg=1)
        xx = np.array([lo, hi], dtype=np.float64)
        ax.plot(xx, slope * xx + intercept, color="black", linestyle="--", linewidth=2.4, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    r = np.corrcoef(xvals, yvals)[0, 1] if len(xvals) > 1 else np.nan
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\nn = {len(xvals)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=f["annotation"],
        bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )
    ax.set_title(title, fontsize=f["panel_title"], weight="bold", pad=14)
    ax.set_xlabel("AHBA reference score", fontsize=f["label"])
    ax.set_ylabel(ylabel, fontsize=f["label"])
    apply_tick_style(ax, label_fontsize=f["tick"])
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.22, linewidth=1.0)


def render_nilearn_alignment_component(
    diag: DlamSubjectDiagnostics,
    *,
    component: int = 0,
    latent_space: str = "spatial",
    cmap: str | None = None,
    clim_pct: tuple[float, float] = (2.0, 98.0),
    node_size: int = 72,
    matched_node_size: int = 135,
    undo_force_left_for_display: bool = True,
    figsize: tuple[float, float] = (29.0, 5.8),
) -> tuple[plt.Figure, np.ndarray]:
    """Render one component: observed pre/post, AHBA sagittal/coronal, scatters."""
    f = _fonts()
    set_academic_style()
    cmap_use = _component_cmap(component, cmap)
    bundle = score_bundle(diag, component=component, latent_space=latent_space)
    coords = display_coords(diag, undo_force_left=undo_force_left_for_display)
    pre_full = full_with_train_values(diag, bundle.pre)
    post_full = full_with_train_values(diag, bundle.post)
    vmin, vmax = robust_limits([pre_full, post_full, bundle.ref_full], pct=clim_pct)

    fig, axes = plt.subplots(
        1,
        7,
        figsize=figsize,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.72, 0.22, 1.05, 1.05]},
    )
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.18, top=0.80, wspace=0.18)
    brain_specs = [
        (f"{diag.subject_id}\npre-aligned", pre_full, "x", False, node_size),
        (f"{diag.subject_id}\npost-aligned", post_full, "x", False, node_size),
        ("AHBA reference", bundle.ref_full, "x", True, max(1, int(0.82 * node_size))),
        ("AHBA reference", bundle.ref_full, "y", True, max(1, int(0.82 * node_size))),
    ]
    for ax, (title, vals, display_mode, highlight_matched, size_use) in zip(axes[:4], brain_specs):
        _plot_nilearn_markers(ax, vals, coords, display_mode, cmap_use, vmin, vmax, size_use)
        if highlight_matched:
            _highlight_matched_on_nilearn(
                ax,
                vals,
                coords,
                diag.train_idx,
                display_mode,
                cmap_use,
                vmin,
                vmax,
                matched_node_size,
            )
        label = title if title == "AHBA reference" else f"GTEx-{title}"
        ax.set_title(label, fontsize=f["panel_title"], weight="bold", pad=16)

    axes[4].axis("off")

    _plot_score_scatter(
        axes[5],
        bundle.ref_obs,
        bundle.pre,
        "Pre vs AHBA",
        "Pre-aligned GTEx score",
        cmap=cmap_use,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_score_scatter(
        axes[6],
        bundle.ref_obs,
        bundle.post,
        "Post vs AHBA",
        "Post-aligned GTEx score",
        cmap=cmap_use,
        vmin=vmin,
        vmax=vmax,
    )

    sm = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_use)
    sm.set_array([])
    cax = fig.add_axes([0.05, 0.30, 0.0065, 0.42])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"LV{int(component) + 1} {latent_space} score", fontsize=f["cbar_label"], fontweight="bold", labelpad=14)
    cbar.ax.tick_params(labelsize=f["cbar_tick"])
    return fig, axes


def render_nilearn_alignment_components(
    diag: DlamSubjectDiagnostics,
    *,
    components: Sequence[int] | None = None,
    latent_space: str = "spatial",
    cmap: str | None = None,
    clim_pct: tuple[float, float] = (2.0, 98.0),
    node_size: int = 72,
    matched_node_size: int = 135,
    undo_force_left_for_display: bool = True,
) -> list[tuple[plt.Figure, np.ndarray]]:
    """Render the same nilearn+scatter panel for every requested component."""
    if components is None:
        components = list(range(int(diag.n_comp)))
    out = []
    for comp in components:
        if int(comp) < int(diag.n_comp):
            out.append(
                render_nilearn_alignment_component(
                    diag,
                    component=int(comp),
                    latent_space=latent_space,
                    cmap=cmap,
                    clim_pct=clim_pct,
                    node_size=node_size,
                    matched_node_size=matched_node_size,
                    undo_force_left_for_display=undo_force_left_for_display,
                )
            )
    return out


def plot_coordinate_alignment_component(
    diag: DlamSubjectDiagnostics,
    *,
    component: int = 0,
    latent_space: str = "spatial",
    axis_x: str = "y",
    axis_y: str = "z",
    cmap: str | None = None,
    undo_force_left_for_display: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Lightweight non-nilearn fallback with the same observed alignment content."""
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    ix, iy = axis_to_idx[str(axis_x).lower()], axis_to_idx[str(axis_y).lower()]
    f = _fonts()
    cmap_use = _component_cmap(component, cmap)
    coords_all = display_coords(diag, undo_force_left=undo_force_left_for_display)
    coords_train = coords_all[np.asarray(diag.train_idx, dtype=int)]
    bundle = score_bundle(diag, component=component, latent_space=latent_space)
    vmin, vmax = robust_limits([bundle.pre, bundle.post, bundle.ref_full], pct=(2.0, 98.0))

    fig, axes = plt.subplots(1, 5, figsize=(21.0, 4.8), constrained_layout=True)
    specs = [
        ("Pre-aligned GTEx", coords_train, bundle.pre, 88, 0.96),
        ("Post-aligned GTEx", coords_train, bundle.post, 88, 0.96),
        ("AHBA reference", coords_all, bundle.ref_full, 38, 0.92),
    ]
    sc = None
    for ax, (title, coords, vals, size, alpha) in zip(axes[:3], specs):
        ax.scatter(coords_all[:, ix], coords_all[:, iy], s=18, color="#dddddd", alpha=0.32, linewidths=0)
        sc = ax.scatter(
            coords[:, ix],
            coords[:, iy],
            c=vals,
            cmap=cmap_use,
            vmin=vmin,
            vmax=vmax,
            s=size,
            alpha=alpha,
            edgecolor="white",
            linewidth=1.0,
        )
        if title == "AHBA reference":
            ax.scatter(coords_train[:, ix], coords_train[:, iy], s=145, facecolor="none", edgecolor="black", linewidth=1.25)
        ax.set_title(title, fontsize=f["panel_title"], weight="bold")
        ax.set_xlabel(f"{str(axis_x).upper()} coord", fontsize=f["label"])
        ax.set_ylabel(f"{str(axis_y).upper()} coord", fontsize=f["label"])
        apply_tick_style(ax, label_fontsize=f["tick"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.grid(False)

    _plot_score_scatter(
        axes[3],
        bundle.ref_obs,
        bundle.pre,
        "Pre vs AHBA",
        "Pre-aligned GTEx score",
        cmap=cmap_use,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_score_scatter(
        axes[4],
        bundle.ref_obs,
        bundle.post,
        "Post vs AHBA",
        "Post-aligned GTEx score",
        cmap=cmap_use,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(sc, ax=axes[:3].tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("Aligned latent score", fontsize=f["cbar_label"])
    cbar.ax.tick_params(labelsize=f["cbar_tick"])
    fig.suptitle(
        f"DLAM alignment | {diag.subject_id} | component {int(component) + 1} | {latent_space}",
        fontsize=f["title"],
        weight="bold",
    )
    return fig, axes


__all__ = [
    "AlignmentScoreBundle",
    "COMPONENT_CMAPS",
    "EDAConfig",
    "cache_run_settings",
    "display_coords",
    "full_with_train_values",
    "load_cache_run_config",
    "plot_coordinate_alignment_component",
    "render_nilearn_alignment_component",
    "render_nilearn_alignment_components",
    "robust_limits",
    "score_bundle",
    "select_subject_for_alignment",
]
