"""Age-progression of PLS-gradient (LV) scores for a single parcel.

A second extension of :mod:`eval_pls_gradients`: instead of rendering one subject's
projection on the brain, we ask how the *molecular LV scores* of a given parcel move
**with age**. For each subject we project their region x gene slice onto the frozen
AHBA PLS basis (LV1-3, same basis :mod:`eval_pls_gradients` freezes), pull the scores
at one parcel, group subjects by GTEx decade-midpoint age, and plot mean +/- SEM per LV
across age.

Works on any prediction stage:
- ``loro_truth`` — held-out harmonized GTEx ground truth (the 12 GTEx parcels), and
- ``loro_fused`` — dense reconstruction, available at every S156 parcel including the
  **extrapolated** ones that GTEx never sampled.

Parcels are addressed by S156 ``parcel_idx`` (int) or native npz parcel name (str), both
resolved through ``prepost['target_meta']``.

Pipeline (notebook is config-first; project once, render many regions cheaply)
-------------------------------------------------------------------------------
1. ``project_all_subjects(basis, view)`` -> ``(n_subjects, n_regions, n_comp)``.
2. ``subject_ages(prepost, view.subjects)`` -> per-subject decade-midpoint age.
3. ``render_region_progression(T_all, ages, regions, target_meta, region, ...)`` —
   single axes, one line+SEM band per LV, colored by the per-LV auto palette.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..latent.pls import project_pls_scores
from .eval_pls_gradients import _COMPONENT_CMAPS

__all__ = [
    "project_all_subjects", "subject_ages", "resolve_region_position",
    "region_progression", "render_region_progression", "render_progression_grid",
]


# ----------------------------------------------------------------- projection
def project_all_subjects(basis, view) -> np.ndarray:
    """Project every subject's region x gene slice onto the frozen AHBA basis.

    ``view.values`` is ``(subjects, regions, genes)``; returns LV scores
    ``(subjects, regions, n_comp)`` on the basis' sign-canonicalized axes. Rows with any
    non-finite gene (e.g. unobserved parcels in the sparse ``loro_truth`` stage) are left
    NaN — sklearn's PLS ``transform`` rejects NaN, so only finite rows are projected — and
    callers drop them per age bin.
    """
    vals = np.asarray(view.values, dtype=np.float64)
    n_s, n_r, n_g = vals.shape
    flat = vals.reshape(n_s * n_r, n_g)
    finite = np.isfinite(flat).all(axis=1)
    ncomp = int(basis["n_comp"])
    out = np.full((n_s * n_r, ncomp), np.nan)
    if finite.any():
        out[finite] = project_pls_scores(basis, flat[finite])
    return out.reshape(n_s, n_r, ncomp)


# --------------------------------------------------------------------- ages
def subject_ages(prepost, subjects) -> np.ndarray:
    """Per-subject GTEx age (numeric decade-midpoint, e.g. 60-69 -> 64).

    Aligned to ``subjects`` (the view's subject order). Pulled from
    ``prepost['gtex_raw']`` (cols ``subject``/``age``); NaN where the subject is absent.
    """
    g = prepost["gtex_raw"]
    age_by_sid = (g.dropna(subset=["age"])
                   .groupby(g["subject"].astype(str))["age"]
                   .first())
    out = np.array([float(age_by_sid.get(str(s), np.nan)) for s in subjects], dtype=float)
    return out


# ----------------------------------------------------------------- addressing
def resolve_region_position(target_meta, regions, region) -> tuple[int, str, int]:
    """Resolve a parcel to ``(position_in_regions, region_name, parcel_idx)``.

    ``region`` is an S156 ``parcel_idx`` (int / digit-str) or a native parcel name
    (``tissue_or_parcel``, e.g. ``LH_Default_2`` / ``Brain - Frontal Cortex (BA9)``).
    ``regions`` is the view/basis region frame (``tissue_or_parcel`` in parcel_idx order),
    so the returned position indexes ``T_all``'s region axis.
    """
    tm = target_meta
    s = str(region)
    if isinstance(region, (int, np.integer)) or s.isdigit():
        pidx = int(region)
        m = tm[tm["parcel_idx"] == pidx]
        if len(m) == 0:
            raise ValueError(f"no parcel_idx == {pidx} in target_meta")
        name = str(m.iloc[0]["tissue_or_parcel"])
    else:
        m = tm[tm["tissue_or_parcel"].astype(str) == s]
        if len(m) == 0:
            raise ValueError(f"unknown parcel name: {region!r}")
        name = s
        pidx = int(m.iloc[0]["parcel_idx"])
    regions = list(regions)
    if name not in regions:
        raise ValueError(f"parcel {name!r} (idx {pidx}) not in the view's region frame")
    return regions.index(name), name, pidx


# ----------------------------------------------------------------- binning
def region_progression(T_all, ages, region_pos, *, n_lv=3):
    """Mean +/- SEM of each LV's parcel score across age bins (decade midpoints).

    ``T_all`` is ``(subjects, regions, n_comp)``; ``ages`` aligns to the subject axis.
    Returns a dict: ``age`` (unique sorted bin centers), ``mean``/``sem`` ``(n_age, n_lv)``,
    ``n`` ``(n_age, n_lv)`` finite-subject counts, ``points`` (list per age of the raw
    ``(n_subj_in_bin, n_lv)`` scores for optional scatter), ``n_lv``.
    """
    scores = np.asarray(T_all)[:, int(region_pos), :]   # (subjects, n_comp)
    ncomp = int(min(n_lv, scores.shape[1]))
    ages = np.asarray(ages, dtype=float)
    have_age = np.isfinite(ages)
    bins = np.unique(ages[have_age])

    mean = np.full((len(bins), ncomp), np.nan)
    sem = np.full((len(bins), ncomp), np.nan)
    n = np.zeros((len(bins), ncomp), dtype=int)
    points = []
    for bi, age in enumerate(bins):
        sel = have_age & (ages == age)
        block = scores[sel, :ncomp]
        points.append(block)
        for lv in range(ncomp):
            col = block[:, lv]
            col = col[np.isfinite(col)]
            n[bi, lv] = col.size
            if col.size:
                mean[bi, lv] = col.mean()
            if col.size > 1:
                sem[bi, lv] = col.std(ddof=1) / np.sqrt(col.size)
    return {"age": bins, "mean": mean, "sem": sem, "n": n, "points": points, "n_lv": ncomp}


def _lv_color(lv):
    """Representative line color for LV ``lv`` sampled from its auto palette."""
    name = _COMPONENT_CMAPS[lv % len(_COMPONENT_CMAPS)]
    return plt.get_cmap(name)(0.62)


# ----------------------------------------------------------------- plotting
def render_region_progression(T_all, ages, regions, target_meta, region, *,
                              stage_label="", n_lv=3, show_scatter=False,
                              font_scale=1.4, export_dir=None, export_tag=None,
                              ax=None, figsize=(6.4, 5.0), title=None, legend=True):
    """LV-score-vs-age progression for one parcel (one axes, one line+SEM band per LV).

    Each LV is a line with a +/-1 SEM shaded band, colored by the per-LV auto palette
    (LV1 cividis / LV2 plasma / LV3 viridis). ``show_scatter=True`` overlays individual
    subjects behind the lines. ``stage_label`` names the data source (e.g. 'ground truth'
    / 'reconstructed'). Pass ``ax`` to draw into an existing axes (no save/show);
    ``legend=False`` suppresses the per-axes legend (use a shared figure legend instead).
    """
    pos, name, pidx = resolve_region_position(target_meta, regions, region)
    prog = region_progression(T_all, ages, pos, n_lv=n_lv)
    age, mean, sem, ncount, points, ncomp = (
        prog["age"], prog["mean"], prog["sem"], prog["n"], prog["points"], prog["n_lv"])

    fs_title = 14.0 * font_scale
    fs_label = 12.0 * font_scale
    fs_tick = 10.0 * font_scale
    fs_leg = 10.0 * font_scale

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for lv in range(ncomp):
        c = _lv_color(lv)
        if show_scatter:
            for bi, block in enumerate(points):
                col = block[:, lv]
                col = col[np.isfinite(col)]
                if col.size:
                    ax.scatter(np.full(col.size, age[bi]), col, s=14, color=c,
                               alpha=0.30, edgecolor="none", zorder=1)
        m, e = mean[:, lv], sem[:, lv]
        ok = np.isfinite(m)
        ax.fill_between(age[ok], (m - e)[ok], (m + e)[ok], color=c, alpha=0.20,
                        linewidth=0, zorder=2)
        ax.errorbar(age[ok], m[ok], yerr=e[ok], color=c, lw=2.0, marker="o", ms=5,
                    capsize=3, elinewidth=1.2, label=f"LV{lv + 1}", zorder=3)

    ax.set_xlabel("Age (decade midpoint)", fontsize=fs_label)
    ax.set_ylabel("Molecular LV score", fontsize=fs_label)
    if title is None:
        n_str = "/".join(str(int(c)) for c in ncount[:, 0]) if len(ncount) else ""
        suffix = f" — {stage_label}" if stage_label else ""
        title = f"{name} (idx {pidx}){suffix}\nn = {n_str}"
    ax.set_title(title, fontsize=fs_title, fontweight="bold", linespacing=1.3)
    ax.tick_params(labelsize=fs_tick)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend:
        ax.legend(fontsize=fs_leg, frameon=False, loc="best")

    if owns_fig and export_dir:
        out = Path(export_dir); out.mkdir(parents=True, exist_ok=True)
        tag = export_tag or f"{name}".replace(" ", "_").replace("/", "-")
        fp = out / f"pls_progression__{tag}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight"); print("saved", fp)
    if owns_fig:
        plt.show()
        plt.close(fig)
    return prog


def render_progression_grid(T_all, ages, regions, target_meta, region_list, *,
                            stage_label="", n_rows=3, n_cols=4, n_lv=3, show_scatter=False,
                            font_scale=1.0, export_dir=None, export_tag=None, figsize_per=(4.0, 3.4)):
    """Grid of per-parcel LV-score-vs-age progressions (one cell per parcel).

    Renders ``region_list`` into an ``n_rows`` x ``n_cols`` grid, each cell a
    :func:`render_region_progression` (compact title = parcel name only; per-cell legends
    suppressed in favor of one shared figure legend). ``stage_label`` titles the whole
    figure (e.g. 'ground truth' / 'reconstructed'). Use for the 12 GTEx-matched parcels.
    """
    region_list = list(region_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows))
    axflat = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axflat):
        if i >= len(region_list):
            ax.axis("off")
            continue
        pos, name, pidx = resolve_region_position(target_meta, regions, region_list[i])
        render_region_progression(T_all, ages, regions, target_meta, region_list[i],
                                  n_lv=n_lv, show_scatter=show_scatter, font_scale=font_scale,
                                  ax=ax, title=f"{name}\n(idx {pidx})", legend=False)
        # de-clutter inner cells: only edge axes keep their labels
        if i % n_cols != 0:
            ax.set_ylabel("")
        if i < len(region_list) - n_cols:
            ax.set_xlabel("")

    handles, labels = axflat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=11 * font_scale + 4,
               ncol=len(labels))
    if stage_label:
        fig.suptitle(f"LV gradient progression — {stage_label}", fontsize=16 * font_scale + 4,
                     fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if export_dir:
        out = Path(export_dir); out.mkdir(parents=True, exist_ok=True)
        tag = export_tag or (stage_label.replace(" ", "_") or "grid")
        fp = out / f"pls_progression_grid__{tag}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight"); print("saved", fp)
    plt.show()
    plt.close(fig)
