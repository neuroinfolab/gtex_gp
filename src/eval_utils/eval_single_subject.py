"""Single-subject evaluation surface.

Houses subject-keyed analyses split out of `eval_population.py`:
  * Per-subject LORO performance summary + ranked plot with percentile
    sparsification (`compute_subject_performance`,
    `select_subjects_by_percentile`, `plot_subject_performance_ranked`).
  * Subject specificity — self vs other-subject truth at the same region
    (`compute_subject_specificity`, `plot_subject_specificity`).
  * Spatial specificity — self-region vs other-regions within the same
    subject's predictions (`compute_spatial_specificity`,
    `plot_spatial_specificity`).

All plots route through `eval_style.py` primitives (`MODEL_COLORS`, `FONT`,
`model_label`, `ordered_models`, `format_legend_label`).
"""
from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .eval_style import (
    FONT,
    MODEL_COLORS,
    apply_tick_style,
    font_size,
    format_legend_label,
    model_label,
    ordered_models,
)
from .eval_style import _resolve_fonts


__all__ = [
    "compute_subject_performance",
    "select_subjects_by_percentile",
    "plot_subject_performance_ranked",
    "compute_subject_specificity",
    "plot_subject_specificity",
    "compute_spatial_specificity",
    "plot_spatial_specificity",
]


# ---------------------------------------------------------------------------
# Shared private helpers (vectorized similarity primitives)
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {"pearson_r": True, "spearman_r": True, "r2": True, "rmse": False}


def _normalize_specificity_metric(metric: str) -> str:
    m = str(metric).lower()
    if m in {"pearson", "pearson_r"}:
        return "pearson_r"
    if m in {"spearman", "spearman_r"}:
        return "spearman_r"
    if m in {"r2", "rmse"}:
        return m
    raise ValueError(
        f"Unsupported metric for subject specificity: {metric!r} "
        f"(use pearson_r, spearman_r, r2, or rmse)"
    )


def _row_z_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True, ddof=0)
    sd_safe = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    Z = (X - mu) / sd_safe
    Z = np.where(np.isfinite(Z), Z, 0.0)
    Z = np.where(np.broadcast_to(sd > 0, Z.shape), Z, 0.0)
    return Z


def _row_rank(X: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.asarray(X, dtype=np.float64)).rank(axis=1, method="average").to_numpy(dtype=np.float64)


def _hex_lighten(color, frac: float) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color), dtype=np.float64)
    return tuple((rgb + (1.0 - rgb) * float(frac)).tolist())


def _pairwise_metric_matrix(
    P: np.ndarray,
    Y: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Vectorized (n_p × n_y) similarity matrix.

    pearson_r / spearman_r:  (Pz @ Yz.T) / G  on (rank-)z-normalized rows
    r2:                      1 − ‖P_i − Y_j‖² / SS_tot(Y_j)  (Y row-centered)
    rmse:                    sqrt(‖P_i − Y_j‖² / G)
    """
    metric = _normalize_specificity_metric(metric)
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    G = int(P.shape[1])
    if metric in {"pearson_r", "spearman_r"}:
        if metric == "spearman_r":
            P = _row_rank(P)
            Y = _row_rank(Y)
        Pz = _row_z_normalize(P)
        Yz = _row_z_normalize(Y)
        return (Pz @ Yz.T) / float(G)

    P_norm2 = (P * P).sum(axis=1, keepdims=True)
    Y_norm2 = (Y * Y).sum(axis=1, keepdims=True)
    PY = P @ Y.T
    SS_res = P_norm2 + Y_norm2.T - 2.0 * PY
    SS_res = np.maximum(SS_res, 0.0)
    if metric == "rmse":
        return np.sqrt(SS_res / float(G))
    Y_mean = Y.mean(axis=1, keepdims=True)
    SS_tot_row = ((Y - Y_mean) ** 2).sum(axis=1)
    SS_tot_safe = np.where(SS_tot_row > 0, SS_tot_row, np.nan)
    return 1.0 - SS_res / SS_tot_safe[None, :]


# ---------------------------------------------------------------------------
# Subject performance summary + percentile selection
# ---------------------------------------------------------------------------


def compute_subject_performance(
    fold_perf_df: pd.DataFrame,
    metric: str = "pearson_r",
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-(model, subject) summary of fold-level metric.

    Returns columns: `model, subject, mean, std, n_folds, percentile_rank`.
    `percentile_rank` ∈ [0, 100] is **performance-oriented**: 0 = worst,
    100 = best, regardless of metric direction. For higher-is-better
    metrics (Pearson r, Spearman ρ, R²) this matches the raw mean order;
    for RMSE it is inverted so a low-RMSE subject still maps to a high
    percentile.

    `fold_perf_df` is the output of `compute_fold_combo_metrics_from_cache`
    or any per-`(subject, fold_key)` table carrying `subject`, `model`, and
    the chosen `metric` column.
    """
    needed = {"subject", "model", metric}
    missing = needed - set(fold_perf_df.columns)
    if missing:
        raise KeyError(f"fold_perf_df missing columns: {sorted(missing)}")
    cols = list(needed)
    if "fold_key" in fold_perf_df.columns:
        cols.append("fold_key")
    d = fold_perf_df[cols].copy()
    d["model"] = d["model"].astype(str).str.lower()
    if models is not None:
        keep = {str(m).lower() for m in models}
        d = d[d["model"].isin(keep)]
    summary = (
        d.groupby(["model", "subject"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), n_folds=(metric, "size"))
    )
    higher_better = bool(_HIGHER_IS_BETTER.get(_normalize_specificity_metric(metric), True))
    summary["percentile_rank"] = (
        summary.groupby("model")["mean"]
        .rank(method="average", pct=True, ascending=higher_better) * 100.0
    )
    return summary


def select_subjects_by_percentile(
    perf_df: pd.DataFrame,
    *,
    metric: str = "pearson_r",
    model: str = "plam",
    n_bands: int | None = None,
    percentiles: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Pick one subject per requested **performance** percentile of `(metric, model)`.

    Performance-oriented: 0 = worst-performing subject, 100 = best.
    For RMSE the underlying `mean` is inverted in the percentile mapping
    (low RMSE → high percentile) so the semantics match Pearson r / R².

    Two modes:
      - `n_bands=N`: pick subjects at the N evenly-spaced percentile anchors
        (e.g. N=10 → 5,15,...,95; N=100 → effectively every subject).
      - `percentiles=(0.10, 0.50, 0.90)`: pick subjects nearest to each
        explicit performance quantile.

    Returns the full per-model row plus `percentile_anchor` (0–100),
    `target_value`, `selection_index`. Anchor collisions on the same
    subject are deduped, keeping the lower anchor.
    """
    if "mean" not in perf_df.columns:
        raise KeyError("perf_df must include 'mean' (output of compute_subject_performance)")
    if (n_bands is None) == (percentiles is None):
        raise ValueError("Pass exactly one of n_bands or percentiles")

    d = perf_df[perf_df["model"].astype(str).str.lower() == str(model).lower()].copy()
    d = d.dropna(subset=["mean"])
    if len(d) == 0:
        raise RuntimeError(f"No subjects with non-null mean for model={model!r}")
    higher_better = bool(_HIGHER_IS_BETTER.get(_normalize_specificity_metric(metric), True))
    # Sort so worst-performing subjects come first regardless of metric direction.
    d = d.sort_values(["mean", "subject"], ascending=[higher_better, True]).reset_index(drop=True)

    if n_bands is not None:
        n = int(n_bands)
        if n < 1:
            raise ValueError("n_bands must be ≥ 1")
        if n == 1:
            anchors = np.array([0.5])
        else:
            anchors = (np.arange(n) + 0.5) / float(n)
    else:
        anchors = np.asarray([float(q) for q in percentiles], dtype=np.float64)
        if not np.all((anchors >= 0.0) & (anchors <= 1.0)):
            raise ValueError("percentiles must lie in [0, 1]")

    # `anchors` are in performance-percentile space; map to raw-mean quantiles
    # by flipping for lower-is-better metrics so the lookup stays a simple
    # `mean.quantile(q)` call.
    raw_quantiles = anchors if higher_better else (1.0 - anchors)

    rows = []
    seen_subjects: set = set()
    for perf_q, raw_q in zip(anchors, raw_quantiles):
        target = float(d["mean"].quantile(float(raw_q)))
        i = (d["mean"] - target).abs().idxmin()
        subj = d.loc[i, "subject"]
        if subj in seen_subjects:
            continue
        seen_subjects.add(subj)
        row = d.loc[i].copy()
        row["percentile_anchor"] = float(perf_q) * 100.0
        row["target_value"] = target
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["selection_index"] = np.arange(len(out))
    preferred = [
        "selection_index", "percentile_anchor", "subject", "model",
        "mean", "std", "n_folds", "percentile_rank", "target_value",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    return out[cols]


_SUBJECT_RANKED_FONTS = {
    "title":  "xxl",
    "xlabel": "l+1",
    "ylabel": "l+2",
    "tick":   "s",       # auto-tuned below by n_subj; caller can override
    "legend": "m+2",
    "legend_title": "m+3",
}


def plot_subject_performance_ranked(
    perf_df: pd.DataFrame,
    metric: str = "pearson_r",
    models: Sequence[str] | None = None,
    sparsify: int | None = 100,
    anchor_model: str = "plam",
    figsize: Tuple[float, float] | None = None,
    dpi: int = 180,
    bottom_quantile: float | None = 0.20,
    top_quantile: float | None = None,
    dynamic_bottom_quantile: bool = True,
    naive_baseline_model: str = "naive",
    font_sizes: Mapping[str, str | int] | None = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Ranked per-subject mean metric across LORO folds, per model.

    Subjects are ordered along the x axis by **performance-ascending**
    anchor-model mean (worst first, best last) regardless of metric
    direction — for RMSE, "worst" is highest-RMSE. The same subject
    occupies the same x position under every model.

    Default `sparsify=100` keeps one subject per percentile of the anchor
    model. Pass `sparsify=None` to render every subject; pass any int N to
    show N evenly-spaced performance percentiles. The plot scales
    horizontally with `n_subj`.

    Shading:
      - `bottom_quantile=0.20` (default) shades the leftmost 20% of subjects.
      - `top_quantile=None` optionally shades the rightmost N%.
      - `dynamic_bottom_quantile=True` (default) **overrides** the static
        `bottom_quantile` with a data-driven cutoff: the smallest performance
        percentile at which the anchor model first beats the
        `naive_baseline_model` ('naive' by default). Beats = strictly better
        in metric direction. Falls back to the static `bottom_quantile` when
        the baseline never gets crossed (or is missing from `perf_df`).

    Each x position carries the subject ID below the axis (rotated) and the
    anchor-model performance percentile above the axis line (horizontal).
    """
    if "mean" not in perf_df.columns:
        raise KeyError("perf_df must include 'mean' (output of compute_subject_performance)")
    summary = perf_df.copy()
    summary["model"] = summary["model"].astype(str).str.lower()
    if models is not None:
        keep = {str(m).lower() for m in models}
        summary = summary[summary["model"].isin(keep)]
    if summary.empty:
        raise RuntimeError("perf_df is empty after model filter")

    anchor = str(anchor_model).lower()
    if anchor not in set(summary["model"].unique()):
        raise ValueError(f"anchor_model={anchor!r} not present in perf_df")

    higher_better = bool(_HIGHER_IS_BETTER.get(_normalize_specificity_metric(metric), True))

    if sparsify is not None and int(sparsify) > 0:
        anchor_subjects = select_subjects_by_percentile(
            summary, metric=metric, model=anchor, n_bands=int(sparsify)
        )
        keep_subjects = list(anchor_subjects["subject"].astype(str))
        anchor_pcts = anchor_subjects.set_index("subject")["percentile_anchor"].astype(float)
    else:
        anchor_d = summary[summary["model"] == anchor]
        # Sort so worst-performing subjects come first (left) regardless of metric direction.
        anchor_d = (
            anchor_d.dropna(subset=["mean"])
            .sort_values(["mean", "subject"], ascending=[higher_better, True])
        )
        keep_subjects = list(anchor_d["subject"].astype(str))
        anchor_pcts = (
            anchor_d.set_index("subject")["percentile_rank"].astype(float)
            if "percentile_rank" in anchor_d.columns
            else pd.Series(dtype=float)
        )

    summary = summary[summary["subject"].astype(str).isin(keep_subjects)].copy()
    pos = {s: i for i, s in enumerate(keep_subjects)}
    summary["_x"] = summary["subject"].astype(str).map(pos)
    summary = summary.sort_values(["model", "_x"]).reset_index(drop=True)

    plot_models = ordered_models(summary["model"].unique())

    n_subj = len(keep_subjects)
    if figsize is None:
        # Wide layout that scales with n_subj. Capped so very large counts
        # don't blow out the rendered page.
        figsize = (min(28.0, max(10.0, 0.16 * n_subj + 4.0)), 5.4)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))

    # Resolve the bottom-shading quantile. When `dynamic_bottom_quantile=True`,
    # walk the subjects in performance-ascending order and find the first
    # one where the anchor model strictly beats the `naive_baseline_model`
    # (direction-aware). That index / n_subj becomes the cutoff. Falls back
    # to the static `bottom_quantile` when no such crossover exists.
    bottom_quantile_used = bottom_quantile
    bottom_quantile_source = "static"
    if bool(dynamic_bottom_quantile):
        baseline = str(naive_baseline_model).lower()
        if baseline != anchor:
            anchor_lookup = (
                summary[summary["model"] == anchor]
                .set_index("subject")["mean"]
            )
            base_lookup = (
                summary[summary["model"] == baseline]
                .set_index("subject")["mean"]
                if baseline in summary["model"].unique()
                else pd.Series(dtype=float)
            )
            if not base_lookup.empty:
                cross_idx = None
                for i, s in enumerate(keep_subjects):
                    a = float(anchor_lookup.get(s, np.nan))
                    b = float(base_lookup.get(s, np.nan))
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    crosses = (a > b) if higher_better else (a < b)
                    if crosses:
                        cross_idx = i
                        break
                if cross_idx is not None:
                    bottom_quantile_used = cross_idx / float(n_subj)
                    bottom_quantile_source = "dynamic"

    # Percentile-cutoff shading: highlights the bottom (and optional top)
    # tails of the anchor distribution. Drawn first (zorder=0) so points
    # render on top.
    if bottom_quantile_used is not None and 0.0 < float(bottom_quantile_used) < 1.0:
        bottom_n = max(1, int(round(float(bottom_quantile_used) * n_subj)))
        ax.axvspan(-0.5, bottom_n - 0.5, color="#cc0000", alpha=0.07, zorder=0)
    if top_quantile is not None and 0.0 < float(top_quantile) < 1.0:
        top_n = max(1, int(round(float(top_quantile) * n_subj)))
        ax.axvspan(n_subj - top_n - 0.5, n_subj - 0.5, color="#1b7837", alpha=0.07, zorder=0)

    # Tick density / size scales with n_subj. The default token is for ~100
    # subjects (one per percentile); fewer subjects bump up, denser bumps
    # down — caller can override via font_sizes['tick'].
    if n_subj <= 20:
        tick_default = "s+1"
    elif n_subj <= 60:
        tick_default = "s"
    elif n_subj <= 150:
        tick_default = "xs"
    else:
        tick_default = "xs-1"
    if font_sizes is None or "tick" not in font_sizes:
        local_override = dict(font_sizes or {})
        local_override["tick"] = tick_default
        fonts = _resolve_fonts(_SUBJECT_RANKED_FONTS, local_override)
    else:
        fonts = _resolve_fonts(_SUBJECT_RANKED_FONTS, font_sizes)

    for m in plot_models:
        dm = summary[summary["model"] == m]
        if dm.empty:
            continue
        c = MODEL_COLORS.get(m, "#777777")
        is_anchor = (m == anchor)
        # Markers only — no connecting lines, no fills.
        ax.errorbar(
            dm["_x"], dm["mean"], yerr=dm["std"].fillna(0.0),
            fmt="o",
            color=c, ecolor=c,
            markersize=4.6 if is_anchor else 4.0,
            linewidth=0.0,
            elinewidth=0.8, capsize=2.0,
            label=model_label(m) + (" (anchor)" if is_anchor else ""),
            alpha=0.92 if is_anchor else 0.78,
            zorder=3 if is_anchor else 2,
        )

    # X-ticks below the axis: subject IDs, rotated. Percentile labels go
    # above the axis line as a horizontal in-axes annotation per tick.
    x_idx = np.arange(n_subj)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(keep_subjects, rotation=80, ha="right", fontsize=fonts["tick"])

    pct_fontsize = max(5, fonts["tick"] - 1)
    for x_i, subj in zip(x_idx, keep_subjects):
        pct = float(anchor_pcts.get(subj, np.nan))
        if not np.isfinite(pct):
            continue
        ax.text(
            float(x_i), 0.012, f"{pct:.0f}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=pct_fontsize, color="#444444",
        )

    direction_word = "ascending" if higher_better else "descending"
    ax.set_xlabel(
        f"Subject (worst → best by {model_label(anchor)} {direction_word} {format_legend_label(metric)})",
        fontsize=fonts["xlabel"],
    )
    ax.set_ylabel(format_legend_label(metric), fontsize=fonts["ylabel"])
    ax.set_title(
        f"Per-Subject Performance Percentile (Mean {format_legend_label(metric)} over LORO Folds)",
        fontsize=fonts["title"],
    )
    ax.grid(True, axis="y", alpha=0.12, linewidth=0.6, linestyle=":")
    ax.set_axisbelow(True)

    # Legend: model markers + (when enabled) labeled patches for the
    # bottom/top shaded percentile regions so the shading is self-documenting.
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    if bottom_quantile_used is not None and 0.0 < float(bottom_quantile_used) < 1.0:
        pct = int(round(float(bottom_quantile_used) * 100))
        if bottom_quantile_source == "dynamic":
            shade_label = (
                f"Bottom {pct}% — {model_label(anchor)} ≤ "
                f"{model_label(naive_baseline_model)}"
            )
        else:
            shade_label = f"Bottom {pct}% (`bottom_quantile`)"
        handles.append(Patch(facecolor="#cc0000", edgecolor="none", alpha=0.07))
        labels.append(shade_label)
    if top_quantile is not None and 0.0 < float(top_quantile) < 1.0:
        handles.append(Patch(facecolor="#1b7837", edgecolor="none", alpha=0.07))
        labels.append(f"Top {int(round(float(top_quantile) * 100))}% (`top_quantile`)")
    ax.legend(
        handles, labels, title="Model",
        frameon=True, fancybox=False,
        fontsize=fonts["legend"], title_fontsize=fonts["legend_title"],
    )
    apply_tick_style(ax, label_fontsize=fonts["tick"])
    fig.tight_layout()
    return fig, ax, summary


# Subject specificity (self vs other-subject truth at the same region)
# ---------------------------------------------------------------------------


def compute_subject_specificity(
    view: Mapping[str, object],
    region_col: str = "parcel_idx",
    metric: str = "pearson_r",
) -> pd.DataFrame:
    """Per-(model, subject, region) self vs other-subject prediction similarity.

    For each LORO sample (subject `s`, region `p`, model `m`):
      - `sim_self`        = metric(pred[s,p,m], truth[s,p]) over genes
      - `sim_other_mean`  = mean metric(pred[s,p,m], truth[s',p]) over s' ≠ s

    Supported metrics: `pearson_r` (default), `spearman_r`, `r2`, `rmse`.
    For RMSE lower is better — `delta = sim_self - sim_other_mean` will
    typically be negative for a specificity-preserving model. The metric
    name is stored in the returned `metric` column.
    """
    metric_norm = _normalize_specificity_metric(metric)
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    genes = list(view["genes"])
    models = ordered_models(view["models"])
    if region_col not in truth_df.columns:
        raise KeyError(f"region_col={region_col!r} not in truth_df columns")
    if "subject_region_key" not in truth_df.columns:
        raise KeyError("truth_df must include subject_region_key")

    G_total = int(len(genes))
    if G_total == 0:
        raise RuntimeError("view has no genes")

    region_indices = truth_df.groupby(region_col, sort=False).indices
    truth_per_region: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for region, idx in region_indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if len(idx_arr) < 2:
            continue
        Y = truth_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)
        truth_per_region[region] = (idx_arr, Y)

    if not truth_per_region:
        raise RuntimeError(f"No region in {region_col!r} has ≥ 2 subjects")

    meta_cols = [
        c for c in ("subject", "subject_region_key", region_col, "gtex_region", "region_group")
        if c in truth_df.columns
    ]

    rows: list[pd.DataFrame] = []
    for model in models:
        pred_df = pred_dfs[model]
        if len(pred_df) != len(truth_df):
            raise ValueError(f"pred_df[{model}] not aligned to truth_df")
        for region, (idx_arr, Y) in truth_per_region.items():
            P = pred_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)
            M = _pairwise_metric_matrix(P, Y, metric_norm)
            self_val = np.diag(M).astype(np.float64).copy()
            M_off = M.copy()
            np.fill_diagonal(M_off, np.nan)
            other_mean = np.nanmean(M_off, axis=1)

            meta = truth_df.iloc[idx_arr][meta_cols].copy().reset_index(drop=True)
            meta["model"] = str(model).lower()
            meta["sim_self"] = self_val
            meta["sim_other_mean"] = other_mean
            meta["delta"] = self_val - other_mean
            meta["metric"] = metric_norm
            rows.append(meta)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Spatial specificity (self-region vs other-regions within same subject)
# ---------------------------------------------------------------------------


def compute_spatial_specificity(
    view: Mapping[str, object],
    metric: str = "pearson_r",
    region_col: str = "parcel_idx",
) -> pd.DataFrame:
    """Per-(model, subject, region) self-region vs other-regions-in-subject similarity.

    For each LORO sample (subject `s`, region `r`, model `m`) with `k`
    covered regions in subject `s`:
      - `sim_self`              = metric(pred[s,r,m], truth[s,r])
      - `sim_other_region_mean` = mean_{r' ≠ r} metric(pred[s,r,m], truth[s,r'])

    The null is the **mean of `k − 1` per-region metrics**, not a metric on
    a pre-averaged truth vector — i.e. for a subject with 10 covered
    regions you get 1 self-metric and 9 other-region metrics per LORO
    sample, then average those 9 to get `sim_other_region_mean`. A model
    that copies a single subject-mean profile to every region will hit
    `sim_other_region_mean` ≈ `sim_self`. A model that genuinely localizes
    will have `sim_self` clear `sim_other_region_mean` by a margin.

    Subjects with fewer than 2 covered regions are skipped (no within-subject
    null possible).
    """
    metric_norm = _normalize_specificity_metric(metric)
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    genes = list(view["genes"])
    models = ordered_models(view["models"])
    if "subject" not in truth_df.columns:
        raise KeyError("truth_df must include 'subject'")
    if region_col not in truth_df.columns:
        raise KeyError(f"region_col={region_col!r} not in truth_df columns")
    if "subject_region_key" not in truth_df.columns:
        raise KeyError("truth_df must include subject_region_key")
    G_total = int(len(genes))
    if G_total == 0:
        raise RuntimeError("view has no genes")

    subject_indices = truth_df.groupby("subject", sort=False).indices
    eligible: dict[object, np.ndarray] = {}
    for subj, idx in subject_indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if len(idx_arr) < 2:
            continue
        eligible[subj] = idx_arr
    if not eligible:
        raise RuntimeError("No subject has ≥ 2 regions; spatial specificity undefined")

    meta_cols = [
        c for c in ("subject", "subject_region_key", region_col, "gtex_region", "region_group")
        if c in truth_df.columns
    ]

    rows: list[pd.DataFrame] = []
    for model in models:
        pred_df = pred_dfs[model]
        if len(pred_df) != len(truth_df):
            raise ValueError(f"pred_df[{model}] not aligned to truth_df")
        for subj, idx_arr in eligible.items():
            P = pred_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)
            Y = truth_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)

            # Pairwise metric matrix M[i, j] = metric(pred[s, r_i], truth[s, r_j]).
            # Diagonal = self (pred[r] vs truth[r]).
            # Row r off-diagonal entries = pred[r] vs every other-region
            # truth in the same subject; their mean = sim_other_region_mean.
            M = _pairwise_metric_matrix(P, Y, metric_norm)
            self_val = np.diag(M).astype(np.float64).copy()
            M_off = M.copy()
            np.fill_diagonal(M_off, np.nan)
            other_val = np.nanmean(M_off, axis=1)

            meta = truth_df.iloc[idx_arr][meta_cols].copy().reset_index(drop=True)
            meta["model"] = str(model).lower()
            meta["sim_self"] = self_val
            meta["sim_other_region_mean"] = other_val
            meta["delta"] = self_val - other_val
            meta["metric"] = metric_norm
            rows.append(meta)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Split-violin plotting (shared between subject + spatial specificity)
# ---------------------------------------------------------------------------

_SPECIFICITY_FONTS = {
    "title": "xl",
    "xlabel": "l",
    "ylabel": "l",
    "tick": "m+1",
    "legend": "m",
    "legend_title": "m",
}


def _plot_specificity_split_violin(
    spec_df: pd.DataFrame,
    *,
    self_col: str,
    other_col: str,
    other_label: str,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    annotate_paired_test: bool,
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if spec_df is None or len(spec_df) == 0:
        raise RuntimeError("spec_df is empty")

    metric_norm = (
        _normalize_specificity_metric(str(spec_df["metric"].iloc[0]))
        if "metric" in spec_df.columns and pd.notna(spec_df["metric"].iloc[0])
        else "pearson_r"
    )
    higher_is_better = bool(_HIGHER_IS_BETTER.get(metric_norm, True))
    metric_label = format_legend_label(metric_norm)
    fonts = _resolve_fonts(_SPECIFICITY_FONTS, font_sizes)

    long = spec_df.melt(
        id_vars=["model"],
        value_vars=[self_col, other_col],
        var_name="condition",
        value_name="value",
    )
    long["model"] = long["model"].astype(str).str.lower()
    long["condition"] = long["condition"].map(
        {self_col: "self", other_col: "other"}
    )
    long = long.dropna(subset=["value"]).copy()

    plot_models = ordered_models(long["model"].unique())
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    sns.violinplot(
        data=long, x="model", y="value",
        hue="condition", hue_order=["self", "other"],
        split=True, order=plot_models,
        palette={"self": "#888888", "other": "#cccccc"},
        inner="quartile", cut=0, ax=ax, linewidth=0.9,
    )

    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    poly_idx = 0
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection):
            continue
        if poly_idx >= 2 * len(plot_models):
            break
        m = plot_models[poly_idx // 2]
        cond = ["self", "other"][poly_idx % 2]
        base = MODEL_COLORS.get(m, "#777777")
        face = base if cond == "self" else _hex_lighten(base, 0.55)
        coll.set_facecolor(face)
        coll.set_edgecolor(base)
        coll.set_linewidth(0.9)
        poly_idx += 1

    ax.set_xticks(np.arange(len(plot_models)))
    ax.set_xticklabels([model_label(m) for m in plot_models], fontsize=fonts["tick"])
    ax.set_xlabel("Model", fontsize=fonts["xlabel"])
    ax.set_ylabel(metric_label, fontsize=fonts["ylabel"])
    ax.set_title(title, fontsize=fonts["title"])
    ax.grid(True, axis="y", alpha=0.18)
    apply_tick_style(ax, label_fontsize=fonts["tick"])

    # Stats-augmented legend: per-model Self/Other patches followed by a
    # compact Wilcoxon line per model. Single legend at lower-left so the
    # comparison key and inferential summary live together.
    stats_lookup: dict[str, str] = {}
    if bool(annotate_paired_test):
        try:
            from scipy.stats import wilcoxon
        except Exception:
            wilcoxon = None
        if wilcoxon is not None:
            alt = "greater" if higher_is_better else "less"
            for m in plot_models:
                d = spec_df[spec_df["model"].astype(str).str.lower() == m]
                d = d.dropna(subset=[self_col, other_col])
                if len(d) < 5:
                    continue
                self_v = d[self_col].to_numpy(dtype=np.float64)
                other_v = d[other_col].to_numpy(dtype=np.float64)
                med_delta = float(np.median(self_v - other_v))
                try:
                    _w, p = wilcoxon(self_v, other_v, alternative=alt)
                    p_val = float(p)
                except Exception:
                    p_val = float("nan")
                p_str = "n/a" if not np.isfinite(p_val) else f"{p_val:.2g}"
                stats_lookup[m] = f"med Δ={med_delta:+.3f}, p={p_str}"

    handles: list = []
    labels: list = []
    blank = Patch(facecolor="none", edgecolor="none")
    for m in plot_models:
        base = MODEL_COLORS.get(m, "#777777")
        light = _hex_lighten(base, 0.55)
        handles.append(Patch(facecolor=base, edgecolor=base))
        labels.append(f"{model_label(m)} — Self")
        handles.append(Patch(facecolor=light, edgecolor=base))
        labels.append(f"{model_label(m)} — {other_label}")
        if m in stats_lookup:
            handles.append(blank)
            labels.append(f"   {stats_lookup[m]}")
    ax.legend(
        handles, labels, title="Comparison (Wilcoxon Δ = self − other)",
        frameon=True, fancybox=False, loc="lower left",
        fontsize=fonts["legend"], title_fontsize=fonts["legend_title"],
        handlelength=1.4, borderaxespad=0.6, labelspacing=0.35,
    )

    fig.tight_layout()
    return fig, ax


def plot_subject_specificity(
    spec_df: pd.DataFrame,
    figsize: Tuple[float, float] = (9.0, 5.0),
    dpi: int = 180,
    annotate_paired_test: bool = True,
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Split-violin: per region, prediction vs that subject's truth (self) vs
    prediction vs the mean of every other subject's truth at the same region.
    """
    return _plot_specificity_split_violin(
        spec_df,
        self_col="sim_self",
        other_col="sim_other_mean",
        other_label="Other Subjects (mean)",
        title="Subject Specificity: LORO Prediction vs Truth — Prediction vs Mean",
        figsize=figsize, dpi=dpi,
        annotate_paired_test=annotate_paired_test,
        font_sizes=font_sizes,
    )


def plot_spatial_specificity(
    spec_df: pd.DataFrame,
    figsize: Tuple[float, float] = (9.0, 5.0),
    dpi: int = 180,
    annotate_paired_test: bool = True,
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Split-violin: per region, prediction vs that region's truth (self) vs
    the mean of `k − 1` per-region metrics comparing the same prediction to
    every other-region truth within the same subject.
    """
    return _plot_specificity_split_violin(
        spec_df,
        self_col="sim_self",
        other_col="sim_other_region_mean",
        other_label="Other Samples (within subject)",
        title="Spatial Specificity: LORO Prediction vs Truth — Prediction vs Other Samples (within subject)",
        figsize=figsize, dpi=dpi,
        annotate_paired_test=annotate_paired_test,
        font_sizes=font_sizes,
    )
