#!/usr/bin/env python3
from __future__ import annotations

"""
Latent-space evaluation: pooled ground-truth PCA recovery and variance spectra.

This module is the import surface for ``eval_pca.ipynb``. It is the fourth
orthogonal axis of cache-backed model evaluation, alongside the population
(``eval_population.py``), single-subject (``eval_single_subject.py``), and
gene-wise surfaces:

- population  -> per-sample accuracy
- gene-wise   -> per-gene predictability
- subject     -> per-subject performance
- **latent**  -> does the model recover the covariance geometry of the
  ground-truth GTEx manifold?

Unlike the legacy ``pca_cached_predictions.ipynb`` (which re-read the LORO
``.npz`` caches through ``results_eda.collect_pooled_sample_prediction_dfs_*``),
every function here consumes the **canonical wide prediction tables** via a
view built by :func:`eval_population.make_prediction_eval_view`. That keeps a
single cache-reading path across the whole eval surface, and means the latent
analyses are strict-held-out-scoped for free (the canonical tables are built
from ``loro_eval_mask`` rows only — see the eval refactor plan).

Core entry points:
- :func:`compute_pca_recovery` / :func:`plot_pca_recovery`
- :func:`compute_pca_variance_spectra` / :func:`plot_pca_variance_spectra`
- :func:`compute_within_parcel_variance` / :func:`plot_within_parcel_variance`
- :func:`plot_sample_gene_heatmap`

Demeaning:
- ``demean_mode='none'`` -> standard PCA (sklearn centers each gene globally).
- ``demean_mode='within_parcel'`` -> subtract the truth parcel-mean vector
  from truth and every prediction within each parcel first, so the analysis
  isolates subject-specific within-parcel variation (the view where naive
  fill is expected to collapse toward zero while learned models should not).

Recovery vs variance — two different questions:
- :func:`compute_pca_recovery` projects each model's predictions into the
  *truth* PCA basis and asks, per component, whether predicted PC scores
  align with true PC scores across pooled samples.
- :func:`compute_pca_variance_spectra` fits PCA on each source separately and
  reports the intrinsic variance geometry of that source.
- :func:`compute_within_parcel_variance` asks whether a source produces
  subject-to-subject spread within a parcel *at all* — a model can have
  ample within-parcel variance yet fail to align it with the true latent
  axes.
"""

from pathlib import Path
import hashlib
import pickle
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .eval_style import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_tick_style,
    build_parcel_label_table,
    format_metric_label,
    model_label,
    ordered_models,
    parcel_label_lookup,
    set_academic_style,
    strip_display_label_prefixes,
)
from .eval_style import _resolve_fonts
from .eval_population import format_gene_list_name

try:  # optional — only needed for show_fit='lowess'
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
except Exception:  # pragma: no cover
    _sm_lowess = None


__all__ = [
    "compute_pca_recovery",
    "compute_pca_variance_spectra",
    "compute_within_parcel_variance",
    "plot_pca_recovery",
    "plot_pca_variance_spectra",
    "plot_within_parcel_variance",
    "plot_sample_gene_heatmap",
    "set_academic_style",
]


TRUTH_COLOR = "#111111"
TRUTH_LABEL = "Truth"

# score_<x> column in pc_df  ->  canonical metric key understood by
# eval_style.format_metric_label (never hardcode metric strings in plot text).
_SCORE_METRIC_KEYS = {
    "score_pearson": "pearson_r",
    "score_spearman": "spearman_r",
    "score_r2": "r2",
    "score_rmse": "rmse",
}

# Token-based font defaults (see eval_style.font_size). Each plotter merges an
# optional `font_sizes={...}` override over these.
_RECOVERY_FONTS = {
    "title": "xxl", "xlabel": "xl", "ylabel": "xl",
    "tick": "l", "legend": "l", "annotation": "m",
}
_SPECTRA_FONTS = dict(_RECOVERY_FONTS)
_VARIANCE_FONTS = dict(_RECOVERY_FONTS)
_HEATMAP_FONTS = {
    "title": "xxl", "xlabel": "xl", "ylabel": "xl",
    "tick": "m", "cbar_label": "l",
}

_RECOVERY_CACHE_DIR = "notebooks/cache/pca_recovery"
_SPECTRA_CACHE_DIR = "notebooks/cache/pca_spectra"


def _format_demean_mode_label(demean_mode: str) -> str:
    mode = str(demean_mode).strip().lower()
    if mode == "within_parcel":
        return "within-parcel demeaned"
    return mode.replace("_", " ")


# ---------------------------------------------------------------------------
# Safe component-wise metric helpers
# ---------------------------------------------------------------------------
def _finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _r2_safe(x_truth: np.ndarray, y_pred: np.ndarray) -> float:
    # x = true PC score, y = predicted PC score. R^2 of pred against truth.
    x, y = _finite_pair(x_truth, y_pred)
    if x.size < 2:
        return np.nan
    ss_res = float(np.sum((x - y) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _rmse_safe(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((x - y) ** 2)))


# ---------------------------------------------------------------------------
# View -> aligned matrices
# ---------------------------------------------------------------------------
def _view_matrices(
    view: Mapping[str, object],
    models: Sequence[str] | None = None,
) -> Tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract aligned matrices from a `make_prediction_eval_view` result.

    Returns ``(X_true, pred_by_model, parcel_idx, sample_keys, genes, model_list)``.
    Rows are pooled subject-parcel samples (strict held-out, since the
    canonical tables are built from ``loro_eval_mask`` rows), columns are
    ``view['genes']``.
    """
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    genes = list(view["genes"])
    model_list = ordered_models(models if models is not None else view["models"])
    if len(genes) == 0:
        raise ValueError("view has no genes")

    X_true = np.asarray(truth_df[genes].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(X_true)):
        raise ValueError(
            "truth matrix has non-finite values — the canonical prediction "
            "tables are strict-held-out and should be fully observed"
        )
    pred_by_model: dict[str, np.ndarray] = {}
    for model in model_list:
        if model not in pred_dfs:
            raise KeyError(f"view is missing a prediction table for model={model!r}")
        Xp = np.asarray(pred_dfs[model][genes].to_numpy(dtype=np.float64))
        if not np.all(np.isfinite(Xp)):
            raise ValueError(f"prediction matrix for model={model!r} has non-finite values")
        pred_by_model[model] = Xp

    parcel_idx = truth_df["parcel_idx"].to_numpy(dtype=np.int64)
    sample_keys = truth_df["subject_region_key"].astype(str).to_numpy()
    return X_true, pred_by_model, parcel_idx, sample_keys, genes, model_list


def _apply_within_parcel_demean(
    X_true: np.ndarray,
    pred_by_model: Mapping[str, np.ndarray],
    parcel_idx: np.ndarray,
    demean_mode: str,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """Subtract the truth parcel-mean vector within each parcel.

    The truth parcel mean is the fixed effect being projected out; both
    truth and every prediction are demeaned by the *same* truth-derived
    vector so they stay in a common residual space. PCA then applies its
    usual global gene-wise centering on the residual matrix.
    """
    mode = str(demean_mode).strip().lower()
    if mode in {"", "none"}:
        return X_true, dict(pred_by_model)
    if mode != "within_parcel":
        raise ValueError("demean_mode must be one of: none, within_parcel")

    X_true_dm = np.asarray(X_true, dtype=np.float64).copy()
    pred_dm = {m: np.asarray(v, dtype=np.float64).copy() for m, v in pred_by_model.items()}
    for p in np.unique(parcel_idx):
        idx = np.where(parcel_idx == int(p))[0]
        if idx.size == 0:
            continue
        mu_p = np.nanmean(X_true_dm[idx, :], axis=0)
        X_true_dm[idx, :] -= mu_p[None, :]
        for m in pred_dm:
            pred_dm[m][idx, :] -= mu_p[None, :]
    return X_true_dm, pred_dm


def _panel_label_from_view(view: Mapping[str, object]) -> str:
    return format_gene_list_name(view.get("eval_gene_list_path"))


# ---------------------------------------------------------------------------
# Disk caching (mirrors eda_core.prepare_pre_post_harmonization_cached)
# ---------------------------------------------------------------------------
def _pca_cache_key(
    sample_keys: Sequence[str],
    genes: Sequence[str],
    models: Sequence[str],
    *,
    num_pcs: int,
    demean_mode: str,
    kind: str,
) -> str:
    h = hashlib.sha256()
    h.update(str(kind).encode())
    h.update(b"\x00")
    h.update("\x01".join(sorted(str(k) for k in sample_keys)).encode())
    h.update(b"\x00")
    h.update("\x01".join(str(g) for g in genes).encode())
    h.update(b"\x00")
    h.update("\x01".join(str(m) for m in models).encode())
    h.update(b"\x00")
    h.update(f"num_pcs={int(num_pcs)};demean={str(demean_mode).lower()}".encode())
    return h.hexdigest()[:16]


def _cache_load(cache_dir: str | Path, digest: str, verbose: bool):
    cache_root = Path(cache_dir)
    cache_path = cache_root / f"{digest}.pkl"
    # Repair NFS/umask-mangled modes (recurring on /scratch/asr655).
    if cache_root.exists():
        try:
            cache_root.chmod(cache_root.stat().st_mode | 0o700)
            if cache_path.exists():
                cache_path.chmod(cache_path.stat().st_mode | 0o600)
        except PermissionError:
            pass
    if cache_path.exists():
        if verbose:
            print(f"[pca_cache] hit  {cache_path}")
        with cache_path.open("rb") as fh:
            return pickle.load(fh)
    return None


def _cache_store(cache_dir: str | Path, digest: str, obj, verbose: bool) -> None:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        cache_root.chmod(cache_root.stat().st_mode | 0o700)
    except PermissionError:
        pass
    cache_path = cache_root / f"{digest}.pkl"
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    try:
        cache_path.chmod(cache_path.stat().st_mode | 0o600)
    except PermissionError:
        pass
    if verbose:
        print(f"[pca_cache] wrote {cache_path}")


def _clamp_num_pcs(num_pcs: int, n_samples: int, n_genes: int, verbose: bool) -> int:
    max_pcs = int(min(n_samples, n_genes))
    num_pcs = int(num_pcs)
    if num_pcs < 1:
        raise ValueError("num_pcs must be >= 1")
    if num_pcs > max_pcs:
        if verbose:
            print(
                f"[pca] num_pcs={num_pcs} exceeds min(n_samples, n_genes)="
                f"{max_pcs}; clamping to {max_pcs}"
            )
        num_pcs = max_pcs
    return num_pcs


# ---------------------------------------------------------------------------
# Compute — PCA recovery
# ---------------------------------------------------------------------------
def compute_pca_recovery(
    view: Mapping[str, object],
    *,
    num_pcs: int = 300,
    demean_mode: str = "none",
    models: Sequence[str] | None = None,
    cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled ground-truth PCA recovery from a canonical prediction view.

    PCA is fit on the pooled truth matrix ``X_true`` (rows = strict held-out
    subject-parcel samples, columns = genes). Each model's prediction matrix
    is projected into that same truth basis and compared component-wise.

    Returns ``(pc_df, summary_df)``:
    - ``pc_df`` — one row per ``(model, component)`` plus ``model='truth'``
      self-recovery reference rows. Columns: ``score_pearson``,
      ``score_spearman``, ``score_r2``, ``score_rmse``,
      ``explained_variance_ratio``, ``cumulative_variance``, ``pc95_cutoff``,
      ``n_samples``, ``n_genes``, ``num_pcs``, ``demean_mode``,
      ``panel_label``.
    - ``summary_df`` — one row per model: mean / variance-weighted PC-score
      Pearson r, PC1 score, 95% cutoff.

    Disk-cached under ``notebooks/cache/pca_recovery/<hash>.pkl``; the hash
    covers the view's sample keys, gene list, model set, ``num_pcs`` and
    ``demean_mode``.
    """
    X_true, pred_by_model, parcel_idx, sample_keys, genes, model_list = _view_matrices(view, models)
    panel_label = _panel_label_from_view(view)
    cdir = cache_dir if cache_dir is not None else _RECOVERY_CACHE_DIR
    digest = _pca_cache_key(
        sample_keys, genes, model_list,
        num_pcs=int(num_pcs), demean_mode=demean_mode, kind="recovery",
    )
    if cache and not force_rebuild:
        hit = _cache_load(cdir, digest, verbose)
        if hit is not None:
            return hit

    X_true, pred_by_model = _apply_within_parcel_demean(X_true, pred_by_model, parcel_idx, demean_mode)
    n_samples, n_genes = int(X_true.shape[0]), int(X_true.shape[1])
    num_pcs = _clamp_num_pcs(num_pcs, n_samples, n_genes, verbose)
    max_pcs = int(min(n_samples, n_genes))
    solver = "randomized" if num_pcs < max_pcs else "full"

    pca = PCA(n_components=num_pcs, svd_solver=solver, random_state=0)
    C_true = np.asarray(pca.fit_transform(X_true), dtype=np.float64)
    exp_var = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cum_var = np.cumsum(exp_var)
    pc95 = int(np.argmax(cum_var >= 0.95)) + 1 if np.any(cum_var >= 0.95) else int(len(cum_var))

    demean_lbl = str(demean_mode).lower()
    base_row = dict(
        pc95_cutoff=pc95, n_samples=n_samples, n_genes=n_genes,
        num_pcs=num_pcs, demean_mode=demean_lbl, panel_label=panel_label,
    )
    rows: list[dict] = []
    summary_rows: list[dict] = []

    # Truth self-recovery reference (a ceiling line for the recovery plot).
    for i in range(num_pcs):
        x = C_true[:, i]
        rows.append({
            "model": "truth", "component": i + 1,
            "score_pearson": _pearson_safe(x, x),
            "score_spearman": _spearman_safe(x, x),
            "score_r2": _r2_safe(x, x),
            "score_rmse": _rmse_safe(x, x),
            "explained_variance_ratio": float(exp_var[i]),
            "cumulative_variance": float(cum_var[i]),
            **base_row,
        })

    for model in model_list:
        C_pred = np.asarray(pca.transform(pred_by_model[model]), dtype=np.float64)
        pc_pear: list[float] = []
        w_num = w_den = 0.0
        for i in range(num_pcs):
            x, y = C_true[:, i], C_pred[:, i]
            pear = _pearson_safe(x, y)
            pc_pear.append(pear)
            if np.isfinite(pear) and np.isfinite(exp_var[i]):
                w_num += float(exp_var[i]) * float(pear)
                w_den += float(exp_var[i])
            rows.append({
                "model": model, "component": i + 1,
                "score_pearson": pear,
                "score_spearman": _spearman_safe(x, y),
                "score_r2": _r2_safe(x, y),
                "score_rmse": _rmse_safe(x, y),
                "explained_variance_ratio": float(exp_var[i]),
                "cumulative_variance": float(cum_var[i]),
                **base_row,
            })
        summary_rows.append({
            "model": model,
            "mean_score_pearson": float(np.nanmean(pc_pear)) if pc_pear else np.nan,
            "var_weighted_score_pearson": float(w_num / w_den) if w_den > 0 else np.nan,
            "pc1_score_pearson": float(pc_pear[0]) if pc_pear else np.nan,
            "pc95_cutoff": pc95, "n_samples": n_samples, "n_genes": n_genes,
            "num_pcs": num_pcs, "demean_mode": demean_lbl, "panel_label": panel_label,
        })

    pc_df = pd.DataFrame(rows)
    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("model").reindex(model_list).reset_index()
    )
    result = (pc_df, summary_df)
    if cache:
        _cache_store(cdir, digest, result, verbose)
    return result


# ---------------------------------------------------------------------------
# Compute — variance spectra
# ---------------------------------------------------------------------------
def compute_pca_variance_spectra(
    view: Mapping[str, object],
    *,
    num_pcs: int = 300,
    demean_mode: str = "none",
    models: Sequence[str] | None = None,
    cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source intrinsic PCA variance geometry.

    PCA is fit *separately* on the truth matrix and on each model's
    prediction matrix; the per-component and cumulative explained-variance
    spectra are returned. This is the "how many PCs does this source need to
    explain its own variance?" question, distinct from
    :func:`compute_pca_recovery` (which always projects into the truth basis).

    Returns ``(spectra_df, summary_df)``. ``spectra_df`` has one row per
    ``(source, component)`` with ``source`` in ``{'truth', *models}``.

    Disk-cached under ``notebooks/cache/pca_spectra/<hash>.pkl``.
    """
    X_true, pred_by_model, parcel_idx, sample_keys, genes, model_list = _view_matrices(view, models)
    panel_label = _panel_label_from_view(view)
    cdir = cache_dir if cache_dir is not None else _SPECTRA_CACHE_DIR
    digest = _pca_cache_key(
        sample_keys, genes, model_list,
        num_pcs=int(num_pcs), demean_mode=demean_mode, kind="spectra",
    )
    if cache and not force_rebuild:
        hit = _cache_load(cdir, digest, verbose)
        if hit is not None:
            return hit

    X_true, pred_by_model = _apply_within_parcel_demean(X_true, pred_by_model, parcel_idx, demean_mode)
    n_samples, n_genes = int(X_true.shape[0]), int(X_true.shape[1])
    num_pcs = _clamp_num_pcs(num_pcs, n_samples, n_genes, verbose)
    max_pcs = int(min(n_samples, n_genes))
    solver = "randomized" if num_pcs < max_pcs else "full"
    demean_lbl = str(demean_mode).lower()

    sources: dict[str, np.ndarray] = {"truth": X_true}
    for model in model_list:
        sources[model] = pred_by_model[model]

    rows: list[dict] = []
    summary_rows: list[dict] = []
    for source, X in sources.items():
        pca = PCA(n_components=num_pcs, svd_solver=solver, random_state=0)
        pca.fit(X)
        exp_var = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cum_var = np.cumsum(exp_var)
        pc95 = int(np.argmax(cum_var >= 0.95)) + 1 if np.any(cum_var >= 0.95) else int(len(cum_var))
        for i in range(num_pcs):
            rows.append({
                "source": source, "component": i + 1,
                "explained_variance_ratio": float(exp_var[i]),
                "cumulative_variance": float(cum_var[i]),
                "pc95_cutoff": pc95, "n_samples": n_samples, "n_genes": n_genes,
                "num_pcs": num_pcs, "demean_mode": demean_lbl, "panel_label": panel_label,
            })
        summary_rows.append({
            "source": source,
            "pc1_explained_variance_ratio": float(exp_var[0]) if len(exp_var) else np.nan,
            "pc95_cutoff": pc95, "n_samples": n_samples, "n_genes": n_genes,
            "num_pcs": num_pcs, "demean_mode": demean_lbl, "panel_label": panel_label,
        })

    spectra_df = pd.DataFrame(rows)
    source_order = ["truth"] + model_list
    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("source").reindex(source_order).reset_index()
    )
    result = (spectra_df, summary_df)
    if cache:
        _cache_store(cdir, digest, result, verbose)
    return result


# ---------------------------------------------------------------------------
# Compute — within-parcel variance
# ---------------------------------------------------------------------------
def compute_within_parcel_variance(
    view: Mapping[str, object],
    *,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean within-parcel gene std per source.

    For each source (truth + each model) and each parcel: the per-gene std
    across that parcel's pooled samples, averaged over genes. Answers "does
    the source produce subject-to-subject spread within a parcel at all?" —
    a companion to PCA recovery, which asks whether that spread aligns with
    the true latent axes. A model can score well here yet still fail
    recovery (variance present but misaligned).

    Long-form output: ``(source, parcel_idx, n_samples, mean_within_parcel_std)``.
    """
    X_true, pred_by_model, parcel_idx, sample_keys, genes, model_list = _view_matrices(view, models)
    panel_label = _panel_label_from_view(view)
    sources: dict[str, np.ndarray] = {"truth": X_true}
    for model in model_list:
        sources[model] = pred_by_model[model]

    rows: list[dict] = []
    uniq = np.unique(parcel_idx)
    for source, X in sources.items():
        for p in uniq:
            idx = np.where(parcel_idx == int(p))[0]
            if idx.size < 2:
                std_mean = np.nan
            else:
                std_mean = float(np.nanmean(np.nanstd(X[idx, :], axis=0, ddof=1)))
            rows.append({
                "source": source,
                "parcel_idx": int(p),
                "n_samples": int(idx.size),
                "mean_within_parcel_std": std_mean,
                "panel_label": panel_label,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _component_xticks(ax, components, stride: int, tick_fontsize: int) -> None:
    comps = sorted({int(c) for c in components})
    if not comps:
        return
    stride = max(1, int(stride))
    ticks = sorted({comps[0]} | {c for c in comps if c % stride == 0} | {comps[-1]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=0, fontsize=tick_fontsize)


def _resolve_sources(present: set[str], sources: Sequence[str] | None) -> list[str]:
    ordered = ["truth"] + [m for m in MODEL_ORDER if m in present]
    if sources is None:
        return ordered
    want = {str(s).lower() for s in sources}
    return [s for s in ordered if s in want]


# ---------------------------------------------------------------------------
# Plot — PCA recovery
# ---------------------------------------------------------------------------
def plot_pca_recovery(
    pc_df: pd.DataFrame,
    *,
    metric: str = "score_pearson",
    models: Sequence[str] | None = None,
    show_truth: bool = True,
    show_lines: bool = False,
    show_fit: str | None = None,
    lowess_frac: float = 0.2,
    x_label_stride: int = 20,
    panel_label: str | None = None,
    figsize: Tuple[float, float] = (9.6, 4.8),
    dpi: int = 200,
    font_sizes: Mapping[str, str | int] | None = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Component-wise PCA recovery curve, one series per model.

    The PCA basis is always the truth basis; each component asks how well a
    model's predicted PC scores align with the true PC scores across pooled
    samples. ``show_truth`` overlays the truth self-recovery ceiling. The
    dashed vertical rule marks the 95% true-data variance cutoff.
    """
    if metric not in _SCORE_METRIC_KEYS:
        raise ValueError(f"metric must be one of {sorted(_SCORE_METRIC_KEYS)}")
    fonts = _resolve_fonts(_RECOVERY_FONTS, font_sizes)
    fit_style = None if show_fit is None else str(show_fit).strip().lower()
    if fit_style not in {None, "lowess"}:
        raise ValueError("show_fit must be None or 'lowess'")
    if fit_style == "lowess" and _sm_lowess is None:
        raise ImportError("show_fit='lowess' requires statsmodels to be installed")

    d = pc_df.copy()
    d["model"] = d["model"].astype(str).str.lower()
    present = set(d["model"])
    model_list = ordered_models(models) if models is not None else [m for m in MODEL_ORDER if m in present]
    d_truth = d[d["model"] == "truth"]
    d_models = d[d["model"].isin(model_list)]
    if len(d_models) == 0 and not (show_truth and len(d_truth)):
        raise RuntimeError("no pooled PCA recovery rows to plot")

    ref = d_models if len(d_models) else d_truth
    n_samples = int(ref["n_samples"].iloc[0])
    n_genes = int(ref["n_genes"].iloc[0])
    pc95 = int(ref["pc95_cutoff"].iloc[0])
    demean_mode = str(ref["demean_mode"].iloc[0]) if "demean_mode" in ref.columns else "none"
    if panel_label is None and "panel_label" in ref.columns:
        panel_label = str(ref["panel_label"].iloc[0])
    panel_label = panel_label or "All Genes"

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)

    def _draw(sub: pd.DataFrame, color: str, label: str, *, truth: bool = False) -> None:
        sub = sub.sort_values("component")
        xx = sub["component"].to_numpy(dtype=np.float64)
        yy = sub[metric].to_numpy(dtype=np.float64)
        z = 5 if truth else 3
        if fit_style == "lowess":
            ax.scatter(
                xx, yy, s=20.0, alpha=0.45, color=color,
                edgecolors="white" if truth else "none",
                linewidths=0.3 if truth else 0.0, label=label, zorder=z - 1,
            )
            ok = np.isfinite(xx) & np.isfinite(yy)
            if int(ok.sum()) >= 4:
                sm = _sm_lowess(yy[ok], xx[ok], frac=float(lowess_frac), return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], lw=2.0, color=color, alpha=0.95, zorder=z)
        elif show_lines:
            ax.plot(xx, yy, marker="o", ms=3.5, lw=1.6, color=color, label=label, zorder=z)
        else:
            ax.scatter(
                xx, yy, s=22.0, alpha=0.85, color=color,
                edgecolors="white" if truth else "none",
                linewidths=0.3 if truth else 0.0, label=label, zorder=z,
            )

    for model in model_list:
        sub = d_models[d_models["model"] == model]
        if len(sub):
            _draw(sub, MODEL_COLORS.get(model, "#444444"), model_label(model))
    if show_truth and len(d_truth):
        _draw(d_truth, TRUTH_COLOR, TRUTH_LABEL, truth=True)

    if pc95 >= 1:
        ax.axvline(
            pc95, color="#555555", ls="--", lw=1.1, alpha=0.8,
            label=f"95% truth variance (PC {pc95})",
        )

    _component_xticks(ax, d["component"], x_label_stride, fonts["tick"])
    ax.set_xlabel("Principal component", fontsize=fonts["xlabel"])
    metric_lbl = format_metric_label(_SCORE_METRIC_KEYS[metric])
    ax.set_ylabel(f"PC-score recovery — {metric_lbl}", fontsize=fonts["ylabel"])
    title = f"Ground-truth PCA recovery — {panel_label}"
    if demean_mode not in {"", "none"}:
        title += f"  [{_format_demean_mode_label(demean_mode)}]"
    ax.set_title(title, fontsize=fonts["title"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=fonts["legend"], loc="upper right")
    ax.text(
        0.99, 0.02, f"n_samples = {n_samples:,}    n_genes = {n_genes:,}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=fonts["annotation"], color="#555555",
    )
    apply_tick_style(ax, label_fontsize=fonts["tick"])
    return fig, ax, d


# ---------------------------------------------------------------------------
# Plot — variance spectra
# ---------------------------------------------------------------------------
def plot_pca_variance_spectra(
    spectra_df: pd.DataFrame,
    *,
    sources: Sequence[str] | None = None,
    panel_label: str | None = None,
    x_label_stride: int = 20,
    show_cumulative: bool = True,
    mark_pc95: bool = True,
    figsize: Tuple[float, float] = (12.6, 4.8),
    dpi: int = 200,
    font_sizes: Mapping[str, str | int] | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Per-component (and cumulative) explained-variance spectra by source.

    All requested sources are overlaid on a shared pair of axes so the
    intrinsic variance geometry of truth and each model can be compared
    directly. Pass ``sources=['truth']`` to isolate one.
    """
    fonts = _resolve_fonts(_SPECTRA_FONTS, font_sizes)
    d = spectra_df.copy()
    d["source"] = d["source"].astype(str).str.lower()
    src_list = _resolve_sources(set(d["source"]), sources)
    d = d[d["source"].isin(src_list)]
    if len(d) == 0:
        raise RuntimeError("no PCA variance spectra rows to plot")

    n_samples = int(d["n_samples"].iloc[0])
    n_genes = int(d["n_genes"].iloc[0])
    demean_mode = str(d["demean_mode"].iloc[0]) if "demean_mode" in d.columns else "none"
    if panel_label is None and "panel_label" in d.columns:
        panel_label = str(d["panel_label"].iloc[0])
    panel_label = panel_label or "All Genes"

    color_map = {"truth": TRUTH_COLOR, **MODEL_COLORS}
    label_map = {"truth": TRUTH_LABEL, **MODEL_LABELS}

    ncols = 2 if show_cumulative else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=figsize if show_cumulative else (figsize[0] / 2.0, figsize[1]),
        dpi=dpi, constrained_layout=True, squeeze=False,
    )
    axes = axes[0]
    ax_ev = axes[0]
    ax_cum = axes[1] if show_cumulative else None

    for source in src_list:
        sub = d[d["source"] == source].sort_values("component")
        if len(sub) == 0:
            continue
        color = color_map.get(source, "#444444")
        label = label_map.get(source, source)
        ax_ev.plot(
            sub["component"], sub["explained_variance_ratio"],
            marker="o", ms=3.0, lw=1.5, color=color, alpha=0.9, label=label,
        )
        if ax_cum is not None:
            ax_cum.plot(
                sub["component"], sub["cumulative_variance"],
                marker="o", ms=3.0, lw=1.5, color=color, alpha=0.9, label=label,
            )
            if mark_pc95:
                cutoff = int(sub["pc95_cutoff"].iloc[0])
                if cutoff >= 1:
                    ax_cum.axvline(cutoff, color=color, ls="--", lw=1.0, alpha=0.4)

    for ax in [a for a in (ax_ev, ax_cum) if a is not None]:
        _component_xticks(ax, d["component"], x_label_stride, fonts["tick"])
        ax.set_xlabel("Principal component", fontsize=fonts["xlabel"])
        ax.grid(alpha=0.22)
        apply_tick_style(ax, label_fontsize=fonts["tick"])

    ax_ev.set_ylabel("Explained variance ratio", fontsize=fonts["ylabel"])
    ax_ev.set_title("Per-component variance", fontsize=fonts["title"])
    ax_ev.legend(frameon=False, fontsize=fonts["legend"], loc="upper right")
    if ax_cum is not None:
        ax_cum.set_ylabel("Cumulative variance", fontsize=fonts["ylabel"])
        ax_cum.set_title("Cumulative variance", fontsize=fonts["title"])
        ax_cum.set_ylim(-0.02, 1.02)
        ax_cum.axhline(0.95, color="#999999", ls=":", lw=1.0, alpha=0.7)
        ax_cum.legend(frameon=False, fontsize=fonts["legend"], loc="upper right")

    sup = (
        f"PCA variance spectra — {panel_label}  "
        f"(n_samples = {n_samples:,}; n_genes = {n_genes:,})"
    )
    if demean_mode not in {"", "none"}:
        sup += f"  [{_format_demean_mode_label(demean_mode)}]"
    fig.suptitle(sup, fontsize=fonts["title"], y=1.05)
    return fig, axes, d


# ---------------------------------------------------------------------------
# Plot — within-parcel variance
# ---------------------------------------------------------------------------
def plot_within_parcel_variance(
    var_df: pd.DataFrame,
    *,
    prepost: Mapping[str, object] | None = None,
    label_mode: str = "gtex",
    sources: Sequence[str] | None = None,
    panel_label: str | None = None,
    figsize: Tuple[float, float] = (13.0, 5.0),
    dpi: int = 200,
    font_sizes: Mapping[str, str | int] | None = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Grouped bar chart of mean within-parcel gene std, one bar group per parcel.

    Truth bars set the reference spread; a model whose bars collapse toward
    zero (especially under ``demean_mode='within_parcel'`` upstream) is not
    producing subject-to-subject variation within that parcel.
    """
    fonts = _resolve_fonts(_VARIANCE_FONTS, font_sizes)
    d = var_df.copy()
    d["source"] = d["source"].astype(str).str.lower()
    src_list = _resolve_sources(set(d["source"]), sources)
    parcels = sorted(int(p) for p in d["parcel_idx"].unique())
    if panel_label is None and "panel_label" in d.columns and len(d):
        panel_label = str(d["panel_label"].iloc[0])

    label_lookup: dict[int, str] = {}
    if prepost is not None:
        label_lookup = parcel_label_lookup(build_parcel_label_table(prepost), label_mode=label_mode)
    parcel_labels = [
        strip_display_label_prefixes(label_lookup.get(int(p), f"parcel {int(p)}")) for p in parcels
    ]

    color_map = {"truth": TRUTH_COLOR, **MODEL_COLORS}
    label_map = {"truth": TRUTH_LABEL, **MODEL_LABELS}

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    n_src = max(1, len(src_list))
    width = 0.8 / n_src
    x = np.arange(len(parcels), dtype=np.float64)
    for i, source in enumerate(src_list):
        sub = d[d["source"] == source].set_index("parcel_idx")
        vals = [float(sub["mean_within_parcel_std"].get(int(p), np.nan)) for p in parcels]
        ax.bar(
            x + i * width - 0.4 + width / 2.0, vals, width=width,
            color=color_map.get(source, "#444444"), label=label_map.get(source, source),
            alpha=0.9, edgecolor="white", linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(parcel_labels, rotation=40, ha="right", fontsize=fonts["tick"])
    ax.set_ylabel("Mean within-parcel gene std", fontsize=fonts["ylabel"])
    ax.set_xlabel("Parcel", fontsize=fonts["xlabel"])
    title = "Within-parcel expression variance by source"
    if panel_label:
        title += f" — {panel_label}"
    ax.set_title(title, fontsize=fonts["title"])
    ax.legend(frameon=False, fontsize=fonts["legend"], ncol=n_src, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    apply_tick_style(ax, label_fontsize=fonts["tick"])
    return fig, ax, d


# ---------------------------------------------------------------------------
# Plot — sample x gene heatmap
# ---------------------------------------------------------------------------
def plot_sample_gene_heatmap(
    view: Mapping[str, object],
    source: str = "truth",
    *,
    prepost: Mapping[str, object] | None = None,
    label_mode: str = "gtex",
    group_by: str | None = "parcel",
    sort_genes_by_mean: bool = False,
    sample_n_genes: int | None = None,
    random_seed: int = 0,
    gene_order: Sequence[str] | None = None,
    cmap: str = "viridis",
    row_scale: float = 0.012,
    figsize: Tuple[float, float] | None = None,
    dpi: int = 200,
    panel_label: str | None = None,
    font_sizes: Mapping[str, str | int] | None = None,
) -> Tuple[plt.Figure, plt.Axes, list[str]]:
    """Sample x gene expression heatmap for one source of a prediction view.

    ``source`` is ``'truth'`` or a model name. Rows are pooled strict
    held-out samples, optionally grouped (white separators + right-edge
    labels) by ``'subject'`` or ``'parcel'``. Pass a fixed ``gene_order``
    (e.g. the truth gene order) to keep columns comparable across sources.
    Returns the gene order used so callers can reuse it.
    """
    fonts = _resolve_fonts(_HEATMAP_FONTS, font_sizes)
    src = str(source).lower()
    truth_df = view["truth_df"]
    if src == "truth":
        df = truth_df
    else:
        if src not in view["pred_dfs"]:
            raise KeyError(f"source must be 'truth' or one of {sorted(view['pred_dfs'])}")
        df = view["pred_dfs"][src]
    if group_by not in (None, "subject", "parcel"):
        raise ValueError("group_by must be None, 'subject', or 'parcel'")

    genes = list(view["genes"])
    gene_set = set(genes)
    if gene_order is not None:
        genes = [g for g in gene_order if g in gene_set]
    elif sample_n_genes is not None and int(sample_n_genes) < len(genes):
        rng = np.random.default_rng(int(random_seed))
        pick = np.sort(rng.choice(len(genes), size=int(sample_n_genes), replace=False))
        genes = [genes[i] for i in pick]
    if sort_genes_by_mean and gene_order is None:
        means = truth_df[genes].mean(axis=0)
        genes = list(means.sort_values().index)
    if len(genes) == 0:
        raise ValueError("no genes selected for the heatmap")

    work = df[["subject", "parcel_idx"] + genes].copy()
    group_values = None
    if group_by is not None:
        sort_cols = ["subject", "parcel_idx"] if group_by == "subject" else ["parcel_idx", "subject"]
        work = work.sort_values(sort_cols).reset_index(drop=True)
        group_values = (work["subject"] if group_by == "subject" else work["parcel_idx"]).to_numpy()
    else:
        work = work.reset_index(drop=True)

    M = work[genes].to_numpy(dtype=np.float64)
    n_rows = int(M.shape[0])
    if figsize is None:
        width = float(min(22.0, 0.085 * len(genes) + 4.5))
        height = float(max(4.0, min(26.0, row_scale * n_rows + 2.0)))
        figsize = (width, height)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")

    if group_values is not None:
        changes = np.where(group_values[1:] != group_values[:-1])[0] + 1
        for c in changes:
            ax.axhline(float(c) - 0.5, color="white", lw=0.7, alpha=0.9)
        bounds = [0, *changes.tolist(), n_rows]
        lk: dict[int, str] = {}
        if prepost is not None and group_by == "parcel":
            lk = parcel_label_lookup(build_parcel_label_table(prepost), label_mode=label_mode)
        centers: list[float] = []
        labels: list[str] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            centers.append((a + b) / 2.0 - 0.5)
            gv = group_values[a]
            if group_by == "parcel":
                labels.append(strip_display_label_prefixes(lk.get(int(gv), f"parcel {int(gv)}")))
            else:
                labels.append(str(gv))
        ax_r = ax.secondary_yaxis("right")
        ax_r.set_yticks(centers)
        ax_r.set_yticklabels(labels, fontsize=fonts["tick"])
        ax_r.tick_params(length=0)

    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=fonts["tick"])
    ax.set_yticks([])
    ax.set_xlabel(f"Genes (n = {len(genes)})", fontsize=fonts["xlabel"])
    ylab = f"Samples (n = {n_rows})"
    if group_by is not None:
        ylab += f"  ·  grouped by {group_by}"
    ax.set_ylabel(ylab, fontsize=fonts["ylabel"])

    src_label = TRUTH_LABEL if src == "truth" else model_label(src)
    title = f"{src_label} — sample × gene expression"
    if panel_label is None:
        panel_label = _panel_label_from_view(view)
    if panel_label:
        title += f"  ({panel_label})"
    ax.set_title(title, fontsize=fonts["title"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.06)
    cbar.set_label("Harmonized expression", fontsize=fonts["cbar_label"])
    cbar.ax.tick_params(labelsize=fonts["tick"])
    return fig, ax, genes
