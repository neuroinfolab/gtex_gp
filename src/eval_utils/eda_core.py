#!/usr/bin/env python3
from __future__ import annotations

"""
Dataset-level EDA and PREPOST reconstruction helpers.

This module is the new import surface for `eval_data.ipynb`. During the first
refactor pass, implementations are delegated to `results_eda.py` so notebooks
can move to focused modules before the implementation is fully relocated.
Do not add new dataset-level functionality to `results_eda.py`; add it here.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval_style import (
    FONT,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    pretty_gtex_label,
    set_academic_style,
)
from .results_eda import (
    EDAConfig,
    available_regions,
    coverage_count_distribution,
    coverage_count_distribution_from_table,
    parcel_subject_count_table,
    plot_coverage_count_distribution,
    plot_parcel_subject_counts,
    plot_region_covariance_side_by_side,
    plot_region_region_spearman_heatmaps,
    plot_subject_region_count_distribution,
    prepare_pre_post_harmonization,
    region_covariance_matrices,
    subject_region_count_distribution,
)
from .results_eda import plot_subject_prepost_heatmaps as _legacy_plot_subject_prepost_heatmaps


__all__ = [
    "EDAConfig",
    "FONT",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "available_regions",
    "coverage_count_distribution",
    "coverage_count_distribution_from_table",
    "parcel_subject_count_table",
    "plot_atlas_median_comparison_heatmaps",
    "plot_coverage_count_distribution",
    "plot_gtex_demographic_breakdown",
    "plot_parcel_subject_counts",
    "plot_region_covariance_side_by_side",
    "plot_region_region_spearman_heatmaps",
    "plot_subject_prepost_heatmaps",
    "plot_subject_region_count_distribution",
    "prepare_atlas_median_comparison",
    "prepare_pre_post_harmonization",
    "prepare_pre_post_harmonization_cached",
    "pretty_gtex_label",
    "region_covariance_matrices",
    "load_eval_gene_list",
    "resolve_dataset_gene_panel",
    "resolve_eval_gene_list_path",
    "set_academic_style",
    "subject_region_count_distribution",
]


def _prepost_cache_key(cfg: "EDAConfig") -> str:
    """Stable 16-hex digest of CFG fields that affect PREPOST output.

    Includes the CSV file mtime + size so the cache invalidates if the
    underlying expression CSV changes. (HVG list is no longer part of
    EDAConfig — for gene_scope='hvg', the gene panel comes from the
    LORO cache itself, which is keyed by `cache_root`.)
    """
    import hashlib
    fields: list[tuple[str, str]] = []
    for attr in (
        "csv_path", "cache_root", "gene_scope",
        "min_observed_parcels", "combat_use_covariates",
        "gtex_rep_mode", "gtex_hemi_mode",
    ):
        if hasattr(cfg, attr):
            fields.append((attr, str(getattr(cfg, attr))))
    # Resolved matching policy (auto-detected from cache when cfg.matching_policy is None).
    # Different policies produce different gtex parcel_idx assignments, so PREPOST blobs
    # must not be shared across them.
    try:
        from src.eval_utils.results_eda import resolve_matching_policy
        fields.append(("matching_policy", resolve_matching_policy(cfg)))
    except Exception:
        fields.append(("matching_policy", str(getattr(cfg, "matching_policy", "centroids") or "centroids")))
    for path_attr in ("csv_path",):
        p = getattr(cfg, path_attr, None)
        if not p:
            continue
        full = Path(p)
        if not full.is_absolute():
            full = full.resolve()
        try:
            stat = full.stat()
            fields.append((f"{path_attr}_mtime_ns", str(int(stat.st_mtime_ns))))
            fields.append((f"{path_attr}_size", str(int(stat.st_size))))
        except FileNotFoundError:
            pass
    blob = repr(sorted(fields)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare_pre_post_harmonization_cached(
    cfg: "EDAConfig",
    *,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Disk-cached `prepare_pre_post_harmonization`.

    First call computes PREPOST and pickles it under
    `out/prepost_cache/<hash>.pkl` (override via `cache_dir`). Subsequent calls
    on the same EDAConfig load from disk in seconds rather than re-fitting
    ComBat. The hash key covers the CSV/HVG paths (with their mtime + size),
    `gene_scope`, `min_observed_parcels`, `combat_use_covariates`,
    `gtex_rep_mode`, `gtex_hemi_mode` — change any of those and the cache is
    rebuilt automatically. Set `force_rebuild=True` to bypass.

    Pickled blob includes the harmonized cubes, eligible-subject metadata,
    and dataframes — typically a few hundred MB at the all-genes scope, much
    smaller for HVG.
    """
    import pickle
    import time

    cache_root = Path(cache_dir) if cache_dir is not None else Path("notebooks/cache/prepost")
    digest = _prepost_cache_key(cfg)
    cache_path = cache_root / f"{digest}.pkl"

    # Repair permissions on dir + existing pickle if NFS/umask created them
    # without owner read/write (recurring on this filesystem).
    if cache_root.exists():
        try:
            cache_root.chmod(cache_root.stat().st_mode | 0o700)
            if cache_path.exists():
                cache_path.chmod(cache_path.stat().st_mode | 0o600)
        except PermissionError:
            pass

    if cache_path.exists() and not force_rebuild:
        if verbose:
            sz = cache_path.stat().st_size / (1024 * 1024)
            print(f"[prepost_cache] hit  {cache_path}  ({sz:.1f} MB)")
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    if verbose:
        print(f"[prepost_cache] miss {cache_path} — computing PREPOST (this is the slow path)...")
    t0 = time.time()
    prepost = prepare_pre_post_harmonization(cfg)
    elapsed = time.time() - t0

    cache_root.mkdir(parents=True, exist_ok=True)
    # Force owner-rwx on the directory so subsequent writes from the same user
    # don't trip on NFS-set 0o000 modes.
    try:
        cache_root.chmod(cache_root.stat().st_mode | 0o700)
    except PermissionError:
        pass

    tmp_path = cache_path.with_suffix(".pkl.tmp")
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(prepost, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        cache_path.chmod(cache_path.stat().st_mode | 0o600)
    except PermissionError as e:
        if verbose:
            print(f"[prepost_cache] WARNING — could not write {cache_path}: {e}")
            print(f"[prepost_cache] returning computed PREPOST without caching; "
                  f"fix perms with `chmod -R u+rwX {cache_root}` or pass cache_dir=...")
    if cache_path.exists() and verbose:
        sz = cache_path.stat().st_size / (1024 * 1024)
        print(f"[prepost_cache] computed in {elapsed:.1f}s, cached to {cache_path} ({sz:.1f} MB)")
    return prepost


def _first_non_null(s: pd.Series) -> object:
    vals = s.dropna()
    if len(vals) == 0:
        return np.nan
    return vals.iloc[0]


def _age_sort_key(value: object) -> Tuple[int, str]:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return (10**9, s)
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if digits:
        return (int(digits), s)
    return (10**8, s)


def plot_gtex_demographic_breakdown(
    prepost: Dict[str, object] | None = None,
    gtex_df: pd.DataFrame | None = None,
    eligible_only: bool = False,
    figsize: Tuple[float, float] = (10.5, 4.0),
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    Plot subject-level GTEx age and sex distributions.

    `gxp_samples.csv` has one GTEx row per donor-tissue sample, so this first
    collapses to one row per subject before counting demographics.
    """
    if gtex_df is None:
        if prepost is None:
            raise ValueError("Either prepost or gtex_df is required")
        gtex_df = prepost["gtex_eligible_raw"] if bool(eligible_only) else prepost["gtex_raw"]
    g = gtex_df.copy()
    if "dataset" in g.columns:
        g = g[g["dataset"].astype(str).str.upper() == "GTEX"].copy()
    required = {"subject", "age", "sex"}
    missing = required.difference(set(g.columns))
    if missing:
        raise ValueError(f"GTEx demographic plot requires columns: {sorted(missing)}")
    demo = (
        g.groupby("subject", as_index=False)
        .agg(age=("age", _first_non_null), sex=("sex", _first_non_null))
        .reset_index(drop=True)
    )
    demo["age"] = demo["age"].fillna("Unknown").astype(str)
    demo["sex"] = demo["sex"].fillna("Unknown").astype(str)

    age_counts = demo["age"].value_counts(dropna=False).rename_axis("age").reset_index(name="n_subjects")
    age_counts = age_counts.sort_values("age", key=lambda s: s.map(_age_sort_key)).reset_index(drop=True)
    sex_counts = demo["sex"].value_counts(dropna=False).rename_axis("sex").reset_index(name="n_subjects")
    sex_counts = sex_counts.sort_values(["sex"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].bar(age_counts["age"].astype(str), age_counts["n_subjects"], color="#4c78a8")
    axes[0].set_title("GTEx age distribution", fontsize=FONT["title"])
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Subjects")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(sex_counts["sex"].astype(str), sex_counts["n_subjects"], color="#f58518")
    axes[1].set_title("GTEx sex distribution", fontsize=FONT["title"])
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Subjects")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
    fig.suptitle(
        f"GTEx subject demographics ({'eligible subjects' if bool(eligible_only) else 'all GTEx subjects'}; n={len(demo)})",
        fontsize=FONT["title"] + 1,
        y=1.04,
    )
    return fig, axes, demo


def _normalize_summary_stat(summary_stat: str) -> str:
    stat = str(summary_stat).strip().lower()
    if stat not in {"mean", "median"}:
        raise ValueError("summary_stat must be one of: mean, median")
    return stat


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_eval_gene_list_path(path_str: str) -> Path:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError("eval_gene_list_path cannot be empty")
    p = Path(raw)
    candidates: List[Path] = []
    root = _repo_root()
    if p.is_absolute():
        candidates.append(p)
    else:
        if p.parts and p.parts[0] == root.name:
            candidates.append((root / Path(*p.parts[1:])).resolve())
        candidates.extend(
            [
                (root / p).resolve(),
                (root.parent / "out" / "raw" / "gene_lists" / p).resolve(),
                (root / "out" / "raw" / "gene_lists" / p).resolve(),
                (root / "data" / "metadata" / "gene_lists" / p).resolve(),
                (root / "data" / "raw" / "gene_lists" / p).resolve(),
                (root / "data" / "metadata" / p).resolve(),
                (root / "data" / "raw" / p).resolve(),
            ]
        )
        if p.suffix == "":
            candidates.extend(
                [
                    (root / f"{raw}.txt").resolve(),
                    (root.parent / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
                    (root / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
                    (root / "data" / "metadata" / "gene_lists" / f"{raw}.txt").resolve(),
                    (root / "data" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
                    (root / "data" / "metadata" / f"{raw}.txt").resolve(),
                    (root / "data" / "raw" / f"{raw}.txt").resolve(),
                ]
            )
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve eval gene list from {path_str!r}")


def load_eval_gene_list(eval_gene_list_path: str) -> List[str]:
    p = resolve_eval_gene_list_path(eval_gene_list_path)
    genes = [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
    if not genes:
        raise ValueError(f"No genes found in eval gene list: {p}")
    return genes


def resolve_dataset_gene_panel(
    prepost: Dict[str, object],
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
) -> List[str]:
    mode = str(gene_mode).strip().lower()
    available = [str(g) for g in prepost["genes"]]
    if mode == "allgenes":
        return available
    if mode != "custom":
        raise ValueError("gene_mode must be one of: allgenes, custom")

    if eval_gene_list_path is not None and str(eval_gene_list_path).strip():
        requested = load_eval_gene_list(str(eval_gene_list_path))
    elif custom_gene_list is not None:
        requested = [str(g) for g in custom_gene_list]
    else:
        raise ValueError("custom gene mode requires eval_gene_list_path or custom_gene_list")

    available_set = set(available)
    keep = [str(g) for g in requested if str(g) in available_set]
    if not keep:
        raise ValueError("Requested custom gene panel has no overlap with PREPOST genes")
    return keep


def _gtex_parcel_subject_summary_matrix(
    gtex_df: pd.DataFrame,
    genes: List[str],
    target_meta: pd.DataFrame,
    summary_stat: str = "median",
) -> np.ndarray:
    stat = _normalize_summary_stat(summary_stat)
    n_parc = int(len(target_meta))
    out = np.full((n_parc, len(genes)), np.nan, dtype=np.float64)
    if len(gtex_df) == 0:
        return out
    subj_parcel = gtex_df.groupby(["subject", "parcel_idx"], as_index=False)[genes].mean()
    if stat == "mean":
        parcel_summary = subj_parcel.groupby("parcel_idx", as_index=False)[genes].mean()
    else:
        parcel_summary = subj_parcel.groupby("parcel_idx", as_index=False)[genes].median()
    for _, row in parcel_summary.iterrows():
        p = int(row["parcel_idx"])
        out[p, :] = row[genes].to_numpy(dtype=np.float64)
    return out


def _parcel_summary_matrix(
    df: pd.DataFrame,
    genes: List[str],
    target_meta: pd.DataFrame,
    summary_stat: str = "median",
) -> np.ndarray:
    stat = _normalize_summary_stat(summary_stat)
    n_parc = int(len(target_meta))
    out = np.full((n_parc, len(genes)), np.nan, dtype=np.float64)
    if len(df) == 0:
        return out
    if stat == "mean":
        grp = df.groupby("parcel_idx", as_index=False)[genes].mean()
    else:
        grp = df.groupby("parcel_idx", as_index=False)[genes].median()
    for _, row in grp.iterrows():
        p = int(row["parcel_idx"])
        out[p, :] = row[genes].to_numpy(dtype=np.float64)
    return out


def prepare_atlas_median_comparison(
    prepost: Dict[str, object],
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    observed_only: bool = True,
    summary_stat: str = "median",
) -> Dict[str, object]:
    stat = _normalize_summary_stat(summary_stat)
    genes = resolve_dataset_gene_panel(
        prepost,
        gene_mode=gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_list_path=eval_gene_list_path,
    )
    target_meta = prepost["target_meta"]
    labels = target_meta["tissue_or_parcel"].astype(str).tolist()

    gtex_raw = prepost["gtex_eligible_raw"]
    gtex_h = prepost["gtex_eligible_h"]
    ahba_raw = prepost["ahba_raw"]
    ahba_h = prepost["ahba_h"]

    g_raw = _gtex_parcel_subject_summary_matrix(gtex_raw, genes, target_meta, summary_stat=stat)
    g_h = _gtex_parcel_subject_summary_matrix(gtex_h, genes, target_meta, summary_stat=stat)
    a_raw = _parcel_summary_matrix(ahba_raw, genes, target_meta, summary_stat=stat)
    a_h = _parcel_summary_matrix(ahba_h, genes, target_meta, summary_stat=stat)

    if bool(observed_only):
        keep = np.isfinite(g_raw).any(axis=1) | np.isfinite(g_h).any(axis=1)
    else:
        keep = np.ones(g_raw.shape[0], dtype=bool)

    gtex_native_by_parcel = (
        gtex_raw.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )
    keep_idx = np.where(keep)[0].tolist()
    gtex_labels = [str(gtex_native_by_parcel.get(int(i), labels[i] if i < len(labels) else "")) for i in keep_idx]
    both_labels = []
    for i in keep_idx:
        ahba_name = labels[i]
        gname = str(gtex_native_by_parcel.get(int(i), ""))
        right = gname if gname else ahba_name
        both_labels.append(f"{ahba_name}: {right}")

    return {
        "genes": genes,
        "parcel_labels": [labels[i] for i in keep_idx],
        "parcel_labels_gtex": gtex_labels,
        "parcel_labels_both": both_labels,
        "parcel_idx": np.asarray(keep_idx, dtype=np.int32),
        "gtex_raw": g_raw[keep, :],
        "ahba_raw": a_raw[keep, :],
        "gtex_h": g_h[keep, :],
        "ahba_h": a_h[keep, :],
        "summary_stat": stat,
    }


def plot_subject_prepost_heatmaps(
    prepost: Dict[str, object],
    subject_id: str,
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    observed_only: bool = True,
    label_mode: str = "gtex",
    label_stride: int = 1,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (13.0, 4.6),
) -> Tuple[plt.Figure, np.ndarray]:
    genes = resolve_dataset_gene_panel(
        prepost,
        gene_mode=gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_list_path=eval_gene_list_path,
    )
    return _legacy_plot_subject_prepost_heatmaps(
        prepost,
        subject_id=subject_id,
        gene_mode="custom",
        gene_list=genes,
        observed_only=observed_only,
        label_mode=label_mode,
        label_stride=label_stride,
        cmap=cmap,
        figsize=figsize,
    )


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    x = x[m]
    y = y[m]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _rmse_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((x[m] - y[m]) ** 2)))


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n == 0:
        return np.nan, np.nan, 0
    return _pearson_safe(x[m], y[m]), _rmse_safe(x[m], y[m]), n


def plot_atlas_median_comparison_heatmaps(
    atlas_cmp: Dict[str, object],
    label_stride: int = 1,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (13.5, 9.0),
) -> Tuple[plt.Figure, np.ndarray]:
    g_raw = np.asarray(atlas_cmp["gtex_raw"], dtype=np.float64)
    a_raw = np.asarray(atlas_cmp["ahba_raw"], dtype=np.float64)
    g_h = np.asarray(atlas_cmp["gtex_h"], dtype=np.float64)
    a_h = np.asarray(atlas_cmp["ahba_h"], dtype=np.float64)
    labels = [str(x) for x in atlas_cmp["parcel_labels"]]
    summary_stat = str(atlas_cmp.get("summary_stat", "median")).lower()
    summary_label = "mean" if summary_stat == "mean" else "median"

    finite_vals = np.r_[g_raw.ravel(), a_raw.ravel(), g_h.ravel(), a_h.ravel()]
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if int(finite_vals.size):
        vmin = float(np.nanpercentile(finite_vals, 1.0))
        vmax = float(np.nanpercentile(finite_vals, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
    else:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    mats = [
        (f"Raw GTEx (eligible-subject {summary_label})", g_raw, axes[0, 0]),
        (f"Raw AHBA (parcel {summary_label})", a_raw, axes[0, 1]),
        (f"ComBat GTEx (eligible-subject {summary_label})", g_h, axes[1, 0]),
        (f"ComBat AHBA (parcel {summary_label})", a_h, axes[1, 1]),
    ]
    im = None
    for title, mat, ax in mats:
        im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=FONT["title"])
        ax.set_xlabel("Genes", fontsize=FONT["label"])
        ax.set_ylabel("AHBA parcels", fontsize=FONT["label"])
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
        ax.minorticks_off()

    step = max(1, int(label_stride))
    y_idx = np.arange(0, len(labels), step, dtype=int)
    y_lab = [labels[i] for i in y_idx.tolist()]
    for ax in axes.ravel().tolist():
        ax.set_yticks(y_idx.tolist())
        ax.set_yticklabels(y_lab, fontsize=FONT["small"])

    r_raw, rmse_raw, n_raw = _pair_metrics(g_raw, a_raw)
    r_h, rmse_h, n_h = _pair_metrics(g_h, a_h)
    axes[0, 0].text(
        0.01,
        1.08,
        f"GTEx vs AHBA raw: r={r_raw:.3f}, rmse={rmse_raw:.3f}, n={n_raw:,}",
        transform=axes[0, 0].transAxes,
        fontsize=FONT["legend"],
        va="bottom",
    )
    axes[1, 0].text(
        0.01,
        -0.16,
        f"GTEx vs AHBA ComBat: r={r_h:.3f}, rmse={rmse_h:.3f}, n={n_h:,}",
        transform=axes[1, 0].transAxes,
        fontsize=FONT["legend"],
        va="top",
    )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    cbar.set_label("Expression value")
    return fig, axes
