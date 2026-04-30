#!/usr/bin/env python3
from __future__ import annotations

"""
Population-level cache-backed model evaluation helpers.

This module is the new import surface for `eval_population.ipynb`. During
the first refactor pass, implementations are delegated to `results_eda.py` so
the notebooks can stop importing the monolithic module while behavior remains
stable. New population-level evaluation functionality should be added here.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re
import textwrap
import warnings
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, ttest_rel
try:
    from IPython.display import display as _display
except Exception:  # pragma: no cover
    _display = None

from .eda_core import EDAConfig, load_eval_gene_list, prepare_pre_post_harmonization
from .eval_style import (
    FONT,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    build_parcel_label_table,
    format_legend_label,
    model_label,
    ordered_models,
    parcel_color_map,
    parcel_group_sort_key,
    parcel_label_lookup,
    set_academic_style,
    strip_display_label_prefixes,
)
from .results_eda import (
    collect_pooled_sample_prediction_dfs_from_cache_gene_subset,
    compute_fold_combo_metrics_from_cache,
    compute_subject_metrics_from_cache,
    compute_subject_metrics_from_cache_gene_subset,
    plot_coverage_vs_accuracy,
    plot_fold_combo_ranked,
    plot_heldout_region_grouped_bars,
    plot_loro_subject_summary_bars as _legacy_plot_loro_subject_summary_bars,
    summarize_heldout_region_performance,
)


__all__ = [
    "EDAConfig",
    "FONT",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "compute_fold_combo_metrics_from_cache",
    "compute_subject_metrics_from_cache",
    "compute_subject_metrics_from_cache_gene_subset",
    "collect_pooled_sample_prediction_dfs_from_cache_gene_subset",
    "build_global_prediction_tables",
    "compute_prediction_metrics",
    "format_gene_list_name",
    "make_prediction_eval_view",
    "paired_ttests_by_subject",
    "plot_coverage_vs_metric",
    "plot_global_prediction_scatter",
    "plot_global_true_pred_scatter_triplet",
    "plot_metric_delta_violins",
    "plot_metric_violins",
    "plot_region_model_metric_heatmap",
    "plot_stratified_scatter",
    "plot_stratified_distribution",
    "format_stratified_metric_table",
    "plot_distance_to_train_vs_metric",
    "compute_stratum_bias",
    "plot_stratum_bias_forest",
    "compute_subject_fold_summary",
    "plot_subject_mean_metric",
    "plot_fold_std_vs_mean",
    "compute_subject_specificity",
    "plot_subject_specificity",
    "plot_coverage_vs_accuracy",
    "plot_fold_combo_ranked",
    "plot_fold_combo_matched_overlay",
    "plot_fold_combo_ranked_overlay",
    "plot_heldout_region_grouped_bars",
    "plot_loro_subject_summary_bars",
    "prepare_pre_post_harmonization",
    "run_subject_metric_panel",
    "select_subjects_by_metric_percentile",
    "set_academic_style",
    "summarize_heldout_region_performance",
]


_META_COLS = {
    "subject",
    "age",
    "sex",
    "subject_region_key",
    "parcel_idx",
    "sample_key",
    "sample_idx_subject",
    "model",
    "eval_gene_mode",
    "eval_gene_path",
    "mixed_space_mode",
    "gtex_region",
    "ahba_region",
    "region_group",
    "gtex_coordinates_raw",
    "gtex_rep_x",
    "gtex_rep_y",
    "gtex_rep_z",
    "ahba_mapped_x",
    "ahba_mapped_y",
    "ahba_mapped_z",
    "mapping_distance",
    "n_samples_subject_region",
    "subject_coverage",
    "model",
}


_SCATTER_REGION_GROUP_ORDER = ["cortical", "subcortical", "cerebellar", "other"]
_SCATTER_REGION_GROUP_COLORS = {
    "cortical": ["#b35806", "#e08214", "#f1a340", "#fdb863", "#7f3b08"],
    "subcortical": ["#2166ac", "#4393c3", "#92c5de", "#762a83", "#9970ab", "#c2a5cf"],
    "subcortical_basal_ganglia": ["#762a83", "#9970ab", "#c2a5cf", "#40004b", "#8e0152"],
    "subcortical_other": ["#2166ac", "#4393c3", "#92c5de", "#053061", "#67a9cf"],
    "cerebellar": ["#1b7837", "#5aae61", "#a6dba0", "#00441b", "#7fbf7b"],
    "other": ["#6b6b6b", "#969696", "#bdbdbd", "#525252"],
}
_SCATTER_REGION_GROUP_BASE = {
    "cortical": "#e08214",
    "subcortical": "#2166ac",
    "cerebellar": "#1b7837",
    "other": "#6b6b6b",
}
_SCATTER_SEX_COLORS = {
    "female": "#f4a261",
    "f": "#f4a261",
    "male": "#7fc97f",
    "m": "#7fc97f",
}
_AGE_ORDER_RE = re.compile(r"\d+")


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


def _r2_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    x = x[m]
    y = y[m]
    ss_res = float(np.sum((x - y) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _gene_cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) not in _META_COLS]


def _resolve_eval_view_genes(
    available_genes: Sequence[str],
    genes: Sequence[str] | None = None,
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    eval_gene_path: str | None = None,
) -> list[str]:
    available = [str(g) for g in available_genes]
    available_set = set(available)
    path = eval_gene_list_path if eval_gene_list_path is not None else eval_gene_path
    if genes is not None:
        requested = [str(g) for g in genes]
    elif path is not None and str(path).strip():
        requested = load_eval_gene_list(str(path))
    else:
        mode = str(gene_mode).strip().lower()
        if mode in {"all", "allgenes", "all_genes"}:
            return available
        if mode != "custom":
            raise ValueError("gene_mode must be one of: allgenes, custom")
        if custom_gene_list is None:
            raise ValueError("custom gene_mode requires custom_gene_list or eval_gene_list_path")
        requested = [str(g) for g in custom_gene_list]
    keep = [str(g) for g in requested if str(g) in available_set]
    if not keep:
        src = eval_gene_list_path if eval_gene_list_path is not None else eval_gene_path
        label = f" from {src}" if src is not None else ""
        raise ValueError(f"Requested gene subset{label} has no overlap with cached prediction table genes")
    return keep


def format_gene_list_name(gene_list_path: str | Path | None, default: str = "All Genes") -> str:
    """Format a gene-list path or basename for plot labels."""
    if gene_list_path is None or not str(gene_list_path).strip():
        return str(default)
    stem = Path(str(gene_list_path)).stem.lower()
    known = {
        "syngo": "SynGO genes",
        "richiardi2015": "Richiardi (2015) genes",
    }
    if stem in known:
        return known[stem]

    parts = stem.split("_")
    source_map = {"ahba": "AHBA", "gtex": "GTEx"}
    source = source_map.get(parts[0], parts[0].upper() if parts else "")
    rest = parts[1:] if parts and parts[0] in source_map else parts
    labels = []
    for part in rest:
        low = str(part).lower()
        if low.endswith("hvg") and low[:-3].isdigit():
            labels.append(f"{low[:-3]} HVG")
        elif low.endswith("deg") and low[:-3].isdigit():
            labels.append(f"{low[:-3]} DEG")
        elif low in {"hvg", "deg"}:
            labels.append(low.upper())
        elif low == "demeaned":
            labels.append("demeaned")
        elif low:
            labels.append(format_legend_label(low))
    return " ".join([source, *labels]).strip() or str(default)


def _normalize_region_filter_label(value: object) -> str:
    return strip_display_label_prefixes(str(value)).strip().lower()


def _normalize_stratify_by(stratify_by: str | None, columns: Sequence[str]) -> str | None:
    if stratify_by is None:
        return None
    raw = str(stratify_by)
    aliases = {
        "region": "gtex_region",
        "regions": "gtex_region",
        "tissue": "gtex_region",
        "tissues": "gtex_region",
        "parcel": "parcel_idx",
        "parcels": "parcel_idx",
    }
    col = aliases.get(raw.strip().lower(), raw)
    if col not in set(map(str, columns)):
        raise KeyError(f"stratify_by={stratify_by!r} is not a column in the eval view")
    return col


def _collapse_region_group(macro_system: object) -> str:
    s = str(macro_system).strip().lower()
    if s == "cerebellar":
        return "cerebellar"
    if s == "subcortical":
        return "subcortical"
    return "cortical"


def _default_eval_table_cache_dir(cfg: EDAConfig, cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        p = Path(cache_dir)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return (Path.cwd() / "out" / "eval_prediction_tables" / str(cfg.gene_scope).lower()).resolve()


def _parquet_paths(cache_dir: Path, models: Sequence[str]) -> tuple[Path, Dict[str, Path], Path]:
    truth_path = cache_dir / "truth.parquet"
    pred_paths = {str(m).lower(): cache_dir / f"pred_{str(m).lower()}.parquet" for m in models}
    manifest_path = cache_dir / "manifest.json"
    return truth_path, pred_paths, manifest_path


def _normalize_cache_permissions(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    def _repair(path: Path) -> None:
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | 0o700)
            for child in path.iterdir():
                _repair(child)
        else:
            path.chmod(mode | 0o600)

    try:
        _repair(cache_dir)
    except PermissionError as e:
        raise PermissionError(
            f"Could not repair permissions under eval table cache directory: {cache_dir}. "
            "If these files are not owned by the notebook user, remove the cache or pass cache_dir=... "
            "to a writable directory."
        ) from e


def _read_global_prediction_tables(cache_dir: Path, models: Sequence[str]) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict[str, object]]:
    _normalize_cache_permissions(cache_dir)
    truth_path, pred_paths, manifest_path = _parquet_paths(cache_dir, models)
    missing = [str(truth_path), str(manifest_path)] + [str(p) for p in pred_paths.values() if not p.exists()]
    missing = [p for p in missing if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached eval table files: {missing}")
    truth_df = pd.read_parquet(truth_path)
    pred_dfs = {m: pd.read_parquet(path) for m, path in pred_paths.items()}
    manifest = json.loads(manifest_path.read_text())
    return truth_df, pred_dfs, manifest


def _write_global_prediction_tables(
    cache_dir: Path,
    truth_df: pd.DataFrame,
    pred_dfs: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, object],
) -> None:
    to_create = [p for p in cache_dir.parents if p.exists()][0]
    parts = cache_dir.relative_to(to_create).parts
    cur = to_create
    cur.chmod(cur.stat().st_mode | 0o700)
    for part in parts:
        cur = cur / part
        cur.mkdir(mode=0o755, exist_ok=True)
        cur.chmod(cur.stat().st_mode | 0o700)
    truth_path, pred_paths, manifest_path = _parquet_paths(cache_dir, pred_dfs.keys())
    probe_path = cache_dir / ".write_probe"
    try:
        probe_path.write_text("ok")
        probe_path.unlink()
    except OSError as e:
        raise PermissionError(
            f"Eval table cache directory is not writable: {cache_dir}. "
            "Pass cache_dir=... to build_global_prediction_tables(...) or repair directory permissions."
        ) from e
    def _write_parquet_bytes(df: pd.DataFrame, path: Path) -> None:
        # Avoid pyarrow opening the filesystem path directly. On this HPC/NFS
        # setup pyarrow can fail open_output_stream even when Python can write.
        data = df.to_parquet(None, index=False)
        path.write_bytes(data)

    try:
        _write_parquet_bytes(truth_df, truth_path)
        for model, df in pred_dfs.items():
            _write_parquet_bytes(df, pred_paths[str(model).lower()])
    except ImportError as e:
        raise ImportError("Parquet caching requires pyarrow or fastparquet in the notebook environment") from e
    except PermissionError as e:
        raise PermissionError(
            f"Parquet writer could not open output files under {cache_dir}. "
            "The directory passed the Python write probe, so this is likely a filesystem/pyarrow "
            "permission interaction or a stale non-writable directory in the notebook kernel. "
            "Try reloading eval_population, repairing permissions recursively, or pass a different cache_dir."
        ) from e
    manifest_path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True, default=str))
    _normalize_cache_permissions(cache_dir)


def _subject_region_metadata(prepost: Dict[str, object]) -> pd.DataFrame:
    g = prepost["gtex_eligible_raw"].copy()
    target = prepost["target_meta"].copy()
    target["parcel_idx"] = target["parcel_idx"].astype(int)

    def _first_nonnull(s: pd.Series) -> object:
        vals = s.dropna()
        return vals.iloc[0] if len(vals) else np.nan

    coord_col = "coordinates" if "coordinates" in g.columns else None
    agg_spec = {
        "age": ("age", _first_nonnull),
        "sex": ("sex", _first_nonnull),
        "gtex_region": ("tissue_or_parcel", _first_nonnull),
        "gtex_rep_x": ("coord_x", "mean"),
        "gtex_rep_y": ("coord_y", "mean"),
        "gtex_rep_z": ("coord_z", "mean"),
        "mapping_distance": ("mapping_distance", "mean"),
        "n_samples_subject_region": ("parcel_idx", "size"),
    }
    if coord_col is not None:
        agg_spec["gtex_coordinates_raw"] = (coord_col, lambda s: " | ".join(sorted(set(str(x) for x in s.dropna().tolist()))))
    meta = g.groupby(["subject", "parcel_idx"], as_index=False).agg(**agg_spec)
    if "gtex_coordinates_raw" not in meta.columns:
        meta["gtex_coordinates_raw"] = ""

    tcols = ["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "macro_system"]
    target_meta = target[tcols].rename(
        columns={
            "tissue_or_parcel": "ahba_region",
            "coord_x": "ahba_mapped_x",
            "coord_y": "ahba_mapped_y",
            "coord_z": "ahba_mapped_z",
        }
    )
    meta = meta.merge(target_meta, on="parcel_idx", how="left")
    meta["region_group"] = meta["macro_system"].map(_collapse_region_group)
    meta = meta.drop(columns=["macro_system"])
    meta["subject"] = meta["subject"].astype(str)
    meta["parcel_idx"] = meta["parcel_idx"].astype(int)
    meta["subject_region_key"] = meta["subject"].astype(str) + "|" + meta["parcel_idx"].astype(str)
    cov = meta.groupby("subject")["parcel_idx"].nunique().rename("subject_coverage").reset_index()
    meta = meta.merge(cov, on="subject", how="left")
    return meta


def _attach_eval_metadata(df: pd.DataFrame, meta: pd.DataFrame, expression_space: str, model: str | None = None) -> pd.DataFrame:
    genes = _gene_cols(df)
    base_cols = ["subject", "parcel_idx"]
    out = df[base_cols + genes].copy()
    out["subject"] = out["subject"].astype(str)
    out["parcel_idx"] = out["parcel_idx"].astype(int)
    out = out.merge(meta, on=["subject", "parcel_idx"], how="left")
    preferred = [
        "subject",
        "age",
        "sex",
        "subject_region_key",
        "parcel_idx",
        "gtex_region",
        "ahba_region",
        "region_group",
        "gtex_coordinates_raw",
        "gtex_rep_x",
        "gtex_rep_y",
        "gtex_rep_z",
        "ahba_mapped_x",
        "ahba_mapped_y",
        "ahba_mapped_z",
        "mapping_distance",
        "n_samples_subject_region",
        "subject_coverage",
    ]
    return out[preferred + genes]


def _manifest_for_global_tables(
    cfg: EDAConfig,
    prepost: Dict[str, object],
    models: Sequence[str],
    truth_df: pd.DataFrame,
    pred_dfs: Mapping[str, pd.DataFrame],
    cache_dir: Path,
) -> dict[str, object]:
    genes = _gene_cols(truth_df)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_kind": "global_prediction_tables",
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "source_cache_root": str(cfg.cache_root),
        "gene_scope": str(cfg.gene_scope),
        "models": [str(m).lower() for m in models],
        "model_cache_dirnames": {
            "naive": str(cfg.naive_cache_dirname),
            "dlam": str(cfg.dlam_cache_dirname),
            "plam": str(cfg.plam_cache_dirname),
        },
        "csv_path": str(cfg.csv_path),
        "hvg_path": str(cfg.hvg_path),
        "truth_expression_space": "harmonized",
        "prediction_expression_space": "harmonized",
        "gtex_rep_mode": str(cfg.gtex_rep_mode),
        "gtex_hemi_mode": str(cfg.gtex_hemi_mode),
        "n_rows": int(len(truth_df)),
        "n_genes": int(len(genes)),
        "n_subjects": int(truth_df["subject"].astype(str).nunique()),
        "n_parcels": int(truth_df["parcel_idx"].nunique()),
        "genes": genes,
        "prepost_subject_count": int(len(prepost.get("subjects", []))),
    }


def build_global_prediction_tables(
    cfg: EDAConfig,
    prepost: Dict[str, object],
    models: Sequence[str] | None = None,
    cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict[str, object]]:
    """Build/load all-gene wide truth and prediction tables for population eval."""
    model_list = ordered_models(models if models is not None else MODEL_ORDER)
    cdir = _default_eval_table_cache_dir(cfg, cache_dir=cache_dir)
    if bool(cache) and not bool(force_rebuild):
        try:
            return _read_global_prediction_tables(cdir, model_list)
        except FileNotFoundError:
            pass

    meta = _subject_region_metadata(prepost)
    truth_ref: pd.DataFrame | None = None
    pred_dfs: Dict[str, pd.DataFrame] = {}
    truth_keys: pd.Series | None = None
    for model in model_list:
        truth_raw, pred_raw = collect_pooled_sample_prediction_dfs_from_cache_gene_subset(
            cfg,
            model=model,
            eval_gene_mode="all",
            eval_gene_path=None,
            n_jobs=n_jobs,
        )
        truth = _attach_eval_metadata(truth_raw, meta, expression_space="harmonized")
        pred = _attach_eval_metadata(pred_raw, meta, expression_space="harmonized", model=model)
        keys = truth["subject_region_key"].astype(str)
        if truth_ref is None:
            truth_ref = truth
            truth_keys = keys
        else:
            if truth_keys is None or not truth_keys.equals(keys):
                raise ValueError(f"Truth row keys for model={model} are not aligned with the reference truth table")
        if not keys.equals(pred["subject_region_key"].astype(str)):
            raise ValueError(f"Prediction row keys for model={model} are not aligned with truth")
        pred_dfs[model] = pred

    if truth_ref is None:
        raise RuntimeError("No model prediction tables were collected")
    manifest = _manifest_for_global_tables(cfg, prepost, model_list, truth_ref, pred_dfs, cdir)
    if bool(cache):
        _write_global_prediction_tables(cdir, truth_ref, pred_dfs, manifest)
    return truth_ref, pred_dfs, manifest


def _top_ids(ids: np.ndarray, n_top: int) -> list[int]:
    if int(n_top) <= 0:
        return []
    vals, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    return [int(vals[i]) for i in order[: int(n_top)].tolist()]


def _ordered_top_ids(
    ids: np.ndarray,
    n_top: int,
    labels: Dict[int, str] | None = None,
) -> list[int]:
    tops = _top_ids(ids, int(n_top))
    if labels is None:
        return tops
    return sorted(tops, key=lambda tid: (*parcel_group_sort_key(labels.get(int(tid), str(tid))), int(tid)))


def _axis_identity_limits(
    arrays: Sequence[np.ndarray],
    quantiles: Tuple[float, float] | None = None,
) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if int(vals.size) == 0:
        return -1.0, 1.0
    if quantiles is None:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    else:
        qlo, qhi = quantiles
        if not (0.0 <= float(qlo) < float(qhi) <= 1.0):
            raise ValueError("axis_limit_quantiles must be a pair within [0, 1] with low < high")
        lo, hi = [float(v) for v in np.nanquantile(vals, [float(qlo), float(qhi)])]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    pad = 0.05 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _nice_interval_ticks(lo: float, hi: float, n_ticks: int = 6) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.linspace(-1.0, 1.0, int(n_ticks))
    raw_step = (hi - lo) / max(int(n_ticks) - 1, 1)
    exponent = np.floor(np.log10(raw_step))
    base = raw_step / (10.0 ** exponent)
    if base <= 1.0:
        nice_base = 1.0
    elif base <= 2.0:
        nice_base = 2.0
    elif base <= 5.0:
        nice_base = 5.0
    else:
        nice_base = 10.0
    step = nice_base * (10.0 ** exponent)
    start = np.floor(lo / step) * step
    stop = np.ceil(hi / step) * step
    ticks = np.arange(start, stop + 0.5 * step, step)
    if len(ticks) > int(n_ticks) + 3:
        ticks = np.linspace(start, stop, int(n_ticks))
    return ticks


def _global_scatter_payload(truth_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    genes = _gene_cols(truth_df)
    if genes != _gene_cols(pred_df):
        raise ValueError("truth_df and pred_df have different gene columns")
    if not truth_df["sample_key"].astype(str).equals(pred_df["sample_key"].astype(str)):
        raise ValueError("truth_df and pred_df sample keys are not aligned")

    x_mat = truth_df[genes].to_numpy(dtype=np.float64)
    y_mat = pred_df[genes].to_numpy(dtype=np.float64)
    n_samples, n_genes = x_mat.shape
    parcel_ids = np.repeat(truth_df["parcel_idx"].to_numpy(dtype=np.int32), n_genes)
    gene_ids = np.tile(np.arange(n_genes, dtype=np.int32), n_samples)
    x = x_mat.ravel()
    y = y_mat.ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    return {
        "x": x[finite],
        "y": y[finite],
        "parcel_ids": parcel_ids[finite],
        "gene_ids": gene_ids[finite],
        "genes": np.asarray(genes, dtype=object),
    }


def _sample_payload(
    payload: Dict[str, np.ndarray],
    max_points: int | None,
    random_seed: int,
) -> Dict[str, np.ndarray]:
    if max_points is None or int(max_points) <= 0:
        return payload
    n = int(len(payload["x"]))
    if n <= int(max_points):
        return payload
    rng = np.random.default_rng(int(random_seed))
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    out = dict(payload)
    for key, val in payload.items():
        arr = np.asarray(val)
        if arr.shape[:1] == (n,):
            out[key] = arr[idx]
    return out


def _scatter_stratified(
    ax: plt.Axes,
    payload: Dict[str, np.ndarray],
    color_by: str,
    top_n: int,
    model_color: str,
    point_size: float,
    alpha: float,
    rasterized: bool,
    parcel_labels: Dict[int, str] | None = None,
    parcel_colors: Dict[int, str] | None = None,
    show_legend_handles: bool = True,
) -> list[Line2D]:
    mode = str(color_by).lower()
    x = payload["x"]
    y = payload["y"]
    if mode == "none":
        ax.scatter(x, y, s=point_size, alpha=alpha, color=model_color, linewidths=0, rasterized=bool(rasterized))
        return []

    if mode not in {"parcel", "gene"}:
        raise ValueError("scatter_color_by must be one of: none, parcel, gene")
    ids = payload["parcel_ids"] if mode == "parcel" else payload["gene_ids"]
    label_map = parcel_labels if mode == "parcel" else None
    tops = _ordered_top_ids(ids, int(top_n), labels=label_map)
    base = np.isin(ids, tops, invert=True)
    ax.scatter(
        x[base],
        y[base],
        s=max(0.4, point_size * 0.55),
        alpha=alpha,
        color="#9a9a9a",
        linewidths=0,
        rasterized=bool(rasterized),
    )

    handles: list[Line2D] = []
    palette = sns.color_palette("tab20", n_colors=max(1, len(tops)))
    for c, tid in zip(palette, tops):
        m = ids == tid
        if mode == "parcel" and parcel_colors is not None and int(tid) in parcel_colors:
            c = parcel_colors[int(tid)]
        ax.scatter(x[m], y[m], s=point_size, alpha=alpha, color=c, linewidths=0, rasterized=bool(rasterized))
        if mode == "gene":
            genes = payload["genes"]
            label = str(genes[int(tid)]) if int(tid) < len(genes) else f"gene {tid}"
        else:
            label = (parcel_labels or {}).get(int(tid), f"parcel {tid}")
        if bool(show_legend_handles):
            handles.append(Line2D([0], [0], marker="o", linestyle="none", color=c, label=format_legend_label(label), markersize=6))
    return handles


def _scatter_hexbin(
    ax: plt.Axes,
    payload: Dict[str, np.ndarray],
    gridsize: int = 90,
    mincnt: int = 1,
    cmap: str = "viridis",
    bins: str = "log",
    rasterized: bool = True,
):
    hb = ax.hexbin(
        payload["x"],
        payload["y"],
        gridsize=int(gridsize),
        mincnt=int(mincnt),
        cmap=str(cmap),
        bins=bins,
        linewidths=0,
        rasterized=bool(rasterized),
    )
    return hb


def plot_global_true_pred_scatter_triplet(
    cfg: EDAConfig,
    models: Sequence[str] | None = None,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    prepost: Dict[str, object] | None = None,
    parcel_label_mode: str = "gtex",
    scatter_color_by: str = "parcel",
    scatter_top_n: int = 10,
    scatter_legend: bool = True,
    scatter_kind: str = "points",
    n_jobs: int | None = None,
    point_size: float | None = None,
    alpha: float | None = None,
    max_points_per_model: int | None = 300_000,
    random_seed: int = 0,
    rasterized: bool = True,
    axis_limits: Tuple[float, float] | None = None,
    axis_limit_quantiles: Tuple[float, float] | None = (0.05, 0.995),
    hexbin_gridsize: int = 90,
    hexbin_mincnt: int = 1,
    hexbin_cmap: str = "viridis",
    hexbin_bins: str = "log",
    figsize: Tuple[float, float] = (17.8, 6.4),
    dpi: int = 220,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    model_list = ordered_models(models if models is not None else MODEL_ORDER)
    scatter_kind_l = str(scatter_kind).lower()
    if scatter_kind_l not in {"points", "hexbin"}:
        raise ValueError("scatter_kind must be one of: points, hexbin")
    frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    payloads: Dict[str, Dict[str, np.ndarray]] = {}
    plot_payloads: Dict[str, Dict[str, np.ndarray]] = {}

    for mi, model in enumerate(model_list):
        truth_df, pred_df = collect_pooled_sample_prediction_dfs_from_cache_gene_subset(
            cfg,
            model=model,
            eval_gene_mode=eval_gene_mode,
            custom_gene_list=custom_gene_list,
            eval_gene_path=eval_gene_path,
            n_jobs=n_jobs,
        )
        frames[model] = (truth_df, pred_df)
        payloads[model] = _global_scatter_payload(truth_df, pred_df)
        plot_payloads[model] = _sample_payload(payloads[model], max_points=max_points_per_model, random_seed=int(random_seed) + mi)

    if axis_limits is None:
        lim_lo, lim_hi = _axis_identity_limits(
            [p["x"] for p in payloads.values()] + [p["y"] for p in payloads.values()],
            quantiles=axis_limit_quantiles,
        )
    else:
        lim_lo, lim_hi = float(axis_limits[0]), float(axis_limits[1])
    n_genes = len(next(iter(payloads.values()))["genes"]) if payloads else 0
    ps = float(point_size) if point_size is not None else (2.0 if n_genes > 500 else 9.0)
    al = float(alpha) if alpha is not None else 1.0

    fig, axes = plt.subplots(1, len(model_list), figsize=figsize, dpi=int(dpi), constrained_layout=False, squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.18, top=0.84, wspace=0.34)
    panel_label = str(eval_gene_path) if eval_gene_path is not None else ("All genes" if str(eval_gene_mode).lower() == "all" else str(eval_gene_mode))
    panel_label = format_legend_label(panel_label)
    parcel_labels: Dict[int, str] = {}
    parcel_colors: Dict[int, str] = {}
    if prepost is not None and str(scatter_color_by).lower() == "parcel":
        parcel_df = build_parcel_label_table(prepost, eligible_only=True)
        parcel_labels = parcel_label_lookup(parcel_df, label_mode=parcel_label_mode)
        parcel_colors = parcel_color_map(parcel_df, palette="functional")

    for ax, model in zip(axes, model_list):
        p = payloads[model]
        p_plot = plot_payloads[model]
        if scatter_kind_l == "hexbin":
            hb = _scatter_hexbin(
                ax,
                p_plot,
                gridsize=hexbin_gridsize,
                mincnt=hexbin_mincnt,
                cmap=hexbin_cmap,
                bins=hexbin_bins,
                rasterized=rasterized,
            )
            handles = []
            cbar = fig.colorbar(hb, ax=ax, shrink=0.80, pad=0.015)
            cbar.set_label("Point density", fontsize=FONT["small"] + 1)
            cbar.ax.tick_params(labelsize=FONT["small"])
        else:
            handles = _scatter_stratified(
                ax,
                p_plot,
                color_by=scatter_color_by,
                top_n=scatter_top_n,
                model_color=MODEL_COLORS.get(model, "#4f4f4f"),
                point_size=ps,
                alpha=al,
                rasterized=rasterized,
                parcel_labels=parcel_labels,
                parcel_colors=parcel_colors,
                show_legend_handles=(bool(scatter_legend) and ax is axes[-1]),
            )
        x = p["x"]
        y = p["y"]
        r = _pearson_safe(x, y)
        r2 = _r2_safe(x, y)
        rmse = _rmse_safe(x, y)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#252525", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ticks = _nice_interval_ticks(lim_lo, lim_hi, n_ticks=6)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        tick_labels = [f"{t:g}" for t in ticks]
        ax.set_xticklabels(tick_labels, visible=True)
        ax.set_yticklabels(tick_labels, visible=True)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelbottom=True,
            labelleft=True,
            labeltop=False,
            labelright=False,
            labelsize=FONT["tick"] + 1,
        )
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_title(f"{model_label(model)} vs Truth", fontsize=FONT["title"] + 2)
        ax.set_xlabel("Held-Out Truth", fontsize=FONT["label"] + 1)
        ax.set_ylabel("Prediction", fontsize=FONT["label"] + 1)
        ax.grid(True, alpha=0.16)
        ax.text(
            0.035,
            0.965,
            f"r={r:.3f}\nR2={r2:.3f}\nRMSE={rmse:.3f}\nn={len(x):,}\nshown={len(p_plot['x']):,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT["small"] + 2,
            bbox={"facecolor": "white", "edgecolor": "#7f7f7f", "alpha": 0.92, "boxstyle": "round,pad=0.28"},
        )
        if handles:
            fig.legend(
                handles=handles,
                title="Region" if str(scatter_color_by).lower() == "parcel" else format_legend_label(str(scatter_color_by)),
                loc="center left",
                bbox_to_anchor=(0.855, 0.50),
                frameon=True,
                fancybox=False,
                edgecolor="#4a4a4a",
                facecolor="white",
                framealpha=0.96,
                fontsize=FONT["small"] + 1,
                title_fontsize=FONT["small"] + 1,
            )

    fig.suptitle(f"Global Held-Out Truth vs Prediction ({panel_label})", fontsize=FONT["title"] + 4, y=0.98)
    return fig, axes, frames


def make_prediction_eval_view(
    truth_df: pd.DataFrame,
    pred_dfs: Mapping[str, pd.DataFrame],
    genes: Sequence[str] | None = None,
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    eval_gene_path: str | None = None,
    models: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    regions: Sequence[str | int] | None = None,
    region_groups: Sequence[str] | None = None,
    sex: Sequence[str] | str | None = None,
    age: Sequence[str] | str | None = None,
    tissues: Sequence[str] | str | None = None,
) -> dict[str, object]:
    """Create an aligned, filtered view over wide truth/prediction tables."""
    model_list = ordered_models(models if models is not None else pred_dfs.keys())
    missing = [m for m in model_list if m not in pred_dfs]
    if missing:
        raise KeyError(f"Missing prediction tables for models: {missing}")
    gene_list = _resolve_eval_view_genes(
        _gene_cols(truth_df),
        genes=genes,
        gene_mode=gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_list_path=eval_gene_list_path,
        eval_gene_path=eval_gene_path,
    )
    missing_genes = [g for g in gene_list if g not in truth_df.columns]
    if missing_genes:
        raise KeyError(f"Genes missing from truth_df: {missing_genes[:8]}")

    mask = pd.Series(True, index=truth_df.index)
    if subjects is not None:
        subject_set = {str(s) for s in subjects}
        mask &= truth_df["subject"].astype(str).isin(subject_set)
    if tissues is not None:
        tissue_vals = [tissues] if isinstance(tissues, str) else list(tissues)
        tissue_set = {_normalize_region_filter_label(t) for t in tissue_vals}
        gtex_region_norm = truth_df["gtex_region"].map(_normalize_region_filter_label)
        tissue_mask = gtex_region_norm.isin(tissue_set)
        if not bool(tissue_mask.any()):
            available = sorted(pd.unique(gtex_region_norm.dropna()))[:20]
            raise ValueError(
                "tissues filter matched no gtex_region rows. "
                f"Requested={sorted(tissue_set)}; available examples={available}"
            )
        mask &= tissue_mask
    if regions is not None:
        region_set = {str(r).lower() for r in regions}
        mask &= (
            truth_df["parcel_idx"].astype(str).str.lower().isin(region_set)
            | truth_df["gtex_region"].astype(str).str.lower().isin(region_set)
            | truth_df["ahba_region"].astype(str).str.lower().isin(region_set)
        )
    if region_groups is not None:
        group_set = {str(g).lower() for g in region_groups}
        mask &= truth_df["region_group"].astype(str).str.lower().isin(group_set)
    if sex is not None:
        sex_vals = [sex] if isinstance(sex, str) else list(sex)
        sex_set = {str(s).lower() for s in sex_vals}
        mask &= truth_df["sex"].astype(str).str.lower().isin(sex_set)
    if age is not None:
        age_vals = [age] if isinstance(age, str) else list(age)
        age_set = {str(a).lower() for a in age_vals}
        mask &= truth_df["age"].astype(str).str.lower().isin(age_set)

    truth_view = truth_df.loc[mask].reset_index(drop=True)
    if len(truth_view) == 0:
        raise RuntimeError("No rows remain after prediction eval view filters")

    key = truth_view["subject_region_key"].astype(str).reset_index(drop=True)
    pred_views: Dict[str, pd.DataFrame] = {}
    for model in model_list:
        p = pred_dfs[model]
        p_view = p.loc[mask].reset_index(drop=True)
        if not key.equals(p_view["subject_region_key"].astype(str).reset_index(drop=True)):
            raise ValueError(f"Prediction rows for model={model} are not aligned with truth view")
        pred_views[model] = p_view

    return {
        "truth_df": truth_view,
        "pred_dfs": pred_views,
        "genes": gene_list,
        "models": model_list,
        "gene_mode": str(gene_mode),
        "eval_gene_list_path": eval_gene_list_path if eval_gene_list_path is not None else eval_gene_path,
        "filters": {
            "subjects": None if subjects is None else list(subjects),
            "tissues": None if tissues is None else ([tissues] if isinstance(tissues, str) else list(tissues)),
            "regions": None if regions is None else list(regions),
            "region_groups": None if region_groups is None else list(region_groups),
            "sex": sex,
            "age": age,
        },
    }


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xr = pd.Series(x[m]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[m]).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_safe(xr, yr)


def _metric_row_values(x: np.ndarray, y: np.ndarray, metrics: Sequence[str]) -> dict[str, float | int]:
    m = np.isfinite(x) & np.isfinite(y)
    xv = x[m]
    yv = y[m]
    out: dict[str, float | int] = {"n_points": int(m.sum())}
    for metric in metrics:
        ml = str(metric).lower()
        if ml in {"pearson", "pearson_r"}:
            out["pearson_r"] = _pearson_safe(xv, yv)
        elif ml in {"spearman", "spearman_r"}:
            out["spearman_r"] = _spearman_safe(xv, yv)
        elif ml == "r2":
            out["r2"] = _r2_safe(xv, yv)
        elif ml == "rmse":
            out["rmse"] = _rmse_safe(xv, yv)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return out


def _normalize_metric_names(metrics: Sequence[str]) -> list[str]:
    out: list[str] = []
    for metric in metrics:
        ml = str(metric).lower()
        if ml in {"pearson", "pearson_r"}:
            name = "pearson_r"
        elif ml in {"spearman", "spearman_r"}:
            name = "spearman_r"
        elif ml in {"r2", "rmse"}:
            name = ml
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if name not in out:
            out.append(name)
    return out


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    n = finite.sum(axis=1).astype(np.float64)
    x0 = np.where(finite, x, 0.0)
    y0 = np.where(finite, y, 0.0)
    sx = x0.sum(axis=1)
    sy = y0.sum(axis=1)
    sxx = (x0 * x0).sum(axis=1)
    syy = (y0 * y0).sum(axis=1)
    sxy = (x0 * y0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - (sx * sy / n)
        vx = sxx - (sx * sx / n)
        vy = syy - (sy * sy / n)
        denom = np.sqrt(vx * vy)
        out = cov / denom
    out[(n < 2) | ~np.isfinite(out)] = np.nan
    return out


def _rowwise_rmse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    n = finite.sum(axis=1).astype(np.float64)
    diff2 = np.where(finite, (x - y) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(diff2.sum(axis=1) / n)
    out[(n < 1) | ~np.isfinite(out)] = np.nan
    return out


def _rowwise_r2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    n = finite.sum(axis=1).astype(np.float64)
    x0 = np.where(finite, x, 0.0)
    y0 = np.where(finite, y, 0.0)
    sx = x0.sum(axis=1)
    sxx = (x0 * x0).sum(axis=1)
    ss_res = np.where(finite, (x - y) ** 2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ss_tot = sxx - (sx * sx / n)
        out = 1.0 - ss_res / ss_tot
    out[(n < 2) | (ss_tot <= 0) | ~np.isfinite(out)] = np.nan
    return out


def _rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if bool(finite.all()):
        return _rowwise_pearson(
            rankdata(x, axis=1, method="average"),
            rankdata(y, axis=1, method="average"),
        )
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    for i in range(x.shape[0]):
        m = finite[i]
        if int(m.sum()) >= 2:
            out[i] = _pearson_safe(
                rankdata(x[i, m], method="average"),
                rankdata(y[i, m], method="average"),
            )
    return out


def _metric_block(args: tuple[np.ndarray, np.ndarray, tuple[str, ...]]) -> dict[str, np.ndarray]:
    x, y, metrics = args
    out: dict[str, np.ndarray] = {"n_points": (np.isfinite(x) & np.isfinite(y)).sum(axis=1).astype(np.int32)}
    if "pearson_r" in metrics:
        out["pearson_r"] = _rowwise_pearson(x, y)
    if "spearman_r" in metrics:
        out["spearman_r"] = _rowwise_spearman(x, y)
    if "r2" in metrics:
        out["r2"] = _rowwise_r2(x, y)
    if "rmse" in metrics:
        out["rmse"] = _rowwise_rmse(x, y)
    return out


def _compute_metric_arrays(
    x: np.ndarray,
    y: np.ndarray,
    metrics: Sequence[str],
    n_jobs: int | None = None,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    metric_names = tuple(_normalize_metric_names(metrics))
    n_rows = int(x.shape[0])
    if n_rows == 0:
        return {"n_points": np.asarray([], dtype=np.int32), **{m: np.asarray([], dtype=np.float64) for m in metric_names}}
    chunk = max(1, int(chunk_size))
    slices = [slice(i, min(i + chunk, n_rows)) for i in range(0, n_rows, chunk)]
    jobs = 1 if n_jobs is None else max(1, int(n_jobs))
    if jobs == 1 or len(slices) == 1:
        blocks = [_metric_block((x[s], y[s], metric_names)) for s in slices]
    else:
        jobs = min(jobs, len(slices))
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            blocks = list(pool.map(_metric_block, [(x[s], y[s], metric_names) for s in slices]))
    keys = ["n_points", *metric_names]
    return {k: np.concatenate([b[k] for b in blocks]) for k in keys}


def compute_prediction_metrics(
    view: Mapping[str, object],
    unit: str = "sample",
    stratify_by: str | None = None,
    metrics: Sequence[str] = ("pearson_r", "spearman_r", "r2", "rmse"),
    genes: Sequence[str] | None = None,
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    eval_gene_path: str | None = None,
    n_jobs: int | None = None,
    chunk_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute prediction metrics from an eval view with explicit unit/aggregation provenance."""
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    genes = _resolve_eval_view_genes(
        list(view["genes"]),
        genes=genes,
        gene_mode=gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_list_path=eval_gene_list_path,
        eval_gene_path=eval_gene_path,
    )
    models = ordered_models(view["models"])
    if not isinstance(truth_df, pd.DataFrame) or not isinstance(pred_dfs, Mapping):
        raise TypeError("view must come from make_prediction_eval_view")
    unit_l = str(unit).lower()
    if unit_l not in {"sample", "gene", "global_flat"}:
        raise ValueError("unit must be one of: sample, gene, global_flat")

    meta_cols = [
        "subject",
        "age",
        "sex",
        "subject_region_key",
        "parcel_idx",
        "gtex_region",
        "ahba_region",
        "region_group",
        "subject_coverage",
    ]
    metric_names = _normalize_metric_names(metrics)
    rows = []
    x_mat = truth_df[genes].to_numpy(dtype=np.float64)
    strat_col = _normalize_stratify_by(stratify_by, truth_df.columns)
    for model in models:
        pred_df = pred_dfs[model]
        y_mat = pred_df[genes].to_numpy(dtype=np.float64)
        if unit_l == "sample":
            d = truth_df[[c for c in meta_cols if c in truth_df.columns]].copy()
            d["model"] = str(model)
            d["unit"] = "sample"
            d["stratify_by"] = "global" if strat_col is None else strat_col
            d["stratum"] = "global" if strat_col is None else truth_df[strat_col].astype(str).to_numpy()
            d["n_genes"] = int(len(genes))
            arrays = _compute_metric_arrays(x_mat, y_mat, metric_names, n_jobs=n_jobs, chunk_size=chunk_size)
            for key, values in arrays.items():
                d[key] = values
            rows.append(d)
        elif unit_l == "gene":
            d = pd.DataFrame(
                {
                    "model": str(model),
                    "gene": [str(g) for g in genes],
                    "unit": "gene",
                    "stratify_by": "global" if strat_col is None else strat_col,
                    "stratum": "global",
                    "n_samples": int(x_mat.shape[0]),
                    "n_genes": 1,
                }
            )
            arrays = _compute_metric_arrays(x_mat.T, y_mat.T, metric_names, n_jobs=n_jobs, chunk_size=chunk_size)
            for key, values in arrays.items():
                d[key] = values
            rows.append(d)
        else:
            row = {
                "model": str(model),
                "unit": "global_flat",
                "stratify_by": "global",
                "stratum": "global",
                "n_samples": int(x_mat.shape[0]),
                "n_genes": int(len(genes)),
            }
            row.update(_metric_row_values(x_mat.ravel(), y_mat.ravel(), metrics))
            rows.append(row)

    metric_df = pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    metric_cols = [c for c in ["pearson_r", "spearman_r", "r2", "rmse"] if c in metric_df.columns]
    group_cols = ["model", "unit", "stratify_by", "stratum"]
    summary_rows = []
    for keys, d in metric_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for metric in metric_cols:
            vals = pd.to_numeric(d[metric], errors="coerce").dropna()
            summary_rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "sem": float(vals.sem(ddof=1)) if len(vals) > 1 else np.nan,
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "n_units": int(len(vals)),
                    "n_samples": int(d["subject_region_key"].nunique()) if "subject_region_key" in d.columns else int(d.get("n_samples", pd.Series([np.nan])).max()),
                    "n_genes": int(d.get("n_genes", pd.Series([len(genes)])).max()),
                    "aggregation": "mean_over_units",
                    "flattening": unit_l,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return metric_df, summary_df


def _scatter_payload_from_view(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    genes: Sequence[str],
) -> Dict[str, np.ndarray]:
    x_mat = truth_df[list(genes)].to_numpy(dtype=np.float64)
    y_mat = pred_df[list(genes)].to_numpy(dtype=np.float64)
    n_samples, n_genes = x_mat.shape
    x = x_mat.ravel()
    y = y_mat.ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    return {
        "x": x[finite],
        "y": y[finite],
        "region_group": np.repeat(truth_df["region_group"].astype(str).to_numpy(), n_genes)[finite],
        "region": np.repeat(truth_df["gtex_region"].astype(str).to_numpy(), n_genes)[finite],
        "subject": np.repeat(truth_df["subject"].astype(str).to_numpy(), n_genes)[finite],
        "sex": np.repeat(truth_df["sex"].astype(str).to_numpy(), n_genes)[finite],
        "age": np.repeat(truth_df["age"].astype(str).to_numpy(), n_genes)[finite],
        "gene": np.tile(np.asarray(list(genes), dtype=object), n_samples)[finite],
    }


def _sample_scatter_payload_from_view(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    genes: Sequence[str],
    max_points: int | None,
    random_seed: int,
    balance_by: str | None = None,
) -> tuple[Dict[str, np.ndarray], int]:
    gene_list = list(genes)
    x_mat = truth_df[gene_list].to_numpy(dtype=np.float64)
    y_mat = pred_df[gene_list].to_numpy(dtype=np.float64)
    n_samples, n_genes = x_mat.shape
    n_total = int(n_samples * n_genes)
    x_flat = x_mat.ravel()
    y_flat = y_mat.ravel()
    rng = np.random.default_rng(int(random_seed))
    if balance_by is None:
        if max_points is None or int(max_points) <= 0 or int(max_points) >= n_total:
            flat_idx = np.arange(n_total, dtype=np.int64)
        else:
            flat_idx = np.sort(rng.choice(n_total, size=int(max_points), replace=False))
    else:
        finite_idx = np.flatnonzero(np.isfinite(x_flat) & np.isfinite(y_flat)).astype(np.int64)
        labels = _scatter_flat_labels(truth_df, gene_list, finite_idx, n_genes, balance_by)
        flat_idx = _balanced_sample_flat_indices(
            finite_idx,
            labels,
            max_points=max_points,
            random_seed=int(random_seed),
        )
    row_idx = flat_idx // n_genes
    gene_idx = flat_idx % n_genes
    x = x_flat[flat_idx]
    y = y_flat[flat_idx]
    finite = np.isfinite(x) & np.isfinite(y)
    row_idx = row_idx[finite]
    gene_idx = gene_idx[finite]
    return (
        {
            "x": x[finite],
            "y": y[finite],
            "region_group": truth_df["region_group"].astype(str).to_numpy()[row_idx],
            "region": truth_df["gtex_region"].astype(str).to_numpy()[row_idx],
            "subject": truth_df["subject"].astype(str).to_numpy()[row_idx],
            "sex": truth_df["sex"].astype(str).to_numpy()[row_idx],
            "age": truth_df["age"].astype(str).to_numpy()[row_idx],
            "gene": np.asarray(gene_list, dtype=object)[gene_idx],
        },
        n_total,
    )


def _scatter_flat_labels(
    truth_df: pd.DataFrame,
    genes: Sequence[str],
    flat_idx: np.ndarray,
    n_genes: int,
    key: str,
) -> np.ndarray:
    row_idx = flat_idx // int(n_genes)
    if key == "gene":
        gene_idx = flat_idx % int(n_genes)
        return np.asarray(list(genes), dtype=object)[gene_idx].astype(str)
    if key == "region_group":
        col = "region_group"
    elif key == "region":
        col = "gtex_region"
    elif key in {"subject", "sex", "age"}:
        col = key
    else:
        raise ValueError(f"Cannot balance scatter sampling by unsupported key: {key}")
    return truth_df[col].astype(str).to_numpy()[row_idx]


def _balanced_sample_flat_indices(
    flat_idx: np.ndarray,
    labels: np.ndarray,
    max_points: int | None,
    random_seed: int,
) -> np.ndarray:
    """Balance candidate flat indices by label, then optionally cap evenly again."""
    if len(flat_idx) == 0:
        return flat_idx
    rng = np.random.default_rng(int(random_seed))
    label_s = pd.Series(labels.astype(str))
    groups = [idx.to_numpy(dtype=np.int64) for _, idx in label_s.groupby(label_s, sort=True).groups.items()]
    groups = [g for g in groups if len(g) > 0]
    if not groups:
        return flat_idx
    min_n = min(len(g) for g in groups)
    balanced_parts = []
    for positions in groups:
        take = positions if len(positions) == min_n else rng.choice(positions, size=min_n, replace=False)
        balanced_parts.append(flat_idx[take])

    cap = None if max_points is None or int(max_points) <= 0 else int(max_points)
    if cap is not None and sum(len(p) for p in balanced_parts) > cap:
        n_groups = len(balanced_parts)
        if cap < n_groups:
            keep_group_idx = set(rng.choice(np.arange(n_groups), size=cap, replace=False).tolist())
            balanced_parts = [p for i, p in enumerate(balanced_parts) if i in keep_group_idx]
            per_group = 1
            extras = 0
        else:
            per_group = cap // n_groups
            extras = cap % n_groups
        capped_parts = []
        extra_groups = set(rng.choice(np.arange(len(balanced_parts)), size=extras, replace=False).tolist()) if extras else set()
        for i, part in enumerate(balanced_parts):
            n_take = min(len(part), per_group + (1 if i in extra_groups else 0))
            capped_parts.append(part if len(part) == n_take else rng.choice(part, size=n_take, replace=False))
        balanced_parts = capped_parts

    out = np.concatenate(balanced_parts).astype(np.int64)
    rng.shuffle(out)
    return out


def _scatter_color_key(color_by: str) -> str | None:
    mode = str(color_by).lower()
    if mode in {"none", ""}:
        return None
    if mode == "region_group":
        return "region_group"
    if mode in {"region", "gtex_region"}:
        return "region"
    if mode in {"subject", "gene", "sex", "age"}:
        return mode
    raise ValueError("color_by must be one of: none, region_group, region, subject, gene, sex, age")


def _scatter_subcortical_palette_key(region: str) -> str:
    s = str(region).lower()
    if any(k in s for k in ["basal ganglia", "caudate", "putamen", "accumbens", "nucleus accumbens"]):
        return "subcortical_basal_ganglia"
    return "subcortical_other"


def _age_sort_key(value: object) -> tuple[int, float, str]:
    s = str(value).strip()
    match = _AGE_ORDER_RE.search(s)
    if match:
        return 0, float(match.group(0)), s.lower()
    return 1, float("inf"), s.lower()


def _global_scatter_color_spec(
    plot_payloads: Mapping[str, Dict[str, np.ndarray]],
    color_by: str,
    top_n: int,
    genes: Sequence[str] | None = None,
) -> tuple[str | None, list[str], dict[str, object]]:
    key = _scatter_color_key(color_by)
    if key is None:
        return None, [], {}
    if key == "region_group":
        order = _SCATTER_REGION_GROUP_ORDER
        palette = _SCATTER_REGION_GROUP_BASE
        present = set()
        for payload in plot_payloads.values():
            present.update(str(v) for v in payload[key].astype(str))
        return key, [v for v in order if v in present], palette

    if key == "region":
        region_ids = np.concatenate([payload["region"].astype(str) for payload in plot_payloads.values()])
        group_ids = np.concatenate([payload["region_group"].astype(str) for payload in plot_payloads.values()])
        counts = pd.Series(region_ids).value_counts()
        top_regions = set(counts.index[: int(top_n)].tolist())
        region_groups = (
            pd.DataFrame({"region": region_ids, "region_group": group_ids})
            .drop_duplicates()
            .groupby("region")["region_group"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        order = []
        palette = {}
        for group in _SCATTER_REGION_GROUP_ORDER:
            group_regions = [
                region
                for region in counts.index.tolist()
                if region in top_regions and region_groups.get(region, "other") == group
            ]
            group_regions = sorted(group_regions, key=lambda r: (-int(counts.loc[r]), format_legend_label(r)))
            counters: dict[str, int] = {}
            for i, region in enumerate(group_regions):
                palette_key = _scatter_subcortical_palette_key(region) if group == "subcortical" else group
                colors = _SCATTER_REGION_GROUP_COLORS.get(palette_key, _SCATTER_REGION_GROUP_COLORS["other"])
                j = counters.get(palette_key, 0)
                order.append(region)
                palette[region] = colors[j % len(colors)]
                counters[palette_key] = j + 1
        return key, order, palette

    if key == "gene":
        if genes is None:
            ids = np.concatenate([payload[key].astype(str) for payload in plot_payloads.values()])
            order = pd.unique(ids).tolist()[: int(top_n)]
        else:
            order = [str(g) for g in list(genes)[: int(top_n)]]
        colors = sns.color_palette("tab20", n_colors=max(1, len(order)))
        palette = {v: colors[i] for i, v in enumerate(order)}
        return key, order, palette

    ids = np.concatenate([payload[key].astype(str) for payload in plot_payloads.values()])
    vals, counts = np.unique(ids, return_counts=True)
    order = vals[np.argsort(-counts, kind="mergesort")[: int(top_n)]].tolist()
    if key == "sex":
        fallback_colors = sns.color_palette("Set2", n_colors=max(1, len(order)))
        palette = {
            v: _SCATTER_SEX_COLORS.get(str(v).strip().lower(), fallback_colors[i])
            for i, v in enumerate(order)
        }
        return key, order, palette
    if key == "age":
        order = sorted(order, key=_age_sort_key)
        palette_name = "tab10" if len(order) <= 10 else "tab20"
        colors = sns.color_palette(palette_name, n_colors=max(1, len(order)))
        palette = {v: colors[i] for i, v in enumerate(order)}
        return key, order, palette
    colors = sns.color_palette("tab20", n_colors=max(1, len(order)))
    palette = {v: colors[i] for i, v in enumerate(order)}
    return key, order, palette


def _scatter_legend_label(value: str, color_key: str | None) -> str:
    if color_key == "gene":
        return str(value).upper()
    return format_legend_label(value)


def _scatter_legend_title(color_by: str, color_key: str | None, n_shown: int, gene_list_label: str | None = None) -> str:
    label = gene_list_label if gene_list_label is not None and str(gene_list_label).strip() else "All Genes"
    if color_key == "gene":
        return f"Top {int(n_shown)} Genes From {label}"
    if str(color_by).lower() == "region_group":
        base = "Region Group"
    else:
        base = format_legend_label(color_by)
    return base


def _scatter_legend_handles(
    order: Sequence[str],
    palette: Mapping[str, object],
    show_other: bool,
    color_key: str | None,
) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=palette[val], label=_scatter_legend_label(val, color_key), markersize=6)
        for val in order
    ]
    if bool(show_other):
        handles.append(Line2D([0], [0], marker="o", linestyle="none", color="#9a9a9a", label="Other", markersize=6))
    return handles


def _plot_categorical_scatter(
    ax: plt.Axes,
    payload: Dict[str, np.ndarray],
    color_key: str | None,
    category_order: Sequence[str],
    category_palette: Mapping[str, object],
    point_size: float,
    alpha: float,
    rasterized: bool,
) -> bool:
    x = payload["x"]
    y = payload["y"]
    if color_key is None:
        ax.scatter(x, y, s=point_size, alpha=alpha, color="#5f5f5f", linewidths=0, rasterized=rasterized)
        return False

    ids = payload[color_key].astype(str)
    order = list(category_order)
    base = ~np.isin(ids, order)
    ax.scatter(x[base], y[base], s=max(0.4, point_size * 0.55), alpha=alpha, color="#9a9a9a", linewidths=0, rasterized=rasterized)
    for val in order:
        m = ids == val
        c = category_palette[val]
        ax.scatter(x[m], y[m], s=point_size, alpha=alpha, color=c, linewidths=0, rasterized=rasterized)
    return bool(base.any())


def plot_global_prediction_scatter(
    view: Mapping[str, object],
    genes: Sequence[str] | None = None,
    gene_mode: str = "allgenes",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_list_path: str | None = None,
    eval_gene_path: str | None = None,
    color_by: str = "region_group",
    top_n: int = 12,
    max_points_per_model: int | None = 100_000,
    balanced_sampling: bool = False,
    axis_limit_quantiles: Tuple[float, float] | None = (0.05, 0.995),
    point_size: float = 1.4,
    alpha: float = 0.45,
    rasterized: bool = True,
    random_seed: int = 0,
    figsize: Tuple[float, float] = (17.8, 6.4),
    dpi: int = 220,
) -> Tuple[plt.Figure, np.ndarray]:
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    gene_list_path = eval_gene_list_path if eval_gene_list_path is not None else eval_gene_path
    if gene_list_path is None:
        gene_list_path = view.get("eval_gene_list_path")
    genes = _resolve_eval_view_genes(
        list(view["genes"]),
        genes=genes,
        gene_mode=gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_list_path=gene_list_path,
        eval_gene_path=None,
    )
    models = ordered_models(view["models"])
    missing = [model for model in models if model not in pred_dfs]
    if missing:
        raise KeyError(f"Missing prediction tables for models: {missing}")
    color_key = _scatter_color_key(color_by)
    balance_by = color_key if bool(balanced_sampling) and color_key is not None else None
    plot_payloads = {}
    n_total_by_model = {}
    for i, model in enumerate(models):
        payload, n_total = _sample_scatter_payload_from_view(
            truth_df,
            pred_dfs[model],
            genes,
            max_points=max_points_per_model,
            random_seed=int(random_seed) + i,
            balance_by=balance_by,
        )
        plot_payloads[model] = payload
        n_total_by_model[model] = n_total

    color_key, category_order, category_palette = _global_scatter_color_spec(plot_payloads, color_by=color_by, top_n=top_n, genes=genes)
    lim_lo, lim_hi = _axis_identity_limits([p["x"] for p in plot_payloads.values()] + [p["y"] for p in plot_payloads.values()], quantiles=axis_limit_quantiles)
    fig, axes = plt.subplots(1, len(models), figsize=figsize, dpi=int(dpi), constrained_layout=False, squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.18, top=0.88, wspace=0.34)
    show_other_in_legend = False
    gene_list_label = format_gene_list_name(gene_list_path, default="All Genes" if custom_gene_list is None else "Custom Gene List")
    n_subjects = int(truth_df["subject"].nunique()) if "subject" in truth_df.columns else 0
    n_samples = int(len(truth_df))
    n_genes = int(len(genes))
    for ax, model in zip(axes, models):
        show_other_in_legend |= _plot_categorical_scatter(
            ax,
            plot_payloads[model],
            color_key=color_key,
            category_order=category_order,
            category_palette=category_palette,
            point_size=point_size,
            alpha=alpha,
            rasterized=rasterized,
        )
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#252525", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ticks = _nice_interval_ticks(lim_lo, lim_hi, n_ticks=6)
        labels = [f"{t:g}" for t in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="both", which="both", bottom=True, left=True, top=False, right=False, labelbottom=True, labelleft=True, labeltop=False, labelright=False, labelsize=FONT["tick"] + 2)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_title(f"{model_label(model)} vs Truth", fontsize=FONT["title"] + 3)
        ax.set_xlabel("Held-Out Truth", fontsize=FONT["label"] + 2)
        ax.set_ylabel("Prediction", fontsize=FONT["label"] + 2)
        ax.grid(True, alpha=0.16)
        if str(model).lower() == "naive":
            n_shown = int(len(plot_payloads[model]["x"]))
            n_total = int(n_total_by_model[model])
            balance_line = f"\nbalanced by {format_legend_label(balance_by)}" if balance_by is not None else ""
            ax.text(
                0.035,
                0.965,
                f"{n_subjects:,} subjects | {n_samples:,} samples\n{n_genes:,} genes ({gene_list_label})\nnum. points shown {n_shown:,} of {n_total:,}{balance_line}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=FONT["small"] + 3,
                bbox={"facecolor": "white", "edgecolor": "#7f7f7f", "alpha": 0.92, "boxstyle": "round,pad=0.28"},
            )
    handles = _scatter_legend_handles(category_order, category_palette, show_other=show_other_in_legend, color_key=color_key)
    if handles:
        title = _scatter_legend_title(color_by, color_key=color_key, n_shown=len(category_order), gene_list_label=gene_list_label)
        fig.legend(handles=handles, title=title, loc="center left", bbox_to_anchor=(0.855, 0.50), frameon=True, fancybox=False, edgecolor="#4a4a4a", facecolor="white", framealpha=0.96, fontsize=FONT["small"] + 2, title_fontsize=FONT["small"] + 2)
    fig.suptitle("Global Held-Out Truth vs Prediction", fontsize=FONT["title"] + 5, y=0.94)
    return fig, axes


def plot_metric_violins(
    metric_df: pd.DataFrame,
    metric: str = "pearson_r",
    x: str = "model",
    hue: str | None = None,
    order: Sequence[str] | None = None,
    figsize: Tuple[float, float] = (8.2, 4.8),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes]:
    d = metric_df.copy()
    d = d[d[metric].notna()].copy()
    if "model" in d.columns:
        d["model"] = d["model"].astype(str).str.lower()
    x_order = ordered_models(d[x].dropna().unique()) if x == "model" and order is None else order
    hue_order = ordered_models(d[hue].dropna().unique()) if hue == "model" else None
    palette = MODEL_COLORS if (x == "model" or hue == "model") else None
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    sns.violinplot(data=d, x=x, y=metric, hue=hue, order=x_order, hue_order=hue_order, cut=0, inner="quartile", palette=palette, ax=ax)
    sns.stripplot(data=d, x=x, y=metric, hue=hue, order=x_order, hue_order=hue_order, dodge=bool(hue), color="black" if hue is None else None, palette=palette if hue == "model" else None, alpha=0.18, size=1.8, ax=ax, legend=False)
    ax.set_title(f"{format_legend_label(metric)} Distribution", fontsize=FONT["title"] + 3)
    ax.set_xlabel(format_legend_label(x), fontsize=FONT["label"] + 1)
    ax.set_ylabel(format_legend_label(metric), fontsize=FONT["label"] + 1)
    if x == "model":
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([model_label(t.get_text()) for t in ax.get_xticklabels()])
    ax.grid(True, axis="y", alpha=0.18)
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [model_label(v) if hue == "model" else format_legend_label(v) for v in labels], title=format_legend_label(hue), frameon=True, fancybox=False)
    return fig, ax


_STRATIFY_TO_COLOR_BY = {
    "sex": "sex",
    "age": "age",
    "gtex_region": "region",
    "region_group": "region_group",
}


def _gtex_region_group_order(metric_df: pd.DataFrame, regions: Sequence[str]) -> list[str]:
    """Order gtex_region values by region_group (cortical → subcortical → cerebellar → other),
    breaking ties alphabetically. Falls back to plain alpha if region_group missing."""
    regions = [str(r) for r in regions]
    if "region_group" not in metric_df.columns:
        return sorted(regions, key=lambda v: str(v).lower())
    region_to_group = (
        metric_df[["gtex_region", "region_group"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .groupby("gtex_region")["region_group"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    group_rank = {g: i for i, g in enumerate(_SCATTER_REGION_GROUP_ORDER)}
    fallback_rank = len(_SCATTER_REGION_GROUP_ORDER)

    def key(region: str) -> tuple[int, str]:
        group = region_to_group.get(region, "other")
        return (group_rank.get(group, fallback_rank), str(region).lower())

    return sorted(regions, key=key)


def _stratum_color_by(strat_col: str) -> str:
    if strat_col not in _STRATIFY_TO_COLOR_BY:
        raise ValueError(
            f"stratify_by={strat_col!r} not supported for stratified scatter; "
            f"supported: {sorted(_STRATIFY_TO_COLOR_BY)}"
        )
    return _STRATIFY_TO_COLOR_BY[strat_col]


def _metric_short_label(metric: str) -> str:
    return {
        "pearson_r": "r",
        "spearman_r": "ρ",
        "r2": "R²",
        "rmse": "RMSE",
    }.get(str(metric).lower(), str(metric))


def _format_metric_value(mean: float, std: float | None) -> str:
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def _stratum_summary_lookup(
    summary_df: pd.DataFrame | None,
    model: str,
    metric: str,
) -> dict[str, tuple[float, float, int]]:
    """{stratum -> (mean, std, n_units)} for a single model/metric."""
    if summary_df is None or summary_df.empty:
        return {}
    d = summary_df[
        (summary_df["model"].astype(str).str.lower() == str(model).lower())
        & (summary_df["metric"] == metric)
    ]
    out: dict[str, tuple[float, float, int]] = {}
    for _, row in d.iterrows():
        mean = float(row["mean"]) if pd.notna(row["mean"]) else float("nan")
        std = float(row["std"]) if pd.notna(row["std"]) else float("nan")
        n = int(row["n_units"]) if pd.notna(row["n_units"]) else 0
        out[str(row["stratum"])] = (mean, std, n)
    return out


def plot_stratified_scatter(
    view: Mapping[str, object],
    stratify_by: str | None = None,
    default_metric: str = "pearson_r",
    max_points_per_model: int | None = 100_000,
    balanced_sampling: bool = True,
    axis_limit_quantiles: Tuple[float, float] | None = (0.05, 0.995),
    point_size: float = 1.4,
    alpha: float = 0.45,
    rasterized: bool = True,
    random_seed: int = 0,
    figsize: Tuple[float, float] = (17.8, 6.8),
    dpi: int = 220,
    top_n: int = 24,
    metric_summary: pd.DataFrame | None = None,
    n_jobs: int | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Three-panel held-out scatter; per-panel legend annotated with this model's
    metric per stratum. Sample-wise metric averaging:
      1) filter view to stratum
      2) per sample (subject_region_key) compute metric across genes
      3) aggregate mean/std across samples in stratum
    Balanced sampling affects only points shown on the scatter, never the metric.
    """
    truth_df = view["truth_df"]
    pred_dfs = view["pred_dfs"]
    genes = list(view["genes"])
    models = ordered_models(view["models"])
    missing = [m for m in models if m not in pred_dfs]
    if missing:
        raise KeyError(f"Missing prediction tables for models: {missing}")

    strat_col = (
        _normalize_stratify_by(stratify_by, truth_df.columns)
        if stratify_by is not None
        else None
    )
    color_by = _stratum_color_by(strat_col) if strat_col is not None else "none"
    color_key = _scatter_color_key(color_by)
    balance_by = color_key if (bool(balanced_sampling) and color_key is not None) else None

    plot_payloads: Dict[str, Dict[str, np.ndarray]] = {}
    n_total_by_model: Dict[str, int] = {}
    for i, model in enumerate(models):
        payload, n_total = _sample_scatter_payload_from_view(
            truth_df,
            pred_dfs[model],
            genes,
            max_points=max_points_per_model,
            random_seed=int(random_seed) + i,
            balance_by=balance_by,
        )
        plot_payloads[model] = payload
        n_total_by_model[model] = n_total

    color_key, category_order, category_palette = _global_scatter_color_spec(
        plot_payloads, color_by=color_by, top_n=top_n, genes=None,
    )

    if metric_summary is None:
        _, metric_summary = compute_prediction_metrics(
            view,
            unit="sample",
            stratify_by=strat_col,
            metrics=[default_metric],
            n_jobs=n_jobs,
        )

    lim_lo, lim_hi = _axis_identity_limits(
        [p["x"] for p in plot_payloads.values()] + [p["y"] for p in plot_payloads.values()],
        quantiles=axis_limit_quantiles,
    )
    fig, axes = plt.subplots(
        1, len(models), figsize=figsize, dpi=int(dpi),
        constrained_layout=False, squeeze=False,
    )
    axes = axes.ravel()
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.16, top=0.86, wspace=0.30)

    metric_short = _metric_short_label(default_metric)
    for ax, model in zip(axes, models):
        show_other = _plot_categorical_scatter(
            ax,
            plot_payloads[model],
            color_key=color_key,
            category_order=category_order,
            category_palette=category_palette,
            point_size=point_size,
            alpha=alpha,
            rasterized=rasterized,
        )
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#252525", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ticks = _nice_interval_ticks(lim_lo, lim_hi, n_ticks=6)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.tick_params(axis="both", labelsize=FONT["tick"] + 2)
        ax.set_title(f"{model_label(model)} vs Truth", fontsize=FONT["title"] + 2)
        ax.set_xlabel("Held-Out Truth", fontsize=FONT["label"] + 1)
        ax.set_ylabel("Prediction", fontsize=FONT["label"] + 1)
        ax.grid(True, alpha=0.16)

        lookup = _stratum_summary_lookup(metric_summary, model, default_metric)
        if color_key is None:
            mean, std, n_units = lookup.get("global", (float("nan"), float("nan"), 0))
            ax.text(
                0.04, 0.96,
                f"{metric_short}={_format_metric_value(mean, std)}\nn={n_units:,} samples",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT["small"] + 3,
                bbox={"facecolor": "white", "edgecolor": "#7f7f7f", "alpha": 0.92, "boxstyle": "round,pad=0.28"},
            )
        else:
            handles = []
            for val in category_order:
                base_label = _scatter_legend_label(val, color_key)
                stat = lookup.get(str(val))
                if stat is not None:
                    mean, std, n_units = stat
                    label = f"{base_label} ({metric_short}={_format_metric_value(mean, std)}, n={n_units})"
                else:
                    label = base_label
                handles.append(
                    Line2D([0], [0], marker="o", linestyle="none",
                           color=category_palette[val], label=label, markersize=6)
                )
            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper left",
                    frameon=True, fancybox=False,
                    edgecolor="#4a4a4a", facecolor="white", framealpha=0.92,
                    fontsize=FONT["small"] + 1,
                    title=format_legend_label(strat_col),
                    title_fontsize=FONT["small"] + 1,
                )

    title_label = format_legend_label(strat_col) if strat_col else "Global"
    fig.suptitle(
        f"Held-Out Truth vs Prediction — {'Stratified by ' + title_label if strat_col else 'Global'}",
        fontsize=FONT["title"] + 4, y=0.96,
    )
    return fig, axes, metric_summary


def plot_stratified_distribution(
    metric_df: pd.DataFrame,
    stratify_by: str | None = None,
    metric: str = "pearson_r",
    kind: str = "box",
    figsize: Tuple[float, float] | None = None,
    dpi: int = 180,
    showfliers: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-sample metric distribution per stratum category, hue=model.

    `kind` is 'box' (default — heavy-tail-friendly) or 'violin'. Uses sample-wise
    metric_df from compute_prediction_metrics(unit='sample', stratify_by=...).
    For stratify_by=None, x=model."""
    kind_l = str(kind).lower()
    if kind_l not in {"box", "violin"}:
        raise ValueError("kind must be one of: box, violin")

    d = metric_df.copy()
    d = d[d[metric].notna()].copy()
    d["model"] = d["model"].astype(str).str.lower()

    if stratify_by is None:
        x_col = "model"
        x_order = ordered_models(d[x_col].dropna().unique())
        hue = None
        hue_order = None
    else:
        x_col = _normalize_stratify_by(stratify_by, d.columns)
        raw_order = d[x_col].dropna().astype(str).unique().tolist()
        if x_col == "age":
            x_order = sorted(raw_order, key=_age_sort_key)
        elif x_col == "region_group":
            x_order = [v for v in _SCATTER_REGION_GROUP_ORDER if v in raw_order] + [
                v for v in sorted(raw_order) if v not in _SCATTER_REGION_GROUP_ORDER
            ]
        elif x_col == "gtex_region":
            x_order = _gtex_region_group_order(d, raw_order)
        else:
            x_order = sorted(raw_order)
        hue = "model"
        hue_order = ordered_models(d["model"].dropna().unique())

    if figsize is None:
        figsize = (max(7.0, 1.0 * len(x_order) + 4.5), 4.8) if stratify_by is not None else (8.2, 4.8)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    palette = MODEL_COLORS if (x_col == "model" or hue == "model") else None
    if kind_l == "box":
        sns.boxplot(
            data=d, x=x_col, y=metric, hue=hue,
            order=x_order, hue_order=hue_order,
            palette=palette, ax=ax,
            fliersize=2.0, linewidth=1.0, showfliers=bool(showfliers),
            width=0.85 if hue is None else 0.78,
        )
    else:
        sns.violinplot(
            data=d, x=x_col, y=metric, hue=hue,
            order=x_order, hue_order=hue_order,
            cut=0, inner="quartile", palette=palette, ax=ax,
        )

    title_axis = format_legend_label(x_col) if stratify_by is not None else "Model"
    ax.set_title(
        f"{format_legend_label(metric)} by {title_axis} (sample-wise)",
        fontsize=FONT["title"] + 3,
    )
    ax.set_xlabel(format_legend_label(x_col), fontsize=FONT["label"] + 1)
    ax.set_ylabel(format_legend_label(metric), fontsize=FONT["label"] + 1)
    if x_col == "gtex_region":
        ax.set_xticklabels(
            [format_legend_label(t.get_text()) for t in ax.get_xticklabels()],
            ha="right", rotation=30,
        )
    if x_col == "model":
        ax.set_xticklabels([model_label(t.get_text()) for t in ax.get_xticklabels()])
    ax.grid(True, axis="y", alpha=0.18)
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [model_label(v) for v in labels], title="Model", frameon=True, fancybox=False)
    return fig, ax


def format_stratified_metric_table(
    metric_df: pd.DataFrame,
    stratify_by: str | None = None,
    metrics: Sequence[str] = ("pearson_r", "r2", "rmse"),
) -> pd.DataFrame:
    """Wide table: rows=stratum, columns=MultiIndex(model, metric|n), cells='mean ± std' or n.
    Aggregation is mean/std over per-sample metrics within each stratum."""
    metric_names = _normalize_metric_names(metrics)
    df = metric_df.copy()
    df["_model"] = df["model"].astype(str).str.lower()
    if stratify_by is None:
        df["_strat"] = "global"
        ordering = ["global"]
        index_name = "stratum"
    else:
        strat_col = _normalize_stratify_by(stratify_by, df.columns)
        df["_strat"] = df[strat_col].astype(str)
        raw_order = df["_strat"].dropna().unique().tolist()
        if strat_col == "age":
            ordering = sorted(raw_order, key=_age_sort_key)
        elif strat_col == "region_group":
            ordering = [v for v in _SCATTER_REGION_GROUP_ORDER if v in raw_order] + [
                v for v in sorted(raw_order) if v not in _SCATTER_REGION_GROUP_ORDER
            ]
        elif strat_col == "gtex_region":
            ordering = _gtex_region_group_order(df, raw_order)
        else:
            ordering = sorted(raw_order)
        index_name = strat_col

    models_present = [m for m in MODEL_ORDER if m in df["_model"].unique().tolist()]
    columns: list[tuple[str, str]] = []
    for m in models_present:
        for mn in metric_names:
            columns.append((model_label(m), mn))
        columns.append((model_label(m), "n"))
    col_index = pd.MultiIndex.from_tuples(columns, names=["model", "metric"])

    rows = []
    for s in ordering:
        row = {}
        for m in models_present:
            d = df[(df["_strat"] == s) & (df["_model"] == m)]
            n = int(d["subject_region_key"].nunique()) if "subject_region_key" in d.columns and len(d) else int(len(d))
            row[(model_label(m), "n")] = n
            for mn in metric_names:
                vals = pd.to_numeric(d.get(mn, pd.Series(dtype=float)), errors="coerce").dropna()
                if len(vals) == 0:
                    row[(model_label(m), mn)] = "n/a"
                else:
                    mean = float(vals.mean())
                    std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
                    row[(model_label(m), mn)] = _format_metric_value(mean, std)
        rows.append([row[c] for c in columns])

    out = pd.DataFrame(rows, index=pd.Index(ordering, name=index_name), columns=col_index)
    return out


def _bias_outcome_transform(metric: str, transform: str = "auto"):
    name = transform
    if name == "auto":
        name = "fisher_z" if str(metric).lower() in {"pearson_r", "spearman_r"} else "identity"
    if name == "fisher_z":
        eps = 1e-6
        fwd = lambda v: np.arctanh(np.clip(np.asarray(v, dtype=np.float64), -1 + eps, 1 - eps))
        inv = lambda z: np.tanh(np.asarray(z, dtype=np.float64))
        return fwd, inv, "fisher_z"
    if name == "identity":
        return (lambda v: np.asarray(v, dtype=np.float64), lambda z: np.asarray(z, dtype=np.float64), "identity")
    raise ValueError("transform must be one of: auto, fisher_z, identity")


def compute_stratum_bias(
    metric_df: pd.DataFrame,
    axis: str = "sex",
    models: Sequence[str] | None = None,
    metric: str = "pearson_r",
    controls: Sequence[str] = ("age", "subject_coverage", "gtex_region"),
    method: str = "lmm",
    transform: str = "auto",
    reference_category: str | None = None,
    fdr_method: str = "fdr_bh",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Per-model bias test: does the per-sample metric differ across `axis` categories
    after controlling for confounders?

    Model (per genome model in `models`):
        outcome ~ C(axis, Treatment(ref)) + sum(C(cat_controls)) + numeric_controls
    Subjects contribute multiple rows (≤ 13 regions/subject), so non-independence is
    handled either by a subject random intercept (`method='lmm'`) or by cluster-robust
    standard errors on `subject` (`method='ols_cluster'`).

    The outcome is Fisher-z transformed when `metric` is a correlation, identity
    otherwise. Reported `adj_mean`/`delta_vs_ref` are back-transformed to the original
    metric scale; `delta_se` is on the transform scale.

    Returns a tidy DataFrame, one row per (model, category), with an extra
    `category='__omnibus__'` row per model carrying the Wald omnibus test of the axis.
    `p_fdr` is Benjamini-Hochberg corrected across all non-reference contrast rows.
    """
    try:
        import statsmodels.api as sm  # noqa: F401
        import statsmodels.formula.api as smf  # noqa: F401
        import patsy
        from statsmodels.stats.multitest import multipletests
        from scipy.stats import norm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_stratum_bias requires statsmodels and patsy. "
            "Install via `pip install statsmodels patsy`."
        ) from exc
    import statsmodels.api as sm

    if axis not in metric_df.columns:
        raise KeyError(f"axis={axis!r} not in metric_df columns")
    method_l = str(method).lower()
    if method_l not in {"lmm", "ols_cluster"}:
        raise ValueError("method must be one of: lmm, ols_cluster")

    needed = {"subject", "model", metric, axis, *controls}
    missing = needed - set(metric_df.columns)
    if missing:
        raise KeyError(f"metric_df missing columns: {sorted(missing)}")

    fwd, inv, transform_name = _bias_outcome_transform(metric, transform)
    df_all = metric_df[list(needed)].copy()
    df_all = df_all.dropna(subset=[metric, axis, "subject", *list(controls)])
    df_all["_y"] = fwd(df_all[metric].to_numpy(dtype=np.float64))
    df_all["model"] = df_all["model"].astype(str).str.lower()
    df_all[axis] = df_all[axis].astype(str).str.strip()

    cat_levels_global = sorted(
        df_all[axis].dropna().unique().tolist(),
        key=_age_sort_key if axis == "age" else (lambda v: str(v).lower()),
    )
    if reference_category is None:
        ref_global = cat_levels_global[0]
    else:
        ref_global = str(reference_category).strip()
        if ref_global not in cat_levels_global:
            raise ValueError(
                f"reference_category={ref_global!r} not in axis levels {cat_levels_global}"
            )

    numeric_controls = [c for c in controls if c in df_all.columns and pd.api.types.is_numeric_dtype(df_all[c])]
    cat_controls = [c for c in controls if c in df_all.columns and not pd.api.types.is_numeric_dtype(df_all[c])]
    rhs_terms = [f"C({axis}, Treatment(reference={ref_global!r}))"]
    rhs_terms += [f"C({c})" for c in cat_controls]
    rhs_terms += list(numeric_controls)
    formula = "_y ~ " + " + ".join(rhs_terms)

    z_crit = float(norm.ppf(0.5 + float(confidence) / 2.0))
    model_list = ordered_models(models if models is not None else df_all["model"].unique())

    rows: list[dict] = []
    for model in model_list:
        d = df_all[df_all["model"] == model].copy()
        cat_levels = [v for v in cat_levels_global if v in set(d[axis].unique())]
        if len(cat_levels) < 2:
            continue

        try:
            y_df, X_df = patsy.dmatrices(formula, data=d, return_type="dataframe")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"patsy failed to build design for model={model}: {exc}") from exc
        design_info = X_df.design_info

        method_used = method_l
        if method_l == "lmm":
            try:
                fit = sm.MixedLM(endog=y_df.values.ravel(), exog=X_df, groups=d["subject"].values).fit(
                    reml=True, method="lbfgs",
                )
                beta = pd.Series(fit.fe_params, index=X_df.columns)
                cov = pd.DataFrame(fit.cov_params().values[: len(beta), : len(beta)],
                                    index=X_df.columns, columns=X_df.columns)
            except Exception as exc:
                warnings.warn(
                    f"compute_stratum_bias: LMM fit failed for model={model!r} "
                    f"({type(exc).__name__}: {exc}); falling back to OLS + cluster-robust SEs.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fit = sm.OLS(y_df, X_df).fit(cov_type="cluster", cov_kwds={"groups": d["subject"].values})
                beta = fit.params
                cov = fit.cov_params()
                method_used = "ols_cluster"
        else:
            fit = sm.OLS(y_df, X_df).fit(cov_type="cluster", cov_kwds={"groups": d["subject"].values})
            beta = fit.params
            cov = fit.cov_params()

        x_means: dict[str, np.ndarray] = {}
        for k in cat_levels:
            d_k = d.copy()
            d_k[axis] = k
            Xk = patsy.build_design_matrices([design_info], d_k, return_type="dataframe")[0]
            Xk = Xk.reindex(columns=X_df.columns, fill_value=0.0)
            x_means[k] = Xk.mean(axis=0).to_numpy(dtype=np.float64)

        ref_xz = float(x_means[ref_global] @ beta.to_numpy(dtype=np.float64))
        single_contrast_stat: dict[str, float] = {}

        for k in cat_levels:
            xk = x_means[k]
            z_hat = float(xk @ beta.to_numpy(dtype=np.float64))
            var = float(xk @ cov.to_numpy(dtype=np.float64) @ xk)
            se_z = float(np.sqrt(max(var, 0.0)))
            n_k = int((d[axis] == k).sum())
            is_ref = bool(k == ref_global)
            row = {
                "axis": axis,
                "metric": metric,
                "model": model,
                "category": k,
                "is_reference": is_ref,
                "reference_category": ref_global,
                "n": n_k,
                "adj_mean": float(inv(z_hat)),
                "adj_mean_ci_lo": float(inv(z_hat - z_crit * se_z)),
                "adj_mean_ci_hi": float(inv(z_hat + z_crit * se_z)),
                "delta_vs_ref": np.nan,
                "delta_se": np.nan,
                "delta_ci_lo": np.nan,
                "delta_ci_hi": np.nan,
                "p_raw": np.nan,
                "transform": transform_name,
                "method": method_used,
            }
            if not is_ref:
                contrast = xk - x_means[ref_global]
                delta_z = float(contrast @ beta.to_numpy(dtype=np.float64))
                var_d = float(contrast @ cov.to_numpy(dtype=np.float64) @ contrast)
                se_d = float(np.sqrt(max(var_d, 0.0)))
                if se_d > 0 and np.isfinite(delta_z):
                    z_stat = delta_z / se_d
                    p = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
                    single_contrast_stat = {"chi2": float(z_stat ** 2), "p": p}
                else:
                    p = float("nan")
                row["delta_vs_ref"] = float(inv(ref_xz + delta_z) - inv(ref_xz))
                row["delta_se"] = se_d
                row["delta_ci_lo"] = float(inv(ref_xz + delta_z - z_crit * se_d) - inv(ref_xz))
                row["delta_ci_hi"] = float(inv(ref_xz + delta_z + z_crit * se_d) - inv(ref_xz))
                row["p_raw"] = p
            rows.append(row)

        axis_idx = [i for i, name in enumerate(X_df.columns) if str(name).startswith(f"C({axis}")]
        omnibus_chi2 = float("nan")
        omnibus_p = float("nan")
        if axis_idx:
            R = np.zeros((len(axis_idx), len(X_df.columns)), dtype=np.float64)
            for i, idx in enumerate(axis_idx):
                R[i, idx] = 1.0
            try:
                w = fit.wald_test(R, use_f=False)
                omnibus_chi2 = float(np.atleast_1d(np.asarray(w.statistic)).ravel()[0])
                omnibus_p = float(np.atleast_1d(np.asarray(w.pvalue)).ravel()[0])
            except Exception:
                pass
        # df=1 fallback: wald_test sometimes returns NaN under cluster-robust covariance
        # for a single restriction; the per-contrast Wald is the same test. Reuse it.
        if (
            len(axis_idx) == 1
            and (not np.isfinite(omnibus_p) or not np.isfinite(omnibus_chi2))
            and single_contrast_stat
        ):
            omnibus_chi2 = float(single_contrast_stat["chi2"])
            omnibus_p = float(single_contrast_stat["p"])
        rows.append({
            "axis": axis,
            "metric": metric,
            "model": model,
            "category": "__omnibus__",
            "is_reference": False,
            "reference_category": ref_global,
            "n": int(len(d)),
            "adj_mean": np.nan,
            "adj_mean_ci_lo": np.nan,
            "adj_mean_ci_hi": np.nan,
            "delta_vs_ref": np.nan,
            "delta_se": np.nan,
            "delta_ci_lo": np.nan,
            "delta_ci_hi": np.nan,
            "p_raw": omnibus_p,
            "transform": transform_name,
            "method": method_used,
            "omnibus_chi2": omnibus_chi2,
            "omnibus_df": int(len(axis_idx)),
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    contrast_mask = (~out["is_reference"]) & (out["category"] != "__omnibus__") & out["p_raw"].notna()
    out["p_fdr"] = np.nan
    if int(contrast_mask.sum()) > 0:
        _, p_fdr, _, _ = multipletests(
            out.loc[contrast_mask, "p_raw"].to_numpy(dtype=np.float64),
            method=fdr_method,
        )
        out.loc[contrast_mask, "p_fdr"] = p_fdr
    return out


def plot_stratum_bias_forest(
    bias_df: pd.DataFrame,
    figsize: Tuple[float, float] | None = None,
    dpi: int = 180,
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Forest plot of category contrasts (Δ vs reference) per model. One error bar per
    (model, non-reference category); zero line marks no bias. Omnibus per-model p
    is shown in a corner box."""
    if bias_df is None or len(bias_df) == 0:
        raise RuntimeError("bias_df is empty")
    d = bias_df[(bias_df["category"] != "__omnibus__") & (~bias_df["is_reference"])].copy()
    d = d[d["delta_vs_ref"].notna()].copy()
    if len(d) == 0:
        raise RuntimeError("No non-reference contrast rows to plot")

    axis = str(bias_df["axis"].iloc[0]) if "axis" in bias_df.columns else "category"
    metric = str(bias_df["metric"].iloc[0]) if "metric" in bias_df.columns else ""
    ref = str(bias_df["reference_category"].iloc[0]) if "reference_category" in bias_df.columns else ""
    method = str(bias_df["method"].iloc[0]) if "method" in bias_df.columns else ""
    transform = str(bias_df["transform"].iloc[0]) if "transform" in bias_df.columns else ""

    model_order = ordered_models(d["model"].unique())
    cat_order = (
        sorted(d["category"].unique(), key=_age_sort_key)
        if axis == "age"
        else sorted(d["category"].unique(), key=lambda v: str(v).lower())
    )
    rows: list[dict] = []
    for m in model_order:
        for c in cat_order:
            sub = d[(d["model"] == m) & (d["category"] == c)]
            if len(sub):
                rows.append(sub.iloc[0].to_dict())
    if not rows:
        raise RuntimeError("No rows after model/category ordering")

    if figsize is None:
        figsize = (9.0, max(2.6, 0.55 * len(rows) + 1.4))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))

    y_positions = np.arange(len(rows), 0, -1, dtype=np.float64)
    for y, row in zip(y_positions, rows):
        c = MODEL_COLORS.get(str(row["model"]), "#777777")
        delta = float(row["delta_vs_ref"])
        ax.errorbar(
            delta, y,
            xerr=[[delta - float(row["delta_ci_lo"])], [float(row["delta_ci_hi"]) - delta]],
            fmt="o", color=c, ecolor=c, markersize=6.5, linewidth=1.4, capsize=3.0,
        )

    ax.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([
        f"{model_label(str(r['model']))} — {format_legend_label(str(r['category']))}"
        for r in rows
    ])
    ax.set_xlabel(
        f"Δ {format_legend_label(metric)} vs {format_legend_label(ref)} "
        f"(adjusted, controlling confounders)",
        fontsize=FONT["label"] + 1,
    )
    ax.set_title(
        f"{format_legend_label(axis)} bias forest — {method.upper()} ({transform})",
        fontsize=FONT["title"] + 2,
    )
    ax.grid(True, axis="x", alpha=0.18)

    xs = (
        [float(r["delta_ci_lo"]) for r in rows]
        + [float(r["delta_ci_hi"]) for r in rows]
        + [0.0]
    )
    xlim_lo, xlim_hi = float(min(xs)), float(max(xs))
    span = max(xlim_hi - xlim_lo, 1e-6)
    right_pad = 0.55 if annotate else 0.10
    ax.set_xlim(xlim_lo - 0.10 * span, xlim_hi + right_pad * span)

    if annotate:
        for y, row in zip(y_positions, rows):
            p_fdr = row.get("p_fdr", np.nan)
            n = int(row.get("n", 0))
            delta = float(row["delta_vs_ref"])
            ci_hi = float(row["delta_ci_hi"])
            star = ""
            if pd.notna(p_fdr):
                if p_fdr < 0.001:
                    star = " ***"
                elif p_fdr < 0.01:
                    star = " **"
                elif p_fdr < 0.05:
                    star = " *"
            label = (
                f"  Δ={delta:+.3f} (95% CI [{float(row['delta_ci_lo']):+.3f}, {ci_hi:+.3f}]), "
                f"p_FDR={p_fdr:.3f}, n={n}{star}"
                if pd.notna(p_fdr)
                else f"  Δ={delta:+.3f}, n={n}"
            )
            ax.text(ci_hi, y, label, va="center", ha="left",
                    fontsize=FONT["small"], color="#333333")

    omnibus = bias_df[bias_df["category"] == "__omnibus__"]
    omnibus_lines: list[str] = []
    if len(omnibus):
        for m in model_order:
            r = omnibus[omnibus["model"] == m]
            if not len(r):
                continue
            p = float(r["p_raw"].iloc[0]) if pd.notna(r["p_raw"].iloc[0]) else float("nan")
            df_o = (
                int(r["omnibus_df"].iloc[0])
                if "omnibus_df" in r.columns and pd.notna(r["omnibus_df"].iloc[0])
                else 0
            )
            p_str = "n/a" if not np.isfinite(p) else f"{p:.3g}"
            omnibus_lines.append(f"{model_label(m)}: omnibus χ²(df={df_o}) p={p_str}")

    fig.tight_layout()
    if omnibus_lines:
        # Place the omnibus summary outside the data area, below the x-axis label,
        # so it can never overlap with the per-row contrast annotations.
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.26))
        fig.text(
            0.99, 0.02, "   |   ".join(omnibus_lines),
            ha="right", va="bottom",
            fontsize=FONT["small"],
            bbox={"facecolor": "white", "edgecolor": "#7f7f7f", "alpha": 0.92, "boxstyle": "round,pad=0.3"},
        )
    return fig, ax


def _row_z_normalize(X: np.ndarray) -> np.ndarray:
    """Z-normalize each row of X for use in vectorized Pearson via inner product.
    Rows with zero variance or non-finite entries get filled with zeros, which
    yields a Pearson r of 0 for those rows — defensible default when comparing
    a constant profile against anything."""
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True, ddof=0)
    sd_safe = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    Z = (X - mu) / sd_safe
    Z = np.where(np.isfinite(Z), Z, 0.0)
    Z = np.where(np.broadcast_to(sd > 0, Z.shape), Z, 0.0)
    return Z


def _hex_lighten(color, frac: float) -> tuple[float, float, float]:
    """Blend a color toward white by `frac` (0=no change, 1=white)."""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color), dtype=np.float64)
    return tuple((rgb + (1.0 - rgb) * float(frac)).tolist())


def compute_subject_specificity(
    view: Mapping[str, object],
    region_col: str = "parcel_idx",
) -> pd.DataFrame:
    """Per-(model, subject, region) self vs other-subject prediction similarity.

    For each LORO sample (subject `s`, region `p`, model `m`):
      - `sim_self`        = Pearson r between `pred[s,p,m]` and `truth[s,p]` over genes
      - `sim_other_mean`  = mean Pearson r between `pred[s,p,m]` and `truth[s',p]` for
                            every other subject s' that has region p

    Vectorized per region as a single (n_p × G) @ (G × n_p) matrix multiply per
    model-region pair. Returns a tidy DataFrame, one row per (model, sample),
    with `sim_self`, `sim_other_mean`, and `delta = sim_self - sim_other_mean`.
    """
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
    truth_z_per_region: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for region, idx in region_indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if len(idx_arr) < 2:
            continue
        Y = truth_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)
        truth_z_per_region[region] = (idx_arr, _row_z_normalize(Y))

    if not truth_z_per_region:
        raise RuntimeError(f"No region in {region_col!r} has ≥ 2 subjects")

    meta_cols = [
        c for c in ("subject", "subject_region_key", region_col, "gtex_region", "region_group")
        if c in truth_df.columns
    ]

    rows: list[pd.DataFrame] = []
    for model in models:
        pred_df = pred_dfs[model]
        if not pred_df.index.equals(truth_df.index):
            # Eval views guarantee aligned index but be defensive.
            if len(pred_df) != len(truth_df):
                raise ValueError(f"pred_df[{model}] not aligned to truth_df")
        for region, (idx_arr, Y_z) in truth_z_per_region.items():
            P = pred_df.iloc[idx_arr][genes].to_numpy(dtype=np.float64)
            P_z = _row_z_normalize(P)
            n = int(idx_arr.shape[0])
            M = (P_z @ Y_z.T) / float(G_total)
            self_sim = np.diag(M).astype(np.float64).copy()
            row_sum = M.sum(axis=1)
            other_mean = (row_sum - self_sim) / float(n - 1)

            meta = truth_df.iloc[idx_arr][meta_cols].copy().reset_index(drop=True)
            meta["model"] = str(model).lower()
            meta["sim_self"] = self_sim
            meta["sim_other_mean"] = other_mean
            meta["delta"] = self_sim - other_mean
            rows.append(meta)

    out = pd.concat(rows, ignore_index=True)
    return out


def plot_subject_specificity(
    spec_df: pd.DataFrame,
    figsize: Tuple[float, float] = (9.0, 5.0),
    dpi: int = 180,
    annotate_paired_test: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Split-violin per model: left half = sim_self, right half = sim_other_mean.

    Each model column is hued by `MODEL_COLORS`; the self half uses the saturated
    model color, the other half is blended ~55% toward white. Optional Wilcoxon
    paired test annotation per model.
    """
    if spec_df is None or len(spec_df) == 0:
        raise RuntimeError("spec_df is empty")

    long = spec_df.melt(
        id_vars=["model"],
        value_vars=["sim_self", "sim_other_mean"],
        var_name="condition",
        value_name="pearson_r",
    )
    long["model"] = long["model"].astype(str).str.lower()
    long["condition"] = long["condition"].map(
        {"sim_self": "self", "sim_other_mean": "other"}
    )
    long = long.dropna(subset=["pearson_r"]).copy()

    model_order = ordered_models(long["model"].unique())
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    sns.violinplot(
        data=long, x="model", y="pearson_r",
        hue="condition", hue_order=["self", "other"],
        split=True, order=model_order,
        palette={"self": "#888888", "other": "#cccccc"},
        inner="quartile", cut=0, ax=ax, linewidth=0.9,
    )

    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    poly_idx = 0
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection):
            continue
        if poly_idx >= 2 * len(model_order):
            break
        m = model_order[poly_idx // 2]
        cond = ["self", "other"][poly_idx % 2]
        base = MODEL_COLORS.get(m, "#777777")
        face = base if cond == "self" else _hex_lighten(base, 0.55)
        coll.set_facecolor(face)
        coll.set_edgecolor(base)
        coll.set_linewidth(0.9)
        poly_idx += 1

    handles: list = []
    for m in model_order:
        base = MODEL_COLORS.get(m, "#777777")
        light = _hex_lighten(base, 0.55)
        handles.append(Patch(facecolor=base, edgecolor=base, label=f"{model_label(m)} — Self"))
        handles.append(Patch(facecolor=light, edgecolor=base, label=f"{model_label(m)} — Other (mean)"))
    ax.legend(
        handles=handles, title="Prediction → Truth",
        frameon=True, fancybox=False, loc="lower right",
        fontsize=FONT["small"], title_fontsize=FONT["small"] + 1,
    )

    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels([model_label(m) for m in model_order], fontsize=FONT["tick"] + 1)
    ax.set_xlabel("Model", fontsize=FONT["label"] + 1)
    ax.set_ylabel("Pearson r (prediction → truth, gene-wise)", fontsize=FONT["label"] + 1)
    ax.set_title(
        "Subject Specificity: Prediction vs Self Truth and Mean Other-Subject Truth (per region)",
        fontsize=FONT["title"] + 2,
    )
    ax.grid(True, axis="y", alpha=0.18)

    if bool(annotate_paired_test):
        try:
            from scipy.stats import wilcoxon
        except Exception:
            wilcoxon = None
        if wilcoxon is not None:
            lines = []
            for m in model_order:
                d = spec_df[spec_df["model"].astype(str).str.lower() == m]
                d = d.dropna(subset=["sim_self", "sim_other_mean"])
                if len(d) < 5:
                    continue
                self_v = d["sim_self"].to_numpy(dtype=np.float64)
                other_v = d["sim_other_mean"].to_numpy(dtype=np.float64)
                med_delta = float(np.median(self_v - other_v))
                try:
                    w_stat, p = wilcoxon(self_v, other_v, alternative="greater")
                    p_val = float(p)
                except Exception:
                    p_val = float("nan")
                p_str = "n/a" if not np.isfinite(p_val) else f"{p_val:.2g}"
                lines.append(f"{model_label(m)}: median Δ={med_delta:+.3f}, Wilcoxon (self>other) p={p_str}")
            if lines:
                fig.tight_layout()
                fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.22))
                fig.text(
                    0.99, 0.02, "   |   ".join(lines),
                    ha="right", va="bottom",
                    fontsize=FONT["small"],
                    bbox={"facecolor": "white", "edgecolor": "#7f7f7f", "alpha": 0.92, "boxstyle": "round,pad=0.3"},
                )
                return fig, ax

    fig.tight_layout()
    return fig, ax


def compute_subject_fold_summary(
    fold_perf_df: pd.DataFrame,
    metric: str = "pearson_r",
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-(model, subject) summary of fold-level metric: mean, std, and fold count."""
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
    return summary


def plot_subject_mean_metric(
    fold_perf_df: pd.DataFrame,
    metric: str = "pearson_r",
    models: Sequence[str] | None = None,
    n_bottom_highlight: int | None = 5,
    figsize: Tuple[float, float] | None = None,
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Ranked per-subject mean metric across all LORO folds, per model.

    Subjects are ordered ascending by their across-model mean, so persistently
    bad subjects float to the left. Optional shaded band highlights the leftmost
    `n_bottom_highlight` subjects.
    """
    summary = compute_subject_fold_summary(fold_perf_df, metric=metric, models=models)
    if summary.empty:
        raise RuntimeError("subject summary is empty")

    overall = summary.groupby("subject")["mean"].mean().sort_values(ascending=True)
    subject_order = overall.index.tolist()
    pos = {s: i for i, s in enumerate(subject_order)}
    summary = summary.copy()
    summary["_x"] = summary["subject"].map(pos)
    model_order = ordered_models(summary["model"].unique())

    if figsize is None:
        figsize = (max(8.0, 0.16 * len(subject_order) + 4.0), 4.6)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))

    if n_bottom_highlight and n_bottom_highlight > 0:
        ax.axvspan(
            -0.5, min(int(n_bottom_highlight), len(subject_order)) - 0.5,
            color="#cc0000", alpha=0.07, zorder=0,
        )

    for m in model_order:
        dm = summary[summary["model"] == m]
        c = MODEL_COLORS.get(m, "#777777")
        ax.errorbar(
            dm["_x"], dm["mean"], yerr=dm["std"].fillna(0.0),
            fmt="o", color=c, ecolor=c,
            markersize=4.6, elinewidth=0.85, capsize=2.0,
            label=model_label(m), alpha=0.85,
        )

    ax.set_xticks(np.arange(len(subject_order)))
    ax.set_xticklabels(subject_order, rotation=80, ha="right", fontsize=FONT["tick"] - 1)
    ax.set_xlabel("Subject (ascending mean across models)", fontsize=FONT["label"])
    ax.set_ylabel(format_legend_label(metric), fontsize=FONT["label"] + 1)
    ax.set_title(
        f"Per-Subject Mean {format_legend_label(metric)} Across LORO Folds",
        fontsize=FONT["title"] + 2,
    )
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(title="Model", frameon=True, fancybox=False)
    fig.tight_layout()
    return fig, ax, summary


def plot_fold_std_vs_mean(
    combo_df: pd.DataFrame,
    metric: str = "mean_pearson",
    figsize: Tuple[float, float] = (8.6, 5.2),
    dpi: int = 180,
    bottom_quantile: float = 0.20,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-fold cross-subject std vs fold mean. Reads as:
      - bottom-left (low mean, low std): structurally hard fold (everyone bad);
      - bottom-right (high mean, low std): easy fold (everyone fine);
      - top-left  (low mean, high std): subject-mixing failure (some subjects drag down);
      - top-right (high mean, high std): mixed-difficulty fold.
    Vertical dotted lines mark each model's `bottom_quantile` threshold.
    """
    metric_to_std = {
        "mean_pearson": "std_pearson",
        "mean_spearman": "std_spearman",
        "mean_r2": "std_r2",
        "mean_rmse": "std_rmse",
    }
    std_col = metric_to_std.get(str(metric))
    if std_col is None or std_col not in combo_df.columns:
        raise ValueError(f"combo_df missing std column for metric={metric}")
    d = combo_df[["model", metric, std_col]].dropna().copy()
    d["model"] = d["model"].astype(str).str.lower()
    model_order = ordered_models(d["model"].unique())

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    for m in model_order:
        dm = d[d["model"] == m]
        c = MODEL_COLORS.get(m, "#777777")
        ax.scatter(
            dm[metric], dm[std_col],
            color=c, alpha=0.55, s=24, linewidths=0,
            label=model_label(m),
        )
    quantiles = d.groupby("model")[metric].quantile(float(bottom_quantile))
    for m in model_order:
        if m in quantiles.index:
            ax.axvline(
                float(quantiles.loc[m]),
                color=MODEL_COLORS.get(m, "#777777"),
                linestyle=":", linewidth=1.0, alpha=0.55,
            )

    ax.set_xlabel(_metric_axis_label(metric), fontsize=FONT["label"] + 1)
    ax.set_ylabel("Per-Fold Std Across Subjects", fontsize=FONT["label"] + 1)
    ax.set_title(
        f"Fold Difficulty Decomposition (mean vs cross-subject std; "
        f"dashed = {int(bottom_quantile * 100)}th-pct cutoff per model)",
        fontsize=FONT["title"] + 2,
    )
    ax.grid(True, alpha=0.18)
    ax.legend(title="Model", frameon=True, fancybox=False, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_distance_to_train_vs_metric(
    combo_df: pd.DataFrame,
    metric: str = "mean_pearson",
    distance_col: str = "dist_to_centroid_train",
    n_bins: int = 20,
    figsize: Tuple[float, float] = (9.4, 5.0),
    dpi: int = 180,
    point_alpha: float = 0.16,
    point_size: float = 9.0,
    show_points: bool = False,
    error_capsize: float = 4.0,
    error_linewidth: float = 1.6,
    y_pad_frac: float = 0.18,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Distance-to-training vs fold metric, summarized in quantile bins.

    Each bin shows a per-model marker at the bin mean with a thick capped errorbar
    (±SEM); markers are connected by a line per model. y-limits auto-zoom to the
    data range with a small padding so model differences are legible. Raw scatter
    is OFF by default; pass `show_points=True` for a faint overlay.

    `combo_df` is the output of `compute_fold_combo_metrics_from_cache(...)` —
    one row per (coverage, fold_key, model) carrying `dist_to_nearest_train` and
    `dist_to_centroid_train`.
    """
    if metric not in {"mean_pearson", "mean_spearman", "mean_r2", "mean_rmse"}:
        raise ValueError("metric must be one of: mean_pearson, mean_spearman, mean_r2, mean_rmse")
    if distance_col not in {"dist_to_nearest_train", "dist_to_centroid_train"}:
        raise ValueError("distance_col must be one of: dist_to_nearest_train, dist_to_centroid_train")
    needed = {"model", metric, distance_col}
    if not needed.issubset(combo_df.columns):
        raise KeyError(f"combo_df missing columns: {sorted(needed - set(combo_df.columns))}")

    d = combo_df[list(needed)].copy()
    d["model"] = d["model"].astype(str).str.lower()
    d = d[pd.to_numeric(d[distance_col], errors="coerce").notna() & pd.to_numeric(d[metric], errors="coerce").notna()].copy()
    if len(d) == 0:
        raise RuntimeError("No rows remain after dropping NaNs in distance/metric")

    model_order = ordered_models(d["model"].dropna().unique())

    # Quantile bins computed across all models so each bin spans the same x range.
    x_all = d[distance_col].to_numpy(dtype=np.float64)
    n_bins_eff = int(min(max(2, n_bins), max(2, len(np.unique(x_all)))))
    edges = np.unique(np.quantile(x_all, np.linspace(0.0, 1.0, n_bins_eff + 1)))
    if len(edges) < 3:
        edges = np.linspace(x_all.min(), x_all.max(), 3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(x_all, edges[1:-1], right=False), 0, len(centers) - 1)
    d["_bin"] = bin_idx
    d["_x_center"] = centers[bin_idx]

    bin_summary = (
        d.groupby(["model", "_bin"], as_index=False)
        .agg(
            x_center=("_x_center", "first"),
            mean=(metric, "mean"),
            sem=(metric, lambda s: float(s.sem(ddof=1)) if len(s.dropna()) > 1 else float("nan")),
            n=(metric, "size"),
        )
        .sort_values(["model", "_bin"])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))

    if bool(show_points):
        for model in model_order:
            dm = d[d["model"] == model]
            ax.scatter(
                dm[distance_col], dm[metric],
                color=MODEL_COLORS.get(model, "#777777"),
                alpha=float(point_alpha), s=float(point_size), linewidths=0,
                zorder=2,
            )

    all_means: list[np.ndarray] = []
    all_sems: list[np.ndarray] = []
    for model in model_order:
        dm = bin_summary[bin_summary["model"] == model].dropna(subset=["mean"]).copy()
        if len(dm) == 0:
            continue
        c = MODEL_COLORS.get(model, "#777777")
        x = dm["x_center"].to_numpy(dtype=np.float64)
        means = dm["mean"].to_numpy(dtype=np.float64)
        sem = np.where(np.isfinite(dm["sem"].to_numpy(dtype=np.float64)),
                        dm["sem"].to_numpy(dtype=np.float64), 0.0)
        all_means.append(means)
        all_sems.append(sem)
        ax.errorbar(
            x, means, yerr=sem,
            fmt="o-",
            color=c, ecolor=c,
            markersize=5.5,
            linewidth=2.0,
            elinewidth=float(error_linewidth),
            capsize=float(error_capsize),
            capthick=float(error_linewidth),
            label=model_label(model),
            zorder=4,
        )

    if all_means:
        means_concat = np.concatenate(all_means)
        sem_concat = np.concatenate(all_sems) if all_sems else np.array([0.0])
        y_lo = float(np.nanmin(means_concat - sem_concat))
        y_hi = float(np.nanmax(means_concat + sem_concat))
        span = max(y_hi - y_lo, 1e-6)
        pad = float(y_pad_frac) * span
        ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(
        f"{_metric_axis_label(metric)} vs {format_legend_label(distance_col)}",
        fontsize=FONT["title"] + 3,
    )
    ax.set_xlabel(f"{format_legend_label(distance_col)} (mm, fold-level)", fontsize=FONT["label"] + 1)
    ax.set_ylabel(_metric_axis_label(metric), fontsize=FONT["label"] + 1)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.legend(title="Model", frameon=True, fancybox=False, loc="best")
    fig.tight_layout()
    return fig, ax, bin_summary


def plot_metric_delta_violins(
    metric_df: pd.DataFrame,
    metric: str = "rmse",
    comparisons: Sequence[tuple[str, str]] = (("dlam", "naive"), ("plam", "naive"), ("plam", "dlam")),
    stratify_by: str | None = None,
    figsize: Tuple[float, float] = (9.4, 4.8),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    need = {"subject_region_key", "model", metric}
    if not need.issubset(metric_df.columns):
        raise ValueError(f"metric_df must include {sorted(need)}")
    idx_cols = ["subject_region_key"]
    if stratify_by is not None:
        idx_cols.append(str(stratify_by))
    wide = metric_df.pivot_table(index=idx_cols, columns="model", values=metric, aggfunc="first").reset_index()
    rows = []
    comparison_order = []
    comparison_palette = {}
    for a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        comparison = f"{model_label(a)} - {model_label(b)}"
        comparison_order.append(comparison)
        comparison_palette[comparison] = MODEL_COLORS.get(a, "#777777") if str(b).lower() == "naive" else "#5f5f5f"
        vals = wide[a] - wide[b]
        for i, v in vals.items():
            row = {
                "comparison": comparison,
                "delta": float(v) if pd.notna(v) else np.nan,
                "metric": metric,
            }
            if stratify_by is not None:
                row[str(stratify_by)] = wide.loc[i, str(stratify_by)]
            rows.append(row)
    delta_df = pd.DataFrame(rows).dropna(subset=["delta"])
    x = str(stratify_by) if stratify_by is not None else "comparison"
    hue = "comparison" if stratify_by is not None else None
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    if stratify_by is None:
        sns.violinplot(
            data=delta_df,
            x=x,
            y="delta",
            order=comparison_order,
            palette=comparison_palette,
            cut=0,
            inner="quartile",
            ax=ax,
        )
    else:
        sns.violinplot(
            data=delta_df,
            x=x,
            y="delta",
            hue=hue,
            hue_order=comparison_order,
            palette=comparison_palette,
            cut=0,
            inner="quartile",
            ax=ax,
        )
    ax.axhline(0.0, color="#222222", linewidth=0.9, linestyle="--")
    ax.set_title(f"Paired Model Delta: {format_legend_label(metric)}", fontsize=FONT["title"] + 3)
    ax.set_xlabel(format_legend_label(x), fontsize=FONT["label"] + 1)
    ax.set_ylabel(f"Delta {format_legend_label(metric)} (first - second)", fontsize=FONT["label"] + 1)
    ax.grid(True, axis="y", alpha=0.18)
    if hue:
        ax.legend(title="Comparison", frameon=True, fancybox=False)
    return fig, ax, delta_df


def plot_region_model_metric_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "pearson_r",
    stratify_by: str = "region_group",
    figsize: Tuple[float, float] = (7.2, 4.4),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    d = summary_df[(summary_df["metric"] == metric) & (summary_df["stratify_by"] == stratify_by)].copy()
    pivot = d.pivot(index="stratum", columns="model", values="mean")
    order = [x for x in ["cortical", "subcortical", "cerebellar"] if x in pivot.index] + [x for x in pivot.index if x not in {"cortical", "subcortical", "cerebellar"}]
    pivot = pivot.loc[order, [m for m in MODEL_ORDER if m in pivot.columns]]
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": format_legend_label(metric)})
    ax.set_title(f"{format_legend_label(metric)} by {format_legend_label(stratify_by)}", fontsize=FONT["title"] + 3)
    ax.set_xlabel("Model", fontsize=FONT["label"] + 1)
    ax.set_ylabel(format_legend_label(stratify_by), fontsize=FONT["label"] + 1)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([model_label(t.get_text()) for t in ax.get_xticklabels()], rotation=0)
    return fig, ax, pivot


def plot_coverage_vs_metric(
    metric_df: pd.DataFrame,
    metric: str = "pearson_r",
    figsize: Tuple[float, float] = (7.2, 4.8),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    d = (
        metric_df.groupby(["subject", "model", "subject_coverage"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
    )
    d["model"] = d["model"].astype(str).str.lower()
    model_order = ordered_models(d["model"].dropna().unique())
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi))
    sns.scatterplot(data=d, x="subject_coverage", y=metric, hue="model", hue_order=model_order, palette=MODEL_COLORS, alpha=0.72, s=26, ax=ax)
    sns.lineplot(data=d, x="subject_coverage", y=metric, hue="model", hue_order=model_order, palette=MODEL_COLORS, estimator="mean", errorbar=None, legend=False, ax=ax)
    ax.set_title(f"Coverage vs {format_legend_label(metric)}", fontsize=FONT["title"] + 3)
    ax.set_xlabel("Observed Regions per Subject", fontsize=FONT["label"] + 1)
    ax.set_ylabel(format_legend_label(metric), fontsize=FONT["label"] + 1)
    ax.grid(True, alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [model_label(v) for v in labels], title="Model", frameon=True, fancybox=False)
    return fig, ax, d


def paired_ttests_by_subject(
    metrics_df: pd.DataFrame,
    label: str = "",
    metric: str = "pearson_r",
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, float]]]:
    if metric not in set(metrics_df.columns):
        raise ValueError(f"metrics_df does not include metric column: {metric}")
    pivot = metrics_df.pivot(index="subject", columns="model", values=metric).dropna()
    model_names = pivot.columns.tolist()
    t_results: Dict[Tuple[str, str], Dict[str, float]] = {}
    print(f"\nPaired t-tests for {label} ({metric}):")
    for model1, model2 in combinations(model_names, 2):
        t_stat, p_val = ttest_rel(pivot[model1], pivot[model2])
        t_results[(str(model1), str(model2))] = {"t_stat": float(t_stat), "p_val": float(p_val)}
        print(f"  {model1} vs {model2}: t = {t_stat:.4f}, p = {p_val:.4g}")
    return pivot, t_results


def select_subjects_by_metric_percentile(
    metrics_df: pd.DataFrame,
    model: str = "plam",
    metric: str = "r2",
    percentiles: Sequence[float] = (0.10, 0.50, 0.90),
    min_coverage: int | None = None,
    max_coverage: int | None = None,
    require_coverage: int | None = None,
) -> pd.DataFrame:
    if metric not in set(metrics_df.columns):
        raise ValueError(f"metrics_df does not include metric column: {metric}")
    d = metrics_df[(metrics_df["model"].astype(str).str.lower() == str(model).lower()) & metrics_df[metric].notna()].copy()
    if require_coverage is not None:
        d = d[d["coverage"].astype(int) == int(require_coverage)].copy()
    else:
        if min_coverage is not None:
            d = d[d["coverage"].astype(int) >= int(min_coverage)].copy()
        if max_coverage is not None:
            d = d[d["coverage"].astype(int) <= int(max_coverage)].copy()
    if len(d) == 0:
        raise RuntimeError("No subjects remain after percentile-selection filters")

    d = d.sort_values([metric, "subject"]).reset_index(drop=True)
    rows = []
    for q in percentiles:
        target = float(d[metric].quantile(float(q)))
        i = (d[metric] - target).abs().idxmin()
        row = d.loc[i].copy()
        row["percentile"] = float(q)
        row["target_value"] = target
        rows.append(row)
    out = pd.DataFrame(rows)
    preferred = ["percentile", "subject", "model", "coverage", metric, "target_value", "pearson_r", "spearman_r", "r2", "rmse"]
    cols = []
    for c in preferred:
        if c in out.columns and c not in cols:
            cols.append(c)
    return out[cols]


def _parse_fold_key(fold_key: str) -> Tuple[int, list[int]]:
    parts = str(fold_key).split("|")
    hold = int(parts[0].split("=")[1].strip())
    train_str = parts[1].split("=")[1].strip() if len(parts) > 1 else ""
    train = [int(x) for x in train_str.split(",") if str(x).strip() != ""]
    return hold, train


def _metric_axis_label(metric: str) -> str:
    labels = {
        "mean_pearson": "Mean fold Pearson r",
        "mean_spearman": "Mean fold Spearman r",
        "mean_r2": r"Mean fold $R^2$",
        "mean_rmse": "Mean fold RMSE",
    }
    if str(metric) not in labels:
        raise ValueError("metric must be one of: mean_pearson, mean_spearman, mean_r2, mean_rmse")
    return labels[str(metric)]


def _rank_fold_combos(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric in {"mean_pearson", "mean_spearman", "mean_r2"}:
        ascending = True
    elif metric == "mean_rmse":
        ascending = False
    else:
        raise ValueError("metric must be one of: mean_pearson, mean_spearman, mean_r2, mean_rmse")
    ranked = d.sort_values([metric, "fold_key"], ascending=[ascending, True]).reset_index(drop=True)
    ranked["rank_index"] = np.arange(len(ranked), dtype=np.int32)
    ranked["rank_percentile"] = np.linspace(0.0, 100.0, len(ranked)) if len(ranked) > 1 else 50.0
    return ranked


def _fold_combo_split_rows(
    ranked: pd.DataFrame,
    model: str,
    metric: str,
    p2label: Dict[int, str],
    rank_kind: str,
) -> list[dict[str, object]]:
    label = model_label(model)
    rows: list[dict[str, object]] = []
    for tag, ix in zip(["worst", "median", "best"], [0, int(len(ranked) // 2), int(len(ranked) - 1)]):
        row = ranked.iloc[ix]
        hold, train = _parse_fold_key(str(row["fold_key"]))
        train_labels = [p2label.get(int(t), str(t)) for t in train]
        rows.append(
            {
                "rank_kind": rank_kind,
                "rank_tag": tag,
                "model": model,
                "model_label": label,
                "metric": metric,
                "score": float(row[metric]),
                "rank_index": int(row["rank_index"]),
                "rank_percentile": float(row["rank_percentile"]),
                "coverage": int(row["coverage"]),
                "n_subjects": int(row["n_subjects"]),
                "fold_key": str(row["fold_key"]),
                "hold_parcel": int(hold),
                "hold_label": p2label.get(int(hold), str(hold)),
                "train_parcels": train,
                "train_labels": train_labels,
            }
        )
    return rows


def _draw_split_panels(
    fig: plt.Figure,
    gs,
    split_df: pd.DataFrame,
    model_list: Sequence[str],
) -> Dict[str, plt.Axes]:
    axes: Dict[str, plt.Axes] = {}
    for i, model in enumerate(model_list[:3]):
        ax = fig.add_subplot(gs[1, i])
        axes[model] = ax
        ax.axis("off")
        model_splits = split_df[split_df["model"] == model].copy()
        color = MODEL_COLORS.get(model, "#333333")
        title = model_label(model)
        ax.set_title(title, fontsize=FONT["title"] + 2, color=color, pad=4)
        y = 0.98
        for _, row in model_splits.iterrows():
            train_text = "; ".join(str(x) for x in row["train_labels"])
            train_wrapped = textwrap.fill(train_text, width=56)
            block = (
                f"{str(row['rank_tag']).upper()}  {row['score']:.3f} | cov={int(row['coverage'])} | n={int(row['n_subjects'])}\n"
                f"hold: {row['hold_label']}\n"
                f"train: {train_wrapped}"
            )
            ax.text(
                0.0,
                y,
                block,
                ha="left",
                va="top",
                fontsize=FONT["small"],
                color="#222222",
                transform=ax.transAxes,
                linespacing=1.12,
            )
            y -= 0.33
    return axes


def plot_fold_combo_ranked_overlay(
    combo_df: pd.DataFrame,
    prepost: Dict[str, object],
    models: Sequence[str] | None = None,
    metric: str = "mean_pearson",
    parcel_label_mode: str = "gtex",
    figsize: Tuple[float, float] = (15.8, 8.8),
    dpi: int = 180,
    show_points: bool = False,
    line_width: float = 2.2,
    point_size: float = 9.0,
) -> Tuple[plt.Figure, Dict[str, plt.Axes], pd.DataFrame]:
    """Overlay per-model fold-combo rankings and summarize worst/median/best splits.

    Rankings are computed independently within each model. The x-axis is rank
    percentile, not a shared fold-key coordinate.
    """
    need = {"model", "fold_key", metric, "coverage", "n_subjects"}
    if not need.issubset(set(combo_df.columns)):
        raise ValueError(f"combo_df must include columns: {sorted(need)}")

    model_list = ordered_models(models if models is not None else MODEL_ORDER)
    label_df = build_parcel_label_table(prepost, eligible_only=True)
    p2label = parcel_label_lookup(label_df, label_mode=parcel_label_mode)

    fig = plt.figure(figsize=figsize, dpi=int(dpi), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.85, 1.55], hspace=0.42, wspace=0.26)
    ax_main = fig.add_subplot(gs[0, :])
    axes: Dict[str, plt.Axes] = {"main": ax_main}

    split_rows = []
    for model in model_list:
        d = combo_df[combo_df["model"].astype(str).str.lower() == model].copy()
        if len(d) == 0:
            continue
        ranked = _rank_fold_combos(d, metric=metric)
        color = MODEL_COLORS.get(model, "#333333")
        label = model_label(model)
        ax_main.plot(
            ranked["rank_percentile"].to_numpy(dtype=np.float64),
            ranked[metric].to_numpy(dtype=np.float64),
            color=color,
            linewidth=float(line_width),
            alpha=0.96,
            label=label,
        )
        if bool(show_points):
            ax_main.scatter(
                ranked["rank_percentile"].to_numpy(dtype=np.float64),
                ranked[metric].to_numpy(dtype=np.float64),
                s=float(point_size),
                color=color,
                alpha=0.36,
                linewidths=0,
                rasterized=True,
            )

        split_rows.extend(_fold_combo_split_rows(ranked, model=model, metric=metric, p2label=p2label, rank_kind="independent"))

    split_df = pd.DataFrame(split_rows)
    if len(split_df) == 0:
        raise RuntimeError("No fold-combo rows found for the requested models")

    ax_main.set_title(
        f"LORO fold-combo ranking by model ({metric}; each model ranked independently)",
        fontsize=FONT["title"] + 5,
    )
    ax_main.set_xlabel("Fold-combo rank percentile (worst -> best within model)", fontsize=FONT["label"] + 3)
    ax_main.set_ylabel(_metric_axis_label(metric), fontsize=FONT["label"] + 3)
    ax_main.set_xlim(0.0, 100.0)
    ax_main.set_xticks([0, 25, 50, 75, 100])
    ax_main.grid(True, axis="both", alpha=0.18)
    ax_main.tick_params(labelsize=FONT["tick"] + 2)
    ax_main.legend(frameon=False, loc="best", fontsize=FONT["legend"] + 2)

    axes.update(_draw_split_panels(fig, gs, split_df, model_list))

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.065, top=0.93)
    return fig, axes, split_df


def plot_fold_combo_matched_overlay(
    combo_df: pd.DataFrame,
    prepost: Dict[str, object],
    models: Sequence[str] | None = None,
    reference_model: str = "plam",
    metric: str = "mean_pearson",
    parcel_label_mode: str = "gtex",
    figsize: Tuple[float, float] = (15.8, 8.8),
    dpi: int = 180,
    show_points: bool = False,
    line_width: float = 2.2,
    point_size: float = 9.0,
) -> Tuple[plt.Figure, Dict[str, plt.Axes], pd.DataFrame, pd.DataFrame]:
    """Overlay models on the same fold-key order, sorted by one reference model."""
    need = {"model", "fold_key", metric, "coverage", "n_subjects"}
    if not need.issubset(set(combo_df.columns)):
        raise ValueError(f"combo_df must include columns: {sorted(need)}")

    model_list = ordered_models(models if models is not None else MODEL_ORDER)
    ref_model = str(reference_model).lower()
    if ref_model not in model_list:
        model_list = list(model_list) + [ref_model]

    label_df = build_parcel_label_table(prepost, eligible_only=True)
    p2label = parcel_label_lookup(label_df, label_mode=parcel_label_mode)

    ref = combo_df[combo_df["model"].astype(str).str.lower() == ref_model].copy()
    if len(ref) == 0:
        raise RuntimeError(f"No combo rows found for reference_model={ref_model}")
    ref_ranked = _rank_fold_combos(ref, metric=metric)
    order = ref_ranked[["fold_key", "rank_index", "rank_percentile"]].copy()

    fig = plt.figure(figsize=figsize, dpi=int(dpi), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.85, 1.55], hspace=0.42, wspace=0.26)
    ax_main = fig.add_subplot(gs[0, :])
    axes: Dict[str, plt.Axes] = {"main": ax_main}

    split_rows = []
    matched_rows = []
    for model in model_list:
        d = combo_df[combo_df["model"].astype(str).str.lower() == model].copy()
        if len(d) == 0:
            continue
        matched = order.merge(d, on="fold_key", how="left").sort_values("rank_index").reset_index(drop=True)
        matched["model"] = model
        matched_rows.append(matched.copy())
        ok = matched[metric].notna()
        color = MODEL_COLORS.get(model, "#333333")
        label = model_label(model)
        ax_main.plot(
            matched.loc[ok, "rank_percentile"].to_numpy(dtype=np.float64),
            matched.loc[ok, metric].to_numpy(dtype=np.float64),
            color=color,
            linewidth=float(line_width),
            alpha=0.96,
            label=label,
        )
        if bool(show_points):
            ax_main.scatter(
                matched.loc[ok, "rank_percentile"].to_numpy(dtype=np.float64),
                matched.loc[ok, metric].to_numpy(dtype=np.float64),
                s=float(point_size),
                color=color,
                alpha=0.36,
                linewidths=0,
                rasterized=True,
            )

        split_ranked = matched.loc[ok].reset_index(drop=True)
        if len(split_ranked) > 0:
            split_rows.extend(_fold_combo_split_rows(split_ranked, model=model, metric=metric, p2label=p2label, rank_kind=f"matched_to_{ref_model}"))

    split_df = pd.DataFrame(split_rows)
    if len(split_df) == 0:
        raise RuntimeError("No matched fold-combo rows found for the requested models")

    ax_main.set_title(
        f"LORO fold-combo ranking matched to {model_label(ref_model)} order ({metric})",
        fontsize=FONT["title"] + 5,
    )
    ax_main.set_xlabel(
        f"Fold-combo rank percentile in {model_label(ref_model)} order (worst -> best)",
        fontsize=FONT["label"] + 3,
    )
    ax_main.set_ylabel(_metric_axis_label(metric), fontsize=FONT["label"] + 3)
    ax_main.set_xlim(0.0, 100.0)
    ax_main.set_xticks([0, 25, 50, 75, 100])
    ax_main.grid(True, axis="both", alpha=0.18)
    ax_main.tick_params(labelsize=FONT["tick"] + 2)
    ax_main.legend(frameon=False, loc="best", fontsize=FONT["legend"] + 2)
    axes.update(_draw_split_panels(fig, gs, split_df, model_list))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.065, top=0.93)
    matched_df = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()
    return fig, axes, split_df, matched_df


def plot_loro_subject_summary_bars(
    metrics_df: pd.DataFrame,
    figsize: Tuple[float, float] = (19.4, 4.25),
    use_sem: bool = True,
    panel_label: str | None = None,
    disable_metrics: Sequence[str] | None = None,
    scatter: bool = False,
    cfg: EDAConfig | None = None,
    prepost: Dict[str, object] | None = None,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    scatter_color_by: str = "parcel",
    scatter_top_n: int = 10,
    parcel_label_mode: str = "gtex",
    scatter_legend: bool = True,
    scatter_kws: Dict[str, object] | None = None,
    scatter_max_points_per_model: int | None = 300_000,
    scatter_random_seed: int = 0,
    scatter_rasterized: bool = True,
    n_jobs: int | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    fig, axes, summary_df = _legacy_plot_loro_subject_summary_bars(
        metrics_df,
        figsize=figsize,
        use_sem=use_sem,
        panel_label=panel_label,
        disable_metrics=disable_metrics,
    )
    if bool(scatter):
        if cfg is None:
            raise ValueError("cfg is required when scatter=True")
        kws = dict(scatter_kws or {})
        kws.setdefault("max_points_per_model", scatter_max_points_per_model)
        kws.setdefault("random_seed", scatter_random_seed)
        kws.setdefault("rasterized", scatter_rasterized)
        scatter_fig, _, _ = plot_global_true_pred_scatter_triplet(
            cfg,
            models=MODEL_ORDER,
            eval_gene_mode=eval_gene_mode,
            custom_gene_list=custom_gene_list,
            eval_gene_path=eval_gene_path,
            prepost=prepost,
            parcel_label_mode=parcel_label_mode,
            scatter_color_by=scatter_color_by,
            scatter_top_n=scatter_top_n,
            scatter_legend=scatter_legend,
            n_jobs=n_jobs,
            **kws,
        )
    return fig, axes, summary_df


def run_subject_metric_panel(
    cfg: EDAConfig,
    label: str,
    models: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    n_jobs: int | None = None,
    scatter: bool = True,
    prepost: Dict[str, object] | None = None,
    scatter_color_by: str = "parcel",
    scatter_top_n: int = 10,
    parcel_label_mode: str = "gtex",
    scatter_kws: Dict[str, object] | None = None,
    scatter_max_points_per_model: int | None = 300_000,
    ttest_metric: str = "pearson_r",
    display_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Dict[str, float]]]:
    model_list = ordered_models(models if models is not None else MODEL_ORDER)
    dfs = [
        compute_subject_metrics_from_cache_gene_subset(
            cfg,
            model=model,
            eval_gene_path=eval_gene_path,
            n_jobs=n_jobs,
        )
        for model in model_list
    ]
    metrics_df = pd.concat(dfs, ignore_index=True)
    if bool(display_outputs) and _display is not None:
        _display(metrics_df.head())
    bar_fig, _, summary_df = plot_loro_subject_summary_bars(
        metrics_df,
        use_sem=True,
        panel_label=label,
        scatter=scatter,
        cfg=cfg,
        prepost=prepost,
        eval_gene_path=eval_gene_path,
        scatter_color_by=scatter_color_by,
        scatter_top_n=scatter_top_n,
        parcel_label_mode=parcel_label_mode,
        scatter_kws=scatter_kws,
        scatter_max_points_per_model=scatter_max_points_per_model,
        n_jobs=n_jobs,
    )
    if bool(display_outputs) and _display is not None:
        _display(summary_df)
    pivot_df, t_results = paired_ttests_by_subject(metrics_df, label=label, metric=ttest_metric)
    if bool(display_outputs) and _display is not None:
        _display(pivot_df.head())
    return metrics_df, summary_df, pivot_df, t_results
