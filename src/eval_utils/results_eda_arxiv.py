#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
except Exception:  # pragma: no cover - optional dependency
    _sm_lowess = None

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_subject_eligibility,
    build_target_parcels,
    map_gtex_to_target,
)

MODEL_ORDER = ["naive", "dlam", "plam"]
MODEL_COLORS = {"naive": "#7f7f7f", "dlam": "#1f77b4", "plam": "#d62728"}
MODEL_LABELS = {"naive": "Naive fill", "dlam": "DLAM", "plam": "PLAM"}
FONT = {"title": 11, "label": 10, "tick": 9, "legend": 9, "small": 8}


@dataclass
class EDAConfig:
    csv_path: str = "data/raw/gxp_samples.csv"
    hvg_path: str = "out/raw/gene_lists/ahba_100hvg.txt"
    cache_root: str = "out/loro_subject_cache"
    naive_cache_dirname: str = "naive"
    dlam_cache_dirname: str = "dlam"
    plam_cache_dirname: str = "plam"
    gene_scope: str = "allgenes"  # hvg
    min_observed_parcels: int = 5
    combat_use_covariates: bool = True
    gtex_rep_mode: str = "centroid"
    gtex_hemi_mode: str = "mirror_left"


def set_academic_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT["label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
        }
    )


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _resolve_eval_gene_path(path_str: str) -> Path:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError("eval gene path/name cannot be empty")
    candidates: List[Path] = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((REPO_ROOT / p).resolve())
        candidates.append((REPO_ROOT.parent / "out" / "raw" / "gene_lists" / p).resolve())
        candidates.append((REPO_ROOT / "out" / "raw" / "gene_lists" / p).resolve())
        candidates.append((REPO_ROOT / "data" / "raw" / "gene_lists" / p).resolve())
        candidates.append((REPO_ROOT / "data" / "raw" / p).resolve())
        if p.suffix == "":
            candidates.append((REPO_ROOT / f"{raw}.txt").resolve())
            candidates.append((REPO_ROOT.parent / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve())
            candidates.append((REPO_ROOT / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve())
            candidates.append((REPO_ROOT / "data" / "raw" / "gene_lists" / f"{raw}.txt").resolve())
            candidates.append((REPO_ROOT / "data" / "raw" / f"{raw}.txt").resolve())
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve eval gene file from {path_str!r}")


def _resolve_hvg_path(path_str: str) -> Path:
    raw = str(path_str).strip()
    candidates: List[Path] = []
    if raw:
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((REPO_ROOT / p).resolve())
    candidates.extend(
        [
            (REPO_ROOT.parent / "out" / "raw" / "gene_lists" / "ahba_100hvg.txt").resolve(),
            (REPO_ROOT / "out" / "raw" / "gene_lists" / "ahba_100hvg.txt").resolve(),
            (REPO_ROOT / "data" / "raw" / "gene_lists" / "ahba_100hvg.txt").resolve(),
            (REPO_ROOT / "data" / "raw" / "ahba_100hvg.txt").resolve(),
            (REPO_ROOT / "ahba_100hvg.txt").resolve(),
        ]
    )
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not resolve HVG gene list path. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


@lru_cache(maxsize=32)
def _cached_gene_header(csv_path_str: str, hvg_path_str: str) -> Dict[str, List[str]]:
    return io_utils.load_gene_header_and_hvg(Path(csv_path_str), Path(hvg_path_str))


@lru_cache(maxsize=64)
def _load_gene_list_from_txt(path_str: str) -> Tuple[str, ...]:
    p = _resolve_eval_gene_path(path_str)
    genes = [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
    if not genes:
        raise ValueError(f"No genes found in eval gene file: {p}")
    return tuple(genes)


def _resolve_eval_gene_set(
    cfg: EDAConfig,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
) -> Tuple[str, str | None, set[str] | None]:
    if eval_gene_path is not None and str(eval_gene_path).strip():
        p = _resolve_eval_gene_path(str(eval_gene_path))
        genes = set(str(g) for g in _load_gene_list_from_txt(str(p)))
        return f"file:{p.stem}", str(p), genes

    mode = str(eval_gene_mode).lower()
    if mode not in {"all", "hvg", "custom"}:
        raise ValueError("eval_gene_mode must be one of: all, hvg, custom")
    if mode == "all":
        return mode, None, None
    if mode == "hvg":
        hdr = _cached_gene_header(str(_resolve_repo_path(cfg.csv_path)), str(_resolve_repo_path(cfg.hvg_path)))
        return mode, str(_resolve_repo_path(cfg.hvg_path)), set(str(g) for g in hdr["genes_hvg"])
    if custom_gene_list is None:
        raise ValueError("custom_gene_list is required when eval_gene_mode='custom'")
    genes = set(str(g) for g in custom_gene_list)
    if not genes:
        raise ValueError("custom_gene_list is empty")
    return mode, None, genes


def _gene_indices_from_names(
    gene_names: Sequence[str],
    eval_gene_set: set[str] | None,
    *,
    context: str,
) -> np.ndarray:
    n_total_genes = int(len(gene_names))
    if eval_gene_set is None:
        return np.arange(n_total_genes, dtype=np.int32)
    gi = np.asarray([i for i, g in enumerate(gene_names) if g in eval_gene_set], dtype=np.int32)
    if int(gi.size) == 0:
        raise RuntimeError(f"No overlap between requested eval genes and cache genes for {context}")
    return gi


def _cache_model_dirname(cfg: EDAConfig, model: str) -> str:
    m = str(model).lower()
    if m == "naive":
        return str(getattr(cfg, "naive_cache_dirname", "naive"))
    if m == "dlam":
        return str(getattr(cfg, "dlam_cache_dirname", "dlam"))
    if m == "plam":
        return str(getattr(cfg, "plam_cache_dirname", "plam"))
    return m


def _model_cache_root(cfg: EDAConfig, model: str) -> Path:
    return (_resolve_repo_path(cfg.cache_root) / str(cfg.gene_scope).lower() / _cache_model_dirname(cfg, model)).resolve()


def _cache_pred_truth_arrays(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(z["loro_fused_subject_h"], dtype=np.float64),
        np.asarray(z["loro_truth_subject_h"], dtype=np.float64),
    )


def _normalize_mixed_space_mode(mixed_space_mode: str | None) -> str | None:
    if mixed_space_mode is None:
        return None
    m = str(mixed_space_mode).strip().lower()
    if m in {"", "none"}:
        return None
    if m not in {
        "harmonized_pred_vs_raw_truth_corr",
        "harmonized_pred_vs_raw_truth_refz_corr",
    }:
        raise ValueError(
            "mixed_space_mode must be one of: "
            "harmonized_pred_vs_raw_truth_corr, "
            "harmonized_pred_vs_raw_truth_refz_corr"
        )
    return m


def _cache_pred_truth_arrays_with_mode(
    z: np.lib.npyio.NpzFile,
    *,
    mixed_space_mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm is None:
        return _cache_pred_truth_arrays(z)
    pred = np.asarray(z["loro_fused_subject_h"], dtype=np.float64)
    truth = np.asarray(z["loro_truth_subject_raw"], dtype=np.float64)
    return pred, truth


def _load_cache_meta(json_path: Path) -> Dict[str, object]:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return {}


def _effective_n_jobs(n_items: int, n_jobs: int | None) -> int:
    if n_items <= 1:
        return 1
    if n_jobs is None:
        return max(1, min(8, n_items, os.cpu_count() or 1))
    return max(1, min(int(n_jobs), n_items))


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xx = pd.Series(x[m]).rank(method="average").to_numpy(dtype=np.float64)
    yy = pd.Series(y[m]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _rmse_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return np.nan
    d = x[m] - y[m]
    return float(np.sqrt(np.mean(d**2)))


def _r2_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    sst = float(np.sum((xx - np.mean(xx)) ** 2))
    if sst < 1e-12:
        return np.nan
    sse = float(np.sum((xx - yy) ** 2))
    return float(1.0 - (sse / sst))


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd > 1e-12, sd, np.nan)
    return (arr - mu[None, :]) / sd[None, :]


def _zscore_cols_with_ref(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sd_arr = np.asarray(sd, dtype=np.float64)
    sd_arr = np.where(sd_arr > 1e-12, sd_arr, np.nan)
    return (arr - mu_arr[None, :]) / sd_arr[None, :]


def _reference_gene_stats(prepost: Dict[str, object], refz_noise_quantile: float = 0.10) -> Dict[str, np.ndarray]:
    genes = list(prepost["genes"])
    raw_grp = (
        prepost["gtex_eligible_raw"]
        .groupby(["subject", "parcel_idx"], as_index=False)[genes]
        .mean(numeric_only=True)
    )
    harm_grp = (
        prepost["gtex_eligible_h"]
        .groupby(["subject", "parcel_idx"], as_index=False)[genes]
        .mean(numeric_only=True)
    )
    raw_mat = raw_grp[genes].to_numpy(dtype=np.float64)
    harm_mat = harm_grp[genes].to_numpy(dtype=np.float64)
    mu_raw = np.nanmean(raw_mat, axis=0)
    sd_raw = np.nanstd(raw_mat, axis=0)
    mu_h = np.nanmean(harm_mat, axis=0)
    sd_h = np.nanstd(harm_mat, axis=0)
    sd_raw = np.where(sd_raw > 1e-12, sd_raw, np.nan)
    sd_h = np.where(sd_h > 1e-12, sd_h, np.nan)
    q = float(refz_noise_quantile)
    if not (0.0 <= q < 1.0):
        raise ValueError("refz_noise_quantile must be in [0.0, 1.0)")
    finite_raw = sd_raw[np.isfinite(sd_raw)]
    finite_h = sd_h[np.isfinite(sd_h)]
    thr_raw = float(np.nanquantile(finite_raw, q)) if finite_raw.size else np.nan
    thr_h = float(np.nanquantile(finite_h, q)) if finite_h.size else np.nan
    keep_mask = np.isfinite(sd_raw) & np.isfinite(sd_h)
    if np.isfinite(thr_raw):
        keep_mask &= sd_raw >= thr_raw
    if np.isfinite(thr_h):
        keep_mask &= sd_h >= thr_h
    return {
        "gene_names": np.asarray(genes, dtype=object),
        "mu_raw": np.asarray(mu_raw, dtype=np.float64),
        "sd_raw": np.asarray(sd_raw, dtype=np.float64),
        "mu_h": np.asarray(mu_h, dtype=np.float64),
        "sd_h": np.asarray(sd_h, dtype=np.float64),
        "keep_mask": np.asarray(keep_mask, dtype=bool),
        "refz_noise_quantile": float(q),
        "thr_raw": float(thr_raw) if np.isfinite(thr_raw) else np.nan,
        "thr_h": float(thr_h) if np.isfinite(thr_h) else np.nan,
    }


def _reference_gene_stats_subset(
    ref_stats: Dict[str, np.ndarray] | None,
    gene_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if ref_stats is None:
        return None
    ref_gene_names = tuple(str(x) for x in ref_stats["gene_names"].tolist())
    idx_map = {g: i for i, g in enumerate(ref_gene_names)}
    gi = np.asarray([idx_map[str(g)] for g in gene_names], dtype=np.int32)
    return (
        np.asarray(ref_stats["mu_raw"], dtype=np.float64)[gi],
        np.asarray(ref_stats["sd_raw"], dtype=np.float64)[gi],
        np.asarray(ref_stats["mu_h"], dtype=np.float64)[gi],
        np.asarray(ref_stats["sd_h"], dtype=np.float64)[gi],
        np.asarray(ref_stats["keep_mask"], dtype=bool)[gi],
    )


def _subject_metrics_row_from_npz(
    npz_path_str: str,
    model: str,
    mixed_space_mode: str | None = None,
    ref_stats: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    npz_path = Path(npz_path_str)
    sid = npz_path.stem
    meta = _load_cache_meta(npz_path.with_suffix(".json"))
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    with np.load(npz_path, allow_pickle=True) as z:
        pred, truth = _cache_pred_truth_arrays_with_mode(z, mixed_space_mode=msm)
        mask = z["loro_eval_mask"].astype(bool)
        pred_use = pred[mask, :]
        truth_use = truth[mask, :]
        if msm == "harmonized_pred_vs_raw_truth_refz_corr":
            gene_names = tuple(str(g) for g in z["gene_names"].tolist())
            ref = _reference_gene_stats_subset(ref_stats, gene_names)
            if ref is None:
                raise ValueError("prepost/reference stats are required for mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr'")
            mu_raw, sd_raw, mu_h, sd_h, keep_mask = ref
            if not np.any(keep_mask):
                raise ValueError("No genes remain after refz low-variance filtering")
            pred_use = _zscore_cols_with_ref(pred_use[:, keep_mask], mu_h[keep_mask], sd_h[keep_mask])
            truth_use = _zscore_cols_with_ref(truth_use[:, keep_mask], mu_raw[keep_mask], sd_raw[keep_mask])
        x = truth_use.ravel()
        y = pred_use.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return {
            "subject": sid,
            "model": model,
            "coverage": int(meta.get("n_gtex_observed_subject", int(mask.sum()))),
            "n_loro_parcels": int(mask.sum()),
            "mixed_space_mode": msm,
            "n_points": int(m.sum()),
            "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "spearman_r": _spearman_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "r2": _r2_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
        }


def _subject_metrics_subset_row_from_npz(
    npz_path_str: str,
    model: str,
    mode_label: str,
    eval_gene_path_resolved: str | None,
    eval_gene_items: Tuple[str, ...] | None,
    mixed_space_mode: str | None = None,
    ref_stats: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    npz_path = Path(npz_path_str)
    sid = npz_path.stem
    meta = _load_cache_meta(npz_path.with_suffix(".json"))
    eval_gene_set = None if eval_gene_items is None else set(eval_gene_items)
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    with np.load(npz_path, allow_pickle=True) as z:
        pred, truth = _cache_pred_truth_arrays_with_mode(z, mixed_space_mode=msm)
        mask = z["loro_eval_mask"].astype(bool)
        gene_names = tuple(str(g) for g in z["gene_names"].tolist())
        n_total_genes = int(len(gene_names))
        gi = _gene_indices_from_names(
            gene_names,
            eval_gene_set,
            context=f"subject={sid} model={model}",
        )
        pred_use = pred[mask, :][:, gi]
        truth_use = truth[mask, :][:, gi]
        if msm == "harmonized_pred_vs_raw_truth_refz_corr":
            ref = _reference_gene_stats_subset(ref_stats, gene_names)
            if ref is None:
                raise ValueError("prepost/reference stats are required for mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr'")
            mu_raw, sd_raw, mu_h, sd_h, keep_mask = ref
            keep_sel = keep_mask[gi]
            if not np.any(keep_sel):
                raise ValueError("No selected genes remain after refz low-variance filtering")
            pred_use = _zscore_cols_with_ref(pred_use[:, keep_sel], mu_h[gi][keep_sel], sd_h[gi][keep_sel])
            truth_use = _zscore_cols_with_ref(truth_use[:, keep_sel], mu_raw[gi][keep_sel], sd_raw[gi][keep_sel])
        x = truth_use.ravel()
        y = pred_use.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return {
            "subject": sid,
            "model": model,
            "coverage": int(meta.get("n_gtex_observed_subject", int(mask.sum()))),
            "n_loro_parcels": int(mask.sum()),
            "n_eval_genes": int(gi.size),
            "n_total_genes": n_total_genes,
            "eval_gene_mode": mode_label,
            "eval_gene_path": eval_gene_path_resolved,
            "mixed_space_mode": msm,
            "n_points": int(m.sum()),
            "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "spearman_r": _spearman_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "r2": _r2_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
        }


def _fold_rows_for_subject_npz(
    npz_path_str: str,
    model: str,
    subject_id: str,
    obs: Sequence[int],
    mode_label: str,
    eval_gene_path_resolved: str | None,
    eval_gene_items: Tuple[str, ...] | None,
) -> List[Dict[str, object]]:
    npz_path = Path(npz_path_str)
    eval_gene_set = None if eval_gene_items is None else set(eval_gene_items)
    obs_set = set(int(x) for x in obs)
    n_obs = int(len(obs_set))
    rows: List[Dict[str, object]] = []
    with np.load(npz_path, allow_pickle=True) as z:
        pred, truth = _cache_pred_truth_arrays(z)
        mask = z["loro_eval_mask"].astype(bool)
        gene_names = tuple(str(x) for x in z["gene_names"].tolist())
        gi = _gene_indices_from_names(
            gene_names,
            eval_gene_set,
            context=f"subject={subject_id} model={model}",
        )
        for hold in np.where(mask)[0].astype(int).tolist():
            train = sorted(obs_set - {int(hold)})
            train_key = ",".join(map(str, train))
            fold_key = f"hold={int(hold)}|train={train_key}"
            x = truth[int(hold), gi]
            y = pred[int(hold), gi]
            m = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "subject": str(subject_id),
                    "model": str(model),
                    "coverage": n_obs,
                    "hold_parcel": int(hold),
                    "train_key": train_key,
                    "fold_key": fold_key,
                    "n_eval_genes": int(gi.size),
                    "eval_gene_mode": mode_label,
                    "eval_gene_path": eval_gene_path_resolved,
                    "n_points": int(m.sum()),
                    "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                    "spearman_r": _spearman_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                    "r2": _r2_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                    "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                }
            )
    return rows


def _pooled_pca_subject_block_from_npz(
    npz_path_str: str,
    model: str,
    mode_label: str,
    eval_gene_path_resolved: str | None,
    eval_gene_items: Tuple[str, ...] | None,
    mixed_space_mode: str | None = None,
    ref_stats: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    npz_path = Path(npz_path_str)
    sid = npz_path.stem
    eval_gene_set = None if eval_gene_items is None else set(eval_gene_items)
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    with np.load(npz_path, allow_pickle=True) as z:
        pred, truth = _cache_pred_truth_arrays_with_mode(z, mixed_space_mode=msm)
        mask = np.asarray(z["loro_eval_mask"], dtype=bool)
        parcel_idx = np.asarray(z["parcel_idx"], dtype=np.int32)
        gene_names = tuple(str(g) for g in z["gene_names"].tolist())
        gi = _gene_indices_from_names(
            gene_names,
            eval_gene_set,
            context=f"subject={sid} model={model}",
        )
        pred_use = np.asarray(pred[mask, :][:, gi], dtype=np.float64)
        truth_use = np.asarray(truth[mask, :][:, gi], dtype=np.float64)
        gene_names_use = tuple(gene_names[i] for i in gi.tolist())
        if msm == "harmonized_pred_vs_raw_truth_refz_corr":
            ref = _reference_gene_stats_subset(ref_stats, gene_names_use)
            if ref is None:
                raise ValueError(
                    "prepost/reference stats are required for mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr'"
                )
            mu_raw, sd_raw, mu_h, sd_h, keep_mask = ref
            if not np.any(keep_mask):
                raise ValueError("No genes remain after refz low-variance filtering")
            pred_use = _zscore_cols_with_ref(pred_use[:, keep_mask], mu_h[keep_mask], sd_h[keep_mask])
            truth_use = _zscore_cols_with_ref(truth_use[:, keep_mask], mu_raw[keep_mask], sd_raw[keep_mask])
            gene_names_use = tuple(np.asarray(gene_names_use, dtype=object)[keep_mask].tolist())
        keys = [(str(sid), int(p)) for p in parcel_idx[mask].tolist()]
        return {
            "subject": str(sid),
            "model": str(model),
            "eval_gene_mode": mode_label,
            "eval_gene_path": eval_gene_path_resolved,
            "mixed_space_mode": msm,
            "keys": keys,
            "pred": pred_use,
            "truth": truth_use,
            "gene_names": gene_names_use,
        }


def _resolve_pooled_pca_truth_keys(
    blocks_by_model: Dict[str, List[Dict[str, object]]],
    model_list: Sequence[str],
) -> List[Tuple[str, int]]:
    key_sets: List[set[Tuple[str, int]]] = []
    for model in model_list:
        blocks = blocks_by_model.get(str(model), [])
        keys_m: set[Tuple[str, int]] = set()
        for block in blocks:
            keys_m.update(block["keys"])
        key_sets.append(keys_m)
    if len(key_sets) == 0:
        return []
    common = set.intersection(*key_sets) if len(key_sets) > 1 else key_sets[0]
    return sorted(common, key=lambda x: (str(x[0]), int(x[1])))


def _build_pooled_truth_pred_mats(
    blocks_by_model: Dict[str, List[Dict[str, object]]],
    model_list: Sequence[str],
    common_keys: Sequence[Tuple[str, int]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, ...]]:
    if len(common_keys) == 0:
        raise RuntimeError("No common strict LORO sample rows remain across the selected models")
    key_order = {k: i for i, k in enumerate(common_keys)}
    truth_mat: np.ndarray | None = None
    pred_by_model: Dict[str, np.ndarray] = {}
    gene_names_ref: Tuple[str, ...] | None = None
    n_samples = int(len(common_keys))

    for model in model_list:
        blocks = blocks_by_model.get(str(model), [])
        if len(blocks) == 0:
            raise RuntimeError(f"No pooled PCA cache rows found for model={model}")
        gene_names_this = tuple(blocks[0]["gene_names"])
        if gene_names_ref is None:
            gene_names_ref = gene_names_this
        elif gene_names_this != gene_names_ref:
            raise ValueError("Gene list mismatch across models while building pooled PCA matrices")
        n_genes = int(len(gene_names_this))
        truth_model = np.full((n_samples, n_genes), np.nan, dtype=np.float64)
        pred_model = np.full((n_samples, n_genes), np.nan, dtype=np.float64)
        for block in blocks:
            keys = block["keys"]
            truth = np.asarray(block["truth"], dtype=np.float64)
            pred = np.asarray(block["pred"], dtype=np.float64)
            for row_i, key in enumerate(keys):
                pos = key_order.get(key)
                if pos is None:
                    continue
                truth_model[pos, :] = truth[row_i, :]
                pred_model[pos, :] = pred[row_i, :]
        if np.any(~np.isfinite(truth_model)):
            raise RuntimeError(f"Truth matrix has missing values after common-key alignment for model={model}")
        if np.any(~np.isfinite(pred_model)):
            raise RuntimeError(f"Prediction matrix has missing values after common-key alignment for model={model}")
        pred_by_model[str(model)] = pred_model
        if truth_mat is None:
            truth_mat = truth_model
    if truth_mat is None or gene_names_ref is None:
        raise RuntimeError("Failed to assemble pooled PCA truth/prediction matrices")
    return truth_mat, pred_by_model, gene_names_ref


def _apply_pooled_pca_demean_mode(
    X_true: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    common_keys: Sequence[Tuple[str, int]],
    demean_mode: str = "none",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mode = str(demean_mode).strip().lower()
    if mode in {"", "none"}:
        return X_true, pred_by_model
    if mode != "within_parcel":
        raise ValueError("demean_mode must be one of: none, within_parcel")

    X_true_dm = np.asarray(X_true, dtype=np.float64).copy()
    pred_dm = {str(k): np.asarray(v, dtype=np.float64).copy() for k, v in pred_by_model.items()}
    parcel_idx = np.asarray([int(p) for _, p in common_keys], dtype=np.int32)
    for p in np.unique(parcel_idx):
        idx = np.where(parcel_idx == int(p))[0]
        if int(idx.size) == 0:
            continue
        mu_p = np.nanmean(X_true_dm[idx, :], axis=0)
        X_true_dm[idx, :] = X_true_dm[idx, :] - mu_p[None, :]
        for model in pred_dm:
            pred_dm[model][idx, :] = pred_dm[model][idx, :] - mu_p[None, :]
    return X_true_dm, pred_dm


def _metrics_panel_label(metrics_df: pd.DataFrame, panel_label: str | None = None) -> str:
    if panel_label is not None and str(panel_label).strip():
        return str(panel_label).strip()
    if "eval_gene_mode" not in metrics_df.columns:
        return ""
    vals = sorted(set(metrics_df["eval_gene_mode"].astype(str).str.lower().dropna().tolist()))
    if len(vals) == 1:
        v = vals[0]
        if v == "hvg":
            return "HVG"
        if v == "all":
            return "All genes"
        if v == "custom":
            return "Custom genes"
        return v
    if len(vals) > 1:
        return "Mixed panels"
    return ""


def _pretty_gtex_label(name: str) -> str:
    s = str(name)
    pref = "brain - "
    if s.lower().startswith(pref):
        return s[len(pref) :]
    return s


def _load_expression(cfg: EDAConfig) -> Dict[str, object]:
    csv_path = _resolve_repo_path(cfg.csv_path)
    hvg_path = _resolve_hvg_path(cfg.hvg_path)
    header = io_utils.load_gene_header_and_hvg(csv_path, hvg_path)
    genes = header["genes_all"] if str(cfg.gene_scope).lower() == "allgenes" else header["genes_hvg"]
    if not genes:
        raise RuntimeError(f"No genes found for scope={cfg.gene_scope}")

    df = io_utils.read_expression_subset(
        csv_path,
        genes,
        rep_mode=str(cfg.gtex_rep_mode).lower(),
        hemi_mode=str(cfg.gtex_hemi_mode).lower(),
    )
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)

    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    elig = build_subject_eligibility(gtex_raw, min_observed_parcels=int(cfg.min_observed_parcels))
    subjects = elig[elig["eligible"]]["subject"].astype(str).tolist()
    return {
        "genes_all": header["genes_all"],
        "genes_hvg": header["genes_hvg"],
        "genes": genes,
        "ahba_raw": ahba_raw,
        "gtex_raw": gtex_raw,
        "target_meta": target_meta,
        "eligibility": elig,
        "eligible_subjects": subjects,
        "csv_path": csv_path,
        "hvg_path": hvg_path,
    }


def prepare_pre_post_harmonization(cfg: EDAConfig) -> Dict[str, object]:
    data = _load_expression(cfg)
    genes = data["genes"]
    ahba_raw = data["ahba_raw"]
    gtex_raw = data["gtex_raw"]
    target_meta = data["target_meta"]
    subjects = data["eligible_subjects"]

    gtex_eligible = gtex_raw[gtex_raw["subject"].astype(str).isin(subjects)].copy()
    hcfg = SimpleNamespace(
        combat_use_covariates=bool(cfg.combat_use_covariates),
    )
    harmonizer = fit_harmonizer(ahba_raw, gtex_eligible, genes, method="combat", cfg=hcfg)
    gtex_h = harmonizer.transform(gtex_eligible, "GTEX")
    ahba_h = harmonizer.transform(ahba_raw, "AHBA")

    n_subj = len(subjects)
    n_parc = int(len(target_meta))
    n_genes = int(len(genes))
    raw_cube = np.full((n_subj, n_parc, n_genes), np.nan, dtype=np.float64)
    harm_cube = np.full((n_subj, n_parc, n_genes), np.nan, dtype=np.float64)
    obs_mask = np.zeros((n_subj, n_parc), dtype=bool)

    sid_to_i = {s: i for i, s in enumerate(subjects)}
    for sid, sdf in gtex_eligible.groupby("subject"):
        i = sid_to_i[str(sid)]
        raw_grp = sdf.groupby("parcel_idx")[genes].mean()
        harm_grp = gtex_h[gtex_h["subject"] == sid].groupby("parcel_idx")[genes].mean()
        for pidx in raw_grp.index.tolist():
            p = int(pidx)
            raw_cube[i, p, :] = raw_grp.loc[pidx, genes].to_numpy(dtype=np.float64)
            obs_mask[i, p] = True
        for pidx in harm_grp.index.tolist():
            p = int(pidx)
            harm_cube[i, p, :] = harm_grp.loc[pidx, genes].to_numpy(dtype=np.float64)

    return {
        "subjects": subjects,
        "genes": genes,
        "genes_all": data["genes_all"],
        "genes_hvg": data["genes_hvg"],
        "target_meta": target_meta,
        "ahba_raw": ahba_raw,
        "ahba_h": ahba_h,
        "gtex_eligible_raw": gtex_eligible,
        "gtex_eligible_h": gtex_h,
        "gtex_raw": gtex_raw,
        "eligibility": data["eligibility"],
        "raw_cube": raw_cube,
        "harm_cube": harm_cube,
        "obs_mask": obs_mask,
    }


def _resolve_region_idx(target_meta: pd.DataFrame, region: int | str) -> int:
    if isinstance(region, int):
        return int(region)
    s = str(region)
    if s.isdigit():
        return int(s)
    m = target_meta[target_meta["tissue_or_parcel"].astype(str) == s]
    if len(m) == 0:
        raise ValueError(f"Unknown region label: {region}")
    return int(m.iloc[0]["parcel_idx"])


def available_regions(prepost: Dict[str, object], min_subjects: int = 2) -> pd.DataFrame:
    target_meta = prepost["target_meta"]
    obs_mask = prepost["obs_mask"]
    n_by_region = obs_mask.sum(axis=0).astype(int)
    out = target_meta[["parcel_idx", "tissue_or_parcel"]].copy()
    out["n_subjects_observed"] = n_by_region
    out = out[out["n_subjects_observed"] >= int(min_subjects)].sort_values(["n_subjects_observed", "parcel_idx"], ascending=[False, True]).reset_index(drop=True)
    return out


def parcel_subject_count_table(
    prepost: Dict[str, object],
    label_mode: str = "ahba",  # ahba | gtex | both
    eligible_only: bool = True,
) -> pd.DataFrame:
    mode = str(label_mode).lower()
    if mode not in {"ahba", "gtex", "both"}:
        raise ValueError("label_mode must be one of: ahba, gtex, both")

    gtex_raw = prepost["gtex_raw"].copy()
    target_meta = prepost["target_meta"][["parcel_idx", "tissue_or_parcel"]].copy().rename(columns={"tissue_or_parcel": "ahba_parcel"})
    elig = prepost["eligibility"].copy()
    all_subjects = sorted(gtex_raw["subject"].astype(str).unique().tolist())
    eligible_subjects = sorted(elig[elig["eligible"]]["subject"].astype(str).tolist())
    subjects_use = set(eligible_subjects) if bool(eligible_only) else set(all_subjects)
    g = gtex_raw[gtex_raw["subject"].astype(str).isin(subjects_use)].copy()

    if "mapped_parcel" not in g.columns:
        raise RuntimeError("Expected `mapped_parcel` column in gtex_raw")
    if "tissue_or_parcel" not in g.columns:
        raise RuntimeError("Expected native GTEx `tissue_or_parcel` column in gtex_raw")

    native_map = (
        g.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )
    subj_counts = g.groupby("parcel_idx")["subject"].nunique().to_dict()
    sample_counts = g.groupby("parcel_idx").size().to_dict()

    out = target_meta.copy()
    out["gtex_native"] = out["parcel_idx"].map(native_map).fillna("")
    out["n_subjects_observed"] = out["parcel_idx"].map(subj_counts).fillna(0).astype(int)
    out["n_samples"] = out["parcel_idx"].map(sample_counts).fillna(0).astype(int)
    out["eligible_only"] = bool(eligible_only)
    out["n_subjects_pool"] = int(len(subjects_use))

    if mode == "ahba":
        out["label"] = out["ahba_parcel"]
    elif mode == "gtex":
        out["label"] = np.where(out["gtex_native"].astype(str).str.len() > 0, out["gtex_native"], out["ahba_parcel"])
    else:
        right = np.where(out["gtex_native"].astype(str).str.len() > 0, out["gtex_native"], out["ahba_parcel"])
        out["label"] = out["ahba_parcel"] + ": " + right

    out = out.sort_values(["n_subjects_observed", "parcel_idx"], ascending=[False, True]).reset_index(drop=True)
    return out


def coverage_count_distribution_from_table(count_df: pd.DataFrame, include_zero: bool = False) -> pd.DataFrame:
    d = count_df.copy()
    if "n_subjects_observed" not in d.columns:
        raise ValueError("count_df must include n_subjects_observed")
    if not bool(include_zero):
        d = d[d["n_subjects_observed"] > 0].copy()
    out = (
        d.groupby("n_subjects_observed", as_index=False)
        .size()
        .rename(columns={"size": "n_parcels"})
        .sort_values("n_subjects_observed")
        .reset_index(drop=True)
    )
    return out


def plot_parcel_subject_counts(
    count_df: pd.DataFrame,
    top_n: int | None = None,
    min_subjects: int = 1,
    figsize: Tuple[float, float] = (10.5, 8.5),
) -> Tuple[plt.Figure, plt.Axes]:
    d = count_df[count_df["n_subjects_observed"] >= int(min_subjects)].copy()
    d = d.sort_values(["n_subjects_observed", "parcel_idx"], ascending=[True, True]).copy()
    if top_n is not None and int(top_n) > 0:
        d = d.sort_values(["n_subjects_observed", "parcel_idx"], ascending=[False, True]).head(int(top_n)).copy()
        d = d.sort_values(["n_subjects_observed", "parcel_idx"], ascending=[True, True]).copy()
    n_pool = int(d["n_subjects_pool"].iloc[0]) if len(d) else 0
    elig_flag = bool(d["eligible_only"].iloc[0]) if len(d) else True
    subtitle = f"n={n_pool} subjects ({'eligible-only' if elig_flag else 'all subjects'})"

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.barh(d["label"], d["n_subjects_observed"], color="#3274A1", alpha=0.9)
    ax.set_xlabel("Subjects observed in parcel", fontsize=FONT["label"])
    ax.set_ylabel("Parcel label", fontsize=FONT["label"])
    ax.set_title(f"Per-parcel subject coverage ({subtitle})", fontsize=FONT["title"])
    ax.grid(True, axis="x", alpha=0.2)
    return fig, ax


def coverage_count_distribution(prepost: Dict[str, object], include_zero: bool = False) -> pd.DataFrame:
    obs_mask = prepost["obs_mask"]
    n_by_region = obs_mask.sum(axis=0).astype(int)
    d = pd.DataFrame({"n_subjects_observed": n_by_region})
    if not bool(include_zero):
        d = d[d["n_subjects_observed"] > 0].copy()
    out = (
        d.groupby("n_subjects_observed", as_index=False)
        .size()
        .rename(columns={"size": "n_parcels"})
        .sort_values("n_subjects_observed")
        .reset_index(drop=True)
    )
    return out


def plot_coverage_count_distribution(
    coverage_df: pd.DataFrame,
    figsize: Tuple[float, float] = (7.2, 3.8),
) -> Tuple[plt.Figure, plt.Axes]:
    d = coverage_df.sort_values("n_subjects_observed").copy()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.bar(d["n_subjects_observed"], d["n_parcels"], color="#3274A1", alpha=0.9)
    ax.set_xlabel("Subjects observed in parcel", fontsize=FONT["label"])
    ax.set_ylabel("Number of parcels", fontsize=FONT["label"])
    ax.set_title("Parcel coverage distribution", fontsize=FONT["title"])
    ax.set_xticks(d["n_subjects_observed"].tolist())
    ax.grid(True, axis="y", alpha=0.2)
    return fig, ax


def subject_region_count_distribution(
    prepost: Dict[str, object],
    eligible_only: bool = True,
) -> pd.DataFrame:
    elig = prepost["eligibility"].copy()
    d = elig[elig["eligible"]].copy() if bool(eligible_only) else elig.copy()
    out = (
        d.groupby("n_obs_parcels", as_index=False)
        .size()
        .rename(columns={"n_obs_parcels": "n_regions_sampled", "size": "n_subjects"})
        .sort_values("n_regions_sampled")
        .reset_index(drop=True)
    )
    out["eligible_only"] = bool(eligible_only)
    out["n_subjects_pool"] = int(len(d))
    return out


def plot_subject_region_count_distribution(
    dist_df: pd.DataFrame,
    figsize: Tuple[float, float] = (7.8, 4.2),
) -> Tuple[plt.Figure, plt.Axes]:
    d = dist_df.sort_values("n_regions_sampled").copy()
    n_pool = int(d["n_subjects_pool"].iloc[0]) if len(d) else 0
    elig_flag = bool(d["eligible_only"].iloc[0]) if len(d) else True
    subtitle = f"n={n_pool} subjects ({'eligible-only' if elig_flag else 'all subjects'})"

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.bar(d["n_regions_sampled"], d["n_subjects"], color="#2E8B57", alpha=0.9)
    ax.set_xlabel("Number of regions sampled per subject", fontsize=FONT["label"])
    ax.set_ylabel("Subject count", fontsize=FONT["label"])
    ax.set_title(f"Subject coverage distribution ({subtitle})", fontsize=FONT["title"])
    ax.set_xticks(d["n_regions_sampled"].tolist())
    ax.grid(True, axis="y", alpha=0.2)
    return fig, ax


def _resolve_gene_panel(
    prepost: Dict[str, object],
    gene_mode: str = "hvg",  # hvg | allgenes | custom
    gene_list: Sequence[str] | None = None,
) -> List[str]:
    mode = str(gene_mode).lower()
    genes_all = [str(g) for g in prepost.get("genes_all", prepost["genes"])]
    genes_hvg = [str(g) for g in prepost.get("genes_hvg", prepost["genes"])]
    if mode == "allgenes":
        return genes_all
    if mode == "hvg":
        return genes_hvg
    if mode == "custom":
        if gene_list is None:
            raise ValueError("gene_list is required when gene_mode='custom'")
        keep = [str(g) for g in gene_list if str(g) in set(genes_all)]
        if not keep:
            raise ValueError("No requested custom genes were found in dataset columns")
        return keep
    raise ValueError("gene_mode must be one of: hvg, allgenes, custom")


def _restrict_genes_to_available(requested: List[str], available: Sequence[str]) -> List[str]:
    avail = set(str(g) for g in available)
    keep = [str(g) for g in requested if str(g) in avail]
    if not keep:
        raise ValueError("Requested gene panel has no overlap with currently loaded genes in PREPOST")
    return keep


def _gtex_parcel_subject_median_matrix(
    gtex_df: pd.DataFrame,
    genes: List[str],
    target_meta: pd.DataFrame,
) -> np.ndarray:
    n_parc = int(len(target_meta))
    out = np.full((n_parc, len(genes)), np.nan, dtype=np.float64)
    if len(gtex_df) == 0:
        return out
    subj_parcel = gtex_df.groupby(["subject", "parcel_idx"], as_index=False)[genes].mean()
    parcel_med = subj_parcel.groupby("parcel_idx", as_index=False)[genes].median()
    for _, row in parcel_med.iterrows():
        p = int(row["parcel_idx"])
        out[p, :] = row[genes].to_numpy(dtype=np.float64)
    return out


def _parcel_median_matrix(
    df: pd.DataFrame,
    genes: List[str],
    target_meta: pd.DataFrame,
) -> np.ndarray:
    n_parc = int(len(target_meta))
    out = np.full((n_parc, len(genes)), np.nan, dtype=np.float64)
    if len(df) == 0:
        return out
    grp = df.groupby("parcel_idx", as_index=False)[genes].median()
    for _, row in grp.iterrows():
        p = int(row["parcel_idx"])
        out[p, :] = row[genes].to_numpy(dtype=np.float64)
    return out


def prepare_atlas_median_comparison(
    prepost: Dict[str, object],
    gene_mode: str = "hvg",
    gene_list: Sequence[str] | None = None,
    observed_only: bool = True,
) -> Dict[str, object]:
    genes = _resolve_gene_panel(prepost, gene_mode=gene_mode, gene_list=gene_list)
    genes = _restrict_genes_to_available(genes, prepost["genes"])
    target_meta = prepost["target_meta"]
    labels = target_meta["tissue_or_parcel"].astype(str).tolist()

    gtex_raw = prepost["gtex_eligible_raw"]
    gtex_h = prepost["gtex_eligible_h"]
    ahba_raw = prepost["ahba_raw"]
    ahba_h = prepost["ahba_h"]

    g_raw = _gtex_parcel_subject_median_matrix(gtex_raw, genes, target_meta)
    g_h = _gtex_parcel_subject_median_matrix(gtex_h, genes, target_meta)
    a_raw = _parcel_median_matrix(ahba_raw, genes, target_meta)
    a_h = _parcel_median_matrix(ahba_h, genes, target_meta)

    if bool(observed_only):
        keep = np.isfinite(g_raw).any(axis=1) | np.isfinite(g_h).any(axis=1)
    else:
        keep = np.ones(g_raw.shape[0], dtype=bool)

    gtex_native_by_parcel = (
        gtex_raw.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
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
    }


def plot_subject_prepost_heatmaps(
    prepost: Dict[str, object],
    subject_id: str,
    gene_mode: str = "hvg",  # hvg | allgenes | custom
    gene_list: Sequence[str] | None = None,
    observed_only: bool = True,
    label_mode: str = "gtex",  # gtex | ahba | both
    label_stride: int = 1,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (13.0, 4.6),
) -> Tuple[plt.Figure, np.ndarray]:
    subjects = [str(s) for s in prepost["subjects"]]
    sid = str(subject_id)
    if sid not in set(subjects):
        raise ValueError(f"Subject not found in eligible PREPOST set: {sid}")
    si = subjects.index(sid)

    genes_req = _resolve_gene_panel(prepost, gene_mode=gene_mode, gene_list=gene_list)
    genes = _restrict_genes_to_available(genes_req, prepost["genes"])
    all_genes = [str(g) for g in prepost["genes"]]
    gi = [all_genes.index(g) for g in genes]

    raw = np.asarray(prepost["raw_cube"][si, :, :], dtype=np.float64)[:, gi]
    harm = np.asarray(prepost["harm_cube"][si, :, :], dtype=np.float64)[:, gi]
    target_meta = prepost["target_meta"]
    ahba_labels = target_meta["tissue_or_parcel"].astype(str).tolist()
    sub_rows = prepost["gtex_eligible_raw"][prepost["gtex_eligible_raw"]["subject"].astype(str) == sid].copy()
    gtex_by_parcel = (
        sub_rows.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )
    mode = str(label_mode).lower()
    if mode not in {"gtex", "ahba", "both"}:
        raise ValueError("label_mode must be one of: gtex, ahba, both")
    labels: List[str] = []
    for pidx, ahba_name in enumerate(ahba_labels):
        gname = gtex_by_parcel.get(int(pidx), "")
        if mode == "ahba":
            labels.append(str(ahba_name))
        elif mode == "both":
            right = str(gname) if str(gname) else str(ahba_name)
            labels.append(f"{ahba_name}: {right}")
        else:  # gtex
            labels.append(str(gname) if str(gname) else str(ahba_name))

    if bool(observed_only):
        keep = np.isfinite(raw).any(axis=1) | np.isfinite(harm).any(axis=1)
        raw = raw[keep, :]
        harm = harm[keep, :]
        labels = [labels[i] for i in np.where(keep)[0].tolist()]

    vals = np.r_[raw.ravel(), harm.ravel()]
    vals = vals[np.isfinite(vals)]
    if int(vals.size):
        vmin = float(np.nanpercentile(vals, 1.0))
        vmax = float(np.nanpercentile(vals, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
    else:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    m0 = axes[0].imshow(raw, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{sid} Raw GTEx", fontsize=FONT["title"])
    axes[1].imshow(harm, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{sid} ComBat GTEx", fontsize=FONT["title"])
    for ax in axes:
        ax.set_xlabel("Genes", fontsize=FONT["label"])
        ax.set_ylabel("AHBA parcels", fontsize=FONT["label"])
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
        ax.minorticks_off()

    step = max(1, int(label_stride))
    y_idx = np.arange(0, len(labels), step, dtype=int)
    y_lab = [labels[i] for i in y_idx.tolist()]
    for ax in axes:
        ax.set_yticks(y_idx.tolist())
        ax.set_yticklabels(y_lab, fontsize=FONT["small"])

    # Row-wise Spearman correlation (parcel-wise, across genes): raw vs ComBat.
    row_rho = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for i in range(raw.shape[0]):
        xv = raw[i, :]
        yv = harm[i, :]
        m = np.isfinite(xv) & np.isfinite(yv)
        if int(m.sum()) < 2:
            continue
        xs = pd.Series(xv[m], dtype=np.float64)
        ys = pd.Series(yv[m], dtype=np.float64)
        row_rho[i] = float(xs.corr(ys, method="spearman"))
    rho_ticks = [f"{row_rho[i]:.2f}" if np.isfinite(row_rho[i]) else "nan" for i in y_idx.tolist()]
    ax_r = axes[1].twinx()
    ax_r.set_ylim(axes[1].get_ylim())
    ax_r.set_yticks(y_idx.tolist())
    ax_r.set_yticklabels(rho_ticks, fontsize=FONT["small"])
    ax_r.set_ylabel("Row-wise Spearman (raw vs ComBat)", fontsize=FONT["label"])
    ax_r.grid(False, which="both")
    ax_r.xaxis.grid(False, which="both")
    ax_r.yaxis.grid(False, which="both")
    ax_r.minorticks_off()

    r, rmse, n = _pair_metrics(raw, harm)
    axes[0].text(
        0.01,
        1.15,
        f"Raw vs ComBat: r={r:.3f}, rmse={rmse:.3f}, n={n:,}",
        transform=axes[0].transAxes,
        fontsize=FONT["legend"],
        va="bottom",
    )
    cbar = fig.colorbar(m0, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Expression value")
    return fig, axes


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
        ("Raw GTEx (eligible-subject median)", g_raw, axes[0, 0]),
        ("Raw AHBA (parcel median)", a_raw, axes[0, 1]),
        ("ComBat GTEx (eligible-subject median)", g_h, axes[1, 0]),
        ("ComBat AHBA (parcel median)", a_h, axes[1, 1]),
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

    # Parcel labels (AHBA labels)
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


def _region_region_spearman(mat: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    # Regions are rows; Spearman over genes between all region pairs.
    return pd.DataFrame(x).T.corr(method="spearman", min_periods=2).to_numpy(dtype=np.float64)


def plot_region_region_spearman_heatmaps(
    atlas_cmp: Dict[str, object],
    label_mode: str = "gtex",  # gtex | ahba | both
    label_stride: int | None = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (12.0, 12.0),
) -> Tuple[plt.Figure, np.ndarray]:
    g_raw = np.asarray(atlas_cmp["gtex_raw"], dtype=np.float64)
    a_raw = np.asarray(atlas_cmp["ahba_raw"], dtype=np.float64)
    g_h = np.asarray(atlas_cmp["gtex_h"], dtype=np.float64)
    a_h = np.asarray(atlas_cmp["ahba_h"], dtype=np.float64)
    mode = str(label_mode).lower()
    if mode not in {"gtex", "ahba", "both"}:
        raise ValueError("label_mode must be one of: gtex, ahba, both")
    if mode == "gtex":
        base_labels = atlas_cmp.get("parcel_labels_gtex", atlas_cmp["parcel_labels"])
    elif mode == "both":
        base_labels = atlas_cmp.get("parcel_labels_both", atlas_cmp["parcel_labels"])
    else:
        base_labels = atlas_cmp["parcel_labels"]
    labels = [str(x) for x in base_labels]

    s_g_raw = _region_region_spearman(g_raw)
    s_a_raw = _region_region_spearman(a_raw)
    s_g_h = _region_region_spearman(g_h)
    s_a_h = _region_region_spearman(a_h)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    mats = [
        ("Raw GTEx (region-region Spearman)", s_g_raw, axes[0, 0]),
        ("Raw AHBA (region-region Spearman)", s_a_raw, axes[0, 1]),
        ("ComBat GTEx (region-region Spearman)", s_g_h, axes[1, 0]),
        ("ComBat AHBA (region-region Spearman)", s_a_h, axes[1, 1]),
    ]
    im = None
    for title, mat, ax in mats:
        im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=str(cmap), vmin=-1.0, vmax=1.0)
        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=FONT["title"])
        ax.set_xlabel("AHBA parcels", fontsize=FONT["label"])
        ax.set_ylabel("AHBA parcels", fontsize=FONT["label"])
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
        ax.minorticks_off()

    if label_stride is None or int(label_stride) <= 0:
        # Auto-thin labels for readability while preserving square panels.
        step = max(1, int(np.ceil(len(labels) / 18)))
    else:
        step = max(1, int(label_stride))
    y_idx = np.arange(0, len(labels), step, dtype=int)
    y_lab = [labels[i] for i in y_idx.tolist()]
    for ax in axes.ravel().tolist():
        ax.set_yticks(y_idx.tolist())
        ax.set_xticks(y_idx.tolist())
        ax.set_yticklabels(y_lab, fontsize=FONT["small"])
        ax.set_xticklabels(y_lab, fontsize=FONT["small"], rotation=90)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    cbar.set_label("Spearman correlation")
    return fig, axes


def region_covariance_matrices(
    prepost: Dict[str, object],
    region: int | str,
    mode: str = "subject",
    measure: str = "corr",  # corr | cov
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    target_meta = prepost["target_meta"]
    subjects = prepost["subjects"]
    genes = prepost["genes"]
    raw_cube = prepost["raw_cube"]
    harm_cube = prepost["harm_cube"]
    obs_mask = prepost["obs_mask"]

    p = _resolve_region_idx(target_meta, region)
    subj_keep = np.where(obs_mask[:, p])[0]
    if int(len(subj_keep)) < 2:
        raise ValueError(
            f"Region {region} (parcel_idx={p}) has <2 observed subjects. "
            "Use `available_regions(prepost)` to choose a valid region."
        )
    X_raw = raw_cube[subj_keep, p, :]
    X_h = harm_cube[subj_keep, p, :]

    use_corr = str(measure).lower() != "cov"
    if str(mode).lower() == "gene":
        # gene-gene relation across subjects at selected parcel
        cov_raw = np.corrcoef(X_raw, rowvar=False) if use_corr else np.cov(X_raw, rowvar=False)
        cov_h = np.corrcoef(X_h, rowvar=False) if use_corr else np.cov(X_h, rowvar=False)
        labels = [str(g) for g in genes]
    else:
        # subject-subject relation across genes at selected parcel
        cov_raw = np.corrcoef(X_raw, rowvar=True) if use_corr else np.cov(X_raw, rowvar=True)
        cov_h = np.corrcoef(X_h, rowvar=True) if use_corr else np.cov(X_h, rowvar=True)
        labels = [str(subjects[i]) for i in subj_keep.tolist()]
    return cov_raw, cov_h, labels


def plot_region_covariance_side_by_side(
    prepost: Dict[str, object],
    region: int | str,
    mode: str = "subject",
    measure: str = "corr",
    figsize: Tuple[float, float] = (11, 4.8),
) -> Tuple[plt.Figure, np.ndarray]:
    cov_raw, cov_h, labels = region_covariance_matrices(prepost, region=region, mode=mode, measure=measure)
    title_mode = "Subject-Subject" if str(mode).lower() != "gene" else "Gene-Gene"
    title_measure = "Correlation" if str(measure).lower() != "cov" else "Covariance"
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if str(measure).lower() != "cov":
        vmin, vmax = -1.0, 1.0
    else:
        vmax = np.nanmax(np.abs(np.r_[cov_raw.ravel(), cov_h.ravel()]))
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        vmin = -vmax
    for ax, mat, title in [
        (axes[0], cov_raw, f"Pre-harmonization ({title_mode} {title_measure})"),
        (axes[1], cov_h, f"Post-ComBat ({title_mode} {title_measure})"),
    ]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")
        ax.set_title(title, fontsize=FONT["title"])
        ax.set_xlabel("Index", fontsize=FONT["label"])
        ax.set_ylabel("Index", fontsize=FONT["label"])
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
        ax.minorticks_off()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Correlation" if str(measure).lower() != "cov" else "Covariance")
    return fig, axes


def compute_subject_metrics_from_cache(
    cfg: EDAConfig,
    model: str,
    mixed_space_mode: str | None = None,
    prepost: Dict[str, object] | None = None,
    refz_noise_quantile: float = 0.10,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    model = str(model).lower()
    root = _model_cache_root(cfg, model)
    if not root.exists():
        raise FileNotFoundError(f"Cache dir not found: {root}")
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm == "harmonized_pred_vs_raw_truth_refz_corr" and prepost is None:
        raise ValueError(
            "mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr' requires prepost=PREPOST "
            "so reference raw/harmonized GTEx statistics can be computed explicitly"
        )
    ref_stats = _reference_gene_stats(prepost, refz_noise_quantile=refz_noise_quantile) if msm == "harmonized_pred_vs_raw_truth_refz_corr" else None
    npz_paths = [str(p) for p in sorted(root.glob("*.npz"))]
    jobs = _effective_n_jobs(len(npz_paths), n_jobs)
    if jobs == 1:
        rows = [
            _subject_metrics_row_from_npz(
                p,
                model=model,
                mixed_space_mode=msm,
                ref_stats=ref_stats,
            )
            for p in npz_paths
        ]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(
                ex.map(
                    _subject_metrics_row_from_npz,
                    npz_paths,
                    [model] * len(npz_paths),
                    [msm] * len(npz_paths),
                    [ref_stats] * len(npz_paths),
                )
            )
    out = pd.DataFrame(rows).sort_values(["coverage", "subject"]).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No npz files found under {root}")
    return out


def compute_subject_metrics_from_cache_gene_subset(
    cfg: EDAConfig,
    model: str,
    eval_gene_mode: str = "all",  # all | hvg | custom
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    mixed_space_mode: str | None = None,
    prepost: Dict[str, object] | None = None,
    refz_noise_quantile: float = 0.10,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    model = str(model).lower()
    root = _model_cache_root(cfg, model)
    if not root.exists():
        raise FileNotFoundError(f"Cache dir not found: {root}")

    mode_label, eval_gene_path_resolved, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm == "harmonized_pred_vs_raw_truth_refz_corr" and prepost is None:
        raise ValueError(
            "mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr' requires prepost=PREPOST "
            "so reference raw/harmonized GTEx statistics can be computed explicitly"
        )
    ref_stats = _reference_gene_stats(prepost, refz_noise_quantile=refz_noise_quantile) if msm == "harmonized_pred_vs_raw_truth_refz_corr" else None

    npz_paths = [str(p) for p in sorted(root.glob("*.npz"))]
    jobs = _effective_n_jobs(len(npz_paths), n_jobs)
    eval_gene_items = None if eval_gene_set is None else tuple(sorted(eval_gene_set))
    if jobs == 1:
        rows = [
            _subject_metrics_subset_row_from_npz(
                p,
                model=model,
                mode_label=mode_label,
                eval_gene_path_resolved=eval_gene_path_resolved,
                eval_gene_items=eval_gene_items,
                mixed_space_mode=msm,
                ref_stats=ref_stats,
            )
            for p in npz_paths
        ]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(
                ex.map(
                    _subject_metrics_subset_row_from_npz,
                    npz_paths,
                    [model] * len(npz_paths),
                    [mode_label] * len(npz_paths),
                    [eval_gene_path_resolved] * len(npz_paths),
                    [eval_gene_items] * len(npz_paths),
                    [msm] * len(npz_paths),
                    [ref_stats] * len(npz_paths),
                )
            )

    out = pd.DataFrame(rows).sort_values(["coverage", "subject"]).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No npz files found under {root}")
    return out


def compute_pooled_pca_recovery_from_cache_gene_subset(
    cfg: EDAConfig,
    num_pcs: int = 10,
    models: Sequence[str] | None = None,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    demean_mode: str = "none",
    mixed_space_mode: str | None = None,
    prepost: Dict[str, object] | None = None,
    refz_noise_quantile: float = 0.20,
    n_jobs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pooled ground-truth PCA recovery from cache-backed strict LORO rows.

    Rows are pooled subject-parcel samples and columns are genes. PCA is fit on
    the truth matrix and each model's pooled prediction matrix is projected into
    that same truth PCA basis for component-wise comparison.

    Demeaning behavior:
    - `demean_mode="none"` uses standard PCA centering only. sklearn PCA
      centers each gene column globally across all pooled samples.
    - `demean_mode="within_parcel"` first subtracts the truth parcel mean
      vector from both truth and prediction rows within each parcel, then PCA
      applies its usual global centering step on the residual matrix.

    Another way to say it:
    - the first step projects out the parcel fixed effect
    - the second step recenters the residual feature space for PCA
    """
    model_list = [str(m).lower() for m in (models if models is not None else MODEL_ORDER)]
    mode_label, eval_gene_path_resolved, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm == "harmonized_pred_vs_raw_truth_refz_corr" and prepost is None:
        raise ValueError(
            "mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr' requires prepost=PREPOST "
            "so reference raw/harmonized GTEx statistics can be computed explicitly"
        )
    ref_stats = (
        _reference_gene_stats(prepost, refz_noise_quantile=refz_noise_quantile)
        if msm == "harmonized_pred_vs_raw_truth_refz_corr"
        else None
    )
    eval_gene_items = None if eval_gene_set is None else tuple(sorted(eval_gene_set))

    blocks_by_model: Dict[str, List[Dict[str, object]]] = {}
    for model in model_list:
        root = _model_cache_root(cfg, model)
        if not root.exists():
            raise FileNotFoundError(f"Cache dir not found: {root}")
        npz_paths = [str(p) for p in sorted(root.glob("*.npz"))]
        jobs = _effective_n_jobs(len(npz_paths), n_jobs)
        if jobs == 1:
            blocks = [
                _pooled_pca_subject_block_from_npz(
                    p,
                    model=model,
                    mode_label=mode_label,
                    eval_gene_path_resolved=eval_gene_path_resolved,
                    eval_gene_items=eval_gene_items,
                    mixed_space_mode=msm,
                    ref_stats=ref_stats,
                )
                for p in npz_paths
            ]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                blocks = list(
                    ex.map(
                        _pooled_pca_subject_block_from_npz,
                        npz_paths,
                        [model] * len(npz_paths),
                        [mode_label] * len(npz_paths),
                        [eval_gene_path_resolved] * len(npz_paths),
                        [eval_gene_items] * len(npz_paths),
                        [msm] * len(npz_paths),
                        [ref_stats] * len(npz_paths),
                    )
                )
        blocks_by_model[str(model)] = blocks

    common_keys = _resolve_pooled_pca_truth_keys(blocks_by_model, model_list)
    X_true, pred_by_model, _ = _build_pooled_truth_pred_mats(blocks_by_model, model_list, common_keys)
    X_true, pred_by_model = _apply_pooled_pca_demean_mode(
        X_true,
        pred_by_model,
        common_keys,
        demean_mode=demean_mode,
    )

    n_samples = int(X_true.shape[0])
    n_genes = int(X_true.shape[1])
    max_pcs = int(min(n_samples, n_genes))
    if int(num_pcs) < 1:
        raise ValueError("num_pcs must be >= 1")
    if int(num_pcs) > max_pcs:
        raise ValueError(f"num_pcs ({num_pcs}) cannot exceed min(n_samples, n_genes) = {max_pcs}")

    # Rows are pooled subject-parcel samples and columns are genes.
    # sklearn PCA centers columns (genes) internally, so no extra transpose
    # or manual centering step is needed here.
    solver = "randomized" if int(num_pcs) < max_pcs else "full"
    pca = PCA(n_components=int(num_pcs), svd_solver=solver, random_state=0)
    C_true = np.asarray(pca.fit_transform(X_true), dtype=np.float64)
    exp_var = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cum_var = np.cumsum(exp_var)
    cutoff_idx_95 = int(np.argmax(cum_var >= 0.95)) if np.any(cum_var >= 0.95) else int(len(cum_var) - 1)

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    truth_rows_added = False
    for model in model_list:
        X_pred = np.asarray(pred_by_model[str(model)], dtype=np.float64)
        C_pred = np.asarray(pca.transform(X_pred), dtype=np.float64)
        pc_corrs: List[float] = []
        weighted_corr_num = 0.0
        weighted_corr_den = 0.0
        for i in range(int(num_pcs)):
            x = C_true[:, i]
            y = C_pred[:, i]
            pear = _pearson_safe(x, y)
            spear = _spearman_safe(x, y)
            r2 = _r2_safe(x, y)
            rmse = _rmse_safe(x, y)
            pc_corrs.append(pear)
            if np.isfinite(pear) and np.isfinite(exp_var[i]):
                weighted_corr_num += float(exp_var[i]) * float(pear)
                weighted_corr_den += float(exp_var[i])
            rows.append(
                {
                    "model": str(model),
                    "component": int(i + 1),
                    "score_pearson": pear,
                    "score_spearman": spear,
                    "score_r2": r2,
                    "score_rmse": rmse,
                    "explained_variance_ratio": float(exp_var[i]),
                    "cumulative_variance": float(cum_var[i]),
                    "pc95_cutoff": int(cutoff_idx_95 + 1),
                    "n_samples": int(n_samples),
                    "n_genes": int(n_genes),
                    "eval_gene_mode": mode_label,
                    "eval_gene_path": eval_gene_path_resolved,
                    "demean_mode": str(demean_mode).lower(),
                    "mixed_space_mode": msm,
                }
            )
        summary_rows.append(
            {
                "model": str(model),
                "mean_score_pearson": float(np.nanmean(pc_corrs)),
                "var_weighted_score_pearson": float(weighted_corr_num / weighted_corr_den) if weighted_corr_den > 0 else np.nan,
                "pc1_score_pearson": float(pc_corrs[0]) if len(pc_corrs) > 0 else np.nan,
                "pc95_cutoff": int(cutoff_idx_95 + 1),
                "n_samples": int(n_samples),
                "n_genes": int(n_genes),
                "num_pcs": int(num_pcs),
                "eval_gene_mode": mode_label,
                "eval_gene_path": eval_gene_path_resolved,
                "demean_mode": str(demean_mode).lower(),
                "mixed_space_mode": msm,
            }
        )
        if not truth_rows_added:
            for i in range(int(num_pcs)):
                x = C_true[:, i]
                pear_t = _pearson_safe(x, x)
                spear_t = _spearman_safe(x, x)
                r2_t = _r2_safe(x, x)
                rmse_t = _rmse_safe(x, x)
                rows.append(
                    {
                        "model": "truth",
                        "component": int(i + 1),
                        "score_pearson": pear_t,
                        "score_spearman": spear_t,
                        "score_r2": r2_t,
                        "score_rmse": rmse_t,
                        "explained_variance_ratio": float(exp_var[i]),
                        "cumulative_variance": float(cum_var[i]),
                        "pc95_cutoff": int(cutoff_idx_95 + 1),
                        "n_samples": int(n_samples),
                        "n_genes": int(n_genes),
                        "eval_gene_mode": mode_label,
                        "eval_gene_path": eval_gene_path_resolved,
                        "demean_mode": str(demean_mode).lower(),
                        "mixed_space_mode": msm,
                    }
                )
            truth_rows_added = True

    pc_df = pd.DataFrame(rows)
    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("model")
        .reindex(model_list)
        .reset_index()
    )
    return pc_df, summary_df


def compute_pooled_pca_variance_spectra_from_cache_gene_subset(
    cfg: EDAConfig,
    num_pcs: int = 50,
    models: Sequence[str] | None = None,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    demean_mode: str = "none",
    mixed_space_mode: str | None = None,
    prepost: Dict[str, object] | None = None,
    refz_noise_quantile: float = 0.20,
    n_jobs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_list = [str(m).lower() for m in (models if models is not None else MODEL_ORDER)]
    mode_label, eval_gene_path_resolved, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm == "harmonized_pred_vs_raw_truth_refz_corr" and prepost is None:
        raise ValueError(
            "mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr' requires prepost=PREPOST "
            "so reference raw/harmonized GTEx statistics can be computed explicitly"
        )
    ref_stats = (
        _reference_gene_stats(prepost, refz_noise_quantile=refz_noise_quantile)
        if msm == "harmonized_pred_vs_raw_truth_refz_corr"
        else None
    )
    eval_gene_items = None if eval_gene_set is None else tuple(sorted(eval_gene_set))

    blocks_by_model: Dict[str, List[Dict[str, object]]] = {}
    for model in model_list:
        root = _model_cache_root(cfg, model)
        if not root.exists():
            raise FileNotFoundError(f"Cache dir not found: {root}")
        npz_paths = [str(p) for p in sorted(root.glob("*.npz"))]
        jobs = _effective_n_jobs(len(npz_paths), n_jobs)
        if jobs == 1:
            blocks = [
                _pooled_pca_subject_block_from_npz(
                    p,
                    model=model,
                    mode_label=mode_label,
                    eval_gene_path_resolved=eval_gene_path_resolved,
                    eval_gene_items=eval_gene_items,
                    mixed_space_mode=msm,
                    ref_stats=ref_stats,
                )
                for p in npz_paths
            ]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                blocks = list(
                    ex.map(
                        _pooled_pca_subject_block_from_npz,
                        npz_paths,
                        [model] * len(npz_paths),
                        [mode_label] * len(npz_paths),
                        [eval_gene_path_resolved] * len(npz_paths),
                        [eval_gene_items] * len(npz_paths),
                        [msm] * len(npz_paths),
                        [ref_stats] * len(npz_paths),
                    )
                )
        blocks_by_model[str(model)] = blocks

    common_keys = _resolve_pooled_pca_truth_keys(blocks_by_model, model_list)
    X_true, pred_by_model, _ = _build_pooled_truth_pred_mats(blocks_by_model, model_list, common_keys)
    X_true, pred_by_model = _apply_pooled_pca_demean_mode(
        X_true,
        pred_by_model,
        common_keys,
        demean_mode=demean_mode,
    )
    n_samples = int(X_true.shape[0])
    n_genes = int(X_true.shape[1])
    max_pcs = int(min(n_samples, n_genes))
    if int(num_pcs) < 1:
        raise ValueError("num_pcs must be >= 1")
    if int(num_pcs) > max_pcs:
        raise ValueError(f"num_pcs ({num_pcs}) cannot exceed min(n_samples, n_genes) = {max_pcs}")

    solver = "randomized" if int(num_pcs) < max_pcs else "full"
    sources: Dict[str, np.ndarray] = {"truth": np.asarray(X_true, dtype=np.float64)}
    for model in model_list:
        sources[str(model)] = np.asarray(pred_by_model[str(model)], dtype=np.float64)

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for source_name, X in sources.items():
        pca = PCA(n_components=int(num_pcs), svd_solver=solver, random_state=0)
        pca.fit(X)
        exp_var = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cum_var = np.cumsum(exp_var)
        cutoff_idx_95 = int(np.argmax(cum_var >= 0.95)) if np.any(cum_var >= 0.95) else int(len(cum_var) - 1)
        for i in range(int(num_pcs)):
            rows.append(
                {
                    "source": str(source_name),
                    "component": int(i + 1),
                    "explained_variance_ratio": float(exp_var[i]),
                    "cumulative_variance": float(cum_var[i]),
                    "pc95_cutoff": int(cutoff_idx_95 + 1),
                    "n_samples": int(n_samples),
                    "n_genes": int(n_genes),
                    "num_pcs": int(num_pcs),
                    "eval_gene_mode": mode_label,
                    "eval_gene_path": eval_gene_path_resolved,
                    "demean_mode": str(demean_mode).lower(),
                    "mixed_space_mode": msm,
                }
            )
        summary_rows.append(
            {
                "source": str(source_name),
                "pc1_explained_variance_ratio": float(exp_var[0]) if len(exp_var) > 0 else np.nan,
                "pc95_cutoff": int(cutoff_idx_95 + 1),
                "n_samples": int(n_samples),
                "n_genes": int(n_genes),
                "num_pcs": int(num_pcs),
                "eval_gene_mode": mode_label,
                "eval_gene_path": eval_gene_path_resolved,
                "demean_mode": str(demean_mode).lower(),
                "mixed_space_mode": msm,
            }
        )

    spectra_df = pd.DataFrame(rows)
    source_order = ["truth"] + model_list
    summary_df = pd.DataFrame(summary_rows).set_index("source").reindex(source_order).reset_index()
    return spectra_df, summary_df


def collect_pooled_sample_prediction_dfs_from_cache_gene_subset(
    cfg: EDAConfig,
    model: str,
    eval_gene_mode: str = "all",
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    mixed_space_mode: str | None = None,
    prepost: Dict[str, object] | None = None,
    refz_noise_quantile: float = 0.20,
    n_jobs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_name = str(model).lower()
    root = _model_cache_root(cfg, model_name)
    if not root.exists():
        raise FileNotFoundError(f"Cache dir not found: {root}")

    mode_label, eval_gene_path_resolved, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )
    msm = _normalize_mixed_space_mode(mixed_space_mode)
    if msm == "harmonized_pred_vs_raw_truth_refz_corr" and prepost is None:
        raise ValueError(
            "mixed_space_mode='harmonized_pred_vs_raw_truth_refz_corr' requires prepost=PREPOST "
            "so reference raw/harmonized GTEx statistics can be computed explicitly"
        )
    ref_stats = (
        _reference_gene_stats(prepost, refz_noise_quantile=refz_noise_quantile)
        if msm == "harmonized_pred_vs_raw_truth_refz_corr"
        else None
    )
    eval_gene_items = None if eval_gene_set is None else tuple(sorted(eval_gene_set))
    npz_paths = [str(p) for p in sorted(root.glob("*.npz"))]
    jobs = _effective_n_jobs(len(npz_paths), n_jobs)
    if jobs == 1:
        blocks = [
            _pooled_pca_subject_block_from_npz(
                p,
                model=model_name,
                mode_label=mode_label,
                eval_gene_path_resolved=eval_gene_path_resolved,
                eval_gene_items=eval_gene_items,
                mixed_space_mode=msm,
                ref_stats=ref_stats,
            )
            for p in npz_paths
        ]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            blocks = list(
                ex.map(
                    _pooled_pca_subject_block_from_npz,
                    npz_paths,
                    [model_name] * len(npz_paths),
                    [mode_label] * len(npz_paths),
                    [eval_gene_path_resolved] * len(npz_paths),
                    [eval_gene_items] * len(npz_paths),
                    [msm] * len(npz_paths),
                    [ref_stats] * len(npz_paths),
                )
            )

    truth_frames: List[pd.DataFrame] = []
    pred_frames: List[pd.DataFrame] = []
    gene_names_ref: Tuple[str, ...] | None = None
    for block in blocks:
        gene_names = tuple(block["gene_names"])
        if gene_names_ref is None:
            gene_names_ref = gene_names
        elif gene_names != gene_names_ref:
            raise ValueError("Gene list mismatch across subjects while collecting pooled sample dfs")
        keys = list(block["keys"])
        subjects = [str(s) for s, _ in keys]
        parcels = [int(p) for _, p in keys]
        sample_index = np.arange(len(keys), dtype=np.int32)
        base = pd.DataFrame(
            {
                "subject": subjects,
                "parcel_idx": parcels,
                "sample_key": [f"{s}|{p}" for s, p in keys],
                "sample_idx_subject": sample_index,
                "model": str(model_name),
                "eval_gene_mode": mode_label,
                "eval_gene_path": eval_gene_path_resolved,
                "mixed_space_mode": msm,
            }
        )
        truth_df = pd.concat(
            [base.copy(), pd.DataFrame(np.asarray(block["truth"], dtype=np.float64), columns=list(gene_names))],
            axis=1,
        )
        pred_df = pd.concat(
            [base.copy(), pd.DataFrame(np.asarray(block["pred"], dtype=np.float64), columns=list(gene_names))],
            axis=1,
        )
        truth_frames.append(truth_df)
        pred_frames.append(pred_df)

    if len(truth_frames) == 0:
        raise RuntimeError(f"No npz files found under {root}")

    truth_all = (
        pd.concat(truth_frames, ignore_index=True)
        .sort_values(["subject", "parcel_idx"])
        .reset_index(drop=True)
    )
    pred_all = (
        pd.concat(pred_frames, ignore_index=True)
        .sort_values(["subject", "parcel_idx"])
        .reset_index(drop=True)
    )
    return truth_all, pred_all


def plot_pooled_pca_variance_spectra(
    spectra_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12.2, 4.6),
    dpi: int = 180,
    panel_label: str | None = None,
    show_fit: str | None = None,
    lowess_frac: float = 0.18,
    x_label_stride: int = 10,
    x_tick_rotation: float = 45.0,
    source: str | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    need = {"source", "component", "explained_variance_ratio", "cumulative_variance", "pc95_cutoff"}
    if not need.issubset(set(spectra_df.columns)):
        raise ValueError(f"spectra_df must include columns: {sorted(need)}")

    d = spectra_df.copy()
    d["source"] = d["source"].astype(str).str.lower()
    source_sel = None if source is None else str(source).lower()
    if source_sel is not None:
        d = d[d["source"] == source_sel].copy()
    source_order = ["truth"] + [m for m in MODEL_ORDER if m in set(d["source"].tolist())]
    d = d[d["source"].isin(source_order)].copy()
    if len(d) == 0:
        raise RuntimeError("No pooled PCA variance spectra rows available to plot")

    n_samples = int(d["n_samples"].iloc[0]) if "n_samples" in d.columns else -1
    n_genes = int(d["n_genes"].iloc[0]) if "n_genes" in d.columns else -1
    demean_mode = str(d["demean_mode"].iloc[0]) if "demean_mode" in d.columns else "none"
    p_lbl = _metrics_panel_label(d.rename(columns={"source": "model"}), panel_label=panel_label)

    color_map = {"truth": "#111111", **MODEL_COLORS}
    label_map = {"truth": "Truth", **MODEL_LABELS}

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    fit_style = None if show_fit is None else str(show_fit).strip().lower()
    if fit_style not in {None, "lowess", "linear", "decay"}:
        raise ValueError("show_fit must be one of: None, lowess, linear, decay")
    if fit_style == "lowess" and _sm_lowess is None:
        raise ImportError("show_fit='lowess' requires statsmodels to be installed")

    def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-b * x) + c

    for source in source_order:
        sub = d[d["source"] == source].sort_values("component").copy()
        if len(sub) == 0:
            continue
        xx = sub["component"].to_numpy(dtype=np.float64)
        yy = sub["explained_variance_ratio"].to_numpy(dtype=np.float64)
        color = color_map.get(source, "#333333")
        label = label_map.get(source, str(source))

        if fit_style is not None:
            axes[0].scatter(xx, yy, s=24.0, alpha=0.50, color=color, edgecolors="none", label=label)
            try:
                valid = np.isfinite(xx) & np.isfinite(yy)
                if fit_style == "decay":
                    valid = valid & (yy > 0)
                if int(np.sum(valid)) >= 4:
                    xfit = xx[valid]
                    yfit = yy[valid]
                    if fit_style == "lowess":
                        smooth = _sm_lowess(yfit, xfit, frac=float(lowess_frac), return_sorted=True)
                        xs = np.asarray(smooth[:, 0], dtype=np.float64)
                        ys = np.asarray(smooth[:, 1], dtype=np.float64)
                    elif fit_style == "linear":
                        slope, intercept = np.polyfit(xfit, yfit, 1)
                        xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 300)
                        ys = slope * xs + intercept
                    else:
                        popt, _ = curve_fit(
                            _exp_decay,
                            xfit,
                            yfit,
                            p0=[float(max(yfit[0] - yfit[-1], 1e-8)), 0.10, float(max(yfit[-1], 1e-8))],
                            bounds=([0.0, 0.0, 0.0], [10.0, 5.0, 1.0]),
                            maxfev=10000,
                        )
                        xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 300)
                        ys = _exp_decay(xs, *popt)
                    axes[0].plot(xs, ys, linewidth=2.0, alpha=0.95, color=color)
                else:
                    axes[0].plot(xx, yy, marker="o", markersize=3.5, linewidth=1.5, color=color, alpha=0.9, label=label)
            except Exception:
                axes[0].plot(xx, yy, marker="o", markersize=3.5, linewidth=1.5, color=color, alpha=0.9, label=label)
        else:
            axes[0].plot(xx, yy, marker="o", markersize=3.5, linewidth=1.5, color=color, alpha=0.9, label=label)

        axes[1].plot(
            sub["component"].to_numpy(dtype=np.float64),
            sub["cumulative_variance"].to_numpy(dtype=np.float64),
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=color,
            alpha=0.9,
            label=label,
        )
        cutoff = int(sub["pc95_cutoff"].iloc[0]) if len(sub) > 0 else -1
        if cutoff >= 1:
            axes[1].axvline(cutoff, color=color, linestyle="--", linewidth=1.0, alpha=0.35)

    comp_ticks = sorted(set(d["component"].astype(int).tolist()))
    stride = max(1, int(x_label_stride))
    for ax in axes:
        ax.set_xticks(comp_ticks)
        ax.set_xticklabels(
            [str(x) if (int(x) % stride == 0 or int(x) == 1) else "" for x in comp_ticks],
            rotation=float(x_tick_rotation),
            ha="right" if float(x_tick_rotation) != 0.0 else "center",
        )
        ax.tick_params(axis="x", which="major", length=4, width=0.9)
        ax.grid(alpha=0.22)

    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Per-Component Variance")
    axes[0].legend(frameon=False, loc="best")

    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].set_title("Cumulative Variance")
    axes[1].set_ylim(-0.01, 1.01)

    title = f"PCA Variance Spectrum ({p_lbl}; n={n_samples}; genes={n_genes})"
    if source_sel is not None:
        title += f" [{label_map.get(source_sel, source_sel)}]"
    if demean_mode not in {"", "none"}:
        title += f" [{demean_mode}]"
    fig.suptitle(title, fontsize=FONT["title"] + 1, y=1.02)

    return fig, axes, d.copy()


def plot_pooled_pca_recovery(
    pc_df: pd.DataFrame,
    metric: str = "score_pearson",
    figsize: Tuple[float, float] = (9.2, 4.6),
    dpi: int = 180,
    panel_label: str | None = None,
    show_fit: str | None = None,
    lowess_frac: float = 0.18,
    show_lines: bool = True,
    x_label_stride: int = 10,
    x_tick_rotation: float = 45.0,
    show_truth: bool = False,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    need = {"model", "component", "explained_variance_ratio", "cumulative_variance", "pc95_cutoff", metric}
    if not need.issubset(set(pc_df.columns)):
        raise ValueError(f"pc_df must include columns: {sorted(need)}")
    if metric not in {"score_pearson", "score_spearman", "score_r2", "score_rmse"}:
        raise ValueError("metric must be one of: score_pearson, score_spearman, score_r2, score_rmse")

    d = pc_df.copy()
    d["model"] = d["model"].astype(str).str.lower()
    d_truth = d[d["model"] == "truth"].copy()
    d = d[d["model"].isin(MODEL_ORDER)].copy()
    if len(d) == 0 and (not bool(show_truth) or len(d_truth) == 0):
        raise RuntimeError("No pooled PCA rows available to plot")

    d = d.sort_values(["model", "component"]).reset_index(drop=True)
    d_ref = d if len(d) else d_truth
    n_samples = int(d_ref["n_samples"].iloc[0]) if "n_samples" in d_ref.columns else -1
    n_genes = int(d_ref["n_genes"].iloc[0]) if "n_genes" in d_ref.columns else -1
    cutoff = int(d_ref["pc95_cutoff"].iloc[0]) if "pc95_cutoff" in d_ref.columns else -1
    p_lbl = _metrics_panel_label(d_ref, panel_label=panel_label)
    demean_mode = str(d_ref["demean_mode"].iloc[0]) if "demean_mode" in d_ref.columns else "none"
    fit_style = None if show_fit is None else str(show_fit).strip().lower()
    if fit_style not in {None, "lowess", "linear", "decay"}:
        raise ValueError("show_fit must be one of: None, lowess, linear, decay")
    if fit_style == "lowess" and _sm_lowess is None:
        raise ImportError("show_fit='lowess' requires statsmodels to be installed")

    if metric == "score_pearson":
        ylab = "PC Score Pearson r"
    elif metric == "score_spearman":
        ylab = "PC Score Spearman r"
    elif metric == "score_r2":
        ylab = r"PC Score $R^2$"
    else:
        ylab = "PC Score RMSE"

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    for model in MODEL_ORDER:
        sub = d[d["model"] == model].copy()
        if len(sub) == 0:
            continue
        xx = sub["component"].to_numpy(dtype=np.float64)
        yy = sub[metric].to_numpy(dtype=np.float64)
        if fit_style is not None:
            ax.scatter(
                xx,
                yy,
                s=26.0,
                alpha=0.55,
                color=MODEL_COLORS[model],
                edgecolors="none",
                label=MODEL_LABELS[model],
            )
            try:
                valid = np.isfinite(xx) & np.isfinite(yy)
                if fit_style == "decay":
                    valid = valid & (yy > 0)
                if int(np.sum(valid)) >= 4:
                    xfit = xx[valid]
                    yfit = yy[valid]
                    if fit_style == "lowess":
                        smooth = _sm_lowess(yfit, xfit, frac=float(lowess_frac), return_sorted=True)
                        xs = np.asarray(smooth[:, 0], dtype=np.float64)
                        ys = np.asarray(smooth[:, 1], dtype=np.float64)
                    elif fit_style == "linear":
                        slope, intercept = np.polyfit(xfit, yfit, 1)
                        xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 300)
                        ys = slope * xs + intercept
                    else:
                        def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                            return a * np.exp(-b * x) + c
                        popt, _ = curve_fit(
                            _exp_decay,
                            xfit,
                            yfit,
                            p0=[float(yfit[0] - yfit[-1]), 0.15, float(yfit[-1])],
                            maxfev=10000,
                        )
                        xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 300)
                        ys = _exp_decay(xs, *popt)
                    ax.plot(
                        xs,
                        ys,
                        linewidth=2.0,
                        alpha=0.95,
                        color=MODEL_COLORS[model],
                    )
            except Exception:
                pass
        else:
            if bool(show_lines):
                ax.plot(
                    xx,
                    yy,
                    marker="o",
                    markersize=4.0,
                    linewidth=1.8,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
            else:
                ax.scatter(
                    xx,
                    yy,
                    s=28.0,
                    alpha=0.9,
                    color=MODEL_COLORS[model],
                    edgecolors="none",
                    label=MODEL_LABELS[model],
                )

    if bool(show_truth):
        if len(d_truth) == 0:
            raise ValueError("show_truth=True requires pc_df to include computed truth rows")
        d_truth = d_truth.sort_values("component").copy()
        tx = d_truth["component"].to_numpy(dtype=np.float64)
        ty = d_truth[metric].to_numpy(dtype=np.float64)
        if fit_style is not None:
            ax.scatter(
                tx,
                ty,
                s=24.0,
                alpha=0.75,
                color="#111111",
                edgecolors="white",
                linewidths=0.3,
                label="Truth",
                zorder=4,
            )
        else:
            ax.scatter(
                tx,
                ty,
                s=30.0,
                alpha=0.85,
                color="#111111",
                edgecolors="white",
                linewidths=0.3,
                label="Truth",
                zorder=4,
            )

    cutoff_label = None
    if cutoff >= 1:
        ax.axvline(cutoff, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
        cutoff_label = "95% true-data variance explained"

    comp_ticks = sorted(set(d["component"].astype(int).tolist()))
    stride = max(1, int(x_label_stride))
    ax.set_xticks(comp_ticks)
    ax.set_xticklabels(
        [str(x) if (int(x) % stride == 0 or int(x) == 1) else "" for x in comp_ticks],
        rotation=float(x_tick_rotation),
        ha="right" if float(x_tick_rotation) != 0.0 else "center",
    )
    ax.tick_params(axis="x", which="major", length=4, width=0.9)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel(ylab)
    title = f"Ground-Truth PCA Recovery ({p_lbl}; n={n_samples}; genes={n_genes})"
    if demean_mode not in {"", "none"}:
        title += f" [{demean_mode}]"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if cutoff_label is not None:
        handles.append(Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.1, alpha=0.8))
        labels.append(cutoff_label)
    ax.legend(handles, labels, frameon=False, loc="best")

    if metric in {"score_pearson", "score_spearman", "score_r2"}:
        yvals = [d[metric].to_numpy(dtype=np.float64)]
        if bool(show_truth) and len(d_truth) > 0:
            yvals.append(d_truth[metric].to_numpy(dtype=np.float64))
        y_all = np.concatenate(yvals)
        ymin = float(np.nanmin(y_all))
        ymax = float(np.nanmax(y_all))
        ax.set_ylim(min(-0.05, ymin - 0.05), max(1.0, ymax + 0.03))

    return fig, ax, d.copy()


def plot_coverage_vs_accuracy(
    metrics_df: pd.DataFrame,
    metric: str = "pearson_r",
    figsize: Tuple[float, float] = (7.2, 4.2),
    panel_label: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if metric not in {"pearson_r", "rmse"}:
        raise ValueError("metric must be one of: pearson_r, rmse")
    d = (
        metrics_df.groupby(["model", "coverage"], as_index=False)[metric]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n_subjects"})
    )
    d["sem"] = d["metric_std"] / np.sqrt(np.maximum(d["n_subjects"], 1))
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    for model in MODEL_ORDER:
        sub = d[d["model"] == model].sort_values("coverage")
        if len(sub) == 0:
            continue
        ax.plot(sub["coverage"], sub["metric_mean"], marker="o", linewidth=1.6, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(
            sub["coverage"].to_numpy(dtype=np.float64),
            (sub["metric_mean"] - sub["sem"]).to_numpy(dtype=np.float64),
            (sub["metric_mean"] + sub["sem"]).to_numpy(dtype=np.float64),
            color=MODEL_COLORS[model],
            alpha=0.15,
        )
    ax.set_xlabel("GTEx observed parcels per subject", fontsize=FONT["label"])
    ylabel = "Mean LORO Pearson r" if metric == "pearson_r" else "Mean LORO RMSE"
    ax.set_ylabel(ylabel, fontsize=FONT["label"])
    p_lbl = _metrics_panel_label(metrics_df, panel_label=panel_label)
    ttl = "Coverage vs LORO accuracy" if p_lbl == "" else f"Coverage vs LORO accuracy ({p_lbl})"
    ax.set_title(ttl, fontsize=FONT["title"])
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    return fig, ax


def plot_loro_subject_summary_bars(
    metrics_df: pd.DataFrame,
    figsize: Tuple[float, float] = (19.4, 4.25),
    use_sem: bool = True,
    panel_label: str | None = None,
    disable_metrics: Sequence[str] | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    need = {"model", "pearson_r", "spearman_r", "r2", "rmse"}
    if not need.issubset(set(metrics_df.columns)):
        raise ValueError(f"metrics_df must include columns: {sorted(need)}")

    d = metrics_df.copy()
    d["model"] = d["model"].astype(str).str.lower()
    d = d[d["model"].isin(MODEL_ORDER)].copy()

    summ = (
        d.groupby("model", as_index=False)
        .agg(
            pearson_mean=("pearson_r", "mean"),
            pearson_std=("pearson_r", "std"),
            spearman_mean=("spearman_r", "mean"),
            spearman_std=("spearman_r", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_subjects=("subject", "nunique"),
        )
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    if bool(use_sem):
        n = np.maximum(summ["n_subjects"].to_numpy(dtype=np.float64), 1.0)
        summ["pearson_err"] = summ["pearson_std"] / np.sqrt(n)
        summ["spearman_err"] = summ["spearman_std"] / np.sqrt(n)
        summ["r2_err"] = summ["r2_std"] / np.sqrt(n)
        summ["rmse_err"] = summ["rmse_std"] / np.sqrt(n)
    else:
        summ["pearson_err"] = summ["pearson_std"]
        summ["spearman_err"] = summ["spearman_std"]
        summ["r2_err"] = summ["r2_std"]
        summ["rmse_err"] = summ["rmse_std"]

    p_lbl = _metrics_panel_label(metrics_df, panel_label=panel_label)
    disabled = {str(x).lower() for x in (disable_metrics or [])}
    n_subs = summ["n_subjects"].to_numpy(dtype=np.int32)
    metric_specs = [
        ("pearson", "Pearson r", True),
        ("spearman", "Spearman r", True),
        ("r2", r"$R^2$", True),
        ("rmse", "RMSE", False),
    ]
    metric_specs = [spec for spec in metric_specs if spec[0] not in disabled]
    if len(metric_specs) == 0:
        raise ValueError("All metrics were disabled; at least one panel must remain")

    n_panels = int(len(metric_specs))
    base_w, base_h = float(figsize[0]), float(figsize[1])
    per_panel_w = base_w / 4.0
    fig_w = per_panel_w * float(n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, base_h), constrained_layout=False)
    if n_panels == 1:
        axes = np.asarray([axes], dtype=object)
    else:
        axes = np.asarray(axes, dtype=object)
    fig.subplots_adjust(wspace=0.36, top=0.86)
    x = np.arange(len(summ), dtype=np.int32)
    labels = [MODEL_LABELS[m] for m in summ["model"].tolist()]
    colors = [MODEL_COLORS[m] for m in summ["model"].tolist()]
    n_pool = int(np.nanmax(summ["n_subjects"].to_numpy(dtype=np.float64))) if len(summ) else 0
    ttl_suffix = f"n={n_pool}" if p_lbl == "" else f"n={n_pool}; {p_lbl}"
    main_title = f"Mean Subject-wise LORO ({ttl_suffix})"
    fig.suptitle(main_title, fontsize=FONT["title"] + 8, y=0.98)

    for ax, (prefix, ylabel, is_corr_like) in zip(axes, metric_specs):
        mean = summ[f"{prefix}_mean"].to_numpy(dtype=np.float64)
        err = np.nan_to_num(summ[f"{prefix}_err"].to_numpy(dtype=np.float64), nan=0.0)
        top = mean + err
        bot = mean - err

        ax.bar(
            x,
            mean,
            yerr=err,
            color=colors,
            alpha=0.92,
            capsize=4,
            ecolor="#3a3a3a",
        )
        ax.set_ylabel(ylabel, fontsize=FONT["label"] + 4)
        ax.set_xticks(x.tolist())
        ax.set_xticklabels(labels, rotation=0, fontsize=FONT["tick"] + 4)
        ax.tick_params(axis="y", labelsize=FONT["tick"] + 3)
        ax.grid(True, axis="y", alpha=0.2)

        finite_top = top[np.isfinite(top)]
        finite_bot = bot[np.isfinite(bot)]
        if finite_top.size == 0 or finite_bot.size == 0:
            y_lo, y_hi = (-1.0, 1.0) if is_corr_like else (0.0, 1.0)
        else:
            y_min = float(np.min(finite_bot))
            y_max = float(np.max(finite_top))
            span = max(y_max - y_min, 1e-6)
            upper_pad = 0.12 * span
            lower_pad = 0.20 * span
            if is_corr_like:
                y_lo = max(-1.0, y_min - lower_pad)
                y_hi = min(1.0, y_max + 2.2 * upper_pad)
                if y_hi - y_lo < 0.08:
                    extra = 0.04
                    y_lo = max(-1.0, y_lo - extra)
                    y_hi = min(1.0, y_hi + extra)
            else:
                y_lo = max(0.0, y_min - lower_pad)
                y_hi = y_max + 2.2 * upper_pad
        ax.set_ylim(y_lo, y_hi)

        y_lo, y_hi = ax.get_ylim()
        label_pad = max(0.012, 0.02 * (y_hi - y_lo))
        if is_corr_like:
            for xi, val, low in zip(x.tolist(), mean.tolist(), bot.tolist()):
                y_txt = max(float(low - label_pad), float(y_lo + 0.01 * (y_hi - y_lo)))
                ax.text(
                    xi,
                    y_txt,
                    f"({val:.3f})",
                    ha="center",
                    va="top",
                    fontsize=FONT["small"] + 3,
                )
        else:
            for xi, val, high in zip(x.tolist(), mean.tolist(), top.tolist()):
                ax.text(
                    xi,
                    float(high + label_pad),
                    f"({val:.3f})",
                    ha="center",
                    va="bottom",
                    fontsize=FONT["small"] + 3,
                )

    return fig, axes, summ


def _parse_fold_key(fold_key: str) -> Tuple[int, List[int]]:
    s = str(fold_key)
    parts = s.split("|")
    hold = int(parts[0].split("=")[1].strip())
    train_str = parts[1].split("=")[1].strip() if len(parts) > 1 else ""
    train = [int(x) for x in train_str.split(",") if str(x).strip() != ""]
    return hold, train


def _target_coords_matrix(prepost: Dict[str, object]) -> np.ndarray:
    target = prepost["target_meta"].copy()
    target = target.sort_values("parcel_idx").reset_index(drop=True)
    parcel_idx = target["parcel_idx"].to_numpy(dtype=np.int32)
    if not np.array_equal(parcel_idx, np.arange(len(target), dtype=np.int32)):
        raise ValueError("target_meta parcel_idx must be contiguous and 0-based for fold-distance computation")
    return target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)


def _fold_distance_metrics(
    hold: int,
    train: Sequence[int],
    coords: np.ndarray,
) -> Dict[str, float]:
    train_idx = np.asarray([int(x) for x in train], dtype=np.int32)
    if int(train_idx.size) == 0:
        return {
            "dist_to_nearest_train": np.nan,
            "dist_to_centroid_train": np.nan,
        }
    hold_xyz = coords[int(hold), :]
    train_xyz = coords[train_idx, :]
    d = np.linalg.norm(train_xyz - hold_xyz[None, :], axis=1)
    train_centroid_xyz = np.mean(train_xyz, axis=0)
    return {
        "dist_to_nearest_train": float(np.min(d)),
        "dist_to_centroid_train": float(np.linalg.norm(train_centroid_xyz - hold_xyz)),
    }


def _attach_fold_distance_columns(df: pd.DataFrame, prepost: Dict[str, object]) -> pd.DataFrame:
    if len(df) == 0 or "fold_key" not in df.columns:
        return df
    coords = _target_coords_matrix(prepost)
    lookup: Dict[str, Dict[str, float]] = {}
    for fk in pd.Series(df["fold_key"]).astype(str).drop_duplicates().tolist():
        hold, train = _parse_fold_key(fk)
        lookup[fk] = _fold_distance_metrics(hold, train, coords)
    dist_df = (
        pd.DataFrame.from_dict(lookup, orient="index")
        .rename_axis("fold_key")
        .reset_index()
    )
    return df.merge(dist_df, on="fold_key", how="left")


def _parcel_to_gtex_label_map(prepost: Dict[str, object]) -> Dict[int, str]:
    g = prepost["gtex_eligible_raw"].copy()
    mapping = (
        g.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )
    ahba = dict(
        zip(
            prepost["target_meta"]["parcel_idx"].astype(int).tolist(),
            prepost["target_meta"]["tissue_or_parcel"].astype(str).tolist(),
        )
    )
    out: Dict[int, str] = {}
    for p, a in ahba.items():
        gname = str(mapping.get(int(p), ""))
        out[int(p)] = gname if gname else str(a)
    return out


def compute_fold_combo_metrics_from_cache(
    cfg: EDAConfig,
    prepost: Dict[str, object],
    eval_gene_mode: str = "all",  # all | hvg | custom
    custom_gene_list: Sequence[str] | None = None,
    coverage_min: int = 5,
    coverage_max: int = 12,
    models: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    n_jobs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_list = [str(m).lower() for m in (models if models is not None else MODEL_ORDER)]
    mode_label, eval_gene_path_resolved, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )

    g = prepost["gtex_eligible_raw"].copy()
    subj_obs = (
        g.groupby("subject")["parcel_idx"]
        .apply(lambda s: sorted(set(int(v) for v in s.tolist())))
        .to_dict()
    )

    rows: List[Dict[str, object]] = []
    eval_gene_items = None if eval_gene_set is None else tuple(sorted(eval_gene_set))
    for model in model_list:
        model_root = _model_cache_root(cfg, model)
        if not model_root.exists():
            continue
        subject_jobs: List[tuple[str, str, List[int]]] = []
        for sid, obs in subj_obs.items():
            n_obs = int(len(obs))
            if n_obs < int(coverage_min) or n_obs > int(coverage_max):
                continue

            npz_path = model_root / f"{sid}.npz"
            if not npz_path.exists():
                continue
            subject_jobs.append((str(npz_path), str(sid), [int(x) for x in obs]))

        jobs = _effective_n_jobs(len(subject_jobs), n_jobs)
        if jobs == 1:
            for npz_path_str, sid, obs in subject_jobs:
                rows.extend(
                    _fold_rows_for_subject_npz(
                        npz_path_str,
                        model=model,
                        subject_id=sid,
                        obs=obs,
                        mode_label=mode_label,
                        eval_gene_path_resolved=eval_gene_path_resolved,
                        eval_gene_items=eval_gene_items,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                mapped = ex.map(
                    _fold_rows_for_subject_npz,
                    [p for p, _, _ in subject_jobs],
                    [model] * len(subject_jobs),
                    [sid for _, sid, _ in subject_jobs],
                    [obs for _, _, obs in subject_jobs],
                    [mode_label] * len(subject_jobs),
                    [eval_gene_path_resolved] * len(subject_jobs),
                    [eval_gene_items] * len(subject_jobs),
                )
                for block in mapped:
                    rows.extend(block)

    fold_perf_df = pd.DataFrame(rows)
    if len(fold_perf_df) == 0:
        raise RuntimeError("No fold-level rows were computed from cache")
    fold_perf_df = _attach_fold_distance_columns(fold_perf_df, prepost)

    combo_df = (
        fold_perf_df.groupby(["coverage", "fold_key", "model"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_pearson=("pearson_r", "mean"),
            std_pearson=("pearson_r", "std"),
            mean_spearman=("spearman_r", "mean"),
            std_spearman=("spearman_r", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_points=("n_points", "mean"),
            dist_to_nearest_train=("dist_to_nearest_train", "first"),
            dist_to_centroid_train=("dist_to_centroid_train", "first"),
        )
        .sort_values(["model", "coverage", "fold_key"])
        .reset_index(drop=True)
    )
    combo_df["eval_gene_mode"] = mode_label
    combo_df["eval_gene_path"] = eval_gene_path_resolved
    return fold_perf_df, combo_df


def plot_fold_combo_ranked(
    combo_df: pd.DataFrame,
    prepost: Dict[str, object],
    model: str = "dlam",
    metric: str = "mean_pearson",  # mean_pearson | mean_spearman | mean_r2 | mean_rmse
    figsize: Tuple[float, float] = (13.8, 7.2),
    dpi: int = 180,
    show_fold_xticklabels: bool = False,
    show_y_axis_label: bool = True,
    style: str = "line+points",  # line+points | points | line
    points_style: str = "solid",  # solid | coverage | dist_to_nearest_train | dist_to_centroid_train
    show_running_average: bool = False,
    running_average_window: int = 20,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    m = str(model).lower()
    if metric not in {"mean_pearson", "mean_spearman", "mean_r2", "mean_rmse"}:
        raise ValueError("metric must be one of: mean_pearson, mean_spearman, mean_r2, mean_rmse")
    style_l = str(style).lower()
    if style_l not in {"line+points", "points", "line"}:
        raise ValueError("style must be one of: line+points, points, line")
    points_style_l = str(points_style).lower()
    if points_style_l not in {"solid", "coverage", "dist_to_nearest_train", "dist_to_centroid_train"}:
        raise ValueError("points_style must be one of: solid, coverage, dist_to_nearest_train, dist_to_centroid_train")

    d = combo_df[combo_df["model"].astype(str).str.lower() == m].copy()
    if len(d) == 0:
        raise RuntimeError(f"No combo rows found for model={m}")

    # Left->right worst->best.
    if metric in {"mean_pearson", "mean_spearman", "mean_r2"}:
        d = d.sort_values(metric, ascending=True).reset_index(drop=True)
        if metric == "mean_pearson":
            ylab = "Mean fold Pearson r"
        elif metric == "mean_spearman":
            ylab = "Mean fold Spearman r"
        else:
            ylab = r"Mean fold $R^2$"
    else:
        d = d.sort_values(metric, ascending=False).reset_index(drop=True)
        ylab = "Mean fold RMSE"

    x = np.arange(len(d), dtype=np.int32)
    y = d[metric].to_numpy(dtype=np.float64)
    running_average_window = max(1, int(running_average_window))
    point_size = 12.0 if bool(show_running_average) and style_l in {"line+points", "points"} else (10.0 if style_l == "points" else 6.5)
    y_plot = (
        pd.Series(y).rolling(window=running_average_window, min_periods=1).mean().to_numpy(dtype=np.float64)
        if bool(show_running_average)
        else y
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=int(dpi), constrained_layout=False)
    base_color = MODEL_COLORS.get(m, "#333333")
    if style_l in {"line+points", "line"}:
        ax.plot(x, y_plot, color=base_color, linewidth=1.7, alpha=0.98)
    if style_l in {"line+points", "points"}:
        if points_style_l in {"coverage", "dist_to_nearest_train", "dist_to_centroid_train"}:
            if points_style_l == "coverage":
                cvals = d["coverage"].to_numpy(dtype=np.float64)
                cbar_label = "Coverage"
            elif points_style_l == "dist_to_nearest_train":
                cvals = d["dist_to_nearest_train"].to_numpy(dtype=np.float64)
                cbar_label = "Dist to Nearest Train"
            else:
                cvals = d["dist_to_centroid_train"].to_numpy(dtype=np.float64)
                cbar_label = "Dist to Centroid Train"
            cmin = float(np.nanmin(cvals))
            cmax = float(np.nanmax(cvals))
            if np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin:
                base_rgb = np.asarray(to_rgb(base_color), dtype=np.float64)
                light_rgb = 1.0 - 0.22 * (1.0 - base_rgb)
                dark_rgb = np.clip(base_rgb * 0.78, 0.0, 1.0)
                model_cmap = LinearSegmentedColormap.from_list(
                    f"{m}_{points_style_l}",
                    [tuple(light_rgb.tolist()), tuple(base_rgb.tolist()), tuple(dark_rgb.tolist())],
                )
                sc = ax.scatter(
                    x,
                    y_plot,
                    s=point_size,
                    c=cvals,
                    cmap=model_cmap,
                    vmin=cmin,
                    vmax=cmax,
                    alpha=0.88 if style_l == "points" else 0.76,
                    linewidths=0,
                )
                cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.015)
                cbar.set_label(cbar_label, fontsize=FONT["label"] + 1)
                cbar.ax.tick_params(labelsize=FONT["tick"] + 1)
            else:
                ax.scatter(
                    x,
                    y_plot,
                    s=point_size,
                    color=base_color,
                    alpha=0.88 if style_l == "points" else 0.76,
                    linewidths=0,
                )
        else:
            ax.scatter(
                x,
                y_plot,
                s=point_size,
                color=base_color,
                alpha=0.88 if style_l == "points" else 0.70,
                linewidths=0,
            )
    n_combo = int(len(d))
    title_suffix = f"{style_l}"
    if bool(show_running_average):
        title_suffix += f" + trailing mean({running_average_window})"
    ax.set_title(
        f"{MODEL_LABELS.get(m, m.upper())}: fold-combo ranking ({metric}; LORO combinations={n_combo:,}; {title_suffix})",
        fontsize=FONT["title"] + 6,
    )
    ax.set_xlabel("Fold combo (worst -> best)", fontsize=FONT["label"] + 5)
    if bool(show_y_axis_label):
        ax.set_ylabel(ylab, fontsize=FONT["label"] + 5)
    else:
        ax.set_ylabel("")
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="y", labelsize=FONT["tick"] + 4)

    # Show only worst/median/best fold_key labels on x-axis.
    idx_w = 0
    idx_m = int(len(d) // 2)
    idx_b = int(len(d) - 1)
    tick_idx = [idx_w, idx_m, idx_b]
    def _compact_fold_tick(row: pd.Series) -> str:
        hold, train = _parse_fold_key(str(row["fold_key"]))
        train_txt = ",".join(str(t) for t in train)
        return f"test:{hold}\ntrain:{train_txt}"

    tick_lbl = [_compact_fold_tick(d.iloc[idx_w]), _compact_fold_tick(d.iloc[idx_m]), _compact_fold_tick(d.iloc[idx_b])]
    if bool(show_fold_xticklabels):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_lbl, rotation=90, ha="center", fontsize=FONT["tick"] + 3)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    fig.subplots_adjust(bottom=0.28, right=0.74)

    # Decode fold keys to GTEx-native tissue labels and print under the plot.
    p2g = _parcel_to_gtex_label_map(prepost)
    picks = []
    for tag, ix in [("worst", idx_w), ("median", idx_m), ("best", idx_b)]:
        row = d.iloc[ix]
        hold, train = _parse_fold_key(str(row["fold_key"]))
        hold_label = p2g.get(int(hold), str(hold))
        train_labels = [p2g.get(int(t), str(t)) for t in train]
        picks.append(
            {
                "rank_tag": tag,
                "model": m,
                "metric": metric,
                "score": float(row[metric]),
                "coverage": int(row["coverage"]),
                "n_subjects": int(row["n_subjects"]),
                "fold_key": str(row["fold_key"]),
                "hold_parcel": int(hold),
                "hold_gtex": str(hold_label),
                "train_parcels": train,
                "train_gtex": train_labels,
            }
        )

    pick_df = pd.DataFrame(picks)
    lines = []
    def _wrap_words(s: str, n_words: int = 7) -> str:
        toks = str(s).split()
        if len(toks) <= n_words:
            return str(s)
        chunks = [" ".join(toks[i : i + n_words]) for i in range(0, len(toks), n_words)]
        return "\n".join(chunks)
    for _, r in pick_df.iterrows():
        train_txt = _wrap_words(", ".join(r["train_gtex"]), n_words=7)
        line = (
            f"{r['rank_tag'].upper()}\n"
            f"hold: {r['hold_gtex']}\n"
            f"train: {train_txt}\n"
            f"score={r['score']:.3f} | coverage={int(r['coverage'])} | n_subj={int(r['n_subjects'])}"
        )
        lines.append(line)
    fig.text(
        0.755,
        0.50,
        "\n\n".join(lines),
        ha="left",
        va="center",
        fontsize=FONT["tick"] + 2,
        bbox={"facecolor": "white", "edgecolor": "#8a8a8a", "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )
    return fig, ax, pick_df


def summarize_heldout_region_performance(
    fold_perf_df: pd.DataFrame,
    prepost: Dict[str, object],
) -> pd.DataFrame:
    need = {"model", "hold_parcel", "pearson_r", "spearman_r", "r2", "rmse", "subject"}
    if not need.issubset(set(fold_perf_df.columns)):
        raise ValueError(f"fold_perf_df must include columns: {sorted(need)}")

    p2g = _parcel_to_gtex_label_map(prepost)
    p2a = dict(
        zip(
            prepost["target_meta"]["parcel_idx"].astype(int).tolist(),
            prepost["target_meta"]["tissue_or_parcel"].astype(str).tolist(),
        )
    )

    d = fold_perf_df.copy()
    d["model"] = d["model"].astype(str).str.lower()
    d = d[d["model"].isin(MODEL_ORDER)].copy()
    d["hold_parcel"] = d["hold_parcel"].astype(int)

    out = (
        d.groupby(["hold_parcel", "model"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            n_folds=("subject", "size"),
            mean_pearson=("pearson_r", "mean"),
            std_pearson=("pearson_r", "std"),
            mean_spearman=("spearman_r", "mean"),
            std_spearman=("spearman_r", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
        )
    )
    out["gtex_label"] = out["hold_parcel"].map(lambda p: p2g.get(int(p), str(p)))
    out["ahba_label"] = out["hold_parcel"].map(lambda p: p2a.get(int(p), str(p)))
    out["label"] = out["gtex_label"]
    return out.sort_values(["hold_parcel", "model"]).reset_index(drop=True)


def plot_heldout_region_grouped_bars(
    heldout_df: pd.DataFrame,
    metric: str = "mean_pearson",  # mean_pearson | mean_spearman | mean_r2 | mean_rmse
    sort_by_model: str = "dlam",
    use_error_bars: bool = False,
    error_kind: str = "std",  # std | sem
    figsize: Tuple[float, float] = (16.0, 7.2),
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    if metric not in {"mean_pearson", "mean_spearman", "mean_r2", "mean_rmse"}:
        raise ValueError("metric must be one of: mean_pearson, mean_spearman, mean_r2, mean_rmse")
    if error_kind not in {"std", "sem"}:
        raise ValueError("error_kind must be one of: std, sem")

    d = heldout_df.copy()
    d["model"] = d["model"].astype(str).str.lower()
    d = d[d["model"].isin(MODEL_ORDER)].copy()
    if len(d) == 0:
        raise RuntimeError("No rows for expected models in heldout_df")

    pivot = d.pivot_table(index=["hold_parcel", "label"], columns="model", values=metric, aggfunc="first").reset_index()
    sort_col = str(sort_by_model).lower()
    if sort_col not in pivot.columns:
        # fall back to first available model
        sort_col = [c for c in MODEL_ORDER if c in pivot.columns][0]

    # Worst->best on x-axis.
    asc = True if metric in {"mean_pearson", "mean_spearman", "mean_r2"} else False
    pivot = pivot.sort_values(sort_col, ascending=asc).reset_index(drop=True)

    # long form aligned to sorted region order
    key_order = pivot[["hold_parcel", "label"]].copy()
    key_order["ord"] = np.arange(len(key_order), dtype=np.int32)
    dd = d.merge(key_order, on=["hold_parcel", "label"], how="inner")
    dd = dd.sort_values(["ord", "model"]).reset_index(drop=True)

    # error bars
    err_col = None
    if use_error_bars:
        if metric == "mean_pearson":
            base = "std_pearson"
        elif metric == "mean_spearman":
            base = "std_spearman"
        elif metric == "mean_r2":
            base = "std_r2"
        else:
            base = "std_rmse"
        if error_kind == "std":
            err_col = base
            dd["err"] = dd[err_col].astype(float)
        else:
            n = np.maximum(dd["n_subjects"].to_numpy(dtype=np.float64), 1.0)
            dd["err"] = dd[base].to_numpy(dtype=np.float64) / np.sqrt(n)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    n_regions = int(len(key_order))
    x = np.arange(n_regions, dtype=np.float64)
    bw = 0.26
    offsets = {"naive": -bw, "dlam": 0.0, "plam": bw}

    for model in MODEL_ORDER:
        sub = dd[dd["model"] == model].sort_values("ord")
        if len(sub) == 0:
            continue
        xx = x + offsets[model]
        yy = sub[metric].to_numpy(dtype=np.float64)
        if use_error_bars:
            ye = sub["err"].to_numpy(dtype=np.float64)
            ax.bar(xx, yy, width=bw * 0.95, color=MODEL_COLORS[model], alpha=0.9, label=MODEL_LABELS[model], yerr=ye, capsize=2, ecolor="#444444")
        else:
            ax.bar(xx, yy, width=bw * 0.95, color=MODEL_COLORS[model], alpha=0.9, label=MODEL_LABELS[model])

    ax.set_title(
        f"Held-out region performance by model ({metric}, sorted by {sort_col.upper()} worst->best)",
        fontsize=FONT["title"] + 5,
    )
    ax.set_xlabel("Held-out region (GTEx label, sorted)", fontsize=FONT["label"] + 4)
    if metric == "mean_pearson":
        ylab = "Mean Pearson r"
    elif metric == "mean_spearman":
        ylab = "Mean Spearman r"
    elif metric == "mean_r2":
        ylab = r"Mean $R^2$"
    else:
        ylab = "Mean RMSE"
    ax.set_ylabel(ylab, fontsize=FONT["label"] + 4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=FONT["legend"] + 4)
    ax.tick_params(axis="y", labelsize=FONT["tick"] + 4)

    y_all = dd[metric].to_numpy(dtype=np.float64)
    if use_error_bars and "err" in dd.columns:
        y_err = dd["err"].to_numpy(dtype=np.float64)
        y_lo = y_all - y_err
        y_hi = y_all + y_err
    else:
        y_lo = y_all
        y_hi = y_all
    finite_lo = y_lo[np.isfinite(y_lo)]
    finite_hi = y_hi[np.isfinite(y_hi)]
    if finite_lo.size and finite_hi.size:
        y_min = float(np.min(finite_lo))
        y_max = float(np.max(finite_hi))
        if metric in {"mean_pearson", "mean_spearman", "mean_r2"}:
            pad = max(0.05, 0.16 * max(y_max - y_min, 1e-6))
            lo = max(-1.0, y_min - pad)
            hi = min(1.0, y_max + pad)
            if hi - lo < 0.12:
                mid = 0.5 * (hi + lo)
                lo = max(-1.0, mid - 0.06)
                hi = min(1.0, mid + 0.06)
            ax.set_ylim(lo, hi)
        else:
            pad = max(0.05, 0.16 * max(y_max - y_min, 1e-6))
            lo = max(0.0, y_min - pad)
            hi = y_max + pad
            if hi - lo < 0.12:
                hi = lo + 0.12
            ax.set_ylim(lo, hi)

    labels = key_order["label"].astype(str).tolist()
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT["tick"] + 4)

    return fig, ax, dd


def select_subject_by_model(
    cfg: EDAConfig,
    mode: str = "median",
    metric: str = "pearson_r",
    model: str = "plam",
) -> str:
    dlam_df = compute_subject_metrics_from_cache(cfg, model=str(model))
    if metric not in {"pearson_r", "rmse"}:
        raise ValueError("metric must be one of: pearson_r, rmse")
    d = dlam_df.dropna(subset=[metric]).copy()
    if len(d) == 0:
        raise RuntimeError(f"No valid {model} subject metrics found")
    if mode == "best":
        row = d.sort_values([metric, "subject"], ascending=[False if metric == "pearson_r" else True, True]).iloc[0]
    elif mode == "worst":
        row = d.sort_values([metric, "subject"], ascending=[True if metric == "pearson_r" else False, True]).iloc[0]
    else:
        # closest to median
        med = float(d[metric].median())
        d["delta_med"] = np.abs(d[metric] - med)
        row = d.sort_values(["delta_med", "subject"], ascending=[True, True]).iloc[0]
    return str(row["subject"])


def _subject_scatter_payload(
    cfg: EDAConfig,
    model: str,
    subject: str,
    eval_gene_mode: str = "all",  # all | hvg | custom
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
) -> Dict[str, np.ndarray]:
    p = (_model_cache_root(cfg, str(model).lower()) / f"{subject}.npz").resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as z:
        pred, truth = _cache_pred_truth_arrays(z)
        mask = z["loro_eval_mask"].astype(bool)
        gene_names = tuple(str(g) for g in z["gene_names"].tolist())

    _, _, eval_gene_set = _resolve_eval_gene_set(
        cfg,
        eval_gene_mode=eval_gene_mode,
        custom_gene_list=custom_gene_list,
        eval_gene_path=eval_gene_path,
    )
    gi = _gene_indices_from_names(
        gene_names,
        eval_gene_set,
        context=f"subject={subject} model={model}",
    )

    g = int(gi.size)
    parcel_ids = np.where(mask)[0]
    x = truth[mask, :][:, gi].ravel()
    y = pred[mask, :][:, gi].ravel()
    parcel_per_point = np.repeat(parcel_ids, g)
    gene_per_point = np.tile(np.arange(g, dtype=np.int32), len(parcel_ids))
    finite = np.isfinite(x) & np.isfinite(y)
    return {
        "x": x[finite],
        "y": y[finite],
        "parcel_ids": parcel_per_point[finite],
        "gene_ids": gene_per_point[finite],
        "n_parcels": int(mask.sum()),
        "n_genes": int(g),
        "n_points": int(finite.sum()),
    }


def _top_ids(ids: np.ndarray, n_top: int) -> List[int]:
    s = pd.Series(ids)
    return s.value_counts().head(int(n_top)).index.astype(int).tolist()


def plot_single_subject_scatter_triplet(
    cfg: EDAConfig,
    subject_mode: str = "median",
    subject_id: str | None = None,
    rank_model: str = "plam",
    color_by: str = "parcel",  # parcel | gene | none | density
    top_n: int = 10,
    eval_gene_mode: str = "all",  # all | hvg | custom
    custom_gene_list: Sequence[str] | None = None,
    eval_gene_path: str | None = None,
    density_gridsize: int = 70,
    density_cmap: str = "magma",
    density_mincnt: int = 1,
    figsize: Tuple[float, float] = (15.0, 4.8),
) -> Tuple[plt.Figure, np.ndarray, str]:
    subject = str(subject_id) if subject_id else select_subject_by_model(
        cfg,
        mode=str(subject_mode),
        metric="pearson_r",
        model=str(rank_model),
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    mode_eval = str(eval_gene_mode).lower()
    mode_color = str(color_by).lower()
    if mode_eval == "all":
        s_bg = 2.0
        s_fg = 3.0
        a_bg = 0.12
        a_fg = 0.28
    else:
        s_bg = 5.0
        s_fg = 7.0
        a_bg = 0.15
        a_fg = 0.45

    for ax, model in zip(axes, MODEL_ORDER):
        p = _subject_scatter_payload(
            cfg,
            model=model,
            subject=subject,
            eval_gene_mode=eval_gene_mode,
            custom_gene_list=custom_gene_list,
            eval_gene_path=eval_gene_path,
        )
        x = p["x"]
        y = p["y"]
        if mode_color == "density":
            ax.hexbin(
                x,
                y,
                gridsize=int(density_gridsize),
                mincnt=int(density_mincnt),
                cmap=str(density_cmap),
                linewidths=0.0,
                bins="log",
            )
        elif mode_color == "none":
            ax.scatter(x, y, s=s_fg, alpha=a_fg, color="#4f4f4f", linewidths=0)
        else:
            ids = p["parcel_ids"] if mode_color == "parcel" else p["gene_ids"]
            tops = _top_ids(ids, top_n)
            base = np.isin(ids, tops, invert=True)
            ax.scatter(x[base], y[base], s=s_bg, alpha=a_bg, color="#9a9a9a", linewidths=0)
            palette = sns.color_palette("tab10", n_colors=max(1, len(tops)))
            for c, tid in zip(palette, tops):
                m = ids == tid
                ax.scatter(x[m], y[m], s=s_fg, alpha=a_fg, color=c, linewidths=0)
        lim_lo = float(np.nanpercentile(np.r_[x, y], 0.5))
        lim_hi = float(np.nanpercentile(np.r_[x, y], 99.5))
        if not np.isfinite(lim_lo) or not np.isfinite(lim_hi) or lim_hi <= lim_lo:
            lim_lo = float(np.nanmin(np.r_[x, y]))
            lim_hi = float(np.nanmax(np.r_[x, y]))
        pad = 0.05 * max(lim_hi - lim_lo, 1e-3)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1)
        ax.set_xlim(lim_lo - pad, lim_hi + pad)
        ax.set_ylim(lim_lo - pad, lim_hi + pad)
        pear = _pearson_safe(x, y)
        rmse = _rmse_safe(x, y)
        if mode_color == "density":
            color_desc = f"density (hexbin, gridsize={int(density_gridsize)})"
        elif mode_color == "none":
            color_desc = "none"
        else:
            color_desc = f"{mode_color} (top {int(top_n)})"
        ax.text(
            0.985,
            0.03,
            f"r={pear:.3f}\nrmse={rmse:.3f}\nn={p['n_points']:,}\ncolored by: {color_desc}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT["small"],
            bbox={"facecolor": "white", "edgecolor": "#808080", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )
        ax.set_title(f"{MODEL_LABELS[model]} ({subject})", fontsize=FONT["title"])
        ax.set_xlabel("True (held-out harmonized GTEx)", fontsize=FONT["label"])
        ax.set_ylabel("Predicted", fontsize=FONT["label"])
        ax.grid(True, alpha=0.15)
    return fig, axes, subject


def select_representative_subject_by_gtex_median(
    prepost: Dict[str, object],
    gene_mode: str = "hvg",  # hvg | allgenes | custom
    gene_list: Sequence[str] | None = None,
    space: str = "harmonized",  # harmonized | raw
    min_coverage: int = 5,
    require_coverage: int | None = None,
) -> Tuple[str, pd.DataFrame]:
    mode = str(space).lower()
    if mode not in {"harmonized", "raw"}:
        raise ValueError("space must be one of: harmonized, raw")

    genes_req = _resolve_gene_panel(prepost, gene_mode=gene_mode, gene_list=gene_list)
    genes = _restrict_genes_to_available(genes_req, prepost["genes"])
    all_genes = [str(g) for g in prepost["genes"]]
    gi = [all_genes.index(g) for g in genes]

    cube = np.asarray(prepost["harm_cube" if mode == "harmonized" else "raw_cube"], dtype=np.float64)[:, :, gi]
    obs_mask = np.asarray(prepost["obs_mask"], dtype=bool)
    subjects = [str(s) for s in prepost["subjects"]]
    coverage = obs_mask.sum(axis=1).astype(int)

    cohort_med = np.nanmedian(cube, axis=0)  # parcel x gene
    rows: List[Dict[str, object]] = []
    for i, sid in enumerate(subjects):
        cov = int(coverage[i])
        if cov < int(min_coverage):
            continue
        if require_coverage is not None and cov != int(require_coverage):
            continue
        m = np.isfinite(cube[i]) & np.isfinite(cohort_med) & obs_mask[i][:, None]
        n = int(m.sum())
        if n < 2:
            continue
        x = cube[i][m]
        y = cohort_med[m]
        rows.append(
            {
                "subject": sid,
                "coverage": cov,
                "n_points": n,
                "pearson_r_to_cohort_median": _pearson_safe(x, y),
                "rmse_to_cohort_median": _rmse_safe(x, y),
            }
        )

    d = pd.DataFrame(rows)
    if len(d) == 0:
        raise RuntimeError("No candidate subjects after representative-subject filtering.")
    d = d.sort_values(["rmse_to_cohort_median", "subject"], ascending=[True, True]).reset_index(drop=True)
    return str(d.iloc[0]["subject"]), d


def _subject_gene_indices(prepost: Dict[str, object], gene_mode: str, gene_list: Sequence[str] | None) -> Tuple[List[str], List[int]]:
    genes_req = _resolve_gene_panel(prepost, gene_mode=gene_mode, gene_list=gene_list)
    genes = _restrict_genes_to_available(genes_req, prepost["genes"])
    all_genes = [str(g) for g in prepost["genes"]]
    gi = [all_genes.index(g) for g in genes]
    return genes, gi


def _ahba_harmonized_median_matrix(prepost: Dict[str, object], genes: List[str]) -> np.ndarray:
    return _parcel_median_matrix(prepost["ahba_h"], genes, prepost["target_meta"])


def _load_subject_prediction_matrix(cfg: EDAConfig, model: str, subject_id: str, gi: List[int]) -> np.ndarray:
    model_l = str(model).lower()
    npz_path = (_model_cache_root(cfg, model_l) / f"{subject_id}.npz").resolve()
    if not npz_path.exists() and model_l == "plam":
        root = (_resolve_repo_path(cfg.cache_root) / str(cfg.gene_scope).lower()).resolve()
        # Fallback search for plam variants (e.g., plam_rank3/plam_rank4/plam_dynamicrank).
        candidates = []
        for d in sorted(root.glob("plam*")):
            if d.is_dir():
                p = (d / f"{subject_id}.npz").resolve()
                if p.exists():
                    candidates.append(p)
        if len(candidates) > 0:
            npz_path = candidates[0]
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing cache for model={model_l}, subject={subject_id}. Tried: {npz_path}. "
            f"If using PLAM rank variants, set cfg.plam_cache_dirname (e.g., plam_rank4/plam_dynamicrank)."
        )
    z = np.load(npz_path, allow_pickle=True)
    return np.asarray(z["loro_fused_subject_h"], dtype=np.float64)[:, gi]


def _reduce_gene_axis_for_render(
    mats: List[np.ndarray],
    labels: List[str],
    max_genes: int | None,
    reduce_mode: str = "mean",  # mean | median
) -> Tuple[List[np.ndarray], List[str]]:
    if max_genes is None:
        return mats, labels
    mg = int(max_genes)
    if mg <= 0:
        return mats, labels
    g = int(mats[0].shape[1]) if len(mats) else 0
    if g <= mg:
        return mats, labels
    # Even-width binning along gene axis for lightweight rendering.
    bins = np.linspace(0, g, num=mg + 1, dtype=np.int32)
    out: List[np.ndarray] = []
    mode = str(reduce_mode).lower()
    for m in mats:
        red = np.full((m.shape[0], mg), np.nan, dtype=np.float64)
        for bi in range(mg):
            lo, hi = int(bins[bi]), int(bins[bi + 1])
            if hi <= lo:
                hi = min(lo + 1, g)
            sl = m[:, lo:hi]
            if mode == "median":
                red[:, bi] = np.nanmedian(sl, axis=1)
            else:
                red[:, bi] = np.nanmean(sl, axis=1)
        out.append(red)
    new_labels = [f"bin{i+1}" for i in range(mg)]
    return out, new_labels


def _subject_label_map(prepost: Dict[str, object], subject_id: str, label_mode: str) -> List[str]:
    target_meta = prepost["target_meta"]
    ahba_labels = target_meta["tissue_or_parcel"].astype(str).tolist()
    sub_rows = prepost["gtex_eligible_raw"][prepost["gtex_eligible_raw"]["subject"].astype(str) == str(subject_id)].copy()
    gtex_by_parcel = (
        sub_rows.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )
    mode = str(label_mode).lower()
    if mode not in {"gtex", "ahba", "both"}:
        raise ValueError("label_mode must be one of: gtex, ahba, both")
    labels: List[str] = []
    for pidx, ahba_name in enumerate(ahba_labels):
        gname = gtex_by_parcel.get(int(pidx), "")
        if mode == "ahba":
            labels.append(str(ahba_name))
        elif mode == "both":
            right = str(gname) if str(gname) else str(ahba_name)
            labels.append(f"{ahba_name}: {right}")
        else:
            labels.append(str(gname) if str(gname) else str(ahba_name))
    return labels


def plot_subject_model_matrix_panel(
    cfg: EDAConfig,
    prepost: Dict[str, object],
    subject_id: str,
    gene_mode: str = "hvg",  # hvg | allgenes | custom
    gene_list: Sequence[str] | None = None,
    label_mode: str = "gtex",  # gtex | ahba | both
    observed_only: bool = False,
    label_stride: int | None = None,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (16.0, 11.0),
    render_max_genes: int | None = 600,
    render_reduce_mode: str = "mean",  # mean | median
) -> Tuple[plt.Figure, np.ndarray]:
    subjects = [str(s) for s in prepost["subjects"]]
    sid = str(subject_id)
    if sid not in set(subjects):
        raise ValueError(f"Subject not found in eligible PREPOST set: {sid}")
    si = subjects.index(sid)

    genes, gi = _subject_gene_indices(prepost, gene_mode=gene_mode, gene_list=gene_list)
    ahba_h = _ahba_harmonized_median_matrix(prepost, genes)
    sparse = np.asarray(prepost["harm_cube"][si, :, :], dtype=np.float64)[:, gi]
    obs = np.asarray(prepost["obs_mask"][si, :], dtype=bool)
    labels = _subject_label_map(prepost, sid, label_mode=label_mode)

    model_preds = {m: _load_subject_prediction_matrix(cfg, m, sid, gi) for m in MODEL_ORDER}
    model_resid = {m: model_preds[m] - ahba_h for m in MODEL_ORDER}

    if bool(observed_only):
        keep = obs
    else:
        keep = np.isfinite(sparse).any(axis=1) | np.isfinite(ahba_h).any(axis=1)
    keep_idx = np.where(keep)[0]
    labels = [labels[i] for i in keep_idx.tolist()]

    ahba_h = ahba_h[keep, :]
    sparse = sparse[keep, :]
    for m in MODEL_ORDER:
        model_preds[m] = model_preds[m][keep, :]
        model_resid[m] = model_resid[m][keep, :]

    render_mats_all = [ahba_h, sparse, model_preds["naive"], model_preds["dlam"], model_preds["plam"], model_resid["naive"], model_resid["dlam"], model_resid["plam"]]
    render_mats_all, _ = _reduce_gene_axis_for_render(render_mats_all, genes, render_max_genes, render_reduce_mode)
    ahba_h, sparse, model_preds["naive"], model_preds["dlam"], model_preds["plam"], model_resid["naive"], model_resid["dlam"], model_resid["plam"] = render_mats_all

    vals = np.r_[
        ahba_h.ravel(),
        sparse.ravel(),
        model_preds["naive"].ravel(),
        model_preds["dlam"].ravel(),
        model_preds["plam"].ravel(),
    ]
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if int(vals.size):
        vmin = float(np.nanpercentile(vals, 1.0))
        vmax = float(np.nanpercentile(vals, 99.0))
    else:
        vmin, vmax = -1.0, 1.0

    resid_vals = np.r_[model_resid["naive"].ravel(), model_resid["dlam"].ravel(), model_resid["plam"].ravel()]
    resid_vals = resid_vals[np.isfinite(resid_vals)]
    if int(resid_vals.size):
        lim_r = float(np.nanpercentile(np.abs(resid_vals), 99.0))
        lim_r = lim_r if np.isfinite(lim_r) and lim_r > 0 else 1.0
    else:
        lim_r = 1.0

    fig, axes = plt.subplots(3, 4, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    col_titles = ["AHBA median (harmonized)", f"Sparse GTEx ({sid})", f"Completed GTEx ({sid})", "Residual (GTEx - AHBA)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=FONT["title"] + 2)

    for r, m in enumerate(MODEL_ORDER):
        mats = [ahba_h, sparse, model_preds[m], model_resid[m]]
        for c, mat in enumerate(mats):
            if c == 3:
                im = axes[r, c].imshow(mat, aspect="auto", interpolation="none", cmap="coolwarm", vmin=-lim_r, vmax=lim_r)
            else:
                im = axes[r, c].imshow(mat, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)
            axes[r, c].set_xlabel("Genes", fontsize=FONT["label"])
            axes[r, c].set_ylabel("")
            axes[r, c].grid(False, which="both")
            axes[r, c].xaxis.grid(False, which="both")
            axes[r, c].yaxis.grid(False, which="both")
            axes[r, c].minorticks_off()
        axes[r, 0].text(
            -0.14,
            0.5,
            MODEL_LABELS[m],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=FONT["title"] + 4,
            fontweight="bold",
        )

    if label_stride is None or int(label_stride) <= 0:
        step = max(1, int(np.ceil(len(labels) / 18)))
    else:
        step = max(1, int(label_stride))
    for ax in axes.ravel().tolist():
        ax.set_yticks([])
        ax.set_yticklabels([])

    fig.colorbar(axes[0, 0].images[0], ax=axes[:, :3].ravel().tolist(), shrink=0.58, label="Expression value")
    fig.colorbar(axes[0, 3].images[0], ax=axes[:, 3].ravel().tolist(), shrink=0.58, label="Residual")
    return fig, axes


def _scatter_with_optional_coloring(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    color_by: str = "parcel",
    top_n: int = 10,
    point_size: float = 7.0,
    alpha: float = 0.45,
    marker: str = "o",
) -> List[Tuple[int, tuple]]:
    mode = str(color_by).lower()
    legend_items: List[Tuple[int, tuple]] = []
    if mode == "none":
        ax.scatter(x, y, s=point_size, alpha=alpha, color="#4f4f4f", linewidths=0, marker=marker)
        return legend_items
    tops = _top_ids(ids, top_n)
    base = np.isin(ids, tops, invert=True)
    ax.scatter(
        x[base],
        y[base],
        s=max(3.0, point_size - 2.0),
        alpha=max(0.12, alpha * 0.35),
        color="#a4a4a4",
        linewidths=0,
        marker=marker,
    )
    palette = sns.color_palette("tab10", n_colors=max(1, len(tops)))
    for c, tid in zip(palette, tops):
        m = ids == tid
        ax.scatter(x[m], y[m], s=point_size, alpha=alpha, color=c, linewidths=0, marker=marker)
        legend_items.append((int(tid), c))
    return legend_items


def plot_subject_alignment_scatter_panel(
    cfg: EDAConfig,
    prepost: Dict[str, object],
    subject_id: str,
    gene_mode: str = "hvg",  # hvg | allgenes | custom
    gene_list: Sequence[str] | None = None,
    parcels: str = "observed",  # observed | all
    color_by: str = "parcel",  # parcel | gene | none
    top_n: int = 10,
    overlay_observed_in_predictions: bool = False,
    figsize: Tuple[float, float] = (18.0, 4.8),
    show_color_legend: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    subjects = [str(s) for s in prepost["subjects"]]
    sid = str(subject_id)
    if sid not in set(subjects):
        raise ValueError(f"Subject not found in eligible PREPOST set: {sid}")
    si = subjects.index(sid)

    genes, gi = _subject_gene_indices(prepost, gene_mode=gene_mode, gene_list=gene_list)
    ahba_h = _ahba_harmonized_median_matrix(prepost, genes)
    obs_h = np.asarray(prepost["harm_cube"][si, :, :], dtype=np.float64)[:, gi]
    obs_mask = np.asarray(prepost["obs_mask"][si, :], dtype=bool)

    eval_mode = str(parcels).lower()
    if eval_mode not in {"observed", "all"}:
        raise ValueError("parcels must be one of: observed, all")

    if eval_mode == "observed":
        row_mask = obs_mask.copy()
    else:
        row_mask = np.isfinite(ahba_h).any(axis=1)

    def _flatten_pair(y_mat: np.ndarray, mask_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xx = ahba_h[mask_rows, :].ravel()
        yy = y_mat[mask_rows, :].ravel()
        parcel_ids = np.repeat(np.where(mask_rows)[0], len(genes))
        gene_ids = np.tile(np.arange(len(genes), dtype=np.int32), int(mask_rows.sum()))
        m = np.isfinite(xx) & np.isfinite(yy)
        return xx[m], yy[m], parcel_ids[m], gene_ids[m]

    x_obs, y_obs, pid_obs, gid_obs = _flatten_pair(obs_h, obs_mask)
    ids_obs = pid_obs if str(color_by).lower() == "parcel" else gid_obs

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    _scatter_with_optional_coloring(
        axes[0], x_obs, y_obs, ids_obs, color_by=color_by, top_n=top_n, point_size=8.0, alpha=0.48, marker="o"
    )
    pear = _pearson_safe(x_obs, y_obs)
    rmse = _rmse_safe(x_obs, y_obs)
    axes[0].set_title(f"{sid} Observed GTEx vs AHBA", fontsize=FONT["title"] + 2)
    m_leg = axes[0].legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#606060", markersize=5, label="Observed")],
        loc="upper left",
        frameon=True,
        fontsize=FONT["small"],
    )
    axes[0].add_artist(m_leg)
    color_desc = "none" if str(color_by).lower() == "none" else f"{str(color_by).lower()} (top {int(top_n)})"
    axes[0].text(
        0.985,
        0.03,
        f"r={pear:.3f}\nrmse={rmse:.3f}\nn={len(x_obs):,}\ncolored by: {color_desc}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT["small"],
        bbox={"facecolor": "white", "edgecolor": "#808080", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )

    for ax, model in zip(axes[1:], MODEL_ORDER):
        pred = _load_subject_prediction_matrix(cfg, model=model, subject_id=sid, gi=gi)
        x, y, pid, gid = _flatten_pair(pred, row_mask)
        ids = pid if str(color_by).lower() == "parcel" else gid

        if bool(overlay_observed_in_predictions):
            axes_obs_x, axes_obs_y, _, _ = _flatten_pair(obs_h, row_mask)
            ax.scatter(axes_obs_x, axes_obs_y, s=8.0, alpha=0.20, color="#8f8f8f", linewidths=0, zorder=1, marker="o")

        _scatter_with_optional_coloring(
            ax, x, y, ids, color_by=color_by, top_n=top_n, point_size=9.5, alpha=0.50, marker="^"
        )
        pr = _pearson_safe(x, y)
        rr = _rmse_safe(x, y)
        ax.set_title(f"{sid} {MODEL_LABELS[model]} vs AHBA", fontsize=FONT["title"] + 2)
        if bool(overlay_observed_in_predictions):
            m_handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#8f8f8f", markersize=5, label="Observed"),
                Line2D([0], [0], marker="^", color="none", markerfacecolor="#606060", markersize=6, label="Predicted"),
            ]
        else:
            m_handles = [Line2D([0], [0], marker="^", color="none", markerfacecolor="#606060", markersize=6, label="Predicted")]
        m_leg = ax.legend(handles=m_handles, loc="upper left", frameon=True, fontsize=FONT["small"])
        ax.add_artist(m_leg)
        color_desc = "none" if str(color_by).lower() == "none" else f"{str(color_by).lower()} (top {int(top_n)})"
        ax.text(
            0.985,
            0.03,
            f"r={pr:.3f}\nrmse={rr:.3f}\nn={len(x):,}\ncolored by: {color_desc}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT["small"],
            bbox={"facecolor": "white", "edgecolor": "#808080", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )

    all_x = []
    all_y = []
    for ax in axes:
        for c in ax.collections:
            offs = c.get_offsets()
            if len(offs):
                all_x.append(np.asarray(offs[:, 0], dtype=np.float64))
                all_y.append(np.asarray(offs[:, 1], dtype=np.float64))
    if all_x:
        xv = np.concatenate(all_x)
        yv = np.concatenate(all_y)
        lim_lo = float(np.nanpercentile(np.r_[xv, yv], 0.5))
        lim_hi = float(np.nanpercentile(np.r_[xv, yv], 99.5))
        if not np.isfinite(lim_lo) or not np.isfinite(lim_hi) or lim_hi <= lim_lo:
            lim_lo = float(np.nanmin(np.r_[xv, yv]))
            lim_hi = float(np.nanmax(np.r_[xv, yv]))
        pad = 0.05 * max(lim_hi - lim_lo, 1e-3)
        for ax in axes:
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.9, alpha=0.75)
            ax.set_xlim(lim_lo - pad, lim_hi + pad)
            ax.set_ylim(lim_lo - pad, lim_hi + pad)

    for ax in axes:
        ax.set_xlabel("Observed AHBA median", fontsize=FONT["label"] + 1)
        ax.tick_params(axis="both", which="major", labelsize=FONT["tick"], length=3.6, width=0.8, direction="out")
        ax.tick_params(axis="both", which="minor", length=2.0, width=0.6, direction="out")
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
        ax.minorticks_on()
    axes[0].set_ylabel("Observed GTEx value", fontsize=FONT["label"] + 1)
    for ax in axes[1:]:
        ax.set_ylabel("Predicted GTEx value", fontsize=FONT["label"] + 1)
    return fig, axes


def resolve_publication_subject(
    cfg: EDAConfig,
    prepost: Dict[str, object] | None = None,
    subject_id: str | None = None,
    mode: str = "representative",  # representative | model_median | model_best | model_worst
    model: str = "plam",
    metric: str = "pearson_r",
    gene_mode: str = "hvg",
    gene_list: Sequence[str] | None = None,
    space: str = "harmonized",
    min_coverage: int = 5,
    require_coverage: int | None = None,
) -> Tuple[str, pd.DataFrame]:
    if subject_id is not None and str(subject_id).strip():
        sid = str(subject_id).strip()
        if prepost is not None:
            if sid not in set(str(s) for s in prepost["subjects"]):
                raise ValueError(f"subject_id={sid} not in eligible PREPOST subjects")
        return sid, pd.DataFrame(
            [
                {
                    "subject": sid,
                    "selection_mode": "explicit_subject_id",
                }
            ]
        )

    sel_mode = str(mode).lower()
    if sel_mode == "representative":
        if prepost is None:
            raise ValueError("prepost is required when mode='representative'")
        sid, tbl = select_representative_subject_by_gtex_median(
            prepost=prepost,
            gene_mode=gene_mode,
            gene_list=gene_list,
            space=space,
            min_coverage=min_coverage,
            require_coverage=require_coverage,
        )
        tbl = tbl.copy()
        tbl["selection_mode"] = "representative"
        return sid, tbl

    model_mode_map = {
        "model_median": "median",
        "model_best": "best",
        "model_worst": "worst",
    }
    if sel_mode not in model_mode_map:
        raise ValueError("mode must be one of: representative, model_median, model_best, model_worst")

    sid = select_subject_by_model(cfg, mode=model_mode_map[sel_mode], metric=str(metric), model=str(model))
    metrics_df = compute_subject_metrics_from_cache(cfg, model=str(model))
    metrics_df["selection_mode"] = sel_mode
    return str(sid), metrics_df
