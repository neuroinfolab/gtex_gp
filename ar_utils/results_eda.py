#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    hvg_path: str = "data/raw/ahba_100hvg.txt"
    cache_root: str = "out/loro_subject_cache"
    naive_cache_dirname: str = "naive"
    dlam_cache_dirname: str = "dlam"
    plam_cache_dirname: str = "plam"
    gene_scope: str = "hvg"  # hvg | allgenes
    min_observed_parcels: int = 5
    combat_use_covariates: bool = True
    gtex_rep_mode: str = "medoid"
    gtex_hemi_mode: str = "native"


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


def _rmse_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return np.nan
    d = x[m] - y[m]
    return float(np.sqrt(np.mean(d**2)))


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
    hvg_path = _resolve_repo_path(cfg.hvg_path)
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
    hcfg = SimpleNamespace(combat_use_covariates=bool(cfg.combat_use_covariates))
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


def compute_subject_metrics_from_cache(cfg: EDAConfig, model: str) -> pd.DataFrame:
    model = str(model).lower()
    root = _model_cache_root(cfg, model)
    if not root.exists():
        raise FileNotFoundError(f"Cache dir not found: {root}")
    rows: List[Dict[str, object]] = []
    for npz_path in sorted(root.glob("*.npz")):
        sid = npz_path.stem
        json_path = npz_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            try:
                import json

                meta = json.loads(json_path.read_text())
            except Exception:
                meta = {}
        z = np.load(npz_path, allow_pickle=True)
        pred = z["loro_fused_subject_h"].astype(np.float64)
        truth = z["loro_truth_subject_h"].astype(np.float64)
        mask = z["loro_eval_mask"].astype(bool)
        x = truth[mask, :].ravel()
        y = pred[mask, :].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "subject": sid,
                "model": model,
                "coverage": int(meta.get("n_gtex_observed_subject", int(mask.sum()))),
                "n_loro_parcels": int(mask.sum()),
                "n_points": int(m.sum()),
                "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            }
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
) -> pd.DataFrame:
    model = str(model).lower()
    root = _model_cache_root(cfg, model)
    if not root.exists():
        raise FileNotFoundError(f"Cache dir not found: {root}")

    mode = str(eval_gene_mode).lower()
    if mode not in {"all", "hvg", "custom"}:
        raise ValueError("eval_gene_mode must be one of: all, hvg, custom")

    eval_gene_set: set[str] | None = None
    if mode == "hvg":
        csv_path = _resolve_repo_path(cfg.csv_path)
        hvg_path = _resolve_repo_path(cfg.hvg_path)
        hdr = io_utils.load_gene_header_and_hvg(csv_path, hvg_path)
        eval_gene_set = set(str(g) for g in hdr["genes_hvg"])
    elif mode == "custom":
        if custom_gene_list is None:
            raise ValueError("custom_gene_list is required when eval_gene_mode='custom'")
        eval_gene_set = set(str(g) for g in custom_gene_list)
        if not eval_gene_set:
            raise ValueError("custom_gene_list is empty")

    rows: List[Dict[str, object]] = []
    for npz_path in sorted(root.glob("*.npz")):
        sid = npz_path.stem
        json_path = npz_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            try:
                import json

                meta = json.loads(json_path.read_text())
            except Exception:
                meta = {}

        z = np.load(npz_path, allow_pickle=True)
        pred = z["loro_fused_subject_h"].astype(np.float64)
        truth = z["loro_truth_subject_h"].astype(np.float64)
        mask = z["loro_eval_mask"].astype(bool)
        gene_names = [str(g) for g in z["gene_names"].tolist()]
        n_total_genes = int(len(gene_names))

        if eval_gene_set is None:
            gi = np.arange(n_total_genes, dtype=np.int32)
        else:
            gi = np.asarray([i for i, g in enumerate(gene_names) if g in eval_gene_set], dtype=np.int32)
            if int(gi.size) == 0:
                raise RuntimeError(
                    f"No overlap between requested eval genes ({mode}) and cache genes for subject {sid} model {model}"
                )

        x = truth[mask, :][:, gi].ravel()
        y = pred[mask, :][:, gi].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "subject": sid,
                "model": model,
                "coverage": int(meta.get("n_gtex_observed_subject", int(mask.sum()))),
                "n_loro_parcels": int(mask.sum()),
                "n_eval_genes": int(gi.size),
                "n_total_genes": n_total_genes,
                "eval_gene_mode": mode,
                "n_points": int(m.sum()),
                "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["coverage", "subject"]).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No npz files found under {root}")
    return out


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
    figsize: Tuple[float, float] = (10.2, 4.2),
    use_sem: bool = False,
    panel_label: str | None = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    need = {"model", "pearson_r", "rmse"}
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
        summ["rmse_err"] = summ["rmse_std"] / np.sqrt(n)
        err_lbl = "SEM"
    else:
        summ["pearson_err"] = summ["pearson_std"]
        summ["rmse_err"] = summ["rmse_std"]
        err_lbl = "SD"

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    x = np.arange(len(summ), dtype=np.int32)
    labels = [MODEL_LABELS[m] for m in summ["model"].tolist()]
    colors = [MODEL_COLORS[m] for m in summ["model"].tolist()]

    p_lbl = _metrics_panel_label(metrics_df, panel_label=panel_label)
    sfx = "" if p_lbl == "" else f" ({p_lbl})"

    bar0 = axes[0].bar(
        x,
        summ["pearson_mean"].to_numpy(dtype=np.float64),
        yerr=summ["pearson_err"].to_numpy(dtype=np.float64),
        color=colors,
        alpha=0.92,
        capsize=4,
        ecolor="#3a3a3a",
    )
    axes[0].set_title(f"Mean Subject LORO Pearson r ({err_lbl} bars){sfx}", fontsize=FONT["title"] + 3)
    axes[0].set_ylabel("Pearson r", fontsize=FONT["label"] + 2)
    axes[0].set_xticks(x.tolist())
    axes[0].set_xticklabels(labels, rotation=0, fontsize=FONT["tick"] + 2)
    axes[0].tick_params(axis="y", labelsize=FONT["tick"] + 2)
    axes[0].grid(True, axis="y", alpha=0.2)

    bar1 = axes[1].bar(
        x,
        summ["rmse_mean"].to_numpy(dtype=np.float64),
        yerr=summ["rmse_err"].to_numpy(dtype=np.float64),
        color=colors,
        alpha=0.92,
        capsize=4,
        ecolor="#3a3a3a",
    )
    axes[1].set_title(f"Mean Subject LORO RMSE ({err_lbl} bars){sfx}", fontsize=FONT["title"] + 3)
    axes[1].set_ylabel("RMSE", fontsize=FONT["label"] + 2)
    axes[1].set_xticks(x.tolist())
    axes[1].set_xticklabels(labels, rotation=0, fontsize=FONT["tick"] + 2)
    axes[1].tick_params(axis="y", labelsize=FONT["tick"] + 2)
    axes[1].grid(True, axis="y", alpha=0.2)

    # Place value labels above each bar in form "(metric; n=...)".
    p_mean = summ["pearson_mean"].to_numpy(dtype=np.float64)
    p_err = np.nan_to_num(summ["pearson_err"].to_numpy(dtype=np.float64), nan=0.0)
    r_mean = summ["rmse_mean"].to_numpy(dtype=np.float64)
    r_err = np.nan_to_num(summ["rmse_err"].to_numpy(dtype=np.float64), nan=0.0)
    n_subs = summ["n_subjects"].to_numpy(dtype=np.int32)

    p_top = p_mean + p_err
    r_top = r_mean + r_err
    p_pad = max(float(np.nanmax(np.abs(p_top))) * 0.08, 0.015)
    r_pad = max(float(np.nanmax(np.abs(r_top))) * 0.08, 0.015)
    # Pearson is bounded in [-1, 1]; fix top at 1.0 for consistent interpretability.
    y0_lo, _ = axes[0].get_ylim()
    axes[0].set_ylim(bottom=y0_lo, top=1.0)
    axes[1].set_ylim(top=float(np.nanmax(r_top) + 3.2 * r_pad))

    y0_lo, y0_hi = axes[0].get_ylim()
    p_bot = p_mean - p_err
    p_label_pad = max(0.012, 0.02 * (y0_hi - y0_lo))
    for xi, val, bot, n_sub in zip(x.tolist(), p_mean.tolist(), p_bot.tolist(), n_subs.tolist()):
        # Place label just below the lower error-bar cap ("bottom T"), clamped to axis range.
        y_txt = max(float(bot - p_label_pad), float(y0_lo + 0.01 * (y0_hi - y0_lo)))
        axes[0].text(
            xi,
            y_txt,
            f"({val:.3f}; n={int(n_sub)})",
            ha="center",
            va="top",
            fontsize=FONT["small"] + 2,
        )
    for xi, val, top, n_sub in zip(x.tolist(), r_mean.tolist(), r_top.tolist(), n_subs.tolist()):
        axes[1].text(
            xi,
            float(top + r_pad),
            f"({val:.3f}; n={int(n_sub)})",
            ha="center",
            va="bottom",
            fontsize=FONT["small"] + 2,
        )

    return fig, axes, summ


def _parse_fold_key(fold_key: str) -> Tuple[int, List[int]]:
    s = str(fold_key)
    parts = s.split("|")
    hold = int(parts[0].split("=")[1].strip())
    train_str = parts[1].split("=")[1].strip() if len(parts) > 1 else ""
    train = [int(x) for x in train_str.split(",") if str(x).strip() != ""]
    return hold, train


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
    coverage_max: int = 11,
    models: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_list = [str(m).lower() for m in (models if models is not None else MODEL_ORDER)]
    mode = str(eval_gene_mode).lower()
    if mode not in {"all", "hvg", "custom"}:
        raise ValueError("eval_gene_mode must be one of: all, hvg, custom")

    eval_gene_set: set[str] | None = None
    if mode == "hvg":
        hdr = io_utils.load_gene_header_and_hvg(_resolve_repo_path(cfg.csv_path), _resolve_repo_path(cfg.hvg_path))
        eval_gene_set = set(str(g) for g in hdr["genes_hvg"])
    elif mode == "custom":
        if custom_gene_list is None:
            raise ValueError("custom_gene_list is required when eval_gene_mode='custom'")
        eval_gene_set = set(str(g) for g in custom_gene_list)
        if not eval_gene_set:
            raise ValueError("custom_gene_list is empty")

    g = prepost["gtex_eligible_raw"].copy()
    subj_obs = (
        g.groupby("subject")["parcel_idx"]
        .apply(lambda s: sorted(set(int(v) for v in s.tolist())))
        .to_dict()
    )

    rows: List[Dict[str, object]] = []
    for model in model_list:
        model_root = _model_cache_root(cfg, model)
        if not model_root.exists():
            continue
        for sid, obs in subj_obs.items():
            n_obs = int(len(obs))
            if n_obs < int(coverage_min) or n_obs > int(coverage_max):
                continue

            npz_path = model_root / f"{sid}.npz"
            if not npz_path.exists():
                continue

            z = np.load(npz_path, allow_pickle=True)
            pred = z["loro_fused_subject_h"].astype(np.float64)
            truth = z["loro_truth_subject_h"].astype(np.float64)
            mask = z["loro_eval_mask"].astype(bool)
            gene_names = [str(x) for x in z["gene_names"].tolist()]

            if eval_gene_set is None:
                gi = np.arange(len(gene_names), dtype=np.int32)
            else:
                gi = np.asarray([i for i, gname in enumerate(gene_names) if gname in eval_gene_set], dtype=np.int32)
                if int(gi.size) == 0:
                    continue

            obs_set = set(int(x) for x in obs)
            for hold in np.where(mask)[0].astype(int).tolist():
                train = sorted(obs_set - {int(hold)})
                train_key = ",".join(map(str, train))
                fold_key = f"hold={int(hold)}|train={train_key}"
                x = truth[int(hold), gi]
                y = pred[int(hold), gi]
                m = np.isfinite(x) & np.isfinite(y)
                rows.append(
                    {
                        "subject": str(sid),
                        "model": str(model),
                        "coverage": n_obs,
                        "hold_parcel": int(hold),
                        "train_key": train_key,
                        "fold_key": fold_key,
                        "n_eval_genes": int(gi.size),
                        "n_points": int(m.sum()),
                        "pearson_r": _pearson_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                        "rmse": _rmse_safe(x[m], y[m]) if int(m.sum()) else np.nan,
                    }
                )

    fold_perf_df = pd.DataFrame(rows)
    if len(fold_perf_df) == 0:
        raise RuntimeError("No fold-level rows were computed from cache")

    combo_df = (
        fold_perf_df.groupby(["coverage", "fold_key", "model"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_pearson=("pearson_r", "mean"),
            std_pearson=("pearson_r", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_points=("n_points", "mean"),
        )
        .sort_values(["model", "coverage", "fold_key"])
        .reset_index(drop=True)
    )
    combo_df["eval_gene_mode"] = mode
    return fold_perf_df, combo_df


def plot_fold_combo_ranked(
    combo_df: pd.DataFrame,
    prepost: Dict[str, object],
    model: str = "dlam",
    metric: str = "mean_pearson",  # mean_pearson | mean_rmse
    figsize: Tuple[float, float] = (13.8, 7.2),
    show_fold_xticklabels: bool = False,
    show_y_axis_label: bool = True,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    m = str(model).lower()
    if metric not in {"mean_pearson", "mean_rmse"}:
        raise ValueError("metric must be one of: mean_pearson, mean_rmse")

    d = combo_df[combo_df["model"].astype(str).str.lower() == m].copy()
    if len(d) == 0:
        raise RuntimeError(f"No combo rows found for model={m}")

    # Left->right worst->best.
    if metric == "mean_pearson":
        d = d.sort_values(metric, ascending=True).reset_index(drop=True)
        ylab = "Mean fold Pearson r"
    else:
        d = d.sort_values(metric, ascending=False).reset_index(drop=True)
        ylab = "Mean fold RMSE"

    x = np.arange(len(d), dtype=np.int32)
    y = d[metric].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=False)
    ax.plot(x, y, color=MODEL_COLORS.get(m, "#333333"), linewidth=1.7, alpha=0.98)
    ax.scatter(x, y, s=6.0, color=MODEL_COLORS.get(m, "#333333"), alpha=0.62, linewidths=0)
    n_combo = int(len(d))
    ax.set_title(
        f"{MODEL_LABELS.get(m, m.upper())}: fold-combo ranking ({metric}; n={n_combo:,})",
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
    need = {"model", "hold_parcel", "pearson_r", "rmse", "subject"}
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
    metric: str = "mean_pearson",  # mean_pearson | mean_rmse
    sort_by_model: str = "dlam",
    use_error_bars: bool = False,
    error_kind: str = "std",  # std | sem
    figsize: Tuple[float, float] = (16.0, 7.2),
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    if metric not in {"mean_pearson", "mean_rmse"}:
        raise ValueError("metric must be one of: mean_pearson, mean_rmse")
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
    asc = True if metric == "mean_pearson" else False
    pivot = pivot.sort_values(sort_col, ascending=asc).reset_index(drop=True)

    # long form aligned to sorted region order
    key_order = pivot[["hold_parcel", "label"]].copy()
    key_order["ord"] = np.arange(len(key_order), dtype=np.int32)
    dd = d.merge(key_order, on=["hold_parcel", "label"], how="inner")
    dd = dd.sort_values(["ord", "model"]).reset_index(drop=True)

    # error bars
    err_col = None
    if use_error_bars:
        base = "std_pearson" if metric == "mean_pearson" else "std_rmse"
        if error_kind == "std":
            err_col = base
            dd["err"] = dd[err_col].astype(float)
        else:
            n = np.maximum(dd["n_folds"].to_numpy(dtype=np.float64), 1.0)
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
        fontsize=FONT["title"] + 1,
    )
    ax.set_xlabel("Held-out region (GTEx label, sorted)", fontsize=FONT["label"] + 1)
    ax.set_ylabel("Mean Pearson r" if metric == "mean_pearson" else "Mean RMSE", fontsize=FONT["label"] + 1)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    labels = key_order["label"].astype(str).tolist()
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT["tick"])

    return fig, ax, dd


def select_subject_by_model(cfg: EDAConfig, mode: str = "median", metric: str = "pearson_r", model: str = "plam") -> str:
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
) -> Dict[str, np.ndarray]:
    p = (_model_cache_root(cfg, str(model).lower()) / f"{subject}.npz").resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    pred = z["loro_fused_subject_h"].astype(np.float64)
    truth = z["loro_truth_subject_h"].astype(np.float64)
    mask = z["loro_eval_mask"].astype(bool)
    gene_names = [str(g) for g in z["gene_names"].tolist()]

    mode = str(eval_gene_mode).lower()
    if mode not in {"all", "hvg", "custom"}:
        raise ValueError("eval_gene_mode must be one of: all, hvg, custom")
    if mode == "all":
        gi = np.arange(len(gene_names), dtype=np.int32)
    elif mode == "hvg":
        hdr = io_utils.load_gene_header_and_hvg(_resolve_repo_path(cfg.csv_path), _resolve_repo_path(cfg.hvg_path))
        hvg_set = set(str(g) for g in hdr["genes_hvg"])
        gi = np.asarray([i for i, g in enumerate(gene_names) if g in hvg_set], dtype=np.int32)
        if int(gi.size) == 0:
            raise RuntimeError(f"No HVG overlap found in cache gene_names for subject={subject}, model={model}")
    else:
        if custom_gene_list is None:
            raise ValueError("custom_gene_list is required when eval_gene_mode='custom'")
        cset = set(str(g) for g in custom_gene_list)
        gi = np.asarray([i for i, g in enumerate(gene_names) if g in cset], dtype=np.int32)
        if int(gi.size) == 0:
            raise RuntimeError(f"No custom gene overlap found in cache gene_names for subject={subject}, model={model}")

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
    density_gridsize: int = 70,
    density_cmap: str = "magma",
    density_mincnt: int = 1,
    figsize: Tuple[float, float] = (15.0, 4.8),
) -> Tuple[plt.Figure, np.ndarray, str]:
    subject = str(subject_id) if subject_id else select_subject_by_model(cfg, mode=str(subject_mode), metric="pearson_r", model=str(rank_model))
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
