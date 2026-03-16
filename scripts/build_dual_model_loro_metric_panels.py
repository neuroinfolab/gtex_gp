#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.metrics import metrics_from_vectors
from src.harmonize import fit_harmonizer
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.models.unified_generative import UnifiedGenerativeConfig, fit_global_atlas_unified, infer_subject_unified
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_eligibility,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)

DET_NAME = "deterministic_latent_transport"
UNI_NAME = "unified_probabilistic_latent_alignment"
BASE_NAME = "naive_ahba_fill"
MODEL_ORDER = [DET_NAME, UNI_NAME, BASE_NAME]
MODEL_LABELS = {
    DET_NAME: "DLAM",
    UNI_NAME: "PLAM",
    BASE_NAME: "Naive fill",
}
MODEL_COLORS = {
    DET_NAME: "#1f77b4",
    UNI_NAME: "#d62728",
    BASE_NAME: "#7f7f7f",
}
FONT = {"title": 9, "label": 8, "tick": 7, "legend": 7, "small": 6.5}


@dataclass
class Config:
    csv_path: str = "/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv"
    hvg_path: str = "/Users/erdem/Documents/github/gtex_gp/ahba_100hvg.txt"
    out_root: str = "/Users/erdem/Downloads/main_scientific_assets"
    gene_scope: str = "allgenes"
    chunk_size: int = 500
    combat_use_covariates: bool = True
    min_observed_parcels: int = 5
    c_min: int = 8
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_length_scale: float = 25.0
    gp_noise: float = 1e-3
    seed: int = 123
    smoke_subjects: int = 0


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build all-gene or HVG LORO metric panels for deterministic, unified, and naive models.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--gene-scope", choices=["allgenes", "hvg"], default=Config.gene_scope)
    p.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    p.add_argument("--combat-use-covariates", default=str(Config.combat_use_covariates).lower())
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--gp-length-scale", type=float, default=Config.gp_length_scale)
    p.add_argument("--gp-noise", type=float, default=Config.gp_noise)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--smoke-subjects", type=int, default=Config.smoke_subjects)
    a = p.parse_args()
    d = vars(a)
    d["combat_use_covariates"] = _parse_bool(d["combat_use_covariates"])
    return Config(**d)


def _suffix(cfg: Config) -> str:
    return "allgene" if str(cfg.gene_scope).lower() == "allgenes" else "hvg"


def _scalar_state(store_abs: bool = False) -> dict:
    return {
        "n_units": 0,
        "n": 0,
        "sx": 0.0,
        "sy": 0.0,
        "sxx": 0.0,
        "syy": 0.0,
        "sxy": 0.0,
        "sabs": 0.0,
        "ssq": 0.0,
        "abs": [] if store_abs else None,
    }


def _update_scalar_state(state: dict, y_true: np.ndarray, y_pred: np.ndarray, units: int = 1) -> None:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) == 0:
        return
    yt = yt[m]
    yp = yp[m]
    err = yt - yp
    state["n_units"] += int(units)
    state["n"] += int(yt.size)
    state["sx"] += float(np.sum(yt))
    state["sy"] += float(np.sum(yp))
    state["sxx"] += float(np.dot(yt, yt))
    state["syy"] += float(np.dot(yp, yp))
    state["sxy"] += float(np.dot(yt, yp))
    state["sabs"] += float(np.sum(np.abs(err)))
    state["ssq"] += float(np.dot(err, err))
    if state["abs"] is not None:
        state["abs"].append(np.abs(err).astype(np.float32, copy=False))


def _finalize_scalar_state(state: dict, include_medae: bool = False) -> dict:
    n = int(state["n"])
    if n == 0:
        out = {"n_points": 0, "pearson_r": np.nan, "rmse": np.nan, "mae": np.nan}
        if include_medae:
            out["medae"] = np.nan
        return out
    sx, sy, sxx, syy, sxy = (float(state[k]) for k in ["sx", "sy", "sxx", "syy", "sxy"])
    num = n * sxy - sx * sy
    den_term_x = n * sxx - sx * sx
    den_term_y = n * syy - sy * sy
    den = np.sqrt(max(den_term_x, 0.0) * max(den_term_y, 0.0))
    pear = float(num / den) if den > 1e-12 else np.nan
    rmse = float(np.sqrt(max(float(state["ssq"]) / n, 0.0)))
    mae = float(float(state["sabs"]) / n)
    out = {"n_points": n, "pearson_r": pear, "rmse": rmse, "mae": mae}
    if include_medae:
        arrs = state.get("abs") or []
        out["medae"] = float(np.median(np.concatenate(arrs))) if arrs else np.nan
    return out


def _vector_state(n: int) -> dict:
    return {
        "n": np.zeros(n, dtype=np.int64),
        "sx": np.zeros(n, dtype=np.float64),
        "sy": np.zeros(n, dtype=np.float64),
        "sxx": np.zeros(n, dtype=np.float64),
        "syy": np.zeros(n, dtype=np.float64),
        "sxy": np.zeros(n, dtype=np.float64),
        "sabs": np.zeros(n, dtype=np.float64),
        "ssq": np.zeros(n, dtype=np.float64),
    }


def _update_vector_state(state: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return
    yt0 = np.where(m, yt, 0.0)
    yp0 = np.where(m, yp, 0.0)
    err = yt0 - yp0
    state["n"] += m.astype(np.int64)
    state["sx"] += yt0
    state["sy"] += yp0
    state["sxx"] += yt0 * yt0
    state["syy"] += yp0 * yp0
    state["sxy"] += yt0 * yp0
    state["sabs"] += np.abs(err)
    state["ssq"] += err * err


def _finalize_vector_state(state: dict) -> dict:
    n = state["n"].astype(np.float64)
    sx, sy, sxx, syy, sxy = state["sx"], state["sy"], state["sxx"], state["syy"], state["sxy"]
    num = n * sxy - sx * sy
    den_x = n * sxx - sx * sx
    den_y = n * syy - sy * sy
    den = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0))
    pear = np.full_like(n, np.nan, dtype=np.float64)
    m = (n >= 2.0) & (den > 1e-12)
    pear[m] = num[m] / den[m]
    rmse = np.full_like(n, np.nan, dtype=np.float64)
    mae = np.full_like(n, np.nan, dtype=np.float64)
    mz = n > 0.0
    rmse[mz] = np.sqrt(np.maximum(state["ssq"][mz] / n[mz], 0.0))
    mae[mz] = state["sabs"][mz] / n[mz]
    return {"n_points": state["n"].copy(), "pearson_r": pear, "rmse": rmse, "mae": mae}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    met = metrics_from_vectors(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64))
    met["mean_abs_error"] = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))) if len(y_true) else np.nan
    return met


def _subject_sort_order(subject_df: pd.DataFrame) -> List[str]:
    u = subject_df[subject_df["model_name"] == UNI_NAME].copy()
    u = u.sort_values(["pearson_r", "rmse", "subject"], ascending=[False, True, True])
    return u["subject"].astype(str).tolist()


def _plot_subject_metrics(subject_df: pd.DataFrame, out_path: Path) -> None:
    order = _subject_sort_order(subject_df)
    pv_p = subject_df.pivot(index="subject", columns="model_name", values="pearson_r").reindex(index=order, columns=MODEL_ORDER)
    pv_r = subject_df.pivot(index="subject", columns="model_name", values="rmse").reindex(index=order, columns=MODEL_ORDER)
    cov = subject_df[["subject", "coverage_tier"]].drop_duplicates().set_index("subject").reindex(order)
    cov_arr = (cov["coverage_tier"].fillna("lt8") == "ge8").astype(int).to_numpy()[:, None]

    summary = (
        subject_df.groupby("model_name")[["pearson_r", "rmse", "delta_rmse_vs_naive"]]
        .mean(numeric_only=True)
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    fig = plt.figure(figsize=(11.4, 10.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.14, 1.0, 1.0], height_ratios=[1.0, 0.42], wspace=0.16, hspace=0.28)
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])
    ax_b2 = fig.add_subplot(gs[1, 2])
    ax_blank.axis("off")

    ax_cov.imshow(cov_arr, aspect="auto", cmap=ListedColormap(["#d9d9d9", "#4daf4a"]), interpolation="none")
    ax_cov.set_title("Coverage", fontsize=FONT["title"])
    ax_cov.set_xticks([])
    ax_cov.set_yticks([])

    sns.heatmap(pv_p, ax=ax_p, cmap="coolwarm", center=0.5, vmin=0.0, vmax=1.0, cbar_kws={"shrink": 0.65, "label": "Pearson r"}, yticklabels=False)
    ax_p.set_title("Subject-level Pearson", fontsize=FONT["title"])
    ax_p.set_xlabel("")
    ax_p.set_ylabel("Subjects (sorted by unified Pearson)", fontsize=FONT["label"])
    ax_p.set_xticklabels([MODEL_LABELS[c] for c in pv_p.columns], rotation=0, ha="center", fontsize=FONT["tick"])

    sns.heatmap(pv_r, ax=ax_r, cmap="viridis_r", cbar_kws={"shrink": 0.65, "label": "RMSE"}, yticklabels=False)
    ax_r.set_title("Subject-level RMSE", fontsize=FONT["title"])
    ax_r.set_xlabel("")
    ax_r.set_ylabel("")
    ax_r.set_xticklabels([MODEL_LABELS[c] for c in pv_r.columns], rotation=0, ha="center", fontsize=FONT["tick"])

    xx = np.arange(len(summary))
    colors = [MODEL_COLORS[m] for m in summary["model_name"]]
    ax_b1.bar(xx, summary["pearson_r"], color=colors)
    ax_b1.set_title("Mean Pearson", fontsize=FONT["title"])
    ax_b1.set_xticks(xx)
    ax_b1.set_xticklabels([MODEL_LABELS[m] for m in summary["model_name"]], rotation=18, ha="right", fontsize=FONT["tick"])
    ax_b1.grid(True, axis="y", alpha=0.15)

    ax_b2.bar(xx, summary["rmse"], color=colors)
    ax_b2.set_title("Mean RMSE", fontsize=FONT["title"])
    ax_b2.set_xticks(xx)
    ax_b2.set_xticklabels([MODEL_LABELS[m] for m in summary["model_name"]], rotation=18, ha="right", fontsize=FONT["tick"])
    ax_b2.grid(True, axis="y", alpha=0.15)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_parcel_metrics(parcel_df: pd.DataFrame, out_path: Path) -> None:
    parcel_order = parcel_df[["parcel_idx", "parcel_name"]].drop_duplicates().sort_values("parcel_idx")["parcel_name"].tolist()
    pv_p = parcel_df.pivot(index="parcel_name", columns="model_name", values="mean_pearson").reindex(index=parcel_order, columns=MODEL_ORDER)
    pv_r = parcel_df.pivot(index="parcel_name", columns="model_name", values="mean_rmse").reindex(index=parcel_order, columns=MODEL_ORDER)

    non_base = parcel_df[parcel_df["model_name"] != BASE_NAME].copy()
    pv_delta = non_base.pivot(index="parcel_name", columns="model_name", values="delta_rmse_vs_naive").fillna(0.0)
    score = pv_delta.mean(axis=1)
    top = score.sort_values(ascending=False).head(5)
    bot = score.sort_values(ascending=True).head(5)
    keep = top.index.tolist() + [x for x in bot.index.tolist() if x not in top.index.tolist()]
    delta_plot = pv_delta.loc[keep].reset_index().melt(id_vars="parcel_name", var_name="model_name", value_name="delta_rmse_vs_naive")

    fig = plt.figure(figsize=(12.8, 11.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.1], wspace=0.28)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])

    sns.heatmap(pv_p, ax=ax_p, cmap="coolwarm", center=0.5, vmin=0.0, vmax=1.0, cbar_kws={"shrink": 0.55, "label": "Pearson r"}, yticklabels=10)
    ax_p.set_title("Parcel-level Pearson", fontsize=FONT["title"])
    ax_p.set_xlabel("")
    ax_p.set_ylabel("Canonical parcel", fontsize=FONT["label"])
    ax_p.set_xticklabels([MODEL_LABELS[c] for c in pv_p.columns], rotation=90, fontsize=FONT["tick"])

    sns.heatmap(pv_r, ax=ax_r, cmap="viridis_r", cbar_kws={"shrink": 0.55, "label": "RMSE"}, yticklabels=False)
    ax_r.set_title("Parcel-level RMSE", fontsize=FONT["title"])
    ax_r.set_xlabel("")
    ax_r.set_ylabel("")
    ax_r.set_xticklabels([MODEL_LABELS[c] for c in pv_r.columns], rotation=90, fontsize=FONT["tick"])

    sns.barplot(data=delta_plot, y="parcel_name", x="delta_rmse_vs_naive", hue="model_name", orient="h", palette=MODEL_COLORS, ax=ax_b)
    ax_b.axvline(0.0, color="black", linewidth=0.8)
    ax_b.set_title("Improvement over naive baseline", fontsize=FONT["title"])
    ax_b.set_xlabel("Naive RMSE - model RMSE", fontsize=FONT["label"])
    ax_b.set_ylabel("")
    if ax_b.legend_:
        labels = [MODEL_LABELS[m] for m in [DET_NAME, UNI_NAME]]
        for txt, lab in zip(ax_b.legend_.texts, labels):
            txt.set_text(lab)
        ax_b.legend(frameon=False, fontsize=FONT["legend"], title="")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_metrics(gene_df: pd.DataFrame, out_path: Path) -> None:
    baseline = gene_df[gene_df["model_name"] == BASE_NAME][["gene", "pearson_r"]].rename(columns={"pearson_r": "baseline_pearson"})
    ord_df = gene_df.merge(baseline, on="gene", how="left")
    gene_order = (
        ord_df[["gene", "baseline_pearson"]]
        .drop_duplicates()
        .sort_values(["baseline_pearson", "gene"], ascending=[False, True], na_position="last")["gene"].tolist()
    )

    pv_p = gene_df.pivot(index="gene", columns="model_name", values="pearson_r").reindex(index=gene_order, columns=MODEL_ORDER)
    pv_r = gene_df.pivot(index="gene", columns="model_name", values="rmse").reindex(index=gene_order, columns=MODEL_ORDER)

    non_base = gene_df[gene_df["model_name"] != BASE_NAME].copy()
    pv_delta = non_base.pivot(index="gene", columns="model_name", values="delta_rmse_vs_naive").fillna(0.0)
    score = pv_delta.mean(axis=1)
    top = score.sort_values(ascending=False).head(6)
    bot = score.sort_values(ascending=True).head(6)
    keep = top.index.tolist() + [x for x in bot.index.tolist() if x not in top.index.tolist()]
    delta_plot = pv_delta.loc[keep].reset_index().melt(id_vars="gene", var_name="model_name", value_name="delta_rmse_vs_naive")
    keep_order = keep[::-1]
    delta_plot["gene"] = pd.Categorical(delta_plot["gene"].astype(str), categories=keep_order, ordered=True)

    fig = plt.figure(figsize=(12.8, 10.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.1], wspace=0.28)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])

    sns.heatmap(pv_p, ax=ax_p, cmap="coolwarm", center=0.5, vmin=-1.0, vmax=1.0, cbar_kws={"shrink": 0.55, "label": "Pearson r"}, yticklabels=False)
    ax_p.set_title("Gene-level Pearson", fontsize=FONT["title"])
    ax_p.set_xlabel("")
    ax_p.set_ylabel("Genes", fontsize=FONT["label"])
    ax_p.set_xticklabels([MODEL_LABELS[c] for c in pv_p.columns], rotation=90, fontsize=FONT["tick"])

    sns.heatmap(pv_r, ax=ax_r, cmap="viridis_r", cbar_kws={"shrink": 0.55, "label": "RMSE"}, yticklabels=False)
    ax_r.set_title("Gene-level RMSE", fontsize=FONT["title"])
    ax_r.set_xlabel("")
    ax_r.set_ylabel("")
    ax_r.set_xticklabels([MODEL_LABELS[c] for c in pv_r.columns], rotation=90, fontsize=FONT["tick"])

    sns.barplot(
        data=delta_plot,
        y="gene",
        x="delta_rmse_vs_naive",
        hue="model_name",
        hue_order=[DET_NAME, UNI_NAME],
        orient="h",
        palette=MODEL_COLORS,
        ax=ax_b,
    )
    ax_b.axvline(0.0, color="black", linewidth=0.8)
    ax_b.set_title("Improvement over naive baseline", fontsize=FONT["title"])
    ax_b.set_xlabel("Naive RMSE - model RMSE", fontsize=FONT["label"])
    ax_b.set_ylabel("")
    if ax_b.legend_:
        labels = [MODEL_LABELS[m] for m in [DET_NAME, UNI_NAME]]
        for txt, lab in zip(ax_b.legend_.texts, labels):
            txt.set_text(lab)
        ax_b.legend(frameon=False, fontsize=FONT["legend"], title="", loc="lower right")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_method_summary_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    d = summary_df.set_index("model_name").reindex(MODEL_ORDER).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0), constrained_layout=True)
    x = np.arange(len(d))
    colors = [MODEL_COLORS[m] for m in d["model_name"]]
    fields = [
        ("mean_subject_pearson", "Mean Pearson"),
        ("mean_subject_rmse", "Mean RMSE"),
        ("mean_delta_rmse_vs_naive", "Naive RMSE - model RMSE"),
        ("frac_subjects_better_than_naive", "Frac. better than naive"),
    ]
    for ax, (field, title) in zip(axes, fields):
        ax.bar(x, d[field], color=colors)
        if "delta" in field:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=FONT["title"])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in d["model_name"]], rotation=20, ha="right", fontsize=FONT["tick"])
        ax.grid(True, axis="y", alpha=0.15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _subject_state_rows(subject_states: dict, subject_info: dict, base_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for sid, info in subject_info.items():
            state = subject_states[model_name].get(sid, _scalar_state(store_abs=True))
            met = _finalize_scalar_state(state, include_medae=True)
            rows.append(
                {
                    "model_name": model_name,
                    "subject": sid,
                    "n_obs_parcels": int(info["n_obs_parcels"]),
                    "pearson_r": float(met["pearson_r"]),
                    "rmse": float(met["rmse"]),
                    "mae": float(met["mae"]),
                    "medae": float(met["medae"]),
                    "coverage_tier": str(info["coverage_tier"]),
                }
            )
    df = pd.DataFrame(rows)
    base = df[df["model_name"] == BASE_NAME][["subject", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    df = df.merge(base, on="subject", how="left")
    df["delta_rmse_vs_naive"] = np.where(df["model_name"] == BASE_NAME, 0.0, df["baseline_rmse"] - df["rmse"])
    return df.sort_values(["model_name", "subject"]).reset_index(drop=True)


def _parcel_rows(parcel_states: dict, target_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pmap = dict(zip(target_meta["parcel_idx"].astype(int).tolist(), target_meta["tissue_or_parcel"].astype(str).tolist()))
    all_pidx = target_meta["parcel_idx"].astype(int).tolist()

    base_map = {}
    for pidx in all_pidx:
        met = _finalize_scalar_state(parcel_states[BASE_NAME][int(pidx)], include_medae=False)
        base_map[int(pidx)] = float(met["rmse"])

    for model_name in MODEL_ORDER:
        for pidx in all_pidx:
            met = _finalize_scalar_state(parcel_states[model_name][int(pidx)], include_medae=False)
            n_folds = int(parcel_states[model_name][int(pidx)].get("n_units", 0))
            delta = np.nan
            if n_folds > 0:
                if model_name == BASE_NAME:
                    delta = 0.0
                elif np.isfinite(base_map[int(pidx)]) and np.isfinite(met["rmse"]):
                    delta = float(base_map[int(pidx)] - met["rmse"])
            rows.append(
                {
                    "model_name": model_name,
                    "parcel_idx": int(pidx),
                    "parcel_name": pmap[int(pidx)],
                    "n_folds": n_folds,
                    "mean_pearson": float(met["pearson_r"]),
                    "mean_rmse": float(met["rmse"]),
                    "mean_mae": float(met["mae"]),
                    "mean_abs_error": float(met["mae"]),
                    "delta_rmse_vs_naive": delta,
                }
            )
    return pd.DataFrame(rows).sort_values(["parcel_idx", "model_name"]).reset_index(drop=True)


def _gene_rows(gene_states: dict, genes: List[str]) -> pd.DataFrame:
    rows = []
    finalized = {m: _finalize_vector_state(gene_states[m]) for m in MODEL_ORDER}
    base_rmse = finalized[BASE_NAME]["rmse"]
    base_n = finalized[BASE_NAME]["n_points"]
    for model_name in MODEL_ORDER:
        cur = finalized[model_name]
        for gi, gene in enumerate(genes):
            rows.append(
                {
                    "model_name": model_name,
                    "gene": gene,
                    "n_points": int(cur["n_points"][gi]),
                    "pearson_r": float(cur["pearson_r"][gi]),
                    "rmse": float(cur["rmse"][gi]),
                    "mae": float(cur["mae"][gi]),
                    "delta_rmse_vs_naive": 0.0 if model_name == BASE_NAME else float(base_rmse[gi] - cur["rmse"][gi]),
                    "baseline_n_points": int(base_n[gi]),
                }
            )
    df = pd.DataFrame(rows)
    df["gene"] = pd.Categorical(df["gene"], categories=genes, ordered=True)
    return df.sort_values(["gene", "model_name"]).reset_index(drop=True)


def _method_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = subject_df[subject_df["model_name"] == model_name].copy()
        rows.append(
            {
                "model_name": model_name,
                "n_subjects": int(sub["subject"].nunique()),
                "mean_subject_pearson": float(sub["pearson_r"].mean()),
                "mean_subject_rmse": float(sub["rmse"].mean()),
                "mean_subject_mae": float(sub["mae"].mean()),
                "mean_delta_rmse_vs_naive": 0.0 if model_name == BASE_NAME else float(sub["delta_rmse_vs_naive"].mean()),
                "frac_subjects_better_than_naive": 0.0 if model_name == BASE_NAME else float((sub["delta_rmse_vs_naive"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    sns.set_theme(style="whitegrid", context="paper")

    out_root = Path(cfg.out_root).resolve()
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    header = io_utils.load_gene_header_and_hvg(Path(cfg.csv_path), Path(cfg.hvg_path))
    genes = header["genes_all"] if str(cfg.gene_scope).lower() == "allgenes" else header["genes_hvg"]
    if len(genes) == 0:
        raise RuntimeError(f"No genes found for scope={cfg.gene_scope}")
    scope_suffix = _suffix(cfg)

    df = io_utils.read_expression_subset(Path(cfg.csv_path), genes)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    elig = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    subjects = elig[elig["eligible"]]["subject"].astype(str).tolist()
    if cfg.smoke_subjects > 0:
        subjects = subjects[: int(cfg.smoke_subjects)]
    if not subjects:
        raise RuntimeError("No eligible subjects for LORO diagnostics.")

    combat_cfg = SimpleNamespace(combat_use_covariates=cfg.combat_use_covariates)
    cfg_hash = io_utils.hash_config(asdict(cfg))
    subject_info = {}
    fold_rows: List[dict] = []

    subject_states = {m: {} for m in MODEL_ORDER}
    parcel_states = {m: {int(p): _scalar_state(store_abs=False) for p in target_meta["parcel_idx"].astype(int).tolist()} for m in MODEL_ORDER}
    gene_states = {m: _vector_state(len(genes)) for m in MODEL_ORDER}

    for i, sid in enumerate(subjects, start=1):
        sub_all = gtex_raw[gtex_raw["subject"] == sid].copy()
        obs_idx_all = sorted(set(int(x) for x in sub_all["parcel_idx"].tolist()))
        coverage_tier = "ge8" if len(obs_idx_all) >= cfg.c_min else "lt8"
        subject_info[sid] = {"n_obs_parcels": len(obs_idx_all), "coverage_tier": coverage_tier}
        print(f"[{scope_suffix} loro] {i}/{len(subjects)} {sid} obs={len(obs_idx_all)}")

        for fold_id, hold in enumerate(obs_idx_all):
            train_mask = ~((gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold)))
            gtex_train = gtex_raw[train_mask].copy()
            harm = fit_harmonizer(ahba_raw, gtex_train, genes, method="combat", cfg=combat_cfg)
            ahba_h = harm.transform(ahba_raw, "AHBA")
            gtex_h = harm.transform(gtex_train, "GTEX")
            ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")

            hold_rows_raw = gtex_raw[(gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold))].copy()
            if len(hold_rows_raw) == 0:
                continue
            hold_h = harm.transform(hold_rows_raw, "GTEX")
            truth = hold_h[genes].to_numpy(dtype=np.float64).mean(axis=0)
            base = np.asarray(ahba_h_full[int(hold), :], dtype=np.float64)

            subj_h_fold = gtex_h[gtex_h["subject"] == sid].copy()
            subj_raw_fold = gtex_train[gtex_train["subject"] == sid].copy()
            obs_idx_fold, xh_fold, xr_fold = build_subject_observed_matrices(subj_h_fold, subj_raw_fold, genes)
            if len(obs_idx_fold) < 2:
                continue

            det_pred_full = None
            try:
                ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
                det_pred, _ = run_subject(
                    {
                        "subject": sid,
                        "obs_idx": obs_idx_fold,
                        "X_obs_h": xh_fold,
                        "X_obs_raw": xr_fold,
                        "coords_full": coords_full,
                        "target_meta": target_meta,
                    },
                    {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_pls["T"]},
                    {
                        "harmonizer": harm,
                        "basis_model": "affine_gl3",
                        "strategy": "constrained_anchor",
                        "spatial_method": "rbf",
                        "n_comp_target": cfg.n_comp_target,
                        "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
                        "rbf_smoothing": cfg.rbf_smoothing,
                        "gp_rbf_length": cfg.gp_length_scale,
                        "seed": cfg.seed,
                        "c_min": cfg.c_min,
                        "distance_d0": 45.0,
                        "distance_tau": 10.0,
                        "uncertainty_shrink": False,
                    },
                    asdict(cfg),
                )
                det_pred_full = np.asarray(det_pred["X_full_h"], dtype=np.float64)
            except Exception:
                det_pred_full = None

            uni_pred_full = None
            uni_std = np.nan
            try:
                ucfg = UnifiedGenerativeConfig(
                    latent_dim=3,
                    max_iters=5,
                    lambda_w=1.0,
                    lambda_z=1.0,
                    lambda_cal_a=10.0,
                    lambda_cal_b=10.0,
                    gp_length_scale=cfg.gp_length_scale,
                    gp_noise=cfg.gp_noise,
                    robust_loss="student_t",
                    heteroscedastic=True,
                    calibration_mode="hier_affine_map",
                    uncertainty_shrink=False,
                    random_state=cfg.seed,
                )
                atlas = fit_global_atlas_unified(ahba_h_full, coords_full, ucfg)
                uni_res = infer_subject_unified(
                    {"subject": sid, "obs_idx": obs_idx_fold, "X_obs_h": xh_fold},
                    atlas,
                    ucfg,
                    fold_ctx={"prior_h": ahba_h_full, "obs_idx": obs_idx_fold},
                )
                uni_pred_full = np.asarray(uni_res["x_hat_h_full"], dtype=np.float64)
                uni_std = float(np.sqrt(np.maximum(np.mean(uni_res["uvar_full"][int(hold), :]), 0.0)))
            except Exception:
                uni_pred_full = None
                uni_std = np.nan

            model_preds = {
                BASE_NAME: (base, 0.0),
                DET_NAME: (det_pred_full[int(hold), :] if det_pred_full is not None else None, np.nan),
                UNI_NAME: (uni_pred_full[int(hold), :] if uni_pred_full is not None else None, uni_std),
            }
            for model_name, (pred, pred_std) in model_preds.items():
                if pred is None:
                    continue
                met = _metrics(truth, pred)
                fold_rows.append(
                    {
                        "model_name": model_name,
                        "subject": sid,
                        "held_out_parcel": int(hold),
                        "fold_id": int(fold_id),
                        "n_obs_parcels": int(len(obs_idx_all)),
                        "coverage_tier": coverage_tier,
                        "pearson_r": float(met["pearson_r"]),
                        "rmse": float(met["rmse"]),
                        "mae": float(met["mae"]),
                        "medae": float(met["medae"]),
                        "baseline_rmse": float(np.sqrt(np.mean((truth - base) ** 2))),
                        "hold_pred_std": float(pred_std),
                        "hold_abs_error_mean": float(met["mean_abs_error"]),
                        "config_hash": cfg_hash,
                    }
                )
                _update_scalar_state(subject_states[model_name].setdefault(sid, _scalar_state(store_abs=True)), truth, pred)
                _update_scalar_state(parcel_states[model_name][int(hold)], truth, pred)
                _update_vector_state(gene_states[model_name], truth, pred)

    if not fold_rows:
        raise RuntimeError(f"No {scope_suffix} LORO payloads were generated.")

    subject_df = _subject_state_rows(subject_states, subject_info)
    parcel_df = _parcel_rows(parcel_states, target_meta)
    gene_df = _gene_rows(gene_states, genes)
    method_summary_df = _method_summary(subject_df)

    subject_df.to_csv(tab_dir / f"loro_subject_metrics_{scope_suffix}.csv", index=False)
    parcel_df.to_csv(tab_dir / f"loro_parcel_metrics_{scope_suffix}.csv", index=False)
    gene_df.to_csv(tab_dir / f"loro_gene_metrics_{scope_suffix}.csv", index=False)
    method_summary_df.to_csv(tab_dir / f"loro_method_summary_{scope_suffix}.csv", index=False)

    _plot_subject_metrics(subject_df, fig_dir / f"loro_subject_metrics_{scope_suffix}.pdf")
    _plot_parcel_metrics(parcel_df, fig_dir / f"loro_parcel_metrics_{scope_suffix}.pdf")
    _plot_gene_metrics(gene_df, fig_dir / f"loro_gene_metrics_{scope_suffix}.pdf")
    _plot_method_summary_bars(method_summary_df, fig_dir / f"loro_method_summary_bars_{scope_suffix}.pdf")

    qc = {
        "models": MODEL_ORDER,
        "gene_scope": cfg.gene_scope,
        "n_subjects": int(subject_df["subject"].nunique()),
        "n_parcels": int(parcel_df["parcel_idx"].nunique()),
        "n_genes": int(gene_df["gene"].nunique()),
        "n_folds": int(len(fold_rows)),
        "build_timestamp": io_utils.utc_timestamp(),
        "config_hash": cfg_hash,
        "figure_files": [
            f"loro_subject_metrics_{scope_suffix}.pdf",
            f"loro_parcel_metrics_{scope_suffix}.pdf",
            f"loro_gene_metrics_{scope_suffix}.pdf",
            f"loro_method_summary_bars_{scope_suffix}.pdf",
        ],
    }
    (tab_dir / f"loro_metric_qc_manifest_{scope_suffix}.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Built {cfg.gene_scope} LORO metric panels under {out_root}")


if __name__ == "__main__":
    main()
