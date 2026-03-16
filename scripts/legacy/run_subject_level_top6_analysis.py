#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.loro import run_loro
from src.harmonize import fit_harmonizer
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)


@dataclass
class Config:
    allgenes_run_root: str = "out/vnext_alignment_allgenes"
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/vnext_alignment_allgenes_subject"
    seed: int = 123
    representative_per_bin: int = 3
    coverage_threshold: int = 8
    generate_raw_space_loro: bool = True
    max_subject_panels: int = 9
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_rbf_length: float = 25.0
    distance_d0: float = 45.0
    distance_tau: float = 10.0
    c_min: int = 8
    whiten_eps: float = 1e-4
    min_group_samples: int = 200
    combat_use_covariates: bool = True
    hier_lambda_a: float = 10.0
    hier_lambda_b: float = 10.0
    hier_base_method: str = "robustz_affine"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Single-subject top6 benchmark and visualization package.")
    p.add_argument("--allgenes-run-root", default=Config.allgenes_run_root)
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--representative-per-bin", type=int, default=Config.representative_per_bin)
    p.add_argument("--coverage-threshold", type=int, default=Config.coverage_threshold)
    p.add_argument("--generate-raw-space-loro", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.generate_raw_space_loro)
    p.add_argument("--max-subject-panels", type=int, default=Config.max_subject_panels)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--gp-rbf-length", type=float, default=Config.gp_rbf_length)
    p.add_argument("--distance-d0", type=float, default=Config.distance_d0)
    p.add_argument("--distance-tau", type=float, default=Config.distance_tau)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--min-group-samples", type=int, default=Config.min_group_samples)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.combat_use_covariates)
    p.add_argument("--hier-lambda-a", type=float, default=Config.hier_lambda_a)
    p.add_argument("--hier-lambda-b", type=float, default=Config.hier_lambda_b)
    p.add_argument("--hier-base-method", default=Config.hier_base_method)
    a = p.parse_args()
    return Config(**vars(a))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_combo(combo: str) -> Tuple[str, str, str, str]:
    parts = str(combo).split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid combo: {combo}")
    return parts[0], parts[1], parts[2], parts[3]


def _fit_harmonizer_for_strategy(
    cfg: Config,
    hm: str,
    strategy: str,
    ahba_train: pd.DataFrame,
    gtex_train: pd.DataFrame,
    genes: List[str],
    subject_subset: List[str] | None = None,
):
    method = "piecewise_robustz" if strategy == "piecewise_harmonization" else hm
    cfg_use = cfg
    if method == "hier_affine" and subject_subset is not None:
        cfg_use = SimpleNamespace(**asdict(cfg), hier_subject_subset=[str(s) for s in subject_subset])
    return fit_harmonizer(ahba_train, gtex_train, genes, method=method, cfg=cfg_use)


def _holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        adj = (n - i) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out


def _rank_biserial(diff: np.ndarray) -> float:
    d = diff[np.isfinite(diff)]
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    pos = float(np.sum(r[d > 0]))
    neg = float(np.sum(r[d < 0]))
    den = pos + neg
    return (pos - neg) / den if den > 0 else np.nan


def _build_subject_score(df: pd.DataFrame) -> pd.DataFrame:
    # higher is better
    out = df.copy()
    out["rank_pearson"] = out.groupby("subject")["pearson_r"].rank(method="average", ascending=False)
    out["rank_rmse"] = out.groupby("subject")["rmse"].rank(method="average", ascending=True)
    out["rank_better"] = out.groupby("subject")["better_than_baseline_rmse"].rank(method="average", ascending=False)
    out["subject_score"] = -(out["rank_pearson"] + out["rank_rmse"] + 0.5 * out["rank_better"])
    return out


def _render_report(md_path: Path, pdf_path: Path) -> None:
    if shutil.which("pandoc") is None or shutil.which("tectonic") is None:
        return
    cmd = [
        "pandoc",
        str(md_path.name),
        "--standalone",
        "--from",
        "markdown+tex_math_dollars",
        "--pdf-engine=tectonic",
        "-o",
        str(pdf_path.name),
    ]
    proc = subprocess.run(cmd, cwd=str(md_path.parent), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PDF render failed:\n{proc.stderr}")


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    sns.set_theme(style="whitegrid", context="talk")

    root = Path(".").resolve()
    run_root = (root / cfg.allgenes_run_root).resolve()
    out_root = (root / cfg.out_root).resolve()
    tdir = out_root / "tables"
    fdir = out_root / "figures"
    for d in [out_root, tdir, fdir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load existing all-gene outputs.
    summ = pd.read_csv(run_root / "tables" / "subject_loro_summary_allgenes.csv")
    folds = pd.read_csv(run_root / "tables" / "subject_loro_folds_allgenes.csv")
    grid = pd.read_csv(run_root / "tables" / "model_grid_summary_allgenes.csv")
    gate = pd.read_csv(run_root / "tables" / "rollout_gate_summary_allgenes.csv")
    combos = gate.sort_values(["gate_pass", "mean_pearson"], ascending=[False, False])["model_combo"].astype(str).tolist()
    combos = combos[:6]

    summ = summ[summ["model_combo"].isin(combos)].copy()
    folds = folds[folds["model_combo"].isin(combos)].copy()

    # Subject-method matrix (primary harmonized).
    sm = _build_subject_score(summ)
    sm["delta_rmse_vs_baseline"] = sm["rmse"] - sm["baseline_rmse"]
    sm.to_csv(tdir / "subject_method_matrix.csv", index=False)

    # Per subject winner map.
    win = (
        sm.sort_values(["subject", "subject_score", "pearson_r", "rmse"], ascending=[True, False, False, True])
        .groupby("subject", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    win = win[
        ["subject", "model_combo", "subject_score", "pearson_r", "rmse", "baseline_rmse", "n_obs_parcels", "coverage_tier"]
    ].rename(columns={"model_combo": "winner_model_combo"})
    win.to_csv(tdir / "per_subject_winner_map.csv", index=False)

    # Global subject ranking by best available score.
    subj_rank = (
        win.sort_values(["subject_score", "pearson_r", "rmse"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    subj_rank["global_rank"] = np.arange(1, len(subj_rank) + 1, dtype=np.int32)
    subj_rank.to_csv(tdir / "subject_ranking_global.csv", index=False)

    # Coverage stratified.
    cov_rows = []
    for (combo, nobs), g in sm.groupby(["model_combo", "n_obs_parcels"]):
        cov_rows.append(
            {
                "space": "harmonized",
                "model_combo": combo,
                "n_obs_parcels": int(nobs),
                "n_subjects": int(g["subject"].nunique()),
                "mean_pearson": float(g["pearson_r"].mean()),
                "mean_rmse": float(g["rmse"].mean()),
                "mean_baseline_rmse": float(g["baseline_rmse"].mean()),
            }
        )
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(tdir / "coverage_stratified_subject_metrics.csv", index=False)

    # Method robustness.
    rob_rows = []
    for combo, g in sm.groupby("model_combo"):
        rob_rows.append(
            {
                "model_combo": combo,
                "n_subjects": int(g["subject"].nunique()),
                "mean_pearson": float(g["pearson_r"].mean()),
                "median_pearson": float(g["pearson_r"].median()),
                "p5_pearson": float(np.percentile(g["pearson_r"], 5)),
                "p10_pearson": float(np.percentile(g["pearson_r"], 10)),
                "mean_rmse": float(g["rmse"].mean()),
                "median_rmse": float(g["rmse"].median()),
                "p90_rmse": float(np.percentile(g["rmse"], 90)),
                "p95_rmse": float(np.percentile(g["rmse"], 95)),
                "iqr_rmse": float(np.percentile(g["rmse"], 75) - np.percentile(g["rmse"], 25)),
                "frac_better_baseline_rmse": float(g["better_than_baseline_rmse"].mean()),
            }
        )
    robustness = pd.DataFrame(rob_rows).sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
    robustness.to_csv(tdir / "method_subject_robustness.csv", index=False)

    # Pairwise stats.
    pair_rows = []
    pvt = sm.pivot_table(index="subject", columns="model_combo", values=["pearson_r", "rmse"], aggfunc="first")
    best_combo = str(robustness.iloc[0]["model_combo"])
    for combo in combos:
        if combo == best_combo:
            continue
        for metric in ["pearson_r", "rmse"]:
            x = pvt[(metric, combo)].to_numpy(dtype=np.float64)
            y = pvt[(metric, best_combo)].to_numpy(dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            d = x[m] - y[m]
            if len(d) == 0:
                wp = np.nan
                sp = np.nan
            else:
                try:
                    wp = float(stats.wilcoxon(d).pvalue)
                except Exception:
                    wp = np.nan
                sp = float(stats.binomtest(int(np.sum(d > 0)), int(np.sum(np.abs(d) > 1e-12)), 0.5).pvalue) if int(np.sum(np.abs(d) > 1e-12)) > 0 else np.nan
            pair_rows.append(
                {
                    "candidate_model_combo": combo,
                    "reference_model_combo": best_combo,
                    "metric": metric,
                    "n_subjects": int(np.sum(m)),
                    "median_delta_candidate_minus_ref": float(np.nanmedian(d)) if len(d) else np.nan,
                    "wilcoxon_p": wp,
                    "sign_test_p": sp,
                    "rank_biserial_effect": _rank_biserial(d),
                }
            )
    pair = pd.DataFrame(pair_rows)
    if len(pair):
        pair["wilcoxon_p_holm"] = _holm(pair["wilcoxon_p"].fillna(1.0).to_numpy())
        pair["sign_test_p_holm"] = _holm(pair["sign_test_p"].fillna(1.0).to_numpy())
    pair.to_csv(tdir / "method_pairwise_stats.csv", index=False)

    # Approx raw-space LORO diagnostics (targeted recomputation on HVG with global harmonizer per method).
    raw_subject_rows = []
    if cfg.generate_raw_space_loro:
        header = io_utils.load_gene_header_and_hvg((root / cfg.csv_path).resolve(), (root / cfg.hvg_path).resolve())
        genes = header["genes_hvg"]
        df = io_utils.read_expression_subset((root / cfg.csv_path).resolve(), genes)
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
        eligible_subjects = sorted(sm["subject"].unique().tolist())

        for combo in combos:
            hm, bm, st, spm = _parse_combo(combo)
            harmonizer = _fit_harmonizer_for_strategy(cfg, hm, st, ahba_raw, gtex_raw, genes)
            ahba_h = harmonizer.transform(ahba_raw, "AHBA")
            gtex_h = harmonizer.transform(gtex_raw, "GTEX")
            ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")
            ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
            ahba_ref_T = ahba_pls["T"]
            for sid in eligible_subjects:
                subj_h = gtex_h[gtex_h["subject"] == sid].copy()
                subj_r = gtex_raw[gtex_raw["subject"] == sid].copy()
                obs_idx, X_obs_h, X_obs_raw = build_subject_observed_matrices(subj_h, subj_r, genes)
                if len(obs_idx) < 3:
                    continue
                idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}

                def _pipeline_raw(hold: int, train_idx: np.ndarray):
                    sb = {
                        "subject": sid,
                        "obs_idx": obs_idx,
                        "X_obs_h": X_obs_h,
                        "X_obs_raw": X_obs_raw,
                        "coords_full": coords_full,
                        "target_meta": target_meta,
                    }
                    ab = {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_ref_T}
                    mb = {
                        "harmonizer": harmonizer,
                        "basis_model": bm,
                        "strategy": st,
                        "spatial_method": spm,
                        "n_comp_target": cfg.n_comp_target,
                        "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
                        "rbf_smoothing": cfg.rbf_smoothing,
                        "gp_rbf_length": cfg.gp_rbf_length,
                        "seed": cfg.seed,
                        "c_min": cfg.c_min,
                        "distance_d0": cfg.distance_d0,
                        "distance_tau": cfg.distance_tau,
                        "uncertainty_shrink": bool(st == "gp_uncertainty"),
                    }
                    pred, _ = run_subject(sb, ab, mb, asdict(cfg), fold_mask=int(hold))
                    y_pred_raw = pred["X_full_raw"][int(hold), :]
                    y_true_raw = X_obs_raw[idx_to_pos[int(hold)], :]
                    return y_pred_raw, y_true_raw

                rows = []
                for hold in obs_idx.tolist():
                    train_idx = np.asarray([p for p in obs_idx.tolist() if int(p) != int(hold)], dtype=np.int32)
                    if len(train_idx) < 2:
                        continue
                    yp, yt = _pipeline_raw(int(hold), train_idx)
                    if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
                        pr = float(stats.pearsonr(yt, yp).statistic)
                    else:
                        pr = np.nan
                    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
                    rows.append((pr, rmse))
                if len(rows) == 0:
                    continue
                arr = np.asarray(rows, dtype=np.float64)
                raw_subject_rows.append(
                    {
                        "subject": sid,
                        "model_combo": combo,
                        "raw_pearson_r": float(np.nanmean(arr[:, 0])),
                        "raw_rmse": float(np.nanmean(arr[:, 1])),
                        "n_folds_raw": int(arr.shape[0]),
                    }
                )
    raw_df = pd.DataFrame(raw_subject_rows)
    raw_df.to_csv(tdir / "raw_space_subject_metrics.csv", index=False)

    # Merge raw metrics into subject matrix.
    sm2 = sm.merge(raw_df, on=["subject", "model_combo"], how="left")
    sm2.to_csv(tdir / "subject_method_matrix.csv", index=False)

    # Representative subjects.
    k = int(cfg.representative_per_bin)
    ranked = subj_rank.copy()
    top = ranked.head(k)
    mid_start = max(0, len(ranked) // 2 - k // 2)
    mid = ranked.iloc[mid_start : mid_start + k]
    bottom = ranked.tail(k)
    reps = pd.concat([top.assign(bin="top"), mid.assign(bin="mid"), bottom.assign(bin="bottom")], ignore_index=True)
    reps = reps.drop_duplicates(subset=["subject"]).head(cfg.max_subject_panels).reset_index(drop=True)
    reps.to_csv(tdir / "representative_subjects.csv", index=False)

    # Cohort figures.
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=sm2, x="model_combo", y="pearson_r", inner="quartile", cut=0)
    plt.xticks(rotation=30, ha="right")
    plt.title("Subject-level LORO Pearson (harmonized)")
    plt.tight_layout()
    plt.savefig(fdir / "method_violin_pearson_harmonized.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=sm2, x="model_combo", y="rmse", inner="quartile", cut=0)
    plt.xticks(rotation=30, ha="right")
    plt.title("Subject-level LORO RMSE (harmonized)")
    plt.tight_layout()
    plt.savefig(fdir / "method_violin_rmse_harmonized.png", dpi=220)
    plt.close()

    if len(raw_df):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=sm2, x="model_combo", y="raw_pearson_r", inner="quartile", cut=0)
        plt.xticks(rotation=30, ha="right")
        plt.title("Subject-level LORO Pearson (raw; HVG targeted recompute)")
        plt.tight_layout()
        plt.savefig(fdir / "method_violin_pearson_raw.png", dpi=220)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=sm2, x="model_combo", y="raw_rmse", inner="quartile", cut=0)
        plt.xticks(rotation=30, ha="right")
        plt.title("Subject-level LORO RMSE (raw; HVG targeted recompute)")
        plt.tight_layout()
        plt.savefig(fdir / "method_violin_rmse_raw.png", dpi=220)
        plt.close()

    # Heatmap methods vs metrics.
    mm = sm2.groupby("model_combo", as_index=False).agg(
        mean_pearson=("pearson_r", "mean"),
        mean_rmse=("rmse", "mean"),
        p95_rmse=("rmse", lambda x: float(np.percentile(x, 95))),
        mean_raw_pearson=("raw_pearson_r", "mean"),
        mean_raw_rmse=("raw_rmse", "mean"),
    )
    hm = mm.set_index("model_combo")
    z = (hm - hm.mean()) / (hm.std(ddof=0) + 1e-8)
    plt.figure(figsize=(10, 5))
    sns.heatmap(z.T, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Method-level metric profile (z-scored)")
    plt.tight_layout()
    plt.savefig(fdir / "subject_rank_heatmap_methods_vs_metrics.png", dpi=220)
    plt.close()

    # Coverage scatter.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sm2, x="n_obs_parcels", y="pearson_r", hue="model_combo", s=20, alpha=0.6)
    plt.title("Coverage vs performance by method (harmonized Pearson)")
    plt.tight_layout()
    plt.savefig(fdir / "coverage_vs_performance_by_method_harmonized.png", dpi=220)
    plt.close()

    if len(raw_df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sm2, x="n_obs_parcels", y="raw_pearson_r", hue="model_combo", s=20, alpha=0.6)
        plt.title("Coverage vs performance by method (raw Pearson)")
        plt.tight_layout()
        plt.savefig(fdir / "coverage_vs_performance_by_method_raw.png", dpi=220)
        plt.close()

    # Winner bar.
    wb = win["winner_model_combo"].value_counts().reset_index()
    wb.columns = ["model_combo", "n_subject_wins"]
    wb["win_fraction"] = wb["n_subject_wins"] / wb["n_subject_wins"].sum()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=wb, x="model_combo", y="win_fraction")
    plt.xticks(rotation=30, ha="right")
    plt.title("Per-subject winner frequency")
    plt.tight_layout()
    plt.savefig(fdir / "per_subject_winner_bar.png", dpi=220)
    plt.close()

    # Tail risk.
    tr = robustness[["model_combo", "p90_rmse", "p95_rmse"]].melt(id_vars="model_combo", var_name="tail", value_name="rmse")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=tr, x="model_combo", y="rmse", hue="tail")
    plt.xticks(rotation=30, ha="right")
    plt.title("Tail risk by method (harmonized RMSE)")
    plt.tight_layout()
    plt.savefig(fdir / "method_tail_risk_p90_p95_rmse.png", dpi=220)
    plt.close()

    # Pairwise delta matrix.
    mlist = combos
    mat = np.full((len(mlist), len(mlist)), np.nan, dtype=np.float64)
    pvt_rmse = sm2.pivot_table(index="subject", columns="model_combo", values="rmse", aggfunc="first")
    for i, a in enumerate(mlist):
        for j, b in enumerate(mlist):
            if i == j:
                mat[i, j] = 0.0
            else:
                d = pvt_rmse[a] - pvt_rmse[b]
                mat[i, j] = float(np.nanmedian(d.to_numpy()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=mlist, yticklabels=mlist, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.title("Pairwise median RMSE deltas (row - col)")
    plt.tight_layout()
    plt.savefig(fdir / "method_pairwise_delta_matrix.png", dpi=220)
    plt.close()

    # Prepare data for representative visual suites (HVG level predictions).
    header = io_utils.load_gene_header_and_hvg((root / cfg.csv_path).resolve(), (root / cfg.hvg_path).resolve())
    genes = header["genes_hvg"]
    df_hvg = io_utils.read_expression_subset((root / cfg.csv_path).resolve(), genes)
    ahba_raw = df_hvg[df_hvg["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df_hvg[df_hvg["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
    ahba_raw_full, _ = build_region_matrix(ahba_raw, genes, target_meta, agg="mean")
    obs_global = np.zeros(len(target_meta), dtype=bool)
    obs_global[np.unique(gtex_raw["parcel_idx"].astype(np.int32))] = True

    method_preds_raw: Dict[Tuple[str, str], np.ndarray] = {}
    method_preds_h: Dict[Tuple[str, str], np.ndarray] = {}
    ahba_h_by_combo: Dict[str, np.ndarray] = {}

    for combo in combos:
        hm, bm, st, spm = _parse_combo(combo)
        harmonizer = _fit_harmonizer_for_strategy(cfg, hm, st, ahba_raw, gtex_raw, genes)
        ahba_h = harmonizer.transform(ahba_raw, "AHBA")
        gtex_h = harmonizer.transform(gtex_raw, "GTEX")
        ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")
        ahba_h_by_combo[combo] = ahba_h_full
        ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
        ahba_ref_T = ahba_pls["T"]
        for sid in reps["subject"].tolist():
            subj_h = gtex_h[gtex_h["subject"] == sid].copy()
            subj_r = gtex_raw[gtex_raw["subject"] == sid].copy()
            obs_idx, X_obs_h, X_obs_raw = build_subject_observed_matrices(subj_h, subj_r, genes)
            if len(obs_idx) < 3:
                continue
            sb = {
                "subject": sid,
                "obs_idx": obs_idx,
                "X_obs_h": X_obs_h,
                "X_obs_raw": X_obs_raw,
                "coords_full": coords_full,
                "target_meta": target_meta,
            }
            ab = {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_ref_T}
            mb = {
                "harmonizer": harmonizer,
                "basis_model": bm,
                "strategy": st,
                "spatial_method": spm,
                "n_comp_target": cfg.n_comp_target,
                "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
                "rbf_smoothing": cfg.rbf_smoothing,
                "gp_rbf_length": cfg.gp_rbf_length,
                "seed": cfg.seed,
                "c_min": cfg.c_min,
                "distance_d0": cfg.distance_d0,
                "distance_tau": cfg.distance_tau,
                "uncertainty_shrink": bool(st == "gp_uncertainty"),
            }
            pred, _ = run_subject(sb, ab, mb, asdict(cfg))
            method_preds_raw[(sid, combo)] = pred["X_full_raw"]
            method_preds_h[(sid, combo)] = pred["X_full_h"]

    # Representative figures.
    for row in reps.itertuples(index=False):
        sid = str(row.subject)
        sdir = fdir / "subjects" / sid
        sdir.mkdir(parents=True, exist_ok=True)

        # sparse observed matrices
        subj = gtex_raw[gtex_raw["subject"] == sid].copy()
        sp_raw = np.full_like(ahba_raw_full, np.nan)
        g = subj.groupby("parcel_idx")[genes].mean()
        for pidx in g.index.tolist():
            sp_raw[int(pidx), :] = g.loc[pidx, genes].to_numpy(dtype=np.float64)

        # heatmap raw and harmonized, 2x4
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        mats = [ahba_raw_full, sp_raw] + [method_preds_raw[(sid, c)] for c in combos if (sid, c) in method_preds_raw][:6]
        titles = ["AHBA raw", f"{sid} sparse GTEx raw"] + [f"{c}" for c in combos]
        vals = np.concatenate([m[np.isfinite(m)] for m in mats if np.any(np.isfinite(m))])
        vmin, vmax = np.percentile(vals, [2, 98]) if vals.size else (-1, 1)
        for i, ax in enumerate(axes.reshape(-1)):
            if i < len(mats):
                im = ax.imshow(np.ma.masked_invalid(mats[i]), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(titles[i], fontsize=8)
            else:
                ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        fig.savefig(sdir / "heatmap_methods_raw.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        mats_h = [ahba_h_by_combo[combos[0]], np.where(np.isfinite(sp_raw), ahba_h_by_combo[combos[0]], np.nan)] + [
            method_preds_h[(sid, c)] for c in combos if (sid, c) in method_preds_h
        ][:6]
        vals = np.concatenate([m[np.isfinite(m)] for m in mats_h if np.any(np.isfinite(m))])
        vmin, vmax = np.percentile(vals, [2, 98]) if vals.size else (-1, 1)
        for i, ax in enumerate(axes.reshape(-1)):
            if i < len(mats_h):
                im = ax.imshow(np.ma.masked_invalid(mats_h[i]), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(titles[i] if i < len(titles) else "", fontsize=8)
            else:
                ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        fig.savefig(sdir / "heatmap_methods_harmonized.png", dpi=220)
        plt.close(fig)

        # scatter plots (allen vs predicted) raw/h
        for mode, pred_store, outn in [
            ("raw", method_preds_raw, "scatter_allen_vs_gtex_methods_raw.png"),
            ("harm", method_preds_h, "scatter_allen_vs_gtex_methods_harmonized.png"),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
            for i, c in enumerate(combos):
                ax = axes.reshape(-1)[i]
                if (sid, c) not in pred_store:
                    ax.axis("off")
                    continue
                pred = pred_store[(sid, c)]
                ref = ahba_raw_full if mode == "raw" else ahba_h_by_combo[c]
                x = ref.reshape(-1)
                y = pred.reshape(-1)
                m = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[m], y[m], s=1.2, alpha=0.22)
                ax.set_title(c, fontsize=8)
                ax.set_xlabel("Allen")
                ax.set_ylabel("GTEx pred")
            for j in range(i + 1, 6):
                axes.reshape(-1)[j].axis("off")
            fig.savefig(sdir / outn, dpi=220)
            plt.close(fig)

        # fold errors by method for this subject.
        fsub = folds[folds["subject"] == sid].copy()
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=fsub, x="model_combo", y="rmse")
        plt.xticks(rotation=30, ha="right")
        plt.title(f"{sid}: fold RMSE by method (harmonized)")
        plt.tight_layout()
        plt.savefig(sdir / "loro_fold_errors_by_method.png", dpi=220)
        plt.close()

        # summary card
        ss = sm2[sm2["subject"] == sid].sort_values(["pearson_r", "rmse"], ascending=[False, True]).copy()
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.axis("off")
        txt = [f"Subject: {sid}", f"n_obs_parcels: {int(ss['n_obs_parcels'].iloc[0]) if len(ss) else 'NA'}", "", "Method metrics (harmonized / raw):"]
        for r in ss.itertuples(index=False):
            txt.append(
                f"- {r.model_combo}: r_h={r.pearson_r:.3f}, rmse_h={r.rmse:.3f}, "
                f"r_raw={getattr(r,'raw_pearson_r',np.nan):.3f}, rmse_raw={getattr(r,'raw_rmse',np.nan):.3f}"
            )
        ax.text(0.01, 0.99, "\n".join(txt), va="top", ha="left", fontsize=10, family="monospace")
        fig.savefig(sdir / "summary_card.png", dpi=220)
        plt.close(fig)

    # Insights.
    insights = []
    winner_counts = win["winner_model_combo"].value_counts(normalize=True)
    top_winner = winner_counts.index[0]
    insights.append(
        {
            "insight": "dominant_winner_fraction",
            "value": float(winner_counts.iloc[0]),
            "detail": f"{top_winner} wins {winner_counts.iloc[0]*100:.1f}% of subjects",
        }
    )
    low = sm2[sm2["n_obs_parcels"] < cfg.coverage_threshold].groupby("model_combo")["pearson_r"].mean().sort_values(ascending=False)
    if len(low):
        insights.append(
            {
                "insight": "best_low_coverage_pearson",
                "value": float(low.iloc[0]),
                "detail": f"{low.index[0]} has highest mean Pearson in n_obs<{cfg.coverage_threshold}",
            }
        )
    if len(raw_df):
        merge_m = sm2.groupby("model_combo", as_index=False).agg(h=("pearson_r", "mean"), r=("raw_pearson_r", "mean"))
        merge_m["delta"] = merge_m["h"] - merge_m["r"]
        r = merge_m.sort_values("delta", ascending=False).iloc[0]
        insights.append(
            {
                "insight": "harmonized_raw_divergence",
                "value": float(r["delta"]),
                "detail": f"{r['model_combo']} has largest harmonized-vs-raw Pearson gap ({r['delta']:.3f})",
            }
        )
    pd.DataFrame(insights).to_csv(tdir / "insight_summary.csv", index=False)

    # Report text and summary.
    summary = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "n_methods": int(len(combos)),
        "n_subjects": int(sm2["subject"].nunique()),
        "methods": combos,
        "global_recommended_method": best_combo,
        "winner_subject_fraction": float(winner_counts.iloc[0]) if len(winner_counts) else np.nan,
        "raw_space_metrics_available": bool(len(raw_df) > 0),
        "representative_subjects": reps["subject"].tolist(),
    }
    io_utils.dump_json(out_root / "summary.json", summary)

    report_lines = [
        "# Single-Subject Top-6 Overnight Benchmark Report",
        "",
        f"- Timestamp (UTC): {summary['timestamp_utc']}",
        f"- Subjects: {summary['n_subjects']}",
        f"- Methods: {summary['n_methods']}",
        f"- Recommended winner: `{best_combo}`",
        "",
        "## Winner frequencies",
        "",
    ]
    for c, frac in winner_counts.items():
        report_lines.append(f"- `{c}`: {frac*100:.1f}% subject wins")
    report_lines += [
        "",
        "## Representative subjects",
        "",
        ", ".join(reps["subject"].astype(str).tolist()),
        "",
        "## Key figures",
        "",
        f"![Harmonized Pearson violin]({(fdir / 'method_violin_pearson_harmonized.png').resolve()})",
        f"![Harmonized RMSE violin]({(fdir / 'method_violin_rmse_harmonized.png').resolve()})",
        f"![Coverage vs harmonized performance]({(fdir / 'coverage_vs_performance_by_method_harmonized.png').resolve()})",
        f"![Winner bar]({(fdir / 'per_subject_winner_bar.png').resolve()})",
        f"![Pairwise delta matrix]({(fdir / 'method_pairwise_delta_matrix.png').resolve()})",
        "",
        "## Notes",
        "",
        "- Raw-space diagnostics were computed via targeted HVG recomputation for the top-6 methods.",
        "- Harmonized-space all-gene LORO remains the primary optimization/evaluation space.",
    ]
    (out_root / "report.md").write_text("\n".join(report_lines) + "\n")

    # Render PDF if available.
    try:
        _render_report(out_root / "report.md", out_root / "report.pdf")
    except Exception:
        pass

    manifest_inputs = [
        run_root / "tables" / "subject_loro_summary_allgenes.csv",
        run_root / "tables" / "subject_loro_folds_allgenes.csv",
        run_root / "tables" / "model_grid_summary_allgenes.csv",
    ]
    manifest = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "config": asdict(cfg),
        "checksums": {str(p): _sha256(p) for p in manifest_inputs if p.exists()},
        "outputs": {
            "summary": str(out_root / "summary.json"),
            "report_md": str(out_root / "report.md"),
            "report_pdf": str(out_root / "report.pdf"),
            "tables": str(tdir),
            "figures": str(fdir),
        },
    }
    io_utils.dump_json(out_root / "report_manifest.json", manifest)
    print(f"Completed subject-level top6 analysis: {out_root}")


if __name__ == "__main__":
    main()
