#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.diagnostics import compute_uncertainty_calibration, plot_uncertainty_reliability, plot_uvar_vs_error
from src.eval.metrics import aggregate_subject_metrics, metrics_from_vectors
from src.harmonize import fit_harmonizer
from src.models.unified_generative import UnifiedGenerativeConfig, fit_global_atlas_unified, infer_subject_unified
from src.spatial.model_coords import model_spatial_coords
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_eligibility,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)


@dataclass
class Config:
    csv_path: str = "/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv"
    winner_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_unified"
    out_root: str = "/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_phase2"
    seed: int = 123
    min_observed_parcels: int = 5
    chunk_size: int = 500
    smoke_subjects: int = 0
    strict_fold_harmonizer: bool = True
    combat_use_covariates: bool = True


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="All-gene LORO validation for unified winner")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--winner-root", default=Config.winner_root)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    p.add_argument("--smoke-subjects", type=int, default=Config.smoke_subjects)
    p.add_argument("--strict-fold-harmonizer", default=str(Config.strict_fold_harmonizer).lower())
    p.add_argument("--combat-use-covariates", default=str(Config.combat_use_covariates).lower())
    a = p.parse_args()
    d = vars(a)
    d["strict_fold_harmonizer"] = _parse_bool(d["strict_fold_harmonizer"])
    d["combat_use_covariates"] = _parse_bool(d["combat_use_covariates"])
    return Config(**d)


def _read_all_genes(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
    return h[6:]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _winner_config(winner_root: Path) -> tuple[str, dict]:
    summary = json.loads((winner_root / "summary.json").read_text())
    winner = str(summary["winner_model"])
    cfg = {"harm": "combat", "cal": "hier_affine_map", "robust": "student_t", "hetero": "gene_var", "uncshrink": "false"}
    for part in winner.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            cfg[k] = v
    return winner, cfg


def _chunk_ranges(n: int, size: int):
    size = int(max(size, 1))
    for s in range(0, n, size):
        e = min(n, s + size)
        yield s, e


def accumulate_chunk_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray, state: dict) -> dict:
    state["true"].append(np.asarray(y_true, dtype=np.float64))
    state["pred"].append(np.asarray(y_pred, dtype=np.float64))
    state["base"].append(np.asarray(y_base, dtype=np.float64))
    return state


def finalize_chunk_metrics(state: dict) -> dict:
    yt = np.concatenate(state["true"]) if len(state["true"]) else np.zeros(0, dtype=np.float64)
    yp = np.concatenate(state["pred"]) if len(state["pred"]) else np.zeros(0, dtype=np.float64)
    yb = np.concatenate(state["base"]) if len(state["base"]) else np.zeros(0, dtype=np.float64)
    met = metrics_from_vectors(yt, yp)
    b_rmse = float(np.sqrt(np.mean((yt - yb) ** 2))) if yt.size else np.nan
    met["baseline_rmse"] = b_rmse
    met["hold_abs_error_mean"] = float(np.mean(np.abs(yt - yp))) if yt.size else np.nan
    return met


def run_allgenes_loro(subjects: list[str], genes: list[str], data_bundle: dict, cfg: Config, winner_cfg: dict):
    ahba_raw = data_bundle["ahba_raw"]
    gtex_raw = data_bundle["gtex_raw"]
    target_meta = data_bundle["target_meta"]
    coords_full = data_bundle["coords_full"]
    coords_model_full = model_spatial_coords(coords_full, fold_hemispheres=True)

    fold_rows = []
    summary_rows = []

    # Optional global prefit (fast mode).
    global_harm = None
    global_ahba_h_full = None
    global_atlas = None
    global_gtex_h = None
    if not cfg.strict_fold_harmonizer:
        hcfg = SimpleNamespace(combat_use_covariates=cfg.combat_use_covariates, whiten_eps=1e-4, hier_lambda_a=10.0, hier_lambda_b=10.0, hier_base_method="robustz_affine", hier_subject_subset=subjects)
        global_harm = fit_harmonizer(ahba_raw, gtex_raw, genes, method=str(winner_cfg.get("harm", "combat")), cfg=hcfg)
        ahba_h = global_harm.transform(ahba_raw, "AHBA")
        global_gtex_h = global_harm.transform(gtex_raw, "GTEX")
        global_ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")
        ucfg = UnifiedGenerativeConfig(
            latent_dim=3,
            max_iters=5,
            lambda_w=1.0,
            lambda_z=1.0,
            lambda_cal_a=10.0,
            lambda_cal_b=10.0,
            gp_length_scale=25.0,
            gp_noise=1e-3,
            gp_optimize=True,
            gp_n_restarts=0,
            robust_loss=str(winner_cfg.get("robust", "student_t")),
            heteroscedastic=str(winner_cfg.get("hetero", "none")).lower() != "none",
            calibration_mode=str(winner_cfg.get("cal", "hier_affine_map")),
            uncertainty_shrink=str(winner_cfg.get("uncshrink", "false")).lower() == "true",
            unc_alpha=0.5,
            unc_beta=0.5,
            unc_m0=0.5,
            unc_tau=0.2,
            random_state=cfg.seed,
        )
        global_atlas = fit_global_atlas_unified(global_ahba_h_full, coords_model_full, ucfg)

    for sid in subjects:
        sub_all = gtex_raw[gtex_raw["subject"] == sid].copy()
        obs_idx_all = sorted(set(int(x) for x in sub_all["parcel_idx"].tolist()))
        if len(obs_idx_all) < cfg.min_observed_parcels:
            continue

        subj_true_list = []
        subj_pred_list = []
        subj_base_list = []
        n_skipped = 0
        n_ok = 0

        for fold_id, hold in enumerate(obs_idx_all):
            train_mask = ~((gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold)))
            gtex_train = gtex_raw[train_mask].copy()
            if cfg.strict_fold_harmonizer:
                hcfg = SimpleNamespace(
                    combat_use_covariates=cfg.combat_use_covariates,
                    whiten_eps=1e-4,
                    hier_lambda_a=10.0,
                    hier_lambda_b=10.0,
                    hier_base_method="robustz_affine",
                    hier_subject_subset=[sid],
                )
                harm = fit_harmonizer(ahba_raw, gtex_train, genes, method=str(winner_cfg.get("harm", "combat")), cfg=hcfg)
                ahba_h = harm.transform(ahba_raw, "AHBA")
                gtex_h = harm.transform(gtex_train, "GTEX")
                ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")
                ucfg = UnifiedGenerativeConfig(
                    latent_dim=3,
                    max_iters=5,
                    lambda_w=1.0,
                    lambda_z=1.0,
                    lambda_cal_a=10.0,
                    lambda_cal_b=10.0,
                    gp_length_scale=25.0,
                    gp_noise=1e-3,
                    gp_optimize=True,
                    gp_n_restarts=0,
                    robust_loss=str(winner_cfg.get("robust", "student_t")),
                    heteroscedastic=str(winner_cfg.get("hetero", "none")).lower() != "none",
                    calibration_mode=str(winner_cfg.get("cal", "hier_affine_map")),
                    uncertainty_shrink=str(winner_cfg.get("uncshrink", "false")).lower() == "true",
                    unc_alpha=0.5,
                    unc_beta=0.5,
                    unc_m0=0.5,
                    unc_tau=0.2,
                    random_state=cfg.seed,
                )
                atlas = fit_global_atlas_unified(ahba_h_full, coords_model_full, ucfg)
            else:
                harm = global_harm
                gtex_h = global_gtex_h
                ahba_h_full = global_ahba_h_full
                atlas = global_atlas
                ucfg = UnifiedGenerativeConfig(
                    latent_dim=3,
                    max_iters=5,
                    lambda_w=1.0,
                    lambda_z=1.0,
                    lambda_cal_a=10.0,
                    lambda_cal_b=10.0,
                    gp_length_scale=25.0,
                    gp_noise=1e-3,
                    gp_optimize=True,
                    gp_n_restarts=0,
                    robust_loss=str(winner_cfg.get("robust", "student_t")),
                    heteroscedastic=str(winner_cfg.get("hetero", "none")).lower() != "none",
                    calibration_mode=str(winner_cfg.get("cal", "hier_affine_map")),
                    uncertainty_shrink=str(winner_cfg.get("uncshrink", "false")).lower() == "true",
                    unc_alpha=0.5,
                    unc_beta=0.5,
                    unc_m0=0.5,
                    unc_tau=0.2,
                    random_state=cfg.seed,
                )

            if cfg.strict_fold_harmonizer:
                subj_h = gtex_h[gtex_h["subject"] == sid].copy()
            else:
                subj_h = gtex_h[(gtex_h["subject"] == sid) & (gtex_h["parcel_idx"] != int(hold))].copy()
            subj_raw_train = gtex_train[gtex_train["subject"] == sid].copy()
            obs_idx_fold, xh_fold, _ = build_subject_observed_matrices(subj_h, subj_raw_train, genes)

            leak_flag = bool(int(hold) in set(obs_idx_fold.tolist()))
            if leak_flag or len(obs_idx_fold) < 2:
                n_skipped += 1
                fold_rows.append(
                    {
                        "subject": sid,
                        "fold_id": int(fold_id),
                        "held_out_parcel": int(hold),
                        "n_train_obs": int(len(obs_idx_fold)),
                        "pearson_r": np.nan,
                        "spearman_rho": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "medae": np.nan,
                        "baseline_rmse": np.nan,
                        "hold_pred_std": np.nan,
                        "hold_abs_error_mean": np.nan,
                        "status": "failed_leakage" if leak_flag else "skipped_too_few_train",
                        "config_hash": io_utils.hash_config(asdict(cfg)),
                        "model_combo": "unified_winner_allgenes",
                    }
                )
                continue

            try:
                res = infer_subject_unified(
                    {"subject": sid, "obs_idx": obs_idx_fold, "X_obs_h": xh_fold},
                    atlas,
                    ucfg,
                    fold_ctx={"prior_h": ahba_h_full, "obs_idx": obs_idx_fold},
                )
                pred = np.asarray(res["x_hat_h_full"][int(hold), :], dtype=np.float64)
                pred_std = float(np.sqrt(np.maximum(np.mean(res["uvar_full"][int(hold), :]), 0.0)))

                hold_rows_raw = gtex_raw[(gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold))].copy()
                if len(hold_rows_raw) == 0:
                    raise RuntimeError("missing held-out rows")
                hold_h = harm.transform(hold_rows_raw, "GTEX")
                truth = hold_h[genes].to_numpy(dtype=np.float64).mean(axis=0)
                base = np.asarray(ahba_h_full[int(hold), :], dtype=np.float64)

                state = {"true": [], "pred": [], "base": []}
                for s, e in _chunk_ranges(len(genes), cfg.chunk_size):
                    accumulate_chunk_metrics(truth[s:e], pred[s:e], base[s:e], state)
                met = finalize_chunk_metrics(state)

                fold_rows.append(
                    {
                        "subject": sid,
                        "fold_id": int(fold_id),
                        "held_out_parcel": int(hold),
                        "n_train_obs": int(len(obs_idx_fold)),
                        "pearson_r": float(met["pearson_r"]),
                        "spearman_rho": float(met["spearman_rho"]),
                        "rmse": float(met["rmse"]),
                        "mae": float(met["mae"]),
                        "medae": float(met["medae"]),
                        "baseline_rmse": float(met["baseline_rmse"]),
                        "hold_pred_std": pred_std,
                        "hold_abs_error_mean": float(met["hold_abs_error_mean"]),
                        "status": "ok",
                        "config_hash": io_utils.hash_config(asdict(cfg)),
                        "model_combo": "unified_winner_allgenes",
                    }
                )
                subj_true_list.append(truth)
                subj_pred_list.append(pred)
                subj_base_list.append(base)
                n_ok += 1
            except Exception as e:
                n_skipped += 1
                fold_rows.append(
                    {
                        "subject": sid,
                        "fold_id": int(fold_id),
                        "held_out_parcel": int(hold),
                        "n_train_obs": int(len(obs_idx_fold)),
                        "pearson_r": np.nan,
                        "spearman_rho": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "medae": np.nan,
                        "baseline_rmse": np.nan,
                        "hold_pred_std": np.nan,
                        "hold_abs_error_mean": np.nan,
                        "status": f"failed:{str(e)[:120]}",
                        "config_hash": io_utils.hash_config(asdict(cfg)),
                        "model_combo": "unified_winner_allgenes",
                    }
                )

        agg = aggregate_subject_metrics(subj_true_list, subj_pred_list, subj_base_list)
        summary_rows.append(
            {
                "subject": sid,
                "n_obs_parcels": int(len(obs_idx_all)),
                "n_folds": int(n_ok),
                "n_skipped_folds": int(n_skipped),
                "n_points": int(agg["n_points"]),
                "pearson_r": float(agg["pearson_r"]),
                "spearman_rho": float(agg["spearman_rho"]),
                "rmse": float(agg["rmse"]),
                "mae": float(agg["mae"]),
                "medae": float(agg["medae"]),
                "baseline_rmse": float(agg["baseline_rmse"]),
                "better_than_baseline_rmse": bool(agg["better_than_baseline_rmse"]),
                "coverage_tier": "lt8" if len(obs_idx_all) < 8 else "ge8",
                "model_combo": "unified_winner_allgenes",
                "config_hash": io_utils.hash_config(asdict(cfg)),
            }
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def _plot_core(model_summary: pd.DataFrame, subject_summary: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(model_summary):
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        vals = model_summary.iloc[0]
        ax.bar(["mean_pearson", "mean_rmse", "mean_baseline_rmse"], [vals["mean_pearson"], vals["mean_rmse"], vals["mean_baseline_rmse"]], color=["#2a9d8f", "#e76f51", "#6c757d"])
        ax.set_title("All-gene winner summary")
        _p = fig_dir / "allgenes_model_ranking.png"
        fig.savefig(_p, dpi=220)
        plt.close(fig)

    if len(subject_summary):
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        sns.scatterplot(data=subject_summary, x="n_obs_parcels", y="pearson_r", hue="coverage_tier", s=16, alpha=0.55, ax=ax)
        ax.set_title("Coverage vs performance (all-gene LORO)")
        _p = fig_dir / "allgenes_coverage_vs_performance.png"
        fig.savefig(_p, dpi=220)
        plt.close(fig)


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)

    csv_path = Path(cfg.csv_path).resolve()
    winner_root = Path(cfg.winner_root).resolve()
    out_root = Path(cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    cfg_dir = out_root / "configs"
    man_dir = out_root / "manifests"
    for p in [table_dir, fig_dir, cfg_dir, man_dir]:
        p.mkdir(parents=True, exist_ok=True)

    winner_model, winner_cfg = _winner_config(winner_root)
    genes_all = _read_all_genes(csv_path)

    io_utils.dump_yaml(cfg_dir / "phase2_config.yaml", asdict(cfg))
    io_utils.dump_json(
        man_dir / "source_manifest.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "source_csv": str(csv_path),
            "source_sha256": _sha256(csv_path),
            "winner_root": str(winner_root),
            "winner_model": winner_model,
            "winner_config": winner_cfg,
            "n_genes_all": int(len(genes_all)),
        },
    )

    df = io_utils.read_expression_subset(csv_path, genes_all)
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

    elig = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    subjects = elig[elig["eligible"]]["subject"].astype(str).tolist()
    if cfg.smoke_subjects > 0:
        subjects = subjects[: int(cfg.smoke_subjects)]

    folds_df, subj_df = run_allgenes_loro(
        subjects,
        genes_all,
        {
            "ahba_raw": ahba_raw,
            "gtex_raw": gtex_raw,
            "target_meta": target_meta,
            "coords_full": coords_full,
        },
        cfg,
        winner_cfg,
    )

    folds_df.to_csv(table_dir / "allgenes_subject_loro_folds.csv", index=False)
    subj_df.to_csv(table_dir / "allgenes_subject_loro_summary.csv", index=False)

    if len(subj_df):
        model_summary = pd.DataFrame(
            [
                {
                    "model_combo": "unified_winner_allgenes",
                    "n_subjects": int(subj_df["subject"].nunique()),
                    "mean_pearson": float(subj_df["pearson_r"].mean()),
                    "median_pearson": float(subj_df["pearson_r"].median()),
                    "mean_rmse": float(subj_df["rmse"].mean()),
                    "mean_baseline_rmse": float(subj_df["baseline_rmse"].mean()),
                    "frac_better_baseline_rmse": float(subj_df["better_than_baseline_rmse"].mean()),
                }
            ]
        )
        model_summary["gate_pass"] = (model_summary["mean_pearson"] >= 0.50) & (model_summary["mean_rmse"] < model_summary["mean_baseline_rmse"])
    else:
        model_summary = pd.DataFrame(
            [
                {
                    "model_combo": "unified_winner_allgenes",
                    "n_subjects": 0,
                    "mean_pearson": np.nan,
                    "median_pearson": np.nan,
                    "mean_rmse": np.nan,
                    "mean_baseline_rmse": np.nan,
                    "frac_better_baseline_rmse": np.nan,
                    "gate_pass": False,
                }
            ]
        )
    model_summary.to_csv(table_dir / "allgenes_model_summary.csv", index=False)

    unc = compute_uncertainty_calibration(folds_df[folds_df["status"] == "ok"].copy())
    unc.to_csv(table_dir / "allgenes_uncertainty_calibration.csv", index=False)

    # coverage stratified uncertainty/error corr
    rows = []
    ok = folds_df[folds_df["status"] == "ok"].copy()
    if len(ok):
        subj_cov = subj_df[["subject", "n_obs_parcels"]].copy()
        ok = ok.merge(subj_cov, on="subject", how="left")
        ok["coverage_bin"] = np.where(ok["n_obs_parcels"] < 8, "lt8", "ge8")
        for cb, d in ok.groupby("coverage_bin"):
            x = d["hold_pred_std"].to_numpy(dtype=np.float64)
            y = d["hold_abs_error_mean"].to_numpy(dtype=np.float64)
            corr = float(stats.pearsonr(x, y).statistic) if len(d) > 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan
            rows.append({"coverage_bin": cb, "corr_uvar_error": corr, "mean_abs_err": float(np.mean(y)), "n_folds": int(len(d))})
    pd.DataFrame(rows).to_csv(table_dir / "allgenes_uncertainty_by_coverage.csv", index=False)

    _plot_core(model_summary, subj_df, fig_dir)
    plot_uncertainty_reliability(unc, fig_dir / "allgenes_uncertainty_reliability.png")
    _ = plot_uvar_vs_error(
        ok.get("hold_pred_std", pd.Series(dtype=float)).to_numpy(dtype=np.float64) if len(ok) else np.zeros(0),
        ok.get("hold_abs_error_mean", pd.Series(dtype=float)).to_numpy(dtype=np.float64) if len(ok) else np.zeros(0),
        fig_dir / "allgenes_uvar_vs_error.png",
    )

    io_utils.dump_json(
        out_root / "summary_phase2.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "winner_model": winner_model,
            "n_subjects": int(subj_df["subject"].nunique()) if len(subj_df) else 0,
            "n_folds_ok": int((folds_df["status"] == "ok").sum()) if len(folds_df) else 0,
            "allgenes_model_summary": model_summary.iloc[0].to_dict(),
            "strict_fold_harmonizer": bool(cfg.strict_fold_harmonizer),
            "smoke_subjects": int(cfg.smoke_subjects),
        },
    )

    print(f"Completed all-gene LORO phase2: {out_root}")


if __name__ == "__main__":
    main()
