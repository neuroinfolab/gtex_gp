#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.diagnostics import (
    calibration_gene_stats,
    plot_calibration_density,
    plot_misfit_vs_distance,
    plot_uncertainty_reliability,
    plot_uvar_vs_error,
    summarize_distance_effect,
    uncertainty_calibration,
)
from src.eval.loro import run_loro
from src.harmonize import fit_harmonizer
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_eligibility,
    build_subject_observed_matrices,
    build_target_parcels,
    compute_global_distance_table,
    map_gtex_to_target,
)


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/vnext_alignment"
    min_observed_parcels: int = 5
    c_min: int = 8
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_rbf_length: float = 25.0
    seed: int = 123
    n_jobs: int = 1
    harmonization_models: str = "zscore_affine,robustz_affine,whiten_zca_affine,combat,hier_affine"
    basis_models: str = "o3,affine_gl3,o3_diagscale"
    mitigation_strategies: str = "baseline,distance_shrink,constrained_anchor,piecewise_harmonization,gp_uncertainty,moe_basis"
    spatial_methods: str = "rbf,gp"
    smoke_subjects: int = 0
    parity_tolerance: float = 0.01
    whiten_eps: float = 1e-4
    min_group_samples: int = 200
    distance_d0: float = 45.0
    distance_tau: float = 10.0
    unc_alpha: float = 0.5
    unc_beta: float = 0.5
    unc_m0: float = 0.5
    unc_tau: float = 0.15
    combat_use_covariates: bool = True
    gtex_rep_mode: str = "medoid"
    gtex_hemi_mode: str = "native"
    hier_lambda_a: float = 10.0
    hier_lambda_b: float = 10.0
    hier_base_method: str = "robustz_affine"


def _parse_csv(v: str) -> List[str]:
    return [x.strip() for x in str(v).split(",") if x.strip()]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run vNext baseline + calibration + uncertainty benchmark (Milestones A-D).")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--gp-rbf-length", type=float, default=Config.gp_rbf_length)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    p.add_argument("--harmonization-models", default=Config.harmonization_models)
    p.add_argument("--basis-models", default=Config.basis_models)
    p.add_argument("--mitigation-strategies", default=Config.mitigation_strategies)
    p.add_argument("--spatial-methods", default=Config.spatial_methods)
    p.add_argument("--smoke-subjects", type=int, default=Config.smoke_subjects)
    p.add_argument("--parity-tolerance", type=float, default=Config.parity_tolerance)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--min-group-samples", type=int, default=Config.min_group_samples)
    p.add_argument("--distance-d0", type=float, default=Config.distance_d0)
    p.add_argument("--distance-tau", type=float, default=Config.distance_tau)
    p.add_argument("--unc-alpha", type=float, default=Config.unc_alpha)
    p.add_argument("--unc-beta", type=float, default=Config.unc_beta)
    p.add_argument("--unc-m0", type=float, default=Config.unc_m0)
    p.add_argument("--unc-tau", type=float, default=Config.unc_tau)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.combat_use_covariates)
    p.add_argument("--gtex-rep-mode", choices=["centroid", "medoid"], default=Config.gtex_rep_mode)
    p.add_argument("--gtex-hemi-mode", choices=["native", "mirror_left"], default=Config.gtex_hemi_mode)
    p.add_argument("--hier-lambda-a", type=float, default=Config.hier_lambda_a)
    p.add_argument("--hier-lambda-b", type=float, default=Config.hier_lambda_b)
    p.add_argument("--hier-base-method", default=Config.hier_base_method)
    a = p.parse_args()
    return Config(**vars(a))


def _combo_id(hm: str, bm: str, st: str, sm: str) -> str:
    return f"{hm}__{bm}__{st}__{sm}"


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _fit_harmonizer_for_strategy(
    cfg: Config,
    hm: str,
    strategy: str,
    ahba_train: pd.DataFrame,
    gtex_train: pd.DataFrame,
    genes_hvg: List[str],
    subject_subset: List[str] | None = None,
):
    method = "piecewise_robustz" if strategy == "piecewise_harmonization" else hm
    cfg_use = cfg
    if method == "hier_affine" and subject_subset is not None:
        cfg_use = SimpleNamespace(**asdict(cfg), hier_subject_subset=[str(s) for s in subject_subset])
    return fit_harmonizer(ahba_train, gtex_train, genes_hvg, method=method, cfg=cfg_use)


def _make_parity_report(
    out_root: Path,
    summary_df: pd.DataFrame,
    tolerance: float,
) -> None:
    parity_dir = out_root / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    legacy_grid = out_root.parent / "o3_subject_model_grid_whitening" / "tables" / "model_grid_summary_hvg.csv"
    legacy_mit = out_root.parent / "o3_subject_misfit_mitigation" / "tables" / "strategy_ranking.csv"

    if legacy_grid.exists():
        lg = pd.read_csv(legacy_grid)
        sub = summary_df[(summary_df["strategy"] == "baseline") & (summary_df["spatial_method"] == "rbf")].copy()
        for _, r in sub.iterrows():
            key = f"{r['harmonization']}__{r['basis_model']}"
            m = lg[lg["model_combo"] == key]
            if len(m) == 0:
                continue
            m0 = m.iloc[0]
            d_p = abs(float(r["mean_pearson"]) - float(m0["mean_pearson"]))
            d_r = abs(float(r["mean_rmse"]) - float(m0["mean_rmse"]))
            rows.append(
                {
                    "parity_group": "legacy_grid_baseline",
                    "model_key": key,
                    "metric": "mean_pearson",
                    "abs_diff": d_p,
                    "tolerance": tolerance,
                    "pass": bool(d_p <= tolerance),
                }
            )
            rows.append(
                {
                    "parity_group": "legacy_grid_baseline",
                    "model_key": key,
                    "metric": "mean_rmse",
                    "abs_diff": d_r,
                    "tolerance": tolerance,
                    "pass": bool(d_r <= tolerance),
                }
            )

    if legacy_mit.exists():
        lm = pd.read_csv(legacy_mit)
        sub = summary_df[(summary_df["harmonization"] == "robustz_affine") & (summary_df["basis_model"] == "affine_gl3") & (summary_df["spatial_method"] == "rbf")].copy()
        for _, r in sub.iterrows():
            st = str(r["strategy"])
            m = lm[lm["strategy"] == st]
            if len(m) == 0:
                continue
            m0 = m.iloc[0]
            d_p = abs(float(r["mean_pearson"]) - float(m0["mean_pearson"]))
            d_r = abs(float(r["mean_rmse"]) - float(m0["mean_rmse"]))
            rows.append(
                {
                    "parity_group": "legacy_mitigation",
                    "model_key": st,
                    "metric": "mean_pearson",
                    "abs_diff": d_p,
                    "tolerance": tolerance,
                    "pass": bool(d_p <= tolerance),
                }
            )
            rows.append(
                {
                    "parity_group": "legacy_mitigation",
                    "model_key": st,
                    "metric": "mean_rmse",
                    "abs_diff": d_r,
                    "tolerance": tolerance,
                    "pass": bool(d_r <= tolerance),
                }
            )

    parity_df = pd.DataFrame(rows)
    parity_df.to_csv(parity_dir / "parity_summary.csv", index=False)
    fail_df = parity_df[~parity_df["pass"]] if len(parity_df) else pd.DataFrame(columns=parity_df.columns)
    fail_df.to_csv(parity_dir / "parity_failures.csv", index=False)

    total = int(len(parity_df))
    passed = int(parity_df["pass"].sum()) if total > 0 else 0
    text = [
        "# Parity Report",
        "",
        f"- tolerance: {tolerance}",
        f"- checks: {total}",
        f"- passed: {passed}",
        f"- failed: {total - passed}",
    ]
    (parity_dir / "parity_report.md").write_text("\n".join(text) + "\n")


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    config_dir = out_root / "configs"
    log_dir = out_root / "logs"
    for d in [out_root, table_dir, fig_dir, config_dir, log_dir, out_root / "parity"]:
        d.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(cfg)
    cfg_hash = io_utils.hash_config(cfg_dict)
    io_utils.dump_yaml(config_dir / "baseline_vnext.yaml", cfg_dict)
    io_utils.dump_json(config_dir / "baseline_vnext.json", cfg_dict)

    header = io_utils.load_gene_header_and_hvg(csv_path, hvg_path)
    genes_hvg = header["genes_hvg"]
    if len(genes_hvg) == 0:
        raise RuntimeError("No overlapping HVGs found.")

    df = io_utils.read_expression_subset(
        csv_path,
        genes_hvg,
        rep_mode=str(cfg.gtex_rep_mode).lower(),
        hemi_mode=str(cfg.gtex_hemi_mode).lower(),
    )
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEX subset empty.")

    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)

    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)

    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    eligibility_df = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    if cfg.smoke_subjects > 0:
        elig = eligibility_df[eligibility_df["eligible"]].head(int(cfg.smoke_subjects))["subject"].tolist()
        eligibility_df["eligible"] = eligibility_df["subject"].isin(set(elig))
    eligibility_df.to_csv(table_dir / "subject_eligibility.csv", index=False)
    eligible_subjects = eligibility_df[eligibility_df["eligible"]]["subject"].astype(str).tolist()

    if len(eligible_subjects) == 0:
        raise RuntimeError("No eligible subjects for run.")

    _, global_obs_mask = build_region_matrix(gtex_raw, genes_hvg, target_meta, agg="mean")
    dist_base = compute_global_distance_table(target_meta, global_obs_mask)

    harmonizers = _parse_csv(cfg.harmonization_models)
    basis_models = _parse_csv(cfg.basis_models)
    strategies = _parse_csv(cfg.mitigation_strategies)
    spatial_methods = _parse_csv(cfg.spatial_methods)

    fold_rows_all = []
    summary_rows_all = []
    harm_diag_rows = []
    distance_rows = []
    uncertainty_rows = []
    gate_rows = []

    for hm in harmonizers:
        for bm in basis_models:
            for st in strategies:
                sm_list = ["gp"] if st == "gp_uncertainty" else spatial_methods
                for sm in sm_list:
                    combo = _combo_id(hm, bm, st, sm)
                    print(f"[combo] {combo}")

                    # Fit global harmonizer for full-subject predictions and diagnostics.
                    harmonizer = _fit_harmonizer_for_strategy(cfg, hm, st, ahba_raw, gtex_raw, genes_hvg)
                    harm_diag = harmonizer.diagnostics()
                    harm_diag_rows.append(
                        {
                            "model_combo": combo,
                            "harmonization": hm,
                            "basis_model": bm,
                            "strategy": st,
                            "spatial_method": sm,
                            **harm_diag,
                        }
                    )

                    ahba_h = harmonizer.transform(ahba_raw, "AHBA")
                    gtex_h = harmonizer.transform(gtex_raw, "GTEX")
                    ahba_h_full, _ = build_region_matrix(ahba_h, genes_hvg, target_meta, agg="mean")
                    ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
                    ahba_ref_T = ahba_pls["T"]

                    stack_h = []
                    stack_raw = []
                    calib_true = []
                    calib_pred = []

                    for sid in eligible_subjects:
                        subj_h = gtex_h[gtex_h["subject"] == sid].copy()
                        subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
                        obs_idx, X_obs_h, X_obs_raw = build_subject_observed_matrices(subj_h, subj_raw, genes_hvg)
                        if len(obs_idx) < cfg.min_observed_parcels:
                            continue

                        subject_bundle = {
                            "subject": sid,
                            "obs_idx": obs_idx,
                            "X_obs_h": X_obs_h,
                            "X_obs_raw": X_obs_raw,
                            "coords_full": coords_full,
                            "target_meta": target_meta,
                        }
                        atlas_bundle = {
                            "ahba_h_full": ahba_h_full,
                            "ahba_ref_T": ahba_ref_T,
                        }
                        method_bundle = {
                            "harmonizer": harmonizer,
                            "basis_model": bm,
                            "strategy": st,
                            "spatial_method": sm,
                            "n_comp_target": cfg.n_comp_target,
                            "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
                            "rbf_smoothing": cfg.rbf_smoothing,
                            "gp_rbf_length": cfg.gp_rbf_length,
                            "seed": cfg.seed,
                            "c_min": cfg.c_min,
                            "distance_d0": cfg.distance_d0,
                            "distance_tau": cfg.distance_tau,
                            "uncertainty_shrink": bool(st == "gp_uncertainty"),
                            "unc_alpha": cfg.unc_alpha,
                            "unc_beta": cfg.unc_beta,
                            "unc_m0": cfg.unc_m0,
                            "unc_tau": cfg.unc_tau,
                        }

                        full_pred, _ = run_subject(subject_bundle, atlas_bundle, method_bundle, cfg_dict)
                        stack_h.append(full_pred["X_full_h"])
                        stack_raw.append(full_pred["X_full_raw"])

                        idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}

                        def _pipeline_for_fold(hold: int, train_idx: np.ndarray):
                            train_mask = ~((gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold)))
                            gtex_train_fold = gtex_raw[train_mask].copy()
                            gtex_train_fold = add_sample_groups(gtex_train_fold, target_meta)
                            ahba_train_fold = ahba_raw.copy()

                            fold_harm = _fit_harmonizer_for_strategy(
                                cfg,
                                hm,
                                st,
                                ahba_train_fold,
                                gtex_train_fold,
                                genes_hvg,
                                subject_subset=[sid],
                            )
                            ahba_h_fold = fold_harm.transform(ahba_train_fold, "AHBA")
                            gtex_h_fold = fold_harm.transform(gtex_train_fold, "GTEX")

                            ahba_h_full_fold, _ = build_region_matrix(ahba_h_fold, genes_hvg, target_meta, agg="mean")
                            ahba_pls_fold = fit_subject_pls(ahba_h_full_fold, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
                            ahba_ref_T_fold = ahba_pls_fold["T"]

                            subj_h_fold = gtex_h_fold[gtex_h_fold["subject"] == sid].copy()
                            subj_raw_fold = gtex_train_fold[gtex_train_fold["subject"] == sid].copy()
                            obs_idx_fold, xh_fold, xr_fold = build_subject_observed_matrices(subj_h_fold, subj_raw_fold, genes_hvg)

                            leak_flag = bool(int(hold) in set(obs_idx_fold.tolist()))
                            if len(obs_idx_fold) < 3:
                                raise RuntimeError("insufficient train parcels in fold")

                            sb_fold = {
                                "subject": sid,
                                "obs_idx": obs_idx_fold,
                                "X_obs_h": xh_fold,
                                "X_obs_raw": xr_fold,
                                "coords_full": coords_full,
                                "target_meta": target_meta,
                            }
                            ab_fold = {"ahba_h_full": ahba_h_full_fold, "ahba_ref_T": ahba_ref_T_fold}
                            mb_fold = dict(method_bundle)
                            mb_fold["harmonizer"] = fold_harm

                            pred_fold, diag_fold = run_subject(sb_fold, ab_fold, mb_fold, cfg_dict)
                            hold_pred = pred_fold["X_full_h"][int(hold), :]

                            hold_pos = idx_to_pos[int(hold)]
                            calib_true.append(X_obs_h[hold_pos, :])
                            calib_pred.append(hold_pred)
                            hold_std = float(np.sqrt(max(float(diag_fold.get("uvar", np.zeros(coords_full.shape[0]))[int(hold)]), 0.0)))
                            return hold_pred, {"leak_flag": leak_flag, "hold_pred_std": hold_std}

                        eval_bundle = {
                            "subject": sid,
                            "obs_idx": obs_idx,
                            "X_obs_h": X_obs_h,
                            "baseline_h_full": ahba_h_full,
                            "config_hash": cfg_hash,
                        }
                        folds_df, summary = run_loro(eval_bundle, _pipeline_for_fold, {"config_hash": cfg_hash})
                        folds_df["model_combo"] = combo
                        folds_df["harmonization"] = hm
                        folds_df["basis_model"] = bm
                        folds_df["strategy"] = st
                        folds_df["spatial_method"] = sm
                        fold_rows_all.append(folds_df)

                        summary.update(
                            {
                                "model_combo": combo,
                                "harmonization": hm,
                                "basis_model": bm,
                                "strategy": st,
                                "spatial_method": sm,
                                "coverage_tier": "high" if int(summary["n_obs_parcels"]) >= cfg.c_min else "low",
                                "deployment_mode": "model" if int(summary["n_obs_parcels"]) >= cfg.c_min else "atlas_prior",
                            }
                        )
                        summary_rows_all.append(summary)

                    if len(stack_h) > 0:
                        mean_h = np.nanmean(np.stack(stack_h, axis=0), axis=0)
                        mean_raw = np.nanmean(np.stack(stack_raw, axis=0), axis=0)
                        gtex_h_tbl = pd.concat(
                            [
                                target_meta[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].rename(columns={"tissue_or_parcel": "parcel_name"}).reset_index(drop=True),
                                pd.DataFrame(mean_h, columns=genes_hvg),
                            ],
                            axis=1,
                        )
                        gtex_raw_tbl = pd.concat(
                            [
                                target_meta[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].rename(columns={"tissue_or_parcel": "parcel_name"}).reset_index(drop=True),
                                pd.DataFrame(mean_raw, columns=genes_hvg),
                            ],
                            axis=1,
                        )
                        gtex_h_tbl.to_csv(table_dir / f"aggregate_hvg_{combo}_mean_harmonized.csv", index=False)
                        gtex_raw_tbl.to_csv(table_dir / f"aggregate_hvg_{combo}_mean_raw.csv", index=False)

                        region_corr = np.zeros(mean_h.shape[0], dtype=np.float64)
                        for r in range(mean_h.shape[0]):
                            if np.std(mean_h[r, :]) > 1e-12 and np.std(ahba_h_full[r, :]) > 1e-12:
                                region_corr[r] = np.corrcoef(mean_h[r, :], ahba_h_full[r, :])[0, 1]
                            else:
                                region_corr[r] = np.nan
                        ddf = dist_base.copy()
                        ddf["model_combo"] = combo
                        ddf["harmonization"] = hm
                        ddf["basis_model"] = bm
                        ddf["strategy"] = st
                        ddf["spatial_method"] = sm
                        ddf["region_pearson_h"] = region_corr
                        ddf["misfit_1mpearson"] = 1.0 - region_corr
                        distance_rows.append(ddf)

                    if len(calib_true) > 0:
                        y_true = np.vstack(calib_true)
                        y_pred = np.vstack(calib_pred)
                        cdf = calibration_gene_stats(y_true, y_pred, genes_hvg)
                        cdf["model_combo"] = combo
                        cdf.to_csv(table_dir / f"calibration_gene_stats_{_sanitize(combo)}.csv", index=False)
                        plot_calibration_density(cdf, fig_dir / f"calibration_slope_intercept_density_{_sanitize(combo)}.png", combo)

    folds_all = pd.concat(fold_rows_all, ignore_index=True) if len(fold_rows_all) else pd.DataFrame()
    summary_all = pd.DataFrame(summary_rows_all)
    harm_diag_df = pd.DataFrame(harm_diag_rows)
    dist_all = pd.concat(distance_rows, ignore_index=True) if len(distance_rows) else pd.DataFrame()

    folds_all.to_csv(table_dir / "subject_loro_folds_hvg.csv", index=False)
    summary_all.to_csv(table_dir / "subject_loro_summary_hvg.csv", index=False)
    harm_diag_df.to_csv(table_dir / "harmonization_diagnostics_extended.csv", index=False)

    if len(summary_all) > 0:
        group_cols = ["model_combo", "harmonization", "basis_model", "strategy", "spatial_method"]
        model_grid = (
            summary_all.groupby(group_cols, as_index=False)
            .agg(
                n_subjects=("subject", "nunique"),
                mean_pearson=("pearson_r", "mean"),
                median_pearson=("pearson_r", "median"),
                mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
                mean_baseline_rmse=("baseline_rmse", "mean"),
                frac_better_baseline_rmse=("better_than_baseline_rmse", "mean"),
            )
            .sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        model_grid = pd.DataFrame()
    model_grid.to_csv(table_dir / "model_grid_summary_hvg.csv", index=False)

    # Coverage-tier gate summaries.
    if len(summary_all) > 0:
        tier = (
            summary_all.groupby(["model_combo", "coverage_tier"], as_index=False)
            .agg(mean_pearson=("pearson_r", "mean"), mean_rmse=("rmse", "mean"), mean_baseline_rmse=("baseline_rmse", "mean"), n_subjects=("subject", "nunique"))
        )
        tier["gate_pass"] = (tier["mean_pearson"] >= 0.50) & (tier["mean_rmse"] < tier["mean_baseline_rmse"])
        tier.to_csv(table_dir / "coverage_tier_gate_summary.csv", index=False)

        overall = (
            summary_all.groupby(["model_combo"], as_index=False)
            .agg(mean_pearson=("pearson_r", "mean"), mean_rmse=("rmse", "mean"), mean_baseline_rmse=("baseline_rmse", "mean"), n_subjects=("subject", "nunique"))
        )
        overall["gate_pass"] = (overall["mean_pearson"] >= 0.50) & (overall["mean_rmse"] < overall["mean_baseline_rmse"])
        overall.to_csv(table_dir / "rollout_gate_summary.csv", index=False)

    # Distance diagnostics by combo.
    if len(dist_all) > 0:
        dist_summary = summarize_distance_effect(dist_all, strategy_col="model_combo")
        dist_summary.to_csv(table_dir / "distance_effect_summary.csv", index=False)
        # Baseline-style plot for best combo.
        if len(model_grid) > 0:
            best_combo = str(model_grid.iloc[0]["model_combo"])
            d_best = dist_all[dist_all["model_combo"] == best_combo]
            plot_misfit_vs_distance(d_best, fig_dir / "misfit_vs_distance_best_combo.png", f"Misfit vs distance ({best_combo})")

    # Uncertainty calibration from fold table.
    if len(folds_all) > 0:
        m = np.isfinite(folds_all.get("hold_pred_std", np.nan)) & np.isfinite(folds_all.get("hold_abs_error_mean", np.nan))
        uc = uncertainty_calibration(
            folds_all.loc[m, "hold_pred_std"].to_numpy(dtype=np.float64),
            folds_all.loc[m, "hold_abs_error_mean"].to_numpy(dtype=np.float64),
            n_bins=10,
        )
        uc.to_csv(table_dir / "uncertainty_calibration.csv", index=False)
        if len(uc) > 0:
            plot_uncertainty_reliability(uc, fig_dir / "uncertainty_reliability.png")
            corr = plot_uvar_vs_error(
                folds_all.loc[m, "hold_pred_std"].to_numpy(dtype=np.float64),
                folds_all.loc[m, "hold_abs_error_mean"].to_numpy(dtype=np.float64),
                fig_dir / "uvar_vs_error_scatter.png",
            )
        else:
            corr = np.nan
    else:
        uc = pd.DataFrame()
        corr = np.nan

    _make_parity_report(out_root, model_grid, tolerance=float(cfg.parity_tolerance))

    summary = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "config_hash": cfg_hash,
        "n_hvg": int(len(genes_hvg)),
        "n_subjects_total": int(len(eligibility_df)),
        "n_subjects_eligible": int(eligibility_df["eligible"].sum()),
        "n_model_combos": int(len(model_grid)),
        "best_combo": str(model_grid.iloc[0]["model_combo"]) if len(model_grid) > 0 else None,
        "best_mean_pearson": float(model_grid.iloc[0]["mean_pearson"]) if len(model_grid) > 0 else np.nan,
        "best_mean_rmse": float(model_grid.iloc[0]["mean_rmse"]) if len(model_grid) > 0 else np.nan,
        "uncertainty_error_corr": float(corr) if np.isfinite(corr) else np.nan,
        "outputs": {
            "tables": str(table_dir),
            "figures": str(fig_dir),
            "parity": str(out_root / "parity"),
        },
    }
    io_utils.dump_json(out_root / "summary.json", summary)

    print("Completed vNext baseline run.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
