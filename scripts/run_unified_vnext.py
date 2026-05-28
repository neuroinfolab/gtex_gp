#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.diagnostics import (
    compute_tail_risk,
    compute_uncertainty_calibration,
    paired_method_tests,
    plot_uncertainty_reliability,
    plot_uvar_vs_error,
)
from src.eval.loro import run_loro
from src.harmonize import fit_harmonizer
from src.models.unified_generative import UnifiedGenerativeConfig, fit_global_atlas_unified, infer_subject_unified
from src.spatial.model_coords import model_spatial_coords, should_fold_hemispheres
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
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/vnext_unified"
    seed: int = 123
    latent_dim: int = 3
    max_iters: int = 5
    min_observed_parcels: int = 5
    c_min: int = 8
    model_variants: str = "unified_homo_none,unified_homo_huber,unified_homo_student,unified_hetero_huber,unified_hetero_student"
    spatial_method: str = "gp"
    uncertainty_shrink: bool = False
    unc_alpha: float = 0.5
    unc_beta: float = 0.5
    unc_m0: float = 0.5
    unc_tau: float = 0.2
    run_all_genes_winner: bool = False
    top_k_winner_allgenes: int = 1
    harmonizer: str = "combat"
    robust_loss: str = "none"
    heteroscedastic: bool = False
    gp_length_scale: float = 25.0
    gp_noise: float = 1e-3
    gp_optimize: bool = True
    gp_n_restarts: int = 0
    lambda_w: float = 1.0
    lambda_z: float = 1.0
    lambda_cal_a: float = 10.0
    lambda_cal_b: float = 10.0
    smoke_subjects: int = 0
    whiten_eps: float = 1e-4
    combat_use_covariates: bool = True
    gtex_rep_mode: str = "centroid"
    gtex_hemi_mode: str = "mirror_left"
    hier_lambda_a: float = 10.0
    hier_lambda_b: float = 10.0
    hier_base_method: str = "robustz_affine"
    hier_subject_subset: list[str] | None = None
    baseline_ref_root: str = "out/vnext_alignment"
    reference_combo: str = "combat__affine_gl3__constrained_anchor__rbf"


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Unified probabilistic model runner (Milestone E/F)")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    p.add_argument("--max-iters", type=int, default=Config.max_iters)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--model-variants", default=Config.model_variants)
    p.add_argument("--spatial-method", default=Config.spatial_method)
    p.add_argument("--uncertainty-shrink", default=str(Config.uncertainty_shrink).lower())
    p.add_argument("--unc-alpha", type=float, default=Config.unc_alpha)
    p.add_argument("--unc-beta", type=float, default=Config.unc_beta)
    p.add_argument("--unc-m0", type=float, default=Config.unc_m0)
    p.add_argument("--unc-tau", type=float, default=Config.unc_tau)
    p.add_argument("--run-all-genes-winner", default=str(Config.run_all_genes_winner).lower())
    p.add_argument("--top-k-winner-allgenes", type=int, default=Config.top_k_winner_allgenes)
    p.add_argument("--harmonizer", default=Config.harmonizer)
    p.add_argument("--robust-loss", default=Config.robust_loss)
    p.add_argument("--heteroscedastic", default=str(Config.heteroscedastic).lower())
    p.add_argument("--gp-length-scale", type=float, default=Config.gp_length_scale)
    p.add_argument("--gp-noise", type=float, default=Config.gp_noise)
    p.add_argument("--gp-optimize", default=str(Config.gp_optimize).lower())
    p.add_argument("--gp-n-restarts", type=int, default=Config.gp_n_restarts)
    p.add_argument("--lambda-w", type=float, default=Config.lambda_w)
    p.add_argument("--lambda-z", type=float, default=Config.lambda_z)
    p.add_argument("--lambda-cal-a", type=float, default=Config.lambda_cal_a)
    p.add_argument("--lambda-cal-b", type=float, default=Config.lambda_cal_b)
    p.add_argument("--smoke-subjects", type=int, default=Config.smoke_subjects)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--gtex-rep-mode", choices=["centroid", "medoid"], default=Config.gtex_rep_mode)
    p.add_argument("--gtex-hemi-mode", choices=["native", "mirror_left"], default=Config.gtex_hemi_mode)
    p.add_argument("--combat-use-covariates", default=str(Config.combat_use_covariates).lower())
    p.add_argument("--hier-lambda-a", type=float, default=Config.hier_lambda_a)
    p.add_argument("--hier-lambda-b", type=float, default=Config.hier_lambda_b)
    p.add_argument("--hier-base-method", default=Config.hier_base_method)
    p.add_argument("--baseline-ref-root", default=Config.baseline_ref_root)
    p.add_argument("--reference-combo", default=Config.reference_combo)
    a = p.parse_args()
    d = vars(a)
    d["uncertainty_shrink"] = _parse_bool(d["uncertainty_shrink"])
    d["run_all_genes_winner"] = _parse_bool(d["run_all_genes_winner"])
    d["heteroscedastic"] = _parse_bool(d["heteroscedastic"])
    d["gp_optimize"] = _parse_bool(d["gp_optimize"])
    d["combat_use_covariates"] = _parse_bool(d["combat_use_covariates"])
    return Config(**d)


def parse_variant(token: str, cfg: Config) -> Dict[str, object]:
    t = token.strip()
    if not t:
        return {}

    spec = {
        "harm": cfg.harmonizer,
        "cal": "hier_affine_map",
        "robust": cfg.robust_loss,
        "hetero": "gene_var" if cfg.heteroscedastic else "none",
        "uncshrink": str(cfg.uncertainty_shrink).lower(),
    }

    if "=" in t and "__" in t:
        for p in t.split("__"):
            if "=" in p:
                k, v = p.split("=", 1)
                spec[k.strip()] = v.strip()
    else:
        lookup = {
            "unified_homo_none": {"robust": "none", "hetero": "none", "uncshrink": "false"},
            "unified_homo_huber": {"robust": "huber", "hetero": "none", "uncshrink": "false"},
            "unified_homo_student": {"robust": "student_t", "hetero": "none", "uncshrink": "false"},
            "unified_hetero_huber": {"robust": "huber", "hetero": "gene_var", "uncshrink": "false"},
            "unified_hetero_student": {"robust": "student_t", "hetero": "gene_var", "uncshrink": "false"},
        }
        if t in lookup:
            spec.update(lookup[t])
        else:
            spec["robust"] = t

    combo = (
        f"unified__harm={spec['harm']}__cal={spec['cal']}"
        f"__robust={spec['robust']}__hetero={spec['hetero']}__uncshrink={spec['uncshrink']}"
    )

    return {
        "token": t,
        "model_combo": combo,
        "harmonizer": str(spec["harm"]),
        "calibration_mode": str(spec["cal"]),
        "robust_loss": str(spec["robust"]),
        "heteroscedastic": str(spec["hetero"]).lower() != "none",
        "uncertainty_shrink": str(spec["uncshrink"]).lower() == "true",
    }


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_core_figures(
    model_grid: pd.DataFrame,
    subject_summary: pd.DataFrame,
    tail_df: pd.DataFrame,
    out_fig: Path,
    ref_combo: str,
) -> None:
    if len(model_grid):
        d = model_grid.sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        sns.barplot(data=d, x="model_combo", y="mean_pearson", ax=axes[0], color="#2a9d8f")
        axes[0].tick_params(axis="x", rotation=75)
        axes[0].set_title("Unified model ranking (mean Pearson)")
        axes[0].set_xlabel("")
        sns.barplot(data=d, x="model_combo", y="mean_rmse", ax=axes[1], color="#e76f51")
        axes[1].tick_params(axis="x", rotation=75)
        axes[1].set_title("Unified model ranking (mean RMSE)")
        axes[1].set_xlabel("")
        _save_fig(fig, out_fig / "unified_model_ranking_hvg.png")

    if len(subject_summary):
        if ref_combo in set(subject_summary["model_combo"]):
            piv = subject_summary.pivot_table(index="subject", columns="model_combo", values="rmse", aggfunc="mean")
            if ref_combo in piv.columns:
                fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                for col in [c for c in piv.columns if c != ref_combo]:
                    xy = piv[[ref_combo, col]].dropna()
                    if len(xy) == 0:
                        continue
                    ax.scatter(xy[ref_combo], xy[col], s=12, alpha=0.35, label=col[:28])
                lims = ax.get_xlim()
                ax.plot(lims, lims, "k--", linewidth=1)
                ax.set_xlabel(f"{ref_combo} RMSE")
                ax.set_ylabel("Unified model RMSE")
                ax.set_title("Unified vs reference baseline (subject RMSE)")
                if len(piv.columns) <= 6:
                    ax.legend(loc="best", fontsize=7)
                _save_fig(fig, out_fig / "unified_vs_baseline_scatter.png")

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.scatterplot(data=subject_summary, x="n_obs_parcels", y="pearson_r", hue="model_combo", s=16, alpha=0.45, ax=ax)
        ax.set_title("Coverage vs performance (Pearson)")
        ax.legend(loc="best", fontsize=7)
        _save_fig(fig, out_fig / "unified_coverage_vs_performance.png")

        # Bad-subject strip plot for winner.
        winner = (
            subject_summary.groupby("model_combo")["pearson_r"].mean().sort_values(ascending=False).index.tolist()[0]
            if len(subject_summary)
            else None
        )
        if winner is not None:
            ww = subject_summary[subject_summary["model_combo"] == winner].copy().sort_values("rmse", ascending=False).head(60)
            fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
            sns.stripplot(data=ww, x="subject", y="rmse", hue="n_obs_parcels", dodge=False, size=5, ax=ax)
            ax.tick_params(axis="x", rotation=90)
            ax.set_title(f"Highest-RMSE subjects ({winner})")
            _save_fig(fig, out_fig / "unified_bad_subjects_stripplot.png")

    if len(tail_df):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        dd = tail_df.sort_values("p95_rmse", ascending=True)
        x = np.arange(len(dd))
        ax.bar(x - 0.18, dd["p90_rmse"], width=0.36, label="P90 RMSE")
        ax.bar(x + 0.18, dd["p95_rmse"], width=0.36, label="P95 RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(dd["model_combo"], rotation=75, ha="right")
        ax.set_title("Tail risk by model")
        ax.legend()
        _save_fig(fig, out_fig / "unified_tail_risk_p90_p95.png")


def _git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)

    root = Path(".").resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    config_dir = out_root / "configs"
    for p in [table_dir, fig_dir, config_dir]:
        p.mkdir(parents=True, exist_ok=True)

    io_utils.dump_yaml(config_dir / "unified_vnext.yaml", asdict(cfg))
    io_utils.dump_json(config_dir / "unified_vnext.json", asdict(cfg))

    header = io_utils.load_gene_header_and_hvg((root / cfg.csv_path).resolve(), (root / cfg.hvg_path).resolve())
    genes_hvg = header["genes_hvg"]
    if len(genes_hvg) == 0:
        raise RuntimeError("No overlapping HVGs found.")

    df = io_utils.read_expression_subset(
        (root / cfg.csv_path).resolve(),
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
    coords_model_full = model_spatial_coords(
        coords_full,
        fold_hemispheres=should_fold_hemispheres(cfg.gtex_hemi_mode, "default"),
    )

    eligibility_df = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    if cfg.smoke_subjects > 0:
        keep = eligibility_df[eligibility_df["eligible"]].head(int(cfg.smoke_subjects))["subject"].tolist()
        eligibility_df["eligible"] = eligibility_df["subject"].isin(set(keep))
    eligibility_df.to_csv(table_dir / "unified_subject_eligibility.csv", index=False)
    eligible_subjects = eligibility_df[eligibility_df["eligible"]]["subject"].astype(str).tolist()
    if len(eligible_subjects) == 0:
        raise RuntimeError("No eligible subjects for unified run.")

    tokens = [t.strip() for t in str(cfg.model_variants).split(",") if t.strip()]
    variants = [parse_variant(t, cfg) for t in tokens]
    variants = [v for v in variants if len(v)]
    pd.DataFrame(variants).to_csv(table_dir / "unified_variant_manifest.csv", index=False)

    config_hash_base = io_utils.hash_config(asdict(cfg))

    all_folds = []
    all_summary = []
    conv_rows = []
    cal_rows = []
    gp_rows = []

    for vidx, v in enumerate(variants, start=1):
        print(f"[{vidx}/{len(variants)}] Running {v['model_combo']}")

        cfg.hier_subject_subset = eligible_subjects
        harm = fit_harmonizer(ahba_raw, gtex_raw, genes_hvg, method=v["harmonizer"], cfg=cfg)
        ahba_h = harm.transform(ahba_raw, "AHBA")
        gtex_h = harm.transform(gtex_raw, "GTEX")
        ahba_h_full, _ = build_region_matrix(ahba_h, genes_hvg, target_meta, agg="mean")

        ucfg = UnifiedGenerativeConfig(
            latent_dim=cfg.latent_dim,
            max_iters=cfg.max_iters,
            lambda_w=cfg.lambda_w,
            lambda_z=cfg.lambda_z,
            lambda_cal_a=cfg.lambda_cal_a,
            lambda_cal_b=cfg.lambda_cal_b,
            gp_length_scale=cfg.gp_length_scale,
            gp_noise=cfg.gp_noise,
            gp_optimize=bool(cfg.gp_optimize),
            gp_n_restarts=int(cfg.gp_n_restarts),
            robust_loss=v["robust_loss"],
            heteroscedastic=bool(v["heteroscedastic"]),
            calibration_mode=v["calibration_mode"],
            uncertainty_shrink=bool(v["uncertainty_shrink"]),
            unc_alpha=cfg.unc_alpha,
            unc_beta=cfg.unc_beta,
            unc_m0=cfg.unc_m0,
            unc_tau=cfg.unc_tau,
            random_state=cfg.seed,
        )

        atlas = fit_global_atlas_unified(ahba_h_full, coords_model_full, ucfg)
        model_hash = io_utils.hash_config({"base": config_hash_base, **v})

        fold_rows = []
        summary_rows = []

        for sid in eligible_subjects:
            subj_h = gtex_h[gtex_h["subject"] == sid].copy()
            subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
            obs_idx, xh, xr = build_subject_observed_matrices(subj_h, subj_raw, genes_hvg)
            if len(obs_idx) < cfg.min_observed_parcels:
                continue

            # Full-fit diagnostics for this subject.
            full_res = infer_subject_unified(
                {"subject": sid, "obs_idx": obs_idx, "X_obs_h": xh},
                atlas,
                ucfg,
                fold_ctx={"prior_h": ahba_h_full, "obs_idx": obs_idx},
            )
            R = np.asarray(full_res["r_align"], dtype=np.float64)
            conv = full_res.get("convergence", {})
            conv_rows.append(
                {
                    "subject": sid,
                    "model_combo": v["model_combo"],
                    "orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(R.shape[0]))),
                    "det_r": float(np.linalg.det(R)),
                    "objective_last": float(conv.get("objective", [np.nan])[-1] if len(conv.get("objective", [])) else np.nan),
                    "delta_r_last": float(conv.get("delta_R", [np.nan])[-1] if len(conv.get("delta_R", [])) else np.nan),
                    "delta_z_last": float(conv.get("delta_Z", [np.nan])[-1] if len(conv.get("delta_Z", [])) else np.nan),
                    "delta_cal_last": float(conv.get("delta_cal", [np.nan])[-1] if len(conv.get("delta_cal", [])) else np.nan),
                    "max_iters_reached": bool(conv.get("max_iters_reached", False)),
                }
            )
            a_sub = np.asarray(full_res.get("a_subj", np.ones(xh.shape[1])), dtype=np.float64)
            b_sub = np.asarray(full_res.get("b_subj", np.zeros(xh.shape[1])), dtype=np.float64)
            cal_rows.append(
                {
                    "subject": sid,
                    "model_combo": v["model_combo"],
                    "a_mean": float(np.nanmean(a_sub)),
                    "a_std": float(np.nanstd(a_sub)),
                    "b_mean": float(np.nanmean(b_sub)),
                    "b_std": float(np.nanstd(b_sub)),
                    "a_abs_shift_mean": float(np.nanmean(np.abs(a_sub - 1.0))),
                    "b_abs_mean": float(np.nanmean(np.abs(b_sub))),
                }
            )
            uvar_full = np.asarray(full_res.get("uvar_full", np.zeros((coords_full.shape[0], 1))), dtype=np.float64)
            gp_rows.append(
                {
                    "subject": sid,
                    "model_combo": v["model_combo"],
                    "uvar_mean": float(np.nanmean(uvar_full)),
                    "uvar_median": float(np.nanmedian(uvar_full)),
                    "uvar_p95": float(np.nanpercentile(uvar_full, 95)),
                }
            )

            def _pipeline_for_fold(hold: int, train_idx: np.ndarray):
                train_mask = ~((gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold)))
                gtex_train = add_sample_groups(gtex_raw[train_mask].copy(), target_meta)

                cfg.hier_subject_subset = [sid]
                harm_fold = fit_harmonizer(ahba_raw, gtex_train, genes_hvg, method=v["harmonizer"], cfg=cfg)
                ahba_h_fold = harm_fold.transform(ahba_raw, "AHBA")
                gtex_h_fold = harm_fold.transform(gtex_train, "GTEX")
                ahba_h_full_fold, _ = build_region_matrix(ahba_h_fold, genes_hvg, target_meta, agg="mean")
                atlas_fold = fit_global_atlas_unified(ahba_h_full_fold, coords_model_full, ucfg)

                subj_h_fold = gtex_h_fold[gtex_h_fold["subject"] == sid].copy()
                subj_raw_fold = gtex_train[gtex_train["subject"] == sid].copy()
                obs_idx_fold, xh_fold, _ = build_subject_observed_matrices(subj_h_fold, subj_raw_fold, genes_hvg)
                leak_flag = bool(int(hold) in set(obs_idx_fold.tolist()))
                if len(obs_idx_fold) < 2:
                    raise RuntimeError("insufficient train parcels in fold")

                pred_fold = infer_subject_unified(
                    {"subject": sid, "obs_idx": obs_idx_fold, "X_obs_h": xh_fold},
                    atlas_fold,
                    ucfg,
                    fold_ctx={"prior_h": ahba_h_full_fold, "obs_idx": obs_idx_fold},
                )
                hold_pred = pred_fold["x_hat_h_full"][int(hold), :]
                hold_std = float(np.sqrt(np.maximum(np.mean(pred_fold["uvar_full"][int(hold), :]), 0.0)))
                convf = pred_fold.get("convergence", {})
                return hold_pred, {
                    "leak_flag": leak_flag,
                    "hold_pred_std": hold_std,
                    "conv_obj_last": float(convf.get("objective", [np.nan])[-1] if len(convf.get("objective", [])) else np.nan),
                    "conv_delta_r_last": float(convf.get("delta_R", [np.nan])[-1] if len(convf.get("delta_R", [])) else np.nan),
                    "conv_delta_z_last": float(convf.get("delta_Z", [np.nan])[-1] if len(convf.get("delta_Z", [])) else np.nan),
                    "conv_delta_cal_last": float(convf.get("delta_cal", [np.nan])[-1] if len(convf.get("delta_cal", [])) else np.nan),
                }

            eval_bundle = {
                "subject": sid,
                "obs_idx": obs_idx,
                "X_obs_h": xh,
                "baseline_h_full": ahba_h_full,
                "config_hash": model_hash,
            }
            folds_df, summary = run_loro(eval_bundle, _pipeline_for_fold, {"config_hash": model_hash, "min_train_obs_loro": 2})
            folds_df["model_combo"] = v["model_combo"]
            folds_df["harmonization"] = v["harmonizer"]
            folds_df["robust_loss"] = v["robust_loss"]
            folds_df["heteroscedastic"] = bool(v["heteroscedastic"])
            folds_df["uncertainty_shrink"] = bool(v["uncertainty_shrink"])
            fold_rows.append(folds_df)

            summary.update(
                {
                    "model_combo": v["model_combo"],
                    "harmonization": v["harmonizer"],
                    "basis_model": "unified_map",
                    "strategy": "uncertainty_shrink" if bool(v["uncertainty_shrink"]) else "baseline",
                    "spatial_method": "gp",
                }
            )
            summary_rows.append(summary)

        variant_folds = pd.concat(fold_rows, ignore_index=True) if len(fold_rows) else pd.DataFrame()
        variant_summary = pd.DataFrame(summary_rows)
        all_folds.append(variant_folds)
        all_summary.append(variant_summary)

    folds_all = pd.concat(all_folds, ignore_index=True) if len(all_folds) else pd.DataFrame()
    summary_all = pd.concat(all_summary, ignore_index=True) if len(all_summary) else pd.DataFrame()
    conv_df = pd.DataFrame(conv_rows)
    cal_df = pd.DataFrame(cal_rows)
    gp_df = pd.DataFrame(gp_rows)

    folds_all.to_csv(table_dir / "unified_subject_loro_folds_hvg.csv", index=False)
    summary_all.to_csv(table_dir / "unified_subject_loro_summary_hvg.csv", index=False)
    conv_df.to_csv(table_dir / "unified_convergence_diagnostics.csv", index=False)
    cal_df.to_csv(table_dir / "unified_calibration_diagnostics.csv", index=False)
    gp_df.to_csv(table_dir / "unified_gp_diagnostics.csv", index=False)

    if len(summary_all):
        model_grid = (
            summary_all.groupby(["model_combo", "harmonization", "basis_model", "strategy", "spatial_method"], as_index=False)
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
        model_grid["gate_pass"] = (model_grid["mean_pearson"] >= 0.50) & (model_grid["mean_rmse"] < model_grid["mean_baseline_rmse"])
    else:
        model_grid = pd.DataFrame(
            columns=[
                "model_combo",
                "harmonization",
                "basis_model",
                "strategy",
                "spatial_method",
                "n_subjects",
                "mean_pearson",
                "median_pearson",
                "mean_rmse",
                "median_rmse",
                "mean_baseline_rmse",
                "frac_better_baseline_rmse",
                "gate_pass",
            ]
        )

    model_grid.to_csv(table_dir / "unified_model_grid_summary_hvg.csv", index=False)

    if len(summary_all):
        bins = pd.cut(summary_all["n_obs_parcels"], bins=[4, 5, 6, 7, 8, 9, 10, 11, 12], labels=["5", "6", "7", "8", "9", "10", "11", "12"], right=False)
        tier = np.where(summary_all["n_obs_parcels"] < cfg.c_min, "lt_cmin", "ge_cmin")
        cov_df = (
            summary_all.assign(n_obs_bin=bins.astype(str), coverage_tier=tier)
            .groupby(["model_combo", "coverage_tier", "n_obs_bin"], as_index=False)
            .agg(
                n_subjects=("subject", "nunique"),
                mean_pearson=("pearson_r", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_baseline_rmse=("baseline_rmse", "mean"),
            )
        )
    else:
        cov_df = pd.DataFrame(columns=["model_combo", "coverage_tier", "n_obs_bin", "n_subjects", "mean_pearson", "mean_rmse", "mean_baseline_rmse"])
    cov_df.to_csv(table_dir / "unified_coverage_tier_summary.csv", index=False)

    unc_cal = compute_uncertainty_calibration(folds_all)
    unc_cal.to_csv(table_dir / "unified_uncertainty_calibration.csv", index=False)

    tail_df = compute_tail_risk(summary_all)
    tail_df.to_csv(table_dir / "unified_tail_risk_summary.csv", index=False)

    stats_rmse = paired_method_tests(summary_all[["subject", "model_combo", "rmse"]].copy(), metric="rmse", reference_model=cfg.reference_combo)
    stats_pr = paired_method_tests(summary_all[["subject", "model_combo", "pearson_r"]].copy(), metric="pearson_r", reference_model=cfg.reference_combo)
    pair_df = pd.concat([stats_rmse, stats_pr], ignore_index=True)
    pair_df.to_csv(table_dir / "unified_pairwise_stats.csv", index=False)

    # Unified vs baseline comparison table.
    base_root = (root / cfg.baseline_ref_root / "tables").resolve()
    base_grid_path = base_root / "model_grid_summary_hvg.csv"
    if base_grid_path.exists() and len(model_grid):
        base_grid = pd.read_csv(base_grid_path)
        ref_rows = base_grid[base_grid["model_combo"].isin([
            "combat__affine_gl3__constrained_anchor__rbf",
            "combat__affine_gl3__distance_shrink__rbf",
            "hier_affine__affine_gl3__distance_shrink__rbf",
        ])].copy()
        ref_rows = ref_rows[["model_combo", "mean_pearson", "mean_rmse", "mean_baseline_rmse", "frac_better_baseline_rmse"]]
        ref_rows["source"] = "baseline_ad"
        uni_rows = model_grid[["model_combo", "mean_pearson", "mean_rmse", "mean_baseline_rmse", "frac_better_baseline_rmse"]].copy()
        uni_rows["source"] = "unified_ef"
        cmp = pd.concat([uni_rows, ref_rows], ignore_index=True)
        if cfg.reference_combo in set(cmp["model_combo"]):
            rr = cmp[cmp["model_combo"] == cfg.reference_combo].iloc[0]
            cmp["delta_pearson_vs_ref"] = cmp["mean_pearson"] - float(rr["mean_pearson"])
            cmp["delta_rmse_vs_ref"] = cmp["mean_rmse"] - float(rr["mean_rmse"])
        cmp.to_csv(table_dir / "unified_vs_baseline_comparison.csv", index=False)
    else:
        pd.DataFrame().to_csv(table_dir / "unified_vs_baseline_comparison.csv", index=False)

    # Figures.
    _plot_core_figures(model_grid, summary_all, tail_df, fig_dir, cfg.reference_combo)
    plot_uncertainty_reliability(unc_cal, fig_dir / "unified_uncertainty_reliability.png")
    _ = plot_uvar_vs_error(
        folds_all.get("hold_pred_std", pd.Series(dtype=float)).to_numpy(dtype=np.float64),
        folds_all.get("hold_abs_error_mean", pd.Series(dtype=float)).to_numpy(dtype=np.float64),
        fig_dir / "unified_uvar_vs_error.png",
    )

    winner = model_grid.iloc[0]["model_combo"] if len(model_grid) else None

    summary = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "git_hash": _git_hash(root),
        "config_hash": config_hash_base,
        "n_hvg": int(len(genes_hvg)),
        "n_subjects": int(summary_all["subject"].nunique()) if len(summary_all) else 0,
        "n_models": int(model_grid.shape[0]),
        "winner_model": winner,
        "gate_pass_count": int(np.sum(model_grid["gate_pass"])) if "gate_pass" in model_grid else 0,
        "reference_combo": cfg.reference_combo,
    }
    io_utils.dump_json(out_root / "summary.json", summary)

    # Report
    lines = [
        "# Unified vNext (Milestones E/F) Report",
        "",
        f"Generated: {summary['timestamp_utc']}",
        f"Git hash: {summary['git_hash']}",
        "",
        "## Cohort",
        f"- Eligible subjects: {summary['n_subjects']}",
        f"- HVGs: {summary['n_hvg']}",
        f"- Unified variants: {summary['n_models']}",
        f"- Gate pass count: {summary['gate_pass_count']}",
        f"- Winner: `{summary['winner_model']}`",
        "",
        "## Main Artifacts",
        "- `tables/unified_model_grid_summary_hvg.csv`",
        "- `tables/unified_subject_loro_summary_hvg.csv`",
        "- `tables/unified_subject_loro_folds_hvg.csv`",
        "- `tables/unified_pairwise_stats.csv`",
        "- `tables/unified_uncertainty_calibration.csv`",
        "- `tables/unified_tail_risk_summary.csv`",
        "",
        "## Figures",
        "- `figures/unified_model_ranking_hvg.png`",
        "- `figures/unified_vs_baseline_scatter.png`",
        "- `figures/unified_coverage_vs_performance.png`",
        "- `figures/unified_uncertainty_reliability.png`",
        "- `figures/unified_uvar_vs_error.png`",
        "- `figures/unified_tail_risk_p90_p95.png`",
        "- `figures/unified_bad_subjects_stripplot.png`",
    ]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")

    # Optional PDF.
    try:
        subprocess.run(
            [
                "pandoc",
                str(out_root / "report.md"),
                "-o",
                str(out_root / "report.pdf"),
                "--pdf-engine=tectonic",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        pass

    io_utils.dump_json(
        out_root / "report_manifest.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "config": asdict(cfg),
            "tables": sorted([p.name for p in table_dir.glob("*.csv")]),
            "figures": sorted([p.name for p in fig_dir.glob("*.png")]),
        },
    )

    print("Unified vNext run complete.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
