#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflows.loro_cache import MODEL_NAMES, SubjectCacheConfig, list_eligible_subjects, process_subject_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch driver for subject-wise LORO cache generation.")
    p.add_argument("--csv-path", default=SubjectCacheConfig.csv_path)
    p.add_argument("--hvg-path", default=SubjectCacheConfig.hvg_path)
    p.add_argument("--out-root", default=SubjectCacheConfig.out_root)
    p.add_argument("--gene-scope", choices=["allgenes", "hvg"], default=SubjectCacheConfig.gene_scope)
    p.add_argument("--use-cache", default=str(SubjectCacheConfig.use_cache).lower())
    p.add_argument("--model", choices=list(MODEL_NAMES) + ["all"], default="all")
    p.add_argument("--subject", default=None, help="Process one explicit subject ID.")
    p.add_argument("--subject-index", type=int, default=None, help="0-based index in eligible subject list.")
    p.add_argument("--from-sbatch-array", action="store_true", help="Use SLURM_ARRAY_TASK_ID (1-based) as subject index.")
    p.add_argument("--max-subjects", type=int, default=0, help="Optional cap for local testing.")
    p.add_argument("--min-observed-parcels", type=int, default=SubjectCacheConfig.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=SubjectCacheConfig.c_min)
    p.add_argument("--n-comp-target", type=int, default=SubjectCacheConfig.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=SubjectCacheConfig.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=SubjectCacheConfig.rbf_smoothing)
    p.add_argument("--gp-length-scale", type=float, default=SubjectCacheConfig.gp_length_scale)
    p.add_argument("--gp-noise", type=float, default=SubjectCacheConfig.gp_noise)
    p.add_argument("--seed", type=int, default=SubjectCacheConfig.seed)
    p.add_argument("--combat-use-covariates", default=str(SubjectCacheConfig.combat_use_covariates).lower())
    p.add_argument("--drop-macro-system-covariate", default=str(SubjectCacheConfig.drop_macro_system_covariate).lower())
    p.add_argument("--latent-dim", type=int, default=SubjectCacheConfig.latent_dim)
    p.add_argument("--dynamic-rank", default=str(SubjectCacheConfig.dynamic_rank).lower())
    p.add_argument("--plam-latent-dim-max", type=int, default=SubjectCacheConfig.plam_latent_dim_max)
    p.add_argument("--plam-max-iters", type=int, default=SubjectCacheConfig.plam_max_iters)
    p.add_argument("--lambda-w", type=float, default=SubjectCacheConfig.lambda_w)
    p.add_argument("--lambda-z", type=float, default=SubjectCacheConfig.lambda_z)
    p.add_argument("--lambda-cal-a", type=float, default=SubjectCacheConfig.lambda_cal_a)
    p.add_argument("--lambda-cal-b", type=float, default=SubjectCacheConfig.lambda_cal_b)
    p.add_argument("--robust-loss", default=SubjectCacheConfig.robust_loss)
    p.add_argument("--heteroscedastic", default=str(SubjectCacheConfig.heteroscedastic).lower())
    p.add_argument("--calibration-mode", default=SubjectCacheConfig.calibration_mode)
    p.add_argument("--uncertainty-shrink", default=str(SubjectCacheConfig.uncertainty_shrink).lower())
    p.add_argument("--atlas-agg", choices=["mean", "median"], default=SubjectCacheConfig.atlas_agg)
    p.add_argument("--gtex-rep-mode", choices=["centroid", "medoid"], default=SubjectCacheConfig.gtex_rep_mode)
    p.add_argument("--gtex-hemi-mode", choices=["native", "mirror_left"], default=SubjectCacheConfig.gtex_hemi_mode)
    p.add_argument(
        "--matching-policy",
        choices=["centroids", "centroids_and_volumes"],
        default=SubjectCacheConfig.matching_policy,
    )
    p.add_argument(
        "--matching-policy-hemi-mode",
        choices=["default", "force_left"],
        default=SubjectCacheConfig.matching_policy_hemi_mode,
    )
    p.add_argument("--collapse-cerebellum", default=str(SubjectCacheConfig.collapse_cerebellum).lower())
    return p.parse_args()


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _cfg_from_args(a: argparse.Namespace) -> SubjectCacheConfig:
    return SubjectCacheConfig(
        csv_path=a.csv_path,
        hvg_path=a.hvg_path,
        out_root=a.out_root,
        gene_scope=a.gene_scope,
        use_cache=_parse_bool(a.use_cache),
        min_observed_parcels=int(a.min_observed_parcels),
        c_min=int(a.c_min),
        n_comp_target=int(a.n_comp_target),
        ridge_alpha_bridge=float(a.ridge_alpha_bridge),
        rbf_smoothing=float(a.rbf_smoothing),
        gp_length_scale=float(a.gp_length_scale),
        gp_noise=float(a.gp_noise),
        seed=int(a.seed),
        combat_use_covariates=_parse_bool(a.combat_use_covariates),
        drop_macro_system_covariate=_parse_bool(a.drop_macro_system_covariate),
        latent_dim=int(a.latent_dim),
        dynamic_rank=_parse_bool(a.dynamic_rank),
        plam_latent_dim_max=int(a.plam_latent_dim_max),
        plam_max_iters=int(a.plam_max_iters),
        lambda_w=float(a.lambda_w),
        lambda_z=float(a.lambda_z),
        lambda_cal_a=float(a.lambda_cal_a),
        lambda_cal_b=float(a.lambda_cal_b),
        robust_loss=str(a.robust_loss),
        heteroscedastic=_parse_bool(a.heteroscedastic),
        calibration_mode=str(a.calibration_mode),
        uncertainty_shrink=_parse_bool(a.uncertainty_shrink),
        atlas_agg=str(a.atlas_agg).lower(),
        gtex_rep_mode=str(a.gtex_rep_mode).lower(),
        gtex_hemi_mode=str(a.gtex_hemi_mode).lower(),
        matching_policy=str(a.matching_policy).lower(),
        matching_policy_hemi_mode=str(a.matching_policy_hemi_mode).lower(),
        collapse_cerebellum=_parse_bool(a.collapse_cerebellum),
    )


def _pick_subjects(eligible: List[str], a: argparse.Namespace) -> List[str]:
    if a.subject:
        return [str(a.subject)]

    idx = a.subject_index
    if bool(a.from_sbatch_array):
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        idx = int(task_id) - 1
    if idx is not None:
        if idx < 0:
            raise IndexError(f"subject index {idx} is negative")
        if idx >= len(eligible):
            # Sbatch arrays are sized to a generous upper bound (full GTEx pool)
            # so the same submission works across policies / floors. Tasks that
            # land beyond the actual eligible count exit cleanly with no work.
            print(
                f"[skip-out-of-range] task_id={idx + 1} n_eligible={len(eligible)} "
                "-- nothing to do, exiting cleanly"
            )
            return []
        return [str(eligible[idx])]

    out = list(eligible)
    if int(a.max_subjects) > 0:
        out = out[: int(a.max_subjects)]
    return out


def main() -> None:
    a = parse_args()
    cfg = _cfg_from_args(a)
    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)

    eligible = list_eligible_subjects(cfg)
    subjects = _pick_subjects(eligible, a)
    models = list(MODEL_NAMES) if str(a.model) == "all" else [str(a.model)]
    print(f"[batch] n_eligible={len(eligible)} n_selected={len(subjects)} models={models}")

    for sid in subjects:
        for m in models:
            process_subject_model(cfg, subject=sid, model_name=m)


if __name__ == "__main__":
    main()
