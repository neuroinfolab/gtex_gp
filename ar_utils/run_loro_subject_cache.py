#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
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
from src.workflows.writeup_pipeline import _resolve_optional_path, _source_signature

MODEL_NAMES = ("naive", "dlam", "plam")


@dataclass
class SubjectCacheConfig:
    csv_path: str = "data/raw/gxp_samples.csv"
    hvg_path: str = "data/raw/ahba_100hvg.txt"
    out_root: str = "out/loro_subject_cache"
    gene_scope: str = "hvg"
    use_cache: bool = True
    min_observed_parcels: int = 5
    c_min: int = 4
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_length_scale: float = 25.0
    gp_noise: float = 1e-3
    seed: int = 123
    combat_use_covariates: bool = True
    latent_dim: int = 3
    dynamic_rank: bool = False
    plam_latent_dim_max: int = 10
    plam_max_iters: int = 5
    lambda_w: float = 1.0
    lambda_z: float = 1.0
    lambda_cal_a: float = 10.0
    lambda_cal_b: float = 10.0
    robust_loss: str = "student_t"
    heteroscedastic: bool = True
    calibration_mode: str = "hier_affine_map"
    uncertainty_shrink: bool = False
    atlas_agg: str = "mean"
    gtex_rep_mode: str = "medoid"
    gtex_hemi_mode: str = "native"


def load_dataset(cfg: SubjectCacheConfig) -> Dict[str, object]:
    if str(cfg.atlas_agg).lower() not in {"mean", "median"}:
        raise ValueError(f"atlas_agg must be 'mean' or 'median', got {cfg.atlas_agg!r}")
    if str(cfg.gtex_rep_mode).lower() not in {"centroid", "medoid"}:
        raise ValueError(f"gtex_rep_mode must be 'centroid' or 'medoid', got {cfg.gtex_rep_mode!r}")
    if str(cfg.gtex_hemi_mode).lower() not in {"native", "mirror_left"}:
        raise ValueError(f"gtex_hemi_mode must be 'native' or 'mirror_left', got {cfg.gtex_hemi_mode!r}")
    csv_path = _resolve_optional_path(cfg.csv_path, ["gxp_samples.csv"])
    hvg_path = _resolve_optional_path(cfg.hvg_path, ["data/raw/ahba_100hvg.txt", "ahba_100hvg.txt"])
    header = io_utils.load_gene_header_and_hvg(csv_path, hvg_path)
    genes = header["genes_all"] if str(cfg.gene_scope).lower() == "allgenes" else header["genes_hvg"]
    if not genes:
        raise RuntimeError(f"No genes found for gene_scope={cfg.gene_scope}")
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
    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    elig = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    eligible_subjects = elig[elig["eligible"]]["subject"].astype(str).tolist()
    return {
        "csv_path": csv_path,
        "hvg_path": hvg_path,
        "genes": genes,
        "ahba_raw": ahba_raw,
        "gtex_raw": gtex_raw,
        "target_meta": target_meta,
        "coords_full": coords_full,
        "eligibility": elig,
        "eligible_subjects": eligible_subjects,
    }


def list_eligible_subjects(cfg: SubjectCacheConfig) -> List[str]:
    bundle = load_dataset(cfg)
    return [str(s) for s in bundle["eligible_subjects"]]


def _npz_path(cfg: SubjectCacheConfig, model_name: str, subject: str) -> Path:
    return (Path(cfg.out_root).resolve() / str(cfg.gene_scope).lower() / model_name / f"{subject}.npz").resolve()


def _meta_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".json")


def _cfg_hash(cfg: SubjectCacheConfig, subject: str, model_name: str) -> str:
    payload = asdict(cfg).copy()
    payload["subject"] = str(subject)
    payload["model_name"] = str(model_name)
    return io_utils.hash_config(payload)


def _cache_valid(npz_path: Path, cfg: SubjectCacheConfig, subject: str, model_name: str, sources: Dict[str, object]) -> bool:
    meta_path = _meta_path(npz_path)
    if not npz_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if meta.get("status") != "ok":
        return False
    if meta.get("cfg_hash") != _cfg_hash(cfg, subject, model_name):
        return False
    if meta.get("sources") != sources:
        return False
    return True


def _fit_full_harmonizer(ahba_raw: pd.DataFrame, gtex_raw: pd.DataFrame, genes: List[str], combat_use_covariates: bool):
    hcfg = SimpleNamespace(combat_use_covariates=bool(combat_use_covariates))
    harm = fit_harmonizer(ahba_raw, gtex_raw, genes, method="combat", cfg=hcfg)
    ahba_h = harm.transform(ahba_raw, "AHBA")
    gtex_h = harm.transform(gtex_raw, "GTEX")
    return harm, ahba_h, gtex_h


def _subject_full_obs_mats(gtex_h: pd.DataFrame, gtex_raw: pd.DataFrame, subject: str, genes: List[str], agg: str):
    sub_h = gtex_h[gtex_h["subject"].astype(str) == str(subject)].copy()
    sub_raw = gtex_raw[gtex_raw["subject"].astype(str) == str(subject)].copy()
    return build_subject_observed_matrices(sub_h, sub_raw, genes, agg=agg)


def _agg_vector(df: pd.DataFrame, genes: List[str], agg: str) -> np.ndarray:
    x = df[genes].to_numpy(dtype=np.float64)
    if agg == "mean":
        return np.nanmean(x, axis=0)
    if agg == "median":
        return np.nanmedian(x, axis=0)
    raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")


def _subject_inverse_df(gtex_raw: pd.DataFrame, subject: str, n_rows: int) -> pd.DataFrame:
    sub = gtex_raw[gtex_raw["subject"].astype(str) == str(subject)].copy().reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"No GTEx rows found for subject={subject}")
    row = sub.iloc[[0]][["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]].copy()
    return pd.concat([row] * int(n_rows), ignore_index=True)


def _inverse_to_subject_raw(harm, x_h: np.ndarray, inverse_df: pd.DataFrame, subject: str) -> np.ndarray:
    x_h_arr = np.asarray(x_h, dtype=np.float64)
    if x_h_arr.ndim == 1:
        x_h_arr = x_h_arr[None, :]
    subj_ids = np.asarray([subject] * x_h_arr.shape[0], dtype=object)
    try:
        return np.asarray(harm.inverse_gtex(x_h_arr, subject_ids=subj_ids, sample_df=inverse_df), dtype=np.float64)
    except TypeError:
        return np.asarray(harm.inverse_gtex(x_h_arr, subject_ids=subj_ids), dtype=np.float64)


def _full_model_fallback(
    model_name: str,
    cfg: SubjectCacheConfig,
    harm,
    ahba_h_full: np.ndarray,
    gtex_h: pd.DataFrame,
    gtex_raw: pd.DataFrame,
    target_meta: pd.DataFrame,
    coords_full: np.ndarray,
    genes: List[str],
    subject: str,
    inverse_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    obs_idx, xh, xr = _subject_full_obs_mats(gtex_h, gtex_raw, subject, genes, str(cfg.atlas_agg).lower())
    n_parcels = int(len(target_meta))
    n_genes = int(len(genes))
    if model_name == "naive":
        sub_mat = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
        pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}
        for p in obs_idx.tolist():
            sub_mat[int(p), :] = xh[pos[int(p)], :]
        x_full_h = np.where(np.isfinite(sub_mat), sub_mat, ahba_h_full).astype(np.float64)
        x_full_raw = _inverse_to_subject_raw(harm, x_full_h, inverse_df, subject)
        x_full_raw[obs_idx, :] = xr
        return x_full_h, x_full_raw
    if model_name == "dlam":
        y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
        ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=int(cfg.n_comp_target), adaptive=True)
        pred, _ = run_subject(
            {
                "subject": str(subject),
                "obs_idx": obs_idx,
                "X_obs_h": xh,
                "X_obs_raw": xr,
                "coords_full": coords_full,
                "target_meta": target_meta,
                "inverse_df": inverse_df,
            },
            {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_pls["T"]},
            {
                "harmonizer": harm,
                "basis_model": "affine_gl3",
                "strategy": "constrained_anchor",
                "spatial_method": "rbf",
                "n_comp_target": int(cfg.n_comp_target),
                "ridge_alpha_bridge": float(cfg.ridge_alpha_bridge),
                "rbf_smoothing": float(cfg.rbf_smoothing),
                "gp_rbf_length": float(cfg.gp_length_scale),
                "seed": int(cfg.seed),
                "c_min": int(cfg.c_min),
                "distance_d0": 45.0,
                "distance_tau": 10.0,
                "uncertainty_shrink": False,
            },
            asdict(cfg),
        )
        return np.asarray(pred["X_full_h"], dtype=np.float64), np.asarray(pred["X_full_raw"], dtype=np.float64)
    if model_name == "plam":
        if bool(cfg.dynamic_rank):
            k_use = int(min(max(1, int(cfg.plam_latent_dim_max)), max(1, int(len(obs_idx)))))
        else:
            k_use = int(cfg.latent_dim)
        ucfg = UnifiedGenerativeConfig(
            latent_dim=int(k_use),
            max_iters=int(cfg.plam_max_iters),
            lambda_w=float(cfg.lambda_w),
            lambda_z=float(cfg.lambda_z),
            lambda_cal_a=float(cfg.lambda_cal_a),
            lambda_cal_b=float(cfg.lambda_cal_b),
            gp_length_scale=float(cfg.gp_length_scale),
            gp_noise=float(cfg.gp_noise),
            robust_loss=str(cfg.robust_loss),
            heteroscedastic=bool(cfg.heteroscedastic),
            calibration_mode=str(cfg.calibration_mode),
            uncertainty_shrink=bool(cfg.uncertainty_shrink),
            random_state=int(cfg.seed),
        )
        atlas_model = fit_global_atlas_unified(ahba_h_full, coords_full, ucfg)
        res = infer_subject_unified({"obs_idx": obs_idx, "X_obs_h": xh}, atlas_model, ucfg)
        xhat_h = np.asarray(res["x_hat_h_full"], dtype=np.float64)
        xhat_h[obs_idx, :] = xh
        xhat_raw = _inverse_to_subject_raw(harm, xhat_h, inverse_df, subject)
        xhat_raw[obs_idx, :] = xr
        return xhat_h, xhat_raw
    raise ValueError(f"Unknown model_name={model_name}")


def process_subject_model(cfg: SubjectCacheConfig, subject: str, model_name: str) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model={model_name}; expected one of {MODEL_NAMES}")
    bundle = load_dataset(cfg)
    subject = str(subject)
    if subject not in set(bundle["eligible_subjects"]):
        raise ValueError(f"Subject {subject} is not eligible under current config")

    csv_path = bundle["csv_path"]
    hvg_path = bundle["hvg_path"]
    sources = {"csv": _source_signature(csv_path), "hvg": _source_signature(hvg_path)}
    npz_path = _npz_path(cfg, model_name, subject)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(cfg.use_cache) and _cache_valid(npz_path, cfg, subject, model_name, sources):
        print(f"[cache-hit] {model_name} {subject} -> {npz_path}")
        return npz_path

    ahba_raw = bundle["ahba_raw"]
    gtex_raw = bundle["gtex_raw"]
    genes = bundle["genes"]
    target_meta = bundle["target_meta"]
    coords_full = bundle["coords_full"]
    atlas_agg = str(cfg.atlas_agg).lower()
    n_parcels = int(len(target_meta))
    n_genes = int(len(genes))
    inverse_df_full = _subject_inverse_df(gtex_raw, subject, n_parcels)

    harm_full, ahba_h_df, gtex_h_df = _fit_full_harmonizer(ahba_raw, gtex_raw, genes, cfg.combat_use_covariates)
    ahba_h_full, _ = build_region_matrix(ahba_h_df, genes, target_meta, agg=atlas_agg)

    obs_idx_all = np.sort(gtex_raw[gtex_raw["subject"].astype(str) == subject]["parcel_idx"].astype(np.int32).unique())
    global_obs_idx = np.sort(gtex_raw["parcel_idx"].astype(np.int32).unique())
    gtex_mask = np.zeros(n_parcels, dtype=bool)
    gtex_mask[global_obs_idx] = True
    coverage_tier = "ge_cmin" if int(len(obs_idx_all)) >= int(cfg.c_min) else "lt_cmin"

    fullfit_subject_h, fullfit_subject_raw = _full_model_fallback(
        model_name=model_name,
        cfg=cfg,
        harm=harm_full,
        ahba_h_full=ahba_h_full,
        gtex_h=gtex_h_df,
        gtex_raw=gtex_raw,
        target_meta=target_meta,
        coords_full=coords_full,
        genes=genes,
        subject=subject,
        inverse_df=inverse_df_full,
    )

    pred_loro = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    truth_loro = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    pred_loro_raw = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    truth_loro_raw = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
    loro_eval_mask = np.zeros(n_parcels, dtype=bool)
    skipped_holds: List[int] = []
    plam_fold_latent_dim = np.full(n_parcels, -1, dtype=np.int32)

    for fold_id, hold in enumerate(obs_idx_all.tolist()):
        hold = int(hold)
        train_mask = ~((gtex_raw["subject"].astype(str) == subject) & (gtex_raw["parcel_idx"] == hold))
        gtex_train = gtex_raw[train_mask].copy()
        hcfg = SimpleNamespace(combat_use_covariates=bool(cfg.combat_use_covariates))
        harm = fit_harmonizer(ahba_raw, gtex_train, genes, method="combat", cfg=hcfg)
        ahba_h = harm.transform(ahba_raw, "AHBA")
        gtex_h = harm.transform(gtex_train, "GTEX")
        ahba_h_mat, _ = build_region_matrix(ahba_h, genes, target_meta, agg=atlas_agg)

        hold_raw = gtex_raw[(gtex_raw["subject"].astype(str) == subject) & (gtex_raw["parcel_idx"] == hold)].copy()
        if len(hold_raw) == 0:
            skipped_holds.append(hold)
            continue
        hold_h = harm.transform(hold_raw, "GTEX")
        truth = _agg_vector(hold_h, genes, atlas_agg)
        truth_raw = _agg_vector(hold_raw, genes, atlas_agg)
        inverse_df_fold_full = _subject_inverse_df(gtex_raw, subject, n_parcels)
        inverse_df_fold_one = inverse_df_fold_full.iloc[[0]].copy()

        if model_name == "naive":
            pred = ahba_h_mat[hold, :]
            pred_raw = _inverse_to_subject_raw(harm, pred, inverse_df_fold_one, subject).reshape(-1)
        elif model_name == "dlam":
            y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
            ahba_pls = fit_subject_pls(ahba_h_mat, y_full, n_comp_target=int(cfg.n_comp_target), adaptive=True)
            subj_h = gtex_h[gtex_h["subject"].astype(str) == subject].copy()
            subj_raw = gtex_train[gtex_train["subject"].astype(str) == subject].copy()
            obs_idx, xh, xr = build_subject_observed_matrices(subj_h, subj_raw, genes, agg=atlas_agg)
            if len(obs_idx) < 2:
                skipped_holds.append(hold)
                continue
            pred_full, _ = run_subject(
                {
                    "subject": subject,
                    "obs_idx": obs_idx,
                    "X_obs_h": xh,
                    "X_obs_raw": xr,
                    "coords_full": coords_full,
                    "target_meta": target_meta,
                    "inverse_df": inverse_df_fold_full,
                },
                {"ahba_h_full": ahba_h_mat, "ahba_ref_T": ahba_pls["T"]},
                {
                    "harmonizer": harm,
                    "basis_model": "affine_gl3",
                    "strategy": "constrained_anchor",
                    "spatial_method": "rbf",
                    "n_comp_target": int(cfg.n_comp_target),
                    "ridge_alpha_bridge": float(cfg.ridge_alpha_bridge),
                    "rbf_smoothing": float(cfg.rbf_smoothing),
                    "gp_rbf_length": float(cfg.gp_length_scale),
                    "seed": int(cfg.seed),
                    "c_min": int(cfg.c_min),
                    "distance_d0": 45.0,
                    "distance_tau": 10.0,
                    "uncertainty_shrink": False,
                },
                asdict(cfg),
                fold_mask=hold,
            )
            pred = np.asarray(pred_full["X_full_h"], dtype=np.float64)[hold, :]
            pred_raw = np.asarray(pred_full["X_full_raw"], dtype=np.float64)[hold, :]
        else:  # plam
            subj_h = gtex_h[gtex_h["subject"].astype(str) == subject].copy()
            subj_raw = gtex_train[gtex_train["subject"].astype(str) == subject].copy()
            obs_idx, xh, _ = build_subject_observed_matrices(subj_h, subj_raw, genes, agg=atlas_agg)
            if len(obs_idx) < 2:
                skipped_holds.append(hold)
                continue
            if bool(cfg.dynamic_rank):
                k_use = int(min(max(1, int(cfg.plam_latent_dim_max)), max(1, int(len(obs_idx)))))
            else:
                k_use = int(cfg.latent_dim)
            ucfg = UnifiedGenerativeConfig(
                latent_dim=int(k_use),
                max_iters=int(cfg.plam_max_iters),
                lambda_w=float(cfg.lambda_w),
                lambda_z=float(cfg.lambda_z),
                lambda_cal_a=float(cfg.lambda_cal_a),
                lambda_cal_b=float(cfg.lambda_cal_b),
                gp_length_scale=float(cfg.gp_length_scale),
                gp_noise=float(cfg.gp_noise),
                robust_loss=str(cfg.robust_loss),
                heteroscedastic=bool(cfg.heteroscedastic),
                calibration_mode=str(cfg.calibration_mode),
                uncertainty_shrink=bool(cfg.uncertainty_shrink),
                random_state=int(cfg.seed),
            )
            atlas_model = fit_global_atlas_unified(ahba_h_mat, coords_full, ucfg)
            res = infer_subject_unified(
                {"subject": subject, "obs_idx": obs_idx, "X_obs_h": xh},
                atlas_model,
                ucfg,
                fold_ctx={"prior_h": ahba_h_mat, "obs_idx": obs_idx},
            )
            pred = np.asarray(res["x_hat_h_full"], dtype=np.float64)[hold, :]
            pred_raw = _inverse_to_subject_raw(harm, pred, inverse_df_fold_one, subject).reshape(-1)
            plam_fold_latent_dim[hold] = int(k_use)

        pred_loro[hold, :] = pred
        truth_loro[hold, :] = truth
        pred_loro_raw[hold, :] = pred_raw
        truth_loro_raw[hold, :] = truth_raw
        loro_eval_mask[hold] = True
        if model_name == "plam":
            print(f"[{model_name} fold] {subject} fold={fold_id} hold={hold} k={int(plam_fold_latent_dim[hold])}")
        else:
            print(f"[{model_name} fold] {subject} fold={fold_id} hold={hold}")

    loro_fused_subject_h = fullfit_subject_h.copy()
    loro_fused_subject_h[loro_eval_mask, :] = pred_loro[loro_eval_mask, :]
    loro_fused_subject_raw = fullfit_subject_raw.copy()
    loro_fused_subject_raw[loro_eval_mask, :] = pred_loro_raw[loro_eval_mask, :]
    imputed_mask = ~loro_eval_mask

    np.savez_compressed(
        npz_path,
        subject_id=np.asarray([subject], dtype=object),
        model_name=np.asarray([model_name], dtype=object),
        gene_names=np.asarray(genes, dtype=object),
        parcel_idx=np.arange(n_parcels, dtype=np.int32),
        fullfit_subject_h=fullfit_subject_h.astype(np.float32),
        fullfit_subject_raw=fullfit_subject_raw.astype(np.float32),
        loro_fused_subject_h=loro_fused_subject_h.astype(np.float32),
        loro_fused_subject_raw=loro_fused_subject_raw.astype(np.float32),
        loro_truth_subject_h=truth_loro.astype(np.float32),
        loro_truth_subject_raw=truth_loro_raw.astype(np.float32),
        gtex_mask=gtex_mask.astype(np.int8),
        loro_eval_mask=loro_eval_mask.astype(np.int8),
        imputed_mask=imputed_mask.astype(np.int8),
        skipped_holds=np.asarray(sorted(set(skipped_holds)), dtype=np.int32),
        plam_fold_latent_dim=plam_fold_latent_dim.astype(np.int32),
    )
    meta = {
        "status": "ok",
        "subject": subject,
        "model_name": model_name,
        "gene_scope": str(cfg.gene_scope).lower(),
        "cfg_hash": _cfg_hash(cfg, subject, model_name),
        "sources": sources,
        "n_genes": int(n_genes),
        "n_parcels": int(n_parcels),
        "n_gtex_observed_global": int(gtex_mask.sum()),
        "n_gtex_observed_subject": int(len(obs_idx_all)),
        "n_loro_eval": int(loro_eval_mask.sum()),
        "coverage_tier": coverage_tier,
        "dynamic_rank": bool(cfg.dynamic_rank),
        "plam_latent_dim_max": int(cfg.plam_latent_dim_max),
        "timestamp": io_utils.utc_timestamp(),
        "config": asdict(cfg),
    }
    _meta_path(npz_path).write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[cache-write] {model_name} {subject} -> {npz_path}")
    return npz_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build subject-wise LORO NPZ cache for one subject.")
    p.add_argument("--subject", required=True)
    p.add_argument("--model", choices=list(MODEL_NAMES) + ["all"], default="all")
    p.add_argument("--csv-path", default=SubjectCacheConfig.csv_path)
    p.add_argument("--hvg-path", default=SubjectCacheConfig.hvg_path)
    p.add_argument("--out-root", default=SubjectCacheConfig.out_root)
    p.add_argument("--gene-scope", choices=["allgenes", "hvg"], default=SubjectCacheConfig.gene_scope)
    p.add_argument("--use-cache", default=str(SubjectCacheConfig.use_cache).lower())
    p.add_argument("--min-observed-parcels", type=int, default=SubjectCacheConfig.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=SubjectCacheConfig.c_min)
    p.add_argument("--n-comp-target", type=int, default=SubjectCacheConfig.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=SubjectCacheConfig.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=SubjectCacheConfig.rbf_smoothing)
    p.add_argument("--gp-length-scale", type=float, default=SubjectCacheConfig.gp_length_scale)
    p.add_argument("--gp-noise", type=float, default=SubjectCacheConfig.gp_noise)
    p.add_argument("--seed", type=int, default=SubjectCacheConfig.seed)
    p.add_argument("--combat-use-covariates", default=str(SubjectCacheConfig.combat_use_covariates).lower())
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
    )


def main() -> None:
    a = parse_args()
    cfg = _cfg_from_args(a)
    models = list(MODEL_NAMES) if str(a.model) == "all" else [str(a.model)]
    for m in models:
        process_subject_model(cfg, subject=str(a.subject), model_name=m)


if __name__ == "__main__":
    main()
