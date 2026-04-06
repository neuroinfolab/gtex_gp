from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

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

META_COLS = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]
DLAM_COMBO = "combat__affine_gl3__constrained_anchor__rbf"
PLAM_COMBO = "unified__harm=combat__cal=hier_affine_map__robust=student_t__hetero=gene_var__uncshrink=false"
MODEL_LABELS = {
    "naive": "Naive fill",
    "dlam": "DLAM",
    "plam": "PLAM",
}


@dataclass
class DatasetBundle:
    csv_path: str
    hvg_path: str
    out_root: str
    genes_all: List[str]
    genes_hvg: List[str]
    ahba_raw: pd.DataFrame
    gtex_raw: pd.DataFrame
    target_meta: pd.DataFrame
    coords_full: np.ndarray
    eligibility: pd.DataFrame
    eligible_subjects: List[str]
    global_obs_mask: np.ndarray
    counts: Dict[str, int]
    source_signatures: Dict[str, Dict[str, Any]]


@dataclass
class ModelRunResult:
    model_name: str
    stage_name: str
    cache_dir: str
    manifest_path: str
    outputs: Dict[str, str]
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    model_name: str
    asset_root: str
    tables: Dict[str, str]
    figures: Dict[str, str]
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepresentativeSubject:
    subject: str
    n_obs_parcels: int
    selection_rule: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetManifest:
    asset_root: str
    figure_paths: List[str]
    table_paths: List[str]
    manuscript_fig_dir: str
    manuscript_table_dir: str


@dataclass
class WriteupRunManifest:
    config_hash: str
    timestamp: str
    csv_path: str
    hvg_path: str
    out_root: str
    n_subjects_eligible: int
    n_genes_all: int
    n_genes_hvg: int
    representative_subject: str
    stages: Dict[str, Any]
    figures: List[str]
    tables: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_optional_path(path_str: str, fallback_names: Iterable[str]) -> Path:
    path = Path(path_str)
    if path.exists():
        return path.resolve()
    repo = _repo_root()
    for name in fallback_names:
        cand = repo / name
        if cand.exists():
            return cand.resolve()
    return path.resolve()


def _source_signature(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _safe_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, Mapping):
            out[k] = _safe_cfg(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = str(v)
    return out


def _cfg_hash(cfg: Mapping[str, Any]) -> str:
    return io_utils.hash_config(_safe_cfg(cfg))


def _stage_paths(out_root: Path, stage: str) -> Dict[str, Path]:
    cache_dir = out_root / "caches" / stage
    manifest_path = out_root / "manifests" / f"{stage}.json"
    status_path = cache_dir / "status.json"
    return {"cache_dir": cache_dir, "manifest_path": manifest_path, "status_path": status_path}


def _outputs_exist(outputs: Mapping[str, str]) -> bool:
    for p in outputs.values():
        if not Path(p).exists():
            return False
    return True


def _stage_valid(out_root: Path, stage: str, cfg_hash: str, sources: Mapping[str, Any]) -> bool:
    paths = _stage_paths(out_root, stage)
    manifest_path = paths["manifest_path"]
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if manifest.get("status") != "ok":
        return False
    if manifest.get("config_hash") != cfg_hash:
        return False
    if manifest.get("sources") != sources:
        return False
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return False
    return _outputs_exist(outputs)


def _write_stage_manifest(out_root: Path, stage: str, cfg_hash: str, sources: Mapping[str, Any], outputs: Mapping[str, str], summary: Mapping[str, Any] | None = None) -> None:
    paths = _stage_paths(out_root, stage)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifest_path"].parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": "ok",
        "config_hash": cfg_hash,
        "sources": sources,
        "outputs": dict(outputs),
        "summary": dict(summary or {}),
        "timestamp": io_utils.utc_timestamp(),
    }
    io_utils.dump_json(paths["manifest_path"], payload)
    io_utils.dump_json(paths["status_path"], {"stage": stage, "status": "ok", "timestamp": payload["timestamp"]})


def _load_stage_manifest(out_root: Path, stage: str) -> Dict[str, Any]:
    return json.loads(_stage_paths(out_root, stage)["manifest_path"].read_text())


def _force_set(cfg: Mapping[str, Any]) -> set[str]:
    return set(str(x) for x in cfg.get("force_rerun_stages", []))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_coord(x: float, y: float, z: float) -> str:
    return f"[({float(x):.6f}, {float(y):.6f}, {float(z):.6f})]"


def _matrix_to_csv(path: Path, mat: np.ndarray, genes: List[str], target_meta: pd.DataFrame) -> None:
    df = pd.DataFrame(mat, columns=genes)
    df.insert(0, "parcel_idx", target_meta["parcel_idx"].to_numpy(dtype=np.int32))
    _ensure_parent(path)
    df.to_csv(path, index=False)


def _shared_harmonized(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    hcfg = SimpleNamespace(combat_use_covariates=bool(cfg.get("combat_use_covariates", True)))
    gtex_eligible = bundle.gtex_raw[bundle.gtex_raw["subject"].astype(str).isin(bundle.eligible_subjects)].copy()
    harmonizer = fit_harmonizer(bundle.ahba_raw, gtex_eligible, bundle.genes_hvg, method="combat", cfg=hcfg)
    ahba_h = harmonizer.transform(bundle.ahba_raw, "AHBA")
    gtex_h = harmonizer.transform(gtex_eligible, "GTEX")
    ahba_raw_mat, _ = build_region_matrix(bundle.ahba_raw, bundle.genes_hvg, bundle.target_meta, agg="mean")
    ahba_h_mat, _ = build_region_matrix(ahba_h, bundle.genes_hvg, bundle.target_meta, agg="mean")
    gtex_sparse_raw = np.full_like(ahba_raw_mat, np.nan, dtype=np.float64)
    gtex_sparse_h = np.full_like(ahba_h_mat, np.nan, dtype=np.float64)
    grp_raw = gtex_eligible.groupby("parcel_idx")[bundle.genes_hvg].mean()
    grp_h = gtex_h.groupby("parcel_idx")[bundle.genes_hvg].mean()
    for idx in grp_raw.index.tolist():
        p = int(idx)
        gtex_sparse_raw[p, :] = grp_raw.loc[idx, bundle.genes_hvg].to_numpy(dtype=np.float64)
    for idx in grp_h.index.tolist():
        p = int(idx)
        gtex_sparse_h[p, :] = grp_h.loc[idx, bundle.genes_hvg].to_numpy(dtype=np.float64)
    gtex_sparse_raw[~bundle.global_obs_mask, :] = np.nan
    gtex_sparse_h[~bundle.global_obs_mask, :] = np.nan
    y_full = np.c_[bundle.coords_full[:, 1], bundle.coords_full[:, 2], np.abs(bundle.coords_full[:, 0])]
    return {
        "harmonizer": harmonizer,
        "ahba_h": ahba_h,
        "gtex_h": gtex_h,
        "gtex_eligible": gtex_eligible,
        "ahba_raw_mat": ahba_raw_mat,
        "ahba_h_mat": ahba_h_mat,
        "gtex_sparse_raw": gtex_sparse_raw,
        "gtex_sparse_h": gtex_sparse_h,
        "y_full": y_full,
    }


def load_dataset_bundle(csv_path: str, hvg_path: str, cfg: Mapping[str, Any]) -> DatasetBundle:
    csv_resolved = _resolve_optional_path(csv_path, ["gxp_samples.csv"])
    hvg_resolved = _resolve_optional_path(hvg_path, ["data/raw/ahba_100hvg.txt", "ahba_100hvg.txt"])
    out_root = Path(str(cfg.get("out_root", _repo_root() / "out" / "notebook_writeup"))).resolve()
    stage = "bundle"
    stage_sources = {
        "csv": _source_signature(csv_resolved),
        "hvg": _source_signature(hvg_resolved),
    }
    cfg_hash = _cfg_hash({k: v for k, v in _safe_cfg(cfg).items() if k not in {"force_rerun_stages"}})
    paths = _stage_paths(out_root, stage)
    bundle_csv = paths["cache_dir"] / "subject_eligibility.csv"
    header = io_utils.load_gene_header_and_hvg(csv_resolved, hvg_resolved)
    df = io_utils.read_expression_subset(
        csv_resolved,
        header["genes_hvg"],
        rep_mode=str(cfg.get("gtex_rep_mode", "medoid")).lower(),
        hemi_mode=str(cfg.get("gtex_hemi_mode", "native")).lower(),
    )
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = build_target_parcels(ahba_raw)
    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)
    eligibility = build_subject_eligibility(gtex_raw, min_observed_parcels=int(cfg.get("min_observed_parcels", 5)))
    eligible = eligibility[eligibility["eligible"]].copy()
    smoke_subjects = int(cfg.get("smoke_subjects", 0) or 0)
    if smoke_subjects > 0:
        eligible = eligible.sort_values("subject").head(smoke_subjects).reset_index(drop=True)
    eligible_subjects = eligible["subject"].astype(str).tolist()
    global_obs_mask = np.zeros(len(target_meta), dtype=bool)
    obs_idx = np.sort(gtex_raw[gtex_raw["subject"].astype(str).isin(eligible_subjects)]["parcel_idx"].astype(np.int32).unique())
    global_obs_mask[obs_idx] = True
    counts = {
        "n_genes_all": int(len(header["genes_all"])),
        "n_genes_hvg": int(len(header["genes_hvg"])),
        "n_ahba_rows": int(len(ahba_raw)),
        "n_gtex_rows": int(len(gtex_raw)),
        "n_subjects_total": int(gtex_raw["subject"].astype(str).nunique()),
        "n_subjects_eligible": int(len(eligible_subjects)),
        "n_parcels": int(len(target_meta)),
    }
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(bundle_csv, index=False)
    _write_stage_manifest(
        out_root,
        stage,
        cfg_hash,
        stage_sources,
        {"subject_eligibility_csv": str(bundle_csv)},
        counts,
    )
    return DatasetBundle(
        csv_path=str(csv_resolved),
        hvg_path=str(hvg_resolved),
        out_root=str(out_root),
        genes_all=header["genes_all"],
        genes_hvg=header["genes_hvg"],
        ahba_raw=ahba_raw,
        gtex_raw=gtex_raw,
        target_meta=target_meta,
        coords_full=target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64),
        eligibility=eligibility,
        eligible_subjects=eligible_subjects,
        global_obs_mask=global_obs_mask,
        counts=counts,
        source_signatures=stage_sources,
    )


def summarize_shared_harmonization(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    shared = _shared_harmonized(bundle, cfg)
    out_root = Path(bundle.out_root)
    stage = "harmonization"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    npz_path = paths["cache_dir"] / "shared_hvg_matrices.npz"
    _ensure_parent(npz_path)
    np.savez_compressed(
        npz_path,
        ahba_raw_mat=shared["ahba_raw_mat"],
        ahba_h_mat=shared["ahba_h_mat"],
        gtex_sparse_raw=shared["gtex_sparse_raw"],
        gtex_sparse_h=shared["gtex_sparse_h"],
        global_obs_mask=bundle.global_obs_mask.astype(np.int8),
    )
    summary = {
        "n_subjects_eligible": bundle.counts["n_subjects_eligible"],
        "n_genes_hvg": bundle.counts["n_genes_hvg"],
        "combat_use_covariates": bool(cfg.get("combat_use_covariates", True)),
    }
    _write_stage_manifest(out_root, stage, cfg_hash, sources, {"shared_matrices_npz": str(npz_path)}, summary)
    return {"manifest_path": str(_stage_paths(out_root, stage)["manifest_path"]), "summary": summary, "outputs": {"shared_matrices_npz": str(npz_path)}}


def _subject_metadata(gtex_raw: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    grp = gtex_raw.groupby("subject", dropna=False)
    for sid, sub in grp:
        row = sub.iloc[0]
        meta[str(sid)] = {
            "age": str(row.get("age", "")),
            "sex": str(row.get("sex", "")),
        }
    return meta


def _save_subject_predictions(cache_dir: Path, subject_ids: List[str], predictions_h: np.ndarray, observed_masks: np.ndarray) -> Dict[str, str]:
    npz_path = cache_dir / "subject_predictions_hvg.npz"
    np.savez_compressed(
        npz_path,
        subject_ids=np.asarray(subject_ids, dtype=object),
        predictions_h=predictions_h.astype(np.float32),
        observed_masks=observed_masks.astype(np.int8),
    )
    return {"subject_predictions_npz": str(npz_path)}


def run_naive_fill(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> ModelRunResult:
    out_root = Path(bundle.out_root)
    stage = "naive"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    outputs = {
        "atlas_mean_harmonized_csv": str(paths["cache_dir"] / "naive_atlas_mean_harmonized.csv"),
        "subject_predictions_npz": str(paths["cache_dir"] / "subject_predictions_hvg.npz"),
    }
    if stage not in _force_set(cfg) and _stage_valid(out_root, stage, cfg_hash, sources):
        manifest = _load_stage_manifest(out_root, stage)
        return ModelRunResult(model_name="naive", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=manifest["outputs"], summary=manifest.get("summary", {}))

    shared = _shared_harmonized(bundle, cfg)
    subject_ids = bundle.eligible_subjects
    n_subj = len(subject_ids)
    n_parcels = len(bundle.target_meta)
    n_genes = len(bundle.genes_hvg)
    preds = np.zeros((n_subj, n_parcels, n_genes), dtype=np.float32)
    obs_masks = np.zeros((n_subj, n_parcels), dtype=bool)
    for i, sid in enumerate(subject_ids):
        sub_h = shared["gtex_h"][shared["gtex_h"]["subject"] == sid].copy()
        grp = sub_h.groupby("parcel_idx")[bundle.genes_hvg].mean()
        sub_mat = np.full((n_parcels, n_genes), np.nan, dtype=np.float64)
        for idx in grp.index.tolist():
            p = int(idx)
            sub_mat[p, :] = grp.loc[idx, bundle.genes_hvg].to_numpy(dtype=np.float64)
            obs_masks[i, p] = True
        preds[i] = np.where(np.isfinite(sub_mat), sub_mat, shared["ahba_h_mat"]).astype(np.float32)

    atlas_mean = preds.mean(axis=0).astype(np.float64)
    _matrix_to_csv(Path(outputs["atlas_mean_harmonized_csv"]), atlas_mean, bundle.genes_hvg, bundle.target_meta)
    outputs.update(_save_subject_predictions(paths["cache_dir"], subject_ids, preds, obs_masks))
    summary = {"n_subjects": n_subj, "n_parcels": n_parcels, "n_genes_hvg": n_genes}
    _write_stage_manifest(out_root, stage, cfg_hash, sources, outputs, summary)
    return ModelRunResult(model_name="naive", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=outputs, summary=summary)


def run_dlam(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> ModelRunResult:
    out_root = Path(bundle.out_root)
    stage = "dlam"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    compat_root = paths["cache_dir"] / "compat_root"
    outputs = {
        "atlas_mean_harmonized_csv": str(compat_root / "tables" / f"aggregate_allgenes_{DLAM_COMBO}_mean_harmonized.csv"),
        "subject_predictions_npz": str(paths["cache_dir"] / "subject_predictions_hvg.npz"),
    }
    if stage not in _force_set(cfg) and _stage_valid(out_root, stage, cfg_hash, sources):
        manifest = _load_stage_manifest(out_root, stage)
        return ModelRunResult(model_name="dlam", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=manifest["outputs"], summary=manifest.get("summary", {}))

    shared = _shared_harmonized(bundle, cfg)
    ahba_h_full = shared["ahba_h_mat"]
    ahba_pls = fit_subject_pls(ahba_h_full, shared["y_full"], n_comp_target=int(cfg.get("n_comp_target", 3)), adaptive=True)
    atlas_bundle = {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_pls["T"]}
    method_bundle = {
        "harmonizer": shared["harmonizer"],
        "basis_model": str(cfg.get("dlam_basis_model", "affine_gl3")),
        "strategy": str(cfg.get("dlam_strategy", "constrained_anchor")),
        "spatial_method": str(cfg.get("dlam_spatial_method", "rbf")),
        "n_comp_target": int(cfg.get("n_comp_target", 3)),
        "ridge_alpha_bridge": float(cfg.get("ridge_alpha_bridge", 1e-2)),
        "rbf_smoothing": float(cfg.get("rbf_smoothing", 0.10)),
        "gp_rbf_length": float(cfg.get("gp_length_scale", 25.0)),
        "seed": int(cfg.get("seed", 123)),
        "c_min": int(cfg.get("c_min", 8)),
        "distance_d0": 45.0,
        "distance_tau": 10.0,
        "uncertainty_shrink": False,
    }
    subject_ids = bundle.eligible_subjects
    n_subj = len(subject_ids)
    n_parcels = len(bundle.target_meta)
    n_genes = len(bundle.genes_hvg)
    preds = np.zeros((n_subj, n_parcels, n_genes), dtype=np.float32)
    obs_masks = np.zeros((n_subj, n_parcels), dtype=bool)
    for i, sid in enumerate(subject_ids):
        subj_h = shared["gtex_h"][shared["gtex_h"]["subject"] == sid].copy()
        subj_raw = shared["gtex_eligible"][shared["gtex_eligible"]["subject"] == sid].copy()
        obs_idx, xh, xr = build_subject_observed_matrices(subj_h, subj_raw, bundle.genes_hvg)
        obs_masks[i, obs_idx] = True
        pred, _ = run_subject(
            {
                "subject": sid,
                "obs_idx": obs_idx,
                "X_obs_h": xh,
                "X_obs_raw": xr,
                "coords_full": bundle.coords_full,
                "target_meta": bundle.target_meta,
            },
            atlas_bundle,
            method_bundle,
            dict(cfg),
        )
        preds[i] = np.asarray(pred["X_full_h"], dtype=np.float32)
    atlas_mean = preds.mean(axis=0).astype(np.float64)
    _matrix_to_csv(Path(outputs["atlas_mean_harmonized_csv"]), atlas_mean, bundle.genes_hvg, bundle.target_meta)
    outputs.update(_save_subject_predictions(paths["cache_dir"], subject_ids, preds, obs_masks))
    summary = {"n_subjects": n_subj, "n_parcels": n_parcels, "n_genes_hvg": n_genes, "compat_root": str(compat_root)}
    _write_stage_manifest(out_root, stage, cfg_hash, sources, outputs, summary)
    return ModelRunResult(model_name="dlam", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=outputs, summary=summary)


def _build_plam_combined_export(bundle: DatasetBundle, shared: Dict[str, Any], subject_ids: List[str], predictions: np.ndarray, obs_masks: np.ndarray, export_path: Path) -> None:
    subject_meta = _subject_metadata(shared["gtex_eligible"])
    rows: List[Dict[str, Any]] = []
    # AHBA rows in harmonized space.
    ahba_h = shared["ahba_h"].copy()
    for row in ahba_h.itertuples(index=False):
        rec = {col: getattr(row, col) for col in META_COLS}
        rec["dataset"] = "AHBA"
        rec["is_imputed"] = 0
        for g in bundle.genes_hvg:
            rec[g] = float(getattr(row, g))
        rows.append(rec)
    tissue = bundle.target_meta["tissue_or_parcel"].astype(str).to_numpy()
    coords = bundle.coords_full
    for i, sid in enumerate(subject_ids):
        meta = subject_meta[str(sid)]
        pred = predictions[i]
        obs = obs_masks[i]
        for p in range(pred.shape[0]):
            rec = {
                "subject": sid,
                "age": meta["age"],
                "sex": meta["sex"],
                "dataset": "GTEX",
                "tissue_or_parcel": str(tissue[p]),
                "coordinates": _format_coord(*coords[p]),
                "is_imputed": int(not bool(obs[p])),
            }
            for gi, g in enumerate(bundle.genes_hvg):
                rec[g] = float(pred[p, gi])
            rows.append(rec)
    export_df = pd.DataFrame(rows, columns=META_COLS + ["is_imputed"] + bundle.genes_hvg)
    _ensure_parent(export_path)
    export_df.to_csv(export_path, index=False)


def run_plam(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> ModelRunResult:
    out_root = Path(bundle.out_root)
    stage = "plam"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    export_root = paths["cache_dir"] / "compat_export_root"
    outputs = {
        "combined_harmonized_csv": str(export_root / "gxp_completed_harmonized_combined.csv"),
        "subject_predictions_npz": str(paths["cache_dir"] / "subject_predictions_hvg.npz"),
        "atlas_mean_harmonized_csv": str(paths["cache_dir"] / "plam_atlas_mean_harmonized.csv"),
    }
    if stage not in _force_set(cfg) and _stage_valid(out_root, stage, cfg_hash, sources):
        manifest = _load_stage_manifest(out_root, stage)
        return ModelRunResult(model_name="plam", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=manifest["outputs"], summary=manifest.get("summary", {}))

    shared = _shared_harmonized(bundle, cfg)
    ucfg = UnifiedGenerativeConfig(
        latent_dim=int(cfg.get("latent_dim", 3)),
        max_iters=int(cfg.get("plam_max_iters", 5)),
        lambda_w=float(cfg.get("lambda_w", 1.0)),
        lambda_z=float(cfg.get("lambda_z", 1.0)),
        lambda_cal_a=float(cfg.get("lambda_cal_a", 10.0)),
        lambda_cal_b=float(cfg.get("lambda_cal_b", 10.0)),
        gp_length_scale=float(cfg.get("gp_length_scale", 25.0)),
        gp_noise=float(cfg.get("gp_noise", 1e-3)),
        robust_loss=str(cfg.get("robust_loss", "student_t")),
        heteroscedastic=bool(cfg.get("heteroscedastic", True)),
        calibration_mode=str(cfg.get("calibration_mode", "hier_affine_map")),
        uncertainty_shrink=bool(cfg.get("uncertainty_shrink", False)),
        unc_alpha=float(cfg.get("unc_alpha", 0.5)),
        unc_beta=float(cfg.get("unc_beta", 0.5)),
        unc_m0=float(cfg.get("unc_m0", 0.5)),
        unc_tau=float(cfg.get("unc_tau", 0.2)),
        random_state=int(cfg.get("seed", 123)),
    )
    atlas_model = fit_global_atlas_unified(shared["ahba_h_mat"], bundle.coords_full, ucfg)
    subject_ids = bundle.eligible_subjects
    n_subj = len(subject_ids)
    n_parcels = len(bundle.target_meta)
    n_genes = len(bundle.genes_hvg)
    preds = np.zeros((n_subj, n_parcels, n_genes), dtype=np.float32)
    obs_masks = np.zeros((n_subj, n_parcels), dtype=bool)
    for i, sid in enumerate(subject_ids):
        subj_h = shared["gtex_h"][shared["gtex_h"]["subject"] == sid].copy()
        subj_raw = shared["gtex_eligible"][shared["gtex_eligible"]["subject"] == sid].copy()
        obs_idx, xh, _ = build_subject_observed_matrices(subj_h, subj_raw, bundle.genes_hvg)
        obs_masks[i, obs_idx] = True
        res = infer_subject_unified({"obs_idx": obs_idx, "X_obs_h": xh}, atlas_model, ucfg)
        xhat = np.asarray(res["x_hat_h_full"], dtype=np.float64)
        xhat[obs_idx, :] = xh
        preds[i] = xhat.astype(np.float32)
    atlas_mean = preds.mean(axis=0).astype(np.float64)
    _matrix_to_csv(Path(outputs["atlas_mean_harmonized_csv"]), atlas_mean, bundle.genes_hvg, bundle.target_meta)
    outputs.update(_save_subject_predictions(paths["cache_dir"], subject_ids, preds, obs_masks))
    _build_plam_combined_export(bundle, shared, subject_ids, preds.astype(np.float64), obs_masks, Path(outputs["combined_harmonized_csv"]))
    summary = {"n_subjects": n_subj, "n_parcels": n_parcels, "n_genes_hvg": n_genes, "compat_root": str(export_root)}
    _write_stage_manifest(out_root, stage, cfg_hash, sources, outputs, summary)
    return ModelRunResult(model_name="plam", stage_name=stage, cache_dir=str(paths["cache_dir"]), manifest_path=str(paths["manifest_path"]), outputs=outputs, summary=summary)


def run_allgene_loro(bundle: DatasetBundle, model_name: str, model_runner: Callable[..., Any] | None, cfg: Mapping[str, Any]) -> EvaluationResult:
    del model_runner
    out_root = Path(bundle.out_root)
    stage = "allgene_loro"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    asset_root = out_root / "manuscript_assets"
    outputs = {
        "subject_metrics_csv": str(asset_root / "tables" / "loro_subject_metrics_allgene.csv"),
        "parcel_metrics_csv": str(asset_root / "tables" / "loro_parcel_metrics_allgene.csv"),
        "gene_metrics_csv": str(asset_root / "tables" / "loro_gene_metrics_allgene.csv"),
        "summary_csv": str(asset_root / "tables" / "loro_method_summary_allgene.csv"),
        "subject_fig": str(asset_root / "figures" / "loro_subject_metrics_allgene.pdf"),
        "parcel_fig": str(asset_root / "figures" / "loro_parcel_metrics_allgene.pdf"),
        "gene_fig": str(asset_root / "figures" / "loro_gene_metrics_allgene.pdf"),
        "summary_fig": str(asset_root / "figures" / "loro_method_summary_bars_allgene.pdf"),
    }
    def _model_summary_from_outputs() -> Dict[str, Any]:
        summary_df = pd.read_csv(outputs["summary_csv"])
        internal_name = {
            "dlam": "deterministic_latent_transport",
            "plam": "unified_probabilistic_latent_alignment",
            "naive": "naive_ahba_fill",
        }[model_name]
        row = summary_df[summary_df["model_name"] == internal_name].iloc[0].to_dict()
        return {
            "requested_model": model_name,
            "mean_subject_pearson": float(row["mean_subject_pearson"]),
            "mean_subject_rmse": float(row["mean_subject_rmse"]),
            "mean_delta_rmse_vs_naive": float(row["mean_delta_rmse_vs_naive"]),
            "frac_subjects_better_than_naive": float(row["frac_subjects_better_than_naive"]),
        }

    if stage not in _force_set(cfg) and _stage_valid(out_root, stage, cfg_hash, sources):
        manifest = _load_stage_manifest(out_root, stage)
        tables = {k: v for k, v in manifest["outputs"].items() if k.endswith("csv")}
        figures = {k: v for k, v in manifest["outputs"].items() if k.endswith("fig")}
        return EvaluationResult(model_name=model_name, asset_root=str(asset_root), tables=tables, figures=figures, summary=_model_summary_from_outputs())

    cmd = [
        "python3",
        str(_repo_root() / "scripts" / "build_dual_model_loro_metric_panels.py"),
        "--csv-path",
        bundle.csv_path,
        "--hvg-path",
        bundle.hvg_path,
        "--out-root",
        str(asset_root),
        "--gene-scope",
        "allgenes",
        "--chunk-size",
        str(int(cfg.get("chunk_size", 500))),
        "--seed",
        str(int(cfg.get("seed", 123))),
        "--smoke-subjects",
        str(int(cfg.get("smoke_subjects", 0))),
    ]
    subprocess.run(cmd, check=True, cwd=str(_repo_root()))
    summary_df = pd.read_csv(outputs["summary_csv"])
    subj_df = pd.read_csv(outputs["subject_metrics_csv"])

    # Compatibility root for PLAM manuscript builder.
    compat_root = out_root / "caches" / "plam_phase2_compat"
    compat_csv = compat_root / "tables" / "allgenes_subject_loro_summary.csv"
    compat_root.joinpath("tables").mkdir(parents=True, exist_ok=True)
    plam_rows = subj_df[subj_df["model_name"] == "unified_probabilistic_latent_alignment"].copy()
    plam_rows = plam_rows.rename(columns={"baseline_rmse": "baseline_rmse"})
    keep_cols = ["subject", "n_obs_parcels", "pearson_r", "rmse", "baseline_rmse", "coverage_tier"]
    plam_rows[keep_cols].to_csv(compat_csv, index=False)
    outputs["plam_compat_subject_summary_csv"] = str(compat_csv)

    summary = {
        "available_models": summary_df["model_name"].astype(str).tolist(),
        "n_subject_rows": int(len(subj_df)),
    }
    _write_stage_manifest(out_root, stage, cfg_hash, sources, outputs, summary)
    tables = {k: v for k, v in outputs.items() if k.endswith("csv")}
    figures = {k: v for k, v in outputs.items() if k.endswith("fig")}
    return EvaluationResult(model_name=model_name, asset_root=str(asset_root), tables=tables, figures=figures, summary=_model_summary_from_outputs())


def select_representative_subject(eval_results: Mapping[str, EvaluationResult], cfg: Mapping[str, Any]) -> RepresentativeSubject:
    out_root = Path(str(cfg.get("out_root", _repo_root() / "out" / "notebook_writeup"))).resolve()
    subj_csv = Path(eval_results["plam"].tables["subject_metrics_csv"])
    df = pd.read_csv(subj_csv)
    use = df[df["model_name"] == "unified_probabilistic_latent_alignment"].copy()
    c_min = int(cfg.get("c_min", 8))
    use = use[use["n_obs_parcels"] >= c_min].copy()
    med_p = float(use["pearson_r"].median())
    med_r = float(use["rmse"].median())
    use["abs_delta_pearson"] = np.abs(use["pearson_r"] - med_p)
    use["abs_delta_rmse"] = np.abs(use["rmse"] - med_r)
    use = use.sort_values(["abs_delta_pearson", "abs_delta_rmse", "subject"], ascending=[True, True, True]).reset_index(drop=True)
    row = use.iloc[0]
    rep = RepresentativeSubject(
        subject=str(row["subject"]),
        n_obs_parcels=int(row["n_obs_parcels"]),
        selection_rule="closest_to_median_plam_subject_performance_among_n_obs_ge_cmin",
        metadata={"pearson_r": float(row["pearson_r"]), "rmse": float(row["rmse"]), "coverage_tier": str(row["coverage_tier"])}
    )
    rep_path = out_root / "tables" / "representative_subject.csv"
    _ensure_parent(rep_path)
    pd.DataFrame([{"subject": rep.subject, "n_obs_parcels": rep.n_obs_parcels, "selection_rule": rep.selection_rule, **rep.metadata}]).to_csv(rep_path, index=False)
    return rep


def _sync_tree(src: Path, dst: Path, pattern: str = "*") -> List[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for path in sorted(src.glob(pattern)):
        if path.is_dir():
            continue
        target = dst / path.name
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def build_manuscript_assets(bundle: DatasetBundle, model_results: Mapping[str, ModelRunResult], eval_results: Mapping[str, EvaluationResult], cfg: Mapping[str, Any]) -> AssetManifest:
    out_root = Path(bundle.out_root)
    stage = "manuscript_assets"
    cfg_hash = _cfg_hash({"stage": stage, **_safe_cfg(cfg)})
    sources = bundle.source_signatures
    paths = _stage_paths(out_root, stage)
    asset_root = out_root / "manuscript_assets"
    fig_dir = asset_root / "figures"
    tab_dir = asset_root / "tables"
    manuscript_root = Path(str(cfg.get("manuscript_root", _repo_root() / "docs" / "manuscript"))).resolve()
    manuscript_fig_dir = manuscript_root / "figs"
    manuscript_tab_dir = manuscript_root / "tables"
    outputs = {
        "asset_root": str(asset_root),
        "manuscript_fig_dir": str(manuscript_fig_dir),
        "manuscript_table_dir": str(manuscript_tab_dir),
    }
    if stage not in _force_set(cfg) and _stage_valid(out_root, stage, cfg_hash, sources):
        manifest = _load_stage_manifest(out_root, stage)
        return AssetManifest(
            asset_root=str(asset_root),
            figure_paths=manifest.get("summary", {}).get("figure_paths", []),
            table_paths=manifest.get("summary", {}).get("table_paths", []),
            manuscript_fig_dir=str(manuscript_fig_dir),
            manuscript_table_dir=str(manuscript_tab_dir),
        )

    dlam_root = Path(model_results["dlam"].summary["compat_root"])
    plam_export_root = Path(model_results["plam"].summary["compat_root"])
    plam_phase2_root = out_root / "caches" / "plam_phase2_compat"
    cmd = [
        "python3",
        str(_repo_root() / "scripts" / "build_dual_model_manuscript_assets.py"),
        "--csv-path",
        bundle.csv_path,
        "--hvg-path",
        bundle.hvg_path,
        "--deterministic-root",
        str(dlam_root),
        "--unified-phase2-root",
        str(plam_phase2_root),
        "--unified-export-root",
        str(plam_export_root),
        "--out-root",
        str(asset_root),
        "--seed",
        str(int(cfg.get("seed", 123))),
        "--c-min",
        str(int(cfg.get("c_min", 8))),
    ]
    subprocess.run(cmd, check=True, cwd=str(_repo_root()))

    copied_figs = _sync_tree(fig_dir, manuscript_fig_dir, "*.pdf")
    copied_tabs = _sync_tree(tab_dir, manuscript_tab_dir, "*.csv")
    if (tab_dir / "figure_qc_manifest.json").exists():
        shutil.copy2(tab_dir / "figure_qc_manifest.json", manuscript_tab_dir / "figure_qc_manifest.json")
        copied_tabs.append(str(manuscript_tab_dir / "figure_qc_manifest.json"))
    summary = {"figure_paths": copied_figs, "table_paths": copied_tabs}
    _write_stage_manifest(out_root, stage, cfg_hash, sources, outputs, summary)
    return AssetManifest(asset_root=str(asset_root), figure_paths=copied_figs, table_paths=copied_tabs, manuscript_fig_dir=str(manuscript_fig_dir), manuscript_table_dir=str(manuscript_tab_dir))


def write_run_manifest(bundle: DatasetBundle, model_results: Mapping[str, ModelRunResult], eval_results: Mapping[str, EvaluationResult], assets: AssetManifest, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out_root = Path(bundle.out_root)
    rep_subject = ""
    rep_csv = out_root / "tables" / "representative_subject.csv"
    if rep_csv.exists():
        rep_df = pd.read_csv(rep_csv)
        if len(rep_df):
            rep_subject = str(rep_df.iloc[0]["subject"])
    manifest = WriteupRunManifest(
        config_hash=_cfg_hash(_safe_cfg(cfg)),
        timestamp=io_utils.utc_timestamp(),
        csv_path=bundle.csv_path,
        hvg_path=bundle.hvg_path,
        out_root=str(out_root),
        n_subjects_eligible=bundle.counts["n_subjects_eligible"],
        n_genes_all=bundle.counts["n_genes_all"],
        n_genes_hvg=bundle.counts["n_genes_hvg"],
        representative_subject=rep_subject,
        stages={
            "models": {k: asdict(v) for k, v in model_results.items()},
            "evaluation": {k: asdict(v) for k, v in eval_results.items()},
            "assets": asdict(assets),
        },
        figures=assets.figure_paths,
        tables=assets.table_paths,
    )
    path = out_root / "manifests" / "run_manifest.json"
    io_utils.dump_json(path, asdict(manifest))
    return asdict(manifest)


def load_workflow_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:
        raise RuntimeError(f"Unable to load config {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config in {p}")
    return data


def run_writeup_workflow(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    bundle = load_dataset_bundle(str(cfg.get("csv_path", "data/raw/gxp_samples.csv")), str(cfg.get("hvg_path", "data/raw/ahba_100hvg.txt")), cfg)
    summarize_shared_harmonization(bundle, cfg)
    naive = run_naive_fill(bundle, cfg)
    dlam = run_dlam(bundle, cfg)
    plam = run_plam(bundle, cfg)
    eval_naive = run_allgene_loro(bundle, "naive", None, cfg)
    eval_dlam = run_allgene_loro(bundle, "dlam", None, cfg)
    eval_plam = run_allgene_loro(bundle, "plam", None, cfg)
    eval_results = {"naive": eval_naive, "dlam": eval_dlam, "plam": eval_plam}
    rep = select_representative_subject(eval_results, cfg)
    assets = build_manuscript_assets(bundle, {"naive": naive, "dlam": dlam, "plam": plam}, eval_results, cfg) if bool(cfg.get("write_manuscript_assets", True)) else AssetManifest(str(Path(bundle.out_root) / "manuscript_assets"), [], [], str(Path(cfg.get("manuscript_root", _repo_root() / "docs" / "manuscript")) / "figs"), str(Path(cfg.get("manuscript_root", _repo_root() / "docs" / "manuscript")) / "tables"))
    manifest = write_run_manifest(bundle, {"naive": naive, "dlam": dlam, "plam": plam}, eval_results, assets, cfg)
    manifest["representative_subject"] = rep.subject
    return manifest


__all__ = [
    "AssetManifest",
    "DatasetBundle",
    "EvaluationResult",
    "ModelRunResult",
    "RepresentativeSubject",
    "WriteupRunManifest",
    "build_manuscript_assets",
    "load_dataset_bundle",
    "load_workflow_config",
    "run_allgene_loro",
    "run_dlam",
    "run_naive_fill",
    "run_plam",
    "run_writeup_workflow",
    "select_representative_subject",
    "summarize_shared_harmonization",
    "write_run_manifest",
]
