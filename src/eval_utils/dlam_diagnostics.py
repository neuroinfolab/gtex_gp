#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_utils.results_eda import EDAConfig, select_subject_by_model, set_academic_style
from src.workflows.loro_cache import SubjectCacheConfig, load_dataset
from src.harmonize import fit_harmonizer
from src.latent.basis_maps import apply_linear, apply_scores, fit_basis_map
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.preprocess import build_region_matrix, build_subject_observed_matrices
from src import io as io_utils


AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}
FONT = {"title": 16, "label": 14, "tick": 13, "legend": 13, "small": 12}


@dataclass
class DlamDiagnosticsConfig:
    csv_path: str = "data/raw/gxp_samples.csv"
    hvg_path: str = "out/raw/gene_lists/ahba_100hvg.txt"
    cache_root: str = "out/loro_subject_cache"
    gene_scope: str = "allgenes"
    min_observed_parcels: int = 5
    combat_use_covariates: bool = True

    # Subject ranking cache dir mapping (for median/best/worst subject selection).
    dlam_cache_dirname: str = "dlam"

    # DLAM fit controls (kept aligned with cache pipeline defaults).
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_length_scale: float = 25.0
    seed: int = 123
    c_min: int = 4
    basis_model: str = "affine_gl3"
    strategy: str = "constrained_anchor"
    spatial_method: str = "rbf"
    diagnostics_cache_root: str = "out/dlam_diagnostics"
    use_diagnostics_cache: bool = True
    write_diagnostics_cache: bool = True


@dataclass
class DlamSubjectDiagnostics:
    subject_id: str
    fold_mode: str
    hold_parcel: Optional[int]
    fold_selector: Optional[str]

    genes: List[str]
    obs_idx_all: np.ndarray
    train_idx: np.ndarray
    coords_full: np.ndarray
    coords_train: np.ndarray
    target_meta: pd.DataFrame
    gtex_label_by_parcel: Dict[int, str]

    ahba_h_full: np.ndarray
    x_obs_h: np.ndarray
    x_obs_raw: np.ndarray
    pred_full_h: np.ndarray

    n_comp: int
    t_obs: np.ndarray
    u_obs: np.ndarray
    t_ref_full: np.ndarray
    t_ref_obs: np.ndarray
    t_prime_obs: np.ndarray
    u_prime_obs: np.ndarray

    basis_model: str
    basis_m: np.ndarray
    basis_b: np.ndarray
    basis_inv_m: np.ndarray
    basis_diagnostics: Dict[str, float]

    hold_truth_h: Optional[np.ndarray] = None
    hold_pred_h: Optional[np.ndarray] = None
    hold_metrics: Dict[str, float] = field(default_factory=dict)
    fold_table: Optional[pd.DataFrame] = None


@dataclass
class _FoldFitPayload:
    subject_id: str
    hold_parcel: Optional[int]
    obs_idx_all: np.ndarray
    train_idx: np.ndarray
    genes: List[str]
    coords_full: np.ndarray
    target_meta: pd.DataFrame
    gtex_label_by_parcel: Dict[int, str]

    ahba_h_full: np.ndarray
    x_obs_h: np.ndarray
    x_obs_raw: np.ndarray
    pred_full_h: np.ndarray

    n_comp: int
    t_obs: np.ndarray
    u_obs: np.ndarray
    t_ref_full: np.ndarray
    t_ref_obs: np.ndarray
    t_prime_obs: np.ndarray
    u_prime_obs: np.ndarray

    basis_model: str
    basis_m: np.ndarray
    basis_b: np.ndarray
    basis_inv_m: np.ndarray
    basis_diagnostics: Dict[str, float]

    hold_truth_h: Optional[np.ndarray]
    hold_pred_h: Optional[np.ndarray]
    hold_metrics: Dict[str, float]


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


def _subject_cache_cfg(cfg: DlamDiagnosticsConfig) -> SubjectCacheConfig:
    return SubjectCacheConfig(
        csv_path=cfg.csv_path,
        hvg_path=cfg.hvg_path,
        gene_scope=cfg.gene_scope,
        min_observed_parcels=cfg.min_observed_parcels,
        combat_use_covariates=cfg.combat_use_covariates,
        n_comp_target=cfg.n_comp_target,
        ridge_alpha_bridge=cfg.ridge_alpha_bridge,
        rbf_smoothing=cfg.rbf_smoothing,
        gp_length_scale=cfg.gp_length_scale,
        seed=cfg.seed,
        c_min=cfg.c_min,
        use_cache=True,
    )


def _get_bundle(cfg: DlamDiagnosticsConfig) -> Dict[str, object]:
    return load_dataset(_subject_cache_cfg(cfg))


def _source_signature(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": int(st.st_size), "mtime": float(st.st_mtime)}


def _cfg_fingerprint(cfg: DlamDiagnosticsConfig) -> Dict[str, object]:
    return {
        "csv_path": str(cfg.csv_path),
        "hvg_path": str(cfg.hvg_path),
        "gene_scope": str(cfg.gene_scope),
        "min_observed_parcels": int(cfg.min_observed_parcels),
        "combat_use_covariates": bool(cfg.combat_use_covariates),
        "n_comp_target": int(cfg.n_comp_target),
        "ridge_alpha_bridge": float(cfg.ridge_alpha_bridge),
        "rbf_smoothing": float(cfg.rbf_smoothing),
        "gp_length_scale": float(cfg.gp_length_scale),
        "seed": int(cfg.seed),
        "c_min": int(cfg.c_min),
        "basis_model": str(cfg.basis_model),
        "strategy": str(cfg.strategy),
        "spatial_method": str(cfg.spatial_method),
    }


def _cache_subject_root(cfg: DlamDiagnosticsConfig, subject_id: str) -> Path:
    return (REPO_ROOT / str(cfg.diagnostics_cache_root) / str(cfg.gene_scope).lower() / str(subject_id)).resolve()


def _fold_metrics_cache_paths(
    cfg: DlamDiagnosticsConfig,
    subject_id: str,
    bundle: Dict[str, object],
) -> Tuple[Path, Path]:
    sigs = {
        "csv": _source_signature(Path(bundle["csv_path"])),
        "hvg": _source_signature(Path(bundle["hvg_path"])),
        "cfg": _cfg_fingerprint(cfg),
        "subject": str(subject_id),
    }
    key = io_utils.hash_config(sigs)
    root = _cache_subject_root(cfg, subject_id)
    return root / f"loro_fold_metrics_{key}.csv", root / f"loro_fold_metrics_{key}.json"


def _diagnostics_cache_paths(
    cfg: DlamDiagnosticsConfig,
    subject_id: str,
    fold_mode: str,
    hold_parcel: Optional[int],
    selector: Optional[str],
    bundle: Dict[str, object],
) -> Tuple[Path, Path]:
    sigs = {
        "csv": _source_signature(Path(bundle["csv_path"])),
        "hvg": _source_signature(Path(bundle["hvg_path"])),
        "cfg": _cfg_fingerprint(cfg),
        "subject": str(subject_id),
        "fold_mode": str(fold_mode),
        "hold_parcel": None if hold_parcel is None else int(hold_parcel),
        "selector": None if selector is None else str(selector),
    }
    key = io_utils.hash_config(sigs)
    root = _cache_subject_root(cfg, subject_id)
    return root / f"diagnostics_{key}.pkl", root / f"diagnostics_{key}.json"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _cache_valid(meta_path: Path, expected: Dict[str, object]) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return bool(meta.get("status") == "ok" and meta.get("key_payload") == expected)


def _pretty_gtex_label(name: str) -> str:
    s = str(name)
    pref = "brain - "
    if s.lower().startswith(pref):
        return s[len(pref) :]
    return s


def _resolve_subject(cfg: DlamDiagnosticsConfig, subject_id: Optional[str], subject_mode: str) -> str:
    if subject_id:
        return str(subject_id)
    eda_cfg = EDAConfig(
        csv_path=cfg.csv_path,
        hvg_path=cfg.hvg_path,
        cache_root=cfg.cache_root,
        gene_scope=cfg.gene_scope,
        min_observed_parcels=cfg.min_observed_parcels,
        combat_use_covariates=cfg.combat_use_covariates,
        dlam_cache_dirname=cfg.dlam_cache_dirname,
    )
    return str(select_subject_by_model(eda_cfg, mode=str(subject_mode), metric="pearson_r", model="dlam"))


def list_subject_observed_parcels(cfg: DlamDiagnosticsConfig, subject_id: str, bundle: Optional[Dict[str, object]] = None) -> List[int]:
    b = bundle if bundle is not None else _get_bundle(cfg)
    g = b["gtex_raw"]
    sid = str(subject_id)
    obs = sorted(set(int(x) for x in g[g["subject"].astype(str) == sid]["parcel_idx"].astype(np.int32).tolist()))
    return obs


def _fit_dlam_fold_payload(
    cfg: DlamDiagnosticsConfig,
    subject_id: str,
    hold_parcel: Optional[int],
    bundle: Optional[Dict[str, object]] = None,
) -> _FoldFitPayload:
    b = bundle if bundle is not None else _get_bundle(cfg)
    sid = str(subject_id)

    ahba_raw = b["ahba_raw"]
    gtex_raw = b["gtex_raw"]
    genes = [str(g) for g in b["genes"]]
    target_meta = b["target_meta"].copy()
    coords_full = np.asarray(b["coords_full"], dtype=np.float64)

    eligible = set(str(s) for s in b["eligible_subjects"])
    if sid not in eligible:
        raise ValueError(f"Subject {sid} is not eligible under current config")

    obs_idx_all = np.sort(gtex_raw[gtex_raw["subject"].astype(str) == sid]["parcel_idx"].astype(np.int32).unique())
    if hold_parcel is not None and int(hold_parcel) not in set(obs_idx_all.astype(int).tolist()):
        raise ValueError(f"hold_parcel={hold_parcel} is not observed for subject {sid}")

    if hold_parcel is None:
        train_mask = np.ones(len(gtex_raw), dtype=bool)
    else:
        train_mask = ~((gtex_raw["subject"].astype(str) == sid) & (gtex_raw["parcel_idx"].astype(np.int32) == int(hold_parcel)))
    gtex_train = gtex_raw[train_mask].copy()

    hcfg = SimpleNamespace(combat_use_covariates=bool(cfg.combat_use_covariates))
    harm = fit_harmonizer(ahba_raw, gtex_train, genes, method="combat", cfg=hcfg)
    ahba_h = harm.transform(ahba_raw, "AHBA")
    gtex_h = harm.transform(gtex_train, "GTEX")
    ahba_h_full, _ = build_region_matrix(ahba_h, genes, target_meta, agg="mean")

    subj_h = gtex_h[gtex_h["subject"].astype(str) == sid].copy()
    subj_raw = gtex_train[gtex_train["subject"].astype(str) == sid].copy()
    train_idx, xh, xr = build_subject_observed_matrices(subj_h, subj_raw, genes)
    if int(len(train_idx)) < 2:
        raise RuntimeError(f"Insufficient training parcels for subject {sid}, hold={hold_parcel}")

    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
    y_obs = y_full[train_idx, :]

    ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=int(cfg.n_comp_target), adaptive=True)
    subj_pls = fit_subject_pls(xh, y_obs, n_comp_target=int(cfg.n_comp_target), adaptive=True)

    k = int(subj_pls["n_comp"])
    t_obs = np.asarray(subj_pls["T"], dtype=np.float64)
    u_obs = np.asarray(subj_pls["U"], dtype=np.float64)
    t_ref_full = np.asarray(ahba_pls["T"], dtype=np.float64)[:, :k]
    t_ref_obs = np.asarray(ahba_pls["T"], dtype=np.float64)[train_idx, :k]

    bmap = fit_basis_map(t_obs, t_ref_obs, model=str(cfg.basis_model))
    t_prime_obs = apply_scores(t_obs, bmap)
    u_prime_obs = apply_linear(u_obs, bmap)

    atlas_bundle = {"ahba_h_full": ahba_h_full, "ahba_ref_T": np.asarray(ahba_pls["T"], dtype=np.float64)}
    method_bundle = {
        "harmonizer": harm,
        "basis_model": str(cfg.basis_model),
        "strategy": str(cfg.strategy),
        "spatial_method": str(cfg.spatial_method),
        "n_comp_target": int(cfg.n_comp_target),
        "ridge_alpha_bridge": float(cfg.ridge_alpha_bridge),
        "rbf_smoothing": float(cfg.rbf_smoothing),
        "gp_rbf_length": float(cfg.gp_length_scale),
        "seed": int(cfg.seed),
        "c_min": int(cfg.c_min),
        "distance_d0": 45.0,
        "distance_tau": 10.0,
        "uncertainty_shrink": False,
    }
    pred_out, _ = run_subject(
        {
            "subject": sid,
            "obs_idx": train_idx,
            "X_obs_h": xh,
            "X_obs_raw": xr,
            "coords_full": coords_full,
            "target_meta": target_meta,
        },
        atlas_bundle,
        method_bundle,
        {
            "c_min": int(cfg.c_min),
            "n_comp_target": int(cfg.n_comp_target),
            "seed": int(cfg.seed),
        },
    )
    pred_full_h = np.asarray(pred_out["X_full_h"], dtype=np.float64)

    hold_truth_h = None
    hold_pred_h = None
    hold_metrics: Dict[str, float] = {
        "pearson_r": np.nan,
        "rmse": np.nan,
        "n_points": 0.0,
    }
    if hold_parcel is not None:
        hold_rows = gtex_raw[(gtex_raw["subject"].astype(str) == sid) & (gtex_raw["parcel_idx"].astype(np.int32) == int(hold_parcel))].copy()
        if len(hold_rows):
            hold_h = harm.transform(hold_rows, "GTEX")
            hold_truth_h = hold_h[genes].to_numpy(dtype=np.float64).mean(axis=0)
            hold_pred_h = np.asarray(pred_full_h[int(hold_parcel), :], dtype=np.float64)
            m = np.isfinite(hold_truth_h) & np.isfinite(hold_pred_h)
            hold_metrics = {
                "pearson_r": _pearson_safe(hold_truth_h[m], hold_pred_h[m]) if int(m.sum()) else np.nan,
                "rmse": _rmse_safe(hold_truth_h[m], hold_pred_h[m]) if int(m.sum()) else np.nan,
                "n_points": float(m.sum()),
            }

    subj_rows = gtex_raw[gtex_raw["subject"].astype(str) == sid].copy()
    gtex_label_by_parcel = (
        subj_rows.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(_pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .to_dict()
    )

    return _FoldFitPayload(
        subject_id=sid,
        hold_parcel=int(hold_parcel) if hold_parcel is not None else None,
        obs_idx_all=obs_idx_all.astype(np.int32),
        train_idx=np.asarray(train_idx, dtype=np.int32),
        genes=genes,
        coords_full=coords_full,
        target_meta=target_meta,
        gtex_label_by_parcel={int(k): str(v) for k, v in gtex_label_by_parcel.items()},
        ahba_h_full=ahba_h_full,
        x_obs_h=np.asarray(xh, dtype=np.float64),
        x_obs_raw=np.asarray(xr, dtype=np.float64),
        pred_full_h=pred_full_h,
        n_comp=k,
        t_obs=np.asarray(t_obs, dtype=np.float64),
        u_obs=np.asarray(u_obs, dtype=np.float64),
        t_ref_full=np.asarray(t_ref_full, dtype=np.float64),
        t_ref_obs=np.asarray(t_ref_obs, dtype=np.float64),
        t_prime_obs=np.asarray(t_prime_obs, dtype=np.float64),
        u_prime_obs=np.asarray(u_prime_obs, dtype=np.float64),
        basis_model=str(bmap.model),
        basis_m=np.asarray(bmap.M, dtype=np.float64),
        basis_b=np.asarray(bmap.b, dtype=np.float64),
        basis_inv_m=np.asarray(bmap.invM, dtype=np.float64),
        basis_diagnostics={k: float(v) for k, v in bmap.diagnostics.items()},
        hold_truth_h=None if hold_truth_h is None else np.asarray(hold_truth_h, dtype=np.float64),
        hold_pred_h=None if hold_pred_h is None else np.asarray(hold_pred_h, dtype=np.float64),
        hold_metrics=hold_metrics,
    )


def compute_dlam_loro_fold_metrics(
    cfg: DlamDiagnosticsConfig,
    subject_id: str,
    bundle: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    b = bundle if bundle is not None else _get_bundle(cfg)
    sid = str(subject_id)
    csv_cache, meta_cache = _fold_metrics_cache_paths(cfg, sid, b)
    key_payload = {
        "csv": _source_signature(Path(b["csv_path"])),
        "hvg": _source_signature(Path(b["hvg_path"])),
        "cfg": _cfg_fingerprint(cfg),
        "subject": sid,
    }
    if bool(cfg.use_diagnostics_cache) and csv_cache.exists() and _cache_valid(meta_cache, key_payload):
        return pd.read_csv(csv_cache)

    holds = list_subject_observed_parcels(cfg, sid, bundle=b)
    rows: List[Dict[str, object]] = []
    for hold in holds:
        fold = _fit_dlam_fold_payload(cfg, sid, hold_parcel=int(hold), bundle=b)
        rows.append(
            {
                "subject": sid,
                "hold_parcel": int(hold),
                "n_obs": int(len(fold.obs_idx_all)),
                "n_train": int(len(fold.train_idx)),
                "n_comp": int(fold.n_comp),
                "pearson_r": float(fold.hold_metrics.get("pearson_r", np.nan)),
                "rmse": float(fold.hold_metrics.get("rmse", np.nan)),
                "n_points": int(fold.hold_metrics.get("n_points", 0)),
            }
        )
    out = pd.DataFrame(rows).sort_values("hold_parcel").reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No observed folds found for subject={sid}")
    if bool(cfg.write_diagnostics_cache):
        csv_cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_cache, index=False)
        _write_json(
            meta_cache,
            {
                "status": "ok",
                "kind": "loro_fold_metrics",
                "subject": sid,
                "gene_scope": str(cfg.gene_scope).lower(),
                "timestamp": io_utils.utc_timestamp(),
                "key_payload": key_payload,
            },
        )
    return out


def _pick_hold_parcel(fold_df: pd.DataFrame, selector: str) -> int:
    sel = str(selector).lower()
    d = fold_df.copy()
    if len(d) == 0:
        raise RuntimeError("Empty fold_df")

    if sel in {"first", "min_hold"}:
        return int(d.sort_values(["hold_parcel"]).iloc[0]["hold_parcel"])

    if sel in {"best", "best_rmse"}:
        return int(d.sort_values(["rmse", "hold_parcel"], ascending=[True, True]).iloc[0]["hold_parcel"])

    if sel in {"worst", "worst_rmse"}:
        return int(d.sort_values(["rmse", "hold_parcel"], ascending=[False, True]).iloc[0]["hold_parcel"])

    if sel in {"median", "median_rmse"}:
        med = float(d["rmse"].median())
        d["delta"] = np.abs(d["rmse"] - med)
        return int(d.sort_values(["delta", "hold_parcel"], ascending=[True, True]).iloc[0]["hold_parcel"])

    if sel in {"best_pearson", "max_pearson"}:
        return int(d.sort_values(["pearson_r", "hold_parcel"], ascending=[False, True]).iloc[0]["hold_parcel"])

    if sel in {"worst_pearson", "min_pearson"}:
        return int(d.sort_values(["pearson_r", "hold_parcel"], ascending=[True, True]).iloc[0]["hold_parcel"])

    raise ValueError(f"Unknown hold selector: {selector}")


def fit_dlam_subject_diagnostics(
    cfg: DlamDiagnosticsConfig,
    subject_id: Optional[str] = None,
    subject_mode: str = "median",
    fold_mode: str = "full",  # full | loro
    hold_parcel: Optional[int] = None,
    hold_selector: str = "median_rmse",
) -> DlamSubjectDiagnostics:
    sid = _resolve_subject(cfg, subject_id=subject_id, subject_mode=subject_mode)
    b = _get_bundle(cfg)
    mode = str(fold_mode).lower()
    if mode not in {"full", "loro"}:
        raise ValueError("fold_mode must be one of: full, loro")

    fold_table = None
    hold = None
    selector_out = None
    if mode == "loro":
        if hold_parcel is None:
            fold_table = compute_dlam_loro_fold_metrics(cfg, sid, bundle=b)
            hold = _pick_hold_parcel(fold_table, hold_selector)
            selector_out = str(hold_selector)
        else:
            hold = int(hold_parcel)
            selector_out = "explicit_hold"

    pkl_cache, meta_cache = _diagnostics_cache_paths(
        cfg=cfg,
        subject_id=sid,
        fold_mode=mode,
        hold_parcel=hold,
        selector=selector_out,
        bundle=b,
    )
    key_payload = {
        "csv": _source_signature(Path(b["csv_path"])),
        "hvg": _source_signature(Path(b["hvg_path"])),
        "cfg": _cfg_fingerprint(cfg),
        "subject": sid,
        "fold_mode": mode,
        "hold_parcel": None if hold is None else int(hold),
        "selector": None if selector_out is None else str(selector_out),
    }
    if bool(cfg.use_diagnostics_cache) and pkl_cache.exists() and _cache_valid(meta_cache, key_payload):
        with open(pkl_cache, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, DlamSubjectDiagnostics) and hasattr(obj, "t_ref_full"):
            # preserve freshly computed fold table when selector path was used
            if obj.fold_table is None and fold_table is not None:
                obj.fold_table = fold_table
            return obj

    payload = _fit_dlam_fold_payload(cfg, sid, hold_parcel=hold, bundle=b)

    diag = DlamSubjectDiagnostics(
        subject_id=payload.subject_id,
        fold_mode=mode,
        hold_parcel=payload.hold_parcel,
        fold_selector=selector_out,
        genes=payload.genes,
        obs_idx_all=payload.obs_idx_all,
        train_idx=payload.train_idx,
        coords_full=payload.coords_full,
        coords_train=payload.coords_full[payload.train_idx, :],
        target_meta=payload.target_meta,
        gtex_label_by_parcel=payload.gtex_label_by_parcel,
        ahba_h_full=payload.ahba_h_full,
        x_obs_h=payload.x_obs_h,
        x_obs_raw=payload.x_obs_raw,
        pred_full_h=payload.pred_full_h,
        n_comp=payload.n_comp,
        t_obs=payload.t_obs,
        u_obs=payload.u_obs,
        t_ref_full=payload.t_ref_full,
        t_ref_obs=payload.t_ref_obs,
        t_prime_obs=payload.t_prime_obs,
        u_prime_obs=payload.u_prime_obs,
        basis_model=payload.basis_model,
        basis_m=payload.basis_m,
        basis_b=payload.basis_b,
        basis_inv_m=payload.basis_inv_m,
        basis_diagnostics=payload.basis_diagnostics,
        hold_truth_h=payload.hold_truth_h,
        hold_pred_h=payload.hold_pred_h,
        hold_metrics=payload.hold_metrics,
        fold_table=fold_table,
    )
    if bool(cfg.write_diagnostics_cache):
        pkl_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_cache, "wb") as f:
            pickle.dump(diag, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(
            meta_cache,
            {
                "status": "ok",
                "kind": "dlam_subject_diagnostics",
                "subject": sid,
                "gene_scope": str(cfg.gene_scope).lower(),
                "timestamp": io_utils.utc_timestamp(),
                "key_payload": key_payload,
            },
        )
    return diag


def diagnostics_summary_table(diag: DlamSubjectDiagnostics) -> pd.DataFrame:
    row = {
        "subject": str(diag.subject_id),
        "fold_mode": str(diag.fold_mode),
        "hold_parcel": -1 if diag.hold_parcel is None else int(diag.hold_parcel),
        "fold_selector": "" if diag.fold_selector is None else str(diag.fold_selector),
        "n_obs": int(len(diag.obs_idx_all)),
        "n_train": int(len(diag.train_idx)),
        "n_comp": int(diag.n_comp),
        "basis_model": str(diag.basis_model),
        "hold_pearson": float(diag.hold_metrics.get("pearson_r", np.nan)),
        "hold_rmse": float(diag.hold_metrics.get("rmse", np.nan)),
        "hold_n_points": int(diag.hold_metrics.get("n_points", 0)),
    }
    return pd.DataFrame([row])


def _axis_idx(name: str) -> int:
    key = str(name).strip().lower()
    if key not in AXIS_TO_IDX:
        raise ValueError(f"axis must be one of x|y|z; got {name}")
    return int(AXIS_TO_IDX[key])


def plot_dlam_bases_panel(
    diag: DlamSubjectDiagnostics,
    component: int = 0,
    latent_space: str = "spatial",  # spatial | expression
    axis_x: str = "y",
    axis_y: str = "z",
    invert_x: bool = False,
    invert_y: bool = False,
    full_brain_ref: bool = True,
    cmap: str | None = None,
    figsize: Tuple[float, float] = (22.0, 5.0),
    point_size: float = 97.0,
    ref_point_size: float = 23.0,
    ref_alpha: float = 0.26,
) -> Tuple[plt.Figure, np.ndarray]:
    comp = int(component)
    if comp < 0 or comp >= int(diag.n_comp):
        raise ValueError(f"component must be in [0, {int(diag.n_comp) - 1}]")
    latent_mode = str(latent_space).strip().lower()
    if latent_mode not in {"spatial", "expression"}:
        raise ValueError("latent_space must be one of: spatial, expression")

    ix = _axis_idx(axis_x)
    iy = _axis_idx(axis_y)
    xy = np.asarray(diag.coords_train, dtype=np.float64)
    xy_all = np.asarray(diag.coords_full, dtype=np.float64)
    if latent_mode == "expression":
        pre = np.asarray(diag.u_obs[:, comp], dtype=np.float64)
        post = np.asarray(diag.u_prime_obs[:, comp], dtype=np.float64)
    else:
        pre = np.asarray(diag.t_obs[:, comp], dtype=np.float64)
        post = np.asarray(diag.t_prime_obs[:, comp], dtype=np.float64)
    ref = np.asarray(diag.t_ref_obs[:, comp], dtype=np.float64)
    # Backward compatibility with older cached diagnostics objects.
    if hasattr(diag, "t_ref_full") and getattr(diag, "t_ref_full") is not None:
        ref_full = np.asarray(diag.t_ref_full[:, comp], dtype=np.float64)
        has_full_ref = True
    else:
        ref_full = np.asarray(ref, dtype=np.float64)
        has_full_ref = False

    if cmap is None:
        cmap_cycle = ("viridis", "plasma", "cividis", "magma", "inferno")
        cmap_use = cmap_cycle[comp % len(cmap_cycle)]
    else:
        cmap_use = str(cmap)

    vals = np.r_[pre, post, ref]
    m = np.isfinite(vals)
    if int(m.sum()):
        lo = float(np.nanpercentile(vals[m], 2.0))
        hi = float(np.nanpercentile(vals[m], 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(vals[m]))
            hi = float(np.nanmax(vals[m]))
    else:
        lo, hi = -1.0, 1.0

    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)

    panels = [
        ("Pre-aligned GTEx score", pre, axes[0]),
        ("Post-aligned GTEx score", post, axes[1]),
        ("AHBA reference score", ref, axes[2]),
    ]
    sc = None
    for title, cvals, ax in panels:
        if bool(full_brain_ref):
            ax.scatter(
                xy_all[:, ix],
                xy_all[:, iy],
                s=float(ref_point_size),
                c="#d0d0d0",
                alpha=float(ref_alpha),
                linewidth=0.0,
                zorder=1,
            )
        sc = ax.scatter(
            xy[:, ix],
            xy[:, iy],
            c=cvals,
            cmap=cmap_use,
            s=float(point_size),
            alpha=0.95,
            edgecolor="white",
            linewidth=0.55,
            vmin=lo,
            vmax=hi,
            zorder=2,
        )
        ax.set_title(f"{title}", fontsize=FONT["title"])
        ax.set_xlabel(f"{axis_x.upper()} coord", fontsize=FONT["label"])
        ax.set_ylabel(f"{axis_y.upper()} coord", fontsize=FONT["label"])
        ax.tick_params(axis="both", labelsize=FONT["tick"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        if bool(invert_x):
            ax.invert_xaxis()
        if bool(invert_y):
            ax.invert_yaxis()
        ax.grid(False)

    ax_ref = axes[2]
    if bool(full_brain_ref):
        for coll in list(ax_ref.collections):
            coll.remove()
        if has_full_ref:
            sc = ax_ref.scatter(
                xy_all[:, ix],
                xy_all[:, iy],
                c=ref_full,
                cmap=cmap_use,
                s=float(ref_point_size),
                alpha=0.95,
                edgecolor="white",
                linewidth=0.25,
                vmin=lo,
                vmax=hi,
                zorder=2,
            )
            # Overlay matched parcels at GTEx marker size for direct visual comparability.
            ax_ref.scatter(
                xy[:, ix],
                xy[:, iy],
                c=ref,
                cmap=cmap_use,
                s=float(point_size),
                alpha=0.98,
                edgecolor="white",
                linewidth=0.55,
                vmin=lo,
                vmax=hi,
                zorder=3,
            )
        else:
            # Fallback for old diagnostics: only matched parcels are available.
            sc = ax_ref.scatter(
                xy[:, ix],
                xy[:, iy],
                c=ref,
                cmap=cmap_use,
                s=float(point_size),
                alpha=0.95,
                edgecolor="white",
                linewidth=0.55,
                vmin=lo,
                vmax=hi,
                zorder=2,
            )
    if bool(invert_x):
        ax_ref.invert_xaxis()
    if bool(invert_y):
        ax_ref.invert_yaxis()
    ax_ref.set_box_aspect(1)

    r_pre = _pearson_safe(pre, ref)
    rmse_pre = _rmse_safe(pre, ref)
    ax = axes[3]
    ax.scatter(pre, ref, s=50, alpha=0.88, color="#7f3c8d", edgecolor="white", linewidth=0.42)
    lo4 = float(np.nanmin(np.r_[pre, ref]))
    hi4 = float(np.nanmax(np.r_[pre, ref]))
    if np.isfinite(lo4) and np.isfinite(hi4) and hi4 > lo4:
        pad = 0.06 * (hi4 - lo4)
        ax.plot([lo4, hi4], [lo4, hi4], "k--", linewidth=1.0)
        ax.set_xlim(lo4 - pad, hi4 + pad)
        ax.set_ylim(lo4 - pad, hi4 + pad)
    ax.set_title("Pre-aligned vs AHBA ref", fontsize=FONT["title"])
    ax.set_xlabel("Pre-aligned score", fontsize=FONT["label"])
    ax.set_ylabel("AHBA reference score", fontsize=FONT["label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.03,
        0.97,
        f"r={r_pre:.3f} (n={int(len(pre))})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT["legend"],
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    r_post = _pearson_safe(post, ref)
    rmse_post = _rmse_safe(post, ref)
    ax = axes[4]
    ax.scatter(post, ref, s=50, alpha=0.88, color="#2f4f6f", edgecolor="white", linewidth=0.42)
    lo5 = float(np.nanmin(np.r_[post, ref]))
    hi5 = float(np.nanmax(np.r_[post, ref]))
    if np.isfinite(lo5) and np.isfinite(hi5) and hi5 > lo5:
        pad = 0.06 * (hi5 - lo5)
        ax.plot([lo5, hi5], [lo5, hi5], "k--", linewidth=1.0)
        ax.set_xlim(lo5 - pad, hi5 + pad)
        ax.set_ylim(lo5 - pad, hi5 + pad)
    ax.set_title("Post-aligned vs AHBA ref", fontsize=FONT["title"])
    ax.set_xlabel("Post-aligned score", fontsize=FONT["label"])
    ax.set_ylabel("AHBA reference score", fontsize=FONT["label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.03,
        0.97,
        f"r={r_post:.3f} (n={int(len(post))})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT["legend"],
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    cbar = fig.colorbar(sc, ax=axes[:3].tolist(), shrink=0.86)
    cbar.set_label("Latent score", fontsize=FONT["label"])
    # Minimal row-style component label on the far left.
    fig.text(
        -0.012,
        0.52,
        f"Latent {comp + 1} ({latent_mode})",
        rotation=90,
        va="center",
        ha="left",
        fontsize=FONT["label"],
    )
    axes[0].set_title("Pre-aligned GTEx score", fontsize=FONT["title"])
    axes[1].set_title("Post-aligned GTEx score", fontsize=FONT["title"])
    axes[2].set_title("AHBA reference score", fontsize=FONT["title"])
    fig.suptitle(
        f"DLAM Bases View | subject={diag.subject_id} | mode={diag.fold_mode}"
        + ("" if diag.hold_parcel is None else f" | hold={int(diag.hold_parcel)}"),
        fontsize=FONT["title"] + 1,
    )
    return fig, axes


def plot_subject_ahba_harmonized_heatmaps(
    diag: DlamSubjectDiagnostics,
    observed_only: bool = True,
    label_mode: str = "gtex",  # gtex | ahba | both
    label_stride: int = 1,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (12.6, 4.8),
) -> Tuple[plt.Figure, np.ndarray]:
    n_parc = int(diag.ahba_h_full.shape[0])
    n_genes = int(diag.ahba_h_full.shape[1])

    subj_mat = np.full((n_parc, n_genes), np.nan, dtype=np.float64)
    subj_mat[diag.train_idx, :] = np.asarray(diag.x_obs_h, dtype=np.float64)

    ahba_mat = np.asarray(diag.ahba_h_full, dtype=np.float64)
    ahba_labels = diag.target_meta["tissue_or_parcel"].astype(str).tolist()

    mode = str(label_mode).lower()
    if mode not in {"gtex", "ahba", "both"}:
        raise ValueError("label_mode must be one of: gtex, ahba, both")

    labels: List[str] = []
    for pidx, ahba_name in enumerate(ahba_labels):
        gname = str(diag.gtex_label_by_parcel.get(int(pidx), ""))
        if mode == "ahba":
            labels.append(str(ahba_name))
        elif mode == "both":
            labels.append(f"{ahba_name}: {gname if gname else ahba_name}")
        else:
            labels.append(gname if gname else str(ahba_name))

    if bool(observed_only):
        keep = np.isfinite(subj_mat).any(axis=1)
        subj_mat = subj_mat[keep, :]
        ahba_mat = ahba_mat[keep, :]
        labels = [labels[i] for i in np.where(keep)[0].tolist()]

    vals = np.r_[subj_mat.ravel(), ahba_mat.ravel()]
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
    m0 = axes[0].imshow(subj_mat, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)
    axes[1].imshow(ahba_mat, aspect="auto", interpolation="none", cmap=str(cmap), vmin=vmin, vmax=vmax)

    left_title = "Subject harmonized GTEx (train parcels)"
    if diag.fold_mode == "full":
        left_title = "Subject harmonized GTEx (observed parcels)"
    axes[0].set_title(left_title, fontsize=FONT["title"])
    axes[1].set_title("AHBA harmonized reference", fontsize=FONT["title"])

    step = max(1, int(label_stride))
    y_idx = np.arange(0, len(labels), step, dtype=int)
    y_lab = [labels[i] for i in y_idx.tolist()]

    for ax in axes:
        ax.set_xlabel("Genes", fontsize=FONT["label"])
        ax.set_ylabel("Parcels", fontsize=FONT["label"])
        ax.set_yticks(y_idx.tolist())
        ax.set_yticklabels(y_lab, fontsize=FONT["small"])
        ax.tick_params(axis="x", labelsize=FONT["tick"])
        ax.grid(False)

    cbar = fig.colorbar(m0, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Harmonized expression", fontsize=FONT["label"])
    fig.suptitle(
        f"Subject vs AHBA Harmonized Heatmaps | subject={diag.subject_id} | mode={diag.fold_mode}",
        fontsize=FONT["title"] + 1,
    )
    return fig, axes


__all__ = [
    "DlamDiagnosticsConfig",
    "DlamSubjectDiagnostics",
    "compute_dlam_loro_fold_metrics",
    "diagnostics_summary_table",
    "fit_dlam_subject_diagnostics",
    "list_subject_observed_parcels",
    "plot_dlam_bases_panel",
    "plot_subject_ahba_harmonized_heatmaps",
    "set_academic_style",
]
