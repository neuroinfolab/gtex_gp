#!/usr/bin/env python3
"""
Diagnose robustz+affine misfit vs distance, then benchmark mitigation strategies
with full subject-level evaluation.

Outputs under out/o3_subject_misfit_mitigation/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
import warnings

import run_o3_subject_level_prediction as base
import run_subject_model_grid as smg


STRATEGIES_DEFAULT = [
    "baseline",
    "distance_shrink",
    "piecewise_harmonization",
    "constrained_anchor",
    "gp_uncertainty",
    "moe_basis",
]


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    out_root: str = "out/o3_subject_misfit_mitigation"
    min_observed_parcels: int = 5
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    seed: int = 123
    strategies: List[str] = None
    distance_d0: float = 45.0
    distance_tau: float = 10.0
    min_group_samples: int = 200
    gp_rbf_length: float = 25.0


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run misfit-distance diagnostics and mitigation strategy benchmark.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--strategies", default=",".join(STRATEGIES_DEFAULT))
    p.add_argument("--distance-d0", type=float, default=Config.distance_d0)
    p.add_argument("--distance-tau", type=float, default=Config.distance_tau)
    p.add_argument("--min-group-samples", type=int, default=Config.min_group_samples)
    p.add_argument("--gp-rbf-length", type=float, default=Config.gp_rbf_length)
    a = p.parse_args()

    strategies = [s.strip() for s in str(a.strategies).split(",") if s.strip()]
    valid = set(STRATEGIES_DEFAULT)
    bad = [s for s in strategies if s not in valid]
    if bad:
        raise RuntimeError(f"Unsupported strategies: {bad}; valid={sorted(valid)}")

    return Config(
        csv_path=a.csv_path,
        hvg_path=a.hvg_path,
        out_root=a.out_root,
        min_observed_parcels=a.min_observed_parcels,
        n_comp_target=a.n_comp_target,
        ridge_alpha_bridge=a.ridge_alpha_bridge,
        rbf_smoothing=a.rbf_smoothing,
        seed=a.seed,
        strategies=strategies,
        distance_d0=float(a.distance_d0),
        distance_tau=float(a.distance_tau),
        min_group_samples=int(a.min_group_samples),
        gp_rbf_length=float(a.gp_rbf_length),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def macro_system(parcel_name: str) -> str:
    s = str(parcel_name)
    if s.startswith("Cerebellar_"):
        return "cerebellar"
    if s.startswith("LH-") or s.startswith("RH-"):
        return "subcortical"
    if "Amygdala" in s or "Hippocampus" in s:
        return "subcortical"
    if s.startswith("LH_Vis") or s.startswith("RH_Vis") or s.startswith("LH_SomMot") or s.startswith("RH_SomMot"):
        return "visual_somatomotor"
    return "cortical_association"


def hemisphere(parcel_name: str) -> str:
    s = str(parcel_name)
    if s.startswith("LH"):
        return "L"
    if s.startswith("RH"):
        return "R"
    return "M"


def add_target_meta(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    out["macro_system"] = out["tissue_or_parcel"].map(macro_system)
    out["hemisphere"] = out["tissue_or_parcel"].map(hemisphere)
    return out


def add_sample_group(df: pd.DataFrame, target_meta: pd.DataFrame) -> pd.DataFrame:
    lk_sys = dict(zip(target_meta["parcel_idx"].tolist(), target_meta["macro_system"].tolist()))
    out = df.copy()
    out["macro_system"] = out["parcel_idx"].map(lk_sys)
    return out


def compute_global_distance_table(target_meta: pd.DataFrame, global_obs_mask: np.ndarray) -> pd.DataFrame:
    coords = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    obs_idx = np.where(global_obs_mask.astype(bool))[0]
    obs_coords = coords[obs_idx, :]
    d2 = ((coords[:, None, :] - obs_coords[None, :, :]) ** 2).sum(axis=2)
    nn = np.argmin(d2, axis=1)
    nearest_obs_idx = obs_idx[nn]
    nearest_name = target_meta.iloc[nearest_obs_idx]["tissue_or_parcel"].astype(str).to_numpy()
    dist = np.sqrt(d2[np.arange(len(coords)), nn])

    out = pd.DataFrame(
        {
            "parcel_idx": target_meta["parcel_idx"].to_numpy(dtype=np.int32),
            "parcel_name": target_meta["tissue_or_parcel"].astype(str).to_numpy(),
            "coord_x": target_meta["coord_x"].to_numpy(dtype=np.float64),
            "coord_y": target_meta["coord_y"].to_numpy(dtype=np.float64),
            "coord_z": target_meta["coord_z"].to_numpy(dtype=np.float64),
            "macro_system": target_meta["macro_system"].astype(str).to_numpy(),
            "hemisphere": target_meta["hemisphere"].astype(str).to_numpy(),
            "is_observed_gtex": global_obs_mask.astype(bool),
            "dist_to_nearest_observed": dist.astype(np.float64),
            "nearest_observed_parcel": nearest_name,
        }
    )
    return out


def fit_piecewise_harmonization(
    ahba_raw: pd.DataFrame,
    gtex_raw: pd.DataFrame,
    genes: List[str],
    target_meta: pd.DataFrame,
    min_group_samples: int,
) -> Dict[str, object]:
    global_cal = smg.fit_harmonizer(
        ahba_raw=ahba_raw,
        gtex_raw=gtex_raw,
        gene_cols=genes,
        n_parcels=len(target_meta),
        model="robustz_affine",
        whiten_eps=1e-4,
    )

    systems = sorted(target_meta["macro_system"].unique().tolist())
    cal_by_system: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []

    for sys in systems:
        a_sub = ahba_raw[ahba_raw["macro_system"] == sys]
        g_sub = gtex_raw[gtex_raw["macro_system"] == sys]
        use_local = len(a_sub) >= min_group_samples and len(g_sub) >= min_group_samples
        if use_local:
            cal = smg.fit_harmonizer(
                ahba_raw=a_sub,
                gtex_raw=g_sub,
                gene_cols=genes,
                n_parcels=len(target_meta),
                model="robustz_affine",
                whiten_eps=1e-4,
            )
            src = "piecewise"
        else:
            cal = global_cal
            src = "global_fallback"
        cal_by_system[sys] = cal
        rows.append(
            {
                "macro_system": sys,
                "ahba_rows": int(len(a_sub)),
                "gtex_rows": int(len(g_sub)),
                "calibration_source": src,
            }
        )

    def _apply_piecewise(df_raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df_raw.copy()
        arr = np.zeros((len(out), len(genes)), dtype=np.float64)
        for sys in systems:
            idx = np.where(out["macro_system"].to_numpy(dtype=object) == sys)[0]
            if idx.size == 0:
                continue
            cal = cal_by_system[sys]
            x = out.iloc[idx][genes].to_numpy(dtype=np.float64)
            arr[idx, :] = smg.harmonize_matrix(x, cal, dataset_name)
        out_genes = pd.DataFrame(arr.astype(np.float32), columns=genes, index=out.index)
        out = pd.concat([out.drop(columns=genes), out_genes], axis=1)
        return out

    def _inverse_piecewise(pred_h: np.ndarray) -> np.ndarray:
        pred_raw = np.zeros_like(pred_h, dtype=np.float64)
        systems_arr = target_meta["macro_system"].astype(str).to_numpy()
        for sys in systems:
            ridx = np.where(systems_arr == sys)[0]
            if ridx.size == 0:
                continue
            cal = cal_by_system[sys]
            pred_raw[ridx, :] = smg.inverse_harmonize_gtex_matrix(pred_h[ridx, :], cal)
        return pred_raw

    return {
        "global_cal": global_cal,
        "cal_by_system": cal_by_system,
        "systems": systems,
        "apply_fn": _apply_piecewise,
        "inverse_fn": _inverse_piecewise,
        "summary_df": pd.DataFrame(rows),
    }


def constrained_nearest_index(
    target_idx: int,
    obs_idx: np.ndarray,
    coords_full: np.ndarray,
    target_meta: pd.DataFrame,
) -> int:
    obs_idx = np.asarray(obs_idx, dtype=np.int32)
    t_hemi = str(target_meta.iloc[target_idx]["hemisphere"])
    t_sys = str(target_meta.iloc[target_idx]["macro_system"])

    cand = obs_idx.copy()
    obs_hemi = target_meta.iloc[cand]["hemisphere"].astype(str).to_numpy()
    obs_sys = target_meta.iloc[cand]["macro_system"].astype(str).to_numpy()

    same_hemi = cand[obs_hemi == t_hemi]
    if same_hemi.size > 0:
        cand = same_hemi
        obs_sys = target_meta.iloc[cand]["macro_system"].astype(str).to_numpy()

    same_sys = cand[obs_sys == t_sys]
    if same_sys.size > 0:
        cand = same_sys

    cxyz = coords_full[cand, :]
    txyz = coords_full[target_idx, :]
    d2 = ((cxyz - txyz[None, :]) ** 2).sum(axis=1)
    return int(cand[int(np.argmin(d2))])


def fit_gp_models(coords_obs: np.ndarray, U_obs: np.ndarray, rbf_length: float) -> List[GaussianProcessRegressor]:
    models = []
    for k in range(U_obs.shape[1]):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=rbf_length, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
            noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)
        )
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=123, optimizer=None)
        gp.fit(coords_obs, U_obs[:, k])
        models.append(gp)
    return models


def gp_predict(models: List[GaussianProcessRegressor], coords_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for mdl in models:
        m, s = mdl.predict(coords_new, return_std=True)
        means.append(m)
        stds.append(s)
    M = np.stack(means, axis=1).astype(np.float64)
    S = np.stack(stds, axis=1).astype(np.float64)
    return M, S


def fit_moe_maps(
    T_obs: np.ndarray,
    U_obs: np.ndarray,
    obs_idx: np.ndarray,
    ahba_ref_T: np.ndarray,
    target_meta: pd.DataFrame,
) -> Tuple[smg.BasisMap, Dict[str, smg.BasisMap], np.ndarray, np.ndarray]:
    k = T_obs.shape[1]
    T_ref_obs = ahba_ref_T[obs_idx, :k]
    global_map = smg.fit_basis_map(T_obs, T_ref_obs, model="affine_gl3")

    sys_arr = target_meta.iloc[obs_idx]["macro_system"].astype(str).to_numpy()
    maps: Dict[str, smg.BasisMap] = {}
    for sys in sorted(set(sys_arr.tolist())):
        pos = np.where(sys_arr == sys)[0]
        if pos.size >= 3:
            maps[sys] = smg.fit_basis_map(T_obs[pos, :], T_ref_obs[pos, :], model="affine_gl3")
        else:
            maps[sys] = global_map

    T_prime = np.zeros_like(T_obs, dtype=np.float64)
    U_prime = np.zeros_like(U_obs, dtype=np.float64)
    for i in range(T_obs.shape[0]):
        sys = sys_arr[i]
        mp = maps.get(sys, global_map)
        T_prime[i, :] = T_obs[i, :] @ mp.M + mp.b
        U_prime[i, :] = U_obs[i, :] @ mp.M

    return global_map, maps, T_prime, U_prime


def invert_moe_scores(
    T_prime_full: np.ndarray,
    target_meta: pd.DataFrame,
    global_map: smg.BasisMap,
    maps: Dict[str, smg.BasisMap],
) -> np.ndarray:
    T_native = np.zeros_like(T_prime_full, dtype=np.float64)
    sys_arr = target_meta["macro_system"].astype(str).to_numpy()
    for r in range(T_prime_full.shape[0]):
        mp = maps.get(sys_arr[r], global_map)
        T_native[r, :] = (T_prime_full[r, :] - mp.b) @ mp.invM
    return T_native


def predict_strategy_full(
    strategy: str,
    obs_idx: np.ndarray,
    X_obs_h: np.ndarray,
    X_obs_raw: np.ndarray,
    coords_full: np.ndarray,
    target_meta: pd.DataFrame,
    ahba_h_full: np.ndarray,
    ahba_ref_T: np.ndarray,
    inverse_fn: Callable[[np.ndarray], np.ndarray],
    cfg: Config,
    return_raw: bool = True,
    apply_anchor_overwrite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    y_obs = np.c_[coords_full[obs_idx, 1], coords_full[obs_idx, 2], np.abs(coords_full[obs_idx, 0])]
    pls = base.fit_subject_pls(X_obs_h, y_obs, n_comp_target=cfg.n_comp_target, adaptive=True)
    k = int(pls["n_comp"])
    T_ref_obs = ahba_ref_T[obs_idx, :k]

    if strategy == "moe_basis":
        global_map, maps, T_prime_obs, U_prime_obs = fit_moe_maps(pls["T"], pls["U"], obs_idx, ahba_ref_T, target_meta)
    else:
        bmap = smg.fit_basis_map(pls["T"], T_ref_obs, model="affine_gl3")
        T_prime_obs = smg.apply_basis_scores(pls["T"], bmap)
        U_prime_obs = smg.apply_basis_linear(pls["U"], bmap)

    coords_obs = coords_full[obs_idx, :]

    if strategy == "constrained_anchor":
        U_full_prime = np.zeros((coords_full.shape[0], U_prime_obs.shape[1]), dtype=np.float64)
        pos_map = {int(p): i for i, p in enumerate(obs_idx.tolist())}
        for r in range(coords_full.shape[0]):
            j = constrained_nearest_index(r, obs_idx, coords_full, target_meta)
            U_full_prime[r, :] = U_prime_obs[pos_map[j], :]
        bridge = base.fit_u_to_t_bridge(U_prime_obs, T_prime_obs, alpha=cfg.ridge_alpha_bridge)
        T_full_prime = U_full_prime @ bridge["M"] + bridge["b"]

    elif strategy == "gp_uncertainty":
        gps = fit_gp_models(coords_obs, U_prime_obs, rbf_length=cfg.gp_rbf_length)
        U_full_prime, U_std_full = gp_predict(gps, coords_full)
        U_obs_pred, U_obs_std = gp_predict(gps, coords_obs)

        sw = 1.0 / (1.0 + np.mean(U_obs_std**2, axis=1))
        ridge = Ridge(alpha=float(cfg.ridge_alpha_bridge), fit_intercept=True)
        ridge.fit(U_obs_pred, T_prime_obs, sample_weight=sw)
        M = ridge.coef_.T.astype(np.float64)
        b = ridge.intercept_.astype(np.float64)
        T_pred = U_full_prime @ M + b

        u = np.mean(U_std_full, axis=1)
        u0 = float(np.median(u[obs_idx])) + 1e-6
        w = u / (u + u0)
        T_ref_full = ahba_ref_T[:, :k]
        T_full_prime = (1.0 - w[:, None]) * T_pred + w[:, None] * T_ref_full

    else:
        spatial = base.fit_spatial_field(coords_obs, U_prime_obs, cfg)
        bridge = base.fit_u_to_t_bridge(U_prime_obs, T_prime_obs, alpha=cfg.ridge_alpha_bridge)
        U_full_prime = base._predict_spatial_field(spatial, coords_full)
        T_full_prime = U_full_prime @ bridge["M"] + bridge["b"]

    if strategy == "moe_basis":
        T_native = invert_moe_scores(T_full_prime, target_meta, global_map, maps)
    else:
        T_native = smg.invert_basis_scores(T_full_prime, bmap)

    X_std = T_native @ pls["P"].T
    X_full_h = X_std * pls["x_scale"] + pls["x_mean"]

    if strategy == "distance_shrink":
        d2 = ((coords_full[:, None, :] - coords_obs[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.min(d2, axis=1))
        w = _sigmoid((d - float(cfg.distance_d0)) / max(float(cfg.distance_tau), 1e-6))
        X_full_h = (1.0 - w[:, None]) * X_full_h + w[:, None] * ahba_h_full

    X_full_h = X_full_h.astype(np.float64)
    if apply_anchor_overwrite:
        X_full_h[obs_idx, :] = X_obs_h

    X_full_raw = None
    if return_raw:
        X_full_raw = inverse_fn(X_full_h)
        X_full_raw = X_full_raw.astype(np.float64)
        if apply_anchor_overwrite:
            X_full_raw[obs_idx, :] = X_obs_raw

    return X_full_h, X_full_raw


def evaluate_subject_loro_strategy(
    strategy: str,
    subject: str,
    obs_idx: np.ndarray,
    X_obs_h: np.ndarray,
    coords_full: np.ndarray,
    target_meta: pd.DataFrame,
    ahba_h_full: np.ndarray,
    ahba_ref_T: np.ndarray,
    inverse_fn: Callable[[np.ndarray], np.ndarray],
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}
    fold_rows = []
    yt_list = []
    yp_list = []
    yb_list = []
    skipped = 0

    for hold in obs_idx.tolist():
        hold = int(hold)
        train_idx = np.asarray([p for p in obs_idx.tolist() if int(p) != hold], dtype=np.int32)
        if len(train_idx) < 3:
            skipped += 1
            continue

        train_pos = [idx_to_pos[int(p)] for p in train_idx.tolist()]
        hold_pos = idx_to_pos[hold]
        X_train_h = X_obs_h[train_pos, :]

        try:
            X_full_h_pred, _ = predict_strategy_full(
                strategy=strategy,
                obs_idx=train_idx,
                X_obs_h=X_train_h,
                X_obs_raw=np.zeros_like(X_train_h),
                coords_full=coords_full,
                target_meta=target_meta,
                ahba_h_full=ahba_h_full,
                ahba_ref_T=ahba_ref_T,
                inverse_fn=inverse_fn,
                cfg=cfg,
                return_raw=False,
                apply_anchor_overwrite=False,
            )
            x_true = X_obs_h[hold_pos, :].astype(np.float64)
            x_pred = X_full_h_pred[hold, :].astype(np.float64)
            x_base = ahba_h_full[hold, :].astype(np.float64)
            met = base._metrics_from_vectors(x_true, x_pred)
            fold_rows.append(
                {
                    "subject": subject,
                    "strategy": strategy,
                    "split": "loro_fold",
                    "holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
                    **met,
                    "baseline_rmse": float(np.sqrt(np.mean((x_true - x_base) ** 2))),
                    "status": "ok",
                }
            )
            yt_list.append(x_true)
            yp_list.append(x_pred)
            yb_list.append(x_base)
        except Exception as e:
            skipped += 1
            fold_rows.append(
                {
                    "subject": subject,
                    "strategy": strategy,
                    "split": "loro_fold",
                    "holdout_parcel_idx": hold,
                    "n_train_parcels": int(len(train_idx)),
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "medae": np.nan,
                    "baseline_rmse": np.nan,
                    "status": f"failed:{str(e)[:120]}",
                }
            )

    folds_df = pd.DataFrame(fold_rows)
    if len(yt_list) == 0:
        summary = {
            "subject": subject,
            "strategy": strategy,
            "split": "loro_summary",
            "n_obs_parcels": int(len(obs_idx)),
            "n_folds": 0,
            "n_skipped_folds": int(skipped),
            "n_points": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
            "baseline_rmse": np.nan,
            "better_than_baseline_rmse": False,
        }
        return folds_df, summary

    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    yb = np.concatenate(yb_list)
    met = base._metrics_from_vectors(yt, yp)
    baseline_rmse = float(np.sqrt(np.mean((yt - yb) ** 2)))
    summary = {
        "subject": subject,
        "strategy": strategy,
        "split": "loro_summary",
        "n_obs_parcels": int(len(obs_idx)),
        "n_folds": int(len(yt_list)),
        "n_skipped_folds": int(skipped),
        "n_points": int(len(yt)),
        **met,
        "baseline_rmse": baseline_rmse,
        "better_than_baseline_rmse": bool(np.isfinite(met["rmse"]) and met["rmse"] < baseline_rmse),
    }
    return folds_df, summary


def compute_region_metrics(pred_h: np.ndarray, ahba_h_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = pred_h.shape[0]
    corr = np.full(n, np.nan, dtype=np.float64)
    rmse = np.full(n, np.nan, dtype=np.float64)
    for r in range(n):
        a = ahba_h_full[r, :]
        p = pred_h[r, :]
        if np.std(a) > 1e-12 and np.std(p) > 1e-12:
            corr[r] = float(np.corrcoef(a, p)[0, 1])
        rmse[r] = float(np.sqrt(np.mean((p - a) ** 2)))
    return corr, rmse


def make_misfit_distance_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    obs = df[df["is_observed_gtex"]]
    unobs = df[~df["is_observed_gtex"]]
    ax.scatter(unobs["dist_to_nearest_observed"], unobs["misfit_1mpearson"], s=28, alpha=0.75, label="unobserved", marker="^")
    ax.scatter(obs["dist_to_nearest_observed"], obs["misfit_1mpearson"], s=34, alpha=0.9, label="observed", marker="o")

    x = df["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
    y = df["misfit_1mpearson"].to_numpy(dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        b1, b0 = np.polyfit(x[m], y[m], deg=1)
        xx = np.linspace(float(np.min(x[m])), float(np.max(x[m])), 200)
        ax.plot(xx, b1 * xx + b0, color="black", lw=2, label=f"fit slope={b1:.4f}")

    ax.set_xlabel("Distance to nearest observed GTEx parcel")
    ax.set_ylabel("Misfit = 1 - region Pearson (harmonized)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_parcel_misfit_brainmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    vals = df["misfit_1mpearson"].to_numpy(dtype=np.float64)
    vmin, vmax = np.nanpercentile(vals, [2, 98]).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sc1 = axes[0].scatter(df["coord_y"], df["coord_z"], c=vals, cmap="RdBu_r", vmin=vmin, vmax=vmax, s=45)
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("Z")
    axes[0].set_title("Y-Z")
    axes[0].grid(True, alpha=0.2)

    sc2 = axes[1].scatter(np.abs(df["coord_x"].to_numpy(dtype=np.float64)), df["coord_z"], c=vals, cmap="RdBu_r", vmin=vmin, vmax=vmax, s=45)
    axes[1].set_xlabel("|X|")
    axes[1].set_ylabel("Z")
    axes[1].set_title("|X|-Z")
    axes[1].grid(True, alpha=0.2)

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Misfit (1 - region Pearson)")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    root = Path(".").resolve()
    csv_path = (root / cfg.csv_path).resolve()
    hvg_path = (root / cfg.hvg_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header = base.load_gene_header_and_hvg(csv_path, hvg_path)
    genes = header["genes_hvg"]
    if len(genes) == 0:
        raise RuntimeError("No HVG overlap found.")

    print("[stage] loading data")
    df = base.read_expression_subset(csv_path, genes)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = base.build_target_parcels(ahba_raw)
    target_meta = add_target_meta(target)

    parcel_lookup = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lookup).astype(np.int32)
    gtex_raw = base.map_gtex_to_target(gtex_raw, target)

    ahba_raw = add_sample_group(ahba_raw, target_meta)
    gtex_raw = add_sample_group(gtex_raw, target_meta)

    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)

    # Global observed mask for distance diagnostics.
    _, global_obs_mask = base.build_region_matrix(gtex_raw, genes, target, agg="mean")
    dist_df = compute_global_distance_table(target_meta, global_obs_mask)

    # Strategy harmonization bundles.
    print("[stage] fitting harmonization bundles")
    robust_cal = smg.fit_harmonizer(
        ahba_raw=ahba_raw,
        gtex_raw=gtex_raw,
        gene_cols=genes,
        n_parcels=len(target),
        model="robustz_affine",
        whiten_eps=1e-4,
    )

    ahba_h_rob = smg.apply_harmonization_df(ahba_raw, genes, robust_cal, "AHBA")
    gtex_h_rob = smg.apply_harmonization_df(gtex_raw, genes, robust_cal, "GTEX")
    ahba_h_full_rob, _ = base.build_region_matrix(ahba_h_rob, genes, target, agg="mean")
    ahba_raw_full, _ = base.build_region_matrix(ahba_raw, genes, target, agg="mean")
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]
    ahba_ref_pls_rob = base.fit_subject_pls(ahba_h_full_rob, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)

    piece = fit_piecewise_harmonization(
        ahba_raw=ahba_raw,
        gtex_raw=gtex_raw,
        genes=genes,
        target_meta=target_meta,
        min_group_samples=cfg.min_group_samples,
    )
    ahba_h_piece = piece["apply_fn"](ahba_raw, "AHBA")
    gtex_h_piece = piece["apply_fn"](gtex_raw, "GTEX")
    ahba_h_full_piece, _ = base.build_region_matrix(ahba_h_piece, genes, target, agg="mean")
    ahba_ref_pls_piece = base.fit_subject_pls(ahba_h_full_piece, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)

    bundle_by_strategy = {}
    for s in cfg.strategies:
        if s == "piecewise_harmonization":
            bundle_by_strategy[s] = {
                "ahba_h": ahba_h_piece,
                "gtex_h": gtex_h_piece,
                "ahba_h_full": ahba_h_full_piece,
                "ahba_ref_T": ahba_ref_pls_piece["T"],
                "inverse_fn": piece["inverse_fn"],
                "harmonization": "piecewise_robustz",
            }
        else:
            bundle_by_strategy[s] = {
                "ahba_h": ahba_h_rob,
                "gtex_h": gtex_h_rob,
                "ahba_h_full": ahba_h_full_rob,
                "ahba_ref_T": ahba_ref_pls_rob["T"],
                "inverse_fn": lambda x, cal=robust_cal: smg.inverse_harmonize_gtex_matrix(x, cal),
                "harmonization": "robustz_affine",
            }

    piece["summary_df"].to_csv(table_dir / "piecewise_harmonization_groups.csv", index=False)

    subjects_all = sorted(gtex_raw["subject"].dropna().astype(str).unique().tolist())
    elig = []
    for sid in subjects_all:
        sub = gtex_raw[gtex_raw["subject"] == sid]
        n_obs = int(sub["parcel_idx"].nunique())
        elig.append({"subject": sid, "n_obs_parcels": n_obs, "eligible": bool(n_obs >= cfg.min_observed_parcels)})
    elig_df = pd.DataFrame(elig)
    elig_df.to_csv(table_dir / "subject_eligibility.csv", index=False)
    eligible_subjects = elig_df[elig_df["eligible"]]["subject"].astype(str).tolist()

    print(f"[stage] eligible subjects={len(eligible_subjects)}")

    fold_rows = []
    summary_rows = []
    parcel_rows = []
    aggregate_cache = {}

    # Run each strategy end-to-end.
    for strategy in cfg.strategies:
        print(f"[strategy] {strategy}")
        bundle = bundle_by_strategy[strategy]
        gtex_h = bundle["gtex_h"]
        ahba_h_full = bundle["ahba_h_full"]
        ahba_ref_T = bundle["ahba_ref_T"]
        inverse_fn = bundle["inverse_fn"]

        pred_h_list = []
        pred_raw_list = []
        pred_sids = []

        for i, sid in enumerate(eligible_subjects):
            subj_h = gtex_h[gtex_h["subject"] == sid].copy()
            subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
            obs_idx, x_obs_h, x_obs_raw = base.build_subject_observed_matrices(subj_h, subj_raw, genes)
            if len(obs_idx) < cfg.min_observed_parcels:
                continue

            try:
                x_full_h, x_full_raw = predict_strategy_full(
                    strategy=strategy,
                    obs_idx=obs_idx,
                    X_obs_h=x_obs_h,
                    X_obs_raw=x_obs_raw,
                    coords_full=coords_full,
                    target_meta=target_meta,
                    ahba_h_full=ahba_h_full,
                    ahba_ref_T=ahba_ref_T,
                    inverse_fn=inverse_fn,
                    cfg=cfg,
                    return_raw=True,
                    apply_anchor_overwrite=True,
                )
                pred_h_list.append(x_full_h.astype(np.float32))
                pred_raw_list.append(x_full_raw.astype(np.float32))
                pred_sids.append(sid)

                folds_df, summ = evaluate_subject_loro_strategy(
                    strategy=strategy,
                    subject=sid,
                    obs_idx=obs_idx,
                    X_obs_h=x_obs_h,
                    coords_full=coords_full,
                    target_meta=target_meta,
                    ahba_h_full=ahba_h_full,
                    ahba_ref_T=ahba_ref_T,
                    inverse_fn=inverse_fn,
                    cfg=cfg,
                )
                if len(folds_df):
                    fold_rows.append(folds_df)
                summary_rows.append(summ)

            except Exception as e:
                summary_rows.append(
                    {
                        "subject": sid,
                        "strategy": strategy,
                        "split": "loro_summary",
                        "n_obs_parcels": int(len(obs_idx)),
                        "n_folds": 0,
                        "n_skipped_folds": int(len(obs_idx)),
                        "n_points": 0,
                        "pearson_r": np.nan,
                        "spearman_rho": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "medae": np.nan,
                        "baseline_rmse": np.nan,
                        "better_than_baseline_rmse": False,
                        "status": f"failed:{str(e)[:120]}",
                    }
                )

            if (i + 1) % 40 == 0:
                print(f"  processed {i+1}/{len(eligible_subjects)} subjects")

        if len(pred_h_list) == 0:
            continue

        pred_h_tensor = np.stack(pred_h_list, axis=0).astype(np.float32)
        pred_raw_tensor = np.stack(pred_raw_list, axis=0).astype(np.float32)
        agg_h = np.nanmean(pred_h_tensor, axis=0).astype(np.float64)
        agg_raw = np.nanmean(pred_raw_tensor, axis=0).astype(np.float64)
        region_corr, region_rmse = compute_region_metrics(agg_h, ahba_h_full)

        strategy_parcel = dist_df.copy()
        strategy_parcel["strategy"] = strategy
        strategy_parcel["region_pearson_h"] = region_corr
        strategy_parcel["misfit_1mpearson"] = 1.0 - region_corr
        strategy_parcel["residual_rmse_h"] = region_rmse
        parcel_rows.append(strategy_parcel)

        aggregate_cache[strategy] = {
            "agg_h": agg_h,
            "agg_raw": agg_raw,
            "ahba_h_full": ahba_h_full,
            "subjects": pred_sids,
        }

        # Save aggregate matrices per strategy.
        base.matrix_to_aligned_table(agg_raw, genes, dist_df[["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]]).to_csv(
            table_dir / f"aggregate_hvg_{strategy}_mean_raw.csv", index=False
        )
        base.matrix_to_aligned_table(agg_h, genes, dist_df[["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z", "is_observed_gtex"]]).to_csv(
            table_dir / f"aggregate_hvg_{strategy}_mean_harmonized.csv", index=False
        )

    # Assemble outputs.
    folds_all_df = pd.concat(fold_rows, ignore_index=True) if len(fold_rows) else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    parcel_all_df = pd.concat(parcel_rows, ignore_index=True) if len(parcel_rows) else pd.DataFrame()

    folds_all_df.to_csv(table_dir / "subject_loro_folds_by_strategy.csv", index=False)
    summary_df.to_csv(table_dir / "subject_loro_summary_by_strategy.csv", index=False)
    parcel_all_df.to_csv(table_dir / "parcel_misfit_distance_by_strategy.csv", index=False)

    baseline_parcel = parcel_all_df[parcel_all_df["strategy"] == "baseline"].copy()
    baseline_parcel.to_csv(table_dir / "parcel_misfit_distance_baseline.csv", index=False)

    # Distance-effect summary.
    drows = []
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    for strategy, d in parcel_all_df.groupby("strategy"):
        x = d["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
        y = d["misfit_1mpearson"].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3:
            pear = float(np.corrcoef(x[m], y[m])[0, 1])
            spear = float(stats.spearmanr(x[m], y[m]).statistic)
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
        else:
            pear, spear, b1, b0 = np.nan, np.nan, np.nan, np.nan

        q = pd.qcut(pd.Series(x, index=d.index), q=4, labels=q_labels, duplicates="drop")
        qmean = d.groupby(q, observed=False)["misfit_1mpearson"].mean().to_dict()
        row = {
            "strategy": strategy,
            "pearson_misfit_distance": pear,
            "spearman_misfit_distance": spear,
            "linear_slope": b1,
            "linear_intercept": b0,
            "q1_mean_misfit": float(qmean.get("Q1", np.nan)),
            "q2_mean_misfit": float(qmean.get("Q2", np.nan)),
            "q3_mean_misfit": float(qmean.get("Q3", np.nan)),
            "q4_mean_misfit": float(qmean.get("Q4", np.nan)),
        }
        drows.append(row)
    dist_eff_df = pd.DataFrame(drows)
    dist_eff_df.to_csv(table_dir / "distance_effect_summary.csv", index=False)

    # Best/worst parcels.
    best_rows = []
    worst_rows = []
    for strategy, d in parcel_all_df.groupby("strategy"):
        best_rows.append(d.nsmallest(10, "misfit_1mpearson"))
        worst_rows.append(d.nlargest(10, "misfit_1mpearson"))
    pd.concat(best_rows, ignore_index=True).to_csv(table_dir / "best_misfit_parcels_by_strategy.csv", index=False)
    pd.concat(worst_rows, ignore_index=True).to_csv(table_dir / "worst_misfit_parcels_by_strategy.csv", index=False)

    # Baseline-vs-strategy subject comparison.
    summ = summary_df[summary_df["split"] == "loro_summary"].copy()
    base_sub = summ[summ["strategy"] == "baseline"][["subject", "rmse", "pearson_r"]].rename(
        columns={"rmse": "baseline_subject_rmse", "pearson_r": "baseline_subject_pearson"}
    )
    summ2 = summ.merge(base_sub, on="subject", how="left")
    summ2["better_than_baseline_subject_rmse"] = summ2["rmse"] < summ2["baseline_subject_rmse"]
    summ2.to_csv(table_dir / "subject_loro_summary_with_baseline_compare.csv", index=False)

    sgrp = (
        summ2.groupby("strategy", as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_pearson=("pearson_r", "mean"),
            median_pearson=("pearson_r", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            frac_better_than_baseline_rmse=("better_than_baseline_subject_rmse", "mean"),
        )
        .reset_index(drop=True)
    )

    sgrp = sgrp.merge(dist_eff_df[["strategy", "linear_slope"]], on="strategy", how="left")
    sgrp["abs_distance_slope"] = np.abs(sgrp["linear_slope"])
    sgrp = sgrp.sort_values(
        ["abs_distance_slope", "mean_pearson", "mean_rmse", "frac_better_than_baseline_rmse"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    sgrp["rank"] = np.arange(1, len(sgrp) + 1, dtype=np.int32)
    sgrp.to_csv(table_dir / "strategy_ranking.csv", index=False)

    # Figures
    print("[stage] generating figures")
    make_misfit_distance_plot(
        baseline_parcel,
        fig_dir / "misfit_vs_distance_baseline.png",
        "Baseline misfit vs distance (robustz_affine + affine_gl3)",
    )
    make_parcel_misfit_brainmap(
        baseline_parcel,
        fig_dir / "parcel_misfit_brainmap_baseline.png",
        "Baseline parcel misfit map",
    )

    # By-strategy misfit-vs-distance and residual-vs-distance.
    order = sgrp["strategy"].tolist()
    n = len(order)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, st in enumerate(order):
        ax = axes[i]
        d = parcel_all_df[parcel_all_df["strategy"] == st]
        ax.scatter(d["dist_to_nearest_observed"], d["misfit_1mpearson"], s=20, alpha=0.75)
        x = d["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
        y = d["misfit_1mpearson"].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, b1 * xx + b0, color="black", lw=1.5)
        ax.set_title(st)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Misfit")
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.savefig(fig_dir / "misfit_vs_distance_by_strategy.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, st in enumerate(order):
        ax = axes[i]
        d = parcel_all_df[parcel_all_df["strategy"] == st]
        ax.scatter(d["dist_to_nearest_observed"], d["residual_rmse_h"], s=20, alpha=0.75)
        x = d["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
        y = d["residual_rmse_h"].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            b1, b0 = np.polyfit(x[m], y[m], deg=1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, b1 * xx + b0, color="black", lw=1.5)
        ax.set_title(st)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Residual RMSE")
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.savefig(fig_dir / "residual_vs_distance_by_strategy.png", dpi=220)
    plt.close(fig)

    # Brain map grid by strategy (Y-Z view)
    vals_all = parcel_all_df["misfit_1mpearson"].to_numpy(dtype=np.float64)
    vmin, vmax = np.nanpercentile(vals_all, [2, 98]).tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, st in enumerate(order):
        ax = axes[i]
        d = parcel_all_df[parcel_all_df["strategy"] == st]
        sc = ax.scatter(d["coord_y"], d["coord_z"], c=d["misfit_1mpearson"], cmap="RdBu_r", vmin=vmin, vmax=vmax, s=28)
        ax.set_title(st)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.85)
    cbar.set_label("Misfit")
    fig.savefig(fig_dir / "parcel_misfit_brainmap_by_strategy.png", dpi=220)
    plt.close(fig)

    # Subject metric violins.
    vv = summ2.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.violinplot(data=vv, x="strategy", y="pearson_r", order=order, ax=axes[0], inner="quartile", cut=0)
    axes[0].set_title("Subject LORO Pearson by strategy")
    axes[0].tick_params(axis="x", rotation=30)
    sns.violinplot(data=vv, x="strategy", y="rmse", order=order, ax=axes[1], inner="quartile", cut=0)
    axes[1].set_title("Subject LORO RMSE by strategy")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(fig_dir / "subject_metric_violin_by_strategy.png", dpi=220)
    plt.close(fig)

    # Ranking overview.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    sns.barplot(data=sgrp, x="strategy", y="abs_distance_slope", order=order, ax=axes[0])
    axes[0].set_title("|Distance slope| (lower better)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=sgrp, x="strategy", y="mean_pearson", order=order, ax=axes[1])
    axes[1].set_title("Mean subject Pearson (higher better)")
    axes[1].tick_params(axis="x", rotation=30)

    sns.barplot(data=sgrp, x="strategy", y="mean_rmse", order=order, ax=axes[2])
    axes[2].set_title("Mean subject RMSE (lower better)")
    axes[2].tick_params(axis="x", rotation=30)
    fig.savefig(fig_dir / "strategy_ranking_overview.png", dpi=220)
    plt.close(fig)

    # Summary + report.
    best = sgrp.iloc[0].to_dict() if len(sgrp) else {}
    summary = {
        "config": asdict(cfg),
        "counts": {
            "n_subjects_total": int(len(subjects_all)),
            "n_subjects_eligible": int(len(eligible_subjects)),
            "n_strategies": int(len(order)),
            "n_parcels": int(len(target_meta)),
            "n_genes_hvg": int(len(genes)),
        },
        "best_strategy": best,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# Misfit Distance Mitigation Report

## Run Summary
- Eligible subjects: {len(eligible_subjects)} / {len(subjects_all)}
- Strategies: {', '.join(order)}
- HVGs: {len(genes)}
- Parcels: {len(target_meta)}

## Baseline Diagnostic
- Baseline: robustz+affine (`baseline`)
- Misfit metric: `1 - region Pearson (harmonized)`
- Diagnostic figures:
  - `figures/misfit_vs_distance_baseline.png`
  - `figures/parcel_misfit_brainmap_baseline.png`

## Strategy Ranking (rule: lower |distance slope|, higher Pearson, lower RMSE, higher better-than-baseline)

```text
{sgrp.to_string(index=False)}
```

## Key outputs
- `tables/parcel_misfit_distance_baseline.csv`
- `tables/parcel_misfit_distance_by_strategy.csv`
- `tables/subject_loro_summary_by_strategy.csv`
- `tables/strategy_ranking.csv`
- `tables/distance_effect_summary.csv`
"""
    (out_root / "report.md").write_text(report)

    print("Completed misfit-distance diagnostics and mitigation benchmark.")
    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
