from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.latent.basis_maps import apply_linear, apply_scores, fit_basis_map, invert_scores
from src.latent.pls import fit_subject_pls
from src.spatial import gp as gp_spatial
from src.spatial import rbf as rbf_spatial


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_u_to_t_bridge(U_prime_obs: np.ndarray, T_prime_obs: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    ridge.fit(U_prime_obs, T_prime_obs, sample_weight=sample_weight)
    return {"M": ridge.coef_.T.astype(np.float64), "b": ridge.intercept_.astype(np.float64)}


def constrained_nearest_index(target_idx: int, obs_idx: np.ndarray, coords_full: np.ndarray, target_meta: pd.DataFrame) -> int:
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


def fit_moe_maps(T_obs: np.ndarray, U_obs: np.ndarray, obs_idx: np.ndarray, ahba_ref_T: np.ndarray, target_meta: pd.DataFrame):
    k = T_obs.shape[1]
    T_ref_obs = ahba_ref_T[obs_idx, :k]
    global_map = fit_basis_map(T_obs, T_ref_obs, model="affine_gl3")

    sys_arr = target_meta.iloc[obs_idx]["macro_system"].astype(str).to_numpy()
    maps = {}
    for sys in sorted(set(sys_arr.tolist())):
        pos = np.where(sys_arr == sys)[0]
        if pos.size >= 3:
            maps[sys] = fit_basis_map(T_obs[pos, :], T_ref_obs[pos, :], model="affine_gl3")
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


def invert_moe_scores(T_prime_full: np.ndarray, target_meta: pd.DataFrame, global_map, maps) -> np.ndarray:
    T_native = np.zeros_like(T_prime_full, dtype=np.float64)
    sys_arr = target_meta["macro_system"].astype(str).to_numpy()
    for r in range(T_prime_full.shape[0]):
        mp = maps.get(sys_arr[r], global_map)
        T_native[r, :] = (T_prime_full[r, :] - mp.b) @ mp.invM
    return T_native


def _fit_spatial(U_prime_obs: np.ndarray, coords_obs: np.ndarray, method_bundle: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray | None]:
    spatial_method = str(method_bundle.get("spatial_method", "rbf")).lower()
    if spatial_method == "gp":
        gp_model = gp_spatial.fit_spatial(
            coords_obs,
            U_prime_obs,
            length_scale=float(method_bundle.get("gp_rbf_length", 25.0)),
            random_state=int(method_bundle.get("seed", 123)),
        )
        return gp_spatial.predict(gp_model, method_bundle["coords_full"])

    rbf_model = rbf_spatial.fit_spatial(coords_obs, U_prime_obs, smoothing=float(method_bundle.get("rbf_smoothing", 0.10)))
    pred, _ = rbf_spatial.predict(rbf_model, method_bundle["coords_full"])
    return pred, None


def run_subject(
    subject_bundle: Dict[str, object],
    atlas_bundle: Dict[str, object],
    method_bundle: Dict[str, object],
    cfg: Dict[str, object],
    fold_mask: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    subject = str(subject_bundle["subject"])
    obs_idx_full = np.asarray(subject_bundle["obs_idx"], dtype=np.int32)
    X_obs_h_full = np.asarray(subject_bundle["X_obs_h"], dtype=np.float64)
    X_obs_raw_full = np.asarray(subject_bundle["X_obs_raw"], dtype=np.float64)
    coords_full = np.asarray(subject_bundle["coords_full"], dtype=np.float64)
    target_meta = subject_bundle["target_meta"]
    harmonizer = method_bundle["harmonizer"]
    inverse_df = subject_bundle.get("inverse_df")

    idx_to_pos = {int(p): i for i, p in enumerate(obs_idx_full.tolist())}
    if fold_mask is None:
        train_idx = obs_idx_full.copy()
        overwrite_idx = obs_idx_full.copy()
    else:
        train_idx = np.asarray([p for p in obs_idx_full.tolist() if int(p) != int(fold_mask)], dtype=np.int32)
        overwrite_idx = train_idx.copy()

    train_pos = np.asarray([idx_to_pos[int(p)] for p in train_idx.tolist()], dtype=np.int32)
    X_train_h = X_obs_h_full[train_pos, :]
    X_train_raw = X_obs_raw_full[train_pos, :]
    coords_obs = coords_full[train_idx, :]

    c_min = int(method_bundle.get("c_min", cfg.get("c_min", 8)))
    deployment_mode = "model"
    if len(train_idx) < c_min:
        deployment_mode = "atlas_prior"
        X_h = np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64).copy()
        subj_ids = np.asarray([subject] * X_h.shape[0], dtype=object)
        try:
            X_raw = harmonizer.inverse_gtex(X_h, subject_ids=subj_ids, sample_df=inverse_df)
        except TypeError:
            X_raw = harmonizer.inverse_gtex(X_h)
        X_h[overwrite_idx, :] = X_train_h
        X_raw[overwrite_idx, :] = X_train_raw
        return {"X_full_h": X_h, "X_full_raw": X_raw}, {
            "deployment_mode": deployment_mode,
            "n_train_obs": int(len(train_idx)),
            "leak_flag": False,
            "uvar": np.zeros(X_h.shape[0], dtype=np.float64),
        }

    Y_train = np.c_[coords_full[train_idx, 1], coords_full[train_idx, 2], np.abs(coords_full[train_idx, 0])]
    pls = fit_subject_pls(X_train_h, Y_train, n_comp_target=int(method_bundle.get("n_comp_target", 3)), adaptive=True)
    k = int(pls["n_comp"])

    strategy = str(method_bundle.get("strategy", "baseline"))
    basis_model = str(method_bundle.get("basis_model", "affine_gl3"))
    T_ref_obs = np.asarray(atlas_bundle["ahba_ref_T"], dtype=np.float64)[train_idx, :k]

    moe_maps = None
    if strategy == "moe_basis":
        global_map, maps, T_prime_obs, U_prime_obs = fit_moe_maps(pls["T"], pls["U"], train_idx, np.asarray(atlas_bundle["ahba_ref_T"]), target_meta)
        moe_maps = (global_map, maps)
    else:
        bmap = fit_basis_map(pls["T"], T_ref_obs, model=basis_model)
        T_prime_obs = apply_scores(pls["T"], bmap)
        U_prime_obs = apply_linear(pls["U"], bmap)

    U_full_prime = None
    U_std_full = None

    if strategy == "constrained_anchor":
        U_full_prime = np.zeros((coords_full.shape[0], U_prime_obs.shape[1]), dtype=np.float64)
        pos_map = {int(p): i for i, p in enumerate(train_idx.tolist())}
        for r in range(coords_full.shape[0]):
            j = constrained_nearest_index(r, train_idx, coords_full, target_meta)
            U_full_prime[r, :] = U_prime_obs[pos_map[j], :]
    else:
        mb = dict(method_bundle)
        mb["coords_full"] = coords_full
        if strategy == "gp_uncertainty":
            mb["spatial_method"] = "gp"
        U_full_prime, U_std_full = _fit_spatial(U_prime_obs, coords_obs, mb)

    if strategy == "gp_uncertainty" and U_std_full is not None:
        # uncertainty-aware bridge on observed points
        mb2 = dict(method_bundle)
        mb2["coords_full"] = coords_obs
        U_obs_pred, U_obs_std = _fit_spatial(U_prime_obs, coords_obs, {**mb2, "spatial_method": "gp"})
        sw = 1.0 / (1.0 + np.mean(U_obs_std**2, axis=1))
        bridge = fit_u_to_t_bridge(U_obs_pred, T_prime_obs, alpha=float(method_bundle.get("ridge_alpha_bridge", 1e-2)), sample_weight=sw)
        T_pred = U_full_prime @ bridge["M"] + bridge["b"]
        u = np.mean(U_std_full, axis=1)
        u0 = float(np.median(u[train_idx])) + 1e-6
        w = u / (u + u0)
        T_ref_full = np.asarray(atlas_bundle["ahba_ref_T"], dtype=np.float64)[:, :k]
        T_full_prime = (1.0 - w[:, None]) * T_pred + w[:, None] * T_ref_full
    else:
        bridge = fit_u_to_t_bridge(U_prime_obs, T_prime_obs, alpha=float(method_bundle.get("ridge_alpha_bridge", 1e-2)))
        T_full_prime = U_full_prime @ bridge["M"] + bridge["b"]

    if strategy == "moe_basis":
        global_map, maps = moe_maps
        T_native = invert_moe_scores(T_full_prime, target_meta, global_map, maps)
    else:
        T_native = invert_scores(T_full_prime, bmap)

    X_std = T_native @ pls["P"].T
    X_full_h = X_std * pls["x_scale"] + pls["x_mean"]

    # Classic distance shrink mitigation.
    if strategy == "distance_shrink":
        d2 = ((coords_full[:, None, :] - coords_obs[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.min(d2, axis=1))
        d0 = float(method_bundle.get("distance_d0", 45.0))
        tau = max(float(method_bundle.get("distance_tau", 10.0)), 1e-6)
        w = _sigmoid((d - d0) / tau)
        X_full_h = (1.0 - w[:, None]) * X_full_h + w[:, None] * np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64)

    # Unified uncertainty-aware shrinkage (distance + uncertainty).
    if bool(method_bundle.get("uncertainty_shrink", False)):
        d2 = ((coords_full[:, None, :] - coords_obs[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.min(d2, axis=1))
        d_scaled = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-8)
        if U_std_full is None:
            uvar = np.zeros_like(d_scaled)
        else:
            uvar = np.mean(U_std_full**2, axis=1)
            uvar = (uvar - np.min(uvar)) / (np.max(uvar) - np.min(uvar) + 1e-8)
        alpha = float(method_bundle.get("unc_alpha", 0.5))
        beta = float(method_bundle.get("unc_beta", 0.5))
        m0 = float(method_bundle.get("unc_m0", 0.5))
        tau = max(float(method_bundle.get("unc_tau", 0.15)), 1e-6)
        score = alpha * d_scaled + beta * uvar
        w = _sigmoid((score - m0) / tau)
        X_full_h = (1.0 - w[:, None]) * X_full_h + w[:, None] * np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64)
    else:
        uvar = np.mean(U_std_full**2, axis=1) if U_std_full is not None else np.zeros(coords_full.shape[0], dtype=np.float64)

    X_full_h = X_full_h.astype(np.float64)
    subj_ids = np.asarray([subject] * X_full_h.shape[0], dtype=object)
    try:
        X_full_raw = harmonizer.inverse_gtex(X_full_h, subject_ids=subj_ids, sample_df=inverse_df)
    except TypeError:
        X_full_raw = harmonizer.inverse_gtex(X_full_h)

    X_full_h[overwrite_idx, :] = X_train_h
    X_full_raw[overwrite_idx, :] = X_train_raw

    return {"X_full_h": X_full_h, "X_full_raw": X_full_raw}, {
        "deployment_mode": deployment_mode,
        "n_train_obs": int(len(train_idx)),
        "n_comp": int(k),
        "leak_flag": False,
        "uvar": np.asarray(uvar, dtype=np.float64),
        "strategy": strategy,
        "basis_model": basis_model,
    }
