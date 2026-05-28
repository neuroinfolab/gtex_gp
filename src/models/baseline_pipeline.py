from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.latent.basis_maps import apply_linear, apply_scores, fit_basis_map, invert_scores
from src.latent.pls import fit_pls_basis, fit_subject_pls, project_pls_scores, project_pls_y_scores, safe_std
from src.spatial import gp as gp_spatial
from src.spatial import rbf as rbf_spatial
from src.spatial.model_coords import model_spatial_coords, should_fold_hemispheres


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pls_spatial_coords(coords_full: np.ndarray) -> np.ndarray:
    return model_spatial_coords(coords_full, fold_hemispheres=True)


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
            noise_level=float(method_bundle.get("gp_noise", 1e-3)),
            alpha=float(method_bundle.get("gp_jitter", 1e-6)),
            optimize=bool(method_bundle.get("gp_optimize", True)),
            n_restarts_optimizer=int(method_bundle.get("gp_n_restarts", 0)),
            random_state=int(method_bundle.get("seed", 123)),
        )
        return gp_spatial.predict(gp_model, method_bundle["coords_full"])

    rbf_model = rbf_spatial.fit_spatial(coords_obs, U_prime_obs, smoothing=float(method_bundle.get("rbf_smoothing", 0.10)))
    pred, _ = rbf_spatial.predict(rbf_model, method_bundle["coords_full"])
    return pred, None


def _fit_residual_interp(
    resid_obs: np.ndarray,
    coords_obs: np.ndarray,
    coords_full: np.ndarray,
    method_bundle: Dict[str, object],
) -> np.ndarray:
    """Interpolate the t-prior residual (coords -> latent score deviation).

    The residual must revert to 0 off-support so the prediction falls back to the
    atlas prior. Default kernel is an inverse-multiquadric RBF (decaying, no
    polynomial tail). ``t_prior_interp`` switches between rbf kernels and a GP;
    the GP is offered but the marginal-likelihood optimizer overfits on the
    handful of shared parcels, so it defaults to a fixed (non-optimized) fit.
    """
    mode = str(method_bundle.get("t_prior_interp", "imq")).lower()
    length_scale = float(method_bundle.get("t_prior_length_scale", 25.0))

    if mode == "gp":
        gp_model = gp_spatial.fit_spatial(
            coords_obs,
            resid_obs,
            length_scale=length_scale,
            noise_level=float(method_bundle.get("t_prior_gp_noise", 1e-2)),
            alpha=float(method_bundle.get("gp_jitter", 1e-6)),
            optimize=bool(method_bundle.get("t_prior_gp_optimize", False)),
            n_restarts_optimizer=int(method_bundle.get("gp_n_restarts", 0)),
            random_state=int(method_bundle.get("seed", 123)),
        )
        pred, _ = gp_spatial.predict(gp_model, coords_full)
        return pred

    smoothing = float(method_bundle.get("rbf_smoothing", 0.10))
    if mode in ("tps", "thin_plate_spline"):
        rbf_model = rbf_spatial.fit_spatial(coords_obs, resid_obs, smoothing=smoothing)
    elif mode in ("gaussian", "imq", "inverse_multiquadric"):
        kernel = "gaussian" if mode == "gaussian" else "inverse_multiquadric"
        eps = 1.0 / max(length_scale, 1e-6)
        rbf_model = rbf_spatial.fit_spatial(
            coords_obs, resid_obs, smoothing=smoothing, kernel=kernel, epsilon=eps, degree=-1
        )
    else:
        raise ValueError(f"unknown t_prior_interp={mode!r}; expected imq|gaussian|tps|gp")
    pred, _ = rbf_spatial.predict(rbf_model, coords_full)
    return pred


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
    coords_full = np.asarray(subject_bundle["coords_full"], dtype=np.float64)
    target_meta = subject_bundle["target_meta"]

    idx_to_pos = {int(p): i for i, p in enumerate(obs_idx_full.tolist())}
    if fold_mask is None:
        train_idx = obs_idx_full.copy()
        overwrite_idx = obs_idx_full.copy()
    else:
        train_idx = np.asarray([p for p in obs_idx_full.tolist() if int(p) != int(fold_mask)], dtype=np.int32)
        overwrite_idx = train_idx.copy()

    train_pos = np.asarray([idx_to_pos[int(p)] for p in train_idx.tolist()], dtype=np.int32)
    X_train_h = X_obs_h_full[train_pos, :]
    fold_hemispheres = bool(method_bundle.get("fold_hemispheres", True))
    if "fold_hemispheres" not in method_bundle:
        fold_hemispheres = should_fold_hemispheres(
            cfg.get("gtex_hemi_mode") if isinstance(cfg, dict) else None,
            cfg.get("matching_policy_hemi_mode") if isinstance(cfg, dict) else None,
        )
    spatial_full = model_spatial_coords(coords_full, fold_hemispheres=fold_hemispheres)
    spatial_obs = spatial_full[train_idx, :]

    c_min = int(method_bundle.get("c_min", cfg.get("c_min", 8)))
    deployment_mode = "model"
    if len(train_idx) < c_min:
        deployment_mode = "atlas_prior"
        X_h = np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64).copy()
        X_h[overwrite_idx, :] = X_train_h
        return {"X_full_h": X_h}, {
            "deployment_mode": deployment_mode,
            "n_train_obs": int(len(train_idx)),
            "leak_flag": False,
            "uvar": np.zeros(X_h.shape[0], dtype=np.float64),
        }

    strategy = str(method_bundle.get("strategy", "baseline"))
    basis_model = str(method_bundle.get("basis_model", "affine_gl3"))

    # The t-prior residual model encodes the atlas through the *subject's* basis,
    # so it needs the retained PLS estimator (fit_pls_basis) to project new genes.
    if strategy == "t_prior_residual":
        ahba_h_full = np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64)
        # The subject's per-gene std (few parcels) is unreliable for low-variance
        # genes and explodes the out-of-sample atlas encode. Floor it by the atlas
        # std (well-estimated over all parcels) so encode/decode stay stable.
        if bool(method_bundle.get("t_prior_atlas_scale_floor", True)):
            x_scale_use = np.maximum(safe_std(X_train_h), safe_std(ahba_h_full))
        else:
            x_scale_use = None
        pls = fit_pls_basis(X_train_h, spatial_obs, n_comp_target=int(method_bundle.get("n_comp_target", 3)), adaptive=True, x_scale=x_scale_use)
    else:
        pls = fit_subject_pls(X_train_h, spatial_obs, n_comp_target=int(method_bundle.get("n_comp_target", 3)), adaptive=True)
    k = int(pls["n_comp"])

    if strategy == "t_prior_residual":
        # Atlas prior, encoded into the subject's own latent frame (no foreign PLS,
        # no affine alignment): T_ahba lives where P decodes.
        T_ahba = np.asarray(project_pls_scores(pls, ahba_h_full), dtype=np.float64)[:, :k]
        T_obs = np.asarray(pls["T"], dtype=np.float64)
        # Subject deviation from the atlas, measured only at the observed parcels,
        # interpolated by a decaying kernel that reverts to 0 (-> atlas) off-support.
        resid_obs = T_obs - T_ahba[train_idx, :]
        resid_full = _fit_residual_interp(resid_obs, spatial_obs, spatial_full, method_bundle)
        T_full = T_ahba + resid_full
        X_std = T_full @ pls["P"].T
        X_full_h = (X_std * pls["x_scale"] + pls["x_mean"]).astype(np.float64)
        X_full_h[overwrite_idx, :] = X_train_h
        return {"X_full_h": X_full_h}, {
            "deployment_mode": deployment_mode,
            "n_train_obs": int(len(train_idx)),
            "n_comp": int(k),
            "leak_flag": False,
            "uvar": np.zeros(coords_full.shape[0], dtype=np.float64),
            "strategy": strategy,
            "basis_model": "subject_pls",
        }

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

    if strategy == "reference_residual":
        ahba_ref_U_raw = atlas_bundle.get("ahba_ref_U")
        if ahba_ref_U_raw is None:
            raise ValueError("dlam_strategy='reference_residual' requires atlas_bundle['ahba_ref_U']")
        ahba_ref_U = np.asarray(ahba_ref_U_raw, dtype=np.float64)[:, :k]

        n_parcels = int(coords_full.shape[0])
        if fold_mask is None:
            ref_idx = np.arange(n_parcels, dtype=np.int32)
        else:
            ref_idx = np.asarray([p for p in range(n_parcels) if int(p) != int(fold_mask)], dtype=np.int32)

        mb_ref = dict(method_bundle)
        mb_ref["coords_full"] = spatial_full
        U_ref_full, U_ref_std = _fit_spatial(ahba_ref_U[ref_idx, :], spatial_full[ref_idx, :], mb_ref)

        delta_U_obs = U_prime_obs - U_ref_full[train_idx, :]
        mb_delta = dict(method_bundle)
        mb_delta["coords_full"] = spatial_full
        delta_U_full, delta_U_std = _fit_spatial(delta_U_obs, spatial_obs, mb_delta)

        U_full_prime = U_ref_full + delta_U_full
        if U_ref_std is not None and delta_U_std is not None:
            U_std_full = np.sqrt(np.maximum(U_ref_std, 0.0) ** 2 + np.maximum(delta_U_std, 0.0) ** 2)
        elif delta_U_std is not None:
            U_std_full = delta_U_std
        else:
            U_std_full = U_ref_std
    elif strategy == "pls_linear_projection":
        U_full = project_pls_y_scores(pls, spatial_full)
        U_full_prime = apply_linear(U_full, bmap)
    elif strategy == "constrained_anchor":
        U_full_prime = np.zeros((coords_full.shape[0], U_prime_obs.shape[1]), dtype=np.float64)
        pos_map = {int(p): i for i, p in enumerate(train_idx.tolist())}
        for r in range(coords_full.shape[0]):
            j = constrained_nearest_index(r, train_idx, coords_full, target_meta)
            U_full_prime[r, :] = U_prime_obs[pos_map[j], :]
    else:
        mb = dict(method_bundle)
        mb["coords_full"] = spatial_full
        if strategy == "gp_uncertainty":
            mb["spatial_method"] = "gp"
        U_full_prime, U_std_full = _fit_spatial(U_prime_obs, spatial_obs, mb)

    if strategy == "gp_uncertainty" and U_std_full is not None:
        # uncertainty-aware bridge on observed points
        mb2 = dict(method_bundle)
        mb2["coords_full"] = spatial_obs
        U_obs_pred, U_obs_std = _fit_spatial(U_prime_obs, spatial_obs, {**mb2, "spatial_method": "gp"})
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
        d2 = ((spatial_full[:, None, :] - spatial_obs[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.min(d2, axis=1))
        d0 = float(method_bundle.get("distance_d0", 45.0))
        tau = max(float(method_bundle.get("distance_tau", 10.0)), 1e-6)
        w = _sigmoid((d - d0) / tau)
        X_full_h = (1.0 - w[:, None]) * X_full_h + w[:, None] * np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64)

    if strategy == "constrained_anchor" and bool(method_bundle.get("anchor_distance_shrink", False)):
        d2 = ((spatial_full[:, None, :] - spatial_obs[None, :, :]) ** 2).sum(axis=2)
        d = np.sqrt(np.min(d2, axis=1))
        d0 = float(method_bundle.get("anchor_distance_d0", 30.0))
        tau = max(float(method_bundle.get("anchor_distance_tau", 8.0)), 1e-6)
        w = _sigmoid((d - d0) / tau)
        X_full_h = (1.0 - w[:, None]) * X_full_h + w[:, None] * np.asarray(atlas_bundle["ahba_h_full"], dtype=np.float64)

    # Unified uncertainty-aware shrinkage (distance + uncertainty).
    if bool(method_bundle.get("uncertainty_shrink", False)):
        d2 = ((spatial_full[:, None, :] - spatial_obs[None, :, :]) ** 2).sum(axis=2)
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
    X_full_h[overwrite_idx, :] = X_train_h

    return {"X_full_h": X_full_h}, {
        "deployment_mode": deployment_mode,
        "n_train_obs": int(len(train_idx)),
        "n_comp": int(k),
        "leak_flag": False,
        "uvar": np.asarray(uvar, dtype=np.float64),
        "strategy": strategy,
        "basis_model": basis_model,
    }
