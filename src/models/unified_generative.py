from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


@dataclass
class UnifiedGenerativeConfig:
    latent_dim: int = 3
    max_iters: int = 5
    lambda_w: float = 1.0
    lambda_z: float = 1.0
    lambda_cal_a: float = 10.0
    lambda_cal_b: float = 10.0
    gp_length_scale: float = 25.0
    gp_noise: float = 1e-3
    gp_jitter: float = 1e-8
    robust_loss: str = "none"  # none|huber|student_t
    huber_delta: float = 1.5
    student_df: float = 4.0
    heteroscedastic: bool = False
    calibration_mode: str = "hier_affine_map"  # global_affine_only|hier_affine_map
    uncertainty_shrink: bool = False
    unc_alpha: float = 0.5
    unc_beta: float = 0.5
    unc_m0: float = 0.5
    unc_tau: float = 0.2
    random_state: int = 123


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ridge_linear(U: np.ndarray, X: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = U.shape
    D = np.c_[np.ones((n, 1), dtype=np.float64), U]
    I = np.eye(k + 1, dtype=np.float64)
    I[0, 0] = 0.0
    B = np.linalg.solve(D.T @ D + lam * I, D.T @ X)
    alpha = B[0, :]
    W = B[1:, :].T
    return W.astype(np.float64), alpha.astype(np.float64)


def _robust_weights(resid: np.ndarray, mode: str, huber_delta: float, student_df: float) -> np.ndarray:
    if mode == "none":
        return np.ones_like(resid, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64)
    mad = np.median(np.abs(r - np.median(r))) + 1e-8
    z = r / (1.4826 * mad)
    if mode == "huber":
        d = float(max(huber_delta, 1e-3))
        w = np.ones_like(z)
        m = np.abs(z) > d
        w[m] = d / np.abs(z[m])
        return w
    if mode == "student_t":
        nu = float(max(student_df, 2.1))
        return (nu + 1.0) / (nu + z**2)
    return np.ones_like(z, dtype=np.float64)


def _fit_gp_component(x_obs: np.ndarray, y_obs: np.ndarray, cfg: UnifiedGenerativeConfig):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=float(cfg.gp_length_scale),
        length_scale_bounds=(1e-2, 1e3),
    )
    kernel += WhiteKernel(noise_level=float(cfg.gp_noise), noise_level_bounds=(1e-8, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(cfg.gp_jitter),
        normalize_y=True,
        random_state=int(cfg.random_state),
        optimizer=None,
    )
    gp.fit(x_obs, y_obs)
    return gp


def fit_global_atlas_unified(ahba_h_full: np.ndarray, coords_full: np.ndarray, cfg: UnifiedGenerativeConfig) -> Dict[str, np.ndarray]:
    X = np.asarray(ahba_h_full, dtype=np.float64)
    coords = np.asarray(coords_full, dtype=np.float64)
    if X.size == 0:
        return {
            "U": np.zeros((0, cfg.latent_dim), dtype=np.float64),
            "W": np.zeros((0, cfg.latent_dim), dtype=np.float64),
            "alpha": np.zeros(0, dtype=np.float64),
            "coords_full": coords,
            "status": "empty",
            "diagnostics": {"atlas_recon_rmse": np.nan},
        }

    X0 = np.nan_to_num(X - np.nanmean(X, axis=0, keepdims=True), nan=0.0)
    Uu, Ss, _ = np.linalg.svd(X0, full_matrices=False)
    k = min(int(cfg.latent_dim), int(Uu.shape[1]))
    U = Uu[:, :k] * Ss[None, :k]

    W, alpha = _ridge_linear(U, X, lam=float(cfg.lambda_w))
    X_hat = alpha[None, :] + U @ W.T
    rmse = float(np.sqrt(np.nanmean((X - X_hat) ** 2)))

    return {
        "U": U.astype(np.float64),
        "W": W.astype(np.float64),
        "alpha": alpha.astype(np.float64),
        "coords_full": coords.astype(np.float64),
        "status": "ok",
        "latent_dim": int(k),
        "gp_kernel": {
            "type": "rbf+white",
            "length_scale": float(cfg.gp_length_scale),
            "noise": float(cfg.gp_noise),
            "jitter": float(cfg.gp_jitter),
        },
        "diagnostics": {"atlas_recon_rmse": rmse},
    }


def update_calibration(
    X_obs: np.ndarray,
    X_star_obs: np.ndarray,
    cfg: UnifiedGenerativeConfig,
    a0: np.ndarray,
    b0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    g = X_obs.shape[1]
    if str(cfg.calibration_mode).lower() == "global_affine_only":
        return np.ones(g, dtype=np.float64), np.zeros(g, dtype=np.float64)

    a = np.ones(g, dtype=np.float64)
    b = np.zeros(g, dtype=np.float64)
    la = float(max(cfg.lambda_cal_a, 1e-8))
    lb = float(max(cfg.lambda_cal_b, 1e-8))

    for j in range(g):
        x = X_obs[:, j]
        s = X_star_obs[:, j]
        m = np.isfinite(x) & np.isfinite(s)
        if int(np.sum(m)) < 2:
            a[j] = a0[j]
            b[j] = b0[j]
            continue
        sm = s[m]
        xm = x[m]
        n = float(sm.size)
        ss = float(np.dot(sm, sm))
        s1 = float(np.sum(sm))
        x1 = float(np.sum(xm))
        sx = float(np.dot(sm, xm))

        # MAP on [a,b] with independent Gaussian priors around [a0,b0].
        A = np.array([[ss + la, s1], [s1, n + lb]], dtype=np.float64)
        rhs = np.array([sx + la * a0[j], x1 + lb * b0[j]], dtype=np.float64)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        a[j] = float(sol[0])
        b[j] = float(sol[1])

    a = np.where(np.isfinite(a), a, a0)
    b = np.where(np.isfinite(b), b, b0)
    a = np.clip(a, 1e-3, 1e3)
    return a, b


def update_latent_z(
    X_obs: np.ndarray,
    U_obs: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    alpha: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cfg: UnifiedGenerativeConfig,
    gene_var_w: np.ndarray,
) -> np.ndarray:
    k = U_obs.shape[1]
    Z_new = np.zeros((U_obs.shape[0], k), dtype=np.float64)

    for i in range(U_obs.shape[0]):
        x = X_obs[i, :]
        u = U_obs[i, :] @ R
        x_tilde = (x - b) / np.clip(a, 1e-8, None)
        pred0 = alpha + u @ W.T
        resid0 = x_tilde - pred0
        rw = _robust_weights(resid0, cfg.robust_loss, cfg.huber_delta, cfg.student_df)
        ww = rw * gene_var_w
        Ww = W * ww[:, None]
        Aw = Ww.T @ W + float(cfg.lambda_z) * np.eye(k)
        bw = Ww.T @ (x_tilde - alpha) + float(cfg.lambda_z) * u
        try:
            z = np.linalg.solve(Aw, bw)
        except np.linalg.LinAlgError:
            z, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        Z_new[i, :] = z

    return Z_new


def update_alignment_R(U_obs: np.ndarray, Z_obs: np.ndarray) -> np.ndarray:
    try:
        R, _ = orthogonal_procrustes(U_obs, Z_obs)
        return np.asarray(R, dtype=np.float64)
    except Exception:
        k = U_obs.shape[1]
        return np.eye(k, dtype=np.float64)


def update_delta_gp(coords_obs: np.ndarray, coords_full: np.ndarray, delta_obs: np.ndarray, cfg: UnifiedGenerativeConfig) -> Tuple[np.ndarray, np.ndarray]:
    k = delta_obs.shape[1]
    n = coords_full.shape[0]
    delta_full = np.zeros((n, k), dtype=np.float64)
    uvar = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        gp = _fit_gp_component(coords_obs, delta_obs[:, j], cfg)
        m, s = gp.predict(coords_full, return_std=True)
        delta_full[:, j] = m
        uvar[:, j] = np.maximum(s, 0.0) ** 2
    return delta_full, uvar


def decode_expression(Z_full: np.ndarray, W: np.ndarray, alpha: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_star = alpha[None, :] + Z_full @ W.T
    X_hat = X_star * a[None, :] + b[None, :]
    return X_star.astype(np.float64), X_hat.astype(np.float64)


def apply_uncertainty_shrinkage(
    x_model: np.ndarray,
    x_prior: np.ndarray,
    dist: np.ndarray,
    uvar: np.ndarray,
    cfg: UnifiedGenerativeConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(dist, dtype=np.float64)
    d_scaled = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-8)
    u = np.asarray(uvar, dtype=np.float64)
    if u.ndim == 2:
        u = np.mean(u, axis=1)
    u_scaled = (u - np.min(u)) / (np.max(u) - np.min(u) + 1e-8)

    score = float(cfg.unc_alpha) * d_scaled + float(cfg.unc_beta) * u_scaled
    w = _sigmoid((score - float(cfg.unc_m0)) / max(float(cfg.unc_tau), 1e-8))
    x_final = (1.0 - w[:, None]) * np.asarray(x_model, dtype=np.float64) + w[:, None] * np.asarray(x_prior, dtype=np.float64)
    return x_final.astype(np.float64), w.astype(np.float64)


def infer_subject_unified(
    subject_data: Dict[str, np.ndarray],
    atlas_model: Dict[str, np.ndarray],
    cfg: UnifiedGenerativeConfig,
    fold_ctx: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    U_atlas = np.asarray(atlas_model["U"], dtype=np.float64)
    W = np.asarray(atlas_model["W"], dtype=np.float64)
    alpha = np.asarray(atlas_model["alpha"], dtype=np.float64)
    coords = np.asarray(atlas_model["coords_full"], dtype=np.float64)

    obs_idx = np.asarray(subject_data["obs_idx"], dtype=np.int32)
    X_obs = np.asarray(subject_data["X_obs_h"], dtype=np.float64)
    n_parcels, k = U_atlas.shape
    n_genes = X_obs.shape[1]

    if obs_idx.size == 0:
        X_star, X_hat = decode_expression(U_atlas, W, alpha, np.ones(n_genes), np.zeros(n_genes))
        return {
            "x_hat_h_full": X_hat,
            "x_star_h_full": X_star,
            "uvar_full": np.ones((n_parcels, k), dtype=np.float64),
            "r_align": np.eye(k, dtype=np.float64),
            "a_subj": np.ones(n_genes, dtype=np.float64),
            "b_subj": np.zeros(n_genes, dtype=np.float64),
            "status": "no_obs",
            "convergence": {"objective": [], "delta_R": [], "delta_Z": [], "delta_cal": []},
        }

    U_obs = U_atlas[obs_idx, :]
    coords_obs = coords[obs_idx, :]
    R = np.eye(k, dtype=np.float64)
    Z_obs = U_obs.copy()
    a0 = np.ones(n_genes, dtype=np.float64)
    b0 = np.zeros(n_genes, dtype=np.float64)
    a = a0.copy()
    b = b0.copy()
    gene_var_w = np.ones(n_genes, dtype=np.float64)

    conv_obj = []
    conv_dR = []
    conv_dZ = []
    conv_dCal = []

    for _ in range(int(max(cfg.max_iters, 1))):
        Z_prev = Z_obs.copy()
        R_prev = R.copy()
        a_prev = a.copy()
        b_prev = b.copy()

        X_star_obs = alpha[None, :] + Z_obs @ W.T
        a, b = update_calibration(X_obs, X_star_obs, cfg, a0, b0)

        Z_obs = update_latent_z(X_obs, U_obs, R, W, alpha, a, b, cfg, gene_var_w)
        R = update_alignment_R(U_obs, Z_obs)

        if bool(cfg.heteroscedastic):
            pred = (alpha[None, :] + Z_obs @ W.T) * a[None, :] + b[None, :]
            resid = X_obs - pred
            gv = np.nanvar(resid, axis=0) + 1e-6
            gene_var_w = 1.0 / gv
            gene_var_w = gene_var_w / (np.mean(gene_var_w) + 1e-8)

        pred_obs = (alpha[None, :] + Z_obs @ W.T) * a[None, :] + b[None, :]
        obj = float(np.nanmean((X_obs - pred_obs) ** 2))
        conv_obj.append(obj)
        conv_dR.append(float(np.linalg.norm(R - R_prev)))
        conv_dZ.append(float(np.linalg.norm(Z_obs - Z_prev)))
        conv_dCal.append(float(np.linalg.norm(a - a_prev) + np.linalg.norm(b - b_prev)))

    delta_obs = Z_obs @ R.T - U_obs
    delta_full, uvar = update_delta_gp(coords_obs, coords, delta_obs, cfg)
    Z_full = (U_atlas + delta_full) @ R
    X_star_full, X_hat_full = decode_expression(Z_full, W, alpha, a, b)

    weights = np.zeros(n_parcels, dtype=np.float64)
    if bool(cfg.uncertainty_shrink):
        if fold_ctx is None or "prior_h" not in fold_ctx:
            raise ValueError("uncertainty_shrink=True requires fold_ctx['prior_h']")
        if "obs_idx" in fold_ctx:
            train_obs = np.asarray(fold_ctx["obs_idx"], dtype=np.int32)
        else:
            train_obs = obs_idx
        coords_train = coords[train_obs, :]
        d2 = ((coords[:, None, :] - coords_train[None, :, :]) ** 2).sum(axis=2)
        dist = np.sqrt(np.min(d2, axis=1))
        X_hat_full, weights = apply_uncertainty_shrinkage(X_hat_full, np.asarray(fold_ctx["prior_h"], dtype=np.float64), dist, uvar, cfg)

    return {
        "x_hat_h_full": X_hat_full.astype(np.float64),
        "x_star_h_full": X_star_full.astype(np.float64),
        "uvar_full": uvar.astype(np.float64),
        "r_align": R.astype(np.float64),
        "a_subj": a.astype(np.float64),
        "b_subj": b.astype(np.float64),
        "z_obs": Z_obs.astype(np.float64),
        "delta_obs": delta_obs.astype(np.float64),
        "weights_shrink": weights.astype(np.float64),
        "status": "ok",
        "convergence": {
            "objective": conv_obj,
            "delta_R": conv_dR,
            "delta_Z": conv_dZ,
            "delta_cal": conv_dCal,
            "max_iters_reached": bool(len(conv_obj) >= int(max(cfg.max_iters, 1))),
        },
    }


# Backward-compatible wrappers

def fit_global_atlas(atlas_bundle: Dict[str, np.ndarray], cfg: UnifiedGenerativeConfig) -> Dict[str, np.ndarray]:
    return fit_global_atlas_unified(
        np.asarray(atlas_bundle.get("ahba_h_full", np.zeros((0, 0))), dtype=np.float64),
        np.asarray(atlas_bundle.get("coords_full", np.zeros((0, 3))), dtype=np.float64),
        cfg,
    )


def infer_subject_map(subject_bundle: Dict[str, np.ndarray], atlas_params: Dict[str, np.ndarray], cfg: UnifiedGenerativeConfig) -> Dict[str, np.ndarray]:
    res = infer_subject_unified(subject_bundle, atlas_params, cfg)
    return {
        "X_hat": res["x_hat_h_full"],
        "U_var": res["uvar_full"],
        "R": res["r_align"],
        "Z_obs": res.get("z_obs", np.zeros((0, 0))),
        "delta_obs": res.get("delta_obs", np.zeros((0, 0))),
        "status": res.get("status", "ok"),
    }
