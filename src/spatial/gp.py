from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


@dataclass
class GPSpatialModel:
    coords_obs: np.ndarray
    U_obs: np.ndarray
    models: list


def fit_spatial(
    coords_obs: np.ndarray,
    U_obs: np.ndarray,
    length_scale: float = 25.0,
    noise_level: float = 1e-3,
    alpha: float = 1e-6,
    optimize: bool = True,
    n_restarts_optimizer: int = 0,
    random_state: int = 123,
) -> GPSpatialModel:
    optimizer = "fmin_l_bfgs_b" if bool(optimize) else None
    models = []
    for k in range(U_obs.shape[1]):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
            noise_level=float(noise_level), noise_level_bounds=(1e-8, 1e1)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(alpha),
            normalize_y=True,
            random_state=random_state,
            optimizer=optimizer,
            n_restarts_optimizer=int(n_restarts_optimizer),
        )
        gp.fit(coords_obs, U_obs[:, k])
        models.append(gp)
    return GPSpatialModel(coords_obs=coords_obs.copy(), U_obs=U_obs.copy(), models=models)


def predict(model: GPSpatialModel, coords_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for mdl in model.models:
        m, s = mdl.predict(coords_new, return_std=True)
        means.append(m)
        stds.append(s)
    M = np.stack(means, axis=1).astype(np.float64)
    S = np.stack(stds, axis=1).astype(np.float64)
    return M, S
