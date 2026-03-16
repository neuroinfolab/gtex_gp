from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class RBFSpatialModel:
    coords_obs: np.ndarray
    U_obs: np.ndarray
    models: list


def fit_spatial(coords_obs: np.ndarray, U_obs: np.ndarray, smoothing: float = 0.10) -> RBFSpatialModel:
    models = []
    for k in range(U_obs.shape[1]):
        try:
            rbf = RBFInterpolator(
                coords_obs,
                U_obs[:, k],
                kernel="thin_plate_spline",
                smoothing=float(smoothing),
            )
            models.append(rbf)
        except Exception:
            models.append(None)
    return RBFSpatialModel(coords_obs=coords_obs.copy(), U_obs=U_obs.copy(), models=models)


def predict(model: RBFSpatialModel, coords_new: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    nc = len(model.models)
    out = np.zeros((coords_new.shape[0], nc), dtype=np.float64)
    for k, mdl in enumerate(model.models):
        if mdl is not None:
            out[:, k] = mdl(coords_new)
        else:
            d2 = ((coords_new[:, None, :] - model.coords_obs[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d2, axis=1)
            out[:, k] = model.U_obs[nn, k]
    return out, None
