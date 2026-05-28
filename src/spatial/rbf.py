from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class RBFSpatialModel:
    coords_obs: np.ndarray
    U_obs: np.ndarray
    models: list


def fit_spatial(
    coords_obs: np.ndarray,
    U_obs: np.ndarray,
    smoothing: float = 0.10,
    kernel: str = "thin_plate_spline",
    epsilon: float | None = None,
    degree: int | None = None,
) -> RBFSpatialModel:
    """Fit per-component scattered-data RBF interpolators.

    Defaults reproduce the original thin-plate-spline behavior (a polyharmonic
    kernel with a polynomial tail that *extrapolates* off-support). Pass a
    decaying ``kernel`` ("gaussian" / "inverse_multiquadric") with ``epsilon``
    (inverse length scale) and ``degree=-1`` (drop the polynomial tail) to get a
    field that reverts to 0 away from the anchors — used by the t-prior residual.
    """
    models = []
    for k in range(U_obs.shape[1]):
        try:
            kw: dict = {"kernel": str(kernel), "smoothing": float(smoothing)}
            if epsilon is not None:
                kw["epsilon"] = float(epsilon)
            if degree is not None:
                kw["degree"] = int(degree)
            rbf = RBFInterpolator(coords_obs, U_obs[:, k], **kw)
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
