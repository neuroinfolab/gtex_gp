from __future__ import annotations

import numpy as np

from src.models.unified_generative import UnifiedGenerativeConfig, update_delta_gp


def test_gp_uncertainty_nonnegative_and_smaller_near_obs():
    rng = np.random.default_rng(2)
    coords_full = rng.normal(size=(40, 3))
    obs_idx = np.arange(8)
    coords_obs = coords_full[obs_idx]
    delta_obs = rng.normal(size=(8, 3))

    cfg = UnifiedGenerativeConfig(latent_dim=3)
    _, uvar = update_delta_gp(coords_obs, coords_full, delta_obs, cfg)

    assert np.isfinite(uvar).all()
    assert (uvar >= 0.0).all()
    near = np.mean(uvar[obs_idx])
    far = np.mean(uvar[20:])
    assert near <= far + 1e-6
