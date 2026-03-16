from __future__ import annotations

import numpy as np

from src.models.unified_generative import (
    UnifiedGenerativeConfig,
    fit_global_atlas_unified,
    infer_subject_unified,
)


def test_unified_updates_finite_and_nonincreasing_objective_smoke():
    rng = np.random.default_rng(123)
    r, g, k = 30, 20, 3
    coords = rng.normal(size=(r, 3))
    X = rng.normal(size=(r, g))

    cfg = UnifiedGenerativeConfig(latent_dim=k, max_iters=4, robust_loss="huber")
    atlas = fit_global_atlas_unified(X, coords, cfg)

    obs_idx = np.arange(10, dtype=np.int32)
    X_obs = X[obs_idx] + 0.05 * rng.normal(size=(10, g))

    res = infer_subject_unified(
        {"obs_idx": obs_idx, "X_obs_h": X_obs},
        atlas,
        cfg,
        fold_ctx={"prior_h": X, "obs_idx": obs_idx},
    )

    assert np.isfinite(res["x_hat_h_full"]).all()
    obj = np.asarray(res["convergence"]["objective"], dtype=np.float64)
    assert obj.size > 0
    assert np.isfinite(obj).all()
    assert obj[-1] <= obj[0] + 1e-6
