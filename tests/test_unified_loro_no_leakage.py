from __future__ import annotations

import numpy as np

from src.eval.loro import run_loro


def test_loro_leakage_flag_propagates_and_excludes_fold():
    obs_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    X_obs_h = np.random.default_rng(0).normal(size=(4, 6))
    baseline = np.random.default_rng(1).normal(size=(10, 6))

    def pipe(hold, train_idx):
        pred = np.zeros(6)
        leak = bool(hold == 1)
        return pred, {"leak_flag": leak, "hold_pred_std": 0.1}

    folds, summary = run_loro(
        {
            "subject": "S",
            "obs_idx": obs_idx,
            "X_obs_h": X_obs_h,
            "baseline_h_full": baseline,
            "config_hash": "x",
        },
        pipe,
        {"min_train_obs_loro": 2, "config_hash": "x"},
    )

    assert "failed_leakage" in set(folds["status"])
    assert summary["n_folds"] == 3
