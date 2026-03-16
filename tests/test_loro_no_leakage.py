import unittest
import numpy as np

from src.eval.loro import run_loro


class TestLORONoLeakage(unittest.TestCase):
    def test_holdout_not_in_train(self):
        obs_idx = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        X = np.random.default_rng(123).normal(size=(5, 6))
        baseline = np.random.default_rng(321).normal(size=(10, 6))

        bundle = {
            "subject": "S1",
            "obs_idx": obs_idx,
            "X_obs_h": X,
            "baseline_h_full": baseline,
            "config_hash": "abc",
        }

        def pipe(hold, train_idx):
            leak = bool(int(hold) in set(train_idx.tolist()))
            pred = X[np.where(obs_idx == hold)[0][0], :]
            return pred, {"leak_flag": leak, "hold_pred_std": 0.1}

        folds, summary = run_loro(bundle, pipe, {"config_hash": "abc"})
        self.assertTrue((folds["status"] == "ok").all())
        self.assertEqual(int(summary["n_folds"]), 5)

    def test_leakage_flag_is_respected(self):
        obs_idx = np.asarray([0, 1, 2], dtype=np.int32)
        X = np.random.default_rng(1).normal(size=(3, 5))
        baseline = np.zeros((8, 5), dtype=float)
        bundle = {
            "subject": "S2",
            "obs_idx": obs_idx,
            "X_obs_h": X,
            "baseline_h_full": baseline,
            "config_hash": "abc",
        }

        def pipe(hold, train_idx):
            pred = np.zeros(5, dtype=float)
            return pred, {"leak_flag": True, "hold_pred_std": 0.2}

        folds, summary = run_loro(bundle, pipe, {"config_hash": "abc"})
        self.assertTrue((folds["status"] == "failed_leakage").all())
        self.assertEqual(int(summary["n_folds"]), 0)


if __name__ == "__main__":
    unittest.main()
