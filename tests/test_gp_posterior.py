import unittest
import numpy as np

from src.spatial.gp import fit_spatial, predict


class TestGPPosterior(unittest.TestCase):
    def test_gp_shapes_and_variance(self):
        rng = np.random.default_rng(123)
        coords = rng.uniform(-1, 1, size=(12, 3))
        U = np.c_[coords[:, 0] + 0.1 * rng.normal(size=12), coords[:, 1] ** 2 + 0.1 * rng.normal(size=12)]
        mdl = fit_spatial(coords, U, length_scale=0.5, random_state=123)
        mean_obs, std_obs = predict(mdl, coords)
        self.assertEqual(mean_obs.shape, U.shape)
        self.assertEqual(std_obs.shape, U.shape)
        self.assertTrue(np.all(np.isfinite(std_obs)))
        self.assertTrue(np.all(std_obs >= 0.0))
        self.assertLess(np.median(std_obs), 1.0)


if __name__ == "__main__":
    unittest.main()
