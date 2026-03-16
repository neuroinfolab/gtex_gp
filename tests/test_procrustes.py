import unittest
import numpy as np

from src.latent.procrustes import fit_o3


class TestProcrustes(unittest.TestCase):
    def test_o3_orthogonality(self):
        rng = np.random.default_rng(123)
        A = rng.normal(size=(20, 3))
        B = rng.normal(size=(20, 3))
        R = fit_o3(A, B)
        err = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
        self.assertLess(err, 1e-6)

    def test_procrustes_improves_fit(self):
        rng = np.random.default_rng(123)
        A = rng.normal(size=(50, 3))
        B = rng.normal(size=(50, 3))
        R = fit_o3(A, B)
        self.assertLessEqual(np.linalg.norm(A @ R - B), np.linalg.norm(A - B) + 1e-10)


if __name__ == "__main__":
    unittest.main()
