import unittest
import numpy as np
import pandas as pd

from src.harmonize.hier_affine import HierAffineHarmonizer


def mk(seed, n, genes, dataset):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(genes)))
    df = pd.DataFrame(x, columns=genes)
    df["subject"] = [f"S{i%5}" for i in range(n)]
    df["age"] = "50-59"
    df["sex"] = "F"
    df["dataset"] = dataset
    df["tissue_or_parcel"] = [f"P{i%10}" for i in range(n)]
    df["coordinates"] = "(0,0,0)"
    df["parcel_idx"] = np.asarray([i % 10 for i in range(n)], dtype=np.int32)
    df["macro_system"] = "cortical_association"
    return df


class TestHierAffine(unittest.TestCase):
    def test_subject_params_exist(self):
        genes = ["G1", "G2", "G3"]
        ah = mk(1, 30, genes, "AHBA")
        gt = mk(2, 40, genes, "GTEX")
        h = HierAffineHarmonizer.fit(ah, gt, genes, lambda_a=5.0, lambda_b=5.0)
        self.assertGreater(len(h.subject_params), 0)
        for a, b in h.subject_params.values():
            self.assertEqual(a.shape[0], len(genes))
            self.assertEqual(b.shape[0], len(genes))
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(np.isfinite(b)))


if __name__ == "__main__":
    unittest.main()
