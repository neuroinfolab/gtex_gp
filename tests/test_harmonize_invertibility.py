import unittest
import numpy as np
import pandas as pd

from src.harmonize import fit_harmonizer


class Cfg:
    whiten_eps = 1e-4
    combat_use_covariates = False
    hier_lambda_a = 10.0
    hier_lambda_b = 10.0
    hier_base_method = "robustz_affine"
    min_group_samples = 5


def _make_df(seed: int, n: int, genes: list[str], dataset: str):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, len(genes)))
    df = pd.DataFrame(arr, columns=genes)
    df["subject"] = [f"S{i%4}" for i in range(n)]
    df["age"] = "40-49"
    df["sex"] = "M"
    df["dataset"] = dataset
    df["tissue_or_parcel"] = [f"P{i%6}" for i in range(n)]
    df["coordinates"] = "(0,0,0)"
    df["parcel_idx"] = np.asarray([i % 6 for i in range(n)], dtype=np.int32)
    df["macro_system"] = "cortical_association"
    return df


class TestHarmonizeInvertibility(unittest.TestCase):
    def test_roundtrip_basic(self):
        genes = ["G1", "G2", "G3", "G4"]
        ah = _make_df(1, 24, genes, "AHBA")
        gt = _make_df(2, 30, genes, "GTEX")
        for method in ["zscore_affine", "robustz_affine", "whiten_zca_affine"]:
            h = fit_harmonizer(ah, gt, genes, method, Cfg())
            hgt = h.transform(gt, "GTEX")[genes].to_numpy(dtype=np.float64)
            xrt = h.inverse_gtex(hgt)
            self.assertTrue(np.all(np.isfinite(xrt)))
            err = float(np.sqrt(np.mean((xrt - gt[genes].to_numpy(dtype=np.float64)) ** 2)))
            self.assertLess(err, 1e-4 if method != "whiten_zca_affine" else 1e-2)


if __name__ == "__main__":
    unittest.main()
