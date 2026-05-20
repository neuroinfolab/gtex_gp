import unittest
import numpy as np
import pandas as pd

from src.harmonize.combat import CombatHarmonizer


def mk(seed, n, genes, dataset):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(genes)))
    df = pd.DataFrame(x, columns=genes)
    df["subject"] = [f"S{i%5}" for i in range(n)]
    df["age"] = ["40-49" if i % 2 == 0 else "50-59" for i in range(n)]
    df["sex"] = ["M" if i % 2 == 0 else "F" for i in range(n)]
    df["dataset"] = dataset
    df["tissue_or_parcel"] = [f"P{i%8}" for i in range(n)]
    df["coordinates"] = "(0,0,0)"
    df["parcel_idx"] = np.asarray([i % 8 for i in range(n)], dtype=np.int32)
    macros = ["cortical_association", "subcortical", "visual_somatomotor", "cerebellar"]
    df["macro_system"] = [macros[i % len(macros)] for i in range(n)]
    return df


class TestCombatIsolation(unittest.TestCase):
    def test_fold_specific_fit_changes_params(self):
        genes = ["G1", "G2", "G3", "G4"]
        ah = mk(1, 32, genes, "AHBA")
        gt = mk(2, 36, genes, "GTEX")
        m1 = CombatHarmonizer.fit(ah, gt, genes, use_covariates=True)
        gt2 = gt.iloc[1:, :].copy()
        m2 = CombatHarmonizer.fit(ah, gt2, genes, use_covariates=True)
        diff = float(np.mean(np.abs(m1.gamma_star - m2.gamma_star)))
        self.assertGreater(diff, 0.0)

    def test_macro_system_covariate_can_be_dropped(self):
        genes = ["G1", "G2"]
        ah = mk(3, 24, genes, "AHBA")
        gt = mk(4, 28, genes, "GTEX")
        with_macro = CombatHarmonizer.fit(
            ah,
            gt,
            genes,
            use_covariates=True,
            drop_macro_system_covariate=False,
        )
        without_macro = CombatHarmonizer.fit(
            ah,
            gt,
            genes,
            use_covariates=True,
            drop_macro_system_covariate=True,
        )
        self.assertEqual(with_macro.beta_cov.shape[0], 5)  # age, sex, 3 macro-system dummies
        self.assertEqual(without_macro.beta_cov.shape[0], 2)  # age, sex only; previous behavior


if __name__ == "__main__":
    unittest.main()
