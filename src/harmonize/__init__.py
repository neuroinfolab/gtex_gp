"""Harmonization model registry for vNext pipeline."""

from .zscore_affine import ZScoreAffineHarmonizer
from .robustz_affine import RobustZAffineHarmonizer
from .whiten_zca_affine import WhitenZCAAffineHarmonizer
from .combat import CombatHarmonizer
from .hier_affine import HierAffineHarmonizer
from .piecewise import PiecewiseRobustZHarmonizer


def fit_harmonizer(train_ahba_df, train_gtex_df, gene_cols, method, cfg):
    m = str(method).lower()
    if m == "zscore_affine":
        return ZScoreAffineHarmonizer.fit(train_ahba_df, train_gtex_df, gene_cols)
    if m == "robustz_affine":
        return RobustZAffineHarmonizer.fit(train_ahba_df, train_gtex_df, gene_cols)
    if m == "whiten_zca_affine":
        return WhitenZCAAffineHarmonizer.fit(
            train_ahba_df,
            train_gtex_df,
            gene_cols,
            eps=float(getattr(cfg, "whiten_eps", 1e-4)),
        )
    if m == "combat":
        return CombatHarmonizer.fit(
            train_ahba_df,
            train_gtex_df,
            gene_cols,
            use_covariates=bool(getattr(cfg, "combat_use_covariates", True)),
            inverse_slope_floor=float(getattr(cfg, "combat_inverse_slope_floor", 0.10)),
        )
    if m == "hier_affine":
        subset = getattr(cfg, "hier_subject_subset", None)
        return HierAffineHarmonizer.fit(
            train_ahba_df,
            train_gtex_df,
            gene_cols,
            lambda_a=float(getattr(cfg, "hier_lambda_a", 10.0)),
            lambda_b=float(getattr(cfg, "hier_lambda_b", 10.0)),
            base_method=str(getattr(cfg, "hier_base_method", "robustz_affine")),
            subject_subset=None if subset is None else [str(s) for s in subset],
        )
    if m == "piecewise_robustz":
        return PiecewiseRobustZHarmonizer.fit(
            train_ahba_df,
            train_gtex_df,
            gene_cols,
            min_group_samples=int(getattr(cfg, "min_group_samples", 200)),
        )
    raise ValueError(f"Unknown harmonization method: {method}")
