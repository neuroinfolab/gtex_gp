from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._utils import fit_genewise_affine_on_overlap, fit_zca


@dataclass
class WhitenZCAAffineHarmonizer:
    genes: List[str]
    ahba_mu: np.ndarray
    ahba_W: np.ndarray
    ahba_Winv: np.ndarray
    gtex_mu: np.ndarray
    gtex_W: np.ndarray
    gtex_Winv: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    ahba_cond: float
    gtex_cond: float
    ahba_shrink: float
    gtex_shrink: float

    @classmethod
    def fit(
        cls,
        ahba_df: pd.DataFrame,
        gtex_df: pd.DataFrame,
        gene_cols: List[str],
        eps: float = 1e-4,
    ) -> "WhitenZCAAffineHarmonizer":
        xa = ahba_df[gene_cols].to_numpy(dtype=np.float64)
        xg = gtex_df[gene_cols].to_numpy(dtype=np.float64)
        az = fit_zca(xa, eps=eps)
        gz = fit_zca(xg, eps=eps)
        ahba_base = (xa - az["mu"][None, :]) @ az["W"]
        gtex_base = (xg - gz["mu"][None, :]) @ gz["W"]
        n_parcels = int(max(ahba_df["parcel_idx"].max(), gtex_df["parcel_idx"].max()) + 1)
        slope, intercept = fit_genewise_affine_on_overlap(
            ahba_base,
            gtex_base,
            ahba_df["parcel_idx"].to_numpy(dtype=np.int32),
            gtex_df["parcel_idx"].to_numpy(dtype=np.int32),
            n_parcels=n_parcels,
        )
        return cls(
            genes=list(gene_cols),
            ahba_mu=az["mu"],
            ahba_W=az["W"],
            ahba_Winv=az["Winv"],
            gtex_mu=gz["mu"],
            gtex_W=gz["W"],
            gtex_Winv=gz["Winv"],
            slope=slope,
            intercept=intercept,
            ahba_cond=float(az["cond_sigma_reg"]),
            gtex_cond=float(gz["cond_sigma_reg"]),
            ahba_shrink=float(az["shrink"]),
            gtex_shrink=float(gz["shrink"]),
        )

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df.copy()
        x = out[self.genes].to_numpy(dtype=np.float64)
        if str(dataset_name).upper() == "AHBA":
            h = (x - self.ahba_mu[None, :]) @ self.ahba_W
        else:
            z = (x - self.gtex_mu[None, :]) @ self.gtex_W
            h = z * self.slope + self.intercept
        out.loc[:, self.genes] = h.astype(np.float32)
        return out

    def inverse_gtex(self, x_h_matrix: np.ndarray, subject_ids: Optional[np.ndarray] = None) -> np.ndarray:
        z = (x_h_matrix - self.intercept) / self.slope
        return z @ self.gtex_Winv + self.gtex_mu[None, :]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "method": "whiten_zca_affine",
            "mean_abs_slope_minus1": float(np.nanmean(np.abs(self.slope - 1.0))),
            "mean_abs_intercept": float(np.nanmean(np.abs(self.intercept))),
            "min_abs_slope": float(np.nanmin(np.abs(self.slope))),
            "max_abs_slope": float(np.nanmax(np.abs(self.slope))),
            "ahba_cov_cond": float(self.ahba_cond),
            "gtex_cov_cond": float(self.gtex_cond),
            "ahba_shrink": float(self.ahba_shrink),
            "gtex_shrink": float(self.gtex_shrink),
        }
