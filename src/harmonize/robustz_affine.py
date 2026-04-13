from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ._utils import fit_genewise_affine_on_overlap, safe_mad_scale


@dataclass
class RobustZAffineHarmonizer:
    genes: List[str]
    ahba_loc: np.ndarray
    ahba_scale: np.ndarray
    gtex_loc: np.ndarray
    gtex_scale: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, ahba_df: pd.DataFrame, gtex_df: pd.DataFrame, gene_cols: List[str]) -> "RobustZAffineHarmonizer":
        xa = ahba_df[gene_cols].to_numpy(dtype=np.float64)
        xg = gtex_df[gene_cols].to_numpy(dtype=np.float64)
        a_loc = np.nanmedian(xa, axis=0)
        a_scale = safe_mad_scale(xa, a_loc)
        g_loc = np.nanmedian(xg, axis=0)
        g_scale = safe_mad_scale(xg, g_loc)
        ahba_base = (xa - a_loc) / a_scale
        gtex_base = (xg - g_loc) / g_scale
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
            ahba_loc=a_loc,
            ahba_scale=a_scale,
            gtex_loc=g_loc,
            gtex_scale=g_scale,
            slope=slope,
            intercept=intercept,
        )

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df.copy()
        x = out[self.genes].to_numpy(dtype=np.float64)
        if str(dataset_name).upper() == "AHBA":
            h = (x - self.ahba_loc) / self.ahba_scale
        else:
            z = (x - self.gtex_loc) / self.gtex_scale
            h = z * self.slope + self.intercept
        out.loc[:, self.genes] = h.astype(np.float32)
        return out

    def diagnostics(self) -> Dict[str, float]:
        return {
            "method": "robustz_affine",
            "mean_abs_slope_minus1": float(np.nanmean(np.abs(self.slope - 1.0))),
            "mean_abs_intercept": float(np.nanmean(np.abs(self.intercept))),
            "min_abs_slope": float(np.nanmin(np.abs(self.slope))),
            "max_abs_slope": float(np.nanmax(np.abs(self.slope))),
        }
