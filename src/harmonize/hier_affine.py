from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .robustz_affine import RobustZAffineHarmonizer
from .zscore_affine import ZScoreAffineHarmonizer


@dataclass
class HierAffineHarmonizer:
    genes: List[str]
    base_method: str
    base: object
    lambda_a: float
    lambda_b: float
    subject_params: Dict[str, Tuple[np.ndarray, np.ndarray]]

    @classmethod
    def fit(
        cls,
        ahba_df: pd.DataFrame,
        gtex_df: pd.DataFrame,
        gene_cols: List[str],
        lambda_a: float = 10.0,
        lambda_b: float = 10.0,
        base_method: str = "robustz_affine",
        subject_subset: Optional[List[str]] = None,
    ) -> "HierAffineHarmonizer":
        bm = str(base_method).lower()
        if bm == "zscore_affine":
            base = ZScoreAffineHarmonizer.fit(ahba_df, gtex_df, gene_cols)
        else:
            base = RobustZAffineHarmonizer.fit(ahba_df, gtex_df, gene_cols)

        ah_h = base.transform(ahba_df, "AHBA")
        gt_h = base.transform(gtex_df, "GTEX")

        # AHBA target means per parcel in base-harmonized space.
        ah_target = ah_h.groupby("parcel_idx")[gene_cols].mean()

        subject_params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        allowed = None if subject_subset is None else {str(s) for s in subject_subset}
        for sid, sdf in gt_h.groupby("subject"):
            sid_s = str(sid)
            if allowed is not None and sid_s not in allowed:
                continue
            parcel_means = sdf.groupby("parcel_idx")[gene_cols].mean()
            common = sorted(set(parcel_means.index.tolist()) & set(ah_target.index.tolist()))
            if len(common) == 0:
                a = np.ones(len(gene_cols), dtype=np.float64)
                b = np.zeros(len(gene_cols), dtype=np.float64)
                subject_params[sid_s] = (a, b)
                continue
            X = parcel_means.loc[common, gene_cols].to_numpy(dtype=np.float64)
            Y = ah_target.loc[common, gene_cols].to_numpy(dtype=np.float64)

            a = np.ones(len(gene_cols), dtype=np.float64)
            b = np.zeros(len(gene_cols), dtype=np.float64)
            prior = np.asarray([1.0, 0.0], dtype=np.float64)
            Lam = np.diag([float(lambda_a), float(lambda_b)])
            ones = np.ones((X.shape[0], 1), dtype=np.float64)

            for gi in range(len(gene_cols)):
                x = X[:, gi]
                y = Y[:, gi]
                m = np.isfinite(x) & np.isfinite(y)
                if int(m.sum()) < 2:
                    a[gi] = 1.0
                    b[gi] = 0.0
                    continue
                A = np.c_[x[m], ones[m]]
                rhs = A.T @ y[m] + Lam @ prior
                mat = A.T @ A + Lam
                try:
                    theta = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    theta, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
                a[gi] = float(theta[0])
                b[gi] = float(theta[1])

            a = np.where(np.abs(a) < 1e-6, 1e-6, a)
            subject_params[sid_s] = (a.astype(np.float64), b.astype(np.float64))

        return cls(
            genes=list(gene_cols),
            base_method=bm,
            base=base,
            lambda_a=float(lambda_a),
            lambda_b=float(lambda_b),
            subject_params=subject_params,
        )

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = self.base.transform(df, dataset_name)
        if str(dataset_name).upper() != "GTEX":
            return out
        x = out[self.genes].to_numpy(dtype=np.float64)
        sids = out["subject"].astype(str).to_numpy()
        x2 = x.copy()
        for sid in np.unique(sids):
            idx = np.where(sids == sid)[0]
            a, b = self.subject_params.get(str(sid), (np.ones(len(self.genes)), np.zeros(len(self.genes))))
            x2[idx, :] = x[idx, :] * a[None, :] + b[None, :]
        out.loc[:, self.genes] = x2.astype(np.float32)
        return out

    def inverse_gtex(self, x_h_matrix: np.ndarray, subject_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if subject_ids is None:
            return self.base.inverse_gtex(x_h_matrix)

        xg = x_h_matrix.astype(np.float64).copy()
        sids = np.asarray(subject_ids).astype(str)
        for sid in np.unique(sids):
            idx = np.where(sids == sid)[0]
            a, b = self.subject_params.get(str(sid), (np.ones(len(self.genes)), np.zeros(len(self.genes))))
            xg[idx, :] = (x_h_matrix[idx, :] - b[None, :]) / a[None, :]
        return self.base.inverse_gtex(xg)

    def diagnostics(self) -> Dict[str, float]:
        slopes = []
        intercepts = []
        for a, b in self.subject_params.values():
            slopes.append(a)
            intercepts.append(b)
        if len(slopes) == 0:
            return {
                "method": "hier_affine",
                "n_subject_params": 0,
                "mean_abs_subject_slope_minus1": np.nan,
                "mean_abs_subject_intercept": np.nan,
            }
        S = np.vstack(slopes)
        B = np.vstack(intercepts)
        return {
            "method": "hier_affine",
            "base_method": self.base_method,
            "n_subject_params": int(len(self.subject_params)),
            "mean_abs_subject_slope_minus1": float(np.nanmean(np.abs(S - 1.0))),
            "mean_abs_subject_intercept": float(np.nanmean(np.abs(B))),
            "lambda_a": float(self.lambda_a),
            "lambda_b": float(self.lambda_b),
        }
