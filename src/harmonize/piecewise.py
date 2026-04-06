from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .robustz_affine import RobustZAffineHarmonizer


@dataclass
class PiecewiseRobustZHarmonizer:
    genes: List[str]
    global_model: RobustZAffineHarmonizer
    by_system: Dict[str, RobustZAffineHarmonizer]
    min_group_samples: int

    @classmethod
    def fit(
        cls,
        ahba_df: pd.DataFrame,
        gtex_df: pd.DataFrame,
        gene_cols: List[str],
        min_group_samples: int = 200,
    ) -> "PiecewiseRobustZHarmonizer":
        g = RobustZAffineHarmonizer.fit(ahba_df, gtex_df, gene_cols)
        systems = sorted(set(ahba_df.get("macro_system", pd.Series([], dtype=str)).astype(str).tolist()))
        by: Dict[str, RobustZAffineHarmonizer] = {}
        for sys in systems:
            a_sub = ahba_df[ahba_df["macro_system"] == sys]
            g_sub = gtex_df[gtex_df["macro_system"] == sys]
            if len(a_sub) < int(min_group_samples) or len(g_sub) < int(min_group_samples):
                by[sys] = g
            else:
                by[sys] = RobustZAffineHarmonizer.fit(a_sub, g_sub, gene_cols)
        return cls(genes=list(gene_cols), global_model=g, by_system=by, min_group_samples=int(min_group_samples))

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df.copy()
        if "macro_system" not in out.columns:
            return self.global_model.transform(out, dataset_name)
        x = out[self.genes].to_numpy(dtype=np.float64)
        o = np.zeros_like(x, dtype=np.float64)
        systems = out["macro_system"].astype(str).to_numpy()
        for sys in np.unique(systems):
            idx = np.where(systems == sys)[0]
            mdl = self.by_system.get(str(sys), self.global_model)
            tmp = out.iloc[idx, :].copy()
            tmp = mdl.transform(tmp, dataset_name)
            o[idx, :] = tmp[self.genes].to_numpy(dtype=np.float64)
        out.loc[:, self.genes] = o.astype(np.float32)
        return out

    def inverse_gtex(
        self,
        x_h_matrix: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        systems: Optional[np.ndarray] = None,
        sample_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        if systems is None:
            return self.global_model.inverse_gtex(x_h_matrix, sample_df=sample_df)
        out = np.zeros_like(x_h_matrix, dtype=np.float64)
        sys_arr = np.asarray(systems).astype(str)
        for sys in np.unique(sys_arr):
            idx = np.where(sys_arr == sys)[0]
            mdl = self.by_system.get(str(sys), self.global_model)
            sub_df = None if sample_df is None else sample_df.iloc[idx, :].copy()
            out[idx, :] = mdl.inverse_gtex(x_h_matrix[idx, :], sample_df=sub_df)
        return out

    def diagnostics(self) -> Dict[str, float]:
        return {
            "method": "piecewise_robustz",
            "n_system_models": int(len(self.by_system)),
            "min_group_samples": int(self.min_group_samples),
            **self.global_model.diagnostics(),
        }
