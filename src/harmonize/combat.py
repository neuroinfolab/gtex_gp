from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ._utils import fit_genewise_affine_on_overlap


def _age_to_numeric(x: object) -> float:
    s = str(x)
    if s.lower() in {"nan", "none", ""}:
        return np.nan
    if "-" in s:
        toks = s.replace(" ", "").split("-")
        try:
            a = float(toks[0])
            b = float(toks[1])
            return 0.5 * (a + b)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return np.nan


_MACRO_SYSTEM_REFERENCE = "cortical_association"
_MACRO_SYSTEM_LEVELS = (
    "cerebellar",
    "subcortical",
    "visual_somatomotor",
)


def _build_covariates(
    df: pd.DataFrame,
    use_covariates: bool,
    drop_macro_system_covariate: bool = False,
) -> np.ndarray:
    n = len(df)
    if not use_covariates:
        return np.zeros((n, 0), dtype=np.float64)
    age = np.asarray([_age_to_numeric(v) for v in df.get("age", pd.Series([np.nan] * n)).tolist()], dtype=np.float64)
    if np.all(~np.isfinite(age)):
        age = np.zeros(n, dtype=np.float64)
    else:
        m = np.nanmean(age)
        age = np.where(np.isfinite(age), age, m)
        sd = np.nanstd(age)
        age = (age - m) / (sd if sd > 1e-8 else 1.0)

    sex_raw = df.get("sex", pd.Series([""] * n)).astype(str).str.upper().str.strip()
    sex_m = (sex_raw == "M").to_numpy(dtype=np.float64)
    covariates = [age, sex_m]

    if not drop_macro_system_covariate:
        macro = df.get("macro_system", pd.Series([_MACRO_SYSTEM_REFERENCE] * n)).astype(str).str.lower().str.strip()
        # Use cortical association as the reference level; one-hot encode the
        # remaining broad systems to preserve spatial macro effects during
        # dataset-batch correction without adding a saturated parcel design.
        for level in _MACRO_SYSTEM_LEVELS:
            covariates.append((macro == level).to_numpy(dtype=np.float64))

    return np.column_stack(covariates).astype(np.float64)


@dataclass
class CombatHarmonizer:
    """
    ComBat-style harmonizer with an additional GTEx->AHBA overlap affine.
    Forward fit / transform, gene by gene: Let x_{i,g} be expression for sample i and gene g, with batch b(i) in {AHBA, GTEx}.
    Let c_i be the row covariates (currently age, sex, and optionally macro_system).
    
    1. Fit covariate effects on the pooled table:
        x_{i,g} ~= beta0_g + c_i^T beta_cov,g + residual_{i,g}
       Store beta_cov,g and define the covariate effect
        cov_e(i, g) = c_i^T beta_cov,g
    
    2. Remove covariates and compute pooled standardization statistics:
        y_{i,g} = x_{i,g} - cov_e(i, g)
        grand_mean_g = mean_i y_{i,g}
        pooled_sd_g  = std_i  y_{i,g}
        s_{i,g} = (y_{i,g} - grand_mean_g) / pooled_sd_g
    
    3. Estimate per-batch mean / variance effects in standardized space:
        gamma_hat_{b,g} = mean_{i: b(i)=b} s_{i,g}
        delta_hat_{b,g} = var_{i: b(i)=b}  s_{i,g}
       Then shrink to gamma_star, delta_star.
    
    4. Remove batch effects:
        s_adj_{i,g} =
            (s_{i,g} - gamma_star_{b(i),g}) / sqrt(delta_star_{b(i),g})
    
    5. Reconstruct corrected expression in the shared corrected space:
        x_corr_{i,g} =
            s_adj_{i,g} * pooled_sd_g + grand_mean_g + cov_e(i, g)
    
    6. Fit a second gene-wise affine map on overlapping parcels only:
        AHBA_parcel_mean_{r,g} ~= slope_g * GTEx_parcel_mean_{r,g} + intercept_g
       This is a post-ComBat calibration from corrected GTEx to corrected AHBA
       on shared parcel support.

    7. Final forward transform:
       AHBA:
           x_h = x_corr
       GTEx:
           x_h = slope * x_corr + intercept
    """
    genes: List[str]
    use_covariates: bool
    drop_macro_system_covariate: bool
    grand_mean: np.ndarray
    pooled_sd: np.ndarray
    gamma_hat: np.ndarray
    delta_hat: np.ndarray
    gamma_star: np.ndarray
    delta_star: np.ndarray
    beta_cov: np.ndarray  # shape: [n_cov, n_genes]
    slope: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(
        cls,
        ahba_df: pd.DataFrame,
        gtex_df: pd.DataFrame,
        gene_cols: List[str],
        use_covariates: bool = True,
        drop_macro_system_covariate: bool = False,
    ) -> "CombatHarmonizer":
        a = ahba_df.copy()
        g = gtex_df.copy()
        a["_batch"] = 0
        g["_batch"] = 1
        comb = pd.concat([a, g], axis=0, ignore_index=True)

        X = comb[gene_cols].to_numpy(dtype=np.float64)
        batch = comb["_batch"].to_numpy(dtype=np.int32)
        cov = _build_covariates(
            comb,
            use_covariates=use_covariates,
            drop_macro_system_covariate=drop_macro_system_covariate,
        )

        n, G = X.shape
        B = 2
        n_cov = cov.shape[1]

        beta_cov = np.zeros((n_cov, G), dtype=np.float64)
        grand_mean = np.zeros(G, dtype=np.float64)
        pooled_sd = np.ones(G, dtype=np.float64)
        S = np.zeros((n, G), dtype=np.float64)
        gamma_hat = np.zeros((B, G), dtype=np.float64)
        delta_hat = np.ones((B, G), dtype=np.float64)

        for gi in range(G):
            y = X[:, gi]
            if n_cov > 0:
                D = np.c_[np.ones(n, dtype=np.float64), cov]
                b, *_ = np.linalg.lstsq(D, y, rcond=None)
                cov_effect = D[:, 1:] @ b[1:]
                beta_cov[:, gi] = b[1:]
            else:
                cov_effect = np.zeros(n, dtype=np.float64)

            y_nocov = y - cov_effect
            gm = float(np.mean(y_nocov))
            sd = float(np.std(y_nocov))
            if sd < 1e-8:
                sd = 1.0
            grand_mean[gi] = gm
            pooled_sd[gi] = sd
            s = (y_nocov - gm) / sd
            S[:, gi] = s
            for bidx in range(B):
                m = batch == bidx
                if np.any(m):
                    gamma_hat[bidx, gi] = float(np.mean(s[m]))
                    dv = float(np.var(s[m]))
                    delta_hat[bidx, gi] = dv if dv > 1e-8 else 1.0

        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.ones_like(delta_hat)
        for bidx in range(B):
            nb = float(np.sum(batch == bidx))
            if nb <= 0:
                gamma_star[bidx, :] = gamma_hat[bidx, :]
                delta_star[bidx, :] = delta_hat[bidx, :]
                continue
            gamma_bar = float(np.mean(gamma_hat[bidx, :]))
            lam_g = nb / (nb + 10.0)
            gamma_star[bidx, :] = lam_g * gamma_hat[bidx, :] + (1.0 - lam_g) * gamma_bar

            ld = np.log(np.clip(delta_hat[bidx, :], 1e-8, None))
            ld_bar = float(np.mean(ld))
            lam_d = nb / (nb + 10.0)
            ld_star = lam_d * ld + (1.0 - lam_d) * ld_bar
            delta_star[bidx, :] = np.exp(ld_star)

        S_adj = np.zeros_like(S)
        for i in range(n):
            bidx = batch[i]
            S_adj[i, :] = (S[i, :] - gamma_star[bidx, :]) / np.sqrt(np.clip(delta_star[bidx, :], 1e-8, None))

        X_corr = np.zeros_like(X)
        for i in range(n):
            cov_e = cov[i, :] @ beta_cov if n_cov > 0 else np.zeros(G, dtype=np.float64)
            X_corr[i, :] = S_adj[i, :] * pooled_sd + grand_mean + cov_e

        a_corr = X_corr[: len(a), :]
        g_corr = X_corr[len(a) :, :]
        n_parcels = int(max(ahba_df["parcel_idx"].max(), gtex_df["parcel_idx"].max()) + 1)
        slope, intercept = fit_genewise_affine_on_overlap(
            a_corr,
            g_corr,
            ahba_df["parcel_idx"].to_numpy(dtype=np.int32),
            gtex_df["parcel_idx"].to_numpy(dtype=np.int32),
            n_parcels=n_parcels,
        )

        return cls(
            genes=list(gene_cols),
            use_covariates=bool(use_covariates),
            drop_macro_system_covariate=bool(drop_macro_system_covariate),
            grand_mean=grand_mean,
            pooled_sd=pooled_sd,
            gamma_hat=gamma_hat,
            delta_hat=delta_hat,
            gamma_star=gamma_star,
            delta_star=delta_star,
            beta_cov=beta_cov,
            slope=slope,
            intercept=intercept,
        )

    def _cov_effect(self, df: pd.DataFrame) -> np.ndarray:
        cov = _build_covariates(
            df,
            use_covariates=self.use_covariates,
            drop_macro_system_covariate=self.drop_macro_system_covariate,
        )
        if cov.shape[1] == 0:
            return np.zeros((len(df), len(self.genes)), dtype=np.float64)
        return cov @ self.beta_cov

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df.copy()
        x = out[self.genes].to_numpy(dtype=np.float64)
        cov_e = self._cov_effect(out)
        y_nocov = x - cov_e
        s = (y_nocov - self.grand_mean[None, :]) / self.pooled_sd[None, :]
        bidx = 0 if str(dataset_name).upper() == "AHBA" else 1
        s_adj = (s - self.gamma_star[bidx][None, :]) / np.sqrt(np.clip(self.delta_star[bidx][None, :], 1e-8, None))
        x_corr = s_adj * self.pooled_sd[None, :] + self.grand_mean[None, :] + cov_e
        if bidx == 1:
            x_corr = x_corr * self.slope[None, :] + self.intercept[None, :]
        out.loc[:, self.genes] = x_corr.astype(np.float32)
        return out

    def diagnostics(self) -> Dict[str, float]:
        return {
            "method": "combat",
            "mean_abs_slope_minus1": float(np.nanmean(np.abs(self.slope - 1.0))),
            "mean_abs_intercept": float(np.nanmean(np.abs(self.intercept))),
            "gamma_var_batch0": float(np.nanvar(self.gamma_star[0, :])),
            "gamma_var_batch1": float(np.nanvar(self.gamma_star[1, :])),
            "delta_mean_batch0": float(np.nanmean(self.delta_star[0, :])),
            "delta_mean_batch1": float(np.nanmean(self.delta_star[1, :])),
        }
