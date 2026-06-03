from __future__ import annotations

from dataclasses import dataclass, field
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
    age_mean: float | None = None,
    age_sd: float | None = None,
) -> np.ndarray:
    """Construct the covariate design matrix for ComBat.

    The continuous ``age`` column is z-scored. At fit time the scaling
    ``(mean, sd)`` is derived from the input DataFrame (the pooled AHBA+GTEx
    table). At transform time the harmonizer must pass in ``age_mean``/``age_sd``
    captured at fit so that the same z-scaling is used on both sides — otherwise
    ``transform`` re-derives the scaling from whatever rows are handed in and the
    add-back step no longer cancels the fit-time covariate removal. In the
    degenerate small-batch case (single subject, single age) the in-data ``sd``
    is 0, the guard collapses the denominator to 1, and ``z(age) = 0`` for every
    row — silently stripping the entire age covariate. Always pass the fit-time
    stats through at transform.
    """
    n = len(df)
    if not use_covariates:
        return np.zeros((n, 0), dtype=np.float64)
    age = np.asarray([_age_to_numeric(v) for v in df.get("age", pd.Series([np.nan] * n)).tolist()], dtype=np.float64)
    if np.all(~np.isfinite(age)):
        age = np.zeros(n, dtype=np.float64)
    else:
        if age_mean is None or age_sd is None:
            m = float(np.nanmean(age))
            sd = float(np.nanstd(age))
        else:
            m = float(age_mean)
            sd = float(age_sd)
        age = np.where(np.isfinite(age), age, m)
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


def _resolve_per_batch_mask(cov_batch_mode: str, n_cov: int) -> np.ndarray:
    """Return a per-column boolean mask saying which covariate columns get
    per-batch coefficients vs a single pooled coefficient shared across batches.

    Column layout from :func:`_build_covariates`: ``[age, sex, *macro_dummies]``.

    Modes:
    - ``'mixed'`` (default) — age and sex per-batch; macro_system pooled.
      Rationale: age and sex are subject-level demographic covariates whose
      batch composition can correlate with the AHBA-vs-GTEx batch identity,
      so a single pooled OLS coefficient confounds with batch (the original
      candidate-1 / candidate-2 defect). Macro_system is parcel-level
      spatial-biological structure that is intrinsically shared across
      datasets, so a pooled coefficient is well-posed and reduces variance.
    - ``'pooled'`` — every covariate column gets a single pooled coefficient
      shared across batches. Reproduces the pre-fix behavior (modulo the
      candidate-1 z-scoring fix).
    - ``'per_batch'`` — every covariate column gets its own batch-specific
      coefficient. Maximally permissive; safe when batch-covariate
      confounding is suspected for every column.
    """
    mode = str(cov_batch_mode).lower()
    if n_cov == 0:
        return np.zeros(0, dtype=bool)
    if mode == "pooled":
        return np.zeros(n_cov, dtype=bool)
    if mode == "per_batch":
        return np.ones(n_cov, dtype=bool)
    if mode == "mixed":
        mask = np.zeros(n_cov, dtype=bool)
        if n_cov >= 1:
            mask[0] = True   # age
        if n_cov >= 2:
            mask[1] = True   # sex
        # macro_system dummies (indices 2:) stay False = pooled
        return mask
    raise ValueError(
        f"cov_batch_mode must be one of 'mixed', 'pooled', 'per_batch'; got {cov_batch_mode!r}"
    )


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
    beta_cov: np.ndarray  # shape: [n_cov, n_genes] — kept for backward
                          # compatibility; populated with the GTEx-batch
                          # coefficients (beta_cov_per_batch[1]).
    slope: np.ndarray
    intercept: np.ndarray
    # Fit-time z-scoring stats for the ``age`` covariate. Captured from the
    # pooled fit DataFrame and reused at transform so that the covariate
    # add-back uses the same scaling as the fit-time removal.
    age_mean_fit: float = 0.0
    age_sd_fit: float = 1.0
    # Per-batch covariate coefficients. Shape (B, n_cov, n_genes) where
    # batch 0 = AHBA, batch 1 = GTEx. ComBat's "preserve biological signal"
    # step adds back ``cov @ beta_cov_per_batch[batch(i)]`` per row. For
    # *pooled* covariate columns (under ``cov_batch_mode``) the coefficient
    # is identical across batches; for *per-batch* columns each batch
    # carries its own coefficient.
    beta_cov_per_batch: np.ndarray = field(default_factory=lambda: np.zeros((2, 0, 0), dtype=np.float64))
    # How covariate columns were treated at fit time: 'mixed' (age+sex
    # per-batch, macro_system pooled — default), 'pooled' (all pooled),
    # or 'per_batch' (all per-batch). Stored for documentation only; the
    # per-column choices are baked into ``beta_cov_per_batch``.
    cov_batch_mode: str = "mixed"

    @classmethod
    def fit(
        cls,
        ahba_df: pd.DataFrame,
        gtex_df: pd.DataFrame,
        gene_cols: List[str],
        use_covariates: bool = True,
        drop_macro_system_covariate: bool = False,
        cov_batch_mode: str = "mixed",
    ) -> "CombatHarmonizer":
        a = ahba_df.copy()
        g = gtex_df.copy()
        a["_batch"] = 0
        g["_batch"] = 1
        comb = pd.concat([a, g], axis=0, ignore_index=True)

        X = comb[gene_cols].to_numpy(dtype=np.float64)
        batch = comb["_batch"].to_numpy(dtype=np.int32)
        # Capture the pooled age stats now so transform can reuse them; passing
        # them in here also makes _build_covariates idempotent w.r.t. the same
        # input it just derived them from.
        _age_pool = np.asarray(
            [_age_to_numeric(v) for v in comb.get("age", pd.Series([np.nan] * len(comb))).tolist()],
            dtype=np.float64,
        )
        _age_pool_finite = _age_pool[np.isfinite(_age_pool)]
        if _age_pool_finite.size > 0:
            age_mean_fit = float(np.mean(_age_pool_finite))
            age_sd_fit = float(np.std(_age_pool_finite))
            if age_sd_fit <= 1e-8:
                age_sd_fit = 1.0
        else:
            age_mean_fit = 0.0
            age_sd_fit = 1.0
        cov = _build_covariates(
            comb,
            use_covariates=use_covariates,
            drop_macro_system_covariate=drop_macro_system_covariate,
            age_mean=age_mean_fit,
            age_sd=age_sd_fit,
        )

        n, G = X.shape
        B = 2
        n_cov = cov.shape[1]

        grand_mean = np.zeros(G, dtype=np.float64)
        pooled_sd = np.ones(G, dtype=np.float64)
        S = np.zeros((n, G), dtype=np.float64)
        gamma_hat = np.zeros((B, G), dtype=np.float64)
        delta_hat = np.ones((B, G), dtype=np.float64)

        # Hybrid OLS: which covariate columns are per-batch vs pooled is
        # controlled by ``cov_batch_mode``. The expanded design has:
        #   - one shared intercept,
        #   - for each per-batch covariate j: B columns (cov[:, j] * 1{batch == b}),
        #   - for each pooled covariate j: 1 column (cov[:, j]).
        # A single OLS solve over the expanded design recovers all coefficients
        # jointly. Pooled-column coefficients are then replicated across
        # batches in ``beta_cov_per_batch`` so the downstream cov_effect lookup
        # ``cov @ beta_cov_per_batch[batch(i)]`` works uniformly.
        mask_per_batch = _resolve_per_batch_mask(cov_batch_mode, n_cov)
        if n_cov > 0:
            cols_list = [np.ones(n, dtype=np.float64)]   # shared intercept
            layout: List[Tuple[int, int | None]] = []    # (cov_index, batch or None=pooled)
            for j in range(n_cov):
                if mask_per_batch[j]:
                    for bi in range(B):
                        cols_list.append(cov[:, j] * (batch == bi).astype(np.float64))
                        layout.append((j, bi))
                else:
                    cols_list.append(cov[:, j])
                    layout.append((j, None))
            D_ext = np.column_stack(cols_list)
            try:
                coef_ext = np.linalg.solve(D_ext.T @ D_ext, D_ext.T @ X)
            except np.linalg.LinAlgError:
                coef_ext, *_ = np.linalg.lstsq(D_ext, X, rcond=None)

            beta_cov_per_batch = np.zeros((B, n_cov, G), dtype=np.float64)
            pos = 1   # skip the shared-intercept row
            for j, bi in layout:
                if bi is None:
                    shared = coef_ext[pos, :]
                    for bb in range(B):
                        beta_cov_per_batch[bb, j, :] = shared
                else:
                    beta_cov_per_batch[bi, j, :] = coef_ext[pos, :]
                pos += 1

            cov_effect_per_sample = np.zeros((n, G), dtype=np.float64)
            for bidx in range(B):
                rows = batch == bidx
                if np.any(rows):
                    cov_effect_per_sample[rows, :] = cov[rows] @ beta_cov_per_batch[bidx]
        else:
            cov_effect_per_sample = np.zeros((n, G), dtype=np.float64)
            beta_cov_per_batch = np.zeros((B, 0, G), dtype=np.float64)

        # Standardization gene-by-gene on (y - per-sample cov_effect).
        for gi in range(G):
            y = X[:, gi]
            y_nocov = y - cov_effect_per_sample[:, gi]
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
            if n_cov > 0:
                cov_e = cov[i, :] @ beta_cov_per_batch[int(batch[i])]
            else:
                cov_e = np.zeros(G, dtype=np.float64)
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

        # beta_cov (legacy single-batch field) carries the GTEx-batch
        # coefficients, since the reconstruction target is GTEx and most
        # diagnostics that read this field care about that batch's signal.
        beta_cov_legacy = (
            beta_cov_per_batch[1]
            if n_cov > 0
            else np.zeros((0, G), dtype=np.float64)
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
            beta_cov=beta_cov_legacy,
            slope=slope,
            intercept=intercept,
            age_mean_fit=age_mean_fit,
            age_sd_fit=age_sd_fit,
            beta_cov_per_batch=beta_cov_per_batch,
            cov_batch_mode=str(cov_batch_mode),
        )

    def _cov_effect(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        cov = _build_covariates(
            df,
            use_covariates=self.use_covariates,
            drop_macro_system_covariate=self.drop_macro_system_covariate,
            age_mean=self.age_mean_fit,
            age_sd=self.age_sd_fit,
        )
        if cov.shape[1] == 0:
            return np.zeros((len(df), len(self.genes)), dtype=np.float64)
        bidx = 0 if str(dataset_name).upper() == "AHBA" else 1
        # Fallback for legacy harmonizers pickled before the per-batch fit
        # landed: their beta_cov_per_batch is (2, 0, 0) so n_cov-mismatch
        # would raise. Use the legacy ``beta_cov`` in that case.
        beta = self.beta_cov_per_batch[bidx]
        if beta.shape != (cov.shape[1], len(self.genes)):
            beta = self.beta_cov
        return cov @ beta

    def transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = df.copy()
        x = out[self.genes].to_numpy(dtype=np.float64)
        cov_e = self._cov_effect(out, dataset_name)
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
