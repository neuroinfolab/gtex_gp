# AHBA-GTEx Expression Harmonization Procedure

This document specifies the manuscript-facing harmonization procedure used by the active `gtex_gp` LORO cache and PREPOST evaluation workflows. It describes the implemented ComBat-style correction, the covariates used to protect biological/design variation, the additional AHBA-referenced GTEx calibration, and the main assumptions and caveats.

Implementation references:

- `src/harmonize/combat.py`: `CombatHarmonizer`
- `src/harmonize/__init__.py`: `fit_harmonizer(..., method="combat")`
- `src/workflows/loro_cache.py`: LORO cache harmonizer fitting and fold isolation
- `src/eval_utils/results_eda.py`: PREPOST global harmonization
- `src/eval_utils/eda_core.py`: PREPOST disk-cache key

## Purpose

AHBA and GTEx measure related gene-expression biology through different assays and sampling designs. AHBA is microarray-based and provides dense spatial coverage over the AHBA-derived target parcel axis. GTEx brain data are RNA-seq TPM-derived and provide subject-level observations over a sparse set of sampled brain tissues. The harmonization step maps both datasets into a shared expression space before model fitting, imputation, and evaluation.

The goal is not to claim that raw microarray intensity and RNA-seq TPM are directly comparable. Instead, the active workflow assumes that, after expression preprocessing and parcel assignment, residual platform differences can be approximated gene-wise by covariate-preserving additive and multiplicative batch corrections, followed by an overlap-based affine calibration of GTEx onto the AHBA-referenced scale.

## Inputs

The harmonizer is fit on row-level expression tables after GTEx-to-AHBA parcel matching:

- AHBA rows with `parcel_idx` over the target parcel axis.
- GTEx rows with `parcel_idx` after the active matching policy.
- A selected gene panel (`allgenes` or `hvg`).
- Row covariates used when `combat_use_covariates=True`.

The current default covariate design is:

| Covariate | Encoding | Purpose |
|---|---|---|
| `age` | numeric midpoint for range labels, mean-imputed if missing, z-scored | preserve age-associated expression variation |
| `sex` | male indicator | preserve sex-associated expression variation |
| `macro_system` | one-hot dummies for `cerebellar`, `subcortical`, and `visual_somatomotor`; `cortical_association` is the reference level | preserve broad spatial-system expression variation |

`macro_system` is defined from the AHBA parcel label by `src/preprocess.py` and has four possible values: `cerebellar`, `subcortical`, `visual_somatomotor`, and `cortical_association`.

## Configuration

Relevant active configuration fields:

| Field | Default | Meaning |
|---|---:|---|
| `combat_use_covariates` | `True` | include biological/design covariates in the ComBat-style design |
| `drop_macro_system_covariate` | `False` | when `True`, reproduce the previous age+sex-only covariate design |

With defaults, the design is `age + sex + macro_system`. Setting `drop_macro_system_covariate=True` keeps `age + sex` but removes macro-system dummies. Setting `combat_use_covariates=False` removes all covariates.

## ComBat-Style Estimation

The implementation treats AHBA and GTEx as two batches:

- batch 0: AHBA
- batch 1: GTEx

For each gene, the procedure is:

1. Fit covariate effects by ordinary least squares on the pooled AHBA+GTEx table:

   ```text
   x_i,g ~= beta0_g + c_i beta_cov,g
   ```

2. Remove the fitted covariate effect and compute pooled standardization statistics:

   ```text
   y_i,g = x_i,g - c_i beta_cov,g
   s_i,g = (y_i,g - grand_mean_g) / pooled_sd_g
   ```

3. Estimate raw batch moments from standardized residuals:

   ```text
   gamma_hat_b,g = mean(s_i,g | batch_i = b)
   delta_hat_b,g = var(s_i,g | batch_i = b)
   ```

4. Shrink the raw gene-wise batch estimates toward shared across-gene batch-level moments:

   ```text
   lambda_b = n_b / (n_b + 10)

   gamma_star_b,g =
       lambda_b * gamma_hat_b,g
     + (1 - lambda_b) * mean_g(gamma_hat_b,g)

   log_delta_star_b,g =
       lambda_b * log(delta_hat_b,g)
     + (1 - lambda_b) * mean_g(log(delta_hat_b,g))
   ```

   Then `delta_star_b,g = exp(log_delta_star_b,g)`.

5. Remove the estimated additive and multiplicative batch effects:

   ```text
   s_adj_i,g = (s_i,g - gamma_star_b,g) / sqrt(delta_star_b,g)
   ```

6. Reconstruct corrected expression in the shared corrected space:

   ```text
   x_corr_i,g = s_adj_i,g * pooled_sd_g + grand_mean_g + c_i beta_cov,g
   ```

This is a ComBat-style moment estimator with empirical shrinkage across genes. It is not a full canonical Johnson et al. empirical-Bayes posterior implementation. The shrinkage serves the same practical purpose of stabilizing noisy gene-wise batch estimates and reducing extreme independent corrections.

## AHBA-Referenced GTEx Affine Calibration

After ComBat-style correction, the implementation fits one additional gene-wise affine map from corrected GTEx to corrected AHBA using parcels observed in both datasets:

```text
AHBA_corrected_parcel_mean_r,g ~= slope_g * GTEx_corrected_parcel_mean_r,g + intercept_g
```

Final transformed values are:

```text
AHBA_h = AHBA_corrected
GTEx_h = slope_g * GTEx_corrected + intercept_g
```

This places GTEx observations onto the AHBA-referenced harmonized scale used by the atlas prior, LORO truth, and model predictions. The global GTEx-to-AHBA affine calibration is distinct from PLAM's subject-specific latent observation calibration.

## PREPOST Workflow

For PREPOST and dataset-level evaluation, the harmonizer is fit once using:

- all AHBA rows
- all eligible GTEx rows

The returned PREPOST object includes raw and harmonized GTEx cubes, AHBA raw/harmonized tables, eligibility metadata, and the observed subject-parcel mask. The disk-cache key includes `combat_use_covariates`, `drop_macro_system_covariate`, the resolved matching policy, and the expression CSV signature so incompatible harmonization settings do not reuse the same PREPOST pickle.

## LORO Workflow

For the subject-wise LORO cache, harmonization is fit separately for each held-out subject-parcel fold:

- all AHBA rows are included
- all GTEx rows are included except the held-out row for the current subject and parcel

The held-out GTEx row is transformed only after fitting the fold-specific harmonizer, producing `loro_truth_subject_h`. Model predictions are evaluated against this harmonized held-out truth. The workflow also fits a full harmonizer for deployable full-subject completion maps using all AHBA rows and all eligible GTEx rows.

## Motivation For Macro-System Covariates

AHBA and GTEx differ not only by assay platform and cohort composition but also by spatial sampling. AHBA has dense target-parcel coverage, while GTEx observes a sparse set of tissue labels. If broad spatial systems are unevenly represented across datasets, an unmodeled spatial composition difference can be absorbed into the dataset batch effect.

Including `macro_system` in the covariate design helps preserve broad spatial-system expression differences while estimating AHBA-vs-GTEx batch effects. The covariate is intentionally coarse: it protects major spatial structure without using a saturated parcel-level design that would be sparse and difficult to identify.

## Modality Considerations

GTEx TPM-derived RNA-seq expression and AHBA microarray intensity have different raw distributional properties. RNA-seq TPM values are nonnegative, right-skewed, and often near-zero heavy before transformation. Microarray intensity values are positive continuous measurements that are typically more Gaussian-like after background correction and normalization, but have a more compressed dynamic range and probe-specific noise.

The active harmonization model assumes that, in the transformed/preprocessed expression space used by `gxp_samples.csv`, cross-platform differences can be represented approximately by gene-wise additive and multiplicative effects after preserving specified covariates. This is a pragmatic approximation, not a full distributional alignment between RNA-seq and microarray measurements.

## Caveats

- With only two highly confounded datasets, technical platform effects, cohort effects, and biological differences cannot be fully separated.
- Covariates can preserve modeled variation only when the design has sufficient support across datasets. They do not solve all non-overlap.
- Macro-system adjustment protects broad spatial-system effects but does not model fine parcel-level spatial variation as a covariate.
- The affine GTEx-to-AHBA calibration uses overlapping parcels and may propagate overlap-specific bias if overlap support is limited or unrepresentative.
- The procedure does not currently apply quantile normalization before ComBat-style correction.
- The implementation is ComBat-style with moment estimates and empirical shrinkage, not a complete canonical empirical-Bayes ComBat implementation.

## Reporting Language

Recommended manuscript phrasing:

> We harmonized AHBA and GTEx expression using a ComBat-style batch-correction procedure with AHBA and GTEx treated as dataset batches. The model included age, sex, and broad macrosystem labels as covariates by default, preserving modeled demographic and spatial-system variation while estimating gene-wise additive and multiplicative dataset effects. Batch moments were estimated from standardized residuals and shrunk toward shared across-gene batch-level moments to stabilize gene-wise corrections. Corrected GTEx expression was then mapped to the AHBA-referenced scale using a gene-wise affine calibration fit on parcels observed in both datasets.

