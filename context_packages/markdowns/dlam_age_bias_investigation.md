# Reconstruction Age-Bias Investigation — Handoff

Status: **resolved**. Two coupled defects in `src/harmonize/combat.py` accounted for the artifact;
both were fixed on 2026-06-02. Candidate 1 (z-scoring degeneracy in `_build_covariates`)
collapsed the recon-vs-truth LV-space bias slope by 88% on its own. The residual exposed a second
issue — the harmonized truth's age slope was substantially larger than raw GTEx's, traced to a
pooled-OLS confounding of `β_age` with the AHBA-vs-GTEx batch effect. Fixed by switching to
per-batch `β_cov`, configurable via `cov_batch_mode ∈ {mixed, pooled, per_batch}` (default
`mixed`: age + sex per-batch, macro_system pooled). After both fixes plus an AHBA-basis refit,
the bias slope in LV space measures **||0.37||** (vs pre-fix 48.0 — a 130× collapse) with per-year
LV slopes of `LV1 +0.021/yr · LV2 −0.009/yr · LV3 +0.020/yr` — essentially noise. Last updated
2026-06-02.

## Formal issue (as observed pre-fix)

The `loro_fused` reconstruction carries a near-linear **age trend on the PLS gradient axes
LV1/LV2** (per-subject LV score rises with subject age) that is **absent in the held-out
ground truth** at the same parcels. The trend appears in essentially every parcel and is the
dominant between-subject signal on LV1/LV2 in the reconstruction. Because it is not present in the
truth, it is an artifact of the reconstruction, not a biological age effect. **The artifact appears
in both DLAM and PLAM reconstructions** (Tested observation 7), so the source lies in machinery the
two models share. Two coupled defects in `src/harmonize/combat.py` were identified and fixed; both
were structurally orthogonal to the model code.

## Where this came up

Building `eval_pls_gradient_progression.ipynb` (LV-score-vs-age progression per parcel; logic in
`src/eval_utils/eval_pls_gradient_progression.py`). The progression plots showed LV1/LV2 climbing
monotonically with age in essentially every parcel (frontal, cerebellar, extrapolated) for the
reconstruction, while the ground truth was flat.

## Setup to reproduce

- Env: `vformer_env` overlay (see `context_packages/markdowns/onbrain_plotting_guide.md`).
  ```bash
  OVL=/scratch/asr655/envs/vformer_env/overlay-25GB-500K.ext3
  singularity exec --overlay "$OVL":ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -lc 'source /ext3/miniforge3/etc/profile.d/conda.sh && conda activate && \
    cd /scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp && PYTHONPATH=$PWD MPLBACKEND=Agg python <script>'
  ```
- Cache: `out/loro_subject_cache_cv_shrink`, model `dlam`, gene scope `allgenes`.
- Projections: freeze the AHBA PLS basis (`pg.fit_ahba_basis`, cached `out/pls_basis/ahba_basis__allgenes.pkl`),
  project every subject's `loro_truth` and `loro_fused` slice with `pgp.project_all_subjects` →
  `(subjects=318, regions=150, LV=3)`. Ages from `pgp.subject_ages` are GTEx decade midpoints
  (24,34,44,54,64,74); per-bin n ≈ 8 / 9 / 29 / 97 / 161 / 14 (heavily older-skewed).
- 12 GTEx–AHBA matched parcels carry held-out truth; the other 138 are extrapolation-only.
  Matched-parcel labels: Cerebellar_Region4/7, LH-Ca, LH-HTH, LH-NAC, LH-Pu, LH-SNr, LH_Amygdala,
  LH_Default_PFC_4 (frontal/BA9), LH_Hippocampus, LH_SalVentAttn_Med_1, LH_SalVentAttn_PFCl_1.
- Scratch diagnostic scripts (one-offs): `/tmp/age_diag.py`, `/tmp/age_amp.py`, `/tmp/age_spatial.py`,
  `/tmp/age_calib.py`, `/tmp/age_shrink.py`, `/tmp/age_dist.py`, `/tmp/gl3_fit.py`, `/tmp/raw_slope.py`.
  Re-create from the numbers below if `/tmp` was cleared.

## Notebook and generated figures

- Notebook: `eval_pls_gradient_progression.ipynb` (repo root). §3 projects three stages
  (`raw`, `loro_truth`, `loro_fused`) onto the frozen AHBA basis; §4 renders 3×4 progression grids
  over the 12 matched parcels, one grid per stage; §5 is a per-parcel selector (matched →
  raw+truth+recon triplet; extrapolated → reconstruction only). Logic in
  `src/eval_utils/eval_pls_gradient_progression.py` (`render_progression_grid`, `render_region_progression`).
- Saved figures (`out/brain_renders/`):
  - `pls_progression_grid__gtex_raw.png` — raw GTEx, 12 matched parcels (age-flat).
  - `pls_progression_grid__gtex_truth.png` — harmonized truth, 12 matched parcels (age-flat).
  - `pls_progression_grid__gtex_recon.png` — reconstruction, 12 matched parcels (LV1/LV2 climb).
  - `pls_progression__<PARCEL>__{raw,loro_truth,loro_fused}.png` — per-parcel selector outputs.
  - `/tmp/age_dist.png` — nearest-training-distance vs age / vs bias (null).
- PLAM analogs of the three progression grids (added 2026-05-28): `pls_progression_grid__gtex_recon.png`
  (PLAM run) shows the same monotone LV1/LV2 climb across all 12 matched parcels; PLAM `_truth.png`
  and `_raw.png` are flat (same input as DLAM). Per-subject PLAM brain-render references for
  side-by-side LV maps: `pls_gradients_many__plam__n6.png`,
  `pls_gradients__plam__GTEX-{T6MN,1HCU6,1KWVE,13PLJ}.png`.

## Tested observations (facts, with numbers)

**1. Recon-only, at identical parcels.** Per-subject mean LV score vs age, at the 12 matched parcels:

| | LV1 slope (r) | LV2 slope (r) | LV3 slope (r) |
|---|---|---|---|
| truth | −0.18 (−0.10) | −0.39 (−0.15) | +0.15 (+0.23) |
| recon | +2.09 (+0.86) | +2.65 (+0.83) | +0.10 (+0.27) |

Sign flip and ~10× magnitude on LV1/LV2 between truth and recon; LV3 is unchanged.

**2. The trend is a global per-subject offset, not regional.** Cross-parcel correlation of the
per-subject LV1 score across subjects = 0.74. recon LV1 score ≈ subject's overall standardized
expression level (r = 0.99). Age-explained variance of the subject-global term: truth 0.01 / 0.02,
recon 0.73 / 0.68 (LV1/LV2). Subject-global term std: truth 18.4 / 26.3 → recon 24.1 / 31.7 (LV1/LV2).

**3. The region-specific (subject-demeaned) part does not match truth.** After removing each
subject's cross-parcel mean over the 12 matched parcels: region-specific |age corr| truth ≈ 0.09,
recon ≈ 0.45; truth↔recon agreement on the per-parcel age slope = −0.10 (LV1/LV2), −0.29 (LV3).
Example per-parcel de-meaned LV1 slope: LH_Default_PFC_4 truth +0.22/yr vs recon −1.23/yr;
Cerebellar_Region7 truth −0.09 vs recon −0.80; LH-HTH truth −0.00 vs recon +0.68. The truth shows
no detectable region-specific age structure on these axes (|r| ≈ noise floor).

**4. The bias coincides with a reconstruction-accuracy gradient over age.** corr(per-subject recon
accuracy [Pearson r from cache], age) = +0.49 (older reconstructed better; consistent with the
sample-wise "Pearson r by Age" boxplot where DLAM/PLAM dip at ages 24/34 and Naive is flat).
corr(accuracy, |LV1 bias|) = −0.78. corr(per-subject LV1 bias, age) = +0.94 (R² ≈ 0.88; the bias is
close to linear in age).

**5. The bias is age-predictable and partly correctable, but not validatable off the matched
parcels.** Cross-parcel CV (fit bias~age on half the matched parcels, correct the other half)
removes the spurious +1.77 LV1 slope but overshoots negative (raw `recon−truth` subtraction
double-counts truth's own small slope). No held-out truth exists at the 138 extrapolated parcels, so
any correction cannot be validated there.

**6. The age slope is absent in the non-harmonized raw data too.** Projecting raw GTEx
(`build_combat_tensor_view`, `raw_matched`) through the same frozen basis, per-subject mean LV score
vs age at the 12 matched parcels: LV1 +0.47/yr (r=+0.23), LV2 −0.86/yr (r=−0.16), LV3 −0.21/yr
(r=−0.10) — small, comparable in magnitude to harmonized truth (LV1 −0.18, LV2 −0.39, LV3 +0.15),
and far below the reconstruction (LV1 +2.09, LV2 +2.65). So the strong LV age slope is **absent in
raw, absent after harmonization, present only at reconstruction** — harmonization does not introduce
a visible LV age slope. (Raw is projected through a basis standardized on harmonized AHBA, so its
absolute LV offset differs; only the within-raw age trend is compared here. Script: `/tmp/raw_slope.py`.)

**7. PLAM exhibits the same artifact.** Re-running the progression analysis with PLAM as the
reconstruction model produces the same monotone LV1/LV2 climb across all 12 matched parcels;
qualitatively as large or larger than DLAM (visual: y-range ~−150 to +250 on LV1/LV2 in
`out/brain_renders/pls_progression_grid__gtex_recon.png` for PLAM). Truth and raw grids for PLAM
are flat (same inputs as DLAM). PLAM and DLAM share almost no internal machinery: PLAM is the joint
generative model `(W, alpha, R, calibration (a,b), GP delta)` with no affine basis map, no
`constrained_anchor`, and no U → bridge ridge → invert chain. The shared surfaces are the ComBat
harmonizer, the AHBA atlas reference `ahba_h_full`, and the frozen AHBA PLS basis used for the
evaluation projection. The frozen basis is excluded mechanically (same transform applied to truth /
raw / recon — only recon climbs). The artifact therefore localizes to the harmonizer's covariate
handling or the atlas's role inside both reconstruction models.

## Mechanisms tested and excluded

**Excluded by measurement:**
- **Coverage** (# observed parcels per subject): corr(coverage, age) = −0.01.
- **Nearest-training-point distance** (the distance that drives anchor shrinkage in the
  `constrained_anchor` spatial reconstruction): corr(mean NN distance, age) = −0.03;
  corr(NN distance, |LV1 bias|) = −0.06. Sanity: corr(NN distance, coverage) = −0.68 (metric valid).
  Plot: `/tmp/age_dist.png`.
- **`affine_gl3` basis-map fit quality (subject→AHBA alignment).** Per-subject gl3 fit R² (subject
  PLS scores mapped to AHBA reference scores on observed parcels) = 0.789 mean; corr(R², age) =
  −0.06; R² by age bin 0.82 / 0.76 / 0.80 / 0.79 / 0.79 / 0.76 (youngest bin is highest).
  `stabilize_invertible` regularization λ = 0 for all 318 subjects. Younger subjects align to AHBA
  as well as older subjects.

**Excluded by reasoning (not a measurement):**
- **Cross-subject dataset imbalance.** DLAM is fit per subject — each fit sees only the AHBA
  reference and that subject's own observed parcels. There is no cross-subject pooling, so the
  older-skewed sample distribution cannot bias an individual subject's reconstruction. (Confirmed
  against the code path; see references.)
- **ComBat as a symmetric shared term.** Age is a ComBat covariate (z-scored age, sex,
  macro_system; `combat.py:_build_covariates`). The held-out truth (`loro_truth`) and the
  reconstruction (`loro_fused`) both pass through the same harmonizer, so a covariate term applied
  identically to both would cancel in `recon − truth`. This excludes a *symmetric* covariate term;
  it does **not** test whether the covariate is handled differently on the truth vs fused paths
  (untested — see investigation areas).

## Additional measured quantities (not yet attributed to a mechanism)

These were measured but do not by themselves establish a cause:
- Regressing per-subject `(recon − truth)` on `(AHBA_ref − truth)` over the 12 matched parcels gives
  a slope (pull fraction) of mean ≈ 0.51 on LV1; pooled corr(recon−truth, AHBA−truth) = 0.64; this
  per-subject slope correlates with age at +0.54.
- The AHBA reference LV1 centroid sits above the GTEx subject centroid by ≈ +9.1 (LV1);
  corr(AHBA−truth gap, age) = +0.10 (LV1), weak.

## Test A (LV-space direction probe, run 2026-05-28)

Run-once test described in the original "Proposed next test": project β_age onto the frozen
AHBA basis as a linear gene-offset direction, project per-subject `(recon − truth)` averaged
over held-out parcels, fit per-gene OLS slope on z(age), project that slope into LV space, and
compare directions by cosine. Inputs: full-data ComBat fit (age + sex + macro), the cached
allgenes basis, the `out/loro_subject_cache_cv_tprior_imq` cache (318 subjects). Script:
`/tmp/age_bias_directions.py`. Numbers:

```
beta_age LV components:        [+24, -192, -77]   ||.|| = 208
AHBA-GTEx centroid LV:         [-38,   -3, -10]   ||.|| =  39
bias-vs-z(age) slope LV:       [+29,  +38,  +1]   ||.|| =  48

cos( beta_age,  bias-vs-age slope ) = -0.673
cos( beta_age,  AHBA - GTEx       ) = +0.050   (β_age and AHBA-GTEx are ≈ orthogonal)
cos( bias slope, AHBA - GTEx      ) = -0.642
```

The two reference directions (β_age and AHBA−GTEx) are essentially orthogonal in LV space, so they
are independent axes of explanation. The bias projects strongly onto **both**. A two-vector linear
fit `bias ≈ a · β_age + b · (AHBA−GTEx)` gives R² = 0.83 with both terms contributing roughly equal
absolute magnitude in LV space (~30 each). Each candidate is therefore real and contributes
comparably; Test A is not a single-candidate discriminator.

## Identified mechanism for the β_age component (candidate 1)

Followup diagnostic (`/tmp/age_sign_diag.py`, 2026-05-28) pins down where the β_age signature
enters:

```
||slope of xh.mean over subjects on z(age)|| = 30.45
cos(slope_xh ,  +beta_age)                   = -0.446
||slope of single-row hold truth on z(age)|| =  7.07   (n=80 (subject, parcel) pairs)
cos(slope_hold_truth, +beta_age)             = -0.041   ≈ flat in beta_age direction

_build_covariates(hold_raw_single_row).age z-values   = [0.]
_build_covariates(gtex_raw) same subject's row z(age) = +0.624
```

**Mechanism.** `src/harmonize/combat.py:_build_covariates` re-computes `z(age) = (age − mean) / sd`
from whatever DataFrame it is handed each call:

```python
m = np.nanmean(age); sd = np.nanstd(age)
age = (age - m) / (sd if sd > 1e-8 else 1.0)
```

At **fit** this is the pooled (AHBA + GTEx) age distribution. At `harm.transform(hold_raw, "GTEX")`,
`hold_raw` is one subject at one parcel — a single age — so `sd = 0`, the guard triggers, every
`z(age) = 0`, and `β_age · z = 0` is added back. The held-out truth therefore has **no β_age
covariate restored**. Meanwhile `harm.transform(gtex_train, "GTEX")` is called on a many-subjects
batch where `sd > 0`, so the subject's xh has a non-zero (but pooled-vs-GTEx-shifted) β_age · z
restored. The model fits on xh, the subject mean `x_mean` over xh absorbs that signature, and the
decode adds it back at every parcel — including the held-out one. Net result on the bias:

- Truth at hold: zero β_age component (z = 0 ⇒ no add-back).
- Recon at hold: a β_age component inherited from `x_mean`; empirically anti-aligned with raw β_age
  (cos = −0.45) because the GTEx affine calibration `slope` multiplies through the cov add-back at
  transform time and reorients the direction in gene space.
- `bias = recon − truth ≈ recon`, so `cos(bias_slope, β_age) ≈ cos(slope_xh, β_age) ≈ −0.45`,
  consistent with the Test A measurement of −0.67 (cache passes through additional model machinery
  that further attenuates the direction).

Sex (binary 0/1) and macro_system (categorical dummies) are not z-scored, so they have no
equivalent transform asymmetry.

## Fix direction for candidate 1 — IMPLEMENTED 2026-06-02

Make `_build_covariates` use **fit-time** z-scoring statistics at transform time, so single-subject
/ single-parcel transforms produce the same `z(age)` they would have had at fit. Three small edits
in `src/harmonize/combat.py`:

1. In `CombatHarmonizer.fit`: compute `age_mean_fit, age_sd_fit` from the pooled fit data (the
   `comb` DataFrame) before any transform is called, and store them on the dataclass.
2. In `_build_covariates`: accept an optional `(age_mean, age_sd)` override; when present, use it
   instead of re-computing from the input df.
3. In `CombatHarmonizer.transform` / `_cov_effect`: pass the stored stats into `_build_covariates`.

Verified by the structural invariant `||transform(1 row) - transform(N rows)[same row]|| = 0`
(previously diverged because single-row local z(age) collapsed to 0).

## Post-fix Test A (2026-06-02, candidate 1 only, t_prior_imq cache rebuilt)

```
                                    PRE-FIX             POST-FIX        change
bias-vs-z(age) slope LV components: [+29, +38, +1]      [-3.5, -4.3, +0.4]
||bias_slope_LV||                   48.0                5.54           −88%
cos(β_age,  bias slope)             -0.673              +0.618          (now on tiny residual)
cos(bias slope, AHBA-GTEx)          -0.642              +0.310          (now on tiny residual)

Per-year LV slopes (vs truth slopes from observation 1):
                      truth        pre-fix recon       post-fix recon
LV1 (/yr)             -0.18        +2.09               -0.29   ← bias gap +2.27 → -0.11
LV2 (/yr)             -0.39        +2.65               -0.35   ← bias gap +3.04 → +0.04
LV3 (/yr)             +0.15        +0.10               +0.03
```

Both candidate-1 (β_age direction) and candidate-2 (AHBA−GTEx direction) components collapsed
together. The two were not independent mechanisms — they were two manifestations of the same
upstream defect. The AHBA−GTEx centroid direction itself moved post-fix (norm 39 → 53, LV2
component flipped from −3 to +19), because correctly adding back the covariate repositions both
batches' centroids in LV space. The pre-fix "atlas pull" reading was implicitly measuring against
an artificially-displaced AHBA centroid.

## Second defect identified post-fix — pooled `β_age` confounds with batch (2026-06-02)

Post-fix Test A confirmed the recon-vs-truth gap collapsed, but **the harmonized truth's absolute
age slope was substantially larger than raw GTEx's** (truth LV1 add-back ≈ +2.0/yr vs raw GTEx
LV1 +0.47/yr). Raw and truth grids should track each other up to harmonization noise; instead the
post-fix truth showed a strong age ramp that wasn't in raw GTEx. Followup diagnostic
(`/tmp/age_beta_by_batch.py`, 2026-06-02) compared `β_age` fit per-batch vs. pooled:

```
                    pooled       AHBA-only    GTEx-only
||β_age||           93.5         16.9         7.1
||LV projection||   208.07       21.22        11.55
LV1 component       +23.7        -14.8        +5.5

cos(pool, AHBA-only)        = -0.743
cos(pool, GTEx-only)        = +0.926
cos(AHBA-only, GTEx-only)   = -0.932    ← the two batches' age signals point ≈ OPPOSITE directions
||pool|| / ||GTEx-only||    = 18.0      ← pooled OLS β_age is ~18× the magnitude of GTEx's own signal
||pool|| / ||AHBA-only||    = 9.8
```

**Mechanism.** ComBat's fit does a single pooled OLS of expression on `(1, z_age, sex, macro)`
without explicitly modeling the batch in the design. AHBA donors skew older than GTEx; age
therefore correlates with batch; and pooled OLS leaks part of the AHBA-vs-GTEx batch effect into
the `β_age` coefficient. The per-batch OLS confirms this: each batch's *own* `β_age` is far
smaller than the pooled fit (factor of 10–20) and the two per-batch directions are nearly
opposite (cos = −0.93). Pooling is inappropriate — there is no shared "age effect on expression"
across the two datasets to recover. Adding back the inflated pooled coefficient at transform
imposes AHBA's batch-confounded age signal on GTEx.

## Fix direction for the pooled-OLS defect — IMPLEMENTED 2026-06-02

Fit `β_cov` **per batch** (AHBA = 0, GTEx = 1) and use each row's batch-specific coefficients for
the remove/add-back step. Four small edits in `src/harmonize/combat.py`:

1. `CombatHarmonizer` dataclass gains `beta_cov_per_batch: np.ndarray` of shape
   `(B, n_cov, n_genes)`. The legacy `beta_cov` (n_cov, n_genes) field is retained and populated
   with the GTEx-batch coefficients for backward compatibility with any diagnostic readers.
2. `fit` replaces the pooled OLS-per-gene loop with two per-batch OLS solves (vectorized over
   all genes via `np.linalg.solve(D_b.T @ D_b, D_b.T @ X_b)`). Per-sample `cov_effect_per_sample`
   is built from each row's batch β, then standardization / gamma / delta proceed unchanged.
3. The `X_corr` reconstruction inside `fit` uses each row's batch β (`cov[i, :] @
   beta_cov_per_batch[batch[i]]`).
4. `_cov_effect(df)` → `_cov_effect(df, dataset_name)`, indexing `beta_cov_per_batch[bidx]`
   where `bidx = 0 if dataset_name=='AHBA' else 1`. `transform` threads `dataset_name` through.

This also automatically per-batches `β_sex` and `β_macro_system` (every covariate gets its own
per-batch coefficient), so any future analysis of dataset-specific sex effects is supported
without further code changes.

Expected post-fix effect on the harmonized truth: GTEx's truth LV1 add-back drops from the
pooled-coefficient ~+2.0/yr to GTEx's own +5.5/12.2 ≈ +0.45/yr — within noise of raw GTEx's
+0.47/yr. The harmonized truth grid should now track the raw GTEx grid in age dependence, both
visually and in LV slope, with the within-batch covariate signal preserved (the original design
intent). The recon-vs-truth bias remains collapsed because reconstruction trains on
GTEx-harmonized data (uses GTEx's β) and held-out truth is GTEx-harmonized (also uses GTEx's β) —
they share the same per-batch coefficient and cancel.

## Configurable mode added 2026-06-02 — `cov_batch_mode`

The per-batch β_cov decision is exposed via `cov_batch_mode` on the harmonizer / fit_harmonizer /
SubjectCacheConfig / CLI / sbatch env:

- **`mixed` (default)** — age and sex per batch; macro_system pooled. Rationale: subject-level
  demographic covariates (age, sex) can confound with batch composition; parcel-level spatial
  covariates (macro_system) are intrinsic biology shared across datasets, so pooling reduces
  variance without introducing confounding.
- **`pooled`** — every covariate column gets a single pooled coefficient (the pre-fix shape, but
  with the candidate-1 z-scoring fix still applied). Use as a reference/ablation.
- **`per_batch`** — every covariate column per batch. Maximally permissive; matches the
  intermediate implementation between the first per-batch fix and the `mixed` default.

The cache hash treats `mixed` as identity so existing caches built before this knob existed stay
valid by default; only non-default modes invalidate.

## Post-both-fixes Test A (2026-06-02, mixed mode, basis refit at 17:32, cache rebuilt at 16:57-58)

```
||beta_age_AHBA||                 = 34.28
||beta_age_GTEx||                 =  9.07

beta_age_GTEx LV components       = [-11.68, -7.33, -2.39]    ||.|| = 13.99
beta_age_AHBA LV components       = [-17.94, -30.36, -36.19]  ||.|| = 50.53
AHBA-GTEx centroid LV components  = [+71.58, +22.56, +22.87]  ||.|| = 78.46

bias-vs-z(age) slope LV components = [+0.26, -0.12, +0.24]    ||.|| =  0.37

Per-year LV bias slopes (recon - truth):
  LV1 = +0.021 /yr
  LV2 = -0.009 /yr
  LV3 = +0.020 /yr

cosines (now on a residual of magnitude 0.37 — essentially noise):
  cos(bias slope, beta_age_GTEx) = -0.528
  cos(bias slope, beta_age_AHBA) = -0.523
  cos(bias slope, AHBA-GTEx    ) = +0.734
```

Collapse trajectory of `||bias_slope_LV||`:
```
48.00   pre-fix (original artifact, both defects present)
 5.54   candidate-1 fix only (z-scoring degeneracy resolved)         -88% from pre-fix
 0.37   candidate-1 + per-batch beta_cov (default 'mixed' mode)      -99% from pre-fix
```

The cosines on the residual are not informative for mechanism — the residual magnitude is at
noise level. Per-year recon-vs-truth slopes on all three LVs are essentially zero (< ±0.03/yr).

## Post-fix progression grid read (2026-06-02, refit basis)

Visual inspection of `out/brain_renders/pls_progression_grid__gtex_{raw,truth,recon}.png` after
both fixes and the AHBA-basis refit:

- **Truth grid**: small absolute LV positions (LV1 ≈ −60 to −80, LV2 ≈ −50 to −90, LV3 ≈ +50)
  with near-flat trajectories per parcel. No monotone climb on any LV.
- **Recon grid**: tracks truth tightly in absolute LV-space (LV1 ≈ −60 to −110, LV2 ≈ −30 to
  −100, LV3 ≈ 0 to +50). Near-flat per-parcel trajectories.
- **Raw grid**: at much more negative absolute LV positions (LV1 ≈ −350 to −500, LV2 ≈ −400 to
  −650, LV3 ≈ −250 to −350) because raw log1p(TPM) GTEx is on a systematically lower numerical
  scale than the AHBA-microarray basis centers on. Trajectories per parcel are again near-flat.
  The shape (parcel-to-parcel ordering) matches truth/recon's shape up to the scale offset.

The qualitative artifact — monotone LV1/LV2 climb across age in both truth and recon — is gone in
all three grids. Truth and recon are now mutually consistent and qualitatively comparable to raw
(up to harmonization-induced scale shift), which was the original modeling intent for ComBat with
preserved biological covariates.

## Candidate areas of the pipeline for investigation

The bias is age-linear (R² ≈ 0.88), independent of every per-subject reconstruction-difficulty
metric tested (coverage, NN distance, gl3 fit quality), and present in **both** DLAM and PLAM
reconstructions. The candidate set is therefore restricted to surfaces those two models share:

1. **Covariate handling asymmetry between truth and reconstruction paths — RESOLVED.** The
   z-scoring degeneracy on single-subject transforms in `_build_covariates` stripped the β_age
   restoration on the held-out-truth path. Resolved by storing fit-time `(age_mean, age_sd)` on
   the harmonizer and reusing them at transform. Post-fix Test A confirmed an 88% collapse of the
   bias slope magnitude.
2. **(Originally listed as "atlas pull") — RESOLVED, but the mechanism was not atlas pull.**
   Post-fix Test A showed the AHBA−GTEx-aligned residual collapsed *together with* the β_age
   component when the candidate-1 fix landed. The two were the same defect viewed through
   different reference vectors; the AHBA−GTEx centroid had been artificially repositioned by the
   broken covariate handling. Subsequently a *second* harmonizer-side defect was identified: the
   pooled OLS for `β_cov` confounds `β_age` with batch (AHBA-GTEx age means differ), inflating the
   pooled coefficient ~10× over each batch's own per-batch fit and imposing AHBA's batch-confounded
   age signature on harmonized GTEx. Resolved by fitting `β_cov` per batch — see "Second defect
   identified post-fix" and "Fix direction for the pooled-OLS defect" sections above.

### Excluded by the PLAM observation (added 2026-05-28)

The following DLAM-specific paths previously listed for investigation are now ruled out because
PLAM exhibits the same artifact without using any of them:

- **The DLAM spatial / U-score reconstruction path** (`apply_linear`/`apply_scores`,
  `constrained_anchor`, `anchor_distance_*`).
- **The affine basis map** (`affine_gl3`, the M-fit-on-T-applied-to-U asymmetry).
- **The U → bridge ridge → invert chain.**

## Tests run (chronological)

**Test A — LV-space direction probe (2026-05-28).** Pre-fix: `cos(bias_slope, β_age) = −0.673`,
`cos(bias_slope, AHBA−GTEx) = −0.642`, β_age and AHBA−GTEx essentially orthogonal. Both candidates
appeared real and roughly equal in magnitude; Test A alone did not discriminate. Follow-up
`age_sign_diag.py` localized the β_age signature to the z-scoring degeneracy in `_build_covariates`.

**Test A-post-fix (candidate 1 fix applied, 2026-06-02).** `||bias_slope_LV||` collapsed 48 → 5.5
(−88%). Per-year LV1 bias `+2.27 → −0.11 /yr`; LV2 bias `+3.04 → +0.04 /yr`. Both the β_age and
AHBA−GTEx components collapsed together — they were the same defect viewed through different
reference vectors.

**Per-batch β_age diagnostic (2026-06-02).** Compared pooled / AHBA-only / GTEx-only `β_age` in
LV space (see "Second defect identified post-fix" section). Established that the pooled coefficient
is ~10× the magnitude of either per-batch coefficient, and that the two batches' age directions
are nearly opposite (cos = −0.93). Pooling is structurally inappropriate when batches have
heterogeneous covariate effects; fixed by per-batch β.

## Verification — all complete (2026-06-02)

1. **LORO cache** at `out/loro_subject_cache_cv_tprior_imq` rebuilt with both fixes applied
   (mtime 16:57-58).
2. **AHBA basis** at `out/pls_basis/ahba_basis__allgenes.pkl` refit on post-fix harmonized AHBA
   (mtime 17:32).
3. **PREPOST cache** at `notebooks/cache/prepost/b0f93ebf16a857f2.pkl` rebuilt (mtime 17:26).
4. **Progression grids** regenerated (mtime 17:37) and visually inspected — see "Post-fix
   progression grid read" section.
5. **Test A** re-run on the rebuilt cache + refit basis — `||bias_slope_LV|| = 0.37`, per-year
   recon-vs-truth slopes all `< ±0.03/yr` — see "Post-both-fixes Test A" section.

## Suggested follow-ups (optional, not blocking)

- Rebuild the `t_prior_gp` and `t_prior_tps` caches with `USE_CACHE=false` to confirm the same
  collapse on the other kernel variants. (The imq result is the primary verification; the other
  kernels share the same harmonizer code path and should behave identically.)
- Rebuild the PLAM cache. The artifact was confirmed present in PLAM pre-fix (observation 7);
  verifying its collapse closes the cross-model loop, but is not necessary for the artifact-fix
  story since both models share the same harmonizer.
- Run the analogous per-batch β diagnostic for `sex` and for each `macro_system` dummy to confirm
  the default `mixed` mode (sex per-batch, macro pooled) is the right call empirically; the
  current default is on *a priori* biological reasoning rather than measurement.

## Practical guidance for the progression notebook in the meantime

The `loro_fused` LV1/LV2 age progression should not be read as biology: both its global and its
region-specific components fail validation against held-out truth (facts 2–3). LV3 is comparatively
unaffected (fact 1). Extrapolated-ROI age trends have no ground truth and are unfalsifiable. A
`subtract_subject_mean` toggle (removes the global offset) and/or residualizing age were discussed
as display mitigations but do not address the source.

## Key code references

- `src/eval_utils/eval_pls_gradient_progression.py` — progression analysis (project all subjects, bin by age).
- `src/eval_utils/eval_pls_gradients.py`, `src/latent/pls.py` — frozen AHBA basis + forward projection.
- `src/latent/basis_maps.py:fit_basis_map` — `affine_gl3` (OLS affine subject→AHBA, `stabilize_invertible`).
- `src/eval_utils/dlam_diagnostics.py:_fit_dlam_fold_payload` — per-subject fold fit (harmonizer,
  subject PLS, basis map, `constrained_anchor` spatial reconstruction with `anchor_distance_shrink`).
- `src/harmonize/combat.py:_build_covariates` — age (z-scored) + sex + macro_system covariates.
- `src/eval/loro.py`, `src/workflows/loro_cache.py` — LORO reconstruction / fused-cube assembly.
