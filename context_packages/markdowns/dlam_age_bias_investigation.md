# DLAM Age-Bias Investigation — Handoff

Status: **open**. The artifact is characterized; its source in the pipeline is **not yet
identified**. This document records only tested facts and the candidate areas of the pipeline left
to investigate. Last updated 2026-05-25.

## Formal issue

The DLAM `loro_fused` reconstruction carries a near-linear **age trend on the PLS gradient axes
LV1/LV2** (per-subject LV score rises with subject age) that is **absent in the held-out
ground truth** at the same parcels. The trend appears in essentially every parcel and is the
dominant between-subject signal on LV1/LV2 in the reconstruction. Because it is not present in the
truth, it is an artifact of the reconstruction, not a biological age effect. The mechanism by which
it enters the reconstruction is unresolved.

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

## Candidate areas of the pipeline for investigation

The bias is age-linear (R² ≈ 0.88) and independent of every per-subject reconstruction-difficulty
metric tested (coverage, NN distance, gl3 fit quality). Untested places where an age-dependent term
could enter the reconstruction:

1. **Covariate handling asymmetry between paths.** Whether the ComBat age covariate is
   applied/removed/re-added differently on the held-out-truth path vs the fused-reconstruction
   assembly. Code: `src/harmonize/combat.py` (`fit`/`transform`, covariate add-back), and the
   reconstruction/assembly path `src/eval/loro.py`, `src/workflows/loro_cache.py`.
2. **The AHBA reference's position relative to GTEx in LV space.** The measured +9.1 LV1 offset
   between AHBA and GTEx centroids, and how the reconstruction uses the reference. Code:
   `src/eval_utils/dlam_diagnostics.py:_fit_dlam_fold_payload` (AHBA PLS, `t_ref_full`), the spatial
   reconstruction (`constrained_anchor`).
3. **The spatial / U-score reconstruction path** (`apply_linear`/`apply_scores`, `constrained_anchor`,
   `anchor_distance_*`) — i.e., the steps after the basis map, which were not separately
   characterized (only the basis-map fit quality and the nearest-distance input were tested).

## Proposed next test (cheap, no retrain)

Extract the fitted ComBat per-gene age coefficient (`β_age`), project it onto the frozen AHBA basis,
and compare its direction in LV space to (a) the measured per-subject bias direction `recon − truth`
and (b) the AHBA-vs-GTEx offset (+9.1 on LV1). Whichever the bias direction matches localizes the
search to area 1 (covariate handling) or area 2 (reference offset). This test has not been run.

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
