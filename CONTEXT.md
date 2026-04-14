# Repo Context

## Purpose

`gtex_gp` is the AHBA<->GTEx atlas-alignment repo used to:
- harmonize AHBA and GTEx gene expression into a shared domain,
- complete sparse GTEx subject maps over the full atlas,
- evaluate completion quality with strict leave-one-region-out (LORO) validation,
- generate manuscript-facing figures/tables.

## Current Development Direction

Near-term work is centered on:
- expanding reusable analysis functions in `src/eval_utils`, `src/viz`, and `src/workflows`,
- iterative feature development in `results_eda_cached_predictions.ipynb`,
- mixed-space raw-truth checks in `results_eda_cached_predictions_raw.ipynb`,
- GTEx/AHBA assignment inspection in `notebooks/coordinate_overlay_3d_mni.ipynb`,
- older variants retained under `notebooks/ar_notebooks/` when historical comparison is useful.

## Main Entrypoints

- `README.md` — top-level usage and commands
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb` — canonical manuscript workflow
- `scripts/build_dual_model_loro_metric_panels.py` — manuscript LORO panel builder
- `src/workflows/loro_cache.py` — per-subject cache builder
- `scripts/run_loro_cache_batch.py` — batch/array subject driver
- `src/eval_utils/results_eda.py` — cache-based EDA utilities
- `src/eval_utils/dlam_diagnostics.py` — DLAM diagnostics fit/cache/plot utilities
- `src/viz/coord_viz.py` — GTEx/AHBA coordinate overlay utilities
- `results_eda_cached_predictions.ipynb` — primary harmonized-space EDA notebook
- `results_eda_cached_predictions_raw.ipynb` — mixed-space raw-truth EDA notebook
- `notebooks/coordinate_overlay_3d_mni.ipynb` — parcel-assignment visualization notebook

## Key Recent Changes (Important)

1. Subject-wise `.npz` caching for naive/DLAM/PLAM under `out/loro_subject_cache/...`.
2. Strict LORO-at-evaluable-parcels + single full-data fallback elsewhere.
3. Cache schema includes masks for downstream modular evaluation:
   - `gtex_mask` (global GTEx parcels),
   - `loro_eval_mask` (subject-specific strict eval parcels),
   - `imputed_mask` (non-LORO parcels).
4. DLAM gate default lowered to `c_min=4` so low-coverage eligible subjects are fit in LORO folds.
5. PLAM dynamic-rank support added:
   - `dynamic_rank=true|false`
   - `plam_latent_dim_max` (default cap `10`)
   - fold-level rank persisted as `plam_fold_latent_dim` in cache `.npz`.
6. Atlas/parcel reduction in the cache pipeline is configurable:
   - `atlas_agg=mean|median`
   - applied consistently to AHBA atlas parcel aggregation, subject observed parcel aggregation, and parcel-averaged LORO truth.
7. GTEx parcel assignment is configurable in the shared loader and active fitting workflows:
   - `gtex_rep_mode=centroid|medoid` (default `centroid`)
   - `gtex_hemi_mode=native|mirror_left` (default `mirror_left`)
   - the chosen representative point is computed before nearest-neighbor parcel mapping.
8. Sbatch launchers under `scripts/sbatch/` now pass:
   - `LATENT_DIM`
   - `DYNAMIC_RANK`
   - `PLAM_MAX_RANK`
   - `ATLAS_AGG`
   - `GTEX_REP_MODE`
   - `GTEX_HEMI_MODE`
9. EDA config now supports model folder remapping (`*_cache_dirname`) for comparisons like:
   - `plam_rank3`
   - `plam_rank4`
   - `plam_dynamicrank`
10. Publication EDA workflow now emphasizes cache-backed plotting with minimal recomputation.
11. DLAM diagnostics were added for single-subject full-fit/LORO latent-alignment inspection.
12. Raw-space evaluation now defaults to mixed-space comparisons against native GTEx truth in the EDA notebooks.
13. Inverse-mapped raw prediction outputs were retired from the active cache/model pipeline; only native raw held-out truth is retained for mixed-space evaluation.

## Core LORO Semantics

- Fold unit is a **subject-observed parcel**.
- For each fold, one parcel from one subject is removed from training.
- Harmonization is re-fit on fold training data.
- Held-out truth is harmonized with that fold harmonizer.
- Strict fold prediction is evaluated only at held-out parcel(s).
- Final `loro_fused_subject_h` is a fused map (strict LORO where available + `fullfit_subject_h` elsewhere).
- Native raw-space held-out truth is still stored as `loro_truth_subject_raw` for mixed-space evaluation against parcel-averaged GTEx truth.

## Data / Path Assumptions

Default raw inputs:
- `data/raw/gxp_samples.csv`
- `data/raw/ahba_100hvg.txt`

Default outputs:
- write-up pipeline: `out/notebook_writeup/`
- subject caches: `out/loro_subject_cache/`
- slurm logs: `out/slurm/`

## Do Not Casually Modify

- LORO fold semantics (held-out parcel policy, fold harmonization behavior)
- cache field names/shapes consumed by EDA notebooks
- sbatch container activation pattern (`source /ext3/env.sh`) without checking cluster impact
- manuscript asset naming conventions used by TeX docs/scripts

## HPC Notes

- Root sbatches are the current launch points:
  - `scripts/sbatch/run_loro_cache_single_subject.sbatch`
  - `scripts/sbatch/run_loro_cache_array.sbatch`
- Scripts are CPU-oriented; no required GPU path for current cache generation.
- Array jobs are one subject per task and support `GENE_SCOPE=hvg|allgenes`.
- Dynamic-rank PLAM launch example:
  - `MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch scripts/sbatch/run_loro_cache_array.sbatch`

## Quick Start (Agent Onboarding)

1. Read `README.md` and this file.
2. Inspect `src/eval_utils/results_eda.py` and `results_eda_cached_predictions.ipynb` first.
3. Use `results_eda_cached_predictions_raw.ipynb` for mixed-space raw-truth checks.
4. Use `notebooks/coordinate_overlay_3d_mni.ipynb` when working on spatial assignment or matching changes.
5. Treat `notebooks/ar_notebooks/` as historical variants unless you intentionally need one.

Last updated at: 2026-04-13
