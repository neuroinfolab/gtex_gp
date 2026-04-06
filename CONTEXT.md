# Repo Context

## Purpose

`gtex_gp` is the AHBA<->GTEx atlas-alignment repo used to:
- harmonize AHBA and GTEx gene expression into a shared domain,
- complete sparse GTEx subject maps over the full atlas,
- evaluate completion quality with strict leave-one-region-out (LORO) validation,
- generate manuscript-facing figures/tables.

## Current Development Direction

Near-term work is centered on:
- expanding `ar_utils` wrappers and reusable analysis functions,
- iterative feature development in `results_eda_cached_predictions_allgenes.ipynb`,
- later synchronization of stable patterns into `results_eda_cached_predictions_hvg.ipynb`.
- rank-comparison EDA via all-genes variants:
  - `results_eda_cached_predictions_allgenes_rank3.ipynb`
  - `results_eda_cached_predictions_allgenes_rank4.ipynb`
  - `results_eda_cached_predictions_allgenes_dynamicrank.ipynb`
- publication-facing cache EDA in:
  - `results_eda_publication_ready.ipynb`
- DLAM single-subject diagnostics in:
  - `results_eda_dlam_single_subject_deepdive.ipynb`

## Main Entrypoints

- `README.md` — top-level usage and commands
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb` — canonical manuscript workflow
- `scripts/build_dual_model_loro_metric_panels.py` — manuscript LORO panel builder
- `ar_utils/run_loro_subject_cache.py` — per-subject cache builder
- `ar_utils/run_loro_cache_batch.py` — batch/array subject driver
- `ar_utils/results_eda.py` — cache-based EDA utilities
- `ar_utils/dlam_diagnostics.py` — DLAM diagnostics fit/cache/plot utilities
- `results_eda_cached_predictions_allgenes.ipynb` — primary EDA notebook
- `results_eda_cached_predictions_hvg.ipynb` — fast-scope counterpart
- `results_eda_publication_ready.ipynb` — publication-ready cached prediction panels

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
   - `gtex_rep_mode=centroid|medoid` (default `medoid`)
   - `gtex_hemi_mode=native|mirror_left` (default `native`)
   - the chosen representative point is computed before nearest-neighbor parcel mapping.
8. Root-level sbatch launchers now pass:
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

## Core LORO Semantics

- Fold unit is a **subject-observed parcel**.
- For each fold, one parcel from one subject is removed from training.
- Harmonization is re-fit on fold training data.
- Held-out truth is harmonized with that fold harmonizer.
- Strict fold prediction is evaluated only at held-out parcel(s).
- Final `loro_fused_subject_h` is a fused map (strict LORO where available + `fullfit_subject_h` elsewhere).
- Raw-space companions (`*_subject_raw`) are also stored using covariate-aware inverse GTEx transforms for predictions and parcel-averaged native GTEx truth for held-out parcels.

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
  - `run_loro_cache_single_subject.sbatch`
  - `run_loro_cache_array.sbatch`
- Scripts are CPU-oriented; no required GPU path for current cache generation.
- Array jobs are one subject per task and support `GENE_SCOPE=hvg|allgenes`.
- Dynamic-rank PLAM launch example:
  - `MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch run_loro_cache_array.sbatch`

## Quick Start (Agent Onboarding)

1. Read `README.md` and this file.
2. Inspect `ar_utils/results_eda.py` and the all-genes notebook first.
3. Use `results_eda_publication_ready.ipynb` for publication panel iteration from caches.
4. Use `results_eda_dlam_single_subject_deepdive.ipynb` for DLAM mechanism figures.
5. Treat all-genes notebook as source of active iteration; port stable behavior into HVG notebook afterward.

Last updated at: 2026-03-30
