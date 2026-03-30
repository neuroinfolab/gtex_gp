# AHBA-GTEx Latent Alignment and Imputation

This repository contains the AHBA<->GTEx atlas-alignment workflow used for manuscript assets and LORO benchmarking across three model families:
- naive atlas-fill baseline
- DLAM (deterministic latent alignment)
- PLAM (probabilistic latent alignment)

## Current Entry Points

Primary write-up pipeline:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

Primary iterative analysis surface (current default for new work):
- `results_eda_cached_predictions_allgenes.ipynb`

Rank-specific all-genes EDA variants:
- `results_eda_cached_predictions_allgenes_rank3.ipynb`
- `results_eda_cached_predictions_allgenes_rank4.ipynb`
- `results_eda_cached_predictions_allgenes_dynamicrank.ipynb`

Companion fast-scope notebook:
- `results_eda_cached_predictions_hvg.ipynb`

Publication-ready cached-results notebook:
- `results_eda_publication_ready.ipynb`

Single-subject cache smoke/debug notebook:
- `loro_single_subject_test.ipynb`

DLAM single-subject diagnostics notebook:
- `results_eda_dlam_single_subject_deepdive.ipynb`

## Recent Workflow Additions (High Level)

- Subject-wise LORO caches now support stable downstream EDA without re-running full notebook workflows.
- PLAM rank experiments are supported via cache directories such as `plam_rank3`, `plam_rank4`, and `plam_dynamicrank`.
- `ar_utils/dlam_diagnostics.py` provides DLAM full-fit/LORO diagnostics with reusable plotting utilities.
- Publication-facing EDA is now centralized in `results_eda_publication_ready.ipynb` using cache-backed utilities.

## Repository Layout

- `src/`: reusable preprocessing, harmonization, models, evaluation
- `scripts/`: manuscript/workflow CLIs (including LORO panel builder)
- `ar_utils/`: subject-wise LORO cache builders + EDA utilities
- `notebooks/`: write-up notebook and paired Python source
- `docs/manuscript/`: TeX manuscript and mapping docs
- `configs/`: notebook workflow configs
- `tests/`: smoke/regression tests
- `legacy/`: archived exploratory material

## Data Paths

Expected defaults:
- `data/raw/gxp_samples.csv`
- `data/raw/ahba_100hvg.txt`

Legacy root-level fallbacks (`gxp_samples.csv`, `ahba_100hvg.txt`) remain supported for migration, but default code paths use `data/raw/`.

## Subject-Wise LORO Cache Workflow (`ar_utils`)

Subject/model caches are written to:
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.npz`
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.json`

`<gene_scope>` is one of `allgenes` or `hvg`.

Each `.npz` includes:
- `predictions_subject_h` (`parcel x gene`): final fused map
- `fallback_subject_h`: model fallback completion map
- `truth_loro_h`: held-out harmonized truth (`NaN` outside `loro_eval_mask`)
- `gtex_mask`: global GTEx-observed parcel mask (cohort-level)
- `loro_eval_mask`: subject-specific parcels with strict held-out predictions
- `imputed_mask`: complement of `loro_eval_mask`
- `plam_fold_latent_dim`: PLAM fold-level latent rank used at each held-out parcel (`-1` for non-PLAM/unset)
- metadata arrays (`gene_names`, `parcel_idx`, `subject_id`, `model_name`, `skipped_holds`)

Fusion behavior:
- strict LORO predictions at `loro_eval_mask`
- single full-data fallback model for all non-LORO parcels

Current defaults:
- eligible subjects: `min_observed_parcels=5`
- DLAM model gate: `c_min=4`
- PLAM dynamic rank: opt-in (`dynamic_rank=false` by default)
- dynamic-rank cap when enabled: `plam_latent_dim_max=10` (or `PLAM_MAX_RANK=10` in sbatch)

## Matching Note (GTEx -> AHBA Parcels)

- Target parcels are defined from AHBA (`tissue_or_parcel`) with parcel centroids.
- GTEx samples are mapped by nearest-neighbor in 3D to AHBA parcel centroids.
- If a row has multiple coordinates in `coordinates`, `gtex_gp` uses the **centroid of all listed coordinates** (not just the first coordinate) before nearest-neighbor assignment.
- GTEx native tissue labels are retained and can be inspected alongside mapped AHBA parcel labels.

## Quick Commands

Single subject (local):

```bash
python3 ar_utils/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --use-cache true
```

Single-subject sbatch:

```bash
SUBJECT_ID=GTEX-14ASI MODEL_NAME=all GENE_SCOPE=allgenes sbatch run_loro_cache_single_subject.sbatch
```

All-subject array sbatch:

```bash
GENE_SCOPE=allgenes sbatch run_loro_cache_array.sbatch
```

Dynamic-rank PLAM array run (all-genes):

```bash
MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch run_loro_cache_array.sbatch
```

HVG array run:

```bash
GENE_SCOPE=hvg sbatch run_loro_cache_array.sbatch
```

## Notes for Contributors

- Current iterative development target is **all-genes** EDA/utilities first.
- After steady-state behavior is reached, mirror updates into HVG notebook defaults.
- `EDAConfig` in `ar_utils/results_eda.py` supports model-directory overrides:
  - `naive_cache_dirname`, `dlam_cache_dirname`, `plam_cache_dirname`
  - useful for side-by-side rank experiments (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`)
- See `CONTEXT.md` for a fast onboarding summary intended for parallel agents.

Last updated at: 2026-03-30
