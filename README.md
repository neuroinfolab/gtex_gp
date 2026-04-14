# AHBA-GTEx Latent Alignment and Imputation

This repository contains the AHBA<->GTEx atlas-alignment workflow used for manuscript assets and LORO benchmarking across three model families:
- naive atlas-fill baseline
- DLAM (deterministic latent alignment)
- PLAM (probabilistic latent alignment)

## Current Entry Points

Primary write-up pipeline:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

Active analysis notebooks:
- `results_eda_cached_predictions.ipynb` — default harmonized-space cache EDA
- `results_eda_cached_predictions_raw.ipynb` — mixed-space raw-truth cache EDA
- `notebooks/coordinate_overlay_3d_mni.ipynb` — GTEx/AHBA spatial assignment inspection

Legacy or exploratory notebook variants now live under:
- `notebooks/ar_notebooks/`

## Recent Workflow Additions (High Level)

- Subject-wise LORO caches now support stable downstream EDA without re-running full notebook workflows.
- PLAM rank experiments are supported via cache directories such as `plam_rank3`, `plam_rank4`, and `plam_dynamicrank`.
- Active cache/eval workflows now keep harmonized predictions plus native raw held-out truth; inverse-mapped raw prediction outputs were retired.
- `src/eval_utils/dlam_diagnostics.py` provides DLAM full-fit/LORO diagnostics with reusable plotting utilities.
- Publication-facing EDA is now centralized in `results_eda_publication_ready.ipynb` using cache-backed utilities.

## Repository Layout

- `src/`: reusable preprocessing, harmonization, models, workflows, evaluation utilities, visualization
- `scripts/`: manuscript/workflow CLIs and cache-generation entrypoints
- `scripts/sbatch/`: Slurm launchers for cache experiments
- `notebooks/`: write-up and active analysis notebooks
- `docs/manuscript/`: TeX manuscript and mapping docs
- `configs/`: notebook workflow configs
- `tests/`: smoke/regression tests
- `legacy/`: archived exploratory material

## Data Paths

Expected defaults:
- `data/raw/gxp_samples.csv`
- `data/raw/ahba_100hvg.txt`

Legacy root-level fallbacks (`gxp_samples.csv`, `ahba_100hvg.txt`) remain supported for migration, but default code paths use `data/raw/`.

## Subject-Wise LORO Cache Workflow

Subject/model caches are written to:
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.npz`
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.json`

`<gene_scope>` is one of `allgenes` or `hvg`.

Each `.npz` includes:
- `fullfit_subject_h` (`parcel x gene`): full-data completion/deployment map
- `loro_fused_subject_h`: final eval-ready fused map
- `loro_truth_subject_h`: held-out harmonized truth (`NaN` outside `loro_eval_mask`)
- `loro_truth_subject_raw`: held-out parcel-averaged native GTEx truth (`NaN` outside `loro_eval_mask`)
- `gtex_mask`: global GTEx-observed parcel mask (cohort-level)
- `loro_eval_mask`: subject-specific parcels with strict held-out predictions
- `imputed_mask`: complement of `loro_eval_mask`
- `plam_fold_latent_dim`: PLAM fold-level latent rank used at each held-out parcel (`-1` for non-PLAM/unset)
- metadata arrays (`gene_names`, `parcel_idx`, `subject_id`, `model_name`, `skipped_holds`)

Fusion behavior:
- strict LORO predictions at `loro_eval_mask`
- single full-data fallback model for all non-LORO parcels

Evaluation note:
- harmonized-space evaluation remains the default
- raw-space follow-up uses mixed-space correlation against `loro_truth_subject_raw` in `results_eda_cached_predictions_raw.ipynb`

Core implementation modules:
- `src/workflows/loro_cache.py`
- `scripts/run_loro_cache_batch.py`
- `scripts/sbatch/run_loro_cache_single_subject.sbatch`
- `scripts/sbatch/run_loro_cache_array.sbatch`
- `src/eval_utils/results_eda.py`
- `src/eval_utils/dlam_diagnostics.py`
- `src/viz/coord_viz.py`

Current defaults:
- eligible subjects: `min_observed_parcels=5`
- DLAM model gate: `c_min=4`
- atlas/parcel aggregation: `atlas_agg=mean` (switchable to `median`)
- GTEx representative point for parcel assignment: `gtex_rep_mode=centroid`
- GTEx hemisphere preprocessing for parcel assignment: `gtex_hemi_mode=mirror_left`
- PLAM dynamic rank: opt-in (`dynamic_rank=false` by default)
- dynamic-rank cap when enabled: `plam_latent_dim_max=10` (or `PLAM_MAX_RANK=10` in sbatch)

## Matching Note (GTEx -> AHBA Parcels)

- Target parcels are defined from AHBA (`tissue_or_parcel`) with parcel centroids.
- GTEx samples are mapped by nearest-neighbor in 3D to AHBA parcel centroids.
- If a row has multiple coordinates in `coordinates`, `gtex_gp` uses the configured representative point across all listed coordinates before nearest-neighbor assignment (`gtex_rep_mode=centroid|medoid`).
- GTEx native tissue labels are retained and can be inspected alongside mapped AHBA parcel labels.

## Quick Commands

Single subject (local):

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --use-cache true
```

Median aggregation variant:

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --atlas-agg median
```

Historical/original behavior before these new spatial options:

- representative point: `centroid`
- hemisphere preprocessing: `native`

Current default fitting behavior:

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes
```

Override example:

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --gtex-rep-mode medoid --gtex-hemi-mode native
```

Single-subject sbatch:

```bash
SUBJECT_ID=GTEX-14ASI MODEL_NAME=all GENE_SCOPE=allgenes sbatch scripts/sbatch/run_loro_cache_single_subject.sbatch
```

All-subject array sbatch:

```bash
GENE_SCOPE=allgenes sbatch scripts/sbatch/run_loro_cache_array.sbatch
```

Dynamic-rank PLAM array run (all-genes):

```bash
MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch scripts/sbatch/run_loro_cache_array.sbatch
```

HVG array run:

```bash
GENE_SCOPE=hvg sbatch scripts/sbatch/run_loro_cache_array.sbatch
```

## Notes for Contributors

- Current iterative development target is **all-genes** EDA/utilities first.
- Keep new work in the active notebooks under `notebooks/`; treat `notebooks/ar_notebooks/` as older exploratory variants unless explicitly reviving one.
- `EDAConfig` in `src/eval_utils/results_eda.py` supports model-directory overrides:
  - `naive_cache_dirname`, `dlam_cache_dirname`, `plam_cache_dirname`
  - useful for side-by-side rank experiments (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`)
- See `CONTEXT.md` for a fast onboarding summary intended for parallel agents.

Last updated at: 2026-04-13
