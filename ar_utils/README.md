# `ar_utils` Subject-Wise LORO Cache Architecture

This folder provides scalable subject-wise LORO cache generation and cache-based EDA helpers.

## Goals

- Persist subject/model outputs to avoid rerunning long LORO loops.
- Keep downstream-friendly tensors (`parcel x gene` per subject file).
- Preserve masks for strict held-out evaluation and full-map imputation analyses.

## Outputs

Default root:
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.npz`
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.json`

`gene_scope`:
- `allgenes` or `hvg`

Each `.npz` contains:
- `fullfit_subject_h` (`150 x G`): full-data completion/deployment map
  - observed subject parcels are preserved
  - missing parcels are imputed by the chosen model
- `fullfit_subject_raw` (`150 x G`): covariate-aware inverse-GTEx version of `fullfit_subject_h`
- `loro_fused_subject_h` (`150 x G`): final fused eval map
  - strict LORO predictions at `loro_eval_mask`
  - `fullfit_subject_h` for non-LORO parcels
- `loro_fused_subject_raw` (`150 x G`): covariate-aware inverse-GTEx version of `loro_fused_subject_h`
- `loro_truth_subject_h` (`150 x G`): held-out harmonized GTEx truth (`NaN` outside eval mask)
- `loro_truth_subject_raw` (`150 x G`): held-out parcel-averaged native GTEx truth (`NaN` outside eval mask)
- `gtex_mask` (`150`): global GTEx-observed parcel mask (cohort-level)
- `loro_eval_mask` (`150`): subject-specific strict LORO eval parcels
- `imputed_mask` (`150`): complement of `loro_eval_mask`
- `skipped_holds`, `gene_names`, `parcel_idx`, `subject_id`, `model_name`

## Scripts

- `run_loro_subject_cache.py`
  - Build cache for one subject and one/all models.

- `run_loro_cache_batch.py`
  - Batch driver supporting:
    - all subjects,
    - explicit subject,
    - subject by index,
    - sbatch-array subject selection.

## SBATCH Launchers (repo root)

- `run_loro_cache_single_subject.sbatch`
  - one-subject smoke test.

- `run_loro_cache_array.sbatch`
  - all-subject array run, one subject per task.

## Typical Usage

Single subject local:

```bash
python3 ar_utils/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --use-cache true
```

Single subject (sbatch):

```bash
SUBJECT_ID=GTEX-14ASI MODEL_NAME=all GENE_SCOPE=allgenes sbatch run_loro_cache_single_subject.sbatch
```

All subjects (sbatch array):

```bash
GENE_SCOPE=allgenes sbatch run_loro_cache_array.sbatch
```

Dynamic-rank PLAM experiment (rank capped by `PLAM_MAX_RANK` and per-fold train coverage):

```bash
MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch run_loro_cache_array.sbatch
```

## Notes

- Cache validity uses config hash + source signatures.
- `use_cache=true` is default and skips valid subject/model recomputation.
- Parcel aggregation defaults to `--atlas-agg mean` and can be switched to `--atlas-agg median`.
- GTEx parcel assignment defaults to `--gtex-rep-mode medoid --gtex-hemi-mode native`.
- Dynamic rank can be enabled for PLAM with `--dynamic-rank true` (or `DYNAMIC_RANK=true` in sbatch launchers).
- This workflow is designed for modular downstream analysis, not only manuscript panel generation.
