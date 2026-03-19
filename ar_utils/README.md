# `ar_utils` Subject-Wise LORO Cache Architecture

This folder adds a scalable, subject-wise cache workflow for strict LORO outputs.

## Goals

- Persist model-specific subject tensors to avoid rerunning long LORO loops.
- Keep shape compatible with downstream code: `subject x parcel x gene` (single-subject files are `parcel x gene`).
- Preserve masks for modular evaluation and comparison against held-out GTEx truth and AHBA distributions.

## Outputs

Default root:

- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.npz`
- `out/loro_subject_cache/<gene_scope>/<model>/<subject>.json`

Default testing scope:

- `gene_scope=hvg` (faster for development/testing).

Each subject/model `.npz` contains:

- `predictions_subject_h` (`150 x G`) final tensor:
  - LORO-held parcels use strict held-out prediction.
  - Remaining parcels use model-specific fallback completion.
- `fallback_subject_h` (`150 x G`) model fallback completion.
- `truth_loro_h` (`150 x G`) held-out harmonized GTEx truth (`NaN` outside `loro_eval_mask`).
- `gtex_mask` (`150`) parcels observed for that subject.
- `loro_eval_mask` (`150`) parcels with successful strict LORO fold prediction.
- `imputed_mask` (`150`) complement of `loro_eval_mask`.
- `held_out_parcels`, `skipped_holds`, `gene_names`, `parcel_idx`, `subject_id`, `model_name`.

## Scripts

- `run_loro_subject_cache.py`
  - Build cache for one subject and one/all models.

- `run_loro_cache_batch.py`
  - Batch driver:
    - all subjects
    - one explicit subject
    - one subject by index
    - one array-task subject (`--from-sbatch-array`)

## SBATCH Launchers (repo root)

- `run_loro_cache_single_subject.sbatch`
  - one-subject smoke test.

- `run_loro_cache_array.sbatch`
  - array execution, one subject per task.

## Typical Usage

Single subject local:

```bash
python3 ar_utils/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope hvg --use-cache true
```

Array (cluster):

```bash
sbatch run_loro_cache_array.sbatch
```

## Notes

- Cache validity is based on config hash + source signatures (csv/hvg path, size, mtime).
- `use_cache=true` is default and skips recomputation when a valid subject/model cache exists.
- This workflow is intentionally independent of the current `allgene_loro` stage manifest so it can scale per subject.
