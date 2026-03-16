# AHBA-GTEx Latent Alignment and Imputation

This repository contains the current Python workflow for transcriptomic alignment between the Allen Human Brain Atlas (AHBA) and GTEx brain samples.

The canonical user entrypoint is a single notebook:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

That notebook orchestrates the full write-up workflow from a raw `gxp_samples.csv` file:
- shared harmonization,
- naive atlas-fill baseline,
- DLAM,
- PLAM,
- all-gene leave-one-region-out evaluation,
- atlas-level figures,
- representative-subject figures,
- manuscript-facing tables and figures.

## Repository layout

- `src/`: reusable model, evaluation, and workflow code
- `scripts/`: thin CLI wrappers around the current workflow code
- `notebooks/`: canonical notebook and paired Python source
- `docs/manuscript/`: TeX source and generated manuscript assets
- `configs/`: full and smoke workflow configs
- `tests/`: smoke and regression tests
- `legacy/`: archived MATLAB and exploratory notebook material

## Data placement

Expected raw input path:
- `data/raw/gxp_samples.csv`

The HVG list is expected at:
- `data/raw/ahba_100hvg.txt`

The workflow also supports the legacy root-level files `gxp_samples.csv` and `ahba_100hvg.txt` as fallbacks for local migration, but the repository default is `data/raw/`.

## Quick start

1. Create an environment and install dependencies.
2. Place `gxp_samples.csv` and `ahba_100hvg.txt` under `data/raw/`.
3. Run the smoke workflow first:

```bash
python3 scripts/run_writeup_workflow.py --config configs/notebook_smoke.yaml
```

4. Run the full workflow:

```bash
python3 scripts/run_writeup_workflow.py --config configs/notebook_full.yaml
```

## Notebook-first workflow

Open:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

The paired source-of-truth script is:
- `notebooks/ahba_gtex_writeup_end_to_end.py`

The notebook is stage-cached and restartable. Heavy outputs are written under:
- `out/notebook_writeup/`

Generated manuscript assets are synchronized to:
- `docs/manuscript/figs/`
- `docs/manuscript/tables/`

Those generated assets are intentionally ignored by Git. The tracked manuscript source is:
- `docs/manuscript/main.tex`

## Notes

- Raw data and generated outputs are intentionally excluded from version control.
- Legacy MATLAB and exploratory notebooks have been archived under `legacy/`.
