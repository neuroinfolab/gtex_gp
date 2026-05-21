# 2D Brain Visualization

Prototype 2D visualization pipeline for GTEx gene expression data across brain parcels. Generates figures with **3 rows × N columns** where rows are anatomical views (sagittal, coronal, axial) and columns are data panels.

## Scripts

| Script | Description | Output folder |
|---|---|---|
| `viz_data.py` | Shared data loader — all scripts import this | — |
| `viz_nilearn.py` | nilearn glass-brain markers, GTEx Input vs Reconstruction | `outputs/v1_nilearn_folder/` |
| `viz_nilearn_wholebrain.py` | Same + third column for Whole-Brain Fit | `outputs/v1_nilearn_with_whole_brain/` |
| `viz_matplotlib.py` | Pure matplotlib bubble scatter with brain silhouette | `outputs/matplotlib/` |
| `viz_plotly.py` | Plotly scatter (PNG export via kaleido) | `outputs/plotly/` |
| `generate_all.py` | Runner that calls all three library approaches | all of the above |

## Layout

```
         GTEx Input (harm.)  |  Reconstruction  |  Whole-Brain Fit
         ──────────────────────────────────────────────────────────
Sagittal │                   │                  │
Coronal  │                   │                  │
Axial    │                   │                  │
```

- **GTEx Input** — combat-harmonized ground truth at LORO eval parcels (`fullfit_subject_h`)
- **Reconstruction** — LORO model output at eval parcels (`loro_fused_subject_h`)
- **Whole-Brain Fit** — dense fullfit across all 150 parcels, LH mirrored to RH

GTEx Input and Reconstruction share the same color scale so they are directly comparable.

## Usage

```bash
cd brain-viz-python/2d_visualization

# All subjects, all models, gene=DPM1 (default)
python generate_all.py

# Specific gene
python generate_all.py --gene MAPT

# Specific subject/model, nilearn only
python viz_nilearn.py --gene DPM1 --subjects GTEX-1117F --models naive dlam

# Just the wholebrain variant
python viz_nilearn_wholebrain.py --gene DPM1
```

### Arguments (all scripts)

| Flag | Default | Description |
|---|---|---|
| `--gene` | `DPM1` | Gene symbol to visualize |
| `--subjects` | all 5 | Space-separated subject IDs |
| `--models` | `naive dlam plam` | Model(s) to run |
| `--approaches` | all | `generate_all.py` only — subset of `nilearn plotly matplotlib` |

## Available subjects and models

**Subjects:** `GTEX-1117F`, `GTEX-13OW8`, `GTEX-11DZ1`, `GTEX-1B996`, `GTEX-1JMPZ`

**Models:** `naive`, `dlam`, `plam`

## Dependencies

```
nilearn
matplotlib
plotly
kaleido==0.2.1   # plotly static export; must be pinned to 0.2.1 for plotly 5.x
scipy
```

All other dependencies (`numpy`, `pandas`, `pyvista`) are shared with `app.py`.

## Notes

- **Raw vs. harmonized**: `v1_gtex_input` (raw counts) and `v2_reconstruction` (harmonized) are on different scales and cannot be compared directly. These scripts use `v1_gtex_harmonized` for the GTEx Input column so both panels are in the same combat-harmonized log₂ space.
- Outputs are gitignored — regenerate locally by running any script above.
