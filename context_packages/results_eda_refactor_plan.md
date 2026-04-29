# eval refactor plan

Date: 2026-04-29

## Current status

The refactor has started and the active target is now the `eval_*` workflow, not continued expansion of `src/eval_utils/results_eda.py`.

Do not touch `src/eval_utils/results_eda_arxiv.py`. It is a read-only backup/reference snapshot.

Current active root notebooks:

- `eval_data.ipynb`: dataset-level EDA, config, PREPOST reconstruction, gene-panel selection, parcel/subject coverage, demographics, global matrix views, and pre/post heatmaps.
- `eval_population.ipynb`: population-level model evaluation over cached LORO predictions, currently centered on canonical all-gene wide prediction tables, global scatters, sample-wise metrics, violins, region-group summaries, and coverage diagnostics.
- `pca_cached_predictions.ipynb`: still a reference for pooled prediction-table/latent-analysis design, not yet fully migrated.
- `results_single_subject_predictions.ipynb`: still pending dedicated single-subject extraction.

Current active modules:

- `src/eval_utils/eval_style.py`
- `src/eval_utils/eda_core.py`
- `src/eval_utils/eval_population.py`

`src/eval_utils/results_eda.py` remains available as a reference and compatibility source while functions are migrated, but new work should land in focused `eval_*` modules.

## Architecture decisions now considered stable

### Naming

- Evaluation modules use the `eval_` prefix.
- Only dataset-level exploratory analysis should be called EDA.
- Population/model evaluation should use `eval_population.py`, not `results_eda.py`.

### Shared style

`eval_style.py` owns shared plotting and naming conventions:

- canonical model order: `naive`, `dlam`, `plam`
- display labels: `Naive`, `DLAM`, `PLAM`
- model colors
- parcel/region color conventions
- GTEx/AHBA label cleanup
- global figure style defaults

Plots should use the shared model order and formal model labels throughout.

### Dataset EDA

`eda_core.py` owns:

- `EDAConfig`
- repository/path resolution
- gene-list path resolution
- PREPOST reconstruction
- dataset-level gene-panel resolution
- GTEx demographic summaries
- parcel/subject coverage tables
- global AHBA/GTEx mean and median matrix views
- pre/post harmonization heatmaps and region diagnostics

Gene-list files should be usable by path, bare basename, or repo-relative path. Practical roots include:

- `data/raw/gene_lists/`
- `out/raw/gene_lists/`

### Population prediction tables

`eval_population.py` now has a canonical population-level data structure:

- one wide `truth_df`
- one wide `pred_dfs[model]` table per model
- rows are subject-region units
- gene columns are expression values
- row metadata includes subject, age, sex, `subject_region_key`, parcel/region labels, region group, mapped coordinates, mapping distance, sample count, and subject coverage

The default cached table location is:

- `out/eval_prediction_tables/<gene_scope>/`

The initial stable cache scope is all genes. Gene-list evaluation should subset these all-gene tables downstream rather than writing separate prediction-table caches per gene list.

Cache metadata should track:

- creation time
- source cache directories
- model cache directory names
- gene scope
- row/gene/subject/parcel counts
- expression-space semantics
- source genes

The current expression-space semantics are harmonized truth and harmonized predictions.

### Population metrics

The default evaluation unit is sample-wise:

- each `subject_region_key` is scored across genes
- global or stratified summaries aggregate those sample-level metrics

This avoids flattening all regions and genes into one pooled vector by default.

Supported current metrics:

- Pearson r
- Spearman r
- R2
- RMSE

Metric computation is vectorized/chunked and exposes `n_jobs` and `chunk_size`. Spearman is the expensive metric because it requires ranking within each evaluated unit.

Gene-wise metrics are the main exception:

- each gene is scored across samples
- summaries aggregate over genes

### Subsetting and stratification

Prediction views should be built from `make_prediction_eval_view(...)`.

Supported subsetting axes:

- gene list / custom genes / `eval_gene_list_path`
- model list
- subjects
- regions / parcels
- region groups
- sex
- age

Supported stratification axes should be explicit columns in the eval view. Common aliases should resolve cleanly:

- `region` / `regions` / `tissue` / `tissues` -> `gtex_region`
- `parcel` / `parcels` -> `parcel_idx`

Gene-list subsetting must be available directly through downstream functions where useful, especially:

- `make_prediction_eval_view(...)`
- `plot_global_prediction_scatter(...)`
- `compute_prediction_metrics(...)`

### Population plotting

Current population plots should remain centered on:

- global true-vs-predicted scatters
- sample-wise metric violins
- paired model-delta violins
- region/region-group summary heatmaps
- subject-coverage vs performance plots
- fold-combo ranking overlays

For very large scatter payloads:

- sample points by default rather than rendering all 38M points
- keep equal aspect
- use quantile-based shared x/y limits
- support coloring by region group, region, subject, or gene
- keep rasterization available

Model plot order and labels must be `Naive`, `DLAM`, `PLAM` across all plot types.

## Target module layout

### `eval_style.py`

Shared plotting and naming only. No cache loading. No PREPOST construction.

Stable responsibilities:

- model order, labels, colors
- region/parcel color conventions
- label formatting
- academic plotting defaults

### `eda_core.py`

Dataset-level EDA and PREPOST reconstruction.

Stable responsibilities:

- `EDAConfig`
- path and gene-list resolution
- PREPOST reconstruction
- demographic and coverage summaries
- global matrix views
- pre/post harmonization diagnostics

### `eval_population.py`

Population-level cached prediction evaluation.

Current responsibilities:

- build/load canonical wide truth/prediction tables
- repair/read/write table cache permissions where possible
- create filtered eval views
- compute sample-wise, gene-wise, and global-flat metrics
- plot global scatters, violins, heatmaps, coverage diagnostics, and fold-combo overlays
- preserve legacy population helpers temporarily where notebooks still need them

Near-term responsibilities:

- make gene-list subset workflows ergonomic across every population plot/metric helper
- reduce dependence on legacy `results_eda.py` internals where practical
- add clearer metric provenance to summaries
- keep all heavy table construction cache-first

### `eval_single_subject.py`

Pending.

Planned responsibilities:

- single-subject cache loading
- subject percentile/representative selection
- single-subject performance tables
- publication-style single-subject scatters/heatmaps/matrix panels
- explicit subject selection APIs by model, metric, percentile, coverage, and gene subset

### `eval_latent.py`

Pending.

Planned responsibilities:

- pooled PCA fitting/evaluation
- reconstruction/recovery analyses
- variance spectra
- PLS fitting and reconstruction analysis
- latent score comparison plots

This should draw from the canonical prediction-table design where possible.

### `results_eda.py`

Reference and compatibility surface during migration.

Rules:

- do not add new eval-refactor features here unless needed as a short-lived bridge
- migrate stable logic into focused `eval_*` modules
- keep it available as implementation reference until notebooks are fully migrated

## Notebook migration plan

### Done or in progress

- Split `results_cached_predictions.ipynb` into:
  - `eval_data.ipynb`
  - `eval_population.ipynb`
- Added dataset demographics and global matrix/heatmap flow to `eval_data.ipynb`.
- Added canonical wide prediction tables, cache loading, global scatters, sample-wise metrics, violins, and region-group summaries to `eval_population.ipynb`.
- Added gene-list path subsetting for population views, scatters, and metrics.

### Next population pass

- Ensure every population plot consumes the canonical eval view or canonical metric tables.
- Normalize model order/labels/colors across remaining fold-combo and legacy plots.
- Add clearer controls for scatter rendering:
  - points vs hexbin
  - max sampled points
  - axis quantiles
  - rasterization
  - coloring axis
- Add metric-table provenance columns for gene subset, unit, stratification, and aggregation.
- Decide which legacy population helpers should be rewritten versus wrapped.

### Single-subject pass

- Create `eval_single_subject.py`.
- Move representative subject selection and single-subject plotting out of `results_eda.py`.
- Update `results_single_subject_predictions.ipynb`.

### Latent pass

- Create `eval_latent.py`.
- Migrate PCA cached-prediction analysis.
- Add PLS fitting/reconstruction analysis.

## Validation checklist

For each patch:

- `python -m py_compile` changed `.py` modules.
- JSON-validate changed notebooks.
- Do not run heavy table/cache construction on the login node.
- Distinguish:
  - source compiles
  - notebook JSON is valid
  - notebook has been rerun end to end

## Open decisions

- Whether to add `eval_cache.py` for low-level `.npz` reading shared by population, single-subject, and latent modules.
- Whether to add `eval_metrics.py` if metric logic grows beyond population use.
- Whether parquet remains the default table-cache format on the HPC filesystem or whether a fallback format should be added.
- Whether global-flat metrics should remain exposed as a diagnostic or be hidden behind an explicit advanced option.
- How much of `results_eda.py` should remain as a permanent public facade after notebooks fully migrate.
