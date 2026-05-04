# eval refactor plan

Last touched: 2026-05-03 (single-subject module bootstrap)

## Current status

Population + single-subject evaluation surfaces are split. `eval_population.ipynb` runs through `src/eval_utils/eval_population.py`; the new `eval_singlesubject.ipynb` runs through `src/eval_utils/eval_single_subject.py`. Dataset-level work lives in `eval_data.ipynb` driven by `src/eval_utils/eda_core.py`.

`results_eda.py` remains a compatibility/reference surface for legacy fold-combo computation (`compute_fold_combo_metrics_from_cache`, `plot_fold_combo_ranked`), the heavyweight single-subject visualizations (matrix panel, alignment scatter, performance triplet, percentile-band representative selection), and a few helpers that haven't earned a dedicated module. `results_eda_arxiv.py` is read-only.

Pending refactor targets (in priority order):
- Migrate the remaining heavyweight single-subject visualizations from `results_eda.py` into `eval_single_subject.py` (`plot_subject_model_matrix_panel`, `plot_subject_alignment_scatter_panel`, `plot_single_subject_scatter_triplet`, `select_representative_subject_by_gtex_median`, `resolve_publication_subject`) and retire `results_single_subject_predictions.ipynb`.
- `eval_latent.py` — pooled PCA, reconstruction, PLS — migrate from `pca_cached_predictions.ipynb`.
- Decide which of the remaining `results_eda.py` helpers (`compute_fold_combo_metrics_from_cache`, the bare ranked plotter) get rewritten into `eval_population.py` / `eval_cache.py` versus permanently re-exported as a public facade.

## Architecture (stable)

### Naming
- Evaluation modules use the `eval_` prefix; only dataset-level work is called EDA.
- Population-level analyses live in `eval_population.py`. Subject-keyed analyses live in `eval_single_subject.py`. Neither lives in `results_eda.py`.

### Shared style — `eval_style.py`
- Canonical model order: `naive, dlam, plam` → display labels `Naive, DLAM, PLAM`.
- `MODEL_COLORS`, region-group palettes, `format_legend_label`, `strip_display_label_prefixes`, `model_label`, `ordered_models`, `FONT`.
- All plotting modules (`eval_population.py`, `eval_single_subject.py`, `eda_core.py`) route through these — no duplicated palettes or label maps.

### Dataset EDA — `eda_core.py`
- `EDAConfig`, repository/path resolution, gene-list resolution by path/basename/repo-relative.
- PREPOST reconstruction, demographics/coverage summaries, pre/post heatmaps.
- Default gene-list roots: `data/raw/gene_lists/`, `out/raw/gene_lists/`.

### Population prediction tables — `eval_population.py`
- Canonical wide tables: one `truth_df`, one `pred_dfs[model]` per model, rows = `subject_region_key`, gene columns = expression.
- Default cache: `out/eval_prediction_tables/<gene_scope>/`. Stable scope = all genes; gene lists subset downstream.
- Manifest tracks creation time, source caches, gene scope, row/gene/subject/parcel counts, expression-space semantics, source genes.

### Population metrics
- Default unit is `sample`: per `subject_region_key`, score across genes; aggregate across samples for stratified summaries.
- Gene-wise is the explicit alternative.
- Supported: `pearson_r`, `spearman_r`, `r2`, `rmse`. Spearman is opt-in (ranking-heavy).
- Vectorized/chunked with `n_jobs` + `chunk_size`.

### Subsetting / stratification
- `make_prediction_eval_view(...)` is the canonical filter entrypoint, shared between population and single-subject modules. Axes: gene list / custom genes / `eval_gene_list_path`, models, subjects, regions / parcels / region groups, sex, age, tissues.
- `region` / `tissue` aliases → `gtex_region`; `parcel`/`parcels` → `parcel_idx`.

### Population plotting
- Held-out-truth-vs-prediction scatters (3-panel `Naive | DLAM | PLAM`) with shared color/legend, balanced sampling keyed to color axis (visual only — never affects metric math), region-group/region/subject/gene/sex/age coloring, age young→old categorical order, sex green/orange, formatted gene-list label and `Brain - ` prefix stripping shared via `eval_style.py`.
- Per-stratum scatter with per-panel legend annotated by that model's metric per category (`plot_stratified_scatter`).
- Box (default) / violin distribution per stratum (`plot_stratified_distribution`, `kind='box'|'violin'`); `gtex_region` ordered cortical → subcortical → cerebellar via `_gtex_region_group_order`.
- Stratified metric tables (`format_stratified_metric_table`) — `MultiIndex(model, [metrics..., n])`, cells `mean ± std`.
- Coverage vs metric, fold-combo ranked overlays (independent + matched), per-model dist-colored ranked curves, distance-to-training (lines + thick errorbars + auto-zoom y).

### Confounder-controlled bias analysis (`compute_stratum_bias` + `plot_stratum_bias_forest`, in `eval_population.py`)
- Per-model fit: `fisher_z(metric) ~ C(axis, Treatment(ref)) + C(cat_controls) + numeric_controls`.
- `method='lmm'` (subject random intercept via `MixedLM`) with explicit fallback warning to `method='ols_cluster'` (cluster-robust SEs on subject) when LMM fails.
- Counterfactual-design adjusted means: `Var(x_avg @ β) = x_avg @ Cov @ x_avgᵀ`. Contrasts back-transformed to original scale.
- Wilks omnibus Wald test per axis term; df=1 fallback uses the contrast Wald when `wald_test` returns NaN under cluster covariance.
- BH-FDR across non-reference contrast p-values.
- Forest plot: per `(model, category)` errorbar at adjusted Δ vs reference, vertical zero line, omnibus footer outside the data area.

### Single-subject analyses — `eval_single_subject.py`
- **Per-subject performance** (`compute_subject_performance`, `select_subjects_by_percentile`, `plot_subject_performance_ranked`, `plot_fold_std_vs_mean`).
  - Input: `fold_perf_df` from `compute_fold_combo_metrics_from_cache` (per-`(subject, fold_key)` LORO output).
  - Output table carries `mean, std, n_folds, percentile_rank` per `(model, subject)`.
  - Selector supports two modes: `n_bands=N` (evenly-spaced percentile anchors) or `percentiles=(...)` (explicit quantiles); anchored on a chosen `(metric, model)` so the same subjects are compared across panels.
  - Ranked plot orders subjects by the **anchor model**'s mean so the same subject sits at the same x position under every model. `sparsify=N` collapses to N anchor subjects with percentile-labeled ticks; otherwise renders all subjects.
- **Subject specificity** (`compute_subject_specificity`, `plot_subject_specificity`).
  - Per LORO sample: `sim_self = metric(pred[s,r], truth[s,r])`; `sim_other_mean = mean_{s' ≠ s} metric(pred[s,r], truth[s',r])`.
  - Vectorized per region as a single (n_p × G) @ (G × n_p) `_pairwise_metric_matrix` call per (model, region).
  - Multi-metric (`pearson_r | spearman_r | r2 | rmse`) — direction flips for RMSE; metric stored in result column.
- **Spatial specificity** (`compute_spatial_specificity`, `plot_spatial_specificity`).
  - Sibling of subject specificity, but the null is **prediction-vs-prediction within the same subject**: `sim_other_region_mean = mean_{r' ≠ r in subject s} metric(pred[s,r], pred[s,r'])`.
  - Distinguishes models that copy a per-subject mean profile to every region (high subject specificity, near-zero spatial specificity) from models that genuinely localize.
  - Reuses the same `_pairwise_metric_matrix` primitive, applied per subject.
- **Shared split-violin plotter** (`_plot_specificity_split_violin`) — saturated/light shading per `MODEL_COLORS`, Wilcoxon footer, direction-aware (alt='greater' vs 'less'); both subject and spatial specificity wrap it. Single source of truth for self-vs-null violins.
- **Vectorized similarity primitives** (`_pairwise_metric_matrix`, `_row_z_normalize`, `_row_rank`) live with their callers in this module — not duplicated in `eval_population.py`.

## Notebook structure (`eval_population.ipynb`)

Top-down:
1. Path setup + dev reload.
2. **Single config cell** (CFG, MODELS, gene list, `VIEW_SUBSET`, `DEFAULT_METRIC='pearson_r'`, scatter knobs, sandbox knobs, LORO knobs, distance knobs, bias-forest knobs). Copy notebook → edit one cell → Restart & Run All.
3. PREPOST + global prediction tables.
4. **Scatter sandbox** — free-form `plot_global_prediction_scatter`.
5. **Sample-Wise Stratification** — four explicit sections, each self-contained:
    - Global → scatter, box, table.
    - Sex → scatter, box, table; opt-in `SHOW_BIAS_FOREST` adds the LMM-controlled forest panel.
    - Age → scatter, box, table.
    - Region (`gtex_region`) → scatter, box, table.
6. Coverage-vs-performance.
7. LORO fold-combo curves (independent + matched overlays + per-model dist-colored ranked curves).
8. Distance-to-training vs metric (lines + thick errorbars, no bars/histogram).

Subject specificity and the LORO bottom-tail per-subject diagnostic moved out to `eval_singlesubject.ipynb`.

## Notebook structure (`eval_singlesubject.ipynb`)

Top-down:
1. Path setup + dev reload (imports both `eval_population` and `eval_single_subject`).
2. **Single config cell** (CFG, MODELS, gene list, `VIEW_SUBSET`, `DEFAULT_METRIC`, `ANCHOR_MODEL='plam'`, `SPARSIFY=10`, `BOTTOM_HIGHLIGHT_N`, `PERFORMANCE_FOLD_METRIC`, `FOLD_COMBO_COVERAGE_*`, `SPECIFICITY_REGION_COL`).
3. PREPOST + global prediction tables + `subject_view`.
4. **Section 1 — Per-subject performance distribution.** LORO fold-combo cache load → `compute_subject_performance` → `select_subjects_by_percentile` (returned as `band_subjects` for downstream illustrative-subject deep dives) → `plot_subject_performance_ranked` (sparsified to anchor percentiles) → optional `plot_fold_std_vs_mean`.
5. **Section 2 — Subject specificity.** `compute_subject_specificity` → median-table + `plot_subject_specificity`.
6. **Section 3 — Spatial specificity.** `compute_spatial_specificity` → median-table + `plot_spatial_specificity`. Markdown explicitly contrasts the two specificity axes (right subject? right region within subject?).

## Module layout (current)

### `eval_style.py`
Shared plotting / naming only.
- Model order, labels, colors; region/parcel color conventions; label formatting; academic plotting defaults.

### `eda_core.py`
Dataset-level EDA + PREPOST reconstruction.
- `EDAConfig`, path & gene-list resolution, PREPOST reconstruction, demographic/coverage summaries, global matrix views, pre/post harmonization diagnostics.

### `eval_population.py`
Population-level cached prediction evaluation. Stable surfaces:
- Cache: `build_global_prediction_tables`, parquet read/write with permission repair.
- Views/metrics: `make_prediction_eval_view`, `compute_prediction_metrics`.
- Scatters: `plot_global_prediction_scatter`, `plot_stratified_scatter`.
- Distributions: `plot_stratified_distribution` (box default, violin opt-in), `plot_metric_violins` (legacy one-axis case), `plot_metric_delta_violins`.
- Tables: `format_stratified_metric_table` (with `_gtex_region_group_order`).
- Coverage / fold-combo / distance: `plot_coverage_vs_metric`, `plot_distance_to_train_vs_metric`.
- Bias analysis: `compute_stratum_bias`, `plot_stratum_bias_forest`.
- Re-exports from `results_eda.py`: `compute_fold_combo_metrics_from_cache`, `plot_fold_combo_ranked` (per-model with `points_style` colorbar), `plot_fold_combo_ranked_overlay`, `plot_fold_combo_matched_overlay`, plus a few pre-refactor helpers.

### `eval_single_subject.py`
Subject-keyed evaluation surfaces. Stable surfaces:
- Subject performance: `compute_subject_performance`, `select_subjects_by_percentile`, `plot_subject_performance_ranked`, `plot_fold_std_vs_mean`.
- Subject specificity: `compute_subject_specificity`, `plot_subject_specificity`.
- Spatial specificity: `compute_spatial_specificity`, `plot_spatial_specificity`.

### `eval_latent.py` *(pending)*
Planned: pooled PCA fit/eval, reconstruction/recovery, variance spectra, PLS fit + reconstruction, latent-score comparison plots. Should consume the canonical wide prediction tables.

### `results_eda.py`
Compatibility/reference. Migrate stable logic out into focused `eval_*` modules; keep re-exports until notebooks fully decouple.

## Notebook migration plan

### Done
- `results_cached_predictions.ipynb` split into `eval_data.ipynb` + `eval_population.ipynb`.
- Dataset demographics + global matrix/heatmap flow in `eval_data.ipynb`.
- Canonical wide tables, scatters, sample-wise metrics, box plots (replaced violins), region-group ordering for `gtex_region`.
- Gene-list path subsetting in views, scatters, metrics.
- Per-section stratified blocks (Global / Sex / Age / Region) in `eval_population.ipynb`.
- Sex bias forest (LMM with subject random intercept; controls for age, coverage, region; FDR; df=1 omnibus fallback; LMM-fallback warning).
- LORO ranked overlays (independent + matched) and per-model dist-colored ranked curves.
- Distance-to-training plot (line + thick errorbars + auto-zoom).
- `eval_single_subject.py` bootstrapped with subject performance + specificity + spatial specificity. Subject specificity and the LORO bottom-tail per-subject diagnostic removed from `eval_population.ipynb` and rehosted in `eval_singlesubject.ipynb`.
- Single config cell at the top of every eval notebook for one-edit reruns.
- Unified subject-percentile selector (`select_subjects_by_percentile` with `n_bands` or `percentiles`) replaces the older `select_subjects_by_metric_percentile` from `eval_population.py`.

### Next
- Migrate heavyweight single-subject visualizations from `results_eda.py` into `eval_single_subject.py` (`plot_subject_model_matrix_panel`, `plot_subject_alignment_scatter_panel`, `plot_single_subject_scatter_triplet`, `select_representative_subject_by_gtex_median`, `resolve_publication_subject`). Add a "percentile-band illustrative subjects" section to `eval_singlesubject.ipynb` keyed off `band_subjects`.
- Wire bias forest into the Age section once the Sex pattern is settled in writing.
- Decide whether `compute_fold_combo_metrics_from_cache` and `plot_fold_combo_ranked` get formally rewritten into `eval_population.py` (or a future `eval_cache.py`) or stay re-exported from `results_eda.py`.
- Begin `eval_latent.py` (PCA → PLS migration from `pca_cached_predictions.ipynb`).

## Validation checklist

For every patch:
- `python -m py_compile` changed `.py` modules.
- JSON-validate changed notebooks.
- Heavy table/cache work runs only on compute nodes, never login.
- Distinguish: source compiles vs notebook JSON valid vs notebook re-run end-to-end.

## Closed decisions
- Default scatter/legend metrics → `pearson_r` (Pearson is more stable than R² and sets the writeup's "headline" number).
- Distribution kind in stratified sections → **box plots** by default; violins opt-in via `kind='violin'` (heavy tails on per-sample Pearson r made violins misleading).
- Region stratification axis → `gtex_region` (full 13 tissues), not `region_group`. Region rows ordered cortical → subcortical → cerebellar everywhere.
- Subject specificity grouping → `parcel_idx` (matches eval unit; alternative `gtex_region` available behind `region_col`).
- Distance plot styling → no bars, no count histogram; line + capped errorbars + auto-zoomed y-axis.
- LMM fallback → emit `RuntimeWarning` and fall through to OLS+cluster transparently; row's `method` column records what actually fit.
- Subject-keyed analyses (per-subject performance, subject specificity, spatial specificity) live in `eval_single_subject.py`, not `eval_population.py` — even when their inputs come from population-level caches.
- Subject percentile selection unified into a single `select_subjects_by_percentile(perf_df, *, metric, model, n_bands=..., percentiles=...)` entrypoint anchored on subject-mean performance; the older sample-wise `select_subjects_by_metric_percentile` is retired.
- Spatial specificity null → **prediction-vs-prediction within subject** (not truth-vs-truth), so the test asks "are predictions spatially differentiated?" rather than "is truth spatially differentiated?" — the latter is a property of GTEx, not the model.
- Subject-performance ranked plot is anchored to one model so the same subject occupies the same x position under every model panel; sparsification picks subjects at evenly-spaced percentiles of that anchor.

## Open decisions
- Whether to add `eval_cache.py` for low-level `.npz` reading shared by population, single-subject, and latent modules.
- Whether `compute_prediction_metrics`'s `unit='global_flat'` stays exposed or gets gated behind an advanced flag.
- Whether parquet remains the default table-cache format on this filesystem or whether a fallback is needed for permission/quota issues.
- How much of `results_eda.py` should remain as a permanent public facade after notebooks fully migrate.
- Whether `compute_stratum_bias` should be re-fit with subject + region random effects when 13 region levels feel too df-heavy.
