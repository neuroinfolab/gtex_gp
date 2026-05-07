# eval refactor plan

Last touched: 2026-05-06 (genewise + Kendall-τ + cached PREPOST + cache layout shift)

## Current status

Four active eval notebooks are wired through focused source modules:
- `eval_data.ipynb` ← `src/eval_utils/eda_core.py` (dataset-level EDA, PREPOST).
- `eval_population.ipynb` ← `src/eval_utils/eval_population.py` (population-level cached prediction evaluation).
- `eval_singlesubject.ipynb` ← `src/eval_utils/eval_single_subject.py` (subject-keyed analyses).
- `eval_population_genewise.ipynb` ← `src/eval_utils/eval_population.py` (per-gene + within-subject spatial Kendall-τ).

Shared style + global infrastructure lives in `src/eval_utils/eval_style.py`. `results_eda.py` is now compatibility/reference only (legacy fold-combo computation, unmigrated single-subject visualizations). `results_eda_arxiv.py` is read-only.

**Cache layout convention** (since 2026-05-06): `out/` holds true prediction artifacts only. Helper-level caches (PREPOST, genewise Kendall) live under `notebooks/cache/<name>/<hash>.pkl`.

Pending refactor targets (in priority order):
- Migrate the remaining heavyweight single-subject visualizations from `results_eda.py` into `eval_single_subject.py` (`plot_subject_model_matrix_panel`, `plot_subject_alignment_scatter_panel`, `plot_single_subject_scatter_triplet`, `select_representative_subject_by_gtex_median`, `resolve_publication_subject`) and retire `results_single_subject_predictions.ipynb`.
- `eval_latent.py` — pooled PCA, reconstruction, PLS — migrate from `pca_cached_predictions.ipynb`.
- Decide whether `compute_fold_combo_metrics_from_cache` and the bare `plot_fold_combo_ranked` get formally rewritten into `eval_population.py` (or a future `eval_cache.py`) or stay re-exported from `results_eda.py`.

## Architecture (stable)

### Naming
- Evaluation modules use the `eval_` prefix; only dataset-level work is called EDA.
- Population-level analyses live in `eval_population.py`. Subject-keyed analyses live in `eval_single_subject.py`. Neither lives in `results_eda.py`.

### Shared style — `eval_style.py`

**Model + region conventions.** Canonical model order `naive, dlam, plam` → display labels `Naive, DLAM, PLAM`. `MODEL_COLORS`, `PARCEL_GROUP_COLORS`, region-group palettes, `model_label`, `ordered_models`, `strip_display_label_prefixes`. All plotting modules route through these — no duplicated palettes or label maps.

**Canonical metric labels.** `METRIC_LABELS` / `METRIC_LABELS_MEAN` and `format_metric_label(metric)`:
- `pearson_r → "Pearson $r$"` (italic *r*)
- `spearman_r → "Spearman $\rho$"`
- `r2 → "$R^2$"`
- `rmse → "RMSE"`
- `kendall_tau → "Kendall $\tau$"`
- `mean_*` variants prefix with `"Mean fold "`.

`format_legend_label` delegates to `format_metric_label` for any known metric key — call sites that previously did `format_legend_label("pearson_r")` automatically get the typeset form. **Never hardcode metric strings in plot text.** A persistent feedback memory documents this convention.

**Token-based font system.** Used by hero plotters; smaller plots can adopt as needed.
- `FONT_TOKENS = {"xs": 7, "s": 9, "m": 10, "l": 12, "xl": 14, "xxl": 16}` × module-level `_FONT_SCALE`.
- `font_size(spec)` resolves a token (`"m"`), token±offset (`"l+2"`, `"xl-1"`), or absolute int.
- `set_font_scale(scale)` re-pushes rcParams so a single knob rescales every plot.
- `_resolve_fonts(defaults, override)` merges per-call `font_sizes={...}` over a plotter's `_DEFAULT_FONT_SIZES` dict and resolves every entry to an int.
- Standardized element keys: `title, xlabel, ylabel, tick, legend, legend_title, annotation, footer, cbar_label, cbar_tick, suptitle, split_panel_title, split_panel_body`.
- Hero plotters that have adopted the system: `plot_global_prediction_scatter`, `plot_fold_combo_ranked_overlay`, `plot_fold_combo_matched_overlay`, `plot_fold_std_vs_mean`, `plot_subject_performance_ranked`, `plot_genewise_ranked_overlay`, `plot_genewise_histogram`, `plot_unitwise_kendall_ranked`. Legacy `FONT["title"] + N` patterns elsewhere coexist; migrate when touching a plot for other reasons.

**Tick visibility enforcement.** `seaborn.set_theme(style="whitegrid")` otherwise hides tick marks. `_TICK_RC` is injected via `sns.set_theme(rc=...)` inside `set_academic_style()`, re-asserted at module import (so ticks are on even before `set_academic_style` is called), and `apply_tick_style(ax, label_fontsize=...)` is available as a per-axes safety net. Hero plotters call it explicitly on their main axis.

**Legacy `FONT` dict** (`{"title": 11, "label": 10, "tick": 9, "legend": 9, "small": 8}`) preserved for unmigrated callers.

### Dataset EDA — `eda_core.py`
- `EDAConfig`, repository/path resolution, gene-list resolution by path/basename/repo-relative.
- PREPOST reconstruction, demographics/coverage summaries, pre/post heatmaps.
- Default gene-list roots: `data/raw/gene_lists/`, `out/raw/gene_lists/`.
- **`prepare_pre_post_harmonization_cached(cfg, *, force_rebuild=False, cache_dir=None)`** — disk-cached PREPOST. Hash key covers EDAConfig fields (`csv_path`, `hvg_path`, `gene_scope`, `min_observed_parcels`, `combat_use_covariates`, `gtex_rep_mode`, `gtex_hemi_mode`) + the input files' mtime + size, so the cache invalidates automatically when underlying data changes. Default cache: `notebooks/cache/prepost/<hash>.pkl`. Defensive chmod-after-mkdir + warn-and-continue on PermissionError covers the historical NFSv4-ACL-zero-mode bug; now redundant after the ACL fix but stays as a safety net.

### Population prediction tables — `eval_population.py`
- Canonical wide tables: one `truth_df`, one `pred_dfs[model]` per model, rows = `subject_region_key`, gene columns = expression.
- Default cache: `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/` (keyed on `cfg.cache_root.name` so iterating LORO caches don't collide). Stable scope = all genes; gene lists subset downstream via `eval_gene_list_path`.
- Manifest tracks creation time, source caches, gene scope, row/gene/subject/parcel counts, expression-space semantics, source genes.

### Population metrics
- Default unit is `sample`: per `subject_region_key`, score across genes; aggregate across samples for stratified summaries.
- Gene-wise is the explicit alternative.
- Supported: `pearson_r`, `spearman_r`, `r2`, `rmse`. Spearman is opt-in (ranking-heavy).
- Vectorized/chunked with `n_jobs` + `chunk_size`.

### Subsetting / stratification
- `make_prediction_eval_view(...)` is the canonical filter entrypoint, shared between population and single-subject modules. Axes: gene list / custom genes / `eval_gene_list_path`, models, subjects, regions / parcels / region groups, sex, age, tissues.
- `region` / `tissue` aliases → `gtex_region`; `parcel`/`parcels` → `parcel_idx`.
- Single-subject illustrative scatters use `make_prediction_eval_view(models=[ANCHOR_MODEL], subjects=[subj])` + `plot_global_prediction_scatter` rather than a bespoke helper — composes existing primitives.

### Population plotting
- Held-out-truth-vs-prediction scatters (3-panel `Naive | DLAM | PLAM`, or single-panel via `models=[…]` filter) with shared color/legend, balanced sampling keyed to color axis (visual only — never affects metric math), region-group/region/subject/gene/sex/age coloring, age young→old categorical order, sex green/orange, gene-list label and `Brain - ` prefix stripping shared via `eval_style.py`.
- Per-stratum scatter with per-panel legend annotated by that model's metric per category (`plot_stratified_scatter`).
- Box (default) / violin distribution per stratum (`plot_stratified_distribution`, `kind='box'|'violin'`); `gtex_region` ordered cortical → subcortical → cerebellar via `_gtex_region_group_order`.
- Stratified metric tables (`format_stratified_metric_table`) — `MultiIndex(model, [metrics..., n])`, cells `mean ± std`.
- Coverage vs metric, distance-to-training (lines + thick errorbars + auto-zoom y).

### Unified LORO fold-combo overlay (`plot_fold_combo_ranked_overlay`)
- Each model's combos ranked independently; x-axis is rank percentile (worst → best). Worst/median/best splits per model render in the bottom row.
- `points_style`:
  - `"line"` (default): connected line per model, colored by `MODEL_COLORS`.
  - `"dist_to_nearest_train"` / `"dist_to_centroid_train"`: scatter per model. Each model gets its own light → base → dark `LinearSegmentedColormap` keyed to the chosen distance, marker shape encodes the model. A single colorbar is shown for `reference_model` only (default `"plam"`); other models' distance ranges share the same vmin/vmax.
- `show_running_average=True` overlays a rolling mean (window `running_average_window`) per model in the model's solid base color, on top of per-point gradient. Useful in scatter mode.
- Companion `plot_fold_combo_matched_overlay` keeps the matched-fold-key version (rank x-axis pinned to the reference model's order).

### Fold-difficulty decomposition (`plot_fold_std_vs_mean`)
- Per-fold cross-subject std vs fold mean. Quadrant reading: low-mean/low-std = structurally hard; low-mean/high-std = subject-mixing failure; high-mean/low-std = easy; high-mean/high-std = mixed.
- `min_subjects=5` (default) drops fold-combos where the std is dominated by sampling noise; the dropped count appears in the figure footer.
- Marker area scales linearly with `n_subjects` (median size shown in the upper-right annotation, "marker size ∝ subjects per fold").
- Lives in `eval_population.py` and is rendered in the LORO section of `eval_population.ipynb`.

### Confounder-controlled bias analysis (`compute_stratum_bias` + `plot_stratum_bias_forest`, in `eval_population.py`)
- Per-model fit: `fisher_z(metric) ~ C(axis, Treatment(ref)) + C(cat_controls) + numeric_controls`.
- `method='lmm'` (subject random intercept via `MixedLM`) with explicit fallback warning to `method='ols_cluster'` (cluster-robust SEs on subject) when LMM fails.
- Counterfactual-design adjusted means: `Var(x_avg @ β) = x_avg @ Cov @ x_avgᵀ`. Contrasts back-transformed to original scale.
- Wilks omnibus Wald test per axis term; df=1 fallback uses the contrast Wald when `wald_test` returns NaN under cluster covariance.
- BH-FDR across non-reference contrast p-values.
- Forest plot: per `(model, category)` errorbar at adjusted Δ vs reference, vertical zero line, omnibus footer outside the data area.

### Single-subject analyses — `eval_single_subject.py`

**Per-subject performance** (`compute_subject_performance`, `select_subjects_by_percentile`, `plot_subject_performance_ranked`).
- Input: `fold_perf_df` from `compute_fold_combo_metrics_from_cache` (per-`(subject, fold_key)` LORO output).
- Output table: `mean, std, n_folds, percentile_rank` per `(model, subject)`. **Performance-oriented percentile**: 0 = worst, 100 = best, regardless of metric direction (RMSE inverted via `_HIGHER_IS_BETTER`).
- `select_subjects_by_percentile(perf_df, *, metric, model, n_bands=N or percentiles=[…])` — direction-aware sort + raw-quantile mapping (perf_q ↔ raw_q flipped for lower-is-better metrics). Anchor collisions deduped, keeps the lower anchor.
- Ranked plot:
  - Subjects sorted **worst → best** along x by anchor-model performance regardless of metric direction.
  - X-tick labels = subject IDs (rotated 80°). Above each tick, a horizontal numeric annotation (`"7"`, `"15"`, `"50"`, …) gives the anchor-model performance percentile via `ax.get_xaxis_transform()`.
  - Markers only (`fmt="o"`); no connecting lines. Anchor model gets larger marker + higher alpha to differentiate.
  - Faint dotted y-grid (`alpha=0.12`, `linewidth=0.6`); `set_axisbelow(True)`.
  - Default `sparsify=100` (one subject per percentile); `figsize` scales with `n_subj` and caps at 28″. Tick density auto-scales with `n_subj` (token-based: `s+1` ≤20, `s` ≤60, `xs` ≤150, `xs-1` more).
  - Percentile-cutoff shading: `bottom_quantile=0.20` (red, leftmost N%) and optional `top_quantile` (green, rightmost N%). Both render at `zorder=0` so points sit on top.
  - **Dynamic bottom shading**: `dynamic_bottom_quantile=True` (default) walks subjects in performance-ascending order and finds the first index where the anchor model strictly beats `naive_baseline_model='naive'` (direction-aware: `>` for higher-better, `<` for RMSE). That index/n_subj **overrides** the static `bottom_quantile`. Falls back to static when no crossover exists or the baseline is missing.
  - Self-documenting legend: model markers + a labeled `Patch` for each shaded region (`"Bottom N% — PLAM ≤ Naive"` for dynamic, `"Bottom N% (\`bottom_quantile\`)"` for static).

**Subject specificity** (`compute_subject_specificity`, `plot_subject_specificity`).
- Per LORO sample: `sim_self = metric(pred[s,r], truth[s,r])`; `sim_other_mean = mean_{s' ≠ s} metric(pred[s,r], truth[s',r])`.
- Vectorized per region as a single `(n_p × G) @ (G × n_p)` `_pairwise_metric_matrix` call per `(model, region)`; off-diagonal mean is NaN-aware.
- Multi-metric (`pearson_r | spearman_r | r2 | rmse`); direction flips for RMSE; metric stored in result column.

**Spatial specificity** (`compute_spatial_specificity`, `plot_spatial_specificity`).
- Sibling of subject specificity; null is **prediction → truth of other regions within the same subject**:
  - `sim_self              = metric(pred[s,r,m], truth[s,r])`
  - `sim_other_region_mean = mean_{r' ≠ r} metric(pred[s,r,m], truth[s,r'])` — i.e. the **mean of `k − 1` per-region metrics** (mean-of-metrics, not metric-of-mean), where `k` is the number of regions covered for subject `s`.
- For a subject with 10 covered regions: 1 self-metric + 9 other-region metrics per LORO sample, then average those 9.
- Same `_pairwise_metric_matrix` primitive: build `M[i,j] = metric(pred[s, r_i], truth[s, r_j])`; diagonal = self, off-diagonal row mean = other-region mean.
- Distinguishes a model that copies a per-subject mean profile to every region (passes subject specificity, fails spatial specificity) from one that genuinely localizes (passes both).

**Shared split-violin plotter** (`_plot_specificity_split_violin`).
- Saturated/light shading per `MODEL_COLORS`; titles drop "higher = better"; stats absorbed into the lower-left legend (per-model "med Δ=…, p=…" rows on transparent handles); legend title: `"Comparison (Wilcoxon Δ = self − other)"`. No separate footer.
- Direction-aware Wilcoxon (`alternative='greater'` for higher-better, `'less'` for RMSE).
- Both subject and spatial specificity wrap it. Single source of truth for self-vs-null violins.

**Vectorized similarity primitives** (`_pairwise_metric_matrix`, `_row_z_normalize`, `_row_rank`, `_normalize_specificity_metric`, `_HIGHER_IS_BETTER`, `_hex_lighten`) live with their callers in this module — not duplicated in `eval_population.py`.

### Genewise per-gene metrics — `eval_population.py`

Per-gene predictability surface, computed across samples (each gene ranked once across the full LORO sample axis).

- **`compute_genewise_metrics(view, models, metrics, *, stratify_by=None, n_jobs=1, chunk_size=200)`** — long-form `(gene, model, metric, value, n_samples[, stratum])` table. `stratify_by='gtex_region'` adds a tissue facet by computing per-tissue per-gene metrics. Vectorized over genes; `chunk_size` bounds memory.
- **`plot_genewise_ranked_overlay(genewise_df, *, metric, group_by=None, anchor='dlam', palette_override=None, gene_label_step_pct=10, labeled_genes=None, ...)`** — direction-aware ranked curve per model (or per (model, group_by) when stratified). Anchors gene-rank x order to `anchor` model so all curves share the same gene ordering and a model's curve = "how this model scores genes in the order DLAM ranks them". `gene_label_step_pct=N` writes gene symbols every N percentile along the anchor curve; `labeled_genes` overrides with a hand-picked list. Tissue palette comes from `_tissue_palette(tissues, region_group_lookup, weights)` aligned with `_global_scatter_color_spec` (4-bucket cortex/subcortex/limbic_midbrain/cerebellum, frequency-weighted ordering).
- **`plot_genewise_histogram(genewise_df, *, metric, group_by=None, ...)`** — per-model (and optionally per-stratum) distribution of per-gene metrics; same palette routing.
- **`compute_gene_rank_lookup(view, *, model, metric)`** — convenience: returns a `gene → rank` dict from a single anchor model, callable once and threaded through `gene_rank_lookup` on `plot_global_prediction_scatter` so per-model loops reuse the precomputed ordering without re-loading other models' tables.

### Within-subject spatial Kendall-τ — `eval_population.py`

One Kendall-τ per `(subject, gene, model)` measuring agreement between predicted and true gene expression *across that subject's covered regions*. Probes spatial-ordering signal independent of subject mean.

- **`compute_within_subject_kendall(view, *, models, min_regions=5, cache=True, force_rebuild=False, cache_dir=None)`** — long-form `(subject_id, gene, model, kendall_tau, p_value, n_regions)` rows. Skips `(subject, gene)` pairs with fewer than `min_regions` covered regions. **Disk-cached**: SHA-256 hash key over the view's selected models + sample set + min_regions. Default cache dir: `notebooks/cache/genewise_kendall/<hash>.pkl`. Run once on **all genes**; sublist analyses filter inline rather than recomputing.
- **`summarize_kendall_per_unit(kendall_long, *, unit_col)`** — collapses the cube along the non-`unit_col` axis: `unit_col='gene'` aggregates across subjects (per-gene median); `unit_col='subject_id'` aggregates across genes (per-subject median). Returns `(unit_col, model, median, mean, n)` long-form.
- **`plot_unitwise_kendall_ranked(summary_df, *, unit_col, metric_col='median', anchor_model='naive', sparsify=100, bottom_quantile=None, top_quantile=None, ...)`** — ranked-curve plot mirroring `plot_subject_performance_ranked` but for Kendall-τ. Anchor-keyed sort, percentile annotations, optional shading. `bottom_quantile=None` is the default (no shading) for Kendall.
- **`filter_kendall_to_genes(kendall_long, genes)`** — inline sublist filter; pairs with `summarize_kendall_per_unit` so `compute_within_subject_kendall(..., all genes) → filter_kendall_to_genes(...) → summarize_kendall_per_unit(...) → plot_unitwise_kendall_ranked(...)` is the canonical sublist flow.

## Notebook structure (`eval_population_genewise.ipynb`)

Per-gene + within-subject spatial analyses. Mirrors the mini-config-per-section pattern.

1. Path setup + dev reload (`importlib.reload(eda_core)` included so the cached PREPOST helper picks up edits).
2. **Top config (global)**: `CFG`, `MODELS`, `N_JOBS`, `EVAL_GENE_LIST_PATH` defaults, `VIEW_SUBSET`. Defines a `build_panel_view(gene_list_path, models=None, **overrides)` closure that wraps `make_prediction_eval_view` with the notebook's defaults so each section can request a panel-specific view in one call.
3. PREPOST + global prediction tables (PREPOST goes through `prepare_pre_post_harmonization_cached`).
4. **Section 1 — Genewise ranked curves.**
    - Mini-config: `GENE_LIST_PATH='gtex_100hvg'` (default sublist), `GENEWISE_METRIC='pearson_r'`, `GENEWISE_RANKED_ANCHOR_MODEL='dlam'`, `GENE_LABEL_STEP_PCT`, `LABELED_GENES=None`.
    - `compute_genewise_metrics` → `plot_genewise_ranked_overlay` (overall) → tissue-stratified version with `stratify_by='gtex_region'`. Companion histogram via `plot_genewise_histogram`.
5. **Section 2 — Tissue-stratified genewise.** Reuses Section 1's `genewise_df` with `group_by='gtex_region'`.
6. **Section 3 — Gene-sublist scatter.**
    - Mini-config: `SUBLIST_RANK_MODEL='dlam'`, `pct_picks` (top/bottom/random percent), `HIGHLIGHT_GENES`.
    - `compute_gene_rank_lookup(view, model=SUBLIST_RANK_MODEL, metric=...)` once → looped per model passing `gene_rank_lookup=` into `plot_global_prediction_scatter` with `gene_selection=`/`highlight_genes=`/`isolate_color_subby=` knobs. Lookup precompute avoids the bug where a per-model view doesn't contain the rank-anchor model's pred table.
7. **Section 4 — Interactive single-gene scatter.**
    - Two `ipywidgets.Dropdown`s (Model + Gene with `'all'` option) wired via explicit `dd.observe(callback)` — *not* `interactive_output` (caused 4× re-renders on inline backend). Single `Output` widget cleared with `_out.clear_output(wait=True)` per change. Inline matplotlib (no `%matplotlib widget`).
8. **Section 5 — Within-subject spatial Kendall-τ.**
    - Mini-config: `KENDALL_GENE_LIST_PATH=None` (all genes — first run is cached, sublists filter inline), `KENDALL_ANCHOR_MODEL='naive'`, `KENDALL_BOTTOM_QUANTILE=None` (no shading), `KENDALL_MIN_REGIONS=5`.
    - `compute_within_subject_kendall(view, models=MODELS, min_regions=KENDALL_MIN_REGIONS)` (cached). Two ranked plots: per-gene (across-subject median) and per-subject (across-gene median).
    - **Sublist subsection**: `KENDALL_SUBLIST_PATH='gtex_100hvg'` → `filter_kendall_to_genes(kendall_long, genes)` → `summarize_kendall_per_unit` → `plot_unitwise_kendall_ranked`. No recompute.

## Notebook structure (`eval_population.ipynb`)

Top-down:
1. Path setup + dev reload (reloads `eval_style` and `eval_population`; explicit `from eval_style import *` so `model_label`/`font_size`/etc. are in scope).
2. **Single config cell** at the top: `CFG`, `MODELS`, gene list, `VIEW_SUBSET`, `DEFAULT_METRIC`, scatter/sandbox/bias/LORO knobs.
3. PREPOST + global prediction tables.
4. **Scatter sandbox** — free-form `plot_global_prediction_scatter`.
5. **Sample-Wise Stratification** — four explicit sections, each self-contained:
    - Global → scatter, box, table.
    - Sex → scatter, box, table; opt-in `SHOW_BIAS_FOREST` adds the LMM-controlled forest panel.
    - Age → scatter, box, table.
    - Region (`gtex_region`) → scatter, box, table.
6. Coverage-vs-performance.
7. **LORO fold-combo curves** — unified `plot_fold_combo_ranked_overlay` (line / dist-colored gradient with reference-model colorbar / running-mean overlay) + matched-overlay companion. No separate per-model loop.
8. **Fold-difficulty decomposition** — `plot_fold_std_vs_mean` (n<5 filter, marker-area scaling, full-set cutoff).
9. Distance-to-training vs metric (lines + thick errorbars, no bars/histogram).

Subject specificity and the per-subject ranked-mean diagnostic moved out to `eval_singlesubject.ipynb`.

## Notebook structure (`eval_singlesubject.ipynb`)

Top-down. Per-section knobs live in **mini-config cells immediately above their use sites** so in-notebook iteration is fluid; only truly global config sits at the top.

1. Path setup + dev reload (reloads all three eval modules; explicit `from eval_style import *`).
2. **Top config (global)**: `CFG`, `MODELS`, `N_JOBS`, cache flags, `EVAL_GENE_LIST_PATH`, `VIEW_SUBSET`, `DEFAULT_METRIC`.
3. PREPOST + global prediction tables + `subject_view`.
4. **Section 1 — Per-subject performance.**
    - Section 1 mini-config: `FOLD_COMBO_COVERAGE_MIN/MAX`, `PERFORMANCE_FOLD_METRIC`, `ANCHOR_MODEL`, `SPARSIFY=100`, `BOTTOM_QUANTILE`, `DYNAMIC_BOTTOM_QUANTILE`, `TOP_QUANTILE`.
    - LORO fold-combo cache load → `compute_subject_performance` → `select_subjects_by_percentile` (returned as `band_subjects`) → `plot_subject_performance_ranked` (markers-only, percentile annotations above axis, dynamic shading).
    - **Illustrative subjects subsection**: mini-config `ILLUSTRATIVE_PERCENTILES=[10,50,90]`, `ILLUSTRATIVE_COLOR_BY='region'`, `ILLUSTRATIVE_FIGSIZE`. Picks subjects nearest to those percentiles from `band_subjects` (or computes inline), then renders **3 single-panel scatters** by passing `models=[ANCHOR_MODEL]` + `subjects=[subj]` through `make_prediction_eval_view` + `plot_global_prediction_scatter`. No new helper — composes existing primitives.
5. **Section 2 — Subject specificity.**
    - Mini-config: `SPECIFICITY_REGION_COL`, `SHOW_SPATIAL_SPECIFICITY`.
    - `compute_subject_specificity` → median-table + `plot_subject_specificity`.
6. **Section 3 — Spatial specificity.**
    - `compute_spatial_specificity` → median-table + `plot_spatial_specificity`. Markdown explicitly contrasts the two specificity axes (right subject? right region within subject?).

## Module layout (current)

### `eval_style.py`
Shared plotting / naming / fonts / metric formatting / tick enforcement.
- Model order, labels, colors; region/parcel color conventions.
- Token system: `FONT_TOKENS`, `font_size`, `set_font_scale`, `_resolve_fonts`. `apply_tick_style(ax)` helper.
- Canonical metric labels: `METRIC_LABELS`, `METRIC_LABELS_MEAN`, `format_metric_label`, `format_legend_label` delegating.
- Legacy `FONT` dict still exported for unmigrated callers.

### `eda_core.py`
Dataset-level EDA + PREPOST reconstruction.
- `EDAConfig`, path & gene-list resolution, PREPOST reconstruction, demographic/coverage summaries, global matrix views, pre/post harmonization diagnostics.
- `prepare_pre_post_harmonization_cached(cfg, *, force_rebuild=False, cache_dir=None)` — disk-cached PREPOST keyed on EDAConfig fields + input file mtime/size; default cache `notebooks/cache/prepost/<hash>.pkl`.

### `eval_population.py`
Population-level cached prediction evaluation. Stable surfaces:
- Cache: `build_global_prediction_tables`, parquet read/write with permission repair.
- Views/metrics: `make_prediction_eval_view`, `compute_prediction_metrics`.
- Scatters: `plot_global_prediction_scatter`, `plot_stratified_scatter`.
- Distributions: `plot_stratified_distribution` (box default, violin opt-in), `plot_metric_violins` (legacy one-axis case), `plot_metric_delta_violins`.
- Tables: `format_stratified_metric_table` (with `_gtex_region_group_order`).
- Coverage / fold-combo / distance: `plot_coverage_vs_metric`, `plot_fold_combo_ranked_overlay` (unified line / dist-gradient / running-mean), `plot_fold_combo_matched_overlay`, `plot_fold_std_vs_mean` (fold-difficulty decomposition), `plot_distance_to_train_vs_metric`.
- Bias analysis: `compute_stratum_bias`, `plot_stratum_bias_forest`.
- Genewise: `compute_genewise_metrics`, `compute_gene_rank_lookup`, `plot_genewise_ranked_overlay`, `plot_genewise_histogram`, `_tissue_palette` (aligned with `_global_scatter_color_spec`).
- Within-subject Kendall-τ: `compute_within_subject_kendall` (disk-cached), `summarize_kendall_per_unit`, `filter_kendall_to_genes`, `plot_unitwise_kendall_ranked`.
- Scatter knobs (extended on `plot_global_prediction_scatter`): `gene_selection`, `gene_rank_metric`, `gene_rank_model`, `gene_rank_lookup`, `highlight_genes`, `base_alpha_factor`, `isolate_stratum`, `isolate_color_subby`, `isolate_color_subby_top_n`, `fit_line_color`, `annotate_metric_in_legend`, `annotation_metric`, `annotation_model`.
- Re-exports from `results_eda.py`: `compute_fold_combo_metrics_from_cache`, `plot_fold_combo_ranked` (per-model with `points_style` colorbar) — kept for legacy callers, retired from `eval_population.ipynb`.

### `eval_single_subject.py`
Subject-keyed evaluation surfaces. Stable surfaces:
- Subject performance: `compute_subject_performance`, `select_subjects_by_percentile`, `plot_subject_performance_ranked` (with `dynamic_bottom_quantile`, direction-aware sort, percentile-anchor annotations).
- Subject specificity: `compute_subject_specificity`, `plot_subject_specificity`.
- Spatial specificity: `compute_spatial_specificity`, `plot_spatial_specificity` (truth-of-other-regions null, mean-of-metrics not metric-of-mean).

### `eval_latent.py` *(pending)*
Planned: pooled PCA fit/eval, reconstruction/recovery, variance spectra, PLS fit + reconstruction, latent-score comparison plots. Should consume the canonical wide prediction tables.

### `results_eda.py`
Compatibility/reference. Migrate stable logic out into focused `eval_*` modules; keep re-exports until notebooks fully decouple. Houses unmigrated heavyweight single-subject visualizations (matrix panel, alignment scatter, performance triplet) used by `results_single_subject_predictions.ipynb`.

## Notebook migration plan

### Done
- `results_cached_predictions.ipynb` split into `eval_data.ipynb` + `eval_population.ipynb`.
- Dataset demographics + global matrix/heatmap flow in `eval_data.ipynb`.
- Canonical wide tables, scatters, sample-wise metrics, box plots (replaced violins), region-group ordering for `gtex_region`.
- Gene-list path subsetting in views, scatters, metrics.
- Per-section stratified blocks (Global / Sex / Age / Region) in `eval_population.ipynb`.
- Sex bias forest (LMM with subject random intercept; controls for age, coverage, region; FDR; df=1 omnibus fallback; LMM-fallback warning).
- Unified `plot_fold_combo_ranked_overlay` with line + per-model gradient cmaps + reference-model colorbar + rolling-mean overlay; the older per-model `plot_fold_combo_ranked` loop is retired from the population notebook.
- Fold-difficulty decomposition (`plot_fold_std_vs_mean`, `min_subjects=5`, marker-area ∝ `n_subjects`, full-set cutoff) is the population notebook's LORO closing panel.
- Distance-to-training plot (line + thick errorbars + auto-zoom).
- `eval_single_subject.py` bootstrapped with subject performance + specificity + spatial specificity. Subject specificity and the per-subject ranked-mean diagnostic moved out of `eval_population.ipynb`.
- Direction-aware percentile semantics (0 = worst, 100 = best) across `compute_subject_performance` / `select_subjects_by_percentile` / `plot_subject_performance_ranked` so RMSE behaves identically to higher-better metrics.
- `dynamic_bottom_quantile` on the ranked plot — auto-shades up to the first anchor>naive crossover; legend self-documents the source.
- Subject specificity / spatial specificity violins reformatted: stats absorbed into lower-left legend, "higher = better" suffix dropped, footer text removed. Spatial specificity null corrected to truth-of-other-regions mean-of-metrics.
- Token-based font system in `eval_style.py`; hero plotters expose `font_sizes={...}` overrides; `set_font_scale(scale)` rescales everything globally.
- Tick visibility enforced via `_TICK_RC` injected into seaborn theme + module-import rcParams update + `apply_tick_style(ax)` per-axes safety; both notebooks' reload cells now `from eval_style import *` to pull `model_label`/`font_size`/etc. into scope.
- Canonical metric formatting routed through `format_metric_label` / `format_legend_label`; persistent feedback memory documents the rule.
- Per-section mini-config cells in `eval_singlesubject.ipynb` replace the single top config block — knobs live above their use sites.
- Illustrative-subject scatters at p10/50/90 implemented inline (no new helper) by passing `models=[ANCHOR_MODEL]` + `subjects=[subj]` through `make_prediction_eval_view` + `plot_global_prediction_scatter`.
- Unified `select_subjects_by_percentile(perf_df, *, metric, model, n_bands=..., percentiles=...)` replaces the older sample-wise `select_subjects_by_metric_percentile`.
- **Genewise migration** into `eval_population.py`: `compute_genewise_metrics` (overall + tissue-stratified), `plot_genewise_ranked_overlay` (anchor-keyed gene order, gene-symbol labels every Nth percentile or hand-picked), `plot_genewise_histogram`. Drives `eval_population_genewise.ipynb` Sections 1–2.
- **Tissue palette unification**: `_tissue_palette` now uses the 4-bucket cortex/subcortex/limbic_midbrain/cerebellum convention and frequency-weighted ordering from `_global_scatter_color_spec`, replacing the old 5-group fallback-to-gray scheme.
- **Gene-sublist scatter knobs** on `plot_global_prediction_scatter`: top/bottom/random/list-order selection (`gene_selection`), `gene_rank_lookup` precompute path so per-model loops don't need every model's pred table, hand-picked `highlight_genes`, `isolate_stratum` + `isolate_color_subby` for sub-frame focus, `annotate_metric_in_legend` for inline metric annotations.
- **Within-subject spatial Kendall-τ**: `compute_within_subject_kendall` (cube of `(subject, gene, model)` τ values, disk-cached), `summarize_kendall_per_unit` (per-gene-across-subjects or per-subject-across-genes median), `filter_kendall_to_genes`, `plot_unitwise_kendall_ranked`. `kendall_tau` registered in `METRIC_LABELS`.
- **Cached PREPOST**: `prepare_pre_post_harmonization_cached` with mtime+size+config hash invalidation; reload-cell pattern (`importlib.reload(eda_core)`) added to all eval notebooks.
- **Cache layout shift**: `out/` reserved for true prediction artifacts; helper-level caches (PREPOST, genewise Kendall) moved to `notebooks/cache/<name>/<hash>.pkl`.
- **ipywidgets-based interactive single-gene scatter**: explicit `Dropdown.observe()` + manual `Output.clear_output(wait=True)` after `interactive_output` caused multi-render artifacts; inline matplotlib (not `%matplotlib widget`).
- **Helper consolidation**: `build_panel_view(gene_list_path, models=None, **overrides)` closure in the genewise notebook keeps cells terse; `compute_gene_rank_lookup` and `filter_kendall_to_genes` extracted from inline notebook code into module helpers.
- **Sublist-by-filter pattern for Kendall**: compute once on all genes (cached), filter inline for any sublist; no recompute path.
- **NFSv4 ACL fix**: diagnosed and repaired across `/scratch/asr655` trees (Conn2Conn, GeneEx2Conn, Seq2GeneEx) — root cause was a collaborator share added with `fd` flag without a matching `OWNER@:fdg` inherit ACE, so new dirs landed at mode 0o000. Repair recipe + persistent memory documented.
- **Matching-policy auto-detect**: `EDAConfig.matching_policy=None` triggers `resolve_matching_policy(cfg)` to read the policy from any subject JSON under `<cache_root>/<gene_scope>/<model>/` (top-level *or* nested `config.matching_policy`); falls back to `'centroids'` for legacy caches. PREPOST applies the resolved policy so its `parcel_idx` values agree with the LORO cache's; the PREPOST cache hash includes the resolved policy so different caches don't collide.
- **Eligibility ordering invariant**: `build_subject_eligibility` runs *before* `apply_gtex_ahba_matching_policy` in both `loro_cache.load_dataset` and `_load_expression`. Without this, `centroids_and_volumes` silently drops subjects that had `brain - cerebellum` + `brain - cerebellar hemisphere` (post-collapse 4 distinct parcels < 5 threshold). Fix recovered 5 subjects on the new cache; matching PREPOST/cache-builder ordering also closed a long-standing 318-vs-313 eligibility discrepancy. Sbatch array bumped to `1-318%64` to re-process the recovered subjects.

### Next
- Migrate heavyweight single-subject visualizations from `results_eda.py` into `eval_single_subject.py` (`plot_subject_model_matrix_panel`, `plot_subject_alignment_scatter_panel`, `plot_single_subject_scatter_triplet`, `select_representative_subject_by_gtex_median`, `resolve_publication_subject`). Retire `results_single_subject_predictions.ipynb`.
- Wire bias forest into the Age section once the Sex pattern is settled in writing.
- Decide whether `compute_fold_combo_metrics_from_cache` and `plot_fold_combo_ranked` get formally rewritten into `eval_population.py` (or a future `eval_cache.py`) or stay re-exported from `results_eda.py`.
- Begin `eval_latent.py` (PCA → PLS migration from `pca_cached_predictions.ipynb`).
- Continue migrating remaining hero plots to the font-token API as they are touched.
- **Principled gene-sublist extraction (HVGs / DEGs)**. Today gene lists live as bare `.txt` files under `data/raw/gene_lists/` and `out/raw/gene_lists/` resolved by `eval_gene_list_path`; provenance (which dataset, which selection method, which parameters, which date) isn't tracked. Plan: add a builder module under `src/data_build/` (or analogue) that produces gene sublists deterministically from the canonical CSV + AHBA / GTEx counts — HVGs via mean-variance trend (configurable n / dispersion threshold), DEGs via per-tissue contrasts (e.g., cerebellum vs cortex, with FDR cut). Each produced file should carry a sidecar manifest (sha256, source CSV mtime, method, params, n_genes) so downstream eval consumers can verify what they're loading. Surfaces consumed: `EDAConfig.hvg_path`, `eval_gene_list_path`, the genewise notebook's sublist defaults.
- **`samples.csv` builder for end-to-end repro**. The current `data/raw/gxp_samples.csv` is hand-assembled from upstream AHBA + GTEx downloads + coordinate parsing. Goal: a documented build pipeline (script entrypoint, ideally an sbatch-backed CLI) that goes raw downloads → harmonized CSV with frozen schema (`subject, age, sex, dataset, tissue_or_parcel, coordinates, <gene cols>`), so any contributor can reproduce the canonical input. Should integrate with the gene-sublist builder above (CSV produces gene header → HVG/DEG selection runs against it). This unblocks: ablating gene scope, swapping atlases, regenerating after upstream releases.

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
- Subject-performance ranked plot is anchored to one model so the same subject occupies the same x position under every model panel; sparsification picks subjects at evenly-spaced **performance** percentiles of that anchor (direction-aware: 0 = worst, 100 = best).
- Spatial specificity null → **mean of `k − 1` per-region metrics** comparing the prediction to each *other* region's truth in the same subject (mean-of-metrics, not metric-of-mean; truth-vs-truth, not pred-vs-pred). Tests "are predictions spatially differentiated within a brain?" without conflating with how spatially differentiated GTEx truth happens to be.
- Subject-performance plot styling → markers only (no connecting lines), subject-ID ticks rotated below axis, horizontal numeric percentile annotations above axis, faint dotted y-grid, percentile-cutoff shading via `bottom_quantile` / `top_quantile` Patches in the legend.
- LORO fold-combo plotting unified → one `plot_fold_combo_ranked_overlay` with `points_style='line' | 'dist_to_nearest_train' | 'dist_to_centroid_train'`. Per-model gradient cmaps; colorbar shown only for the reference model; running-mean overlay is opt-in.
- Fold-difficulty decomposition → require `n_subjects >= 5` (n<5 std is sampling noise); marker area ∝ `n_subjects`; cutoff computed on the full pre-filter set so it agrees with the LORO ranked overlay's bottom tail.
- Illustrative-subject scatters → no bespoke helper; compose `make_prediction_eval_view(models=[ANCHOR_MODEL], subjects=[subj])` + `plot_global_prediction_scatter`. Three figures, each one panel.
- Notebook config layout → only truly global knobs at the top; per-section knobs in mini-config cells immediately above their use sites for fluid iteration.
- Font sizes → token-based system (`xs/s/m/l/xl/xxl`, `±N` offsets, `set_font_scale` global multiplier). Hero plots accept `font_sizes={...}` overrides; muscle-memory element keys (`title, xlabel, ylabel, tick, legend, legend_title, annotation, footer, cbar_label, cbar_tick`).
- Metric labels → always route through `format_metric_label` / `format_legend_label`; never hardcode "Pearson R" / "R^2" / "rho" in plot text. Persistent feedback memory documents the rule.
- Tick visibility → enforce via rcParams injected through `sns.set_theme(rc=...)` + module-import update + `apply_tick_style` per-axes safety. Re-asserted in hero plotters.
- Eval-table cache is keyed on `cfg.cache_root.name` → `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/`. Different LORO caches (`loro_subject_cache`, `loro_subject_cache_coord_vol`, ...) get isolated subtrees, so iterating model variants only requires changing `cache_root` (rebuild if the matching subtree doesn't exist yet).
- Cache layout → `out/` holds **true prediction artifacts only** (`build_global_prediction_tables`, LORO fold-combo caches). Helper-level caches (PREPOST reconstruction, genewise Kendall) live under `notebooks/cache/<name>/<hash>.pkl`. Safe to delete and recompute; never confused with prediction outputs.
- Within-subject Kendall-τ aggregation default → **median**, not mean. Per-unit summary stores both but the ranked plot defaults to `metric_col='median'`. Bottom-quantile shading off by default for Kendall (signal too noisy for naïve crossover semantics).
- Kendall sublist analyses → compute once on **all genes** (cached), filter inline via `filter_kendall_to_genes`. No recompute path; the cache is the source of truth.
- Gene-sublist scatter rank lookup → precompute via `compute_gene_rank_lookup(view, model=..., metric=...)` once and pass `gene_rank_lookup=` into `plot_global_prediction_scatter` per model. Avoids the bug where a per-model view doesn't contain the rank-anchor model's pred table.
- Tissue palette → aligned to `_global_scatter_color_spec`'s 4-bucket convention (cortex, subcortex, limbic_midbrain, cerebellum) with frequency-weighted region ordering; no fallback-to-gray for basal_ganglia/limbic_midbrain.
- Notebook-helper closures → `build_panel_view(gene_list_path, models=None, **overrides)`-style closures live in the notebook (capture `view_root`, `MODELS`, `CFG`) rather than as module helpers; module helpers stay config-agnostic.
- Interactive widgets → explicit `Dropdown.observe(callback)` + manual `Output.clear_output(wait=True)` per change. **Do not** use `interactive_output` on the inline backend (causes 4× re-renders); **do not** require `%matplotlib widget` (extra ipympl dependency, layout fragility).
- NFSv4 ACL convention on `/scratch/asr655` → every collaborator share must add a matching `A:fdg:OWNER@:rwaDxtTnNcy` inherit ACE alongside the group ACE; otherwise new dirs/files land at mode 0o000. Defensive `chmod` after `mkdir` in helpers stays as a safety net even after the tree-wide ACL repair.
- Matching policy resolution → `EDAConfig.matching_policy=None` ⇒ auto-detect from cache subject JSONs (top-level or nested `config.matching_policy`), fallback `'centroids'` for legacy caches. Notebooks change `cache_root` only; never hardcode policy in the notebook config. PREPOST disk-cache hash includes the resolved policy so distinct caches don't collide.
- Subject eligibility is computed on the centroid-mapped `parcel_idx` *before* `apply_gtex_ahba_matching_policy`. Eligibility = "subject sampled ≥`min_observed_parcels` distinct anatomical regions"; it must not depend on cerebellar/cortical bucketing applied downstream. Both `loro_cache.load_dataset` and `_load_expression` must keep this ordering or PREPOST and the cache builder will diverge silently.

## Open decisions
- Whether to add `eval_cache.py` for low-level `.npz` reading shared by population, single-subject, and latent modules.
- Whether `compute_prediction_metrics`'s `unit='global_flat'` stays exposed or gets gated behind an advanced flag.
- Whether parquet remains the default table-cache format on this filesystem or whether a fallback is needed for permission/quota issues.
- How much of `results_eda.py` should remain as a permanent public facade after notebooks fully migrate.
- Whether `compute_stratum_bias` should be re-fit with subject + region random effects when 13 region levels feel too df-heavy.
- Whether to migrate the remaining (non-hero) plotters to the font-token system in a single sweep, or keep doing it opportunistically as plots are touched.
- Where the gene-sublist builder lives — new `src/data_build/gene_lists.py`, or a subpackage under `src/eval_utils/`? Whether outputs land in `data/raw/gene_lists/` (committed) or `out/gene_lists/` (regenerable). Whether sidecar manifests are JSON next-to-file or a single registry index.
- Scope of the `samples.csv` builder — does it own raw download/caching or assume upstream files are present? Whether it lives under `scripts/` (CLI entrypoint) or `src/data_build/` (importable). Whether the schema is locked in `EDAConfig` or a separate dataclass that EDAConfig references.
