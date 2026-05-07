# AHBA-GTEx Latent Alignment and Imputation

This repository contains the AHBA<->GTEx atlas-alignment workflow used for manuscript assets and LORO benchmarking across three model families:
- naive atlas-fill baseline
- DLAM (deterministic latent alignment)
- PLAM (probabilistic latent alignment)

## Current Entry Points

Primary write-up pipeline:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

Active analysis notebooks (eval refactor):
- `eval_data.ipynb` — dataset-level EDA: PREPOST, demographics, coverage, global matrices, harmonization heatmaps.
- `eval_population.ipynb` — population-level evaluation over cached LORO predictions: global scatters, sample-wise stratifications (Global / Sex / Age / Region), sex-bias forest, LORO fold-combo overlays + dist-colored gradients, fold-difficulty decomposition, distance-to-training.
- `eval_singlesubject.ipynb` — subject-keyed analyses: per-subject performance distribution with percentile-anchored ranked plot (dynamic-shading at anchor>naive crossover), illustrative single-model scatters at p10/50/90, subject specificity (self vs other-subject truth at same region), spatial specificity (self vs mean of `k − 1` per-region metrics within subject).
- `eval_population_genewise.ipynb` — per-gene analyses: gene-wise ranked + histogram overall and tissue-stratified; gene-sublist 3-model scatters with rank-based selection (`top` / `bottom` / `random` / `list_order`) or hand-picked highlights; ipywidgets dropdown for single-gene scatter (model + gene); within-subject spatial Kendall-τ with all-genes default + cached sublist filter.
- `notebooks/coordinate_overlay_3d_mni.ipynb` — GTEx/AHBA spatial assignment inspection.

Legacy / reference notebooks at repo root: `results_cached_predictions.ipynb`, `results_single_subject_predictions.ipynb`, `pca_cached_predictions.ipynb`. Older exploratory variants live under `notebooks/`.

## Recent Workflow Additions (High Level)

- Subject-wise LORO caches under `out/loro_subject_cache/` enable cache-backed analysis without re-running fits.
- Population eval uses canonical wide all-gene truth/prediction tables under `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/` (subdir keyed on `cfg.cache_root.name` so different LORO caches stay isolated — change `cache_root` and the eval tables route to the matching subtree, rebuild if it doesn't exist yet). Gene-list analyses subset downstream via `eval_gene_list_path`.
- Sample-wise metrics are the default unit: per `subject_region_key`, score across genes, then aggregate. Spearman is opt-in (ranking-heavy).
- Sex-bias forest in `eval_population.py` (Fisher-z(metric) ~ axis + controls; LMM with subject random intercept, OLS+cluster fallback; BH-FDR; df=1 omnibus fallback).
- Unified LORO fold-combo plot: `plot_fold_combo_ranked_overlay` supports `points_style='line' | 'dist_to_nearest_train' | 'dist_to_centroid_train'` with per-model gradient cmaps and an optional rolling-mean overlay.
- Fold-difficulty decomposition (`plot_fold_std_vs_mean`) excludes `n<5` fold-combos, scales marker area to `n_subjects`, draws cutoffs computed on the full pre-filter set.
- `src/eval_utils/eval_single_subject.py` is the canonical home for subject-keyed analyses (perf summary, percentile selector, ranked plot with `dynamic_bottom_quantile`, subject specificity, spatial specificity).
- **Genewise surface in `eval_population.py`**: `compute_genewise_metrics(view, ..., stratify_by=None)` (per-gene Pearson r / R² / RMSE with optional tissue stratification), `plot_genewise_ranked_overlay` (group-by model or tissue, gene labels on the curve, per-line n_samples in legend), `plot_genewise_histogram`. Tissue palette matches the existing region scatter via the shared `_tissue_palette`.
- **Within-subject spatial Kendall-τ** in `eval_population.py`: `compute_within_subject_kendall(view, ..., min_regions=5, cache=True)` produces a long `(subject, gene, model, kendall_tau, n_regions)` table (disk-cached); `summarize_kendall_per_unit(unit_col='gene'|'subject')` aggregates with median/mean/std/n_valid; `plot_unitwise_kendall_ranked` is the generalized ranked plot for either x-axis. `filter_kendall_to_genes(kendall_long, genes)` reuses an all-genes pickle for sublist analyses with no recompute.
- **Gene-sublist scatter knobs** on `plot_global_prediction_scatter`: `gene_selection='top'|'bottom'|'random'|'list_order'` with `gene_rank_lookup` precomputed once via `compute_gene_rank_lookup(view, model, metric)`; `highlight_genes=[…]` hand-pick override; `base_alpha_factor` for fading "Other" points; `isolate_stratum` + `isolate_color_subby` for side-by-side panels with sub-gradient coloring.
- **Disk-cached PREPOST** via `prepare_pre_post_harmonization_cached(cfg)` in `eda_core.py` — first call computes (slow ComBat fit), subsequent calls load in seconds. Hash key covers EDAConfig fields + CSV/HVG file mtime+size.
- **Cache layout convention**: `out/` holds true prediction artifacts only (`loro_subject_cache/`, `eval_prediction_tables/`, `slurm/`, etc.). Helper-level caches live under `notebooks/cache/{prepost,genewise_kendall}/<hash>.pkl` — safe to delete and recompute.
- **ipywidgets-based interactive single-gene scatter** in Section 4 of `eval_population_genewise.ipynb` — explicit dropdowns + `observe` callbacks (no `%matplotlib widget` / `ipympl` dependency required); inline static figures with one render path.
- Token-based font system (`FONT_TOKENS = {xs, s, m, l, xl, xxl}`, `font_size("m+1")`, `set_font_scale(1.2)`); hero plotters expose `font_sizes={...}` overrides for per-call surgery without leaving the token grammar.
- Canonical metric label formatting (`Pearson r`, `Spearman ρ`, `R²`, `RMSE`, `Kendall τ`) routed through `eval_style.format_metric_label` / `format_legend_label`. Never hardcode metric strings in plot text.
- Tick visibility enforced globally via the rcParams updates in `set_academic_style` and the `apply_tick_style(ax)` per-axes safety helper.
- Per-section mini-config cells: only global config (CFG, MODELS, gene list, view subset, default metric) sits at the top of each eval notebook; section-local knobs live immediately above their use sites for fluid in-notebook iteration. The genewise notebook adds a `build_panel_view(gene_list_path, models=None)` helper closure to consolidate `make_prediction_eval_view` boilerplate.
- DLAM diagnostics utilities live in `src/eval_utils/dlam_diagnostics.py`.
- **GTEx<->AHBA matching policy auto-detected from the cache.** `EDAConfig.matching_policy` defaults to `None`; `resolve_matching_policy(cfg)` reads the field from any subject JSON in `<cache_root>/<gene_scope>/<model>/` (top-level or nested under `config.matching_policy`) and falls back to `'centroids'` for legacy caches. PREPOST applies the resolved policy so its parcel_idx values agree with the cache's. Switch `cfg.cache_root` between `loro_subject_cache` and `loro_subject_cache_coord_vol` and the right policy is applied automatically.
- **Subject eligibility is computed *before* the matching policy** in both `loro_cache.load_dataset` and `_load_expression`. This keeps eligibility a property of the underlying data (distinct centroid-mapped parcels per subject, ≥`min_observed_parcels`) rather than an artefact of the cerebellum/cortex bucketing applied downstream — `centroids_and_volumes` no longer drops subjects that had cerebellum + cerebellar hemisphere by collapsing them to one parcel before the eligibility check.

## Repository Layout

- `src/`: reusable preprocessing, harmonization, models, workflows, evaluation utilities, visualization
- `scripts/`: manuscript/workflow CLIs and cache-generation entrypoints
- `scripts/sbatch/`: Slurm launchers for cache experiments
- `notebooks/`: write-up and supporting analysis notebooks
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
- harmonized-space evaluation is the default
- raw-space follow-up uses mixed-space correlation against `loro_truth_subject_raw` in legacy `results_*.ipynb`

Core implementation modules:
- `src/workflows/loro_cache.py`
- `scripts/run_loro_cache_batch.py`
- `scripts/sbatch/run_loro_cache_single_subject.sbatch`
- `scripts/sbatch/run_loro_cache_array.sbatch`
- `src/eval_utils/eda_core.py`
- `src/eval_utils/eval_population.py`
- `src/eval_utils/eval_single_subject.py`
- `src/eval_utils/eval_style.py`
- `src/eval_utils/results_eda.py` (compatibility/reference)
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
- The formal manuscript-facing matching policy is documented in `docs/manuscript/parcel_matching.md`, with exploratory diagnostics in `parcel_assignment.ipynb`. That policy keeps default centroid matching for manually validatable subcortical structures, adds Brodmann/Schaefer voxel-overlap overrides for BA9, BA24, and BA110 cortical labels, and defines cerebellar duplicate handling with assignment to `Cerebellar_Region7`.

## Quick Commands

Single subject (local):

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --use-cache true
```

Median aggregation variant:

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes --atlas-agg median
```

Default fitting behavior:

```bash
python3 scripts/run_loro_cache_batch.py --subject GTEX-14ASI --model all --gene-scope allgenes
```

Override example (legacy spatial settings):

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
- Keep new analysis work in the active `eval_*` notebooks at the repo root; treat `notebooks/` and the `results_*.ipynb` notebooks at root as historical unless explicitly reviving one. `notebooks/cache/` is an active helper-cache location and is not historical.
- `EDAConfig` (in `eda_core.py`) supports model-directory overrides (`naive_cache_dirname`, `dlam_cache_dirname`, `plam_cache_dirname`) for side-by-side rank experiments (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`).
- New eval-refactor work should target `src/eval_utils/eda_core.py`, `src/eval_utils/eval_population.py`, `src/eval_utils/eval_single_subject.py`, and the future `eval_latent.py`. Treat `results_eda.py` as compatibility/reference during migration; do not add new functionality there.
- Always route metric labels through `eval_style.format_metric_label` (or `format_legend_label`); never hardcode `Pearson R` / `R^2` / `rho` / `tau` strings in plots or tables.
- Use the font-token system (`font_size`, `_DEFAULT_FONT_SIZES`, `font_sizes={...}` per-call overrides, `set_font_scale`) when adding or tuning hero plots — single global scale knob.
- Helper-level caches (PREPOST, genewise Kendall) belong under `notebooks/cache/<name>/<hash>.pkl`. `out/` is reserved for true prediction artifacts.
- When adding a new collaborator share via `nfs4_setfacl` on `/scratch/asr655/...`, always re-assert the inheritable OWNER@ ACE on the same tree, otherwise newly-created files land with mode `0o000` (see CONTEXT.md → "NFSv4 ACL gotcha").
- Do not edit `src/eval_utils/results_eda_arxiv.py`; it is a backup snapshot.
- See `CONTEXT.md` for a fast onboarding summary intended for parallel agents, and `context_packages/results_eda_refactor_plan.md` for the live refactor plan.

Last updated at: 2026-05-06
