# AHBA-GTEx Latent Alignment and Imputation

This repository contains the AHBA<->GTEx atlas-alignment workflow used for manuscript assets and LORO benchmarking across three model families:
- naive atlas-fill baseline
- DLAM (deterministic latent alignment)
- PLAM (probabilistic latent alignment)

## Current Entry Points

Primary write-up pipeline:
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb`

Active analysis notebooks live under `notebooks/eval/` and `notebooks/preprocessing/`:

- **Dataset-level EDA** (`notebooks/preprocessing/eval_data.ipynb`) — PREPOST, demographics, coverage, global matrices, harmonization heatmaps.
- **Model evaluation** (`notebooks/eval/results/`, seven notebooks; per-section CFG + ToC pattern):
  - `eval_population.ipynb` — population evaluation over cached LORO predictions: global scatters, sample-wise stratifications (Global / Sex / Age), sex-bias forest, LORO fold-combo overlays + dist-colored gradients, fold-difficulty decomposition, distance-to-training.
  - `eval_population_genewise.ipynb` — per-gene analyses: gene-wise ranked + histogram overall and tissue-stratified; gene-sublist 3-model scatters; ipywidgets dropdown single-gene scatter.
  - `eval_population_kendall.ipynb` — within-subject spatial Kendall-τ with all-genes default + cached sublist filter (gene-wise and subject-wise ranked).
  - `eval_population_subjectwise.ipynb` — subject-keyed analyses: per-subject performance distribution with percentile-anchored ranked plot, four illustrative-subject views (region / gene-stratified / focus-gene-isolated / focus-gene-by-region), subject specificity, spatial specificity.
  - `eval_regional.ipynb` — region-stratified performance + DLAM gene-highlighted region fan.
  - `eval_distributional.ipynb` — distributional comparison on matched held-out parcels and on fullfit-imputed parcels (`src/eval_utils/eval_distributional.py`).
  - `eval_pca.ipynb` — pooled PCA recovery: standard, within-parcel demeaned, within-parcel variance effects, sample×gene heatmaps, parcel scatters (`src/eval_utils/eval_latent.py`).
- **Embedding diagnostics** (`notebooks/eval/embeddings/`) — six cov/nocov variants for GTEx-only and cross-dataset.
- **Gradients / on-brain** (`notebooks/eval/gradients/`) — DLAM alignment, PLS gradient progression, PLS-gradients-on-brain. Plus `notebooks/eval/eval_gxp_onbrain_views.ipynb` at the eval root.
- **Tensor visualizers** (`notebooks/eval/tensors/`, unified `TensorView`/`JointTensorView` surface in `src/eval_utils/eval_samples.py`; see `context_packages/markdowns/samples_visualizer.md`):
  - `eval_gxp_samples.ipynb` — raw GTEx + AHBA standalone then joint (CSV via `build_native_tensor_view`).
  - `eval_gxp_samples_combat.ipynb` — ComBat joint, pre (`raw_matched`) and post (`harmonized`), from PREPOST cubes.
  - `eval_gxp_samples_pipeline.ipynb` — end-to-end pipeline view.
  - `eval_gxp_samples_predictions_joint.ipynb` + `eval_gxp_samples_predictions_joint_alpha.ipynb` — LORO + full-fit (`loro_truth` / `loro_recon` / `loro_fused` / `fullfit`) joint-vs-AHBA on the superset axis.
  - `eval_gxp_onbrain_view_prep.ipynb` — prep for on-brain renders.
- **Variant / sweep notebooks at the repo root** read from per-variant LORO cache roots:
  - DLAM `t_prior_residual` kernel sweep: `eval_population_genewise_tprior_{gp,imq,tps}.ipynb`, `eval_population_subjectwise_tprior_imq.ipynb`, `eval_population_tprior_imq.ipynb`.
  - Legacy spatial-interpolation sweep: `eval_population_genewise_{anchor,gp,rbf,rbf_shrink}.ipynb`.
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
- **Disk-cached PREPOST** via `prepare_pre_post_harmonization_cached(cfg)` in `eda_core.py` — first call computes (slow ComBat fit), subsequent calls load in seconds. Hash key covers EDAConfig fields + CSV file mtime+size, including `combat_use_covariates`, `drop_macro_system_covariate`, and resolved matching-policy settings.
- **Cache layout convention**: `out/` holds true prediction artifacts only (`loro_subject_cache/`, `eval_prediction_tables/`, `slurm/`, etc.). Helper-level caches live under `notebooks/cache/{prepost,genewise_kendall}/<hash>.pkl` — safe to delete and recompute.
- **ipywidgets-based interactive single-gene scatter** in Section 4 of `eval_population_genewise.ipynb` — explicit dropdowns + `observe` callbacks (no `%matplotlib widget` / `ipympl` dependency required); inline static figures with one render path.
- Token-based font system (`FONT_TOKENS = {xs, s, m, l, xl, xxl}`, `font_size("m+1")`, `set_font_scale(1.2)`); hero plotters expose `font_sizes={...}` overrides for per-call surgery without leaving the token grammar.
- Canonical metric label formatting (`Pearson r`, `Spearman ρ`, `R²`, `RMSE`, `Kendall τ`) routed through `eval_style.format_metric_label` / `format_legend_label`. Never hardcode metric strings in plot text.
- Tick visibility enforced globally via the rcParams updates in `set_academic_style` and the `apply_tick_style(ax)` per-axes safety helper.
- Per-section mini-config cells: only global config (CFG, MODELS, gene list, view subset, default metric) sits at the top of each eval notebook; section-local knobs live immediately above their use sites for fluid in-notebook iteration. The genewise notebook adds a `build_panel_view(gene_list_path, models=None)` helper closure to consolidate `make_prediction_eval_view` boilerplate.
- DLAM diagnostics utilities live in `src/eval_utils/dlam_diagnostics.py`.
- **Expression harmonization** is documented in `docs/manuscript/expression_harmonization.md`. The active ComBat-style default uses age, sex, and macro-system covariates, then applies a gene-wise GTEx-to-AHBA affine calibration over overlapping parcels. Set `drop_macro_system_covariate=True` to reproduce the previous age+sex-only covariate design.
- **GTEx<->AHBA matching policy auto-detected from the cache.** `EDAConfig.matching_policy`, `matching_policy_hemi_mode`, and `collapse_cerebellum` default to cache metadata when unset. PREPOST applies the resolved settings and includes them in its cache hash so parcel_idx values agree with the cache's.
- **Subject eligibility follows the matching policy** in both `loro_cache.load_dataset` and `_load_expression`: policy is applied first, then `build_subject_eligibility` measures distinct post-policy parcels. `centroids_and_volumes` now keeps `brain - cerebellar hemisphere` and `brain - cerebellum` separate by default (`Cerebellar_Region4` and `Cerebellar_Region7`); `collapse_cerebellum=true` restores the one-cerebellar-parcel collapse.
- **Canonical LORO cache variants** currently use force-left matching: `out/loro_subject_cache_c` = `centroids + force_left + collapse_cerebellum=false`; `out/loro_subject_cache_cv` = `centroids_and_volumes + force_left + collapse_cerebellum=false`; `out/loro_subject_cache_cv_collapse` = `centroids_and_volumes + force_left + collapse_cerebellum=true`.
- **Sample tensor notebooks and render sbatches** default to `data/raw/gxp_samples.csv`. The older `gxp_samples_arxiv.csv` is retained as the archived full/reference CSV, not the active default.
- **Dynamic sbatch array sizing**: `run_loro_cache_array.sbatch` is fixed at `--array=1-385%64` (full GTEx subject pool ceiling). Tasks beyond the policy's eligible count exit cleanly with a `[skip-out-of-range]` log line via `scripts/run_loro_cache_batch.py`. The same submission line works across all `MATCHING_POLICY` × `MIN_OBSERVED_PARCELS` configurations.
- **Sample visualizer plan**: `context_packages/samples_visualizer.md` is the live plan for explaining the full tensor pipeline: original GTEx/AHBA inputs, the raw joint ground-truth view on a shared AHBA superset region frame (`JointTensorView` + `plot_joint_tensor_voxels`, with translucent GTEx imputation-target cells driven by `future_imputation_mask`), ComBat-harmonized matched data, strict LORO imputed data, and full-brain GTEx imputed data. UMAPs are planned primarily for raw matched, ComBat harmonized, and full-brain imputed stages.
- **Embedding diagnostics**: `src/eval_utils/eval_embeddings.py` emits a shared subject-region x gene matrix contract. PREPOST is the direct source for pre/post ComBat; `TensorView` is the adapter for LORO/fullfit and visualizer-derived views. PCA/UMAP helpers apply explicit feature preprocessing (`center`, `standardize`, `none`) and reuse the region-stratified eval palette from `eval_style`.
- **DLAM atlas-prior residual variant (`t_prior_residual`)**: a new DLAM strategy in `src/models/baseline_pipeline.py` that keeps the subject PLS as the sole frame, projects the harmonized atlas through the *subject's* gene loadings to obtain a dense atlas reference in the subject's own latent coordinates, measures the subject-specific residual only at observed parcels, and propagates it with a decaying-kernel RBF that vanishes off-support. The affine basis map, latent spatial transport, U-field RBF, ridge bridge, and inverse-affine step of the original DLAM are bypassed. Residual kernel switch — `imq` (default) | `gaussian` | `tps` | `gp` — plumbed through `SubjectCacheConfig.t_prior_{interp,length_scale,gp_noise,gp_optimize}` → CLI flags on `loro_cache.py` and `scripts/run_loro_cache_batch.py` → env vars on `run_loro_cache_array.sbatch`. The atlas-scale floor (`t_prior_atlas_scale_floor`, default `True`) is intentionally code-only — not on the CLI surface — for ablation hygiene. Method documented as `\section{DLAM Atlas Prior Residual Variant}` in `docs/manuscript/main_edits.tex`.
- **Refactor migrations completed**: `src/eval_utils/eval_latent.py` is now live (pooled PCA recovery + within-parcel demeaned PCA + sample×gene heatmaps + parcel scatters, drives `eval_pca.ipynb`); `src/eval_utils/eval_distributional.py` + `eval_distributional.ipynb` are new (distributional comparison on matched and fullfit-imputed parcels). Notebook splits: Kendall promoted out of the genewise notebook into `eval_population_kendall.ipynb`; region stratification promoted into `eval_regional.ipynb`; subjectwise renamed from `eval_singlesubject.ipynb` into `eval_population_subjectwise.ipynb` with illustrative views expanded from one to four (region / gene-stratified / focus-gene-isolated / focus-gene-by-region). All seven base eval notebooks relocated from the repo root into `notebooks/eval/results/`.

## Repository Layout

- `src/`: reusable preprocessing, harmonization, models, workflows, evaluation utilities, visualization
- `scripts/`: manuscript/workflow CLIs and cache-generation entrypoints
- root `run_loro_cache_*.sbatch` and `render_*gxp_tensor*.sbatch`: Slurm launchers for cache and tensor-render jobs
- `notebooks/`: write-up and supporting analysis notebooks
- `docs/manuscript/`: TeX manuscript and mapping docs
- `configs/`: notebook workflow configs
- `tests/`: smoke/regression tests
- `legacy/`: archived exploratory material

## Data Paths

Expected defaults:
- `data/raw/gxp_samples.csv`
- `data/metadata/gene_lists/ahba_100hvg.txt`
- `data/raw/gxp_samples_arxiv.csv` is the archived/reference sample table; active notebooks and sbatches default to `data/raw/gxp_samples.csv`.
- `docs/samples_builder.md` — detailed pre-`gxp_samples.csv` builder and GTEx
  filtering workflow (`SMRIN > 6`, TPM/read-count thresholds, AHBA overlap
  audit), strict duplicate-subject handling within GTEx tissue files, and
  atlas-coordinate file fallback behavior when local metadata mirrors are
  incomplete

Legacy root-level fallbacks (`gxp_samples.csv`, `ahba_100hvg.txt`) remain supported for migration, but default code paths use `data/raw/` for expression CSVs and `data/metadata/` for gene lists / atlas metadata.

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
- `run_loro_cache_single_subject.sbatch`
- `run_loro_cache_array.sbatch`
- `src/eval_utils/eda_core.py`
- `src/eval_utils/eval_population.py`
- `src/eval_utils/eval_single_subject.py`
- `src/eval_utils/eval_embeddings.py`
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
- The formal manuscript-facing matching policy is documented in `docs/manuscript/parcel_matching.md`, with exploratory diagnostics and final one-to-one mapping tables in `parcel_assignment.ipynb`. The policy keeps centroid matching for manually validatable labels, adds Brodmann/Schaefer voxel-overlap overrides for cortical labels, assigns cerebellar hemisphere/cerebellum explicitly under `centroids_and_volumes`, and only collapses cerebellar labels when `collapse_cerebellum=true`.

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
- Keep new analysis work in the active eval notebooks under `notebooks/eval/results/` (plus `notebooks/eval/{embeddings,gradients,tensors}/` for adjacent families); treat `notebooks/` and the `results_*.ipynb` notebooks at the repo root as historical unless explicitly reviving one. The variant sweep notebooks at the repo root (e.g. `eval_population_genewise_tprior_*.ipynb`) are active. `notebooks/cache/` is an active helper-cache location and is not historical.
- `EDAConfig` (in `eda_core.py`) supports model-directory overrides (`naive_cache_dirname`, `dlam_cache_dirname`, `plam_cache_dirname`) for side-by-side rank experiments (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`).
- New eval-refactor work should target `src/eval_utils/eda_core.py`, `src/eval_utils/eval_population.py`, `src/eval_utils/eval_single_subject.py`, `src/eval_utils/eval_latent.py`, and `src/eval_utils/eval_distributional.py`. Treat `results_eda.py` as compatibility/reference during migration; do not add new functionality there. The heavyweight single-subject visualizations (matrix panel, alignment scatter, performance triplet) still live in `results_eda.py` and remain on the migration backlog.
- Always route metric labels through `eval_style.format_metric_label` (or `format_legend_label`); never hardcode `Pearson R` / `R^2` / `rho` / `tau` strings in plots or tables.
- Use the font-token system (`font_size`, `_DEFAULT_FONT_SIZES`, `font_sizes={...}` per-call overrides, `set_font_scale`) when adding or tuning hero plots — single global scale knob.
- Helper-level caches (PREPOST, genewise Kendall) belong under `notebooks/cache/<name>/<hash>.pkl`. `out/` is reserved for true prediction artifacts.
- When adding a new collaborator share via `nfs4_setfacl` on `/scratch/asr655/...`, always re-assert the inheritable OWNER@ ACE on the same tree, otherwise newly-created files land with mode `0o000` (see CONTEXT.md → "NFSv4 ACL gotcha").
- Do not edit `src/eval_utils/results_eda_arxiv.py`; it is a backup snapshot.
- See `CONTEXT.md` for a fast onboarding summary intended for parallel agents, `context_packages/markdowns/results_eda_refactor_plan.md` for the live eval refactor plan, and `context_packages/markdowns/samples_visualizer.md` for the sample/tensor visualizer plan.

Last updated at: 2026-05-28
