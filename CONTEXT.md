# Repo Context

## Purpose

`gtex_gp` is the AHBA<->GTEx atlas-alignment repo used to:
- harmonize AHBA and GTEx gene expression into a shared domain,
- complete sparse GTEx subject maps over the full atlas,
- evaluate completion quality with strict leave-one-region-out (LORO) validation,
- generate manuscript-facing figures/tables.

## Current Development Direction

Near-term work spans (a) the eval refactor and (b) the DLAM atlas-prior residual (`t_prior_residual`) modeling thread.

**Eval notebooks** are now organized under `notebooks/eval/`:
- `notebooks/eval/results/` — seven base eval notebooks (population, genewise, kendall, subjectwise, regional, distributional, pca).
- `notebooks/eval/embeddings/` — six cross-dataset and GTEx-only embedding diagnostics (cov / nocov variants).
- `notebooks/eval/gradients/` — DLAM alignment + PLS gradient progression + PLS-gradients-on-brain.
- `notebooks/eval/tensors/` — raw, ComBat, pipeline, and prediction-joint tensor visualizers.
- `notebooks/eval/eval_gxp_onbrain_views.ipynb` — top-level on-brain renders.
- `notebooks/preprocessing/eval_data.ipynb` — dataset-level EDA (moved out of /results since it isn't a model-eval notebook).

**Sweep / variant notebooks at the repo root**: `eval_population_genewise_{anchor,gp,rbf,rbf_shrink}.ipynb` (legacy interpolation sweep) and `eval_population_{genewise,subjectwise,}_tprior_{gp,imq,tps}.ipynb` (active t-prior atlas-prior residual sweep). These read from per-variant `out/loro_subject_cache_*` roots.

**Canonical implementation surfaces** under `src/eval_utils/`: `eval_population.py`, `eval_single_subject.py`, `eval_style.py`, `eda_core.py`, `eval_latent.py` (PCA / latent recovery — now landed), `eval_distributional.py` (distributional eval — new), plus the adjacent families `eval_embeddings.py`, `eval_flattened.py`, `eval_onbrain.py`, `eval_pls_gradients.py`, `eval_pls_gradient_progression.py`, `eval_dlam_alignment.py`, `eval_samples.py`, and `dlam_diagnostics.py`. `results_eda.py` remains compatibility/reference only and still houses the heavyweight single-subject visualizations (matrix panel, alignment scatter, performance triplet) that haven't yet migrated to `eval_single_subject.py`. `results_eda_arxiv.py` is a backup snapshot.

Refactor plan and status: `context_packages/markdowns/results_eda_refactor_plan.md` (kept current with the seven-notebook layout, the t-prior modeling thread, and the standing Next items: heavyweight single-subject migration, age bias forest, gene-sublist builder, samples.csv builder).

Sample/tensor visualizer plan: `context_packages/samples_visualizer.md`.
`src/eval_utils/eval_samples.py` is the unified surface: one `TensorView`
contract (with `pipeline_stage`, `future_imputation_mask`, `matching`), one
cube adapter `build_tensor_view_from_cube(...)`, three source facades
(`build_native_tensor_view` CSV, `build_combat_tensor_view` PREPOST,
`build_prediction_tensor_view` npz cache), and two renderers
(`plot_gtex_tensor_voxels` single, `plot_joint_tensor_voxels` joint, both with
`mask_render_mode` two-pass and data-derived future-imputation rendering).
Three live notebooks, one per pipeline layer:
- `eval_gxp_samples.ipynb` — raw GTEx+AHBA, standalone then joint.
- `eval_gxp_samples_combat.ipynb` — ComBat joint pre/post (PREPOST cubes).
- `eval_gxp_samples_predictions_single.ipynb` — standalone `loro_truth`/`loro_recon`
  single tensors, matched native-GTEx-parcel axis, dense region ticks.
- `eval_gxp_samples_predictions_joint.ipynb` — `loro_truth`/`loro_recon`/`loro_fused`/`fullfit`
  joint-vs-AHBA on the superset axis, plus dense `loro_fused`/`fullfit` standalones.
Native single-dataset reference renders are archived under `notebooks/arxiv/`.
`region_ordering` (`target_parcel`/`region_matched`/`region_matched_superset`)
is shared across cube facades; `region_matched_superset` makes GTEx share the
AHBA superset axis with `future_imputation_mask` on padded/extrapolated cells.
UMAPs (raw matched, harmonized, full-fit) remain the open next stage.

## Main Entrypoints

- `README.md` — top-level usage and commands
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb` — canonical manuscript workflow
- `scripts/build_dual_model_loro_metric_panels.py` — manuscript LORO panel builder
- `src/workflows/loro_cache.py` — per-subject cache builder
- `scripts/run_loro_cache_batch.py` — batch/array subject driver
- `src/eval_utils/eda_core.py` — dataset-level EDA, PREPOST reconstruction (cached via `prepare_pre_post_harmonization_cached`), path/gene-list resolution
- `docs/manuscript/expression_harmonization.md` — formal AHBA-GTEx ComBat-style harmonization procedure, covariates, affine GTEx-to-AHBA calibration, and caveats
- `src/eval_utils/eval_style.py` — shared model/region labels, colors, font-token system, metric-label formatter (incl. Kendall τ), tick-style enforcer
- `src/eval_utils/eval_population.py` — population-level cached prediction evaluation (sample-wise + gene-wise + within-subject spatial Kendall)
- `src/eval_utils/eval_single_subject.py` — subject-keyed analyses (perf summary, specificity, spatial specificity)
- `src/eval_utils/eval_embeddings.py` — embedding matrix contract plus PCA/UMAP plots; PREPOST is the direct pre/post source and `TensorView` is the LORO/fullfit adapter
- `src/eval_utils/results_eda.py` — legacy/reference compatibility surface during migration
- `src/eval_utils/dlam_diagnostics.py` — DLAM diagnostics fit/cache/plot utilities
- `src/viz/coord_viz.py` — GTEx/AHBA coordinate overlay utilities
- `notebooks/eval/results/{eval_population,eval_population_genewise,eval_population_kendall,eval_population_subjectwise,eval_regional,eval_distributional,eval_pca}.ipynb` — active eval notebooks (seven, per-section CFG + ToC pattern)
- `notebooks/preprocessing/eval_data.ipynb` — dataset-level EDA (PREPOST, demographics, coverage)
- `notebooks/eval/tensors/{eval_gxp_samples,eval_gxp_samples_combat,eval_gxp_samples_pipeline,eval_gxp_samples_predictions_joint,eval_gxp_samples_predictions_joint_alpha,eval_gxp_onbrain_view_prep}.ipynb` — tensor visualizer family
- `notebooks/eval/embeddings/eval_{cross_dataset,gtex}_embeddings{_cov,_nocov,}.ipynb` — embedding diagnostics (cov / nocov variants)
- `notebooks/eval/gradients/{eval_dlam_alignment,eval_pls_gradient_progression,eval_pls_gradients_onbrain}.ipynb` — DLAM alignment + PLS gradient + on-brain projections
- `notebooks/eval/eval_gxp_onbrain_views.ipynb` — on-brain rendering surface
- Variant sweep notebooks at repo root: `eval_population_genewise_tprior_{gp,imq,tps}.ipynb`, `eval_population_{subjectwise_,}tprior_imq.ipynb`, and the legacy `eval_population_genewise_{anchor,gp,rbf,rbf_shrink}.ipynb`
- `notebooks/coordinate_overlay_3d_mni.ipynb` (a.k.a. `coordinate_assignment_3dmni.ipynb` at repo root) — parcel-assignment visualization notebook

Legacy notebooks kept for reference: `results_cached_predictions.ipynb`, `results_single_subject_predictions.ipynb`, `pca_cached_predictions.ipynb`.

## Key Recent Changes (Important)

1. Subject-wise `.npz` caching for naive/DLAM/PLAM under `out/loro_subject_cache/...`.
2. Strict LORO-at-evaluable-parcels + single full-data fallback elsewhere.
3. Cache schema includes masks for downstream modular evaluation:
   - `gtex_mask` (global GTEx parcels),
   - `loro_eval_mask` (subject-specific strict eval parcels),
   - `imputed_mask` (non-LORO parcels).
4. DLAM gate default lowered to `c_min=4` so low-coverage eligible subjects are fit in LORO folds.
5. PLAM dynamic-rank support (`dynamic_rank`, `plam_latent_dim_max`, fold-level rank in cache).
6. Atlas/parcel reduction is configurable (`atlas_agg=mean|median`).
7. GTEx parcel assignment is configurable (`gtex_rep_mode=centroid|medoid`, `gtex_hemi_mode=native|mirror_left`).
8. Root sbatch launchers thread `LATENT_DIM`, `DYNAMIC_RANK`, `PLAM_MAX_RANK`, `ATLAS_AGG`, `GTEX_REP_MODE`, `GTEX_HEMI_MODE`, `MATCHING_POLICY_HEMI_MODE`, and `COLLAPSE_CEREBELLUM`.
9. EDA config supports model folder remapping (`*_cache_dirname`) for rank comparisons (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`).
10. Population eval centers on canonical wide all-gene truth/prediction tables under `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/` (e.g. `out/eval_prediction_tables/loro_subject_cache_cv/allgenes/`). The subdir is keyed on `cfg.cache_root.name` so different LORO caches do not collide; switching `cache_root` in a notebook routes reads/writes automatically. Gene-list analyses subset downstream via `eval_gene_list_path`.
11. Population metrics default to sample-wise: per-`subject_region_key` scoring across genes, then aggregation across samples or strata. Spearman is opt-in.
12. Sex-bias analysis available via `compute_stratum_bias` + `plot_stratum_bias_forest` (Fisher-z(metric) ~ axis + controls; LMM with subject random intercept, OLS+cluster fallback; BH-FDR; df=1 omnibus fallback).
13. LORO ranked overlay + matched overlay + per-fold dist-colored scatter unified into a single `plot_fold_combo_ranked_overlay` (line / `dist_to_nearest_train` / `dist_to_centroid_train` modes; per-model gradient cmaps; reference-model colorbar; rolling-mean overlay).
14. Fold-difficulty decomposition (`plot_fold_std_vs_mean`) drops `n<5` fold-combos, scales marker area to `n_subjects`, computes the bottom-tail cutoff on the full pre-filter set.
15. `eval_single_subject.py` houses subject performance summary + percentile selector (direction-aware: 0 = worst, 100 = best regardless of metric direction), the ranked-with-percentile-anchor plot (with optional `dynamic_bottom_quantile` that auto-shades up to the first crossover where the anchor model beats `naive_baseline_model`), subject specificity (self vs other-subject truth at the same region), and spatial specificity (self vs mean of `k − 1` per-region metrics within the same subject's truth).
16. **Genewise surface** in `eval_population.py`: `compute_genewise_metrics(view, ..., stratify_by=None)` produces the per-(model, gene[, stratum]) table; `plot_genewise_ranked_overlay` (with `group_by='model'|'gtex_region'`, gene labels on the curve at every Nth percentile, n_samples in legend) and `plot_genewise_histogram` (overlapping per-group distributions with median rules) cover the visualizations. Tissue-stratified plots use the same shared region palette as the rest of the repo via `_tissue_palette` (region_group buckets + n_samples-weighted ordering matching `_global_scatter_color_spec`).
17. **Gene-sublist scatters** in `plot_global_prediction_scatter`: `gene_selection='top'|'bottom'|'random'|'list_order'` with `gene_rank_lookup` precomputed once via `compute_gene_rank_lookup(view, model, metric)`, plus `highlight_genes=[…]` hand-pick override and `base_alpha_factor` for fading "Other" points. Stratum isolation (`isolate_stratum`, `isolate_color_subby`) for side-by-side panels with sub-gradient coloring.
18. **Within-subject spatial Kendall-τ** in `eval_population.py`: `compute_within_subject_kendall(view, ..., min_regions=5, cache=True)` produces a long table `(subject, gene, model, kendall_tau, n_regions)` keyed for disk caching. `summarize_kendall_per_unit(unit_col='gene'|'subject')` aggregates with median/mean/std + n_valid; `plot_unitwise_kendall_ranked` is the generalized ranked plot for either axis. `filter_kendall_to_genes(kendall_long, genes)` lets sublist analyses reuse the cached all-genes table without recomputing.
19. **Disk-cached PREPOST** via `prepare_pre_post_harmonization_cached(cfg)` in `eda_core.py`. Hash key covers `(csv_path, cache_root, gene_scope, min_observed_parcels, combat_use_covariates, drop_macro_system_covariate, gtex_rep_mode, gtex_hemi_mode, resolved_matching_policy, resolved_matching_policy_hemi_mode, resolved_collapse_cerebellum)` plus the CSV file's `(mtime_ns, size)` — `hvg_path` is no longer in EDAConfig (HVG gene panel for `gene_scope='hvg'` is sourced from the LORO cache's subject `gene_names` instead). First call computes (slow ComBat fit), subsequent calls load in seconds. Cache lives at `notebooks/cache/prepost/<hash>.pkl`. Same cache pattern at `notebooks/cache/genewise_kendall/<hash>.pkl` for the Kendall long table. **Out is reserved for true prediction artifacts; helper-level caches live under `notebooks/cache/`.**
20. **ipywidgets-based interactive single-gene scatter** in Section 4 of `eval_population_genewise.ipynb` — explicit dropdowns + `observe` callbacks (no `%matplotlib widget` / `ipympl` dependency). Inline static figures, one render path, one Output widget. `'all'` model option renders all 3 panels in one figure; specific model renders single panel.
21. Token-based font system in `eval_style.py`: `FONT_TOKENS = {xs, s, m, l, xl, xxl}`, `font_size("token±N")`, `set_font_scale(scale)`, `_resolve_fonts(defaults, override)`. Hero plotters declare `_DEFAULT_FONT_SIZES` and accept `font_sizes={...}` overrides. Single global rescale knob.
22. Tick visibility forced via `_TICK_RC` injected into `sns.set_theme(rc=...)`, re-asserted at module import, and `apply_tick_style(ax)` available for per-axes safety.
23. Canonical metric labels (`Pearson r`, `Spearman ρ`, `R²`, `RMSE`, `Kendall τ`) are routed through `format_metric_label` / `format_legend_label` — never hardcode metric strings in plot text.
24. Per-section mini-config cells in eval notebooks: only truly global knobs sit at the top; section-local knobs (LORO coverage, anchor model, percentile shading, illustrative subjects, specificity, gene panel, Kendall sublist) live immediately above their use sites. The genewise notebook adds a single `build_panel_view(gene_list_path, models=None)` helper closure to consolidate `make_prediction_eval_view` boilerplate across all sections.
25. The ongoing eval refactor moves stable functionality out of monolithic `results_eda.py` into focused `eval_*` modules. Do not build new refactor functionality in `results_eda.py` unless it is a short-lived compatibility bridge.
26. **Matching policy auto-detect**: `EDAConfig.matching_policy`, `matching_policy_hemi_mode`, and `collapse_cerebellum` default to metadata read from any subject JSON under `<cache_root>/<gene_scope>/<model>/`, with legacy fallbacks `centroids`, `default`, and `false`. PREPOST applies the resolved settings and includes them in its disk-cache hash.
27. **Eligibility ordering invariant**: `build_subject_eligibility` runs *after* `apply_gtex_ahba_matching_policy` in both `loro_cache.load_dataset` and `_load_expression`. `centroids_and_volumes` now assigns `brain - cerebellar hemisphere` to `Cerebellar_Region4` and `brain - cerebellum` to `Cerebellar_Region7` by default; `collapse_cerebellum=true` explicitly restores the old one-cerebellar-parcel collapse to `Cerebellar_Region7`. PREPOST and the cache builder must apply policy → eligibility in this order or eligible sets can diverge.
28. **Dynamic sbatch array**: root `run_loro_cache_array.sbatch` uses `--array=1-385%64` (the full GTEx subject pool ceiling). `scripts/run_loro_cache_batch.py:_pick_subjects` exits cleanly with `[skip-out-of-range] task_id=N n_eligible=M` when the array index exceeds the policy's eligible count. The same launcher works across `MATCHING_POLICY`, `MATCHING_POLICY_HEMI_MODE`, `COLLAPSE_CEREBELLUM`, and `MIN_OBSERVED_PARCELS` combinations.
29. **EDAConfig slimmed to data-lineage essentials**: `csv_path`, `cache_root`, `gene_scope` are the primary uniqueness keys; `hvg_path` is removed (HVG list is sourced from the LORO cache's subject npz `gene_names` for `gene_scope='hvg'`). `plam_cache_dirname` (and its naive/dlam siblings) stay in the dataclass for variant-rank model dirs (e.g. `plam_rank4`) but are not surfaced in notebook CFGs unless overridden. The four eval notebooks (`eval_data`, `eval_population`, `eval_population_genewise`, `eval_singlesubject`) carry minimal CFG cells with `csv_path` + `cache_root` + `gene_scope` only, plus a Table-of-Contents markdown right after the title. Default sublist gene panel is `gtex_100hvg` everywhere a sublist is needed (Section 3 scatter, Kendall sublist, illustrative-subject view B); top-level analyses default to `gene_scope='allgenes'` with `EVAL_GENE_LIST_PATH=None` (all genes from the cached tables). **Current cache-root convention**: `out/loro_subject_cache_c` = `centroids + force_left + collapse_cerebellum=false`; `out/loro_subject_cache_cv` = `centroids_and_volumes + force_left + collapse_cerebellum=false`; `out/loro_subject_cache_cv_collapse` = `centroids_and_volumes + force_left + collapse_cerebellum=true`.
30. **Data layout: `data/raw/` vs `data/metadata/`**: raw expression CSVs (`gxp_samples.csv`) stay in `data/raw/`; gene lists and atlas info moved to `data/metadata/{gene_lists,atlas_info}/`. Module-level resolvers (`AtlasOverlapPaths`, `default_cerebellar_gene_list_paths`, `_resolve_eval_gene_path`, `_genes_from_cache`, the loro_cache HVG fallback list) try `data/metadata/...` first and fall back to `data/raw/...` so legacy paths still resolve.
31. **Pre-`gxp_samples.csv` GTEx gene filtering** now has an explicit audit path in `src/samples/{gtex,build}.py` and `docs/samples_builder.md`: restrict to brain samples, apply `SMRIN > 6`, compute GTEx cohort-wide TPM/read-count thresholds on the retained samples, then intersect that GTEx gene set with the AHBA preprocessing universe before any optional small manuscript panel is applied.
32. **`build_gxp_samples(...)` now owns the formal GTEx filter interface**: the builder accepts `gtex_sample_attributes_path`, `gtex_rin_threshold`, `gtex_tpm_threshold`, `gtex_reads_threshold`, `gtex_min_fraction`, `gtex_assay_freeze`, and `return_overlap_info`. Notebook exploration may still call `compute_gtex_ahba_overlap_report(...)` directly, but formal `gxp_samples.csv` creation should pass these arguments through the builder.
33. **GTEx duplicate subject IDs are now a hard error, not a silent average**: after parsing GTEx sample IDs down to subject IDs within a tissue file, duplicate subjects raise instead of being merged by mean. This preserves the invariant "one GTEx row per subject per tissue" and avoids unreviewed expression averaging in the canonical CSV.
34. **Atlas coordinate CSV resolution is schema-aware**: when building GTEx tissue coordinate lists, the builder requires `mni_x`, `mni_y`, `mni_z` for the Brodmann/S156 coordinate tables and skips incomplete local mirrors that only contain label metadata. This prevents late failures when `data/metadata/atlas_info` lacks full coordinate columns but the upstream `GeneEx2Conn_data/atlas_info` copy is complete.
35. **Sample visualizer direction**: the visualizer should tell the pipeline story in five stages: original GTEx/AHBA inputs, raw region-matched data, ComBat-harmonized matched data, strict LORO imputed data, and full-brain GTEx imputed data. Keep raw input tensors separate from model-space tensors, and carry explicit dataset / expression-space / region-axis-kind / value-source / mask-source lineage on tensor views.
36. **Joint tensor layering**: cross-dataset views go through three reusable layers in `eval_samples.py` — per-dataset `TensorView` from `build_sampled_tensor(..., region_ordering="region_matched_superset")`, then `JointTensorView` from `build_joint_tensor_view(gtex_view, ahba_view)` (validates identical genes/regions/axis-kind and records a `matched_region_count`), then `plot_joint_tensor_voxels(joint_view, ...)` for rendering. `dataset="combined"` is the convenience wrapper. GTEx's superset-padded AHBA-only cells carry `future_imputation_mask=True` and render translucent; AHBA's unobserved cells stay solid gray. Raw GTEx (`log1p(TPM)`) and AHBA (microarray intensity) are not on a shared scale — the raw combined plot uses per-dataset normalization or twin colorbars; harmonized/predicted joint pairs can collapse to one colorbar.
37. **ComBat-style harmonization covariates**: active defaults use `combat_use_covariates=True` and `drop_macro_system_covariate=False`, so the design is age + sex + macro_system. Macro-system is encoded with `cortical_association` as reference and dummies for `cerebellar`, `subcortical`, and `visual_somatomotor`. Set `drop_macro_system_covariate=True` to reproduce the previous age+sex-only behavior.
38. **Embedding diagnostics first pass**: `eval_embeddings.ipynb` builds GTEx ground-truth and post-ComBat matrices directly from PREPOST, then builds the LORO reconstruction matrix from `build_prediction_tensor_view(..., stage='loro_recon')`. All stages emit the same `RegionEmbeddingMatrix` contract before PCA/UMAP. Feature preprocessing defaults to `center`; region, macro-system, and region-group coloring all route through shared `eval_style` palette helpers.
39. **`t_prior_residual` DLAM strategy** in `src/models/baseline_pipeline.py`: a new variant of DLAM that keeps the subject PLS as the sole frame, projects the harmonized atlas through the *subject's* gene loadings to obtain a dense atlas reference in the subject's latent coordinates (`Z_A^{(s)} = X_A^{(h)} W_s`), measures the subject-specific residual only at observed parcels, and propagates it with a decaying-kernel RBF that vanishes off-support so predictions relax to the atlas reference rather than extrapolate unboundedly. The affine basis map, the latent spatial transport, the U-field RBF, the ridge bridge, and the inverse-affine step of the original DLAM are all bypassed for this strategy. Residual interpolator kernel switch — `imq` (default) | `gaussian` | `tps` | `gp` — wired through `_fit_residual_interp` in `baseline_pipeline.py`; per-gene standardization used at PLS fit time is shared with the atlas projection and lower-bounded by the atlas-side standard deviation as an implementation safeguard against blow-up on low-variance genes. Pipeline plumbing: `SubjectCacheConfig.t_prior_{interp,length_scale,gp_noise,gp_optimize,atlas_scale_floor}` → method_bundle keys → CLI flags on both `loro_cache.py` and `scripts/run_loro_cache_batch.py` (the atlas-scale floor is *intentionally not* on the CLI surface — it stays a code-level default of `True` for ablation hygiene). The cache hash excludes `t_prior_*` keys when the strategy is not `t_prior_residual` so existing naive/plam/anchor caches stay valid. Method documented as a parallel `\section{DLAM Atlas Prior Residual Variant}` in `docs/manuscript/main_edits.tex` directly after the original DLAM section.
40. **`run_loro_cache_array.sbatch` carries t-prior env vars** — `T_PRIOR_INTERP`, `T_PRIOR_LENGTH_SCALE`, `T_PRIOR_GP_NOISE`, `T_PRIOR_GP_OPTIMIZE` — forwarded to the batch driver. The atlas-scale floor remains unexposed. Canonical per-variant sweep cache roots follow the convention `out/loro_subject_cache_cv_tprior_{gp,imq,tps}` (centroids_and_volumes + force_left matching, allgenes). Reading caches from these roots is what drives the variant sweep notebooks at the repo root.
41. **`eval_latent.py` landed** (was previously pending): pooled PCA fit/eval, reconstruction recovery, within-parcel demeaned PCA, within-parcel variance effects, sample×gene heatmap utilities, parcel prediction scatters. Drives `notebooks/eval/results/eval_pca.ipynb` end-to-end. PLS comparison surfaces from the original `pca_cached_predictions.ipynb` remain to migrate.
42. **`eval_distributional.py` + `eval_distributional.ipynb`** (new): distributional comparison of truth vs prediction on matched held-out parcels and on fullfit-imputed parcels.
43. **Kendall split into its own notebook** (`notebooks/eval/results/eval_population_kendall.ipynb`); the genewise notebook keeps the per-gene predictability surface and the interactive single-gene scatter. **Region stratification promoted** into its own `eval_regional.ipynb` (adds the DLAM gene-highlighted region fan). **Subjectwise renamed** to `eval_population_subjectwise.ipynb` (joins the `eval_population_*` naming family) and its illustrative-subject section expanded to four views — region / gene-stratified / focus-gene-isolated / focus-gene-colored-by-region.

## Core LORO Semantics

- Fold unit is a **subject-observed parcel**.
- For each fold, one parcel from one subject is removed from training.
- Harmonization is re-fit on fold training data.
- Held-out truth is harmonized with that fold harmonizer.
- Strict fold prediction is evaluated only at held-out parcel(s).
- Final `loro_fused_subject_h` is a fused map (strict LORO where available + `fullfit_subject_h` elsewhere).
- Native raw-space held-out truth is still stored as `loro_truth_subject_raw` for mixed-space evaluation.

## Data / Path Assumptions

Default raw inputs:
- `data/raw/gxp_samples.csv`
- `data/metadata/gene_lists/ahba_100hvg.txt`

`data/raw/gxp_samples.csv` is the active sample table for LORO and sample/tensor eval notebooks. `data/raw/gxp_samples_arxiv.csv` is retained as the archived/reference sample table.

Default outputs (true prediction artifacts only — `out/` is reserved for these):
- write-up pipeline: `out/notebook_writeup/`
- subject caches: `out/loro_subject_cache/`
- canonical wide eval tables: `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/`
- slurm logs: `out/slurm/`

Helper-level caches (not predictions; safe to delete and recompute):
- PREPOST: `notebooks/cache/prepost/<hash>.pkl`
- Genewise Kendall: `notebooks/cache/genewise_kendall/<hash>.pkl`

## Do Not Casually Modify

- LORO fold semantics (held-out parcel policy, fold harmonization behavior)
- cache field names/shapes consumed by EDA notebooks
- sbatch container activation pattern (`source /ext3/env.sh`) without checking cluster impact
- manuscript asset naming conventions used by TeX docs/scripts
- canonical metric formatting (`format_metric_label`) — adopted to keep plots/tables consistent
- `notebooks/cache/` layout — `prepare_pre_post_harmonization_cached` and `compute_within_subject_kendall` default there; moving the dirs invalidates existing pickles
- eligibility ordering (`apply_gtex_ahba_matching_policy` *before* `build_subject_eligibility`) — see Key Recent Change #27. Eligibility must be measured against the active policy's parcel scheme; reversing the order admits subjects whose post-policy coverage is below `min_observed_parcels`.

## HPC Notes

- Root sbatches are the current launch points:
  - `run_loro_cache_single_subject.sbatch`
  - `run_loro_cache_array.sbatch`
- Scripts are CPU-oriented; no required GPU path for current cache generation.
- Array jobs are one subject per task and support `GENE_SCOPE=hvg|allgenes`.
- Dynamic-rank PLAM launch example:
  - `MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch run_loro_cache_array.sbatch`

### NFSv4 ACL gotcha (NYU vast / `/scratch/asr655`)

`/scratch/asr655` enforces NFSv4 ACLs on top of POSIX mode. When a collaborator is granted access via `nfs4_setfacl -a 'A:fd:user@hpc.nyu.edu:...'` *without* also adding `fd` inheritance to OWNER@, newly-created files/dirs land with mode `0o000` because OWNER@ is not propagated. Repair recipe (run once per affected tree):

```bash
find <ROOT> -type d ! -perm -u+rwx -exec chmod u+rwx {} + 2>/dev/null
find <ROOT> -type f ! -perm -u+rw  -exec chmod u+rw  {} + 2>/dev/null
find <ROOT> -type d -exec nfs4_setfacl -a 'A:fdg:OWNER@:rwaDxtTnNcy' {} +
```

Top-level `/scratch/asr655` already has the inheritable OWNER@ ACE applied; new trees created from now on inherit correctly. Trees imported via `tar`/`rsync` without `--acls` need the chmod sweep. Defensive code in `prepare_pre_post_harmonization_cached` (chmod-after-mkdir + warn-and-continue on PermissionError) provides a safety net.

## Quick Start (Agent Onboarding)

1. Read `README.md` and this file.
2. For active eval-refactor work, inspect `context_packages/markdowns/results_eda_refactor_plan.md`, then the seven notebooks under `notebooks/eval/results/` (`eval_population`, `eval_population_genewise`, `eval_population_kendall`, `eval_population_subjectwise`, `eval_regional`, `eval_distributional`, `eval_pca`) and the focused `src/eval_utils/eval_*` modules. For dataset-level EDA, use `notebooks/preprocessing/eval_data.ipynb`.
3. For DLAM `t_prior_residual` work, see `src/models/baseline_pipeline.py` (the strategy branch + `_fit_residual_interp`), `src/spatial/rbf.py` (the kernel switch), and the manuscript section `\section{DLAM Atlas Prior Residual Variant}` in `docs/manuscript/main_edits.tex`. Sweep notebooks live at the repo root (`eval_population_genewise_tprior_{gp,imq,tps}.ipynb`) and read from `out/loro_subject_cache_cv_tprior_{gp,imq,tps}`.
4. For sample/tensor visualizer work, inspect `context_packages/markdowns/samples_visualizer.md`, then `src/eval_utils/eval_samples.py` (the unified surface) and the notebooks under `notebooks/eval/tensors/`, plus `src/eval_utils/eda_core.py` (PREPOST) and the LORO cache readers in `src/eval_utils/eval_population.py`.
5. For gradient / on-brain / embedding work, use the focused dirs under `notebooks/eval/{gradients,embeddings}/` and the matching modules `eval_pls_gradients.py`, `eval_pls_gradient_progression.py`, `eval_onbrain.py`, `eval_embeddings.py`, `eval_dlam_alignment.py`.
6. Use `src/eval_utils/results_eda.py` as a legacy reference, not a default implementation target.
7. Do not edit `src/eval_utils/results_eda_arxiv.py`; it is a backup snapshot.
8. Use `notebooks/coordinate_overlay_3d_mni.ipynb` (or repo-root `coordinate_assignment_3dmni.ipynb`) when working on spatial assignment or matching changes.
9. Treat `notebooks/` legacy notebooks and the `results_*.ipynb` notebooks at repo root as historical unless intentionally reviving one. `notebooks/cache/` is for active helper-level caches and is fine to populate.

Last updated at: 2026-05-28
