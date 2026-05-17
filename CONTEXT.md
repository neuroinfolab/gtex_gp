# Repo Context

## Purpose

`gtex_gp` is the AHBA<->GTEx atlas-alignment repo used to:
- harmonize AHBA and GTEx gene expression into a shared domain,
- complete sparse GTEx subject maps over the full atlas,
- evaluate completion quality with strict leave-one-region-out (LORO) validation,
- generate manuscript-facing figures/tables.

## Current Development Direction

Near-term work is centered on the eval refactor split across four active notebooks:
- `eval_data.ipynb` — dataset-level EDA (PREPOST, demographics, coverage, heatmaps).
- `eval_population.ipynb` — population-level cached prediction evaluation (global scatters, sample-wise stratifications, sex-bias forest, LORO fold-combo overlays + dist-colored gradients, fold-difficulty decomposition, distance-to-training).
- `eval_singlesubject.ipynb` — subject-keyed analyses (per-subject performance distribution, percentile-anchored illustrative scatters, subject specificity, spatial specificity).
- `eval_population_genewise.ipynb` — per-gene analyses (gene-wise ranked + histogram overall and tissue-stratified, gene-sublist scatters with rank-based selection or hand-picked highlights, ipywidgets single-gene scatter, within-subject spatial Kendall-τ).

The corresponding source modules under `src/eval_utils/` (`eval_population.py`, `eval_single_subject.py`, `eval_style.py`, `eda_core.py`, `dlam_diagnostics.py`) are the canonical implementation surfaces. `results_eda.py` is now compatibility/reference only and houses the heavyweight single-subject visualizations (matrix panel, alignment scatter, performance triplet) that haven't yet migrated. `results_eda_arxiv.py` is a backup snapshot.

Refactor plan and status: `context_packages/results_eda_refactor_plan.md`.

## Main Entrypoints

- `README.md` — top-level usage and commands
- `notebooks/ahba_gtex_writeup_end_to_end.ipynb` — canonical manuscript workflow
- `scripts/build_dual_model_loro_metric_panels.py` — manuscript LORO panel builder
- `src/workflows/loro_cache.py` — per-subject cache builder
- `scripts/run_loro_cache_batch.py` — batch/array subject driver
- `src/eval_utils/eda_core.py` — dataset-level EDA, PREPOST reconstruction (cached via `prepare_pre_post_harmonization_cached`), path/gene-list resolution
- `src/eval_utils/eval_style.py` — shared model/region labels, colors, font-token system, metric-label formatter (incl. Kendall τ), tick-style enforcer
- `src/eval_utils/eval_population.py` — population-level cached prediction evaluation (sample-wise + gene-wise + within-subject spatial Kendall)
- `src/eval_utils/eval_single_subject.py` — subject-keyed analyses (perf summary, specificity, spatial specificity)
- `src/eval_utils/results_eda.py` — legacy/reference compatibility surface during migration
- `src/eval_utils/dlam_diagnostics.py` — DLAM diagnostics fit/cache/plot utilities
- `src/viz/coord_viz.py` — GTEx/AHBA coordinate overlay utilities
- `eval_data.ipynb` / `eval_population.ipynb` / `eval_singlesubject.ipynb` / `eval_population_genewise.ipynb` — active eval notebooks
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
8. Sbatch launchers under `scripts/sbatch/` thread `LATENT_DIM`, `DYNAMIC_RANK`, `PLAM_MAX_RANK`, `ATLAS_AGG`, `GTEX_REP_MODE`, `GTEX_HEMI_MODE`.
9. EDA config supports model folder remapping (`*_cache_dirname`) for rank comparisons (`plam_rank3`, `plam_rank4`, `plam_dynamicrank`).
10. Population eval centers on canonical wide all-gene truth/prediction tables under `out/eval_prediction_tables/<cache_root_basename>/<gene_scope>/` (e.g. `out/eval_prediction_tables/loro_subject_cache_coord_vol/allgenes/`). The subdir is keyed on `cfg.cache_root.name` so different LORO caches do not collide; switching `cache_root` in a notebook routes reads/writes automatically. Gene-list analyses subset downstream via `eval_gene_list_path`.
11. Population metrics default to sample-wise: per-`subject_region_key` scoring across genes, then aggregation across samples or strata. Spearman is opt-in.
12. Sex-bias analysis available via `compute_stratum_bias` + `plot_stratum_bias_forest` (Fisher-z(metric) ~ axis + controls; LMM with subject random intercept, OLS+cluster fallback; BH-FDR; df=1 omnibus fallback).
13. LORO ranked overlay + matched overlay + per-fold dist-colored scatter unified into a single `plot_fold_combo_ranked_overlay` (line / `dist_to_nearest_train` / `dist_to_centroid_train` modes; per-model gradient cmaps; reference-model colorbar; rolling-mean overlay).
14. Fold-difficulty decomposition (`plot_fold_std_vs_mean`) drops `n<5` fold-combos, scales marker area to `n_subjects`, computes the bottom-tail cutoff on the full pre-filter set.
15. `eval_single_subject.py` houses subject performance summary + percentile selector (direction-aware: 0 = worst, 100 = best regardless of metric direction), the ranked-with-percentile-anchor plot (with optional `dynamic_bottom_quantile` that auto-shades up to the first crossover where the anchor model beats `naive_baseline_model`), subject specificity (self vs other-subject truth at the same region), and spatial specificity (self vs mean of `k − 1` per-region metrics within the same subject's truth).
16. **Genewise surface** in `eval_population.py`: `compute_genewise_metrics(view, ..., stratify_by=None)` produces the per-(model, gene[, stratum]) table; `plot_genewise_ranked_overlay` (with `group_by='model'|'gtex_region'`, gene labels on the curve at every Nth percentile, n_samples in legend) and `plot_genewise_histogram` (overlapping per-group distributions with median rules) cover the visualizations. Tissue-stratified plots use the same shared region palette as the rest of the repo via `_tissue_palette` (region_group buckets + n_samples-weighted ordering matching `_global_scatter_color_spec`).
17. **Gene-sublist scatters** in `plot_global_prediction_scatter`: `gene_selection='top'|'bottom'|'random'|'list_order'` with `gene_rank_lookup` precomputed once via `compute_gene_rank_lookup(view, model, metric)`, plus `highlight_genes=[…]` hand-pick override and `base_alpha_factor` for fading "Other" points. Stratum isolation (`isolate_stratum`, `isolate_color_subby`) for side-by-side panels with sub-gradient coloring.
18. **Within-subject spatial Kendall-τ** in `eval_population.py`: `compute_within_subject_kendall(view, ..., min_regions=5, cache=True)` produces a long table `(subject, gene, model, kendall_tau, n_regions)` keyed for disk caching. `summarize_kendall_per_unit(unit_col='gene'|'subject')` aggregates with median/mean/std + n_valid; `plot_unitwise_kendall_ranked` is the generalized ranked plot for either axis. `filter_kendall_to_genes(kendall_long, genes)` lets sublist analyses reuse the cached all-genes table without recomputing.
19. **Disk-cached PREPOST** via `prepare_pre_post_harmonization_cached(cfg)` in `eda_core.py`. Hash key now covers `(csv_path, cache_root, gene_scope, min_observed_parcels, combat_use_covariates, gtex_rep_mode, gtex_hemi_mode, resolved_matching_policy)` plus the CSV file's `(mtime_ns, size)` — `hvg_path` is no longer in EDAConfig (HVG gene panel for `gene_scope='hvg'` is sourced from the LORO cache's subject `gene_names` instead). First call computes (slow ComBat fit), subsequent calls load in seconds. Cache lives at `notebooks/cache/prepost/<hash>.pkl`. Same cache pattern at `notebooks/cache/genewise_kendall/<hash>.pkl` for the Kendall long table. **Out is reserved for true prediction artifacts; helper-level caches live under `notebooks/cache/`.**
20. **ipywidgets-based interactive single-gene scatter** in Section 4 of `eval_population_genewise.ipynb` — explicit dropdowns + `observe` callbacks (no `%matplotlib widget` / `ipympl` dependency). Inline static figures, one render path, one Output widget. `'all'` model option renders all 3 panels in one figure; specific model renders single panel.
21. Token-based font system in `eval_style.py`: `FONT_TOKENS = {xs, s, m, l, xl, xxl}`, `font_size("token±N")`, `set_font_scale(scale)`, `_resolve_fonts(defaults, override)`. Hero plotters declare `_DEFAULT_FONT_SIZES` and accept `font_sizes={...}` overrides. Single global rescale knob.
22. Tick visibility forced via `_TICK_RC` injected into `sns.set_theme(rc=...)`, re-asserted at module import, and `apply_tick_style(ax)` available for per-axes safety.
23. Canonical metric labels (`Pearson r`, `Spearman ρ`, `R²`, `RMSE`, `Kendall τ`) are routed through `format_metric_label` / `format_legend_label` — never hardcode metric strings in plot text.
24. Per-section mini-config cells in eval notebooks: only truly global knobs sit at the top; section-local knobs (LORO coverage, anchor model, percentile shading, illustrative subjects, specificity, gene panel, Kendall sublist) live immediately above their use sites. The genewise notebook adds a single `build_panel_view(gene_list_path, models=None)` helper closure to consolidate `make_prediction_eval_view` boilerplate across all sections.
25. The ongoing eval refactor moves stable functionality out of monolithic `results_eda.py` into focused `eval_*` modules. Do not build new refactor functionality in `results_eda.py` unless it is a short-lived compatibility bridge.
26. **Matching policy auto-detect**: `EDAConfig.matching_policy` defaults to `None`; `resolve_matching_policy(cfg)` reads the policy from any subject JSON under `<cache_root>/<gene_scope>/<model>/` (top-level or nested `config.matching_policy`), falling back to `'centroids'` for legacy caches that pre-date the field. PREPOST applies the resolved policy so its parcel_idx assignments agree with the LORO cache's. Switching `cfg.cache_root` is sufficient — no notebook config edits needed. The PREPOST disk-cache hash includes the resolved policy so different caches get distinct pickles.
27. **Eligibility ordering invariant**: `build_subject_eligibility` runs *after* `apply_gtex_ahba_matching_policy` in both `loro_cache.load_dataset` and `_load_expression`. The active matching policy IS the analysis bucketing, so `min_observed_parcels` is a *post-policy* floor: subjects with fewer than k distinct parcels under the chosen policy never enter the cache and are filtered identically by PREPOST. Practical implications under `centroids_and_volumes`: 5 boundary subjects (who carry both `brain - cerebellum` and `brain - cerebellar hemisphere` and otherwise have exactly 3 other tissues) drop from k=5 pre-collapse to k=4 post-collapse and are excluded at floor=5; lowering the floor to 4 admits 17 additional subjects (sparse-sampled regardless of policy). PREPOST and the cache builder must apply policy → eligibility in this order or the eligible sets will diverge silently.
28. **Dynamic sbatch array**: `run_loro_cache_array.sbatch` uses `--array=1-385%64` (the full GTEx subject pool ceiling). `scripts/run_loro_cache_batch.py:_pick_subjects` exits cleanly with `[skip-out-of-range] task_id=N n_eligible=M` when the array index exceeds the policy's eligible count. The same submission line works across all `MATCHING_POLICY` × `MIN_OBSERVED_PARCELS` combinations.
29. **EDAConfig slimmed to data-lineage essentials**: `csv_path`, `cache_root`, `gene_scope` are the primary uniqueness keys; `hvg_path` is removed (HVG list is sourced from the LORO cache's subject npz `gene_names` for `gene_scope='hvg'`). `plam_cache_dirname` (and its naive/dlam siblings) stay in the dataclass for variant-rank model dirs (e.g. `plam_rank4`) but are not surfaced in notebook CFGs unless overridden. The four eval notebooks (`eval_data`, `eval_population`, `eval_population_genewise`, `eval_singlesubject`) carry minimal CFG cells with `csv_path` + `cache_root` + `gene_scope` only, plus a Table-of-Contents markdown right after the title. Default sublist gene panel is `gtex_100hvg` everywhere a sublist is needed (Section 3 scatter, Kendall sublist, illustrative-subject view B); top-level analyses default to `gene_scope='allgenes'` with `EVAL_GENE_LIST_PATH=None` (all genes from the cached tables). **Cache-root naming convention**: `out/loro_subject_cache_c` (centroids) | `out/loro_subject_cache_cv` (centroids_and_volumes); the `_c`/`_cv` suffix encodes the matching policy that built the cache and is read back by `resolve_matching_policy(cfg)` from the subject JSONs at PREPOST time.
30. **Data layout: `data/raw/` vs `data/metadata/`**: raw expression CSVs (`gxp_samples.csv`) stay in `data/raw/`; gene lists and atlas info moved to `data/metadata/{gene_lists,atlas_info}/`. Module-level resolvers (`AtlasOverlapPaths`, `default_cerebellar_gene_list_paths`, `_resolve_eval_gene_path`, `_genes_from_cache`, the loro_cache HVG fallback list) try `data/metadata/...` first and fall back to `data/raw/...` so legacy paths still resolve.
31. **Pre-`gxp_samples.csv` GTEx gene filtering** now has an explicit audit path in `src/samples/{gtex,build}.py` and `docs/samples_builder.md`: restrict to brain samples, apply `SMRIN > 6`, compute GTEx cohort-wide TPM/read-count thresholds on the retained samples, then intersect that GTEx gene set with the AHBA preprocessing universe before any optional small manuscript panel is applied.
32. **`build_gxp_samples(...)` now owns the formal GTEx filter interface**: the builder accepts `gtex_sample_attributes_path`, `gtex_rin_threshold`, `gtex_tpm_threshold`, `gtex_reads_threshold`, `gtex_min_fraction`, `gtex_assay_freeze`, and `return_overlap_info`. Notebook exploration may still call `compute_gtex_ahba_overlap_report(...)` directly, but formal `gxp_samples.csv` creation should pass these arguments through the builder.
33. **GTEx duplicate subject IDs are now a hard error, not a silent average**: after parsing GTEx sample IDs down to subject IDs within a tissue file, duplicate subjects raise instead of being merged by mean. This preserves the invariant "one GTEx row per subject per tissue" and avoids unreviewed expression averaging in the canonical CSV.
34. **Atlas coordinate CSV resolution is schema-aware**: when building GTEx tissue coordinate lists, the builder requires `mni_x`, `mni_y`, `mni_z` for the Brodmann/S156 coordinate tables and skips incomplete local mirrors that only contain label metadata. This prevents late failures when `data/metadata/atlas_info` lacks full coordinate columns but the upstream `GeneEx2Conn_data/atlas_info` copy is complete.

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
- `data/raw/ahba_100hvg.txt`

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
  - `scripts/sbatch/run_loro_cache_single_subject.sbatch`
  - `scripts/sbatch/run_loro_cache_array.sbatch`
- Scripts are CPU-oriented; no required GPU path for current cache generation.
- Array jobs are one subject per task and support `GENE_SCOPE=hvg|allgenes`.
- Dynamic-rank PLAM launch example:
  - `MODEL_NAME=plam GENE_SCOPE=allgenes DYNAMIC_RANK=true PLAM_MAX_RANK=10 sbatch scripts/sbatch/run_loro_cache_array.sbatch`

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
2. For active eval-refactor work, inspect `context_packages/results_eda_refactor_plan.md`, then `eval_data.ipynb` / `eval_population.ipynb` / `eval_singlesubject.ipynb` / `eval_population_genewise.ipynb`, and the focused `src/eval_utils/eval_*` modules.
3. Use `src/eval_utils/results_eda.py` as a legacy reference, not a default implementation target.
4. Do not edit `src/eval_utils/results_eda_arxiv.py`; it is a backup snapshot.
5. Use `notebooks/coordinate_overlay_3d_mni.ipynb` (or repo-root `coordinate_assignment_3dmni.ipynb`) when working on spatial assignment or matching changes.
6. Treat `notebooks/` and the `results_*.ipynb` notebooks at repo root as historical/legacy unless intentionally reviving one. `notebooks/cache/` is for active helper-level caches and is fine to populate.

Last updated at: 2026-05-14
