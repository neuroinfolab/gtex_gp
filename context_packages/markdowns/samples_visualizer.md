# samples visualizer plan

Last touched: 2026-05-20

## Goal

Build a tensor-first visualization workflow that explains the raw GTEx and AHBA
sample datasets, then uses the same visual grammar for matched, harmonized,
imputed, and embedded model-space views.

The visualizer should support the full narrative:

1. original GTEx and AHBA sample tensors,
2. raw region-matched GTEx/AHBA tensors,
3. ComBat-harmonized matched tensors,
4. GTEx LORO imputed tensors,
5. GTEx full-brain imputed tensors,
6. UMAPs for the matched, harmonized, and full-brain stages.

The current GTEx tensor view is in a good default place and should be treated as
the reference composition for the next notebook split.

## Strategy Overview

This plan now has three coordinated visualization surfaces. They should share
the same source facades, lineage metadata, stage names, region labels, and
`eval_style` color conventions.

### 1. Tensors

Tensors are the primary data-lineage view. They show the actual
subject x region x gene matrices at each stage:

- raw native GTEx and AHBA inputs,
- raw region-matched GTEx/AHBA tensors,
- ComBat-harmonized matched tensors,
- strict LORO held-out and fused prediction tensors,
- full-brain GTEx imputation tensors.

The tensor surface answers: what values exist, what is missing, what is
observed, and what is an imputation target?

### 2. Embeddings

Embeddings are the variance-structure view. They reuse `TensorView` and
`JointTensorView` outputs, flattening them into well-defined sample matrices
for PCA and UMAP:

- region-wise rows: subject-region samples, features = genes,
- gene-wise rows: genes, features = subject-region samples.

PCA should be the first diagnostic for each embedding family because axes,
variance explained, and obvious batch separation are easier to audit. UMAP
then provides a nonlinear neighborhood view once the input matrix and coloring
semantics are validated.

### 3. On-Brain Visualizations

On-brain views are the spatial interpretation layer. They should be used after
tensor and embedding diagnostics identify a stage, component, cluster, or
outlier worth spatializing. They should show parcel-level summaries on the
target AHBA axis: observed support, macro-system distribution, PCA/UMAP scores
aggregated by parcel, imputation uncertainty, or per-parcel model deviation.

## Architecture

Three layers, in order. Every visualization downstream of the raw CSV flows
through them in this order. There are no parallel paths.

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Source loaders   (one per data origin)                       │
│    CSVSource        → native GTEx tissue / AHBA parcel space    │
│    PrePostSource    → raw-matched + ComBat-harmonized cubes     │
│    NpzCacheSource   → LORO truth/fused + full-brain predictions │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Adapter          build_tensor_view_from_cube(...)            │
│                     single chokepoint, source-agnostic          │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Joint + render   build_joint_tensor_view(...)                │
│                     plot_joint_tensor_voxels(...)               │
└─────────────────────────────────────────────────────────────────┘
```

### Data contracts (definitive)

`TensorView` fields — every view from any source carries the same set:

```
values                  (n_subj, n_region, n_gene)  float32
observed_mask           (n_subj, n_region)          bool
future_imputation_mask  (n_subj, n_region)          bool
subjects                list[str]
regions                 list[str]
genes                   list[str]
dataset                 'GTEx' | 'AHBA'
region_axis_kind        'gtex_tissue' | 'ahba_parcel' | 'target_parcel'
pipeline_stage          one of the stage values below
matching                dict  (always present for matched/target views)
```

`pipeline_stage` is a closed enum (`PIPELINE_STAGES`):

```
'native_tissue'   GTEx CSV, native tissue axis
'native_parcel'   AHBA CSV, parcel axis (full superset)
'raw_matched'     PREPOST raw_cube (pre-ComBat), target_parcel
'harmonized'      PREPOST harm_cube (post-ComBat), target_parcel
'loro_truth'      Npz loro_truth_subject_h — held-out truth at LORO parcels (sparse)
'loro_recon'      loro_fused_subject_h masked to loro_eval_mask — held-out preds only (sparse)
'loro_fused'      Npz loro_fused_subject_h — held-out preds + full-fit extrapolation (dense,
                  non-LORO cells flagged future_imputation)
'fullfit'         Npz fullfit_subject_h — full-fit (truth-anchored + extrapolated)
```

### Rendering semantic for `future_imputation_mask` (data-derived, not a mode)

A single mask drives two visual behaviors, chosen per cell by **data presence** —
no render-mode flag:

- `future_imputation_mask=True` **+ NaN value** → solid `future_imputation_rgba`
  gray. This is true padding (Phase 3 raw GTEx superset: AHBA-only parcels with
  no GTEx data).
- `future_imputation_mask=True` **+ finite value** → cmap(value) at
  `future_imputation_rgba` alpha. This is a real but non-anchored prediction
  (Phase 5 `loro_fused` extrapolation cells). The value stays visible; alpha
  flags it. Set alpha `1.0` for a fully-opaque first pass; titrate `< 1.0` to
  encode uncertainty.
- `future_imputation_mask=False` → normal cmap (or missing gray if unobserved).

Both renderers (`plot_gtex_tensor_voxels`, `plot_joint_tensor_voxels`) implement
this identically and both support `mask_render_mode='two_pass' | 'single_pass'`
(two-pass paints opaque cells first, then the flagged cells on top — cleaner
seam).

Voxel edges on the flagged (future/extrapolation) cells are controlled by
`future_edgecolor` / `future_linewidth`. Both default to `None`, meaning they
**inherit** the main `edgecolor` / `linewidth` — i.e. white borders everywhere,
consistent with the observed cells. Pass `future_edgecolor='none',
future_linewidth=0` to hide the borders on flagged cells.

### Global region display ordering

Display order is reversed at the view-construction layer only — matching/dedup
stays on the canonical order (so which GTEx tissue wins a collided parcel is
unchanged). Two shared helpers (`_display_region_axis`, `_hemisphere_display_order`)
are called by both the native path (`_resolve_region_ordered_dataset`) and the
cube path (`_apply_region_ordering`), so every tensor shares the axis convention:

- **Matched block** is reversed → the cerebellum-matched parcel leads, frontal
  cortex trails (was the reverse).
- **AHBA remaining block** is hemisphere-ordered: `LH_*` parcels first
  (reverse-sorted), then `RH_*` (reverse-sorted), then non-hemisphere parcels
  (cerebellar, no `LH`/`RH` prefix) pushed to the end.

`matching` dict — always the same shape:

```
{
  'region_ordering':     str   (informational; not part of compat fingerprint)
  'policy':              str
  'hemi_mode':           str
  'gtex_rep_mode':       str
  'gtex_hemi_mode':      str
  'collapse_cerebellum': bool
  'gtex_to_ahba':        dict[str, str]   (full 11 pairs, GTEx-rank ordered)
}
```

`matching_metadata_compatible(a, b)` compares the fingerprint
(`policy`/`hemi_mode`/`gtex_rep_mode`/`gtex_hemi_mode`/`collapse_cerebellum`)
plus the `gtex_to_ahba` dict. `build_joint_tensor_view` enforces this on every
joint stack.

### Source facades (three, one per origin)

Each facade is a thin wrapper around `build_tensor_view_from_cube`. No facade
contains data-shaping logic — only "where does the cube come from."

```python
build_native_tensor_view(
    samples,            # CSV path or DataFrame
    *,
    dataset,            # 'GTEx' | 'AHBA'
    region_ordering,    # 'dataset' | 'region_matched' | 'region_matched_superset'
    matching_policy, matching_policy_hemi_mode,
    gtex_rep_mode, gtex_hemi_mode, collapse_cerebellum,
    gene_panel, n_subjects, n_regions, n_genes, random_seed,
) -> TensorView   # pipeline_stage in {'native_tissue', 'native_parcel'}

build_combat_tensor_view(
    prepost,
    *,
    dataset, stage,     # stage in {'raw_matched', 'harmonized'}
    cfg, matching_context,
    region_ordering,    # 'target_parcel' | 'region_matched' | 'region_matched_superset'
    gene_panel, n_subjects, n_regions, n_genes, random_seed,
) -> TensorView

build_prediction_tensor_view(
    cache_root, model_name,
    *,
    dataset, stage,     # stage in {'loro_truth','loro_recon','loro_fused','fullfit'}
    cfg, matching_context,
    region_ordering,    # same set as above
    gene_panel, n_subjects, n_regions, n_genes, random_seed,
) -> TensorView
```

`region_ordering` for the two cube facades shares one helper
(`_apply_region_ordering`): `target_parcel` is the bare `parcel_idx` order;
`region_matched` keeps the matched 11 only; `region_matched_superset` puts the
matched 11 first then the remaining parcels (`region_axis_kind` flips to
`ahba_parcel`, and GTEx unmatched columns are flagged future_imputation). The
cube reader `stack_subject_cubes(...)` is fully implemented (per-subject
`.npz` → `(n_subj, n_parc, n_gene)`, with an optional `extra_mask_field` used
to load `loro_eval_mask` for `loro_recon` sparsifying and `loro_fused` flagging).

The facades share argument naming wherever the meaning is the same. Anything
that differs (e.g. CSV vs cache_root) is explicit in the signature.

### Cross-stage helper (one)

```python
expand_to_ahba_superset(view, *, full_ahba_axis) -> TensorView
```

Pads the parcel axis from the matched 11 to the full AHBA superset and sets
`future_imputation_mask=True` on the padded columns. Used when a single figure
must show multiple pipeline stages on the same axis length.

### What's deliberately out of scope

- HVG path. `gene_scope='allgenes'` is the only supported scope. Gene
  sub-panels are applied post-cube via `gene_panel=...`.
- `plot_sampled_tensor(..., dataset='combined')` convenience wrapper. Use the
  three explicit facades + `build_joint_tensor_view`.
- Auto-detection of matching policy at view-build time. The cfg's policy is
  authoritative; mismatches raise.
- float64 cubes. All cubes are float32 from the source-loader boundary onward.
- Per-call CSV re-reads. `MatchingContext.from_cfg(cfg)` is cached per
  cfg-fingerprint.
- Multiple renderers. `plot_joint_tensor_voxels` is the only joint renderer;
  `plot_gtex_tensor_voxels` is the only single-view renderer.

## Notebooks (current, shipped)

Three live notebooks, one per pipeline layer. Each shows standalone +/or joint
views in a `SMALL` toggle (fast iteration vs full-density figure). The `SMALL`
preset is identical across notebooks (30 GTEx / 6 AHBA / 100 genes / 75 regions
/ DPI 300) for A/B comparability.

- `eval_gxp_samples.ipynb` — **raw** (Tier A, CSV via `build_native_tensor_view`)
  - standalone GTEx, standalone AHBA, then the raw joint tensor
  - region frame `region_matched_superset`; GTEx unmatched parcels translucent
- `eval_gxp_samples_combat.ipynb` — **ComBat** (Tier B, PREPOST)
  - the joint tensor pre-ComBat (`raw_matched`) and post-ComBat (`harmonized`)
  - harmonized joint uses `normalization='shared'` (one biological scale)
- `eval_gxp_samples_predictions.ipynb` — **LORO + full-fit** (Tier B, npz cache)
  - standalone + joint (vs AHBA harmonized reference) for, in order:
    `loro_truth` → `loro_recon` → `loro_fused` → `fullfit`.
    Each tensor has its own cell (no plot loops).
  - default `MODEL_NAME='dlam'`; extrapolation alpha `1.0` (opaque first pass)

Archived reference notebooks (native single-dataset input space, CSV-driven):
`notebooks/arxiv/eval_gtex_gxp_samples.ipynb`,
`notebooks/arxiv/eval_ahba_gxp_samples.ipynb`. These are the only place the
native GTEx 12-tissue axis is rendered (PREPOST collapses GTEx into target
parcels, so it cannot reproduce that view).

## Current GTEx Baseline

Current defaults from `eval_gtex_gxp_samples.ipynb` (the split raw-GTEx
notebook). These are the live reference; the AHBA notebook mirrors the same
parameter surface with smaller subject/region/gene counts.

```python
CSV_PATH = Path("data/raw/gxp_samples.csv")
ACTIVE_GENE_LIST = "richiardi2015"  # also: allgenes_stable_r0.2

REGION_ORDERING = "region_matched_superset"  # 'dataset' | 'region_matched' | 'region_matched_superset'
MATCHING_POLICY = "centroids_and_volumes"    # 'centroids' | 'centroids_and_volumes'
MATCHING_POLICY_HEMI_MODE = "force_left"     # 'default' | 'force_left'
GTEX_REP_MODE = "centroid"
GTEX_HEMI_MODE = "mirror_left"

N_SUBJECTS = 75
N_GENES = 100
N_REGIONS = None

MIN_REGIONS_PER_SUBJECT = 5
RANDOM_SEED = 42
MISSING_ALPHA = 1.0

FIGSIZE = "auto"
AXES_BBOX = "auto"
COLORBAR_BBOX = "auto"
BOX_ASPECT = "compressed_data"
BOX_ZOOM = 1
DPI = 400
TITLE_PAD = 0.0
AXIS_ASSIGNMENT = ("gene", "region", "subject")
SHOW_GRID = False
ELEV = 20.0
AZIM = 24.0

SUBJECT_TICK_STEP = 3
GENE_TICK_STEP = 5
REGION_TICK_STEP = 1
TICK_LABEL_PAD = -3.0

SUBJECT_LABEL_MODE = "full"
GENE_LABEL_MODE = "full"

SHOW_X_AXIS_LABEL = True
SHOW_Y_AXIS_LABEL = None
SHOW_Z_AXIS_LABEL = None
SHOW_AXIS_LABELS = False

SHOW_SUBJECT_TICKLABELS = True
SHOW_REGION_TICKLABELS = True
SHOW_REGION_LABEL_COLORS = False
SHOW_GENE_TICKLABELS = False

SHOW_AXIS_LINES = True
SHOW_TICK_LINES = True
SHOW_LEGEND = False
```

Note: notebooks and render sbatches should default to `data/raw/gxp_samples.csv`.
Use notebook-level subject/gene/region sampling controls for quick iteration.

Current implementation notes:

- `src/eval_utils/eval_samples.py` owns the tensor helpers and voxel renderer.
- raw GTEx and AHBA tensors now share the same `TensorView` contract.
- `plot_sampled_tensor(..., dataset="GTEx"|"AHBA")` is the notebook-facing
  entry point for dataset-specific raw sample tensors.
- `region_ordering` controls how the region axis is chosen:
  - `"dataset"` keeps each dataset's own full native ordering.
  - `"region_matched"` uses the GTEx-to-AHBA matching policy to build paired
    GTEx/AHBA region axes in the same order.
  - `"region_matched_superset"` keeps matched AHBA parcels first, then appends
    the remaining AHBA parcels. For AHBA this surfaces observed parcels on the
    full axis. For GTEx this pads the unmatched AHBA-only positions with NaN
    values and flags them in `future_imputation_mask` (see Joint Tensor View
    below), so GTEx now shares the full AHBA region frame instead of
    collapsing to the matched prefix.
- matched orderings are sampled by ordered prefix rather than random region
  draw so the shared matched block stays at the start of the axis.
- `matching_policy` defaults to `"centroids_and_volumes"` to match the current
  modeling pipeline; `"centroids"` remains available for pure Euclidean
  matching.
- `matching_policy_hemi_mode` defaults to `"default"`; `"force_left"` restricts
  centroid candidates to LH-prefixed parcels and uses the left BA10 volume
  prior for generic cortex.
- `FIGSIZE`, `AXES_BBOX`, `COLORBAR_BBOX`, and `BOX_ZOOM` support `"auto"`.
- `BOX_ASPECT="compressed_data"` expands the semantic region axis by `1.3`.
- y/region and z/subject tick labels have smaller tensor-specific defaults.
- bare gene-list names resolve both top-level gene-list files and
  `data/metadata/gene_lists/allgenes_stability_dk/*.csv`.
- the title reports actual plotted dimensions after sampling, not just
  requested dimensions.
- colorbar label is dataset-aware: `log1p(TPM)` for GTEx and
  `Microarray Intensity` for AHBA.
- root sbatch wrappers render long-running PNGs through the same parameter
  surface used by the notebooks. Presets per dataset (full / repr / small):
  - `render_gtex_gxp_tensor_{full,repr,small}.sbatch`
  - `render_ahba_gxp_tensor_{full,repr,small}.sbatch`
  - `_full` matches manuscript-density renders (e.g. GTEx: 200 subjects /
    225 genes / DPI 500; AHBA: 6 subjects / 225 genes / DPI 500, full region
    axis, ticks off for hero figure use).
  - `_repr` matches the split-notebook representative defaults (GTEx: 75
    subjects / 100 genes / full AHBA superset axis, ticks on; AHBA: 6
    subjects / 100 genes / 75 regions, ticks on including genes).
  - `_small` is the smoke/illustrative preset (GTEx: 30 subjects / 50 genes /
    `region_matched` ordering with subject ticks; AHBA: 6 subjects / 50 genes
    / 60 regions on `region_matched_superset` with subject ticks).
  - default outputs write to `out/tensor_renders/<name>_<preset>.png`.

## Tensor Families

### Raw Explanatory Tensors

These come directly from `data/raw/gxp_samples.csv`.

GTEx raw tensor:

- dataset: `GTEX`
- axes by default: `x=gene`, `y=native GTEx tissue`, `z=subject`
- region axis kind: `gtex_tissue`
- missingness unit: whole subject x tissue gene vector

AHBA raw tensor:

- dataset: `AHBA`
- axes by default: `x=gene`, `y=AHBA parcel`, `z=subject`
- region axis kind: `ahba_parcel` or `target_parcel`, depending on source rows
- same `TensorView` layout as GTEx: subjects x regions x genes
- expected shape pattern: fewer subjects than GTEx, many more regions

### Cross-Dataset Tensors

These belong in `eval_gxp_samples.ipynb`.

Matched shared-region view:

- GTEx and AHBA restricted to the regions/parcels linked by the matching path.
- The point is to show the directly comparable support before harmonization.
- This should line up with the same matching semantics used later by the model
  path, not an ad hoc display-only join.
- tensor control: `region_ordering="region_matched"`.
- current default matching policy: `matching_policy="centroids_and_volumes"`.

Outer-joined region view:

- Both datasets share one ordered region axis defined by AHBA's superset:
  matched parcels first (current 11), then the remaining AHBA parcels.
- Matched columns/regions are aligned.
- GTEx's unmatched AHBA-only positions are padded empty and carry
  `future_imputation_mask=True` so the renderer can draw them translucent;
  AHBA's own unobserved subject x parcel cells remain normal solid-gray
  missingness.
- This is the visual bridge from sparse GTEx tissue observations to full AHBA
  target-space completion.
- tensor control: `region_ordering="region_matched_superset"` for both
  `dataset="GTEx"` and `dataset="AHBA"`; the convenience entry point is
  `dataset="combined"` (see Joint Tensor View).

### Joint Tensor View

The combined plot stacks GTEx and AHBA along the subject axis on a shared
region x gene frame. Construction is layered into three reusable pieces so
the same surface can later host ComBat-harmonized, LORO, and full-brain
imputed pairs.

Data contract additions in `eval_samples.py`:

- keep `TensorView` as the single-dataset contract; add
  `future_imputation_mask` (subjects x regions) alongside `observed_mask` for
  superset-padded GTEx so the renderer can distinguish translucent
  imputation targets from solid-gray AHBA missingness.
- add `JointTensorView` for two-dataset stacks:
  - `values`, `observed_mask`, `future_imputation_mask`
  - `subjects`, `subject_datasets`, `dataset_slices`
  - `regions`, `genes`
  - `matched_region_count`
  - `source_views` (the original per-dataset `TensorView`s for lineage)

Construction layers:

1. dataset tensor construction: `build_sampled_tensor(..., dataset="GTEx"|"AHBA",
   region_ordering="region_matched_superset")` for each dataset, using
   identical `gene_panel`, `random_seed`, `matching_policy`,
   `matching_policy_hemi_mode`, `gtex_rep_mode`, `gtex_hemi_mode`.
2. shared-frame joint construction: `build_joint_tensor_view(gtex_view,
   ahba_view, *, stack_order=("GTEx", "AHBA"))` validates identical gene list,
   identical region list, compatible `region_axis_kind`, and a known matched
   prefix length, then concatenates subjects and records `dataset_slices`.
3. rendering: `plot_joint_tensor_voxels(joint_view, ...)` consumes only a
   valid `JointTensorView`; it does not touch raw samples. This is the
   reusable surface for harmonized/predicted pairs.

Convenience entry point:

- `plot_sampled_tensor(..., dataset="combined")` internally calls the modular
  path and returns `(fig, ax, joint_view, selection_summary)`.

Renderer behavior for the joint view:

- shared x = genes, shared y = AHBA superset region frame,
  stacked z = subjects from both datasets (GTEx block then AHBA block by
  default, controllable via `stack_order`).
- GTEx superset-padded AHBA-only regions render translucent (driven by
  `future_imputation_mask`); AHBA unobserved subject x region cells stay
  solid gray.
- subject tick labels stay dataset-prefixed (`GTEx-*` and `AHBA-*`).
- optional thin dataset separator between the two subject blocks.
- raw GTEx (`log1p(TPM)`) and AHBA (microarray intensity) are not on a
  shared biological scale; the raw combined plot avoids implying one — first
  implementation uses one colormap with per-dataset normalization or two
  colorbars. Later harmonized/ComBat joint tensors can collapse to a single
  shared colorbar.

### Model-Space Tensors

These come from the matching / harmonization / cache path.

Raw matched tensors:

- source path includes `read_expression_subset(...)`,
  `build_target_parcels(...)`, `map_gtex_to_target(...)`, and matching-policy
  application.
- useful for showing GTEx/AHBA geometry before ComBat.

ComBat-harmonized tensors:

- source: `eda_core.prepare_pre_post_harmonization_cached(...)`
- key fields include `raw_cube`, `harm_cube`, `obs_mask`,
  `gtex_eligible_raw`, `gtex_eligible_h`, `ahba_raw`, `ahba_h`, and
  `target_meta`.
- UMAPs should be paired with these tensor views to show domain alignment.

LORO imputation tensors:

- source: subject `.npz` caches under `out/loro_subject_cache...`
- key arrays: `loro_truth_subject_h`, `loro_fused_subject_h`,
  `loro_eval_mask`
- best used as strict evaluation views.

Full-brain imputation tensors:

- source: `fullfit_subject_h`
- complete GTEx subject maps across target parcels
- UMAPs should show where full GTEx completions sit relative to AHBA
  harmonized structure.

## Embedding Analysis Plan

Embedding analysis should be built on top of the shipped tensor facades, not as
a parallel data-loading path. Every embedding payload should record:

- source view(s): `TensorView` or `JointTensorView`,
- `pipeline_stage`,
- feature axis (`genes` or `subject_region_samples`),
- row axis (`subject_region` or `gene`),
- dataset labels,
- region / parcel labels,
- macro-system labels,
- observation and imputation masks,
- preprocessing applied before PCA/UMAP.

### Embedding Input Matrices

Region-wise embedding matrix:

- row = one subject-region observation or prediction,
- columns = genes,
- default use: study sample-level variance structure across regions/datasets,
- expected row count: GTEx has many more rows than AHBA in shared-region views
  because GTEx contributes many subjects per sampled region.

Gene-wise embedding matrix:

- row = one gene,
- columns = flattened subject-region samples,
- default use: study how genes organize by cross-sample/cross-region expression
  pattern,
- coloring is defined from the pre-stage gene summary, typically mean
  expression per gene with a viridis gradient,
- later extension: repeat this view over full-fit/full-brain region axes.

### Stage Comparisons

Primary stage sequence:

1. `raw_matched` / pre-ComBat,
2. `harmonized` / post-ComBat,
3. `loro_recon` / `loro_fused` strict LORO imputation views,
4. `fullfit` / full-fit GTEx completion.

For each sequence, keep the sampled genes, regions, subjects, and random seed
fixed when possible so stage-to-stage movement is interpretable.

### Within-GTEx Variance Retention

Purpose: verify that harmonization and imputation do not erase meaningful
within-GTEx biological/spatial structure.

Region-wise within-GTEx view:

- input: GTEx-only `TensorView` at `raw_matched`, `harmonized`, and LORO stages,
- row = subject-region,
- columns = genes, often all genes (~11k) or a stable gene panel,
- color = region label or macro-system,
- expected result: region structure should remain visible post-ComBat and after
  LORO, not collapse into one undifferentiated GTEx cloud.

Gene-wise within-GTEx view:

- input: GTEx-only views at the same stages,
- row = gene,
- columns = flattened subject-region samples,
- color = pre-stage mean expression per gene using viridis,
- expected result: broad gene-expression landscape should be retained while
  stage-specific distortions or compression become visible.

### Cross-Dataset Batch-Correction Diagnostics

Purpose: evaluate whether AHBA-vs-GTEx separation decreases after harmonization
while region/macro-system organization becomes more prominent.

Shared-region region-wise view:

- input: `JointTensorView` restricted to shared/matched regions first,
- row = subject-region sample,
- columns = genes,
- plot PCA first, then UMAP,
- color = region label using global `eval_style` definitions,
- marker = dataset (`GTEx` small circles, `AHBA` larger triangles),
- expected pre-ComBat pattern: stronger dataset-wise separation,
- expected post-ComBat pattern: weaker dataset separation and more
  region-wise/macro-system clustering across datasets.

Stage progression:

- pre-ComBat (`raw_matched`) -> post-ComBat (`harmonized`) -> LORO imputed
  (`loro_recon` / `loro_fused`),
- use the same visual grammar and sampled support across panels,
- optionally draw low-alpha arrows or paired centroids only after static panels
  are interpretable.

### Full-Brain Imputation Embeddings

Purpose: evaluate whether full-brain GTEx completions occupy a plausible
distribution relative to AHBA across the full target parcel axis.

Full-brain region-wise view:

- GTEx rows: subject x all target parcels from `fullfit_subject_h`,
- AHBA rows: AHBA subject x sampled/available parcels from the harmonized AHBA
  reference,
- columns = genes,
- plot PCA first, then UMAP,
- dataset markers remain visible despite GTEx row dominance:
  - GTEx = smaller circles,
  - AHBA = larger triangles,
- use balanced or stratified downsampling when row-count imbalance hides AHBA.

Coloring for full-brain parcel views should be more structured than the
matched-region palette:

- cortical parcels: yellow/orange/red family,
- subcortical parcels: blue/purple family,
- cerebellar parcels: green gradients,
- implementation should build from `eval_style` rather than adding ad hoc
  notebook palettes.

### PCA Before UMAP

For each embedding family, PCA is the first required plot:

- expose variance explained,
- make dataset separation and major spatial axes auditable,
- verify scaling/centering decisions before nonlinear embedding,
- provide a deterministic comparison surface.

UMAP should then use the PCA-validated matrix. Recommended first-pass controls:

- fixed `random_state`,
- same sample matrix across compared stages,
- documented feature scaling,
- optional PCA pre-reduction when using all genes,
- side-by-side panels rather than overlaid stage encodings for the first pass.

## On-Brain Visualization Strategy

On-brain plots should be used to translate tensor/embedding findings back to
the AHBA target parcel axis. They are not the first diagnostic; they are the
spatial follow-up once PCA/UMAP identifies a pattern.

Candidate on-brain payloads:

- per-parcel observed support and future-imputation support,
- macro-system and hemisphere coverage,
- per-parcel PCA score centroids from region-wise embeddings,
- per-parcel UMAP cluster membership or neighborhood density,
- pre-to-post ComBat displacement summarized by parcel,
- LORO/fullfit deviation from AHBA harmonized reference,
- uncertainty or imputation-mask summaries.

Visualization rules:

- keep target parcel order and labels consistent with `TensorView.regions`,
- use the same macro-system color families as embedding plots,
- summarize subject-level quantities to parcel-level statistics explicitly
  (mean, median, variance, or quantile),
- keep GTEx observed, LORO-imputed, and fullfit-extrapolated support visually
  distinguishable.

## Implementation Plan

### Phase 1: Split GTEx notebook

Status: implemented as `eval_gtex_gxp_samples.ipynb`.

Keep:

- current GTEx defaults,
- GTEx tissue tensor builder,
- sampled gene-list flow,
- observation mask handling,
- current 3D tensor renderer.

Remove from this notebook:

- AHBA-specific planning cells,
- cross-dataset comparison cells,
- model-space/cache views.

Outcome:

- a clean raw GTEx input tensor notebook that can be rerun quickly with
  alternate subject/gene counts.

### Phase 2: Create AHBA raw notebook

Status: implemented as `eval_ahba_gxp_samples.ipynb`.

AHBA builder support in `eval_samples.py`:

- filter `gxp_samples.csv` to AHBA rows,
- use the same subject x region x gene `TensorView` contract as GTEx,
- sample fewer subjects,
- sample many more regions/parcels,
- reuse the same gene-panel resolver and gene sampling path,
- return the existing `TensorView` contract.

Expected AHBA defaults should differ from GTEx:

- smaller `N_SUBJECTS`,
- all available AHBA regions by default,
- same `N_GENES` and deterministic gene-list handling as GTEx,
- `AXIS_ASSIGNMENT=("gene", "region", "subject")`,
- larger region tick interval for full-parcel views.

### Phase 2b: Batch rendering wrappers

Status: implemented.

Use the root-level sbatch wrappers when tensor rendering is too slow for the
notebook kernel. Three presets per dataset:

- `render_gtex_gxp_tensor_full.sbatch` / `render_ahba_gxp_tensor_full.sbatch` —
  manuscript-density renders (large subject/gene counts, ticks off).
- `render_gtex_gxp_tensor_repr.sbatch` / `render_ahba_gxp_tensor_repr.sbatch` —
  representative split-notebook defaults (mid-density, ticks on).
- `render_gtex_gxp_tensor_small.sbatch` / `render_ahba_gxp_tensor_small.sbatch` —
  small smoke-test / illustrative renders (low subject/gene counts, ticks on
  for subject only; GTEx uses `region_matched` to focus on the shared 11).

All wrappers expose the same control surface as the notebooks through
environment variables and write images under `out/tensor_renders/<name>_<preset>.png`
with a sidecar `.selection.csv` describing the sampled subjects, regions, and
genes.

### Phase 3: Joint ground-truth tensor in `eval_gxp_samples.ipynb`

Refactor `eval_gxp_samples.ipynb` into the joint-tensor / comparison notebook.
First milestone is the raw ground-truth combined view; later milestones reuse
the same `JointTensorView` surface for harmonized and predicted pairs.

Stage 3a — region_matched_superset semantics for GTEx:

- GTEx with `region_ordering="region_matched_superset"` now returns the full
  AHBA superset region axis instead of collapsing to the matched prefix.
- matched positions get the mapped GTEx values; unmatched AHBA-only positions
  are padded (NaN values) and marked in `future_imputation_mask`.
- `TensorView` gains `future_imputation_mask`; existing single-dataset
  renderers ignore it by default, the joint renderer uses it for translucent
  styling.

Stage 3b — joint tensor construction:

- add `JointTensorView` dataclass and `build_joint_tensor_view(gtex_view,
  ahba_view, *, stack_order=("GTEx", "AHBA"))`.
- validate identical genes, identical regions, compatible
  `region_axis_kind`, and a recorded `matched_region_count`.
- concatenate subjects along the subject axis and record per-dataset slices.

Stage 3c — joint renderer:

- add `plot_joint_tensor_voxels(joint_view, ...)` reusing the current voxel
  styling.
- translucent GTEx imputation-target mask, solid AHBA missing mask, optional
  dataset separator on the subject axis.
- per-dataset normalization (or twin colorbars) for the raw combined view.

Stage 3d — convenience entry point:

- `build_sampled_joint_tensor(...)` wires the modular path end-to-end.
- `plot_sampled_tensor(..., dataset="combined")` wraps that and returns
  `(fig, ax, joint_view, selection_summary)`.

Stage 3e — notebook wiring (test surface):

`eval_gxp_samples.ipynb` becomes the joint-tensor test surface:

1. build GTEx view with `region_matched_superset`, `force_left`.
2. build AHBA view with the same gene list, seed, region ordering, and
   matching policy.
3. assert same regions and same genes.
4. build `JointTensorView`.
5. plot combined tensor via the modular call.
6. once stable, swap to the convenience `dataset="combined"` call.

The single-dataset notebooks (`eval_gtex_gxp_samples.ipynb`,
`eval_ahba_gxp_samples.ipynb`) stay focused on their own dataset views.

Later comparison views reusing the same `JointTensorView` surface:

1. raw matched GTEx/AHBA tensors (matched-only prefix),
2. ComBat harmonized matched tensors,
3. LORO truth vs fused predictions,
4. `fullfit` GTEx imputation,
5. UMAPs for stages 2, 3, and 5.

### Phase 4: Unified refactor — SHIPPED

The Architecture above is implemented in `src/eval_utils/eval_samples.py`:

- `TensorView.pipeline_stage` (closed enum `PIPELINE_STAGES`); `values` cast to
  float32 in `__post_init__`; `_validate_pipeline_stage` enforces the enum.
- Tier B adapter: `MatchingContext.from_cfg(cfg, prepost)` (lru-cached pair
  table), `build_ahba_cube_from_prepost(...)`, `build_tensor_view_from_cube(...)`
  (single chokepoint).
- Three facades: `build_native_tensor_view` (CSV; renamed from the old
  `build_sampled_tensor`, no shim), `build_combat_tensor_view` (PREPOST),
  `build_prediction_tensor_view` (npz cache).
- `stack_subject_cubes(...)` fully implemented (per-subject npz → cube), with
  `extra_mask_field` for `loro_eval_mask`.
- `plot_joint_tensor_voxels` gained `normalization='per_dataset_minmax' |
  'shared'`. Both renderers gained `future_imputation_rgba`, `mask_render_mode`,
  and the data-derived future-imputation rendering (see Architecture).
- `_apply_region_ordering` shared by both cube facades for `target_parcel` /
  `region_matched` / `region_matched_superset`.
- `expand_to_ahba_superset(...)` cross-stage helper.

### Phase 5: Predictions — SHIPPED

`build_prediction_tensor_view` + `eval_gxp_samples_predictions.ipynb` cover the
four prediction stages (`loro_truth`, `loro_fused`, `loro_fused`, `fullfit`)
in both standalone and joint-vs-AHBA-reference form. The `loro_fused` stage is
the "all predicted" view: same cube as `loro_fused` but with non-LORO cells
flagged via `loro_eval_mask` so extrapolation is visually distinguishable. No
new renderer, no new joint logic — pure application of the Architecture.

## Design Constraints

1. Keep values and masks separate.
   - `values` is expression.
   - `observed_mask` is observation support.
   - translucent/unmatched regions should be driven by mask/lineage metadata,
     not by overloading expression values.

2. Sample late.
   - subset subjects, genes, and regions before materializing display tensors.

3. Preserve raw-space versus model-space lineage.
   - raw GTEx tissue space is not target-parcel space.
   - raw AHBA parcel space is not a GTEx tissue axis.
   - matched/harmonized/cache tensors must declare their region axis kind.

4. Avoid new layout knobs unless repeated usage proves they are needed.
   - current auto layout plus tick intervals should remain the main control
     surface.

5. Every tensor should carry explicit metadata:
   - dataset,
   - expression space,
   - region axis kind,
   - value source,
   - mask source,
   - matching lineage if applicable.

## Immediate Next Step

Phases 1–5 are shipped: raw, ComBat, LORO, and full-fit tensors all render in
both standalone and joint form across the three live notebooks, on the unified
`TensorView` / `build_tensor_view_from_cube` / two-renderer architecture.

Open follow-ups (not blocking):

1. Titrate `future_imputation_rgba` alpha in the predictions notebook to encode
   prediction uncertainty (currently opaque `1.0` first pass).
2. Add a regression test exercising the cube facades (`build_combat_tensor_view`,
   `build_prediction_tensor_view`, `stack_subject_cubes`, `loro_fused` masking)
   — `tests/test_eval_samples.py` currently only covers `build_native_tensor_view`.
3. Extend the first-pass embedding layer (`src/eval_utils/eval_embeddings.py`,
   `eval_embeddings.ipynb`) beyond GTEx ground-truth / post-ComBat / LORO-recon
   into joint AHBA-GTEx and fullfit stages while preserving the shared matrix
   contract. Use PREPOST directly where it is the clean source; use `TensorView`
   where the view is cache- or visualizer-derived.
