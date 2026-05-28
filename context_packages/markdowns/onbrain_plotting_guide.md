# On-Brain Plotting in `gtex_gp` — Handoff Brief

Briefing for agents creating on-brain visualizations (nilearn dots + yabplot cortical/subcortical
surfaces) for GTEx/AHBA TensorViews in this repo. Reference implementations:
`eval_pls_gradients_onbrain.ipynb` (repo root) and `src/eval_utils/eval_onbrain.py` +
`eval_pls_gradients.py`.

## Environment & execution
- **Env:** `vformer_env` singularity overlay (Python 3.12); nilearn/yabplot/pyvista/vtk live there. Enter with:
  ```bash
  singularity exec --nv --overlay /scratch/asr655/envs/vformer_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
    'source /ext3/miniforge3/etc/profile.d/conda.sh && conda activate; <cmd>'
  ```
  Use `:ro` for read/plot, `:rw` only to install (single-writer; the lock holder shows as process
  **`starter`**, not `singularity` — find via `lsof`/`fuser`, `kill -TERM` it).
- **Headless render:** set `MPLBACKEND=Agg` (or `module://matplotlib_inline.backend_inline` to capture
  inline images), `PYVISTA_OFF_SCREEN=true`, `pv.OFF_SCREEN=True`. OSMesa works; ignore the
  `bad X server connection` warning. No xvfb needed; `pv.start_xvfb()` doesn't exist in pyvista 0.48.
- **Repo imports:** `gtex_gp` is NOT pip-installed in `vformer_env`, so `from src...` needs the **repo
  root on sys.path** — run with cwd=repo root or
  `export PYTHONPATH=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp`. Relative data paths
  (`data/raw/...`, `out/...`) also resolve from repo root.

## Core modules (already built — reuse these)
- **`src/eval_utils/eval_onbrain.py`** — the rendering toolkit. Notebooks are config-first: toggles +
  one-line calls into here.
  - `subject_gene_values(view, subject, gene)`, `slice_value_dict(...)`, `shared_clim(arrays, pct)`,
    `region_coords(prepost, regions)`, `search(items, substr)`, `build_panels(...)`.
  - `render_nilearn_row(panels, gene, coords, ...)` — nilearn glass-brain dot row.
  - **`BrainSurfaceRenderer(atlas_cache='data/brain_atlas_4s')`** — single-PyVista-scene cortex+subcortex
    renderer (yabplot can't overlay cortex-data + subcortex-data in one call, so this ports the
    collaborator's `BrainRenderer`). `.render(value_by_label, scope, *, vmin, vmax, cmap, zoom, camera,
    hemispheres)`. Scopes in `SCOPE_CFG`: `cortical`, `subcortical_cerebellar`, `joint`. `hemispheres` ∈
    `left|right|both`. Cameras: `left_lateral|left_medial|right_lateral|right_medial|superior|anterior`.
    Loads LH+RH midthickness + all subcortical VTKs once.
  - `render_scope_row(renderer, scope, panels, gene, ...)`.
- **`src/eval_utils/eval_pls_gradients.py`** — the PLS-gradient analysis; a worked example of a full
  brain+scatter figure (see formatting patterns below).
- **`src/latent/pls.py`** — `fit_pls_basis(X, Y)` (returns a reusable projector: keeps the sklearn
  estimator + sign flips) and `project_pls_scores(basis, X_new)`. The original `fit_subject_pls`
  doesn't expose a forward transform.

## Data conventions (critical for correctness)
- **TensorViews:** `build_combat_tensor_view` (AHBA, stage `harmonized`) and
  `build_prediction_tensor_view` (GTEx, stages `loro_truth|loro_recon|loro_fused|fullfit`). `.values` is
  `(subjects, regions, genes)`, plus `.subjects/.regions/.genes/.observed_mask`. Pass the same
  `MatchingContext` + `tier_b` so all views share one region/gene frame (assert `.regions`/`.genes` equal).
- **Region ↔ brain:** `prepost['target_meta']` has `parcel_idx, tissue_or_parcel, coord_x/y/z`. Parcel
  labels (`LH_Vis_1`, `RH_Default_*`, `Cerebellar_Region7`, `LH-GPe`…) **match the vendored atlas labels
  exactly** — render by label, no coordinate work needed. For nilearn, MNI coords come from
  `coord_x/y/z` reindexed to `view.regions`.
- **Vendored atlas:** `data/brain_atlas_4s/` (5.9 MB: `cortical/atlas.csv` vertex→id + `atlas.txt` LUT,
  54 subcortical `.vtk`), copied from the collaborator's 4S156 cache. Rebuilding the cortical projection
  needs Connectome Workbench (`wb_command`), which is absent — reuse the prebuilt cache.
- **Hemisphere:** matching is `force_left`/`mirror_left`, so GTEx coords collapse left. For **yabplot
  surfaces** you do NOT need coord-mirroring — the atlas `LH_`/`RH_` labels place parcels anatomically,
  so `hemispheres='both'` is the correct full brain. (DLAM's
  `eval_dlam_alignment.display_coords(undo_force_left=True)` mirroring is only for the nilearn coord path:
  flips display x by `target_meta.hemisphere` L→-|x|, R→+|x|.)

## Notebook conventions (follow these)
- **Config-first:** numbered markdown sections, each opening with a toggle block, then one call. All heavy
  logic lives in `src/eval_utils/`, not inline.
- **Autoreload** at top: `%load_ext autoreload` / `%autoreload 2`.
- **Derive config from the cache, don't hardcode hparams:** `pg.build_cfg_from_cache(cache_root, model,
  gene_scope)` reads a per-subject JSON sidecar's `config` and rebuilds `EDAConfig` (matching policy,
  hemi modes, rep mode, `n_comp_target`…). Prevents drift between plotting and the model fit.
  **Gotcha:** `cfg_hash` carries per-subject bits (one hash per subject) — check consistency on the
  fit-field subset, not the hash.
- **Subject selection:** `select_subject_by_recon_percentile(cfg, model, percentile, metric='pearson_r')`
  ranks subjects by held-out recon quality (via `compute_subject_metrics_from_cache`; cols
  `subject/coverage/pearson_r/r2/rmse`); percentile 100 = best. Load one subject:
  `build_prediction_tensor_view(..., subjects=[SUBJECT])`.
- **PLS basis (if needed):** `fit_ahba_basis(prepost, n_comp)` mirrors DLAM exactly
  (`build_region_matrix(prepost['ahba_h'], genes, target_meta)` + `Y=[coord_y, coord_z, |coord_x|]`);
  `T_ahba` == DLAM's `t_ref_full`. Project a subject's `loro_fused` with `project_subject`. Cache to
  `out/pls_basis/`.

## Plot / formatting patterns (publication style)
- **Mixed brain+scatter grids:** use `fig.subfigures(1, 2, width_ratios=[2.0,1.2], wspace=0.0)` — brains
  tight in the left subfigure (`gridspec_kw wspace≈0.02`), scatters with their own room in the right
  (`hspace≈0.4`). Avoids scatter y-axes colliding with brains.
- **Brains:** `imshow(renderer.render(...))` + `axis('off')`; LV/row labels via
  `ax.text(-0.10, 0.5, ..., rotation=90, transform=ax.transAxes)`.
- **Scatters:** `set_box_aspect(1)` (square box, not data-equal) + `set_anchor('W')` to keep them close
  to the brains; remove top/right spines; light grid (`color='0.9'`); stats in a rounded `bbox` text
  (top-left); slim colorbar with `outline.set_visible(False)`.
- **Fonts:** single `font_scale` multiplier; unify header + column-title sizes; multi-line headers via
  `\n` + `linespacing`.
- **Color:** diverging `RdBu_r` (symmetric clim) for signed scores; `cmap='auto'` follows DLAM's per-LV
  palette (`eval_dlam_alignment.COMPONENT_CMAPS` = cividis/plasma/viridis, percentile clim).
- **Output:** save to `out/brain_renders/` at `dpi≈200`. To avoid Jupyter **double-rendering**, render
  functions must end with `plt.show(); plt.close(fig)` and **not** return the Figure.

## Validation workflow
- Validate end-to-end with `nbclient`
  (`NotebookClient(nb, resources={'metadata':{'path':repo_root}})`) **or**, while the user is
  live-editing the notebook, a **standalone /tmp script** (load → basis → project → render to `/tmp`) so
  you don't overwrite their file. Then view the PNG.
- The PREPOST cache (`notebooks/cache/prepost/*.pkl`, ~9 GB) loads in seconds on a hit; the node has
  ~500 GB RAM, so full loads (all genes × all subjects) are fine.

## Gotchas
- Notebook files have drifted to the **repo root** repeatedly (IDE/Jupyter); confirm the path before
  editing and `find` for stray copies.
- yabplot/pooch downloads the midthickness surface to `~/.cache/yabplot` (~5 MB; home quota).
- Route metric/label text through `src/eval_utils/eval_style.py` (`model_label`, `format_metric_label`);
  never hardcode metric strings.
- Reference analysis to mirror conventions: `eval_dlam_alignment.ipynb` + `src/eval_utils/dlam_diagnostics.py`.
