# `subject_bundle_GTEX-1B996/` — local viz assets for one GTEx subject

A self-contained download bundle (~44 MB) for experimenting with the 3D gene-expression viewer locally. Default subject: **GTEX-1B996**. Includes atlas geometry, one subject across all three model variants, gene-list presets, and the canonical parcel-order mapping.

---

## Contents

```
subject_bundle_GTEX-1B996/
├── atlas_info/
│   ├── atlas-4S156Parcels_dseg_reformatted.csv
│   └── atlas-4S156Parcels_space-MNI152NLin6Asym_dseg.nii.gz
├── cache/
│   ├── naive/    GTEX-1B996.npz + .json
│   ├── dlam/     GTEX-1B996.npz + .json
│   └── plam/     GTEX-1B996.npz + .json
├── gene_lists/
│   ├── ahba_100hvg.txt / 250hvg.txt / 500hvg.txt
│   ├── gtex_100hvg.txt / gtex_100hvg_demeaned.txt
│   ├── gtex_10deg.txt / gtex_25deg.txt
│   ├── richiardi2015.txt
│   └── syngo.txt
├── parcel_order.json
└── README.md
```

---

## `atlas_info/` — the geometry backbone

### `atlas-4S156Parcels_dseg_reformatted.csv` (22 KB)

A lookup table with one row per atlas parcel (156 rows). Key columns:

- `index` — integer 1..156, matches the voxel values in the NIfTI
- `label` — human-readable name (`LH_Vis_1`, `Amygdala-L`, `Cerebellar_Region3`)
- `hemisphere` — `L` / `R` / `B`
- `structure` — `cortex` or `subcortex` (controls which `yabplot` function renders it)
- `mni_x, mni_y, mni_z` — parcel centroid in MNI space (mm)
- Network labels (`label_7network`, `label_17network`) — for filtering/coloring by functional network

**Use it for:** splitting parcels into cortex vs. subcortex (since `yabplot` renders them separately), tooltips, network filters, centroid-based distance calculations.

### `atlas-4S156Parcels_space-MNI152NLin6Asym_dseg.nii.gz` (160 KB)

The 3D volumetric atlas. Every voxel holds an integer = the `index` of the parcel it belongs to (0 = background).

**Use it for:** feeding into `yabplot.atlas_builder.build_cortical_atlas(...)` and `build_subcortical_atlas(...)` — these produce the surface + 3D meshes your scalars will be painted onto. **One-time setup**, then cache the output meshes.

---

## `cache/{naive,dlam,plam}/GTEX-1B996.npz`

Three flavors of the same subject, one per model variant. Each `.npz` is a compressed bundle of numpy arrays — all `(150, ~13618)` or `(150,)` shaped — aligned to the alphabetical parcel order in `parcel_order.json`.

**Load:**
```python
import numpy as np
d = np.load("cache/plam/GTEX-1B996.npz", allow_pickle=True)
```

**Keys you'll use:**

| Key | Shape | For which view |
|---|---|---|
| `gene_names` | `(G,)` str | Resolve "BDNF" → column index |
| `gtex_mask` | `(150,)` int8 | **V1** — which parcels the subject actually contributed GTEx data to |
| `loro_truth_subject_raw` | `(150, G)` | **V1 Input** — measured GTEx values (NaN outside `gtex_mask`) |
| `loro_eval_mask` | `(150,)` int8 | **V2/V3** — which parcels were held out in LORO eval |
| `loro_fused_subject_h` | `(150, G)` | **V2 Reconstruction** — model's prediction at held-out parcels |
| `loro_truth_subject_h` | `(150, G)` | **V3 Ground Truth** — AHBA truth at held-out parcels (harmonized space) |
| `fullfit_subject_h` | `(150, G)` | **V4 Whole-brain** — dense prediction everywhere |

PLAM only has extras (`plam_fullfit_uvar_latent`, `plam_loro_uvar_latent`) for uncertainty overlays.

**Why three models?** Different algorithms: `naive` (baseline regression), `dlam` (Deep Linear Additive), `plam` (Probabilistic Linear Additive + uncertainty). V1 and V3 are model-independent — they're the same across all three files. Only V2 and V4 change with model choice.

### `cache/*/GTEX-1B996.json`

Sidecar metadata — run config, gene scope, timestamp, counts. Good for tooltips ("Model X fit on N parcels, G genes, on date T") but not needed for rendering.

---

## `gene_lists/*.txt`

Plain text, one gene symbol per line. These are **curated dropdown presets** so users don't scroll through 13,618 genes.

| File | Count | Use when |
|---|---|---|
| `ahba_100hvg.txt` | 100 | Default — top highly-variable genes in AHBA |
| `ahba_250hvg.txt` / `ahba_500hvg.txt` | 250 / 500 | Wider AHBA HVG nets |
| `gtex_100hvg.txt` | 100 | HVGs from GTEx side |
| `gtex_100hvg_demeaned.txt` | 100 | GTEx HVGs after tissue-mean removal |
| `gtex_10deg.txt` / `gtex_25deg.txt` | 75 / 169 | Differentially expressed across tissues |
| `richiardi2015.txt` | 135 | Curated brain-relevant genes |
| `syngo.txt` | 1,601 | SynGO synaptic ontology |

**Load:**
```python
with open("gene_lists/ahba_100hvg.txt") as f:
    preset = [line.strip() for line in f if line.strip()]
```

---

## `parcel_order.json`

The 150 alphabetical AHBA parcel labels — **the canonical row order** for every `(150, G)` array in every `.npz`. Precomputed so you don't need the 887 MB `gxp_samples.csv` locally.

**Use it to align the atlas CSV to `.npz` rows:**
```python
import json, pandas as pd
labels = json.load(open("parcel_order.json"))["ahba_labels_alphabetical"]
atlas = pd.read_csv("atlas_info/atlas-4S156Parcels_dseg_reformatted.csv")
atlas_aligned = (atlas[atlas["label"].isin(labels)]
                 .set_index("label").loc[labels].reset_index())
# now atlas_aligned.iloc[i] describes parcel_idx = i in the .npz arrays
```

---

## How the pieces fit together for V1 (input view)

```python
import numpy as np, json, pandas as pd

# ---- one-time load --------------------------------------------------
labels = json.load(open("parcel_order.json"))["ahba_labels_alphabetical"]
atlas  = pd.read_csv("atlas_info/atlas-4S156Parcels_dseg_reformatted.csv")
atlas_aligned = atlas.set_index("label").loc[labels].reset_index()
# Build yabplot meshes ONCE from the NIfTI (slow; cache the output)

# ---- per subject (change on subject dropdown) -----------------------
d = np.load("cache/plam/GTEX-1B996.npz", allow_pickle=True)

# ---- per gene (change on gene dropdown — just slice, no I/O) --------
g = int(np.where(d["gene_names"] == "BDNF")[0][0])
v1 = np.where(d["gtex_mask"].astype(bool),
              d["loro_truth_subject_raw"][:, g], np.nan)   # (150,)

# Feed {labels[i]: v1[i]} into yabplot (skipping NaNs) + a pyvista cutaway clip
```

**Rule of thumb:** the bundle has one *geometry* source (atlas files), one *axis definition* (`parcel_order.json`), one *data source* per subject/model (`.npz`), and UX helpers (`gene_lists/`).

---

## Gotchas

1. **Parcel order is alphabetical AHBA labels**, not atlas CSV `index` order. Always align the atlas CSV to `parcel_order.json` before zipping values with labels.
2. `loro_truth_*` arrays are **NaN outside the relevant mask** (`gtex_mask` for `..._raw`, `loro_eval_mask` for `..._h`). Skip NaNs before handing to `yabplot` or they'll render as black patches.
3. **6 of the 156 atlas parcels have no AHBA coverage** and are absent from the 150-axis: `Cerebellar_Region9`, `LH-MN`, `RH-EXA`, `RH-STH`, `RH-VeP`, `RH_SomMot_2`. Hide or grey them out.
4. **Gene name matching is exact string** — check case/punctuation once against `d["gene_names"]`.
5. `yabplot` wants cortex and subcortex passed to separate functions (`plot_cortical` vs `plot_subcortical`) — use the `structure` column in the atlas CSV to split your 150-vector.
