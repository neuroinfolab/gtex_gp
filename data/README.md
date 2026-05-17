# Data Layout

This repository is code-first. Raw data and derived outputs are not tracked.

Expected local layout:
- `data/raw/`: raw inputs such as `gxp_samples.csv`
- `data/derived/`: local derived data products

The canonical tracked smoke fixtures live under:
- `tests/fixtures/`

The workflow defaults to:
- `data/raw/gxp_samples.csv`
- `out/raw/gene_lists/ahba_100hvg.txt`

It also supports the legacy root-level fallbacks `gxp_samples.csv` and `ahba_100hvg.txt` for local migration convenience.

## `data/raw/gxp_samples.csv`

`gxp_samples.csv` is the main combined AHBA + GTEx sample table consumed by `gtex_gp`.

It was originally created in the `neuroVformer` repo via:

- notebook: [`eval_refgenome_hvg.ipynb`](/scratch/asr655/neuroinformatics/Seq2GeneEx/neuroVformer/notebooks/eval_refgenome_hvg.ipynb)
- builder: [`data_utils.py`](/scratch/asr655/neuroinformatics/Seq2GeneEx/neuroVformer/data/data_utils.py#L812), function `build_combined_samples_df(...)`

### High-Level Mapping Idea

The GTEx rows in this table do not carry one native measured coordinate per donor sample. Instead, a manual atlas-based mapping was defined from each GTEx sampled brain tissue to a set of atlas parcels:

- cortical tissues map through a Brodmann-area atlas (`BA`)
- subcortical tissues map through the Schaefer-156 / Tian subcortical atlas mapping (`S156`)
- cerebellar tissues are also included through the same `S156`-side mapping resources

Those mappings are defined using atlas/map CSVs external to this repo, including:

- `MaptoBA.csv`
- `MaptoS156.csv`
- `overlay_brodmann_2mm_MNI_reformatted.csv`
- `atlas-4S156Parcels_dseg_reformatted.csv`

For each GTEx brain tissue, these files define a set of MNI coordinates associated with that tissue. Every donor row for that GTEx tissue inherits the same coordinate list.

### Table Structure

Each row contains:

- `subject`
- `age`
- `sex`
- `dataset`
- `tissue_or_parcel`
- `coordinates`
- one column per gene

Row semantics differ by dataset:

- `dataset == "GTEx"`: one row per donor per GTEx brain tissue; gene values come from GTEx brain GCT files, averaged within donor for that tissue, then `log1p` transformed; `coordinates` is a list of atlas-derived MNI coordinates for that tissue, shared across donors
- `dataset == "AHBA"`: one row per AHBA donor per parcel; gene values come from AHBA parcel-level expression tables; `coordinates` is typically a single `(x, y, z)` parcel coordinate

Current builder note:

- GTEx duplicate sample-columns that collapse to the same parsed subject ID
  within a tissue file are now treated as an error during rebuild, rather than
  being silently averaged into one row.
- Optional pre-build GTEx filtering (`SMRIN > 6`, TPM/read-count thresholds,
  GTEx ∩ AHBA overlap audit) is documented in
  [`docs/samples_builder.md`](../docs/samples_builder.md).

### Downstream Note

In `gtex_gp`, the GTEx `coordinates` list is later reduced to a single centroid and then mapped to one AHBA parcel by nearest-neighbor distance. So the spatial assignment used by the modeling pipeline is a downstream simplification of the atlas-derived coordinate lists stored in this CSV.

For the formal manuscript-facing GTEx-to-AHBA matching policy, including cortical Brodmann/Schaefer overrides, cerebellar duplicate handling, and subcortical manual validation against 4S156 atlas metadata, see [`docs/manuscript/parcel_matching.md`](../docs/manuscript/parcel_matching.md). The companion exploratory notebook is [`parcel_assignment.ipynb`](../parcel_assignment.ipynb).

## Parcel Ordering Contract (Aligned With `out/README.md`)

For model/cache outputs (especially `.npz` matrices under `out/loro_subject_cache/`), parcel row order is defined by pipeline `parcel_idx`, not by row order in atlas CSV files.

`parcel_idx` is built from AHBA labels in `data/raw/gxp_samples.csv`:
1. filter rows with `dataset == "AHBA"` (or `dataset_upper == "AHBA"`)
2. take unique `tissue_or_parcel`
3. sort labels alphabetically
4. assign contiguous indices `0..P-1`

Important implication for visualization:
- `atlas-4S156Parcels_dseg_reformatted.csv` (`label` column) is not guaranteed to be in this same order.
- To align atlas metadata to `.npz` parcel rows, filter atlas labels to the AHBA-used set and sort by label ascending.

See `out/README.md` for cache-structure and `.npz` key definitions.

### Atlas Labels Missing From AHBA Samples

Comparing:
- `gxp_samples.csv` with `dataset == "AHBA"` using `tissue_or_parcel`
- `atlas-4S156Parcels_dseg_reformatted.csv` using `label`

`150/156` labels overlap exactly. The following `6` atlas labels are present in the reformatted atlas CSV but absent from AHBA sample `tissue_or_parcel`:

- `Cerebellar_Region9`
- `LH-MN`
- `RH-EXA`
- `RH-STH`
- `RH-VeP`
- `RH_SomMot_2`
