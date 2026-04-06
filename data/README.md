# Data Layout

This repository is code-first. Raw data and derived outputs are not tracked.

Expected local layout:
- `data/raw/`: raw inputs such as `gxp_samples.csv`
- `data/derived/`: local derived data products

The canonical tracked smoke fixtures live under:
- `tests/fixtures/`

The workflow defaults to:
- `data/raw/gxp_samples.csv`
- `data/raw/ahba_100hvg.txt`

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

### Downstream Note

In `gtex_gp`, the GTEx `coordinates` list is later reduced to a single centroid and then mapped to one AHBA parcel by nearest-neighbor distance. So the spatial assignment used by the modeling pipeline is a downstream simplification of the atlas-derived coordinate lists stored in this CSV.
