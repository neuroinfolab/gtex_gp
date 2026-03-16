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
