#!/usr/bin/env python3
"""
Download GTEx gene expression (median TPM by tissue) via GTEx Portal API.
Uses the latest GTEx release (v10) with continuous TPM values.
Saves to data/gtex/median_tpm_by_tissue.json.
"""
import json
import sys
import time
from pathlib import Path

import requests

# GTEx API v2 - latest release (v10)
GTEX_API_BASE = "https://gtexportal.org/api/v2"
DATASET_ID = "gtex_v10"
ITEMS_PER_PAGE = 1000

# Brain tissue IDs used by the GTEx API (must match tissue_to_mni mapping in build script)
BRAIN_TISSUE_IDS = [
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Substantia_nigra",
]


def fetch_top_expressed_genes(tissue_id: str, page: int = 0) -> dict:
    """Fetch one page of top-expressed genes (median TPM) for a tissue."""
    url = f"{GTEX_API_BASE}/expression/topExpressedGene"
    params = {
        "tissueSiteDetailId": tissue_id,
        "datasetId": DATASET_ID,
        "page": page,
        "itemsPerPage": ITEMS_PER_PAGE,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "gtex"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "median_tpm_by_tissue.json"

    result = {}
    for tissue_id in BRAIN_TISSUE_IDS:
        print(f"Fetching {tissue_id}...", end=" ", flush=True)
        tissue_data = {}  # geneSymbol -> median TPM
        page = 0
        while True:
            try:
                resp = fetch_top_expressed_genes(tissue_id, page)
            except requests.RequestException as e:
                print(f"\nAPI error for {tissue_id} page {page}: {e}", file=sys.stderr)
                sys.exit(1)
            data = resp.get("data", [])
            paging = resp.get("paging_info", {})
            for row in data:
                tissue_data[row["geneSymbol"]] = float(row["median"])
            total = paging.get("totalNumberOfItems", 0)
            n_pages = paging.get("numberOfPages", 1)
            page += 1
            if page >= n_pages or not data:
                break
            time.sleep(0.3)  # gentle rate limiting
        result[tissue_id] = tissue_data
        print(f"{len(tissue_data)} genes")

    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
