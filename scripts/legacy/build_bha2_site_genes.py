#!/usr/bin/env python3
"""
Parse BHA2 unzipped data: read transcriptomics.csv and iPA_nROIS.csv for a given resolution,
produce site x genes matrix and xyz (MNI) table. Saves to data/bha2/.
Run from project root. Requires BHA2 data already extracted under data/bha2/data/.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

RESOLUTIONS = [183, 391, 568, 729, 964, 1242, 1584, 1795, 2165]

def main():
    parser = argparse.ArgumentParser(description="Build BHA2 site x genes and xyz")
    parser.add_argument("--resolution", type=int, default=391, choices=RESOLUTIONS,
                        help="Parcellation resolution (number of ROIs)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to unzipped BHA2 data (default: project/data/bha2)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    base = args.data_dir or (root / "data" / "bha2")
    # Unzip may create data/bha2/data/iPA_391/ or data/bha2/iPA_391/
    for parent in (base / "data", base):
        res_dir = parent / f"iPA_{args.resolution}"
        if res_dir.exists():
            break
    else:
        raise FileNotFoundError(
            f"BHA2 resolution dir iPA_{args.resolution} not found under {base}. "
            "Run download_bha2.py first and unzip to data/bha2/."
        )

    # Expected files (names from Zenodo description)
    coords_file = res_dir / f"iPA_{args.resolution}.csv"
    trans_file = res_dir / "transcriptomics.csv"
    for f in (coords_file, trans_file):
        if not f.exists():
            raise FileNotFoundError(f"Expected file not found: {f}")

    # Load MNI coordinates and ROI labels
    coords_df = pd.read_csv(coords_file)
    # Common column names for MNI: X_MNI,Y_MNI,Z_MNI or x,y,z or mni_x, mni_y, mni_z or X,Y,Z
    xyz_cols = None
    for cand in (["X_MNI", "Y_MNI", "Z_MNI"], ["x", "y", "z"], ["mni_x", "mni_y", "mni_z"], ["X", "Y", "Z"], ["x_mni", "y_mni", "z_mni"]):
        if all(c in coords_df.columns for c in cand):
            xyz_cols = cand
            break
    if xyz_cols is None:
        # Try first three numeric columns
        numeric = coords_df.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 3:
            xyz_cols = numeric.columns[:3].tolist()
        else:
            raise ValueError("Could not find x,y,z columns in coords file. Columns: " + str(coords_df.columns.tolist()))
    site_id_col = "ROI_number" if "ROI_number" in coords_df.columns else ("ROI" if "ROI" in coords_df.columns else coords_df.columns[0])
    if site_id_col not in coords_df.columns:
        site_id_col = coords_df.columns[0]
    xyz = coords_df[xyz_cols].values.astype(float)
    site_ids = coords_df[site_id_col].astype(str).values

    # Load transcriptomics: may be genes x ROIs (first col = gene_symbol, rest = ROI 1,2,...) or ROIs x genes
    trans_df = pd.read_csv(trans_file)
    first_col = trans_df.columns[0]
    # If first column looks like gene names and rest are numeric (1,2,...), layout is genes x ROIs
    if trans_df.shape[1] > 1 and str(trans_df.columns[1]).isdigit():
        gene_names = trans_df[first_col].astype(str).tolist()
        n_sites = len(site_ids)
        roi_cols = [c for c in trans_df.columns[1:1 + n_sites]]
        if len(roi_cols) < n_sites:
            roi_cols = trans_df.columns[1:].tolist()[:n_sites]
        Y = trans_df[roi_cols].values.T.astype(float)  # sites x genes
    else:
        # ROIs x genes
        if trans_df.shape[0] != len(site_ids):
            trans_ids = trans_df[first_col].astype(str).values
            if set(trans_ids) >= set(site_ids):
                trans_df = trans_df.set_index(first_col).reindex(site_ids).reset_index(drop=True)
            else:
                trans_df = trans_df.iloc[: len(site_ids)]
        gene_cols = trans_df.select_dtypes(include=[np.number]).columns.tolist()
        if not gene_cols:
            gene_cols = [c for c in trans_df.columns if c != site_id_col]
        gene_names = list(gene_cols)
        Y = trans_df[gene_cols].values.astype(float)

    # Ensure same number of sites
    n_sites = len(site_ids)
    if Y.shape[0] != n_sites:
        Y = Y[:n_sites]
    if xyz.shape[0] != n_sites:
        xyz = xyz[:n_sites]
        site_ids = site_ids[:n_sites]

    out_dir = root / "data" / "bha2"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"site_x_genes_bha2_{args.resolution}"
    site_df = pd.DataFrame(Y, index=site_ids, columns=gene_names)
    site_df.index.name = "site_id"
    site_df.to_csv(out_dir / f"{prefix}.csv")
    xyz_df = pd.DataFrame({"site_id": site_ids, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    xyz_df.to_csv(out_dir / f"xyz_bha2_{args.resolution}.csv", index=False)
    print(f"Saved {site_df.shape[0]} x {site_df.shape[1]} to {out_dir / f'{prefix}.csv'}")
    print(f"Saved xyz to {out_dir / f'xyz_bha2_{args.resolution}.csv'}")

if __name__ == "__main__":
    main()
