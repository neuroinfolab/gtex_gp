#!/usr/bin/env python3
"""
Build GTEx brain site x genes matrix and xyz (MNI) from median TPM by tissue
and tissue_to_mni.csv. Saves to data/gtex/site_x_genes_gtex.csv and xyz_gtex.csv.
Run from project root. Requires download_gtex_brain.py to have run.

Prefers: median_tpm_by_tissue.json (GTEx v10 API, continuous TPM).
Fallbacks: GCT median TPM file or Harmonizome gene_attribute_matrix.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Map GTEx API tissue IDs to tissue_to_mni.csv tissue names
API_TO_MNI_TISSUE = {
    "Brain_Amygdala": "Brain - Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24": "Brain - Anterior cingulate cortex (BA24)",
    "Brain_Caudate_basal_ganglia": "Brain - Caudate (basal ganglia)",
    "Brain_Cerebellar_Hemisphere": "Brain - Cerebellar Hemisphere",
    "Brain_Cerebellum": "Brain - Cerebellum",
    "Brain_Cortex": "Brain - Cortex",
    "Brain_Frontal_Cortex_BA9": "Brain - Frontal Cortex (BA9)",
    "Brain_Hippocampus": "Brain - Hippocampus",
    "Brain_Hypothalamus": "Brain - Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia": "Brain - Nucleus accumbens (basal ganglia)",
    "Brain_Putamen_basal_ganglia": "Brain - Putamen (basal ganglia)",
    "Brain_Spinal_cord_cervical_c-1": "Brain - Spinal cord (cervical c-1)",
    "Brain_Substantia_nigra": "Brain - Substantia nigra",
}


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "gtex"
    api_json_path = data_dir / "median_tpm_by_tissue.json"
    gct_path = data_dir / "GTEx_Analysis_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    if not gct_path.exists():
        gct_path = data_dir / "GTEx_Analysis_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    harm_path = data_dir / "gene_attribute_matrix.txt.gz"
    if not api_json_path.exists() and not gct_path.exists() and not harm_path.exists():
        raise FileNotFoundError(
            f"GTEx file not found in {data_dir}. Run download_gtex_brain.py first."
        )
    tissue_mni_path = data_dir / "tissue_to_mni.csv"
    if not tissue_mni_path.exists():
        raise FileNotFoundError(f"tissue_to_mni.csv not found: {tissue_mni_path}")

    mni_df = pd.read_csv(tissue_mni_path)
    mni_df["tissue"] = mni_df["tissue"].str.strip()
    brain_tissues = set(mni_df["tissue"].tolist())

    # GTEx v10 API JSON (continuous median TPM)
    if api_json_path.exists():
        with open(api_json_path) as f:
            api_data = json.load(f)
        # Get union of genes (order from first tissue)
        genes_set = set()
        for tid, g2tpm in api_data.items():
            genes_set.update(g2tpm.keys())
        gene_names = sorted(genes_set)
        site_ids = []
        xyz_list = []
        rows = []
        for api_tid, mni_tissue in API_TO_MNI_TISSUE.items():
            if api_tid not in api_data or mni_tissue not in brain_tissues:
                continue
            match = mni_df[mni_df["tissue"] == mni_tissue]
            if match.empty:
                continue
            site_ids.append(mni_tissue)
            r = match.iloc[0]
            xyz_list.append([float(r["mni_x"]), float(r["mni_y"]), float(r["mni_z"])])
            g2tpm = api_data[api_tid]
            rows.append([g2tpm.get(g, 0.0) for g in gene_names])
        xyz = np.array(xyz_list)
        Y = np.array(rows, dtype=float)
        out_sites = pd.DataFrame(Y, index=site_ids, columns=gene_names)
        out_sites.index.name = "site_id"
        out_sites.to_csv(data_dir / "site_x_genes_gtex.csv")
        xyz_df = pd.DataFrame({"site_id": site_ids, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        xyz_df.to_csv(data_dir / "xyz_gtex.csv", index=False)
        print(f"Saved {out_sites.shape[0]} x {out_sites.shape[1]} to {data_dir / 'site_x_genes_gtex.csv'} (GTEx v10 continuous TPM)")
        print(f"Saved xyz to {data_dir / 'xyz_gtex.csv'}")
        return

    # Harmonizome format: genes x tissues (tab, first col = Gene) – discrete values
    if harm_path.exists():
        df = pd.read_csv(harm_path, sep="\t", compression="gzip")
        name_col = df.columns[0]
        gene_names = df[name_col].astype(str).tolist()
        tissue_cols = [c for c in df.columns[1:] if c in brain_tissues]
        if not tissue_cols:
            tissue_cols = [c for c in df.columns[1:] if "Brain" in str(c)]
        site_ids = []
        xyz_list = []
        for t in tissue_cols:
            match = mni_df[mni_df["tissue"] == t]
            if match.empty:
                match = mni_df[mni_df["tissue"].str.lower() == t.lower()]
            if match.empty:
                continue
            site_ids.append(t)
            r = match.iloc[0]
            xyz_list.append([float(r["mni_x"]), float(r["mni_y"]), float(r["mni_z"])])
        xyz = np.array(xyz_list)
        Y = df[site_ids].astype(float).values.T
        out_sites = pd.DataFrame(Y, index=site_ids, columns=gene_names)
        out_sites.index.name = "site_id"
        out_sites.to_csv(data_dir / "site_x_genes_gtex.csv")
        xyz_df = pd.DataFrame({"site_id": site_ids, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        xyz_df.to_csv(data_dir / "xyz_gtex.csv", index=False)
        print(f"Saved {out_sites.shape[0]} x {out_sites.shape[1]} to {data_dir / 'site_x_genes_gtex.csv'}")
        print(f"Saved xyz to {data_dir / 'xyz_gtex.csv'}")
        return

    # GCT format
    open_fn = gzip.open if str(gct_path).endswith(".gz") else open
    with open_fn(gct_path, "rt") as f:
        line1 = f.readline()
        line2 = f.readline()
        n_genes, n_samples = map(int, line2.strip().split()[:2])
        col_line = f.readline()
    cols = col_line.strip().split("\t")
    # GCT: Name Description sample1 sample2 ...
    sample_cols = [c for c in cols[2:] if c in brain_tissues]
    if not sample_cols:
        # Try name normalization: GTEx may use "Brain - Amygdala" etc.
        all_tissues = cols[2:]
        sample_cols = [c for c in all_tissues if any(bt.lower() in c.lower() for bt in brain_tissues)]
    if not sample_cols:
        sample_cols = [c for c in cols[2:] if c.startswith("Brain")]
    if not sample_cols:
        raise ValueError(
            "No brain tissue columns found in GCT. Columns (first 20): " + str(cols[:20])
        )

    # Read full GCT
    df = pd.read_csv(gct_path, sep="\t", skiprows=2)
    name_col = df.columns[0]
    gene_col = name_col  # gene id or name in first column
    gene_names = df[gene_col].astype(str).tolist()

    # Keep only tissues we have MNI for and that exist in GCT
    site_ids = []
    xyz_list = []
    for t in sample_cols:
        if t not in df.columns:
            continue
        match = mni_df[mni_df["tissue"] == t]
        if match.empty:
            match = mni_df[mni_df["tissue"].str.lower() == t.lower()]
        if match.empty:
            match = mni_df[mni_df["tissue"].str.contains(t.split("-")[-1].strip(), case=False, na=False)]
        if not match.empty:
            site_ids.append(t)
            r = match.iloc[0]
            xyz_list.append([float(r["mni_x"]), float(r["mni_y"]), float(r["mni_z"])])
    if not site_ids:
        site_ids = [c for c in sample_cols if c in df.columns]
        xyz_list = [[0.0, 0.0, 0.0]] * len(site_ids)
    xyz = np.array(xyz_list)
    Y = df[site_ids].astype(float).values.T  # site x genes

    out_sites = pd.DataFrame(Y, index=site_ids, columns=gene_names)
    out_sites.index.name = "site_id"
    out_sites.to_csv(data_dir / "site_x_genes_gtex.csv")
    xyz_df = pd.DataFrame({"site_id": site_ids, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    xyz_df.to_csv(data_dir / "xyz_gtex.csv", index=False)
    print(f"Saved {out_sites.shape[0]} x {out_sites.shape[1]} to {data_dir / 'site_x_genes_gtex.csv'}")
    print(f"Saved xyz to {data_dir / 'xyz_gtex.csv'}")

if __name__ == "__main__":
    main()
