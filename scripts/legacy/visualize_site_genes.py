#!/usr/bin/env python3
"""
Visualize site×genes matrices for BHA2 and GTEx:
- Heatmaps using the same gene set (HVG intersection) for both datasets
- 3D scatter of sampling sites (MNI xyz) for each dataset
- Combined 3D scatter: BHA2 and GTEx in same space to see overlap
Saves figures to out/figs/. Run from project root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_hvg(root, n=100):
    path = root / "ahba_100hvg.txt"
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes[:n]

def match_genes(gene_list, columns, n=100):
    """Return column names that match gene_list (case-insensitive), up to n."""
    col_upper = {c.upper(): c for c in columns}
    matched = []
    for g in gene_list:
        if g.upper() in col_upper:
            matched.append(col_upper[g.upper()])
        if len(matched) >= n:
            break
    return matched[:n]

def common_genes(hvg, bha2_cols, gtex_cols, n=100):
    """Return (bha2_names, gtex_names) for genes in HVG present in both, same order."""
    bha2_upper = {c.upper(): c for c in bha2_cols}
    gtex_upper = {c.upper(): c for c in gtex_cols}
    bha2_list = []
    gtex_list = []
    for g in hvg[:n]:
        gu = g.upper()
        if gu in bha2_upper and gu in gtex_upper:
            bha2_list.append(bha2_upper[gu])
            gtex_list.append(gtex_upper[gu])
    return bha2_list, gtex_list

def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "out" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    hvg = load_hvg(root, n=100)

    bha2_sites_path = root / "data" / "bha2" / "site_x_genes_bha2_391.csv"
    bha2_xyz_path = root / "data" / "bha2" / "xyz_bha2_391.csv"
    gtex_sites_path = root / "data" / "gtex" / "site_x_genes_gtex.csv"
    gtex_xyz_path = root / "data" / "gtex" / "xyz_gtex.csv"

    bha2_loaded = bha2_sites_path.exists() and bha2_xyz_path.exists()
    gtex_loaded = gtex_sites_path.exists() and gtex_xyz_path.exists()

    if bha2_loaded:
        bha2_df = pd.read_csv(bha2_sites_path, index_col=0)
        bha2_xyz = pd.read_csv(bha2_xyz_path)
    if gtex_loaded:
        gtex_df = pd.read_csv(gtex_sites_path, index_col=0)
        gtex_xyz = pd.read_csv(gtex_xyz_path)

    # ---- Common gene set for both heatmaps ----
    if bha2_loaded and gtex_loaded:
        genes_bha2, genes_gtex = common_genes(hvg, bha2_df.columns, gtex_df.columns, 100)
        if genes_bha2 and genes_gtex:
            print("Common HVG genes in BHA2 and GTEx: %d (same set used for both heatmaps)" % len(genes_bha2))
            # BHA2 heatmap (same genes)
            Z = bha2_df[genes_bha2].values.astype(float)
            Z = np.nan_to_num(Z, nan=0.0)
            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(Z, aspect="auto", cmap="viridis")
            ax.set_yticks(np.linspace(0, Z.shape[0] - 1, min(10, Z.shape[0]), dtype=int))
            ax.set_yticklabels(bha2_df.index[ax.get_yticks()], fontsize=6)
            ax.set_xticks(np.linspace(0, len(genes_bha2) - 1, min(20, len(genes_bha2)), dtype=int))
            ax.set_xticklabels([genes_bha2[i] for i in ax.get_xticks()], rotation=45, ha="right", fontsize=6)
            ax.set_xlabel("Gene")
            ax.set_ylabel("Site (ROI)")
            ax.set_title("BHA2: site × genes (HVG shared with GTEx, n=%d)" % len(genes_bha2))
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_bha2_hvg100.png", dpi=150)
            plt.close()
            print("Saved heatmap_bha2_hvg100.png")
            # GTEx heatmap (same genes, same order)
            Zg = gtex_df[genes_gtex].values.astype(float)
            Zg = np.nan_to_num(Zg, nan=0.0)
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(Zg, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(gtex_df.index)))
            ax.set_yticklabels(gtex_df.index, fontsize=8)
            ax.set_xticks(np.linspace(0, len(genes_gtex) - 1, min(20, len(genes_gtex)), dtype=int))
            ax.set_xticklabels([genes_gtex[i] for i in ax.get_xticks()], rotation=45, ha="right", fontsize=6)
            ax.set_xlabel("Gene")
            ax.set_ylabel("Site (tissue)")
            ax.set_title("GTEx brain: site × genes (HVG shared with BHA2, n=%d)" % len(genes_gtex))
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_gtex_hvg100.png", dpi=150)
            plt.close()
            print("Saved heatmap_gtex_hvg100.png")

    # ---- Individual 3D scatters ----
    if bha2_loaded:
        x, y, z = bha2_xyz["x"].values, bha2_xyz["y"].values, bha2_xyz["z"].values
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=8, alpha=0.7)
        ax.set_xlabel("MNI X"); ax.set_ylabel("MNI Y"); ax.set_zlabel("MNI Z")
        ax.set_title("BHA2: spatial distribution of 391 sites")
        plt.savefig(out_dir / "scatter3d_bha2_sites.png", dpi=150)
        plt.close()
        print("Saved scatter3d_bha2_sites.png")
    if gtex_loaded:
        x, y, z = gtex_xyz["x"].values, gtex_xyz["y"].values, gtex_xyz["z"].values
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=80, alpha=0.8)
        for i, sid in enumerate(gtex_xyz["site_id"]):
            ax.text(x[i], y[i], z[i], sid, fontsize=6)
        ax.set_xlabel("MNI X"); ax.set_ylabel("MNI Y"); ax.set_zlabel("MNI Z")
        ax.set_title("GTEx brain: spatial distribution of 13 sites")
        plt.savefig(out_dir / "scatter3d_gtex_sites.png", dpi=150)
        plt.close()
        print("Saved scatter3d_gtex_sites.png")

    # ---- Combined 3D: BHA2 + GTEx in same space ----
    if bha2_loaded and gtex_loaded:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        x1, y1, z1 = bha2_xyz["x"].values, bha2_xyz["y"].values, bha2_xyz["z"].values
        x2, y2, z2 = gtex_xyz["x"].values, gtex_xyz["y"].values, gtex_xyz["z"].values
        ax.scatter(x1, y1, z1, s=12, alpha=0.5, c="C0", label="BHA2 (391 sites)")
        ax.scatter(x2, y2, z2, s=120, alpha=0.9, c="C1", marker="^", label="GTEx (13 sites)")
        ax.set_xlabel("MNI X"); ax.set_ylabel("MNI Y"); ax.set_zlabel("MNI Z")
        ax.set_title("BHA2 and GTEx sampling sites in MNI space (overlap)")
        ax.legend()
        plt.savefig(out_dir / "scatter3d_bha2_gtex_overlap.png", dpi=150)
        plt.close()
        print("Saved scatter3d_bha2_gtex_overlap.png")

    # Fallback: heatmaps with dataset-specific genes if no overlap
    if bha2_loaded and not (gtex_loaded and common_genes(hvg, bha2_df.columns, gtex_df.columns, 1)[0]):
        genes_use = match_genes(hvg, bha2_df.columns, 100)
        if genes_use:
            Z = bha2_df[genes_use].values.astype(float)
            Z = np.nan_to_num(Z, nan=0.0)
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.imshow(Z, aspect="auto", cmap="viridis")
            ax.set_ylabel("Site (ROI)"); ax.set_xlabel("Gene")
            ax.set_title("BHA2: site × genes (top 100 HVG)")
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_bha2_hvg100.png", dpi=150)
            plt.close()
    if gtex_loaded and not (bha2_loaded and common_genes(hvg, bha2_df.columns if bha2_loaded else [], gtex_df.columns, 1)[0]):
        genes_use = match_genes(hvg, gtex_df.columns, 100)
        if genes_use:
            Z = gtex_df[genes_use].values.astype(float)
            Z = np.nan_to_num(Z, nan=0.0)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(Z, aspect="auto", cmap="viridis")
            ax.set_ylabel("Site (tissue)"); ax.set_xlabel("Gene")
            ax.set_title("GTEx: site × genes (top 100 HVG)")
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_gtex_hvg100.png", dpi=150)
            plt.close()

    print("Figures saved to", out_dir)

if __name__ == "__main__":
    main()
