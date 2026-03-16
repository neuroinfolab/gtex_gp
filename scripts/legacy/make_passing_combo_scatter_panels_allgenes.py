#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.preprocess import build_target_parcels, map_gtex_to_target


def parse_args():
    p = argparse.ArgumentParser(description="Create all-genes Allen-vs-GTEx scatter suite for passing combos.")
    p.add_argument("--run-root", default="out/vnext_alignment_allgenes")
    p.add_argument("--csv-path", default="gxp_samples.csv")
    p.add_argument("--out-fig-dir", default="")
    p.add_argument("--out-tab-dir", default="")
    p.add_argument("--max-points-per-combo", type=int, default=250000)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _parse_combo(combo: str) -> Tuple[str, str, str, str]:
    parts = str(combo).split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid combo: {combo}")
    return parts[0], parts[1], parts[2], parts[3]


def _choose_combos(table_dir: Path) -> List[str]:
    gate_path = table_dir / "rollout_gate_summary_allgenes.csv"
    if gate_path.exists():
        gate = pd.read_csv(gate_path)
        if "gate_pass" in gate.columns:
            g = gate[gate["gate_pass"] == True].copy()  # noqa: E712
            if len(g):
                return g.sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True])["model_combo"].astype(str).tolist()
        if "model_combo" in gate.columns and len(gate):
            return gate.sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True])["model_combo"].astype(str).tolist()
    sel_path = table_dir / "selected_combos.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        if "model_combo" in sel.columns:
            return sel["model_combo"].astype(str).tolist()
    return []


def _sample_long(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def main():
    a = parse_args()
    np.random.seed(a.seed)

    root = Path(".").resolve()
    run_root = (root / a.run_root).resolve()
    table_dir = run_root / "tables"
    fig_dir = (Path(a.out_fig_dir).resolve() if a.out_fig_dir else (run_root / "figures" / "allgenes_passing_combo_scatter"))
    out_tab = (Path(a.out_tab_dir).resolve() if a.out_tab_dir else (run_root / "tables" / "allgenes_passing_combo_scatter"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    combos = _choose_combos(table_dir)
    if len(combos) == 0:
        raise RuntimeError("No combos found in all-genes run root.")

    first_raw = pd.read_csv(table_dir / f"aggregate_allgenes_{combos[0]}_mean_raw.csv")
    meta_cols = ["parcel_idx", "parcel_name", "coord_x", "coord_y", "coord_z"]
    genes = [c for c in first_raw.columns if c not in meta_cols]

    # Build shared parcel metadata and observed mask from source GTEx mapping.
    src = io_utils.read_expression_subset((root / a.csv_path).resolve(), genes)
    ahba_src = src[src["dataset_upper"] == "AHBA"].copy()
    gtex_src = src[src["dataset_upper"] == "GTEX"].copy()
    target = build_target_parcels(ahba_src)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_src["parcel_idx"] = ahba_src["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_src = map_gtex_to_target(gtex_src, target)
    obs_idx = set(gtex_src["parcel_idx"].unique().tolist())

    # Allen raw fixed reference.
    ahba_raw = ahba_src.groupby("parcel_idx")[genes].mean().reset_index()
    ahba_raw = target[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].merge(ahba_raw, on="parcel_idx", how="left")
    ahba_raw = ahba_raw.rename(columns={"tissue_or_parcel": "parcel_name"}).sort_values("parcel_idx").reset_index(drop=True)

    region_ids = ahba_raw["parcel_idx"].to_numpy(dtype=np.int32)
    region_names = ahba_raw["parcel_name"].astype(str).to_numpy()
    reg_palette = sns.color_palette("husl", n_colors=len(region_ids))
    reg_color = {int(region_ids[i]): reg_palette[i] for i in range(len(region_ids))}

    # top genes for gene-color panel
    top_genes = pd.Series(np.nanvar(ahba_raw[genes].to_numpy(dtype=float), axis=0), index=genes).sort_values(ascending=False).head(20).index.tolist()
    gene_palette = sns.color_palette("tab20", n_colors=len(top_genes))
    gene_color = {top_genes[i]: gene_palette[i] for i in range(len(top_genes))}

    all_long = []
    reg_corr_rows = []
    gene_corr_rows = []

    for combo in combos:
        raw_path = table_dir / f"aggregate_allgenes_{combo}_mean_raw.csv"
        h_path = table_dir / f"aggregate_allgenes_{combo}_mean_harmonized.csv"
        if not raw_path.exists() or not h_path.exists():
            continue
        gtex_raw = pd.read_csv(raw_path).sort_values("parcel_idx").reset_index(drop=True)
        gtex_h = pd.read_csv(h_path).sort_values("parcel_idx").reset_index(drop=True)

        # Harmonized Allen reference recomputed by combo harmonizer.
        hm, _, _, _ = _parse_combo(combo)
        harm = fit_harmonizer(ahba_src, gtex_src, genes, method=hm, cfg=type("C", (), {})())
        ahba_h = harm.transform(ahba_src, "AHBA")
        ahba_h = ahba_h.groupby("parcel_idx")[genes].mean().reset_index()
        ahba_h = target[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].merge(ahba_h, on="parcel_idx", how="left")
        ahba_h = ahba_h.rename(columns={"tissue_or_parcel": "parcel_name"}).sort_values("parcel_idx").reset_index(drop=True)

        for scale, a_tbl, g_tbl in [("raw", ahba_raw, gtex_raw), ("harmonized", ahba_h, gtex_h)]:
            reg_rep = np.repeat(region_ids, len(genes))
            reg_name_rep = np.repeat(region_names, len(genes))
            gene_rep = np.tile(np.asarray(genes), len(region_ids))
            obs_rep = np.repeat(np.asarray([int(r) in obs_idx for r in region_ids]), len(genes))
            long_df = pd.DataFrame(
                {
                    "combo": combo,
                    "scale": scale,
                    "parcel_idx": reg_rep,
                    "parcel_name": reg_name_rep,
                    "gene": gene_rep,
                    "allen_value": a_tbl[genes].to_numpy(dtype=float).reshape(-1),
                    "gtex_value": g_tbl[genes].to_numpy(dtype=float).reshape(-1),
                    "is_observed_gtex": obs_rep,
                }
            )
            all_long.append(long_df)

            # correlations
            A = a_tbl[genes].to_numpy(dtype=float)
            G = g_tbl[genes].to_numpy(dtype=float)
            for i, pid in enumerate(region_ids):
                x = A[i, :]
                y = G[i, :]
                r = np.corrcoef(x, y)[0, 1] if (np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
                reg_corr_rows.append({"combo": combo, "scale": scale, "parcel_idx": int(pid), "parcel_name": region_names[i], "pearson_r": r})
            for j, g in enumerate(genes):
                x = A[:, j]
                y = G[:, j]
                r = np.corrcoef(x, y)[0, 1] if (np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
                gene_corr_rows.append({"combo": combo, "scale": scale, "gene": g, "pearson_r": r})

    long_all = pd.concat(all_long, ignore_index=True)
    reg_corr_df = pd.DataFrame(reg_corr_rows)
    gene_corr_df = pd.DataFrame(gene_corr_rows)
    long_all.to_csv(out_tab / "long_all.csv", index=False)
    reg_corr_df.to_csv(out_tab / "region_correlations_all.csv", index=False)
    gene_corr_df.to_csv(out_tab / "gene_correlations_all.csv", index=False)

    combos = sorted(long_all["combo"].unique().tolist())
    n = len(combos)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    def _facet_setup(title: str):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows), constrained_layout=True, sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        fig.suptitle(title, y=1.01)
        return fig, axes

    for scale in ["raw", "harmonized"]:
        base = long_all[long_all["scale"] == scale].copy()

        # 1) region color
        fig, axes = _facet_setup(f"All-genes {scale}: Allen vs GTEx (color by region)")
        for i, combo in enumerate(combos):
            ax = axes[i]
            d = _sample_long(base[base["combo"] == combo], a.max_points_per_combo, a.seed)
            u = d[~d["is_observed_gtex"]]
            o = d[d["is_observed_gtex"]]
            ax.scatter(u["allen_value"], u["gtex_value"], c=[reg_color[int(x)] for x in u["parcel_idx"]], s=1, alpha=0.2, marker="^", linewidths=0, rasterized=True)
            ax.scatter(o["allen_value"], o["gtex_value"], c=[reg_color[int(x)] for x in o["parcel_idx"]], s=1.5, alpha=0.25, marker="o", linewidths=0, rasterized=True)
            ax.set_title(combo, fontsize=8)
            ax.grid(True, alpha=0.15)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.savefig(fig_dir / f"scatter_region_color_faceted_{scale}.png", dpi=220)
        plt.close(fig)

        # 2) gene color (top genes only for readability)
        fig, axes = _facet_setup(f"All-genes {scale}: Allen vs GTEx (color by gene, top20)")
        for i, combo in enumerate(combos):
            ax = axes[i]
            d0 = base[(base["combo"] == combo) & (base["gene"].isin(top_genes))]
            d = _sample_long(d0, min(a.max_points_per_combo, len(d0)), a.seed)
            u = d[~d["is_observed_gtex"]]
            o = d[d["is_observed_gtex"]]
            ax.scatter(u["allen_value"], u["gtex_value"], c=[gene_color[g] for g in u["gene"]], s=5, alpha=0.35, marker="^", linewidths=0)
            ax.scatter(o["allen_value"], o["gtex_value"], c=[gene_color[g] for g in o["gene"]], s=6, alpha=0.4, marker="o", linewidths=0)
            ax.set_title(combo, fontsize=8)
            ax.grid(True, alpha=0.15)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.savefig(fig_dir / f"scatter_gene_color_faceted_{scale}.png", dpi=220)
        plt.close(fig)

        # 3) region corr color
        fig, axes = _facet_setup(f"All-genes {scale}: Allen vs GTEx (color by region correlation)")
        for i, combo in enumerate(combos):
            ax = axes[i]
            d = _sample_long(base[base["combo"] == combo], a.max_points_per_combo, a.seed)
            cmap = reg_corr_df[(reg_corr_df["combo"] == combo) & (reg_corr_df["scale"] == scale)][["parcel_idx", "pearson_r"]]
            d = d.merge(cmap, on="parcel_idx", how="left")
            m = np.isfinite(d["pearson_r"].to_numpy())
            u = (~d["is_observed_gtex"].to_numpy()) & m
            o = d["is_observed_gtex"].to_numpy() & m
            sc = ax.scatter(d.loc[u, "allen_value"], d.loc[u, "gtex_value"], c=d.loc[u, "pearson_r"], s=1, alpha=0.22, marker="^", linewidths=0, cmap="coolwarm", vmin=-1, vmax=1, rasterized=True)
            ax.scatter(d.loc[o, "allen_value"], d.loc[o, "gtex_value"], c=d.loc[o, "pearson_r"], s=1.5, alpha=0.28, marker="o", linewidths=0, cmap="coolwarm", vmin=-1, vmax=1, rasterized=True)
            ax.set_title(combo, fontsize=8)
            ax.grid(True, alpha=0.15)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, label="Pearson r")
        fig.savefig(fig_dir / f"scatter_region_corrcolor_faceted_{scale}.png", dpi=220)
        plt.close(fig)

        # 4) gene corr color
        fig, axes = _facet_setup(f"All-genes {scale}: Allen vs GTEx (color by gene correlation)")
        for i, combo in enumerate(combos):
            ax = axes[i]
            d = _sample_long(base[base["combo"] == combo], a.max_points_per_combo, a.seed)
            cmap = gene_corr_df[(gene_corr_df["combo"] == combo) & (gene_corr_df["scale"] == scale)][["gene", "pearson_r"]]
            d = d.merge(cmap, on="gene", how="left")
            m = np.isfinite(d["pearson_r"].to_numpy())
            u = (~d["is_observed_gtex"].to_numpy()) & m
            o = d["is_observed_gtex"].to_numpy() & m
            sc = ax.scatter(d.loc[u, "allen_value"], d.loc[u, "gtex_value"], c=d.loc[u, "pearson_r"], s=1, alpha=0.22, marker="^", linewidths=0, cmap="coolwarm", vmin=-1, vmax=1, rasterized=True)
            ax.scatter(d.loc[o, "allen_value"], d.loc[o, "gtex_value"], c=d.loc[o, "pearson_r"], s=1.5, alpha=0.28, marker="o", linewidths=0, cmap="coolwarm", vmin=-1, vmax=1, rasterized=True)
            ax.set_title(combo, fontsize=8)
            ax.grid(True, alpha=0.15)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, label="Pearson r")
        fig.savefig(fig_dir / f"scatter_gene_corrcolor_faceted_{scale}.png", dpi=220)
        plt.close(fig)

    # Legend maps
    pd.DataFrame({"parcel_idx": region_ids, "parcel_name": region_names}).to_csv(out_tab / "region_legend.csv", index=False)
    pd.DataFrame({"gene": top_genes}).to_csv(out_tab / "gene_legend_top20.csv", index=False)
    pd.DataFrame({"combo": combos}).to_csv(out_tab / "combo_manifest.csv", index=False)
    print(f"done: {fig_dir}")


if __name__ == "__main__":
    main()
