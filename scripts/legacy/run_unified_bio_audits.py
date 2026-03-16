#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.preprocess import add_sample_groups, add_target_meta, build_target_parcels, map_gtex_to_target


MARKERS = ["GFAP", "AQP4", "MBP", "MOBP", "SLC17A7", "GAD1", "GAD2", "TH", "PDGFRA", "CLDN5"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Biological plausibility audits for unified phase2")
    p.add_argument("--csv-path", default="/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv")
    p.add_argument("--winner-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified")
    p.add_argument("--completed-gtex-path", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_final/gxp_completed_harmonized_gtex_only.csv")
    p.add_argument("--out-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_phase2")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _read_all_genes(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
    return h[6:]


def _winner_config(winner_root: Path) -> tuple[str, dict]:
    summary = json.loads((winner_root / "summary.json").read_text())
    winner = str(summary["winner_model"])
    cfg = {"harm": "combat", "cal": "hier_affine_map", "robust": "student_t", "hetero": "gene_var", "uncshrink": "false"}
    for part in winner.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            cfg[k] = v
    return winner, cfg


def _stream_parcel_means(csv_path: Path, genes: list[str], parcel_to_idx: dict[str, int], chunk_size: int = 128) -> np.ndarray:
    R = len(parcel_to_idx)
    G = len(genes)
    sums = np.zeros((R, G), dtype=np.float64)
    counts = np.zeros(R, dtype=np.int64)

    usecols = ["tissue_or_parcel", *genes]
    for ch in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        idx = ch["tissue_or_parcel"].astype(str).map(parcel_to_idx)
        m = idx.notna().to_numpy()
        if not np.any(m):
            continue
        idxv = idx[m].astype(int).to_numpy()
        X = ch.loc[m, genes].to_numpy(dtype=np.float64)
        for rid in np.unique(idxv):
            mm = idxv == rid
            sums[rid, :] += np.sum(X[mm, :], axis=0)
            counts[rid] += int(np.sum(mm))

    means = np.full((R, G), np.nan, dtype=np.float64)
    m = counts > 0
    means[m, :] = sums[m, :] / counts[m][:, None]
    return means


def _pair_name(name: str) -> str | None:
    if name.startswith("LH_"):
        return "RH_" + name[3:]
    if name.startswith("RH_"):
        return "LH_" + name[3:]
    if name.startswith("LH-"):
        return "RH-" + name[3:]
    if name.startswith("RH-"):
        return "LH-" + name[3:]
    return None


def _subject_shift_stats(csv_path: Path, genes: list[str], chunk_size: int = 64) -> pd.DataFrame:
    states = {}
    usecols = ["subject", "is_imputed", *genes]
    for ch in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        ch["subject"] = ch["subject"].astype(str)
        for (sid, imp), d in ch.groupby(["subject", "is_imputed"], observed=True):
            X = d[genes].to_numpy(dtype=np.float64)
            key = (sid, int(imp))
            if key not in states:
                states[key] = {"sum": 0.0, "sumsq": 0.0, "n": 0}
            states[key]["sum"] += float(np.sum(X))
            states[key]["sumsq"] += float(np.sum(X * X))
            states[key]["n"] += int(X.size)

    rows = []
    subjects = sorted(set(k[0] for k in states.keys()))
    for sid in subjects:
        o = states.get((sid, 0), {"sum": np.nan, "sumsq": np.nan, "n": 0})
        i = states.get((sid, 1), {"sum": np.nan, "sumsq": np.nan, "n": 0})
        if o["n"] <= 0 or i["n"] <= 0:
            rows.append({"subject": sid, "obs_mean": np.nan, "imp_mean": np.nan, "mean_shift": np.nan, "obs_var": np.nan, "imp_var": np.nan, "var_ratio": np.nan, "pass_shift": False})
            continue
        om = o["sum"] / o["n"]
        im = i["sum"] / i["n"]
        ov = max(o["sumsq"] / o["n"] - om * om, 0.0)
        iv = max(i["sumsq"] / i["n"] - im * im, 0.0)
        vr = iv / ov if ov > 1e-12 else np.nan
        ms = im - om
        rows.append(
            {
                "subject": sid,
                "obs_mean": float(om),
                "imp_mean": float(im),
                "mean_shift": float(ms),
                "obs_var": float(ov),
                "imp_var": float(iv),
                "var_ratio": float(vr),
                "pass_shift": bool(np.isfinite(vr) and abs(ms) < 0.25 and 0.5 <= vr <= 2.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    csv_path = Path(args.csv_path).resolve()
    winner_root = Path(args.winner_root).resolve()
    completed_path = Path(args.completed_gtex_path).resolve()
    out_root = Path(args.out_root).resolve()
    tab_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    winner_model, winner_cfg = _winner_config(winner_root)
    genes_all = _read_all_genes(csv_path)

    # Build AHBA parcel map and harmonized AHBA reference using winner harmonizer.
    df = io_utils.read_expression_subset(csv_path, genes_all)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)

    hcfg = SimpleNamespace(combat_use_covariates=True, whiten_eps=1e-4, hier_lambda_a=10.0, hier_lambda_b=10.0, hier_base_method="robustz_affine", hier_subject_subset=sorted(gtex_raw["subject"].astype(str).unique().tolist()))
    harmonizer = fit_harmonizer(ahba_raw, gtex_raw, genes_all, method=str(winner_cfg.get("harm", "combat")), cfg=hcfg)
    ahba_h = harmonizer.transform(ahba_raw, "AHBA")

    # AHBA parcel mean matrix (R x G)
    R = len(target_meta)
    G = len(genes_all)
    ahba_mean = np.full((R, G), np.nan, dtype=np.float64)
    for rid in range(R):
        sub = ahba_h[ahba_h["parcel_idx"] == rid]
        if len(sub):
            ahba_mean[rid, :] = sub[genes_all].to_numpy(dtype=np.float64).mean(axis=0)

    parcel_to_idx = dict(zip(target_meta["tissue_or_parcel"].astype(str).tolist(), target_meta["parcel_idx"].astype(int).tolist()))
    gtex_mean = _stream_parcel_means(completed_path, genes_all, parcel_to_idx, chunk_size=int(args.chunk_size))

    # 1) Marker regional rank correlations.
    marker_present = [g for g in MARKERS if g in genes_all]
    marker_rows = []
    for g in marker_present:
        gi = genes_all.index(g)
        x = ahba_mean[:, gi]
        y = gtex_mean[:, gi]
        m = np.isfinite(x) & np.isfinite(y)
        rho = float(stats.spearmanr(x[m], y[m]).statistic) if np.sum(m) > 3 else np.nan
        p = float(stats.spearmanr(x[m], y[m]).pvalue) if np.sum(m) > 3 else np.nan
        marker_rows.append({"gene": g, "spearman_rho": rho, "p_value": p, "pass_marker": bool(np.isfinite(rho) and rho >= 0.30)})
    marker_df = pd.DataFrame(marker_rows)
    marker_df.to_csv(tab_dir / "marker_region_correlation.csv", index=False)

    # 2) Hemisphere symmetry checks per gene.
    names = target_meta["tissue_or_parcel"].astype(str).tolist()
    idx_by_name = {n: i for i, n in enumerate(names)}
    pairs = []
    for n in names:
        if n.startswith("LH_") or n.startswith("LH-"):
            p = _pair_name(n)
            if p in idx_by_name:
                pairs.append((idx_by_name[n], idx_by_name[p], n, p))

    sym_rows = []
    if len(pairs):
        left_idx = np.asarray([a for a, _, _, _ in pairs], dtype=np.int32)
        right_idx = np.asarray([b for _, b, _, _ in pairs], dtype=np.int32)
        for gi, g in enumerate(genes_all):
            al = ahba_mean[left_idx, gi]
            ar = ahba_mean[right_idx, gi]
            gl = gtex_mean[left_idx, gi]
            gr = gtex_mean[right_idx, gi]
            ma = np.isfinite(al) & np.isfinite(ar)
            mg = np.isfinite(gl) & np.isfinite(gr)
            ca = float(stats.pearsonr(al[ma], ar[ma]).statistic) if np.sum(ma) > 3 and np.std(al[ma]) > 1e-12 and np.std(ar[ma]) > 1e-12 else np.nan
            cg = float(stats.pearsonr(gl[mg], gr[mg]).statistic) if np.sum(mg) > 3 and np.std(gl[mg]) > 1e-12 and np.std(gr[mg]) > 1e-12 else np.nan
            sym_rows.append({"gene": g, "corr_lh_rh_ahba": ca, "corr_lh_rh_gtex": cg, "pass_symmetry": bool(np.isfinite(cg) and cg >= 0.5)})
    sym_df = pd.DataFrame(sym_rows)
    sym_df.to_csv(tab_dir / "hemisphere_symmetry_checks.csv", index=False)

    # 3) Regional outlier scan per gene.
    diff = gtex_mean - ahba_mean
    mu = np.nanmean(diff, axis=0)
    sd = np.nanstd(diff, axis=0)
    z = (diff - mu[None, :]) / np.where(sd[None, :] > 1e-12, sd[None, :], np.nan)
    out_rows = []
    for gi, g in enumerate(genes_all):
        zz = z[:, gi]
        m = np.isfinite(zz)
        if not np.any(m):
            out_rows.append({"gene": g, "n_outlier_parcels_absz_gt3": np.nan, "max_abs_z": np.nan, "pass_outlier": False})
            continue
        n_out = int(np.sum(np.abs(zz[m]) > 3.0))
        maz = float(np.max(np.abs(zz[m])))
        out_rows.append({"gene": g, "n_outlier_parcels_absz_gt3": n_out, "max_abs_z": maz, "pass_outlier": bool(maz < 6.0)})
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(tab_dir / "regional_outlier_scan.csv", index=False)

    # 4) Imputed-vs-observed shift per subject.
    shift_df = _subject_shift_stats(completed_path, genes_all, chunk_size=max(32, int(args.chunk_size // 2)))
    shift_df.to_csv(tab_dir / "imputed_observed_shift.csv", index=False)

    # Figures
    if len(marker_df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        pm = [parcel_to_idx[k] for k in names]
        ahm = np.stack([ahba_mean[:, genes_all.index(g)] for g in marker_present], axis=1)
        gtm = np.stack([gtex_mean[:, genes_all.index(g)] for g in marker_present], axis=1)
        sns.heatmap(ahm, ax=axes[0], cmap="vlag", cbar=True, yticklabels=False, xticklabels=marker_present)
        axes[0].set_title("AHBA markers by parcel")
        sns.heatmap(gtm, ax=axes[1], cmap="vlag", cbar=True, yticklabels=False, xticklabels=marker_present)
        axes[1].set_title("GTEx completed markers by parcel")
        fig.savefig(fig_dir / "marker_heatmap_ahba_vs_gtex.png", dpi=220)
        plt.close(fig)

    if len(sym_df):
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.scatter(sym_df["corr_lh_rh_ahba"], sym_df["corr_lh_rh_gtex"], s=8, alpha=0.35)
        lim = [-1, 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("AHBA LH-RH corr")
        ax.set_ylabel("GTEx LH-RH corr")
        ax.set_title("Hemisphere symmetry agreement")
        fig.savefig(fig_dir / "hemisphere_symmetry_scatter.png", dpi=220)
        plt.close(fig)

    if len(shift_df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        sns.violinplot(y=shift_df["mean_shift"], ax=axes[0], color="#2a9d8f")
        axes[0].set_title("Imputed - observed mean shift")
        vr = np.log2(np.clip(shift_df["var_ratio"].to_numpy(dtype=np.float64), 1e-6, None))
        sns.violinplot(y=vr, ax=axes[1], color="#e76f51")
        axes[1].set_title("log2(var_ratio imputed/observed)")
        fig.savefig(fig_dir / "imputed_observed_shift_violin.png", dpi=220)
        plt.close(fig)

    io_utils.dump_json(
        out_root / "bio_audit_summary.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "winner_model": winner_model,
            "n_genes": int(len(genes_all)),
            "n_markers_present": int(len(marker_present)),
            "marker_pass_fraction": float(marker_df["pass_marker"].mean()) if len(marker_df) else np.nan,
            "symmetry_pass_fraction": float(sym_df["pass_symmetry"].mean()) if len(sym_df) else np.nan,
            "outlier_pass_fraction": float(out_df["pass_outlier"].mean()) if len(out_df) else np.nan,
            "shift_pass_fraction": float(shift_df["pass_shift"].mean()) if len(shift_df) else np.nan,
        },
    )

    print(f"Completed biological audits: {out_root}")


if __name__ == "__main__":
    main()
