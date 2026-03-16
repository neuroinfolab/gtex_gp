#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _bins_curve(std_vals: np.ndarray, err_vals: np.ndarray, n_bins: int = 10):
    m = np.isfinite(std_vals) & np.isfinite(err_vals)
    s = std_vals[m]
    e = err_vals[m]
    if s.size < 10:
        return np.array([]), np.array([])
    q = pd.qcut(s, q=min(n_bins, max(2, np.unique(s).size)), labels=False, duplicates="drop")
    q = np.asarray(q, dtype=float)
    xs, ys = [], []
    for b in sorted(np.unique(q[np.isfinite(q)]).astype(int).tolist()):
        mm = q == float(b)
        xs.append(float(np.mean(s[mm])))
        ys.append(float(np.mean(e[mm])))
    return np.asarray(xs), np.asarray(ys)


def main():
    ap = argparse.ArgumentParser(description="Compact winner vs reference figure pack")
    ap.add_argument("--unified-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified")
    ap.add_argument("--baseline-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment")
    ap.add_argument("--reference-combo", default="combat__affine_gl3__constrained_anchor__rbf")
    ap.add_argument("--out-dir", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified/figures/compare_winner_vs_reference")
    args = ap.parse_args()

    uroot = Path(args.unified_root)
    broot = Path(args.baseline_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_grid = pd.read_csv(uroot / "tables" / "unified_model_grid_summary_hvg.csv")
    winner = model_grid.sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True]).iloc[0]["model_combo"]

    us = pd.read_csv(uroot / "tables" / "unified_subject_loro_summary_hvg.csv")
    bs = pd.read_csv(broot / "tables" / "subject_loro_summary_hvg.csv")
    uf = pd.read_csv(uroot / "tables" / "unified_subject_loro_folds_hvg.csv")
    bf = pd.read_csv(broot / "tables" / "subject_loro_folds_hvg.csv")

    us_w = us[us["model_combo"] == winner].copy()
    bs_r = bs[bs["model_combo"] == args.reference_combo].copy()

    pair = us_w[["subject", "pearson_r", "rmse", "baseline_rmse"]].merge(
        bs_r[["subject", "pearson_r", "rmse", "baseline_rmse"]],
        on="subject",
        suffixes=("_winner", "_reference"),
        how="inner",
    )
    pair["delta_pearson"] = pair["pearson_r_winner"] - pair["pearson_r_reference"]
    pair["delta_rmse"] = pair["rmse_winner"] - pair["rmse_reference"]

    summary = pd.DataFrame(
        [
            {
                "winner_model": winner,
                "reference_model": args.reference_combo,
                "n_subjects_paired": int(len(pair)),
                "mean_pearson_winner": float(np.mean(pair["pearson_r_winner"])),
                "mean_pearson_reference": float(np.mean(pair["pearson_r_reference"])),
                "mean_rmse_winner": float(np.mean(pair["rmse_winner"])),
                "mean_rmse_reference": float(np.mean(pair["rmse_reference"])),
                "mean_delta_pearson": float(np.mean(pair["delta_pearson"])),
                "median_delta_pearson": float(np.median(pair["delta_pearson"])),
                "mean_delta_rmse": float(np.mean(pair["delta_rmse"])),
                "median_delta_rmse": float(np.median(pair["delta_rmse"])),
                "fraction_better_rmse": float(np.mean(pair["delta_rmse"] < 0.0)),
                "fraction_better_pearson": float(np.mean(pair["delta_pearson"] > 0.0)),
            }
        ]
    )
    summary.to_csv(out / "winner_vs_reference_summary.csv", index=False)
    pair.to_csv(out / "winner_vs_reference_subject_pair.csv", index=False)

    # 1) Distribution overlays
    dist = pd.concat(
        [
            pd.DataFrame({"model": "winner", "pearson_r": pair["pearson_r_winner"], "rmse": pair["rmse_winner"]}),
            pd.DataFrame({"model": "reference", "pearson_r": pair["pearson_r_reference"], "rmse": pair["rmse_reference"]}),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.violinplot(data=dist, x="model", y="pearson_r", inner="quartile", cut=0, ax=axes[0])
    axes[0].set_title("Subject Pearson distribution")
    sns.violinplot(data=dist, x="model", y="rmse", inner="quartile", cut=0, ax=axes[1])
    axes[1].set_title("Subject RMSE distribution")
    fig.savefig(out / "compare_subject_distributions.png", dpi=220)
    plt.close(fig)

    # 2) Paired scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(pair["pearson_r_reference"], pair["pearson_r_winner"], s=14, alpha=0.5)
    lim = [min(pair["pearson_r_reference"].min(), pair["pearson_r_winner"].min()), max(pair["pearson_r_reference"].max(), pair["pearson_r_winner"].max())]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("Reference Pearson")
    axes[0].set_ylabel("Winner Pearson")
    axes[0].set_title("Per-subject Pearson (paired)")

    axes[1].scatter(pair["rmse_reference"], pair["rmse_winner"], s=14, alpha=0.5)
    lim2 = [min(pair["rmse_reference"].min(), pair["rmse_winner"].min()), max(pair["rmse_reference"].max(), pair["rmse_winner"].max())]
    axes[1].plot(lim2, lim2, "k--", lw=1)
    axes[1].set_xlabel("Reference RMSE")
    axes[1].set_ylabel("Winner RMSE")
    axes[1].set_title("Per-subject RMSE (paired)")
    fig.savefig(out / "compare_subject_paired_scatter.png", dpi=220)
    plt.close(fig)

    # 3) Delta hist
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.histplot(pair["delta_pearson"], kde=True, ax=axes[0], color="#2a9d8f")
    axes[0].axvline(0.0, color="k", linestyle="--", lw=1)
    axes[0].set_title("Delta Pearson (winner-reference)")
    sns.histplot(pair["delta_rmse"], kde=True, ax=axes[1], color="#e76f51")
    axes[1].axvline(0.0, color="k", linestyle="--", lw=1)
    axes[1].set_title("Delta RMSE (winner-reference)")
    fig.savefig(out / "compare_subject_delta_hist.png", dpi=220)
    plt.close(fig)

    # 4) Uncertainty reliability (winner) + reference mean error line.
    uw = uf[uf["model_combo"] == winner].copy()
    br = bf[bf["model_combo"] == args.reference_combo].copy()
    xs, ys = _bins_curve(
        uw.get("hold_pred_std", pd.Series(dtype=float)).to_numpy(dtype=float),
        uw.get("hold_abs_error_mean", pd.Series(dtype=float)).to_numpy(dtype=float),
        n_bins=10,
    )
    ref_mean_abs = float(np.nanmean(br.get("hold_abs_error_mean", pd.Series(dtype=float)).to_numpy(dtype=float)))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if xs.size:
        ax.plot(xs, ys, marker="o", lw=2, label="Winner reliability curve")
    ax.axhline(ref_mean_abs, color="crimson", linestyle="--", lw=1.5, label="Reference mean abs error")
    ax.set_xlabel("Predicted std (winner, binned mean)")
    ax.set_ylabel("Empirical abs error")
    ax.set_title("Uncertainty calibration: winner vs reference")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.savefig(out / "compare_uncertainty_reliability.png", dpi=220)
    plt.close(fig)

    # Lightweight markdown summary
    lines = [
        "# Winner vs Reference Compact Comparison",
        "",
        f"- Winner: `{winner}`",
        f"- Reference: `{args.reference_combo}`",
        f"- Paired subjects: `{len(pair)}`",
        "",
        "## Key numbers",
        f"- Mean Pearson (winner vs reference): `{summary.iloc[0]['mean_pearson_winner']:.4f}` vs `{summary.iloc[0]['mean_pearson_reference']:.4f}`",
        f"- Mean RMSE (winner vs reference): `{summary.iloc[0]['mean_rmse_winner']:.4f}` vs `{summary.iloc[0]['mean_rmse_reference']:.4f}`",
        f"- Mean delta Pearson: `{summary.iloc[0]['mean_delta_pearson']:.4f}`",
        f"- Mean delta RMSE: `{summary.iloc[0]['mean_delta_rmse']:.4f}`",
        f"- Fraction better RMSE: `{summary.iloc[0]['fraction_better_rmse']:.3f}`",
        "",
        "## Figures",
        "- compare_subject_distributions.png",
        "- compare_subject_paired_scatter.png",
        "- compare_subject_delta_hist.png",
        "- compare_uncertainty_reliability.png",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(f"Completed compact comparison pack: {out}")


if __name__ == "__main__":
    main()
