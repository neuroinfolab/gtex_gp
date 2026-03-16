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


def parse_args():
    p = argparse.ArgumentParser(description="Generate standard evaluation plots/tables from vNext outputs.")
    p.add_argument("--run-root", default="out/vnext_alignment")
    p.add_argument("--baseline-combo", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = Path(a.run_root).resolve()
    tdir = root / "tables"
    fdir = root / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(tdir / "subject_loro_folds_hvg.csv")
    summ = pd.read_csv(tdir / "subject_loro_summary_hvg.csv")
    grid = pd.read_csv(tdir / "model_grid_summary_hvg.csv")

    if len(grid) == 0:
        print("No model summary rows found.")
        return

    baseline_combo = str(a.baseline_combo).strip() or str(grid.iloc[0]["model_combo"])

    # coverage vs performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.scatterplot(data=summ, x="n_obs_parcels", y="pearson_r", hue="model_combo", s=20, alpha=0.6, ax=axes[0], legend=False)
    axes[0].set_title("Coverage vs Pearson")
    axes[0].grid(True, alpha=0.2)
    sns.scatterplot(data=summ, x="n_obs_parcels", y="rmse", hue="model_combo", s=20, alpha=0.6, ax=axes[1], legend=False)
    axes[1].set_title("Coverage vs RMSE")
    axes[1].grid(True, alpha=0.2)
    fig.savefig(fdir / "coverage_vs_performance.png", dpi=220)
    plt.close(fig)

    # baseline vs model scatter
    base = summ[summ["model_combo"] == baseline_combo][["subject", "rmse", "pearson_r"]].rename(columns={"rmse": "base_rmse", "pearson_r": "base_pearson"})
    delta = summ.merge(base, on="subject", how="inner")
    delta = delta[delta["model_combo"] != baseline_combo].copy()
    if len(delta) > 0:
        delta["delta_rmse"] = delta["rmse"] - delta["base_rmse"]
        delta["delta_pearson"] = delta["pearson_r"] - delta["base_pearson"]
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        sns.scatterplot(data=delta, x="delta_rmse", y="delta_pearson", hue="model_combo", s=24, alpha=0.7, ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"Baseline vs model deltas (baseline={baseline_combo})")
        ax.grid(True, alpha=0.2)
        fig.savefig(fdir / "baseline_vs_model_scatter.png", dpi=220)
        plt.close(fig)

    # rollout gate table refresh
    gate = (
        summ.groupby("model_combo", as_index=False)
        .agg(mean_pearson=("pearson_r", "mean"), mean_rmse=("rmse", "mean"), mean_baseline_rmse=("baseline_rmse", "mean"), n_subjects=("subject", "nunique"))
    )
    gate["gate_pass"] = (gate["mean_pearson"] >= 0.50) & (gate["mean_rmse"] < gate["mean_baseline_rmse"])
    gate = gate.sort_values(["gate_pass", "mean_pearson"], ascending=[False, False])
    gate.to_csv(tdir / "rollout_gate_summary.csv", index=False)

    # uncertainty file check
    ufile = tdir / "uncertainty_calibration.csv"
    if ufile.exists():
        uc = pd.read_csv(ufile)
        if len(uc) > 0:
            fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
            ax.plot(uc["pred_std_mean"], uc["abs_err_mean"], marker="o")
            ax.set_title("Uncertainty reliability")
            ax.set_xlabel("Predicted std")
            ax.set_ylabel("Empirical abs error")
            ax.grid(True, alpha=0.2)
            fig.savefig(fdir / "uncertainty_reliability_replot.png", dpi=220)
            plt.close(fig)

    print("Evaluation refresh complete.")
    print(f"Run root: {root}")


if __name__ == "__main__":
    main()
