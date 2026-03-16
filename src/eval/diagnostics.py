from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def summarize_distance_effect(df: pd.DataFrame, strategy_col: str = "strategy") -> pd.DataFrame:
    rows = []
    for key, d in df.groupby(strategy_col):
        x = d["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
        y = d["misfit_1mpearson"].to_numpy(dtype=np.float64)
        pear = float(stats.pearsonr(x, y).statistic) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan
        spear = float(stats.spearmanr(x, y).statistic)
        b1, b0 = np.polyfit(x, y, deg=1)
        q = pd.qcut(x, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        qmean = d.groupby(q, observed=False)["misfit_1mpearson"].mean().to_dict()
        rows.append(
            {
                strategy_col: key,
                "pearson_misfit_distance": pear,
                "spearman_misfit_distance": spear,
                "linear_slope": float(b1),
                "linear_intercept": float(b0),
                "q1_mean_misfit": float(qmean.get("Q1", np.nan)),
                "q2_mean_misfit": float(qmean.get("Q2", np.nan)),
                "q3_mean_misfit": float(qmean.get("Q3", np.nan)),
                "q4_mean_misfit": float(qmean.get("Q4", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def plot_misfit_vs_distance(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    obs = df[df["is_observed_gtex"]]
    unobs = df[~df["is_observed_gtex"]]
    ax.scatter(unobs["dist_to_nearest_observed"], unobs["misfit_1mpearson"], s=24, alpha=0.75, label="unobserved", marker="^")
    ax.scatter(obs["dist_to_nearest_observed"], obs["misfit_1mpearson"], s=28, alpha=0.9, label="observed", marker="o")
    x = df["dist_to_nearest_observed"].to_numpy(dtype=np.float64)
    y = df["misfit_1mpearson"].to_numpy(dtype=np.float64)
    b1, b0 = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xx, b1 * xx + b0, color="black", linewidth=2.0, label=f"slope={b1:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Distance to nearest observed GTEx parcel")
    ax.set_ylabel("Misfit (1 - region Pearson in harmonized space)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def calibration_gene_stats(y_true: np.ndarray, y_pred: np.ndarray, gene_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for gi, g in enumerate(gene_names):
        yt = y_true[:, gi]
        yp = y_pred[:, gi]
        m = np.isfinite(yt) & np.isfinite(yp)
        if int(m.sum()) < 2:
            rows.append({"gene": g, "slope": np.nan, "intercept": np.nan, "r2": np.nan})
            continue
        b1, b0 = np.polyfit(yp[m], yt[m], deg=1)
        yhat = b1 * yp[m] + b0
        sse = float(np.sum((yt[m] - yhat) ** 2))
        sst = float(np.sum((yt[m] - np.mean(yt[m])) ** 2))
        r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan
        rows.append({"gene": g, "slope": float(b1), "intercept": float(b0), "r2": float(r2)})
    return pd.DataFrame(rows)


def plot_calibration_density(calib_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sns.kdeplot(calib_df["slope"].dropna(), ax=axes[0], fill=True)
    axes[0].set_title("Slope distribution")
    axes[0].set_xlabel("Pred->Obs slope")
    sns.kdeplot(calib_df["intercept"].dropna(), ax=axes[1], fill=True)
    axes[1].set_title("Intercept distribution")
    axes[1].set_xlabel("Pred->Obs intercept")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def uncertainty_calibration(
    pred_std: np.ndarray,
    abs_error: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    m = np.isfinite(pred_std) & np.isfinite(abs_error)
    ps = pred_std[m]
    ae = abs_error[m]
    if ps.size == 0:
        return pd.DataFrame(columns=["bin", "pred_std_mean", "abs_err_mean", "n"])
    n_unique = int(np.unique(ps).size)
    if n_unique <= 1:
        return pd.DataFrame(
            [
                {
                    "bin": 0,
                    "pred_std_mean": float(np.mean(ps)),
                    "abs_err_mean": float(np.mean(ae)),
                    "n": int(ps.size),
                }
            ]
        )
    q = pd.qcut(ps, q=min(n_bins, max(2, n_unique)), labels=False, duplicates="drop")
    q_arr = np.asarray(q, dtype=np.float64)
    valid = np.isfinite(q_arr)
    if not np.any(valid):
        return pd.DataFrame(columns=["bin", "pred_std_mean", "abs_err_mean", "n"])
    rows = []
    for b in sorted(np.unique(q_arr[valid]).astype(int).tolist()):
        bb = q_arr == float(b)
        rows.append(
            {
                "bin": b,
                "pred_std_mean": float(np.mean(ps[bb])),
                "abs_err_mean": float(np.mean(ae[bb])),
                "n": int(np.sum(bb)),
            }
        )
    return pd.DataFrame(rows)


def compute_uncertainty_calibration(folds_df: pd.DataFrame) -> pd.DataFrame:
    if len(folds_df) == 0:
        return pd.DataFrame(columns=["model_combo", "bin", "pred_std_mean", "abs_err_mean", "n"])
    rows = []
    for model, d in folds_df.groupby("model_combo"):
        dd = uncertainty_calibration(
            d.get("hold_pred_std", pd.Series(dtype=float)).to_numpy(dtype=np.float64),
            d.get("hold_abs_error_mean", pd.Series(dtype=float)).to_numpy(dtype=np.float64),
        )
        if len(dd) == 0:
            continue
        dd.insert(0, "model_combo", model)
        rows.append(dd)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model_combo", "bin", "pred_std_mean", "abs_err_mean", "n"])


def plot_uncertainty_reliability(calib_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if len(calib_df) > 0:
        for model, d in calib_df.groupby("model_combo"):
            ax.plot(d["pred_std_mean"], d["abs_err_mean"], marker="o", linewidth=1.4, alpha=0.8, label=model[:30])
        if calib_df["model_combo"].nunique() <= 8:
            ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Predicted std (binned mean)")
    ax.set_ylabel("Empirical absolute error (binned mean)")
    ax.set_title("Uncertainty reliability")
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_uvar_vs_error(pred_std: np.ndarray, abs_error: np.ndarray, out_path: Path) -> float:
    m = np.isfinite(pred_std) & np.isfinite(abs_error)
    x = pred_std[m]
    y = abs_error[m]
    corr = float(stats.pearsonr(x, y).statistic) if x.size > 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(x, y, s=8, alpha=0.25)
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"uvar vs error (pearson={corr:.3f})" if np.isfinite(corr) else "uvar vs error")
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return corr


def compute_tail_risk(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) == 0:
        return pd.DataFrame(columns=["model_combo", "p90_rmse", "p95_rmse", "p10_pearson", "p05_pearson"])
    rows = []
    for model, d in summary_df.groupby("model_combo"):
        rm = d["rmse"].to_numpy(dtype=np.float64)
        pr = d["pearson_r"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model_combo": model,
                "p90_rmse": float(np.nanpercentile(rm, 90)),
                "p95_rmse": float(np.nanpercentile(rm, 95)),
                "p10_pearson": float(np.nanpercentile(pr, 10)),
                "p05_pearson": float(np.nanpercentile(pr, 5)),
            }
        )
    return pd.DataFrame(rows)


def paired_method_tests(subject_method_df: pd.DataFrame, metric: str = "rmse", reference_model: str | None = None) -> pd.DataFrame:
    if len(subject_method_df) == 0:
        return pd.DataFrame(columns=["model_combo", "reference_model", "metric", "n_pairs", "median_delta", "mean_delta", "wilcoxon_p", "sign_p", "rank_biserial"])

    models = sorted(subject_method_df["model_combo"].astype(str).unique().tolist())
    ref = reference_model if reference_model in models else models[0]
    ref_df = subject_method_df[subject_method_df["model_combo"] == ref][["subject", metric]].rename(columns={metric: "ref_metric"})

    rows = []
    for model in models:
        if model == ref:
            continue
        cur = subject_method_df[subject_method_df["model_combo"] == model][["subject", metric]].rename(columns={metric: "cur_metric"})
        m = cur.merge(ref_df, on="subject", how="inner")
        if len(m) == 0:
            continue
        delta = (m["cur_metric"] - m["ref_metric"]).to_numpy(dtype=np.float64)
        try:
            w = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox", correction=False)
            p_w = float(w.pvalue)
        except Exception:
            p_w = np.nan

        pos = int(np.sum(delta > 0))
        neg = int(np.sum(delta < 0))
        n_eff = pos + neg
        if n_eff > 0:
            b = stats.binomtest(min(pos, neg), n=n_eff, p=0.5, alternative="two-sided")
            p_s = float(b.pvalue)
            rb = float((pos - neg) / n_eff)
        else:
            p_s = np.nan
            rb = np.nan

        rows.append(
            {
                "model_combo": model,
                "reference_model": ref,
                "metric": metric,
                "n_pairs": int(len(delta)),
                "median_delta": float(np.nanmedian(delta)),
                "mean_delta": float(np.nanmean(delta)),
                "wilcoxon_p": p_w,
                "sign_p": p_s,
                "rank_biserial": rb,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Holm correction on Wilcoxon p-values.
    ps = out["wilcoxon_p"].to_numpy(dtype=np.float64)
    order = np.argsort(np.nan_to_num(ps, nan=1.0))
    m = int(len(ps))
    holm = np.full(m, np.nan, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        p = ps[idx]
        if not np.isfinite(p):
            holm[idx] = np.nan
            continue
        adj = min(1.0, p * (m - rank))
        running = max(running, adj)
        holm[idx] = running
    out["wilcoxon_p_holm"] = holm
    return out
