#!/usr/bin/env python3
"""
Standalone spatial-transcriptomic alignment for AHBA and GTEx separately.

For each dataset (AHBA, GTEx), this script:
1) Builds subject x region x gene atlas (mean and median)
2) Runs PCA + PLSR/CCA/PLSC between expression and coordinates
3) Evaluates cross-validation (coordinate prediction + component alignment)
4) Generates spatial overlays for transcriptomic scores (X) and spatial scores (Y)
5) Generates X-vs-Y scatter plots (full fit and out-of-fold)
6) Writes dataset-specific and combined reports

Run:
    python3 scripts/run_standalone_spatial_alignment.py
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    out_root: str = "out/standalone_alignment"

    random_seed: int = 123
    n_components: int = 3
    n_pca: int = 100

    cv_splits: int = 10
    cv_repeats: int = 10
    oof_score_repeats: int = 5

    test_fraction: float = 0.25


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


def parse_coordinate_centroid(coord_text: str) -> Tuple[float, float, float]:
    toks = COORD_PATTERN.findall(str(coord_text))
    if not toks:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray([[float(a), float(b), float(c)] for a, b, c in toks], dtype=np.float64)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def load_dataset_df(csv_path: Path, dataset_name: str, gene_cols: List[str]) -> pd.DataFrame:
    usecols = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"] + gene_cols
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    ds = df["dataset"].astype(str).str.upper().str.strip()
    df = df[ds == dataset_name.upper()].copy().reset_index(drop=True)

    xyz = np.vstack([parse_coordinate_centroid(c) for c in df["coordinates"]])
    df["coord_x"] = xyz[:, 0]
    df["coord_y"] = xyz[:, 1]
    df["coord_z"] = xyz[:, 2]
    df["coord_abs_x"] = np.abs(df["coord_x"])
    df = df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    return df


def build_subject_region_atlas(df: pd.DataFrame, gene_cols: List[str], agg: str) -> pd.DataFrame:
    atlas = df.groupby(["subject", "tissue_or_parcel"], as_index=False)[gene_cols].agg(agg)
    reg_coords = (
        df.groupby("tissue_or_parcel", as_index=False)[["coord_x", "coord_y", "coord_z", "coord_abs_x"]]
        .mean()
    )
    atlas = atlas.merge(reg_coords, on="tissue_or_parcel", how="left")
    return atlas


def tensorize_subject_region(atlas_df: pd.DataFrame, gene_cols: List[str]):
    subjects = sorted(atlas_df["subject"].astype(str).unique())
    regions = sorted(atlas_df["tissue_or_parcel"].astype(str).unique())

    sidx = {s: i for i, s in enumerate(subjects)}
    ridx = {r: i for i, r in enumerate(regions)}

    tensor = np.full((len(subjects), len(regions), len(gene_cols)), np.nan, dtype=np.float32)
    for _, row in atlas_df.iterrows():
        i = sidx[str(row["subject"])]
        j = ridx[str(row["tissue_or_parcel"])]
        tensor[i, j, :] = row[gene_cols].to_numpy(dtype=np.float32)
    return tensor, subjects, regions


def manual_cv_predict(estimator_ctor, X: np.ndarray, Y: np.ndarray, cv: KFold) -> np.ndarray:
    pred = np.zeros_like(Y, dtype=np.float64)
    for tr, te in cv.split(X):
        est = estimator_ctor()
        est.fit(X[tr], Y[tr])
        pred[te] = est.predict(X[te])
    return pred


def cross_validate_estimators(
    Xp: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    estimators = {
        "PLSR": lambda nc: PLSRegression(n_components=nc),
        "CCA": lambda nc: CCA(n_components=nc, max_iter=1000),
        "PLSC": lambda nc: PLSCanonical(n_components=nc, max_iter=1000),
    }

    rows = []
    for nc in range(1, n_components + 1):
        for nm, ctor in estimators.items():
            for rep in range(n_repeats):
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
                pred = manual_cv_predict(lambda: ctor(nc), Xp, Y, cv)
                rows.append(
                    {
                        "estimator": nm,
                        "n_components": nc,
                        "repeat": rep,
                        "r2": float(r2_score(Y, pred, multioutput="variance_weighted")),
                        "mae": float(mean_absolute_error(Y, pred)),
                    }
                )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["estimator", "n_components"], as_index=False)[["r2", "mae"]]
        .mean()
        .sort_values(["r2", "mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return raw, summary


def align_full_scores(pls: PLSRegression) -> Tuple[np.ndarray, np.ndarray]:
    xs = pls.x_scores_.copy()
    ys = pls.y_scores_.copy()
    for c in range(xs.shape[1]):
        r = np.corrcoef(xs[:, c], ys[:, c])[0, 1]
        if r < 0:
            ys[:, c] *= -1
    return xs, ys


def compute_test_y_scores(model: PLSRegression, Y_test: np.ndarray) -> np.ndarray:
    y = Y_test - model._y_mean
    if hasattr(model, "_y_std"):
        y = y / model._y_std
    ys = y @ model.y_weights_
    return ys


def oof_component_alignment(
    Xp: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_collect = [[] for _ in range(n_components)]
    y_collect = [[] for _ in range(n_components)]

    for rep in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000 + rep)
        for tr, te in cv.split(Xp):
            mod = PLSRegression(n_components=n_components)
            mod.fit(Xp[tr], Y[tr])

            x_te = mod.transform(Xp[te])
            y_te = compute_test_y_scores(mod, Y[te])

            for c in range(n_components):
                r_train = np.corrcoef(mod.x_scores_[:, c], mod.y_scores_[:, c])[0, 1]
                sign = 1.0 if np.isfinite(r_train) and r_train >= 0 else -1.0
                x_collect[c].append(x_te[:, c])
                y_collect[c].append(sign * y_te[:, c])

    rows = []
    scatter_rows = []
    for c in range(n_components):
        x = np.concatenate(x_collect[c])
        y = np.concatenate(y_collect[c])
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        rows.append(
            {
                "component": c + 1,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
                "n_points": int(len(x)),
            }
        )
        scatter_rows.append(pd.DataFrame({"component": c + 1, "x_score": x, "y_score": y}))

    return pd.DataFrame(rows), pd.concat(scatter_rows, ignore_index=True)


def full_component_alignment(xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in range(xs.shape[1]):
        pr = stats.pearsonr(xs[:, c], ys[:, c])
        sr = stats.spearmanr(xs[:, c], ys[:, c])
        rows.append(
            {
                "component": c + 1,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
                "n_points": int(len(xs[:, c])),
            }
        )
    return pd.DataFrame(rows)


def plot_spatial_overlays(df: pd.DataFrame, score_cols: List[str], title_prefix: str, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    views = [("coord_y", "coord_z", "Y-Z"), ("coord_abs_x", "coord_z", "|X|-Z")]

    for c in range(3):
        comp = score_cols[c]
        vals = df[comp].to_numpy()
        vmin, vmax = np.percentile(vals, [2, 98])
        for j, (xc, yc, ttl) in enumerate(views):
            ax = axes[c, j]
            sc = ax.scatter(
                df[xc],
                df[yc],
                c=vals,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                s=28,
                alpha=0.9,
                linewidths=0,
            )
            ax.set_title(f"{title_prefix} {comp} on sites ({ttl})")
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(comp)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_alignment_scatter(
    df: pd.DataFrame,
    full_alignment: pd.DataFrame,
    oof_scatter: pd.DataFrame,
    dataset_name: str,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for c in range(1, 4):
        # full-fit row
        ax = axes[0, c - 1]
        x = df[f"X_C{c}"].to_numpy()
        y = df[f"Y_C{c}"].to_numpy()
        ax.scatter(x, y, s=22, alpha=0.65)
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, "r-", lw=2)
        row = full_alignment[full_alignment["component"] == c].iloc[0]
        ax.set_title(f"Full C{c}: r={row['pearson_r']:.3f}, rho={row['spearman_rho']:.3f}")
        ax.set_xlabel(f"X_C{c}")
        ax.set_ylabel(f"Y_C{c}")
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)

        # oof row
        ax2 = axes[1, c - 1]
        sub = oof_scatter[oof_scatter["component"] == c]
        x2 = sub["x_score"].to_numpy()
        y2 = sub["y_score"].to_numpy()
        ax2.scatter(x2, y2, s=12, alpha=0.4)
        m2, b2 = np.polyfit(x2, y2, 1)
        xx2 = np.linspace(x2.min(), x2.max(), 100)
        ax2.plot(xx2, m2 * xx2 + b2, "r-", lw=2)
        pr = stats.pearsonr(x2, y2).statistic
        sr = stats.spearmanr(x2, y2).statistic
        ax2.set_title(f"OOF C{c}: r={pr:.3f}, rho={sr:.3f}")
        ax2.set_xlabel(f"OOF X_C{c}")
        ax2.set_ylabel(f"OOF Y_C{c}")
        ax2.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax2.axvline(0, color="k", lw=0.8, alpha=0.5)

    fig.suptitle(f"{dataset_name}: Transcriptomic vs Spatial score alignment")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_coord_prediction_oof(Xp: np.ndarray, Y: np.ndarray, dataset_name: str, out_path: Path, seed: int):
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    pred = manual_cv_predict(lambda: PLSRegression(n_components=3), Xp, Y, cv)

    labels = ["Y", "Z", "|X|"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for i in range(3):
        ax = axes[i]
        ax.scatter(Y[:, i], pred[:, i], s=18, alpha=0.6)
        lo = min(np.min(Y[:, i]), np.min(pred[:, i]))
        hi = max(np.max(Y[:, i]), np.max(pred[:, i]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r2 = r2_score(Y[:, i], pred[:, i])
        mae = mean_absolute_error(Y[:, i], pred[:, i])
        ax.set_title(f"{labels[i]} OOF: R2={r2:.3f}, MAE={mae:.2f}")
        ax.set_xlabel(f"Observed {labels[i]}")
        ax.set_ylabel(f"Predicted {labels[i]}")

    fig.suptitle(f"{dataset_name}: PLSR(3) out-of-fold coordinate prediction")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cv_selection(cv_raw: pd.DataFrame, dataset_name: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.lineplot(data=cv_raw, x="n_components", y="r2", hue="estimator", marker="o", ax=axes[0])
    axes[0].set_title(f"{dataset_name}: CV R2")
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("R2")

    sns.lineplot(data=cv_raw, x="n_components", y="mae", hue="estimator", marker="o", ax=axes[1])
    axes[1].set_title(f"{dataset_name}: CV MAE")
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("MAE")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_coverage_heatmap(coverage_df: pd.DataFrame, dataset_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
    sns.heatmap(coverage_df, cmap="viridis", cbar_kws={"label": "Observed (1/0)"}, ax=ax)
    ax.set_title(f"{dataset_name}: subject-region coverage")
    ax.set_xlabel("Region")
    ax.set_ylabel("Subject")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_single_dataset(dataset_name: str, cfg: Config, gene_cols: List[str], root: Path) -> Dict:
    print(f"\n=== Running standalone pipeline for {dataset_name} ===")
    df = load_dataset_df(root / cfg.csv_path, dataset_name, gene_cols)

    ds_root = root / cfg.out_root / dataset_name.lower()
    atlas_dir = ds_root / "atlas"
    fig_dir = ds_root / "figures"
    tab_dir = ds_root / "tables"
    for p in [ds_root, atlas_dir, fig_dir, tab_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # atlas
    atlas_mean = build_subject_region_atlas(df, gene_cols, "mean")
    atlas_median = build_subject_region_atlas(df, gene_cols, "median")

    mean_tensor, subjects, regions = tensorize_subject_region(atlas_mean, gene_cols)
    median_tensor, _, _ = tensorize_subject_region(atlas_median, gene_cols)

    coverage = np.isfinite(mean_tensor).any(axis=2).astype(int)
    coverage_df = pd.DataFrame(coverage, index=subjects, columns=regions)
    coverage_df.to_csv(tab_dir / "subject_region_coverage_matrix.csv")

    mean_vs_median_absdiff = float(np.nanmean(np.abs(mean_tensor - median_tensor)))
    np.savez_compressed(
        atlas_dir / f"{dataset_name.lower()}_subject_region_gene_atlas.npz",
        mean_tensor=mean_tensor,
        median_tensor=median_tensor,
        subjects=np.asarray(subjects),
        regions=np.asarray(regions),
        genes=np.asarray(gene_cols),
    )

    # matrices
    X = df[gene_cols].to_numpy(dtype=np.float64)
    Y = df[["coord_y", "coord_z", "coord_abs_x"]].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_pca = int(min(cfg.n_pca, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed)
    Xp = pca.fit_transform(Xs)

    pca_var = pd.DataFrame(
        {
            "component": np.arange(1, n_pca + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_var.to_csv(tab_dir / "pca_variance_profile.csv", index=False)

    # CV estimator selection
    cv_raw, cv_summary = cross_validate_estimators(
        Xp,
        Y,
        n_components=cfg.n_components,
        n_splits=cfg.cv_splits,
        n_repeats=cfg.cv_repeats,
        seed=cfg.random_seed,
    )
    cv_raw.to_csv(tab_dir / "estimator_selection_cv_raw.csv", index=False)
    cv_summary.to_csv(tab_dir / "estimator_selection_cv_summary.csv", index=False)

    best = cv_summary.iloc[0]

    # holdout test with PLSR(3)
    Xtr, Xte, Ytr, Yte = train_test_split(
        Xp,
        Y,
        test_size=cfg.test_fraction,
        random_state=cfg.random_seed,
        shuffle=True,
    )
    pls_holdout = PLSRegression(n_components=cfg.n_components)
    pls_holdout.fit(Xtr, Ytr)
    Yte_pred = pls_holdout.predict(Xte)
    holdout = {
        "test_r2": float(r2_score(Yte, Yte_pred, multioutput="variance_weighted")),
        "test_mae": float(mean_absolute_error(Yte, Yte_pred)),
        "n_test": int(len(Yte)),
    }

    # full model scores
    pls = PLSRegression(n_components=cfg.n_components)
    pls.fit(Xp, Y)
    xs, ys = align_full_scores(pls)

    # component alignment full + OOF
    full_align = full_component_alignment(xs, ys)
    full_align.to_csv(tab_dir / "x_vs_y_alignment_fullfit.csv", index=False)

    oof_align, oof_scatter = oof_component_alignment(
        Xp,
        Y,
        n_components=cfg.n_components,
        n_splits=cfg.cv_splits,
        n_repeats=cfg.oof_score_repeats,
        seed=cfg.random_seed,
    )
    oof_align.to_csv(tab_dir / "x_vs_y_alignment_oof.csv", index=False)
    oof_scatter.to_csv(tab_dir / "x_vs_y_alignment_oof_scatter_points.csv", index=False)

    # save sample scores for overlays
    score_df = df[["subject", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "coord_abs_x"]].copy()
    for c in range(cfg.n_components):
        score_df[f"X_C{c+1}"] = xs[:, c]
        score_df[f"Y_C{c+1}"] = ys[:, c]
    score_df.to_csv(tab_dir / "sample_x_y_scores.csv", index=False)

    # plots
    plot_spatial_overlays(score_df, ["X_C1", "X_C2", "X_C3"], "Transcriptomic", fig_dir / "fig1_transcriptomic_scores_overlay.png")
    plot_spatial_overlays(score_df, ["Y_C1", "Y_C2", "Y_C3"], "Spatial", fig_dir / "fig2_spatial_scores_overlay.png")
    plot_alignment_scatter(score_df, full_align, oof_scatter, dataset_name, fig_dir / "fig3_x_vs_y_alignment_full_and_oof.png")
    plot_coord_prediction_oof(Xp, Y, dataset_name, fig_dir / "fig4_coordinate_prediction_oof.png", seed=cfg.random_seed)
    plot_cv_selection(cv_raw, dataset_name, fig_dir / "fig5_estimator_cv_selection.png")
    plot_coverage_heatmap(coverage_df, dataset_name, fig_dir / "fig6_subject_region_coverage_heatmap.png")

    # concise summary json
    summary = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_regions": int(df["tissue_or_parcel"].nunique()),
        "n_genes": int(len(gene_cols)),
        "mean_vs_median_absdiff": mean_vs_median_absdiff,
        "best_estimator": {
            "estimator": str(best["estimator"]),
            "n_components": int(best["n_components"]),
            "cv_r2": float(best["r2"]),
            "cv_mae": float(best["mae"]),
        },
        "holdout": holdout,
        "x_y_alignment_fullfit": full_align.to_dict(orient="records"),
        "x_y_alignment_oof": oof_align.to_dict(orient="records"),
    }
    with open(ds_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{dataset_name}: best={summary['best_estimator']} holdout={holdout}")
    return summary


def write_combined_report(root: Path, cfg: Config, summaries: List[Dict]):
    out_root = root / cfg.out_root
    by_name = {s["dataset"]: s for s in summaries}

    def _fmt_align(s, key):
        rows = s[key]
        lines = []
        for r in rows:
            lines.append(
                f"C{int(r['component'])}: pearson={r['pearson_r']:.3f} (p={r['pearson_p']:.2e}), "
                f"spearman={r['spearman_rho']:.3f} (p={r['spearman_p']:.2e}), n={int(r['n_points'])}"
            )
        return "\n".join(lines)

    ahba = by_name.get("AHBA")
    gtex = by_name.get("GTEX")

    md = [
        "# Standalone AHBA and GTEx Spatial-Transcriptomic Alignment Report",
        "",
        "This report addresses:",
        "1. Does X-vs-Y component alignment hold under cross-validation?",
        "2. Does the same behavior appear in GTEx when analyzed standalone?",
        "",
    ]

    for name, s in [("AHBA", ahba), ("GTEx", gtex)]:
        md.extend(
            [
                f"## {name} standalone results",
                f"- Samples: {s['n_rows']}",
                f"- Subjects: {s['n_subjects']}",
                f"- Regions: {s['n_regions']}",
                f"- Genes: {s['n_genes']}",
                f"- Mean-vs-median atlas abs diff: {s['mean_vs_median_absdiff']:.6f}",
                f"- Best CV estimator: {s['best_estimator']['estimator']}({s['best_estimator']['n_components']}) "
                f"R2={s['best_estimator']['cv_r2']:.3f}, MAE={s['best_estimator']['cv_mae']:.3f}",
                f"- Holdout PLSR(3): R2={s['holdout']['test_r2']:.3f}, MAE={s['holdout']['test_mae']:.3f}",
                "",
                f"Full-fit X-vs-Y alignment:",
                "```text",
                _fmt_align(s, "x_y_alignment_fullfit"),
                "```",
                "",
                f"Out-of-fold (CV) X-vs-Y alignment:",
                "```text",
                _fmt_align(s, "x_y_alignment_oof"),
                "```",
                "",
                f"Figures: `{cfg.out_root}/{name.lower()}/figures/`",
                "",
            ]
        )

    md.extend(
        [
            "## Bottom line",
            "- If OOF component correlations stay high, alignment holds under CV.",
            "- Compare AHBA and GTEx OOF C1-C3 values directly above.",
            "",
        ]
    )

    (out_root / "standalone_alignment_report.md").write_text("\n".join(md))


def main():
    cfg = Config()
    root = Path(".").resolve()

    header = pd.read_csv(root / cfg.csv_path, nrows=0)
    gene_cols = header.columns.tolist()[6:]

    sns.set_context("talk")
    sns.set_style("whitegrid")

    summaries = []
    summaries.append(run_single_dataset("AHBA", cfg, gene_cols, root))
    summaries.append(run_single_dataset("GTEX", cfg, gene_cols, root))

    out_root = root / cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "combined_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    write_combined_report(root, cfg, summaries)
    print(f"\nDone. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
