#!/usr/bin/env python3
"""
GTEx<->AHBA genetic-axis agreement and 3x3 basis-change hypothesis test.

Outputs under out/axis_basis_change/:
- summary.json
- tables/*.csv
- figures/*.png
- report.md
"""

from __future__ import annotations

import json
import math
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    out_root: str = "out/axis_basis_change"

    random_seed: int = 123
    n_components: int = 3
    n_pca: int = 100

    n_perm: int = 2000


COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")


GTEX_TO_DIV = {
    "brain - cortex": "Frontal",
    "brain - frontal cortex (ba9)": "Frontal",
    "brain - cerebellum": "Cerebellum",
    "brain - cerebellar hemisphere": "Cerebellum",
    "brain - caudate (basal ganglia)": "Caudate",
    "brain - nucleus accumbens (basal ganglia)": "Accumbens",
    "brain - putamen (basal ganglia)": "Putamen",
    "brain - hypothalamus": "Hypothalamus",
    "brain - hippocampus": "Hippocampus",
    "brain - anterior cingulate cortex (ba24)": "AntCing",
    "brain - substantia nigra": "Nigra",
    "brain - amygdala": "Amygdala",
}


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


def compute_pls_axes(df: pd.DataFrame, gene_cols: List[str], cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    X = df[gene_cols].to_numpy(dtype=np.float64)
    Y = df[["coord_y", "coord_z", "coord_abs_x"]].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_pca = int(min(cfg.n_pca, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed)
    Xp = pca.fit_transform(Xs)

    pls = PLSRegression(n_components=cfg.n_components)
    pls.fit(Xp, Y)

    xs = pls.x_scores_.copy()
    ys = pls.y_scores_.copy()

    signs = []
    for c in range(cfg.n_components):
        r = np.corrcoef(xs[:, c], ys[:, c])[0, 1]
        s = 1.0
        if np.isfinite(r) and r < 0:
            s = -1.0
            xs[:, c] *= -1
            ys[:, c] *= -1
        signs.append(float(s))

    out = df[["subject", "tissue_or_parcel", "coord_x", "coord_y", "coord_z", "coord_abs_x"]].copy()
    for c in range(cfg.n_components):
        out[f"X_C{c+1}"] = xs[:, c]
        out[f"Y_C{c+1}"] = ys[:, c]

    meta = {
        "n_samples": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_regions": int(df["tissue_or_parcel"].nunique()),
        "n_genes": int(len(gene_cols)),
        "n_pca": int(n_pca),
        "sign_flips": signs,
    }
    return out, meta


def region_mean_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["X_C1", "X_C2", "X_C3", "Y_C1", "Y_C2", "Y_C3", "coord_x", "coord_y", "coord_z", "coord_abs_x"]
    return score_df.groupby("tissue_or_parcel", as_index=False)[cols].mean()


def build_pairing_nearest(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ah_reg = region_mean_scores(ahba_scores)
    gt_reg = region_mean_scores(gtex_scores)

    Axyz = ah_reg[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    Gxyz = gt_reg[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)

    d2 = ((Gxyz[:, None, :] - Axyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(gt_reg)), idx])

    pair = pd.DataFrame(
        {
            "pairing": "nearest_parcel",
            "gtex_tissue": gt_reg["tissue_or_parcel"].values,
            "ahba_region": ah_reg.loc[idx, "tissue_or_parcel"].values,
            "distance": dist,
        }
    )

    for c in [1, 2, 3]:
        pair[f"G_X_C{c}"] = gt_reg[f"X_C{c}"].values
        pair[f"A_X_C{c}"] = ah_reg.loc[idx, f"X_C{c}"].values
        pair[f"G_Y_C{c}"] = gt_reg[f"Y_C{c}"].values
        pair[f"A_Y_C{c}"] = ah_reg.loc[idx, f"Y_C{c}"].values

    pair["gtex_division"] = pair["gtex_tissue"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))
    return pair.sort_values("gtex_tissue").reset_index(drop=True)


def assign_ahba_divisions_from_nearest_gtex(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ah_reg = region_mean_scores(ahba_scores)
    gt_reg = region_mean_scores(gtex_scores)

    gt_reg = gt_reg.copy()
    gt_reg["division"] = gt_reg["tissue_or_parcel"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))

    Axyz = ah_reg[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    Gxyz = gt_reg[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)

    d2 = ((Axyz[:, None, :] - Gxyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)

    ah_reg = ah_reg.copy()
    ah_reg["division"] = gt_reg.loc[idx, "division"].values
    return ah_reg


def build_pairing_coarse(ahba_scores: pd.DataFrame, gtex_scores: pd.DataFrame) -> pd.DataFrame:
    ah_reg_div = assign_ahba_divisions_from_nearest_gtex(ahba_scores, gtex_scores)
    gt_reg = region_mean_scores(gtex_scores)
    gt_reg = gt_reg.copy()
    gt_reg["division"] = gt_reg["tissue_or_parcel"].map(lambda x: GTEX_TO_DIV.get(str(x).lower(), "Other"))

    Adiv = ah_reg_div.groupby("division", as_index=False)[[f"X_C{i}" for i in [1, 2, 3]] + [f"Y_C{i}" for i in [1, 2, 3]]].mean()
    Gdiv = gt_reg.groupby("division", as_index=False)[[f"X_C{i}" for i in [1, 2, 3]] + [f"Y_C{i}" for i in [1, 2, 3]]].mean()

    shared = sorted(set(Adiv["division"]) & set(Gdiv["division"]))
    Adiv = Adiv[Adiv["division"].isin(shared)].sort_values("division").reset_index(drop=True)
    Gdiv = Gdiv[Gdiv["division"].isin(shared)].sort_values("division").reset_index(drop=True)

    pair = pd.DataFrame({"pairing": "coarse_division", "division": shared})
    for c in [1, 2, 3]:
        pair[f"G_X_C{c}"] = Gdiv[f"X_C{c}"].values
        pair[f"A_X_C{c}"] = Adiv[f"X_C{c}"].values
        pair[f"G_Y_C{c}"] = Gdiv[f"Y_C{c}"].values
        pair[f"A_Y_C{c}"] = Adiv[f"Y_C{c}"].values
    return pair


def zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def fit_so3(X: np.ndarray, Y: np.ndarray) -> Dict:
    C = X.T @ Y
    U, S, Vt = np.linalg.svd(C)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    pred = X @ Q
    return {
        "name": "SO3",
        "matrix": Q,
        "intercept": np.zeros(3),
        "singular_values": S,
        "pred": pred,
    }


def fit_o3(X: np.ndarray, Y: np.ndarray) -> Dict:
    C = X.T @ Y
    U, S, Vt = np.linalg.svd(C)
    Q = U @ Vt
    pred = X @ Q
    return {
        "name": "O3",
        "matrix": Q,
        "intercept": np.zeros(3),
        "singular_values": S,
        "pred": pred,
    }


def fit_affine(X: np.ndarray, Y: np.ndarray) -> Dict:
    X1 = np.c_[X, np.ones(len(X))]
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    M = B[:3, :]
    b = B[3, :]
    pred = X @ M + b
    return {
        "name": "AFFINE",
        "matrix": M,
        "intercept": b,
        "singular_values": np.linalg.svd(X.T @ Y, compute_uv=False),
        "pred": pred,
    }


def evaluate_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    p = []
    s = []
    for i in range(3):
        if np.std(y_true[:, i]) < 1e-12 or np.std(y_pred[:, i]) < 1e-12:
            p.append(np.nan)
            s.append(np.nan)
        else:
            p.append(float(stats.pearsonr(y_true[:, i], y_pred[:, i]).statistic))
            s.append(float(stats.spearmanr(y_true[:, i], y_pred[:, i]).statistic))

    return {
        "global_r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pearson_c1": p[0],
        "pearson_c2": p[1],
        "pearson_c3": p[2],
        "spearman_c1": s[0],
        "spearman_c2": s[1],
        "spearman_c3": s[2],
        "mean_pearson": float(np.nanmean(p)),
        "mean_spearman": float(np.nanmean(s)),
    }


def loro_predict(X: np.ndarray, Y: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    pred = np.zeros_like(Y)
    true = np.zeros_like(Y)

    for i in range(n):
        idx = np.array([j for j in range(n) if j != i], dtype=int)
        Xtr = X[idx]
        Ytr = Y[idx]
        Xte = X[i : i + 1]
        Yte = Y[i : i + 1]

        if model_name == "SO3":
            m = fit_so3(Xtr, Ytr)
            yhat = Xte @ m["matrix"]
        elif model_name == "O3":
            m = fit_o3(Xtr, Ytr)
            yhat = Xte @ m["matrix"]
        elif model_name == "AFFINE":
            m = fit_affine(Xtr, Ytr)
            yhat = Xte @ m["matrix"] + m["intercept"]
        else:
            raise ValueError(model_name)

        pred[i] = yhat[0]
        true[i] = Yte[0]

    return pred, true


def run_permutation_test(X: np.ndarray, Y: np.ndarray, model_name: str, n_perm: int, rng: np.random.Generator) -> Dict:
    obs_pred, obs_true = loro_predict(X, Y, model_name)
    obs = evaluate_pred(obs_true, obs_pred)
    obs_r2 = obs["global_r2"]

    null_r2 = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(len(Y))
        Yp = Y[perm]
        p_pred, p_true = loro_predict(X, Yp, model_name)
        null_r2[i] = evaluate_pred(p_true, p_pred)["global_r2"]

    pval = (np.sum(null_r2 >= obs_r2) + 1) / (n_perm + 1)
    return {
        "observed_loro_r2": float(obs_r2),
        "null_mean_r2": float(np.mean(null_r2)),
        "null_std_r2": float(np.std(null_r2, ddof=0)),
        "perm_p": float(pval),
    }


def flatten_matrix_row(
    pairing: str,
    direction: str,
    model: str,
    M: np.ndarray,
    b: np.ndarray,
    singular_values: np.ndarray,
) -> Dict:
    row = {
        "pairing": pairing,
        "direction": direction,
        "model": model,
        "det": float(np.linalg.det(M)) if M.shape == (3, 3) else np.nan,
        "orthogonality_error": float(np.linalg.norm(M.T @ M - np.eye(3), ord="fro")) if M.shape == (3, 3) else np.nan,
        "condition_number": float(np.linalg.cond(M)) if M.shape == (3, 3) else np.nan,
        "intercept_1": float(b[0]),
        "intercept_2": float(b[1]),
        "intercept_3": float(b[2]),
        "sv_1": float(singular_values[0]),
        "sv_2": float(singular_values[1]),
        "sv_3": float(singular_values[2]),
    }
    for i in range(3):
        for j in range(3):
            row[f"m_{i+1}{j+1}"] = float(M[i, j])
    return row


def make_spatial_overlay_figure(df: pd.DataFrame, dataset_name: str, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    views = [("coord_y", "coord_z", "Y-Z"), ("coord_abs_x", "coord_z", "|X|-Z")]
    for c in [1, 2, 3]:
        vals = df[f"X_C{c}"].to_numpy()
        vmin, vmax = np.percentile(vals, [2, 98])
        for j, (xc, yc, ttl) in enumerate(views):
            ax = axes[c - 1, j]
            sc = ax.scatter(df[xc], df[yc], c=vals, cmap="coolwarm", vmin=vmin, vmax=vmax, s=30, alpha=0.9, linewidths=0)
            ax.set_title(f"{dataset_name} X_C{c} on sites ({ttl})")
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(f"X_C{c}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_pre_post_scatter(
    X: np.ndarray,
    Y: np.ndarray,
    Y_so3: np.ndarray,
    Y_aff: np.ndarray,
    pairing: str,
    direction: str,
    split_label: str,
    out_path: Path,
):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    methods = [("Pre", X), ("SO3", Y_so3), ("Affine", Y_aff)]

    for c in [1, 2, 3]:
        y_true = Y[:, c - 1]
        for j, (name, pred_mat) in enumerate(methods):
            ax = axes[c - 1, j]
            y_pred = pred_mat[:, c - 1]
            ax.scatter(y_true, y_pred, s=30, alpha=0.7)
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
                r = stats.pearsonr(y_true, y_pred).statistic
            else:
                r = np.nan
            ax.set_title(f"C{c} {name} (r={r:.3f})")
            ax.set_xlabel("Target")
            ax.set_ylabel("Predicted")

    fig.suptitle(f"{pairing} | {direction} | {split_label}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_matrix_heatmap_figure(M_so3: np.ndarray, M_o3: np.ndarray, M_aff: np.ndarray, pairing: str, direction: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    mats = [("SO3", M_so3), ("O3", M_o3), ("Affine B", M_aff)]
    for ax, (name, M) in zip(axes, mats):
        sns.heatmap(M, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar=False, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("target axis")
        ax.set_ylabel("source axis")
    fig.suptitle(f"Transform matrices | {pairing} | {direction}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_performance_summary(ins_df: pd.DataFrame, lo_df: pd.DataFrame, out_path: Path):
    ins = ins_df.copy(); ins["split"] = "insample"
    lo = lo_df.copy(); lo["split"] = "loro"
    df = pd.concat([ins, lo], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.barplot(data=df, x="model", y="global_r2", hue="split", ax=axes[0])
    axes[0].set_title("Global R2 by model")
    axes[0].set_xlabel("Model")

    sns.barplot(data=df, x="model", y="mean_pearson", hue="split", ax=axes[1])
    axes[1].set_title("Mean component Pearson by model")
    axes[1].set_xlabel("Model")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_residual_boxplot(loro_point_df: pd.DataFrame, out_path: Path):
    df = loro_point_df.copy()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    sns.boxplot(data=df, x="model", y="residual_norm", hue="pairing_direction", ax=ax)
    ax.set_title("LORO residual magnitude by model and pairing/direction")
    ax.set_xlabel("Model")
    ax.set_ylabel("||residual||")
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_report(
    cfg: Config,
    out_root: Path,
    data_summary: Dict,
    pairings_meta: Dict,
    ins_df: pd.DataFrame,
    lo_df: pd.DataFrame,
    bidir_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    verdict: Dict,
):
    ins_txt = ins_df.to_string(index=False)
    lo_txt = lo_df.to_string(index=False)
    bi_txt = bidir_df.to_string(index=False)
    pm_txt = perm_df.to_string(index=False)

    md = f"""# GTEx<->AHBA Axis Basis-Change Report

## Hypothesis
H1: A simple orthonormal 3x3 matrix (SO(3)) can translate GTEx genetic axes (C1-C3) to AHBA axes and vice versa.

## Data Summary
- AHBA rows: {data_summary['ahba_rows']}
- GTEx rows: {data_summary['gtex_rows']}
- AHBA subjects: {data_summary['ahba_subjects']}
- GTEx subjects: {data_summary['gtex_subjects']}
- AHBA regions: {data_summary['ahba_regions']}
- GTEx regions: {data_summary['gtex_regions']}
- Genes used: {data_summary['n_genes']}

## Pairings
- Nearest-parcel pairs: {pairings_meta['nearest_pairs']}
- Coarse-division pairs: {pairings_meta['coarse_pairs']}

## In-sample alignment metrics
```text
{ins_txt}
```

## Leave-one-region-out (LORO) metrics
```text
{lo_txt}
```

## Bidirectional consistency
```text
{bi_txt}
```

## Permutation falsification (LORO R2)
```text
{pm_txt}
```

## Decision Rule (pre-registered)
Primary evidence: SO(3) on nearest-parcel pairing must satisfy in BOTH directions:
1. LORO global R2 >= 0.50
2. LORO mean component Pearson >= 0.60
3. Permutation p < 0.05

## Verdict
- GTEx->AHBA pass: {verdict['nearest_so3_g2a_pass']}
- AHBA->GTEx pass: {verdict['nearest_so3_a2g_pass']}
- Overall: **{verdict['overall_verdict']}**

## Interpretation
{verdict['interpretation']}

## Figures
See `out/axis_basis_change/figures/` for:
- AHBA and GTEx spatial overlays of genetic scores
- Pre/post transform scatter plots (full and LORO)
- Transform matrix heatmaps
- Performance summaries and residual diagnostics
"""

    (out_root / "report.md").write_text(md)


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.random_seed)

    root = Path(".").resolve()
    csv_path = root / cfg.csv_path

    out_root = root / cfg.out_root
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0)
    gene_cols = header.columns.tolist()[6:]

    # 1) Recompute standalone axes from raw source
    ahba_raw = load_dataset_df(csv_path, "AHBA", gene_cols)
    gtex_raw = load_dataset_df(csv_path, "GTEx", gene_cols)

    ahba_scores, ahba_meta = compute_pls_axes(ahba_raw, gene_cols, cfg)
    gtex_scores, gtex_meta = compute_pls_axes(gtex_raw, gene_cols, cfg)

    ahba_scores.to_csv(table_dir / "ahba_scores.csv", index=False)
    gtex_scores.to_csv(table_dir / "gtex_scores.csv", index=False)

    # 2) Pairings
    nearest_pair = build_pairing_nearest(ahba_scores, gtex_scores)
    coarse_pair = build_pairing_coarse(ahba_scores, gtex_scores)

    nearest_pair.to_csv(table_dir / "pairing_nearest_parcel.csv", index=False)
    coarse_pair.to_csv(table_dir / "pairing_coarse_division.csv", index=False)

    pairings = {
        "nearest_parcel": nearest_pair,
        "coarse_division": coarse_pair,
    }

    transform_rows = []
    ins_rows = []
    lo_rows = []
    bidir_rows = []
    perm_rows = []
    loro_point_rows = []

    models_store = {}

    # 3-4) Fit models and evaluate
    for pname, ptab in pairings.items():
        GX = ptab[["G_X_C1", "G_X_C2", "G_X_C3"]].to_numpy(dtype=np.float64)
        AX = ptab[["A_X_C1", "A_X_C2", "A_X_C3"]].to_numpy(dtype=np.float64)

        GXz, Gmu, Gsd = zscore_cols(GX)
        AXz, Amu, Asd = zscore_cols(AX)

        directions = {
            "GTEx_to_AHBA": (GXz, AXz),
            "AHBA_to_GTEx": (AXz, GXz),
        }

        for dname, (Xsrc, Ytgt) in directions.items():
            # fit models
            m_so3 = fit_so3(Xsrc, Ytgt)
            m_o3 = fit_o3(Xsrc, Ytgt)
            m_aff = fit_affine(Xsrc, Ytgt)

            models = [m_so3, m_o3, m_aff]

            models_store[(pname, dname)] = {
                "SO3": m_so3,
                "O3": m_o3,
                "AFFINE": m_aff,
            }

            # matrix table rows
            for m in models:
                transform_rows.append(
                    flatten_matrix_row(
                        pairing=pname,
                        direction=dname,
                        model=m["name"],
                        M=m["matrix"],
                        b=m["intercept"],
                        singular_values=m["singular_values"],
                    )
                )

            # insample metrics
            for m in models:
                met = evaluate_pred(Ytgt, m["pred"])
                met.update({"pairing": pname, "direction": dname, "model": m["name"]})
                ins_rows.append(met)

            # LORO
            for model_name in ["SO3", "O3", "AFFINE"]:
                pred, true = loro_predict(Xsrc, Ytgt, model_name)
                met = evaluate_pred(true, pred)
                met.update({"pairing": pname, "direction": dname, "model": model_name})
                lo_rows.append(met)

                # point-wise residual diagnostics
                res = np.linalg.norm(true - pred, axis=1)
                for i in range(len(res)):
                    loro_point_rows.append(
                        {
                            "pairing": pname,
                            "direction": dname,
                            "pairing_direction": f"{pname}|{dname}",
                            "model": model_name,
                            "point_index": i,
                            "residual_norm": float(res[i]),
                        }
                    )

                # permutation falsification
                perm = run_permutation_test(Xsrc, Ytgt, model_name, cfg.n_perm, rng)
                perm.update({"pairing": pname, "direction": dname, "model": model_name})
                perm_rows.append(perm)

            # scatter figures (full and loro) using SO3 and Affine
            so3_pred = m_so3["pred"]
            aff_pred = m_aff["pred"]
            make_pre_post_scatter(
                X=Xsrc,
                Y=Ytgt,
                Y_so3=so3_pred,
                Y_aff=aff_pred,
                pairing=pname,
                direction=dname,
                split_label="Full",
                out_path=fig_dir / f"scatter_full_{pname}_{dname}.png",
            )

            so3_loro, y_loro = loro_predict(Xsrc, Ytgt, "SO3")
            aff_loro, _ = loro_predict(Xsrc, Ytgt, "AFFINE")
            make_pre_post_scatter(
                X=Xsrc,
                Y=y_loro,
                Y_so3=so3_loro,
                Y_aff=aff_loro,
                pairing=pname,
                direction=dname,
                split_label="LORO",
                out_path=fig_dir / f"scatter_loro_{pname}_{dname}.png",
            )

            # matrix heatmaps
            make_matrix_heatmap_figure(
                M_so3=m_so3["matrix"],
                M_o3=m_o3["matrix"],
                M_aff=m_aff["matrix"],
                pairing=pname,
                direction=dname,
                out_path=fig_dir / f"matrix_heatmap_{pname}_{dname}.png",
            )

    ins_df = pd.DataFrame(ins_rows)
    lo_df = pd.DataFrame(lo_rows)
    tf_df = pd.DataFrame(transform_rows)
    perm_df = pd.DataFrame(perm_rows)
    loro_point_df = pd.DataFrame(loro_point_rows)

    # 4.3 bidirectional consistency
    for pname in pairings.keys():
        # SO3 consistency
        so3_fwd = models_store[(pname, "GTEx_to_AHBA")]["SO3"]["matrix"]
        so3_rev = models_store[(pname, "AHBA_to_GTEx")]["SO3"]["matrix"]
        err_so3 = float(np.linalg.norm(so3_rev - so3_fwd.T, ord="fro"))

        o3_fwd = models_store[(pname, "GTEx_to_AHBA")]["O3"]["matrix"]
        o3_rev = models_store[(pname, "AHBA_to_GTEx")]["O3"]["matrix"]
        err_o3 = float(np.linalg.norm(o3_rev - o3_fwd.T, ord="fro"))

        aff_fwd = models_store[(pname, "GTEx_to_AHBA")]["AFFINE"]["matrix"]
        aff_rev = models_store[(pname, "AHBA_to_GTEx")]["AFFINE"]["matrix"]
        err_aff = float(np.linalg.norm(aff_rev - np.linalg.pinv(aff_fwd), ord="fro"))

        bidir_rows.extend(
            [
                {"pairing": pname, "model": "SO3", "inverse_consistency_error": err_so3},
                {"pairing": pname, "model": "O3", "inverse_consistency_error": err_o3},
                {"pairing": pname, "model": "AFFINE", "inverse_consistency_error": err_aff},
            ]
        )

    bidir_df = pd.DataFrame(bidir_rows)

    # save tables
    tf_df.to_csv(table_dir / "transform_matrices.csv", index=False)
    ins_df.to_csv(table_dir / "alignment_metrics_insample.csv", index=False)
    lo_df.to_csv(table_dir / "alignment_metrics_loro.csv", index=False)
    bidir_df.to_csv(table_dir / "alignment_metrics_bidirectional.csv", index=False)
    perm_df.to_csv(table_dir / "permutation_test_results.csv", index=False)

    # figures common
    make_spatial_overlay_figure(ahba_scores, "AHBA", fig_dir / "spatial_overlay_ahba_genetic_axes.png")
    make_spatial_overlay_figure(gtex_scores, "GTEx", fig_dir / "spatial_overlay_gtex_genetic_axes.png")
    make_performance_summary(ins_df, lo_df, fig_dir / "performance_summary_models.png")
    make_residual_boxplot(loro_point_df, fig_dir / "residual_diagnostics_loro.png")

    # verdict
    def row_for(df, pairing, direction, model):
        r = df[(df["pairing"] == pairing) & (df["direction"] == direction) & (df["model"] == model)]
        if len(r) != 1:
            raise RuntimeError(f"Missing row for {pairing} {direction} {model}")
        return r.iloc[0]

    g2a = row_for(lo_df, "nearest_parcel", "GTEx_to_AHBA", "SO3")
    a2g = row_for(lo_df, "nearest_parcel", "AHBA_to_GTEx", "SO3")
    p_g2a = row_for(perm_df, "nearest_parcel", "GTEx_to_AHBA", "SO3")
    p_a2g = row_for(perm_df, "nearest_parcel", "AHBA_to_GTEx", "SO3")

    g2a_pass = bool((g2a["global_r2"] >= 0.50) and (g2a["mean_pearson"] >= 0.60) and (p_g2a["perm_p"] < 0.05))
    a2g_pass = bool((a2g["global_r2"] >= 0.50) and (a2g["mean_pearson"] >= 0.60) and (p_a2g["perm_p"] < 0.05))

    if g2a_pass and a2g_pass:
        overall = "SUPPORTED"
        interp = "Nearest-parcel SO(3) passes bidirectionally under LORO and permutation testing, supporting a simple orthonormal basis change."
    else:
        overall = "NOT_SUPPORTED"
        interp = "Nearest-parcel SO(3) fails at least one preregistered criterion (LORO R2, mean Pearson, or permutation p) in one or both directions; this falsifies the strict simple orthonormal translation hypothesis under tested conditions."

    verdict = {
        "nearest_so3_g2a_pass": g2a_pass,
        "nearest_so3_a2g_pass": a2g_pass,
        "overall_verdict": overall,
        "interpretation": interp,
        "thresholds": {
            "loro_global_r2_min": 0.50,
            "loro_mean_pearson_min": 0.60,
            "perm_p_max": 0.05,
        },
    }

    # summary json
    data_summary = {
        "ahba_rows": int(len(ahba_raw)),
        "gtex_rows": int(len(gtex_raw)),
        "ahba_subjects": int(ahba_raw["subject"].nunique()),
        "gtex_subjects": int(gtex_raw["subject"].nunique()),
        "ahba_regions": int(ahba_raw["tissue_or_parcel"].nunique()),
        "gtex_regions": int(gtex_raw["tissue_or_parcel"].nunique()),
        "n_genes": int(len(gene_cols)),
    }

    pairings_meta = {
        "nearest_pairs": int(len(nearest_pair)),
        "coarse_pairs": int(len(coarse_pair)),
    }

    models_meta = {
        "fit_models": ["SO3", "O3", "AFFINE"],
        "directions": ["GTEx_to_AHBA", "AHBA_to_GTEx"],
        "pairings": ["nearest_parcel", "coarse_division"],
    }

    cv_meta = {
        "insample_rows": int(len(ins_df)),
        "loro_rows": int(len(lo_df)),
        "bidirectional_rows": int(len(bidir_df)),
    }

    perm_meta = {
        "n_perm": int(cfg.n_perm),
        "rows": int(len(perm_df)),
    }

    summary = {
        "config": asdict(cfg),
        "data_summary": data_summary,
        "pairings": pairings_meta,
        "models": models_meta,
        "cv": cv_meta,
        "permutation_tests": perm_meta,
        "verdict": verdict,
    }

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    build_report(
        cfg=cfg,
        out_root=out_root,
        data_summary=data_summary,
        pairings_meta=pairings_meta,
        ins_df=ins_df,
        lo_df=lo_df,
        bidir_df=bidir_df,
        perm_df=perm_df,
        verdict=verdict,
    )

    print("Completed axis basis-change analysis.")
    print("Outputs:", out_root)


if __name__ == "__main__":
    sns.set_context("talk")
    sns.set_style("whitegrid")
    main()
