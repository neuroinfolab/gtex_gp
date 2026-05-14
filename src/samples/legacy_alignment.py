from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .coordinates import (
    BA_ATLAS_KEY_COL,
    GTEX_ATLAS_COL,
    S156_ATLAS_KEY_COL,
    atlas_gtex_to_tissue_label,
    normalize_atlas_key,
)
from .gtex import BRAIN_TISSUE_NAME_MAP


def _normalize_key_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(spearmanr(x, y)[0])


def _pairwise_row_corr(mat: np.ndarray, *, method: str) -> np.ndarray:
    n = mat.shape[0]
    out = np.eye(n, dtype=float)
    corr_fn = _safe_pearson if method == "pearson" else _safe_spearman
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_fn(mat[i, :], mat[j, :])
            out[i, j] = r
            out[j, i] = r
    return out


def read_gene_list(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        # Stability CSVs are stored as region x gene matrices:
        # `label,<gene1>,<gene2>,...` with parcel labels down the first column.
        # For gene-panel selection, the intended flat panel is the set of
        # non-label columns.
        if "label" in df.columns and len(df.columns) > 1:
            cols = [str(c) for c in df.columns if str(c).strip().lower() != "label"]
            if cols:
                return cols
        for col in ("gene", "gene_symbol", "symbol", "gene_name"):
            if col in df.columns:
                return [str(x) for x in df[col].dropna().tolist()]
        return [str(x) for x in df.iloc[:, 0].dropna().tolist()]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def subset_tissue_gene_df(df: pd.DataFrame, genes: list[str] | str | Path | None) -> pd.DataFrame:
    gene_list = read_gene_list(genes) if not isinstance(genes, list) else genes
    if not gene_list:
        return df
    lookup = {str(c).upper(): c for c in df.columns}
    keep = [lookup[str(g).upper()] for g in gene_list if str(g).upper() in lookup]
    return df.loc[:, keep].copy()


def load_gtex_median_tpm_df(
    tpm_file_path: str | Path,
    *,
    gene_id_style: str = "symbol",
    genes: list[str] | str | Path | None = None,
) -> pd.DataFrame:
    """Load GTEx median TPM GCT as a tissue x gene table for legacy alignment."""
    df = pd.read_csv(tpm_file_path, sep="\t", skiprows=2)
    tissue_cols = [c for c in BRAIN_TISSUE_NAME_MAP if c in df.columns]
    if not tissue_cols:
        tissue_cols = [c for c in df.columns if str(c).startswith("Brain_") and "Spinal_cord" not in str(c)]
    if gene_id_style == "symbol":
        gene_index = df["Description"].astype(str)
    elif gene_id_style == "ensembl":
        gene_index = df["Name"].astype(str).str.split(".").str[0]
    else:
        raise ValueError("gene_id_style must be 'symbol' or 'ensembl'")
    out = df[tissue_cols].copy()
    out.index = gene_index
    out = out.T
    out.index = [BRAIN_TISSUE_NAME_MAP.get(t, atlas_gtex_to_tissue_label(t.replace("_", " "))) for t in out.index]
    out = out[~out.index.duplicated(keep="first")]
    out = subset_tissue_gene_df(out, genes)
    out.index.name = "tissue"
    out.columns.name = "gene"
    return out


def aggregate_parcels_to_tissues_by_key(
    ahba_df: pd.DataFrame,
    atlas_df: pd.DataFrame,
    *,
    atlas_key_col: str,
    tissue_aggregation_style: str,
    ahba_key_col: str = "label",
    gtex_col: str = GTEX_ATLAS_COL,
) -> pd.DataFrame:
    if ahba_key_col in ahba_df.columns:
        ahba_keys = _normalize_key_series(ahba_df[ahba_key_col])
        gene_cols = [c for c in ahba_df.columns if c != ahba_key_col]
    else:
        ahba_keys = _normalize_key_series(ahba_df.index.to_series())
        gene_cols = list(ahba_df.columns)

    tissue_to_keys: dict[str, list[str]] = {}
    for _, row in atlas_df.iterrows():
        tissue = atlas_gtex_to_tissue_label(row.get(gtex_col, ""))
        key = normalize_atlas_key(row.get(atlas_key_col))
        if tissue and key:
            tissue_to_keys.setdefault(tissue, []).append(key)

    rows = []
    labels = []
    for tissue, keys in tissue_to_keys.items():
        block = ahba_df.loc[ahba_keys.isin(keys), gene_cols].dropna(how="all")
        if len(block):
            if tissue_aggregation_style == "mean":
                rows.append(block.mean(axis=0, skipna=True))
            else:
                rows.append(block.median(axis=0, skipna=True))
        else:
            rows.append(pd.Series(np.nan, index=gene_cols))
        labels.append(tissue)
    return pd.DataFrame(rows, index=labels, columns=gene_cols)


def aggregate_ahba_to_gtex_tpm(
    true_median_tpm_df: pd.DataFrame,
    *,
    processing_style: str = "raw",
    subject_aggregation_style: str = "median",
    tissue_aggregation_style: str = "median",
    log1p: bool = False,
    mirror_interpolate: bool = False,
    ahba_ba_path: str | Path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/AHBA/BA",
    ahba_ba_atlas_map: str | Path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/atlas_info/AtlasMaps/MaptoBA.csv",
    ahba_s156_path: str | Path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/AHBA/S156",
    ahba_s156_atlas_map: str | Path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/atlas_info/AtlasMaps/MaptoS156.csv",
    symbol_id_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    ba_atlas = pd.read_csv(ahba_ba_atlas_map)
    s156_atlas = pd.read_csv(ahba_s156_atlas_map)
    suffix = "_mirror_interpolate" if mirror_interpolate else ""
    ba_file = Path(ahba_ba_path) / "microarray" / processing_style / f"AHBA_BA_{subject_aggregation_style}{suffix}.csv"
    s156_file = Path(ahba_s156_path) / "microarray" / processing_style / f"AHBA_schaefer156_{subject_aggregation_style}{suffix}.csv"
    ahba_ba = pd.read_csv(ba_file, index_col=0, keep_default_na=True)
    ahba_s156 = pd.read_csv(s156_file, index_col=0, keep_default_na=True)

    cortical = aggregate_parcels_to_tissues_by_key(
        ahba_ba,
        ba_atlas,
        atlas_key_col=BA_ATLAS_KEY_COL,
        tissue_aggregation_style=tissue_aggregation_style,
    )
    subcortical = aggregate_parcels_to_tissues_by_key(
        ahba_s156,
        s156_atlas,
        atlas_key_col=S156_ATLAS_KEY_COL,
        tissue_aggregation_style=tissue_aggregation_style,
    )
    combined = pd.concat([cortical, subcortical], axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]

    target_genes = true_median_tpm_df.columns.tolist()
    target_tissues = true_median_tpm_df.index.tolist()
    col_map = {}
    for c in combined.columns:
        col_map.setdefault(str(c).split(".")[0], c)
    if symbol_id_dict:
        for symbol, eid in symbol_id_dict.items():
            if symbol in combined.columns:
                col_map.setdefault(str(eid).split(".")[0], symbol)

    aligned_pairs = [(g, col_map.get(str(g).split(".")[0])) for g in target_genes]
    aligned_pairs = [(g, c) for g, c in aligned_pairs if c is not None]
    out = pd.DataFrame(index=target_tissues, columns=target_genes, dtype=float)
    common_tissues = combined.index.intersection(target_tissues)
    for target_gene, source_col in aligned_pairs:
        out.loc[common_tissues, target_gene] = combined.loc[common_tissues, source_col].to_numpy()
    if log1p:
        out = np.log1p(out)
    return out


def _alignment_payload(source_df: pd.DataFrame, comparison_df: pd.DataFrame) -> dict[str, object]:
    common_tissues = source_df.index.intersection(comparison_df.index)
    common_genes = source_df.columns.intersection(comparison_df.columns)
    source_full = np.log1p(source_df.loc[common_tissues, common_genes].to_numpy(dtype=float))
    comp_full = comparison_df.loc[common_tissues, common_genes].to_numpy(dtype=float)
    joint_finite = np.isfinite(source_full) & np.isfinite(comp_full)
    overlap_gene_mask = joint_finite.sum(axis=0) >= 2
    common_genes = common_genes[overlap_gene_mask]
    source = source_full[:, overlap_gene_mask]
    comp = comp_full[:, overlap_gene_mask]
    gene_r = np.array([_safe_pearson(source[:, i], comp[:, i]) for i in range(source.shape[1])])
    tissue_r = np.array([_safe_pearson(source[i, :], comp[i, :]) for i in range(source.shape[0])])
    source_corr = _pairwise_row_corr(source, method="pearson")
    comp_corr = _pairwise_row_corr(comp, method="pearson")
    source_sp = _pairwise_row_corr(source, method="spearman")
    comp_sp = _pairwise_row_corr(comp, method="spearman")
    tri = np.triu_indices(source.shape[0], k=1)
    gene_r2 = np.sign(gene_r) * gene_r**2
    tissue_r2 = np.sign(tissue_r) * tissue_r**2
    return {
        "common_tissues": common_tissues,
        "common_genes": common_genes,
        "n_overlap_genes": int(overlap_gene_mask.sum()),
        "source": source,
        "comparison": comp,
        "gene_r": gene_r,
        "tissue_r": tissue_r,
        "gene_r2": gene_r2,
        "tissue_r2": tissue_r2,
        "source_corr": source_corr,
        "comparison_corr": comp_corr,
        "source_spearman_corr": source_sp,
        "comparison_spearman_corr": comp_sp,
        "rsa_pearson": _safe_pearson(source_corr[tri], comp_corr[tri]),
        "rsa_spearman": _safe_pearson(source_sp[tri], comp_sp[tri]),
    }


def evaluate_tissue_gene_alignment(source_df: pd.DataFrame, comparison_df: pd.DataFrame) -> dict[str, float]:
    payload = _alignment_payload(source_df, comparison_df)
    return {
        "n_common_tissues": int(len(payload["common_tissues"])),
        "n_overlap_genes": int(payload["n_overlap_genes"]),
        "mean_gene_r": float(np.nanmean(payload["gene_r"])),
        "mean_tissue_r": float(np.nanmean(payload["tissue_r"])),
        "rsa_pearson": float(payload["rsa_pearson"]),
        "rsa_spearman": float(payload["rsa_spearman"]),
    }


def evaluate_gtex_tpm_df(
    source_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    show_metric: str = "r",
    silence_plotting: bool = False,
    figsize_heatmaps: tuple[float, float] = (16, 5),
    figsize_corrmats: tuple[float, float] = (10, 5),
    cmap_heatmap: str = "viridis",
    cmap_corrmat: str = "RdBu_r",
) -> dict[str, object]:
    """Original-notebook style GTEx-vs-AHBA tissue/gene evaluation."""
    if show_metric not in {"r", "r2"}:
        raise ValueError("show_metric must be 'r' or 'r2'")
    payload = _alignment_payload(source_df, comparison_df)
    metrics = {
        "n_common_tissues": int(len(payload["common_tissues"])),
        "n_overlap_genes": int(payload["n_overlap_genes"]),
        "mean_gene_r": float(np.nanmean(payload["gene_r"])),
        "mean_tissue_r": float(np.nanmean(payload["tissue_r"])),
        "mean_gene_r2": float(np.nanmean(payload["gene_r2"])),
        "mean_tissue_r2": float(np.nanmean(payload["tissue_r2"])),
        "std_gene_r": float(np.nanstd(payload["gene_r"])),
        "std_tissue_r": float(np.nanstd(payload["tissue_r"])),
        "std_gene_r2": float(np.nanstd(payload["gene_r2"])),
        "std_tissue_r2": float(np.nanstd(payload["tissue_r2"])),
        "rsa_pearson": float(payload["rsa_pearson"]),
        "rsa_spearman": float(payload["rsa_spearman"]),
    }
    if silence_plotting:
        return metrics
    figs = plot_gtex_tpm_evaluation(
        payload,
        metrics,
        show_metric=show_metric,
        figsize_heatmaps=figsize_heatmaps,
        figsize_corrmats=figsize_corrmats,
        cmap_heatmap=cmap_heatmap,
        cmap_corrmat=cmap_corrmat,
    )
    return {**metrics, "figures": figs}


def evaluate_ahba_combination_grid(
    source_df: pd.DataFrame,
    *,
    processing_styles: list[str] | None = None,
    subject_aggregation_styles: list[str] | None = None,
    tissue_aggregation_styles: list[str] | None = None,
    log1p_options: list[bool] | None = None,
    mirror_interpolate_options: list[bool] | None = None,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for processing, subject_agg, tissue_agg, log1p, mirror in itertools.product(
        processing_styles or ["raw", "minmax", "srs"],
        subject_aggregation_styles or ["mean", "median"],
        tissue_aggregation_styles or ["mean", "median"],
        log1p_options or [True, False],
        mirror_interpolate_options or [True, False],
    ):
        comp = aggregate_ahba_to_gtex_tpm(
            source_df,
            processing_style=processing,
            subject_aggregation_style=subject_agg,
            tissue_aggregation_style=tissue_agg,
            log1p=log1p,
            mirror_interpolate=mirror,
            **kwargs,
        )
        row = {
            "processing_style": processing,
            "subject_aggregation_style": subject_agg,
            "tissue_aggregation_style": tissue_agg,
            "log1p": log1p,
            "mirror_interpolate": mirror,
            **evaluate_tissue_gene_alignment(source_df, comp),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    metric_cols = ["mean_gene_r", "mean_tissue_r", "rsa_pearson", "rsa_spearman"]
    df["mean"] = df[metric_cols].mean(axis=1)
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def plot_gtex_tpm_evaluation(
    payload: dict[str, object],
    metrics: dict[str, float],
    *,
    show_metric: str = "r",
    figsize_heatmaps: tuple[float, float] = (17, 5.5),
    figsize_corrmats: tuple[float, float] = (12, 6.5),
    cmap_heatmap: str = "viridis",
    cmap_corrmat: str = "RdBu_r",
):
    import matplotlib.pyplot as plt

    source = payload["source"]
    comp = payload["comparison"]
    labels = [str(x).replace("brain - ", "") for x in payload["common_tissues"]]
    gene_scores = payload["gene_r2"] if show_metric == "r2" else payload["gene_r"]
    tissue_scores = payload["tissue_r2"] if show_metric == "r2" else payload["tissue_r"]
    metric_label = "R2" if show_metric == "r2" else "Pearson r"
    n_genes = len(payload["common_genes"])
    xtick_idx = np.linspace(0, max(n_genes - 1, 0), min(100, n_genes), dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(figsize_heatmaps[0], 2 * (figsize_heatmaps[1] + 1)), dpi=160)
    im0 = axes[0].imshow(source, aspect="auto", cmap="viridis")
    axes[0].set_title("Source GTEx log(TPM+1)")
    axes[0].set_ylabel("Tissue")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xticks(xtick_idx)
    axes[0].set_xticklabels(["" for _ in xtick_idx])
    ax0r = axes[0].twinx()
    ax0r.set_ylim(axes[0].get_ylim())
    ax0r.set_yticks(np.arange(len(labels)))
    ax0r.set_yticklabels(["1.00" for _ in labels])
    ax0r.set_ylabel(f"Tissue {metric_label}")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.08)
    cbar0.set_ticks([])
    im1 = axes[1].imshow(comp, aspect="auto", cmap=cmap_heatmap)
    axes[1].set_title("AHBA aggregated to GTEx tissues")
    axes[1].set_xlabel(f"Gene ({metric_label})")
    axes[1].set_ylabel("Tissue")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xticks(xtick_idx)
    axes[1].set_xticklabels([f"{gene_scores[i]:.2f}" for i in xtick_idx], rotation=90, fontsize=6)
    ax1r = axes[1].twinx()
    ax1r.set_ylim(axes[1].get_ylim())
    ax1r.set_yticks(np.arange(len(labels)))
    ax1r.set_yticklabels([f"{v:.2f}" for v in tissue_scores], fontsize=7)
    ax1r.set_ylabel(f"Tissue {metric_label}")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.08)
    cbar1.set_ticks([])
    fig.tight_layout(rect=(0, 0, 0.94, 1))

    fig_corr, corr_axes = plt.subplots(2, 2, figsize=(figsize_corrmats[0] * 1.4, figsize_corrmats[1] * 1.8), dpi=160)
    for ax, mat, title in [
        (corr_axes[0, 0], payload["source_corr"], "GTEx source tissue correlation\n(Pearson)"),
        (corr_axes[0, 1], payload["source_spearman_corr"], "GTEx source tissue correlation\n(Spearman)"),
        (corr_axes[1, 0], payload["comparison_corr"], "AHBA comparison tissue correlation\n(Pearson)"),
        (corr_axes[1, 1], payload["comparison_spearman_corr"], "AHBA comparison tissue correlation\n(Spearman)"),
    ]:
        im = ax.imshow(mat, aspect="equal", cmap=cmap_corrmat, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_corr.suptitle(
        (
            f"GTEx vs AHBA tissue-RSA | Pearson={metrics['rsa_pearson']:.3f} | "
            f"Spearman={metrics['rsa_spearman']:.3f} | overlap genes={metrics['n_overlap_genes']}"
        ),
        y=1.02,
    )
    fig_corr.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, fig_corr


def plot_tissue_gene_alignment(source_df: pd.DataFrame, comparison_df: pd.DataFrame, *, show_metric: str = "r"):
    metrics = evaluate_gtex_tpm_df(source_df, comparison_df, show_metric=show_metric, silence_plotting=False)
    return metrics["figures"]
