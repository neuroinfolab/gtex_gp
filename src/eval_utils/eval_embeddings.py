#!/usr/bin/env python3
from __future__ import annotations

"""
Embedding diagnostics for expression embedding matrices.

Rows are subject-region samples; columns are genes. PREPOST is the clean source
for pre/post ComBat matrices, while `TensorView` is an adapter for visualizer,
LORO, and fullfit-derived views. Both routes emit the same
`RegionEmbeddingMatrix` contract before PCA/UMAP.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.preprocess import macro_system as infer_macro_system

from .eval_samples import JointTensorView, TensorView
from .eval_style import (
    REGION_SCATTER_GROUP_BASE,
    REGION_SCATTER_GROUP_ORDER,
    apply_tick_style,
    collapse_macro_system_to_region_group,
    format_legend_label,
    ordered_region_scatter_values,
    region_scatter_palette,
    set_academic_style,
)


FEATURE_PREPROCESS_OPTIONS = ("center", "standardize", "none")
EMBEDDING_COLOR_OPTIONS = (
    "region",
    "gtex_native_region",
    "macro_system",
    "region_group",
    "age",
    "sex",
    "dataset",
    # gene-wise (rows = genes)
    "peak_region",
    "peak_region_group",
    "gene_mean",
    "gene_cv",
    "network_label",
    "network_label_17network",
)
_CONTINUOUS_COLOR_KEYS = {"age", "gene_mean", "gene_cv"}
_SEX_COLORS = {"female": "#f4a261", "f": "#f4a261", "male": "#7fc97f", "m": "#7fc97f"}
_DATASET_COLORS = {"GTEx": "#4c78a8", "AHBA": "#f58518"}
_DATASET_MARKERS = {"GTEx": "o", "AHBA": "^"}
_DATASET_POINT_SIZES = {"GTEx": 5.0, "AHBA": 44.0}
_DATASET_ALPHAS = {"GTEx": 0.36, "AHBA": 0.92}
_MACRO_SYSTEM_COLORS = {
    "visual_somatomotor": "#fdb863",
    "vis_somatomotor": "#fdb863",
    "cortical_association": "#d7301f",
    "association": "#d7301f",
    "subcortical": "#1f78b4",
    "basal_ganglia": "#6a3d9a",
    "limbic_midbrain": "#8c6bb1",
    "cerebellar": "#1b9e77",
    "other": "#6b6b6b",
    "unknown": "#969696",
}
_NETWORK_LABEL_COLORS = {
    "Vis": "#fdb863",
    "SomMot": "#d95f02",
    "DorsAttn": "#1f78b4",
    "SalVentAttn": "#33a02c",
    "Limbic": "#b15928",
    "Cont": "#6a3d9a",
    "Default": "#d7301f",
    "Subcortical": "#7570b3",
    "Cerebellar": "#1b9e77",
    "Unknown": "#969696",
}


@dataclass
class RegionEmbeddingMatrix:
    X: np.ndarray
    metadata: pd.DataFrame
    genes: list[str]
    stage: str
    dataset: str
    region_axis_kind: str
    row_axis: str = "subject_region"
    dropped_rows: int = 0


@dataclass
class FeaturePreprocessStats:
    mode: str
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    zero_scale_features: int = 0


@dataclass
class EmbeddingResult:
    coordinates: pd.DataFrame
    method: str
    feature_preprocess: str
    stage: str
    dataset: str
    region_axis_kind: str
    genes: list[str]
    fit_info: dict[str, object] = field(default_factory=dict)


def _normalize_feature_preprocess(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in FEATURE_PREPROCESS_OPTIONS:
        raise ValueError(f"feature_preprocess must be one of {FEATURE_PREPROCESS_OPTIONS} (got {mode!r})")
    return normalized


def preprocess_expression_features(
    X: np.ndarray,
    mode: str = "center",
) -> tuple[np.ndarray, FeaturePreprocessStats]:
    """Apply gene-wise preprocessing before PCA/UMAP.

    `center` subtracts each gene mean, `standardize` additionally divides by
    each gene standard deviation, and `none` leaves expression values unchanged.
    """
    mode = _normalize_feature_preprocess(mode)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values; filter or impute before embedding")
    if mode == "none":
        return X_arr.copy(), FeaturePreprocessStats(mode=mode)

    mean = X_arr.mean(axis=0)
    X_proc = X_arr - mean
    if mode == "center":
        return X_proc, FeaturePreprocessStats(mode=mode, mean=mean)

    scale = X_proc.std(axis=0, ddof=0)
    zero = ~np.isfinite(scale) | (scale <= 0)
    scale_safe = scale.copy()
    scale_safe[zero] = 1.0
    X_proc = X_proc / scale_safe
    return X_proc, FeaturePreprocessStats(
        mode=mode,
        mean=mean,
        scale=scale_safe,
        zero_scale_features=int(zero.sum()),
    )


def build_region_embedding_matrix_from_tensor_view(
    view: TensorView,
    *,
    require_all_genes_finite: bool = True,
) -> RegionEmbeddingMatrix:
    """Flatten a `TensorView` into subject-region rows x gene columns."""
    values = np.asarray(view.values, dtype=np.float64)
    obs_mask = np.asarray(view.observed_mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"TensorView values must be 3D subject x region x gene, got {values.shape}")
    if obs_mask.shape != values.shape[:2]:
        raise ValueError(f"observed_mask shape {obs_mask.shape} does not match values shape {values.shape[:2]}")

    row_mask = obs_mask.copy()
    if require_all_genes_finite:
        row_mask &= np.isfinite(values).all(axis=2)
    else:
        row_mask &= np.isfinite(values).any(axis=2)
    subj_i, region_i = np.where(row_mask)
    X = values[subj_i, region_i, :]

    future = view.future_imputation_mask
    if future is None:
        future_vals = np.zeros(len(subj_i), dtype=bool)
    else:
        future_vals = np.asarray(future, dtype=bool)[subj_i, region_i]

    regions = [str(r) for r in view.regions]
    macro = [infer_macro_system(regions[j]) for j in region_i.tolist()]
    region_group = [collapse_macro_system_to_region_group(v) for v in macro]
    stage = str(view.pipeline_stage or "unknown")
    meta = pd.DataFrame(
        {
            "dataset": str(view.dataset),
            "subject": [str(view.subjects[i]) for i in subj_i.tolist()],
            "subject_index": subj_i.astype(int),
            "region_index": region_i.astype(int),
            "region": [regions[j] for j in region_i.tolist()],
            "macro_system": macro,
            "region_group": region_group,
            "stage": stage,
            "pipeline_stage": stage,
            "region_axis_kind": str(view.region_axis_kind),
            "future_imputation": future_vals.astype(bool),
        }
    )
    dropped = int(obs_mask.sum()) - int(row_mask.sum())
    return RegionEmbeddingMatrix(
        X=X,
        metadata=meta,
        genes=[str(g) for g in view.genes],
        stage=stage,
        dataset=str(view.dataset),
        region_axis_kind=str(view.region_axis_kind),
        dropped_rows=dropped,
    )


def build_region_embedding_matrix_from_joint_tensor_view(
    joint_view: JointTensorView,
    *,
    require_all_genes_finite: bool = True,
) -> RegionEmbeddingMatrix:
    """Flatten a `JointTensorView` into subject-region rows x gene columns."""
    values = np.asarray(joint_view.values, dtype=np.float64)
    obs_mask = np.asarray(joint_view.observed_mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"JointTensorView values must be 3D subject x region x gene, got {values.shape}")
    if obs_mask.shape != values.shape[:2]:
        raise ValueError(f"observed_mask shape {obs_mask.shape} does not match values shape {values.shape[:2]}")

    row_mask = obs_mask.copy()
    if require_all_genes_finite:
        row_mask &= np.isfinite(values).all(axis=2)
    else:
        row_mask &= np.isfinite(values).any(axis=2)
    subj_i, region_i = np.where(row_mask)
    X = values[subj_i, region_i, :]

    future = np.asarray(joint_view.future_imputation_mask, dtype=bool)
    future_vals = future[subj_i, region_i]
    regions = [str(r) for r in joint_view.regions]
    datasets = [str(joint_view.subject_datasets[i]) for i in subj_i.tolist()]
    stage_by_dataset = {
        str(k): str(getattr(v, "pipeline_stage", None) or "unknown")
        for k, v in joint_view.source_views.items()
    }
    pipeline_stage = [stage_by_dataset.get(ds, "unknown") for ds in datasets]
    macro = [infer_macro_system(regions[j]) for j in region_i.tolist()]
    region_group = [collapse_macro_system_to_region_group(v) for v in macro]
    meta = pd.DataFrame(
        {
            "dataset": datasets,
            "subject": [str(joint_view.subjects[i]) for i in subj_i.tolist()],
            "subject_index": subj_i.astype(int),
            "region_index": region_i.astype(int),
            "region": [regions[j] for j in region_i.tolist()],
            "macro_system": macro,
            "region_group": region_group,
            "stage": pipeline_stage,
            "pipeline_stage": pipeline_stage,
            "region_axis_kind": str(joint_view.region_axis_kind),
            "future_imputation": future_vals.astype(bool),
        }
    )

    unique_stage = sorted(set(pipeline_stage))
    stage = unique_stage[0] if len(unique_stage) == 1 else "joint"
    dropped = int(obs_mask.sum()) - int(row_mask.sum())
    return RegionEmbeddingMatrix(
        X=X,
        metadata=meta,
        genes=[str(g) for g in joint_view.genes],
        stage=stage,
        dataset="joint",
        region_axis_kind=str(joint_view.region_axis_kind),
        dropped_rows=dropped,
    )


def build_prepost_region_embedding_matrix(
    prepost: Mapping[str, object],
    *,
    dataset: str = "GTEx",
    stage: str = "raw",
    require_all_genes_finite: bool = True,
) -> RegionEmbeddingMatrix:
    """Flatten PREPOST subject-parcel observations into samples x genes.

    This is the preferred route for pre/post ComBat embeddings. GTEx uses the
    PREPOST cubes directly; AHBA uses the PREPOST dataframes grouped to
    subject-parcel means. All rows are in the active parcel-matching policy.
    """
    ds = str(dataset).strip().upper()
    stage_key = str(stage).strip().lower()
    if stage_key in {"raw", "raw_matched", "pre", "ground_truth"}:
        stage_label = "raw_matched"
        source_key = "raw"
    elif stage_key in {"harm", "harmonized", "post", "combat"}:
        stage_label = "harmonized"
        source_key = "harmonized"
    else:
        raise ValueError("stage must be one of: raw, raw_matched, ground_truth, harmonized, post, combat")

    genes = [str(g) for g in prepost["genes"]]
    target_meta = prepost["target_meta"].copy()
    target_meta["parcel_idx"] = target_meta["parcel_idx"].astype(int)
    target_meta = target_meta.set_index("parcel_idx", drop=False)

    if ds == "GTEX":
        cube_key = "raw_cube" if source_key == "raw" else "harm_cube"
        cube = np.asarray(prepost[cube_key], dtype=np.float64)
        obs_mask = np.asarray(prepost["obs_mask"], dtype=bool)
        subjects = [str(s) for s in prepost["subjects"]]
    elif ds == "AHBA":
        df_key = "ahba_raw" if source_key == "raw" else "ahba_h"
        df = prepost[df_key].copy()
        df["subject"] = df["subject"].astype(str)
        df["parcel_idx"] = df["parcel_idx"].astype(int)
        grouped = df.groupby(["subject", "parcel_idx"], as_index=False)[genes].mean()
        subjects = sorted(grouped["subject"].unique().tolist())
        sid_to_i = {s: i for i, s in enumerate(subjects)}
        cube = np.full((len(subjects), len(target_meta), len(genes)), np.nan, dtype=np.float64)
        obs_mask = np.zeros((len(subjects), len(target_meta)), dtype=bool)
        i_idx = grouped["subject"].map(sid_to_i).to_numpy(dtype=np.intp)
        p_idx = grouped["parcel_idx"].to_numpy(dtype=np.intp)
        cube[i_idx, p_idx, :] = grouped[genes].to_numpy(dtype=np.float64, copy=False)
        obs_mask[i_idx, p_idx] = True
    else:
        raise ValueError("dataset must be 'GTEx' or 'AHBA'")

    if cube.ndim != 3:
        raise ValueError(f"PREPOST cube must be 3D subject x parcel x gene, got {cube.shape}")
    if obs_mask.shape != cube.shape[:2]:
        raise ValueError(f"obs_mask shape {obs_mask.shape} does not match cube shape {cube.shape[:2]}")

    row_mask = obs_mask.copy()
    if require_all_genes_finite:
        row_mask &= np.isfinite(cube).all(axis=2)
    else:
        row_mask &= np.isfinite(cube).any(axis=2)
    subj_i, parcel_i = np.where(row_mask)
    X = cube[subj_i, parcel_i, :]

    meta = pd.DataFrame(
        {
            "dataset": "GTEx" if ds == "GTEX" else "AHBA",
            "subject": [subjects[i] for i in subj_i],
            "subject_index": subj_i.astype(int),
            "parcel_idx": parcel_i.astype(int),
            "region_index": parcel_i.astype(int),
            "stage": stage_label,
            "pipeline_stage": stage_label,
            "region_axis_kind": "target_parcel",
            "future_imputation": False,
        }
    )
    parcel_cols = ["tissue_or_parcel", "macro_system", "hemisphere", "coord_x", "coord_y", "coord_z"]
    available_cols = [c for c in parcel_cols if c in target_meta.columns]
    meta = meta.merge(
        target_meta[["parcel_idx", *available_cols]].reset_index(drop=True),
        on="parcel_idx",
        how="left",
    )
    meta["region"] = meta["tissue_or_parcel"].astype(str)
    if "macro_system" in meta.columns:
        meta["macro_system"] = meta["macro_system"].astype(str)
        meta["region_group"] = meta["macro_system"].map(collapse_macro_system_to_region_group)
    else:
        meta["macro_system"] = "unknown"
        meta["region_group"] = "other"

    gtex_raw = prepost.get("gtex_eligible_raw")
    if ds == "GTEX" and isinstance(gtex_raw, pd.DataFrame) and {"subject", "parcel_idx"}.issubset(gtex_raw.columns):
        agg: dict[str, object] = {}
        for col in ("age", "sex"):
            if col in gtex_raw.columns:
                agg[col] = "first"
        if "tissue_or_parcel" in gtex_raw.columns:
            agg["tissue_or_parcel"] = lambda s: "; ".join(sorted(set(str(v) for v in s.dropna().tolist())))
        if agg:
            sample_meta = (
                gtex_raw.assign(parcel_idx=gtex_raw["parcel_idx"].astype(int), subject=gtex_raw["subject"].astype(str))
                .groupby(["subject", "parcel_idx"], as_index=False)
                .agg(agg)
            )
            if "tissue_or_parcel" in sample_meta.columns:
                sample_meta = sample_meta.rename(columns={"tissue_or_parcel": "gtex_native_regions"})
            meta = meta.merge(sample_meta, on=["subject", "parcel_idx"], how="left")
    if "gtex_native_regions" in meta.columns:
        meta["gtex_native_region"] = meta["gtex_native_regions"].astype(str)

    dropped = int(obs_mask.sum()) - int(row_mask.sum())
    return RegionEmbeddingMatrix(
        X=X,
        metadata=meta,
        genes=genes,
        stage=stage_label,
        dataset="GTEx" if ds == "GTEX" else "AHBA",
        region_axis_kind="target_parcel",
        dropped_rows=dropped,
    )


def build_gtex_region_embedding_matrix(
    prepost: Mapping[str, object],
    *,
    stage: str = "raw",
    require_all_genes_finite: bool = True,
) -> RegionEmbeddingMatrix:
    """Backward-compatible alias for GTEx PREPOST embeddings."""
    return build_prepost_region_embedding_matrix(
        prepost,
        dataset="GTEx",
        stage=stage,
        require_all_genes_finite=require_all_genes_finite,
    )


def add_gtex_native_region_labels(
    matrix: RegionEmbeddingMatrix,
    prepost: Mapping[str, object],
    *,
    source_col: str = "region",
    output_col: str = "gtex_native_region",
) -> RegionEmbeddingMatrix:
    """Attach native GTEx tissue labels to an embedding matrix.

    Useful for cache-derived matrices, where rows are indexed by AHBA target
    parcels but the desired legend is the native GTEx tissue that mapped there.
    """
    gtex = prepost.get("gtex_eligible_raw", prepost.get("gtex_raw"))
    if not isinstance(gtex, pd.DataFrame) or "parcel_idx" not in gtex.columns or "tissue_or_parcel" not in gtex.columns:
        raise KeyError("prepost must contain gtex_eligible_raw/gtex_raw with parcel_idx and tissue_or_parcel")
    target = prepost["target_meta"][["parcel_idx", "tissue_or_parcel"]].copy()
    target["parcel_idx"] = target["parcel_idx"].astype(int)
    target["region"] = target["tissue_or_parcel"].astype(str)
    parcel_to_native = (
        gtex.assign(parcel_idx=gtex["parcel_idx"].astype(int))
        .groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(str(v) for v in s.dropna().tolist()))))
        .to_dict()
    )
    region_to_native = {
        str(row.region): str(parcel_to_native.get(int(row.parcel_idx), str(row.region)))
        for row in target.itertuples(index=False)
    }
    out = RegionEmbeddingMatrix(
        X=matrix.X,
        metadata=matrix.metadata.copy(),
        genes=list(matrix.genes),
        stage=matrix.stage,
        dataset=matrix.dataset,
        region_axis_kind=matrix.region_axis_kind,
        row_axis=matrix.row_axis,
        dropped_rows=matrix.dropped_rows,
    )
    out.metadata[output_col] = out.metadata[source_col].astype(str).map(region_to_native).fillna(out.metadata[source_col].astype(str))
    meta_cols = [c for c in ("age", "sex") if c in gtex.columns]
    if meta_cols and "subject" in out.metadata.columns:
        subject_meta = (
            gtex.assign(subject=gtex["subject"].astype(str))
            .groupby("subject", as_index=False)[meta_cols]
            .first()
        )
        out.metadata = out.metadata.drop(columns=[c for c in meta_cols if c in out.metadata.columns], errors="ignore")
        out.metadata = out.metadata.merge(subject_meta, on="subject", how="left")
    return out


def add_parcel_network_labels(
    matrix: RegionEmbeddingMatrix,
    atlas_csv: str | Path = "data/metadata/atlas_info/atlas-4S156Parcels_dseg_reformatted.csv",
    *,
    label_col: str = "label",
) -> RegionEmbeddingMatrix:
    """Attach 7-network and 17-network atlas labels to parcel-frame rows."""
    atlas = pd.read_csv(atlas_csv)
    required = {label_col, "network_label", "network_label_17network"}
    missing = sorted(required.difference(atlas.columns))
    if missing:
        raise ValueError(f"atlas_csv is missing required columns: {missing}")

    lookup = (
        atlas[[label_col, "network_label", "network_label_17network"]]
        .drop_duplicates(subset=[label_col])
        .rename(columns={label_col: "region"})
    )
    meta = matrix.metadata.copy()
    meta = meta.drop(columns=["network_label", "network_label_17network"], errors="ignore")
    meta = meta.merge(lookup, on="region", how="left")

    fallback = meta.get("macro_system", pd.Series("unknown", index=meta.index)).map(collapse_macro_system_to_region_group)
    fallback = fallback.map(
        {
            "cortical": "Unknown",
            "subcortical": "Subcortical",
            "cerebellar": "Cerebellar",
            "other": "Unknown",
        }
    ).fillna("Unknown")
    meta["network_label"] = meta["network_label"].fillna(fallback)
    meta["network_label_17network"] = meta["network_label_17network"].fillna(fallback)
    return RegionEmbeddingMatrix(
        X=matrix.X,
        metadata=meta,
        genes=list(matrix.genes),
        stage=matrix.stage,
        dataset=matrix.dataset,
        region_axis_kind=matrix.region_axis_kind,
        row_axis=matrix.row_axis,
        dropped_rows=matrix.dropped_rows,
    )


def _peak_region_per_gene(X: np.ndarray, region_labels: Sequence[str]) -> list[str]:
    """For each gene (column of `X`), the region (row label) with the highest
    mean expression across that region's samples."""
    regions = np.asarray([str(r) for r in region_labels])
    uniq = list(pd.unique(regions))
    means = np.vstack([np.nanmean(X[regions == r], axis=0) for r in uniq])  # (n_region, n_gene)
    peak_idx = np.nanargmax(means, axis=0)
    return [uniq[int(i)] for i in peak_idx]


def build_gene_embedding_matrix(
    matrix: RegionEmbeddingMatrix,
    *,
    attribute_matrix: RegionEmbeddingMatrix | None = None,
    peak_over: str = "region",
) -> RegionEmbeddingMatrix:
    """Transpose a region matrix to **gene rows x subject-region-sample features**.

    Per-gene color attributes are attached to the metadata:
      - `peak_region`        : label of maximal mean expression (region palette)
      - `peak_region_group`  : its coarse anatomical group
      - `gene_mean`          : mean expression across samples (continuous)
      - `gene_cv`            : coefficient of variation across samples (continuous)

    `peak_over` selects the metadata column to compute the peak over — e.g.
    `"region"` (target parcels, ~150) or `"gtex_native_region"` (native GTEx
    tissues, ~11, cleaner legend). Falls back to `"region"` if the column is absent.

    Attributes are computed from `attribute_matrix` if given (e.g. the ground-truth
    raw_matched matrix) so the same gene keeps the same color across stages;
    otherwise from `matrix` itself.
    """
    genes = [str(g) for g in matrix.genes]
    src = attribute_matrix if attribute_matrix is not None else matrix
    if [str(g) for g in src.genes] != genes:
        raise ValueError("attribute_matrix.genes must match matrix.genes")

    X_gene = np.asarray(matrix.X, dtype=np.float64).T  # genes x samples
    X_src = np.asarray(src.X, dtype=np.float64)        # samples x genes

    gene_mean = X_src.mean(axis=0)
    gene_std = X_src.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_cv = np.where(np.abs(gene_mean) > 1e-9, gene_std / np.abs(gene_mean), np.nan)

    peak_col = peak_over if peak_over in src.metadata.columns else "region"
    if peak_col not in src.metadata.columns:
        raise KeyError("attribute source metadata must contain a 'region' column")
    peak_region = _peak_region_per_gene(X_src, src.metadata[peak_col].astype(str).tolist())
    if "region_group" in src.metadata.columns:
        rg_map = (
            src.metadata[[peak_col, "region_group"]].dropna().drop_duplicates().astype(str)
            .set_index(peak_col)["region_group"].to_dict()
        )
    else:
        rg_map = {}
    peak_region_group = [rg_map.get(r, "other") for r in peak_region]

    meta = pd.DataFrame(
        {
            "gene": genes,
            "dataset": matrix.dataset,
            "stage": matrix.stage,
            "pipeline_stage": matrix.stage,
            "peak_region": peak_region,
            "peak_region_group": peak_region_group,
            "gene_mean": gene_mean,
            "gene_cv": gene_cv,
        }
    )
    return RegionEmbeddingMatrix(
        X=X_gene,
        metadata=meta,
        genes=[f"sample_{i}" for i in range(X_gene.shape[1])],
        stage=matrix.stage,
        dataset=matrix.dataset,
        region_axis_kind=matrix.region_axis_kind,
        row_axis="gene",
        dropped_rows=0,
    )


def compute_pca_embedding(
    matrix: RegionEmbeddingMatrix,
    *,
    n_components: int = 2,
    feature_preprocess: str = "center",
    random_state: int = 0,
) -> EmbeddingResult:
    """Compute a PCA embedding from a PREPOST region matrix."""
    from sklearn.decomposition import PCA

    X_proc, stats = preprocess_expression_features(matrix.X, mode=feature_preprocess)
    n_comp = int(min(max(1, n_components), X_proc.shape[0], X_proc.shape[1]))
    pca = PCA(n_components=n_comp, random_state=int(random_state))
    coords = pca.fit_transform(X_proc)
    out = matrix.metadata.copy()
    for i in range(n_comp):
        out[f"PC{i + 1}"] = coords[:, i]
    if n_comp >= 2:
        out["embed_x"] = out["PC1"]
        out["embed_y"] = out["PC2"]
    fit_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components": n_comp,
        "preprocess_zero_scale_features": stats.zero_scale_features,
    }
    return EmbeddingResult(
        coordinates=out,
        method="PCA",
        feature_preprocess=stats.mode,
        stage=matrix.stage,
        dataset=matrix.dataset,
        region_axis_kind=matrix.region_axis_kind,
        genes=matrix.genes,
        fit_info=fit_info,
    )


def compute_umap_embedding(
    matrix: RegionEmbeddingMatrix,
    *,
    n_pca: int = 200,
    feature_preprocess: str = "center",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 0,
    umap_kwargs: Mapping[str, object] | None = None,
) -> EmbeddingResult:
    """Compute UMAP after PCA pre-reduction of the gene feature axis."""
    from sklearn.decomposition import PCA
    try:
        import umap  # type: ignore
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("UMAP embedding requires `umap-learn` in the active notebook environment") from e

    X_proc, stats = preprocess_expression_features(matrix.X, mode=feature_preprocess)
    n_pca_eff = int(min(max(2, n_pca), X_proc.shape[0], X_proc.shape[1]))
    rng = None if random_state is None else int(random_state)
    X_umap = PCA(n_components=n_pca_eff, random_state=rng).fit_transform(X_proc)
    kwargs = dict(umap_kwargs or {})
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=rng,
        **kwargs,
    )
    coords = reducer.fit_transform(X_umap)
    out = matrix.metadata.copy()
    out["UMAP1"] = coords[:, 0]
    out["UMAP2"] = coords[:, 1]
    out["embed_x"] = out["UMAP1"]
    out["embed_y"] = out["UMAP2"]
    fit_info = {
        "n_pca": n_pca_eff,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "random_state": rng,
        "preprocess_zero_scale_features": stats.zero_scale_features,
    }
    return EmbeddingResult(
        coordinates=out,
        method="UMAP",
        feature_preprocess=stats.mode,
        stage=matrix.stage,
        dataset=matrix.dataset,
        region_axis_kind=matrix.region_axis_kind,
        genes=matrix.genes,
        fit_info=fit_info,
    )


def embedding_color_spec(
    df: pd.DataFrame,
    *,
    color_by: str = "region",
    max_legend_items: int | None = 40,
) -> tuple[str, list[str], dict[str, object]]:
    """Return color key, ordered legend values, and palette for an embedding df."""
    color_key = str(color_by).strip().lower()
    if color_key == "native":
        color_key = "gtex_native_region"
    if color_key == "macro":
        color_key = "macro_system"
    if color_key not in EMBEDDING_COLOR_OPTIONS:
        raise ValueError(f"color_by must be one of {EMBEDDING_COLOR_OPTIONS} (got {color_by!r})")
    if color_key not in df.columns:
        raise KeyError(f"Embedding dataframe does not contain {color_key!r}")
    if color_key in _CONTINUOUS_COLOR_KEYS:
        return color_key, [], {}

    if color_key in {"region", "gtex_native_region", "peak_region"}:
        group_source = color_key
        # group-membership column used for the region palette's shading buckets
        rg_col = "peak_region_group" if color_key == "peak_region" else "region_group"
        rg_lookup = (
            df[[group_source, rg_col]]
            .dropna()
            .drop_duplicates()
            .astype(str)
            .set_index(group_source)[rg_col]
            .to_dict()
            if rg_col in df.columns
            else None
        )
        weights = df[group_source].astype(str).value_counts().astype(int).to_dict()
        palette = region_scatter_palette(df[group_source].astype(str).unique(), region_group_lookup=rg_lookup, weights=weights)
        order = ordered_region_scatter_values(df[group_source].astype(str).unique(), region_group_lookup=rg_lookup, weights=weights)
        if max_legend_items is not None:
            counts = df[group_source].astype(str).value_counts()
            allowed = set(counts.index[: int(max_legend_items)].tolist())
            order = [v for v in order if v in allowed]
        return color_key, order, palette

    if color_key in {"region_group", "peak_region_group"}:
        present = set(df[color_key].astype(str))
        order = [v for v in REGION_SCATTER_GROUP_ORDER if v in present] + sorted(
            v for v in present if v not in REGION_SCATTER_GROUP_ORDER
        )
        palette = {v: REGION_SCATTER_GROUP_BASE.get(v, "#777777") for v in order}
        return color_key, order, palette

    if color_key == "sex":
        raw = [str(v) for v in df[color_key].dropna().astype(str).unique()]
        order = sorted(raw, key=format_legend_label)
        palette = {
            v: _SEX_COLORS.get(v.strip().lower(), f"C{i % 10}")
            for i, v in enumerate(order)
        }
        return color_key, order, palette

    if color_key == "dataset":
        preferred = ["GTEx", "AHBA"]
        present = {str(v) for v in df[color_key].dropna().astype(str).unique()}
        order = [v for v in preferred if v in present] + sorted(v for v in present if v not in preferred)
        palette = {v: _DATASET_COLORS.get(v, f"C{i % 10}") for i, v in enumerate(order)}
        return color_key, order, palette

    if color_key in {"network_label", "network_label_17network"}:
        preferred = [
            "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic",
            "Cont", "Default", "Subcortical", "Cerebellar", "Unknown",
        ]
        present = {str(v) for v in df[color_key].dropna().astype(str).unique()}
        order = [v for v in preferred if v in present] + sorted(v for v in present if v not in preferred)
        palette = {v: _NETWORK_LABEL_COLORS.get(v, f"C{i % 10}") for i, v in enumerate(order)}
        return color_key, order, palette

    order = sorted(df[color_key].dropna().astype(str).unique(), key=format_legend_label)
    palette = {
        v: _MACRO_SYSTEM_COLORS.get(_macro_system_palette_key(v), REGION_SCATTER_GROUP_BASE.get("other", "#777777"))
        for v in order
    }
    return color_key, order, palette


def _macro_system_palette_key(value: object) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if key in _MACRO_SYSTEM_COLORS:
        return key
    if "visual" in key or "somato" in key or key.startswith("vis"):
        return "visual_somatomotor"
    if "assoc" in key or "association" in key:
        return "cortical_association"
    if "cerebell" in key:
        return "cerebellar"
    if "basal" in key or "ganglia" in key:
        return "basal_ganglia"
    if "limbic" in key or "midbrain" in key:
        return "limbic_midbrain"
    if "subcort" in key:
        return "subcortical"
    if key in {"", "nan", "none"}:
        return "unknown"
    return key


def _marker_values(df: pd.DataFrame, marker_by: str | None) -> tuple[str | None, list[str], dict[str, str], dict[str, float]]:
    if marker_by is None:
        return None, [], {}, {}
    marker_key = str(marker_by).strip().lower()
    if marker_key not in df.columns:
        raise KeyError(f"Embedding dataframe does not contain marker_by={marker_by!r}")
    raw = {str(v) for v in df[marker_key].dropna().astype(str).unique()}
    if marker_key == "dataset":
        preferred = ["GTEx", "AHBA"]
        order = [v for v in preferred if v in raw] + sorted(v for v in raw if v not in preferred)
        markers = {v: _DATASET_MARKERS.get(v, "o") for v in order}
        sizes = {v: _DATASET_POINT_SIZES.get(v, 8.0) for v in order}
        return marker_key, order, markers, sizes
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    order = sorted(raw, key=format_legend_label)
    markers = {v: marker_cycle[i % len(marker_cycle)] for i, v in enumerate(order)}
    sizes = {v: 8.0 for v in order}
    return marker_key, order, markers, sizes


def _continuous_color_limits(df: pd.DataFrame, color_key: str) -> tuple[float, float]:
    vals = pd.to_numeric(df[color_key], errors="coerce")
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _counted_label(value: object, counts: Mapping[str, int]) -> str:
    label = format_legend_label(value)
    return f"{label} ({int(counts.get(str(value), 0)):,})"


def _max_counts_by_panel(dfs: list[pd.DataFrame], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for df in dfs:
        panel_counts = df[key].astype(str).value_counts().astype(int).to_dict()
        for value, count in panel_counts.items():
            counts[value] = max(counts.get(value, 0), int(count))
    return counts


def _legend_marker_size(point_size: float) -> float:
    return max(4.0, min(12.0, float(point_size) ** 0.5))


def _marker_alpha(marker_value: str, marker_alpha_map: Mapping[str, float] | None, default_alpha: float) -> float:
    if marker_alpha_map is None:
        return float(default_alpha)
    return float(marker_alpha_map.get(str(marker_value), default_alpha))


def plot_embedding(
    embedding: EmbeddingResult | pd.DataFrame,
    *,
    color_by: str = "region",
    ax=None,
    title: str | None = None,
    point_size: float = 8,
    alpha: float = 0.82,
    max_legend_items: int | None = 40,
    show_legend: bool = True,
    continuous_vmin: float | None = None,
    continuous_vmax: float | None = None,
    show_colorbar: bool = True,
    continuous_cmap: str = "viridis",
    marker_by: str | None = None,
    marker_map: Mapping[str, str] | None = None,
    marker_size_map: Mapping[str, float] | None = None,
    marker_alpha_map: Mapping[str, float] | None = None,
):
    """Scatter an embedding with eval-population region colors."""
    set_academic_style()
    df = embedding.coordinates if isinstance(embedding, EmbeddingResult) else embedding
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.6))
    color_key, order, palette = embedding_color_spec(df, color_by=color_by, max_legend_items=max_legend_items)
    marker_key, marker_order, markers, marker_sizes = _marker_values(df, marker_by)
    if marker_map:
        markers.update({str(k): str(v) for k, v in marker_map.items()})
    if marker_size_map:
        marker_sizes.update({str(k): float(v) for k, v in marker_size_map.items()})
    marker_groups = marker_order if marker_key is not None else ["__all__"]

    sc = None
    if color_key in _CONTINUOUS_COLOR_KEYS:
        vmin, vmax = _continuous_color_limits(df, color_key)
        if continuous_vmin is not None:
            vmin = float(continuous_vmin)
        if continuous_vmax is not None:
            vmax = float(continuous_vmax)
        for marker_value in marker_groups:
            sub = df if marker_key is None else df[df[marker_key].astype(str) == marker_value]
            vals = pd.to_numeric(sub[color_key], errors="coerce")
            sc = ax.scatter(
                sub["embed_x"], sub["embed_y"],
                c=vals,
                cmap=continuous_cmap,
                vmin=vmin,
                vmax=vmax,
                s=marker_sizes.get(marker_value, point_size) if marker_key is not None else point_size,
                marker=markers.get(marker_value, "o") if marker_key is not None else "o",
                alpha=_marker_alpha(marker_value, marker_alpha_map, alpha) if marker_key is not None else alpha,
                linewidths=0,
            )
    else:
        for marker_value in marker_groups:
            sub = df if marker_key is None else df[df[marker_key].astype(str) == marker_value]
            colors = sub[color_key].astype(str).map(palette).fillna("#777777")
            sc = ax.scatter(
                sub["embed_x"], sub["embed_y"],
                c=colors,
                s=marker_sizes.get(marker_value, point_size) if marker_key is not None else point_size,
                marker=markers.get(marker_value, "o") if marker_key is not None else "o",
                alpha=_marker_alpha(marker_value, marker_alpha_map, alpha) if marker_key is not None else alpha,
                linewidths=0,
            )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if title:
        ax.set_title(title)
    apply_tick_style(ax)
    if color_key in _CONTINUOUS_COLOR_KEYS and show_colorbar:
        ax.figure.colorbar(sc, ax=ax, label=format_legend_label(color_key), fraction=0.046, pad=0.04)
    elif show_legend and order:
        counts = df[color_key].astype(str).value_counts().astype(int).to_dict()
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=palette.get(v, "#777777"),
                   markeredgecolor="none", label=_counted_label(v, counts))
            for v in order
        ]
        ax.legend(handles=handles, title=format_legend_label(color_key), loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=False, borderaxespad=0.0)
    return ax


def plot_embedding_pair(
    left: EmbeddingResult,
    right: EmbeddingResult,
    *,
    color_by: str = "region",
    titles: tuple[str, str] | None = None,
    figsize: tuple[float, float] = (12.0, 4.8),
    max_legend_items: int | None = 40,
    point_size: float = 8,
    alpha: float = 0.82,
    continuous_cmap: str = "viridis",
    marker_by: str | None = None,
    marker_map: Mapping[str, str] | None = None,
    marker_size_map: Mapping[str, float] | None = None,
    marker_alpha_map: Mapping[str, float] | None = None,
):
    """Plot two embeddings side by side with one shared legend on the right."""
    set_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if titles is None:
        titles = (left.stage, right.stage)
    combined = pd.concat([left.coordinates, right.coordinates], ignore_index=True)
    color_key, order, palette = embedding_color_spec(combined, color_by=color_by, max_legend_items=max_legend_items)
    vmin = vmax = None
    if color_key in _CONTINUOUS_COLOR_KEYS:
        vmin, vmax = _continuous_color_limits(combined, color_key)
    marker_key, marker_order, markers, marker_sizes = _marker_values(combined, marker_by)
    if marker_map:
        markers.update({str(k): str(v) for k, v in marker_map.items()})
    if marker_size_map:
        marker_sizes.update({str(k): float(v) for k, v in marker_size_map.items()})
    plot_embedding(left, color_by=color_by, ax=axes[0], title=titles[0], show_legend=False,
                   max_legend_items=max_legend_items, point_size=point_size, alpha=alpha,
                   continuous_vmin=vmin, continuous_vmax=vmax, show_colorbar=False,
                   continuous_cmap=continuous_cmap, marker_by=marker_by,
                   marker_map=markers, marker_size_map=marker_sizes,
                   marker_alpha_map=marker_alpha_map)
    plot_embedding(right, color_by=color_by, ax=axes[1], title=titles[1], show_legend=False,
                   max_legend_items=max_legend_items, point_size=point_size, alpha=alpha,
                   continuous_vmin=vmin, continuous_vmax=vmax, show_colorbar=False,
                   continuous_cmap=continuous_cmap, marker_by=marker_by,
                   marker_map=markers, marker_size_map=marker_sizes,
                   marker_alpha_map=marker_alpha_map)

    if color_key in _CONTINUOUS_COLOR_KEYS:
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap=continuous_cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=axes, label=format_legend_label(color_key), fraction=0.046, pad=0.04)
    elif order:
        counts = _max_counts_by_panel([left.coordinates, right.coordinates], color_key)
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=palette.get(v, "#777777"),
                   markeredgecolor="none", label=_counted_label(v, counts))
            for v in order
        ]
        fig.legend(handles=handles, title=format_legend_label(color_key), loc="center left",
                   bbox_to_anchor=(1.01, 0.5), frameon=True, fancybox=False)
    if marker_key is not None and marker_order:
        marker_counts = _max_counts_by_panel([left.coordinates, right.coordinates], marker_key)
        marker_handles = [
            Line2D([0], [0], marker=markers.get(v, "o"), linestyle="",
                   markersize=_legend_marker_size(marker_sizes.get(v, point_size)),
                   markerfacecolor="#666666", markeredgecolor="none", label=_counted_label(v, marker_counts))
            for v in marker_order
        ]
        fig.legend(handles=marker_handles, title=format_legend_label(marker_key), loc="lower left",
                   bbox_to_anchor=(1.01, 0.05), frameon=True, fancybox=False)
    return fig, axes


def plot_embedding_sequence(
    embeddings: list[EmbeddingResult],
    *,
    color_by: str = "region",
    titles: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    max_legend_items: int | None = 40,
    point_size: float = 8,
    alpha: float = 0.82,
    continuous_cmap: str = "viridis",
    marker_by: str | None = None,
    marker_map: Mapping[str, str] | None = None,
    marker_size_map: Mapping[str, float] | None = None,
    marker_alpha_map: Mapping[str, float] | None = None,
):
    """Plot an ordered embedding sequence with one shared legend."""
    if not embeddings:
        raise ValueError("embeddings must contain at least one EmbeddingResult")
    set_academic_style()
    n = len(embeddings)
    if figsize is None:
        figsize = (4.8 * n, 4.6)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True, squeeze=False)
    axes_1d = axes[0]
    if titles is None:
        titles = [e.stage for e in embeddings]
    combined = pd.concat([e.coordinates for e in embeddings], ignore_index=True)
    color_key, order, palette = embedding_color_spec(combined, color_by=color_by, max_legend_items=max_legend_items)
    vmin = vmax = None
    if color_key in _CONTINUOUS_COLOR_KEYS:
        vmin, vmax = _continuous_color_limits(combined, color_key)
    marker_key, marker_order, markers, marker_sizes = _marker_values(combined, marker_by)
    if marker_map:
        markers.update({str(k): str(v) for k, v in marker_map.items()})
    if marker_size_map:
        marker_sizes.update({str(k): float(v) for k, v in marker_size_map.items()})
    for ax, emb, title in zip(axes_1d, embeddings, titles):
        plot_embedding(
            emb,
            color_by=color_by,
            ax=ax,
            title=title,
            show_legend=False,
            max_legend_items=max_legend_items,
            point_size=point_size,
            alpha=alpha,
            continuous_vmin=vmin,
            continuous_vmax=vmax,
            show_colorbar=False,
            continuous_cmap=continuous_cmap,
            marker_by=marker_by,
            marker_map=markers,
            marker_size_map=marker_sizes,
            marker_alpha_map=marker_alpha_map,
        )

    if color_key in _CONTINUOUS_COLOR_KEYS:
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap=continuous_cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=axes_1d.tolist(), label=format_legend_label(color_key), fraction=0.046, pad=0.04)
    elif order:
        counts = _max_counts_by_panel([e.coordinates for e in embeddings], color_key)
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=palette.get(v, "#777777"),
                   markeredgecolor="none", label=_counted_label(v, counts))
            for v in order
        ]
        fig.legend(handles=handles, title=format_legend_label(color_key), loc="center left",
                   bbox_to_anchor=(1.01, 0.5), frameon=True, fancybox=False)
    if marker_key is not None and marker_order:
        marker_counts = _max_counts_by_panel([e.coordinates for e in embeddings], marker_key)
        marker_handles = [
            Line2D([0], [0], marker=markers.get(v, "o"), linestyle="",
                   markersize=_legend_marker_size(marker_sizes.get(v, point_size)),
                   markerfacecolor="#666666", markeredgecolor="none", label=_counted_label(v, marker_counts))
            for v in marker_order
        ]
        fig.legend(handles=marker_handles, title=format_legend_label(marker_key), loc="lower left",
                   bbox_to_anchor=(1.01, 0.05), frameon=True, fancybox=False)
    return fig, axes_1d


def plot_joint_embedding_sequence(
    embeddings: list[EmbeddingResult],
    *,
    color_by: str = "region",
    titles: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    max_legend_items: int | None = 40,
    point_size: float = 8,
    alpha: float = 0.82,
    equal_dataset_point_size: bool = False,
    continuous_cmap: str = "viridis",
    dataset_marker_map: Mapping[str, str] | None = None,
    dataset_point_size_map: Mapping[str, float] | None = None,
    dataset_alpha_map: Mapping[str, float] | None = None,
):
    marker_sizes: dict[str, float] | None = {"GTEx": point_size, "AHBA": point_size} if equal_dataset_point_size else None
    if dataset_point_size_map:
        marker_sizes = dict(marker_sizes or {})
        marker_sizes.update({str(k): float(v) for k, v in dataset_point_size_map.items()})
    marker_alphas = dict(_DATASET_ALPHAS)
    if dataset_alpha_map:
        marker_alphas.update({str(k): float(v) for k, v in dataset_alpha_map.items()})
    return plot_embedding_sequence(
        embeddings,
        color_by=color_by,
        titles=titles,
        figsize=figsize,
        max_legend_items=max_legend_items,
        point_size=point_size,
        alpha=alpha,
        continuous_cmap=continuous_cmap,
        marker_by="dataset",
        marker_map=dataset_marker_map,
        marker_size_map=marker_sizes,
        marker_alpha_map=marker_alphas,
    )


__all__ = [
    "EMBEDDING_COLOR_OPTIONS",
    "FEATURE_PREPROCESS_OPTIONS",
    "EmbeddingResult",
    "FeaturePreprocessStats",
    "RegionEmbeddingMatrix",
    "add_gtex_native_region_labels",
    "add_parcel_network_labels",
    "build_prepost_region_embedding_matrix",
    "build_gtex_region_embedding_matrix",
    "build_region_embedding_matrix_from_joint_tensor_view",
    "build_gene_embedding_matrix",
    "build_region_embedding_matrix_from_tensor_view",
    "compute_pca_embedding",
    "compute_umap_embedding",
    "embedding_color_spec",
    "plot_embedding",
    "plot_embedding_pair",
    "plot_embedding_sequence",
    "plot_joint_embedding_sequence",
    "preprocess_expression_features",
]
