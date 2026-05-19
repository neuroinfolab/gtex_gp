#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.io import parse_coordinate_representative
from src.preprocess import build_target_parcels, map_gtex_to_target
from src.samples import validate_gxp_samples
from src.samples.schema import META_COLS
from src.spatial.parcel_matching import (
    MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
    apply_gtex_ahba_matching_policy,
)

from .eval_style import PARCEL_GROUP_COLORS, _resolve_fonts, apply_tick_style, pretty_gtex_label


GTEX_TENSOR_REGION_ORDER = [
    "brain - frontal cortex (ba9)",
    "brain - anterior cingulate cortex (ba24)",
    "brain - cortex",
    "brain - nucleus accumbens (basal ganglia)",
    "brain - caudate (basal ganglia)",
    "brain - putamen (basal ganglia)",
    "brain - amygdala",
    "brain - hippocampus",
    "brain - hypothalamus",
    "brain - substantia nigra",
    "brain - cerebellar hemisphere",
    "brain - cerebellum",
]

GTEX_TENSOR_REGION_GROUP = {
    "brain - frontal cortex (ba9)": "cortical",
    "brain - anterior cingulate cortex (ba24)": "cortical",
    "brain - cortex": "cortical",
    "brain - nucleus accumbens (basal ganglia)": "basal_ganglia",
    "brain - caudate (basal ganglia)": "basal_ganglia",
    "brain - putamen (basal ganglia)": "basal_ganglia",
    "brain - amygdala": "limbic_midbrain",
    "brain - hippocampus": "limbic_midbrain",
    "brain - hypothalamus": "limbic_midbrain",
    "brain - substantia nigra": "limbic_midbrain",
    "brain - cerebellar hemisphere": "cerebellar",
    "brain - cerebellum": "cerebellar",
}

GTEX_TENSOR_GROUP_COLOR = {group: colors[0] for group, colors in PARCEL_GROUP_COLORS.items()}
TENSOR_AXIS_COMPONENTS = ("gene", "subject", "region")
DEFAULT_TENSOR_AXIS_ASSIGNMENT = ("gene", "region", "subject")
REGION_ORDERING_DATASET = "dataset"
REGION_ORDERING_MATCHED = "region_matched"
REGION_ORDERING_MATCHED_SUPERSET = "region_matched_superset"
REGION_ORDERINGS = (
    REGION_ORDERING_DATASET,
    REGION_ORDERING_MATCHED,
    REGION_ORDERING_MATCHED_SUPERSET,
)
_COMPRESSED_BOX_ASPECT_FACTORS = {
    "gene": 1.0,
    "region": 1.3,
    "subject": 1.0,
}
_VOXEL_TENSOR_FONTS = {
    "title": "l",
    "xlabel": "l",
    "ylabel": "l",
    "zlabel": "l",
    "xtick": "s-1",
    "ytick": "s-2",
    "ztick": "s-1",
    "cbar_label": "s",
    "legend": "s",
}
_VOXEL_TENSOR_AXIS_LABEL_PADS = {
    "gene": 4.0,
    "region": 32.0,
    "subject": 32.0,
}
_VOXEL_TENSOR_AXIS_LABEL_ONLY_PADS = {
    "gene": -6.0,
    "region": -6.0,
    "subject": -6.0,
}


@dataclass
class TensorView:
    values: np.ndarray
    observed_mask: np.ndarray
    subjects: list[str]
    regions: list[str]
    genes: list[str]
    dataset: str
    region_axis_kind: str
    future_imputation_mask: np.ndarray | None = None


@dataclass
class JointTensorView:
    values: np.ndarray
    observed_mask: np.ndarray
    future_imputation_mask: np.ndarray
    subjects: list[str]
    subject_datasets: list[str]
    dataset_slices: dict
    regions: list[str]
    genes: list[str]
    region_axis_kind: str
    matched_region_count: int
    source_views: dict


def build_joint_tensor_view(
    gtex_view: TensorView,
    ahba_view: TensorView,
    *,
    stack_order: Sequence[str] = ("GTEx", "AHBA"),
    matched_region_count: int | None = None,
) -> JointTensorView:
    if list(gtex_view.genes) != list(ahba_view.genes):
        raise ValueError(
            "gtex_view and ahba_view must share the same gene list and order."
        )
    if list(gtex_view.regions) != list(ahba_view.regions):
        raise ValueError(
            "gtex_view and ahba_view must share the same region list and order. "
            "Build both views with the same region_ordering / matching policy."
        )
    if gtex_view.region_axis_kind != ahba_view.region_axis_kind:
        raise ValueError(
            "region_axis_kind mismatch: "
            f"{gtex_view.region_axis_kind!r} vs {ahba_view.region_axis_kind!r}. "
            "Both views must live on the same region frame (e.g. 'ahba_parcel')."
        )
    stack_order = tuple(str(x) for x in stack_order)
    if set(stack_order) != {"GTEx", "AHBA"} or len(stack_order) != 2:
        raise ValueError("stack_order must be a permutation of ('GTEx', 'AHBA').")

    if matched_region_count is None:
        if (
            gtex_view.future_imputation_mask is not None
            and gtex_view.future_imputation_mask.any()
        ):
            per_region_future = gtex_view.future_imputation_mask.any(axis=0)
            matched_region_count = int(np.argmax(per_region_future))
        else:
            matched_region_count = len(gtex_view.regions)
    matched_region_count = int(matched_region_count)

    views_by_name = {"GTEx": gtex_view, "AHBA": ahba_view}
    ordered_views = [views_by_name[name] for name in stack_order]

    subjects: list[str] = []
    subject_datasets: list[str] = []
    dataset_slices: dict[str, slice] = {}
    cursor = 0
    for name, view in zip(stack_order, ordered_views):
        start = cursor
        subjects.extend(view.subjects)
        subject_datasets.extend([name] * len(view.subjects))
        cursor += len(view.subjects)
        dataset_slices[name] = slice(start, cursor)

    values = np.concatenate([v.values for v in ordered_views], axis=0)
    observed_mask = np.concatenate([v.observed_mask for v in ordered_views], axis=0)
    future_pieces = []
    for v in ordered_views:
        piece = (
            v.future_imputation_mask
            if v.future_imputation_mask is not None
            else np.zeros_like(v.observed_mask)
        )
        future_pieces.append(piece)
    future_imputation_mask = np.concatenate(future_pieces, axis=0)

    return JointTensorView(
        values=values,
        observed_mask=observed_mask,
        future_imputation_mask=future_imputation_mask,
        subjects=subjects,
        subject_datasets=subject_datasets,
        dataset_slices=dataset_slices,
        regions=list(gtex_view.regions),
        genes=list(gtex_view.genes),
        region_axis_kind=gtex_view.region_axis_kind,
        matched_region_count=matched_region_count,
        source_views={"GTEx": gtex_view, "AHBA": ahba_view},
    )


def summarize_joint_selection(joint_view: JointTensorView) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": joint_view.subjects,
            "dataset": joint_view.subject_datasets,
        }
    )


def load_gxp_samples_table(csv_path: str | Path = "data/raw/gxp_samples.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_gxp_samples(df)
    return df


def load_gene_panel(path_or_name: str | Path) -> list[str]:
    raw = str(path_or_name).strip()
    path = Path(raw)
    if not path.exists():
        root = Path(__file__).resolve().parents[2]
        candidates = [
            (root / raw).resolve(),
            (root / "data" / "metadata" / "gene_lists" / raw).resolve(),
            (root / "data" / "metadata" / "gene_lists" / f"{raw}.csv").resolve(),
            (root / "data" / "metadata" / f"{raw}.txt").resolve(),
            (root / "data" / "metadata" / "gene_lists" / f"{raw}.txt").resolve(),
            (root / "data" / "metadata" / "gene_lists" / "allgenes_stability_dk" / raw).resolve(),
            (root / "data" / "metadata" / "gene_lists" / "allgenes_stability_dk" / f"{raw}.csv").resolve(),
            (root / "data" / "raw" / "gene_lists" / raw).resolve(),
            (root / "data" / "raw" / "gene_lists" / f"{raw}.csv").resolve(),
            (root / "data" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
            (root / "out" / "raw" / "gene_lists" / raw).resolve(),
            (root / "out" / "raw" / "gene_lists" / f"{raw}.csv").resolve(),
            (root / "out" / "raw" / "gene_lists" / f"{raw}.txt").resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise FileNotFoundError(f"Could not resolve gene panel {path_or_name!r}")
    if path.suffix.lower() == ".csv":
        panel_df = pd.read_csv(path)
        if "label" in panel_df.columns and len(panel_df.columns) > 1:
            return [str(c) for c in panel_df.columns if str(c).strip().lower() != "label"]
        for col in ("gene", "gene_symbol", "symbol", "gene_name"):
            if col in panel_df.columns:
                return [str(x) for x in panel_df[col].dropna().tolist()]
        return [str(x) for x in panel_df.iloc[:, 0].dropna().tolist()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def sample_subjects(df: pd.DataFrame, n: int = 10, seed: int = 123) -> list[str]:
    subjects = sorted(df["subject"].astype(str).unique().tolist())
    if n >= len(subjects):
        return subjects
    rng = np.random.default_rng(seed)
    picked = rng.choice(subjects, size=n, replace=False).tolist()
    return sorted(str(x) for x in picked)


def sample_subjects_by_region_coverage(
    df: pd.DataFrame,
    *,
    n: int = 10,
    seed: int = 123,
    min_regions: int = 5,
    region_order: Sequence[str] | None = None,
) -> list[str]:
    return sample_subjects_by_dataset_region_coverage(
        df,
        dataset="GTEx",
        n=n,
        seed=seed,
        min_regions=min_regions,
        region_order=region_order,
    )


def sample_subjects_by_dataset_region_coverage(
    df: pd.DataFrame,
    *,
    dataset: str,
    n: int = 10,
    seed: int = 123,
    min_regions: int = 5,
    region_order: Sequence[str] | None = None,
) -> list[str]:
    dataset_key = str(dataset).upper()
    subset = df.loc[df["dataset"].astype(str).str.upper().eq(dataset_key)].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["tissue_or_parcel"] = subset["tissue_or_parcel"].astype(str)
    if region_order is not None:
        valid_regions = {str(r) for r in region_order}
        subset = subset[subset["tissue_or_parcel"].isin(valid_regions)]
    coverage = (
        subset.groupby("subject")["tissue_or_parcel"]
        .nunique()
        .rename("n_regions")
        .reset_index()
    )
    eligible = coverage.loc[coverage["n_regions"].ge(int(min_regions)), "subject"].astype(str).tolist()
    eligible = sorted(eligible)
    if not eligible:
        raise ValueError(f"No {dataset} subjects have at least {min_regions} observed regions.")
    if n >= len(eligible):
        return eligible
    rng = np.random.default_rng(seed)
    picked = rng.choice(eligible, size=n, replace=False).tolist()
    return sorted(str(x) for x in picked)


def sample_genes(panel_genes: Sequence[str], available_genes: Sequence[str], n: int = 20, seed: int = 123) -> list[str]:
    available_upper = {str(g).upper(): str(g) for g in available_genes}
    genes = [available_upper[g.upper()] for g in panel_genes if str(g).upper() in available_upper]
    genes = list(dict.fromkeys(genes))
    if n >= len(genes):
        return genes
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.choice(genes, size=n, replace=False).tolist()]


def sample_regions(region_order: Sequence[str], n: int | None = None, seed: int = 123) -> list[str]:
    regions = [str(r) for r in region_order]
    if n is None or int(n) >= len(regions):
        return regions
    rng = np.random.default_rng(seed)
    sampled = {str(x) for x in rng.choice(regions, size=int(n), replace=False).tolist()}
    return [r for r in regions if r in sampled]


def _take_region_prefix(region_order: Sequence[str], n: int | None = None) -> list[str]:
    regions = [str(r) for r in region_order]
    if n is None or int(n) >= len(regions):
        return regions
    return regions[: int(n)]


def build_dataset_region_tensor(
    df: pd.DataFrame,
    *,
    dataset: str,
    subjects: Sequence[str],
    genes: Sequence[str],
    region_order: Sequence[str],
    region_axis_kind: str | None = None,
    future_imputation_regions: Sequence[str] | None = None,
) -> TensorView:
    dataset_label = str(dataset)
    dataset_key = dataset_label.upper()
    subset = df.loc[df["dataset"].astype(str).str.upper().eq(dataset_key)].copy()
    subset["subject"] = subset["subject"].astype(str)
    subset["tissue_or_parcel"] = subset["tissue_or_parcel"].astype(str)
    dupes = subset.groupby(["subject", "tissue_or_parcel"]).size()
    dupes = dupes[dupes.gt(1)]
    if not dupes.empty:
        raise ValueError(
            f"Expected one {dataset_label} row per subject x region. "
            f"Found duplicates: {dupes.head().to_dict()}"
        )

    subjects = [str(s) for s in subjects]
    genes = [str(g) for g in genes]
    regions = [str(r) for r in region_order]
    values = np.full((len(subjects), len(regions), len(genes)), np.nan, dtype=np.float64)
    observed_mask = np.zeros((len(subjects), len(regions)), dtype=bool)
    subject_to_i = {s: i for i, s in enumerate(subjects)}
    region_to_j = {r: j for j, r in enumerate(regions)}

    subset = subset[subset["subject"].isin(subjects) & subset["tissue_or_parcel"].isin(regions)]
    for _, row in subset.iterrows():
        i = subject_to_i[row["subject"]]
        j = region_to_j[row["tissue_or_parcel"]]
        values[i, j, :] = row[genes].to_numpy(dtype=np.float64)
        observed_mask[i, j] = True

    future_imputation_mask = np.zeros((len(subjects), len(regions)), dtype=bool)
    if future_imputation_regions:
        future_set = {str(r) for r in future_imputation_regions}
        for j, region in enumerate(regions):
            if region in future_set:
                future_imputation_mask[:, j] = True

    return TensorView(
        values=values,
        observed_mask=observed_mask,
        subjects=subjects,
        regions=regions,
        genes=genes,
        dataset=dataset_label,
        region_axis_kind=region_axis_kind or f"{dataset_key.lower()}_region",
        future_imputation_mask=future_imputation_mask,
    )


def _canonical_dataset_label(dataset: str) -> str:
    dataset_key = str(dataset).strip().upper()
    if dataset_key == "GTEX":
        return "GTEx"
    if dataset_key == "AHBA":
        return "AHBA"
    raise ValueError("dataset must be 'GTEx' or 'AHBA'.")


def _default_region_axis_kind(dataset: str) -> str:
    return "gtex_tissue" if _canonical_dataset_label(dataset) == "GTEx" else "ahba_parcel"


def _default_region_order(df: pd.DataFrame, dataset: str) -> list[str]:
    dataset_label = _canonical_dataset_label(dataset)
    if dataset_label == "GTEx":
        return list(GTEX_TENSOR_REGION_ORDER)
    subset = df.loc[df["dataset"].astype(str).str.upper().eq(dataset_label.upper())]
    return sorted(subset["tissue_or_parcel"].astype(str).unique().tolist())


def _with_representative_coordinates(
    df: pd.DataFrame,
    *,
    gtex_rep_mode: str = "centroid",
    gtex_hemi_mode: str = "mirror_left",
) -> pd.DataFrame:
    if {"coord_x", "coord_y", "coord_z"}.issubset(df.columns):
        return df.copy()
    out = df.copy()
    xyz = np.vstack(
        [
            parse_coordinate_representative(c, rep_mode=gtex_rep_mode, hemi_mode=gtex_hemi_mode)
            for c in out["coordinates"]
        ]
    )
    out["coord_x"] = xyz[:, 0]
    out["coord_y"] = xyz[:, 1]
    out["coord_z"] = xyz[:, 2]
    out = out.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    return out


def _matched_region_pairs(
    df: pd.DataFrame,
    *,
    matching_policy: str = MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
    matching_policy_hemi_mode: str = "default",
    collapse_cerebellum: bool = False,
    gtex_rep_mode: str = "centroid",
    gtex_hemi_mode: str = "mirror_left",
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spatial_df = _with_representative_coordinates(
        df,
        gtex_rep_mode=gtex_rep_mode,
        gtex_hemi_mode=gtex_hemi_mode,
    )
    ahba_raw = spatial_df.loc[spatial_df["dataset"].astype(str).str.upper().eq("AHBA")].copy()
    gtex_raw = spatial_df.loc[spatial_df["dataset"].astype(str).str.upper().eq("GTEX")].copy()
    if ahba_raw.empty:
        raise ValueError("region_ordering requires AHBA rows in the samples table.")
    if gtex_raw.empty:
        raise ValueError("region_ordering requires GTEx rows in the samples table.")

    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_mapped = map_gtex_to_target(gtex_raw, target)
    gtex_mapped = apply_gtex_ahba_matching_policy(
        gtex_mapped,
        target,
        matching_policy=str(matching_policy).lower(),
        matching_policy_hemi_mode=str(matching_policy_hemi_mode).lower(),
        collapse_cerebellum=bool(collapse_cerebellum),
        validate_expected=True,
        repo_root=repo_root,
    )

    pairs = (
        gtex_mapped[["tissue_or_parcel", "parcel_idx", "mapped_parcel"]]
        .drop_duplicates()
        .rename(columns={"tissue_or_parcel": "gtex_region", "mapped_parcel": "ahba_region"})
    )
    gtex_rank = {str(region): i for i, region in enumerate(GTEX_TENSOR_REGION_ORDER)}
    pairs["_rank"] = pairs["gtex_region"].map(lambda x: gtex_rank.get(str(x), len(gtex_rank)))
    pairs = (
        pairs.sort_values(["_rank", "parcel_idx", "gtex_region"])
        .drop_duplicates(["parcel_idx"], keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return pairs, gtex_mapped, ahba_raw


def _resolve_region_ordered_dataset(
    df: pd.DataFrame,
    *,
    dataset: str,
    region_ordering: str = REGION_ORDERING_DATASET,
    region_order: Sequence[str] | None = None,
    matching_policy: str = MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
    matching_policy_hemi_mode: str = "default",
    collapse_cerebellum: bool = False,
    gtex_rep_mode: str = "centroid",
    gtex_hemi_mode: str = "mirror_left",
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    dataset_label = _canonical_dataset_label(dataset)
    ordering = str(region_ordering).strip().lower()
    if ordering not in REGION_ORDERINGS:
        raise ValueError(f"region_ordering must be one of {REGION_ORDERINGS}, got {region_ordering!r}")

    if ordering == REGION_ORDERING_DATASET:
        dataset_df = df.loc[df["dataset"].astype(str).str.upper().eq(dataset_label.upper())].copy()
        all_regions = [str(r) for r in region_order] if region_order is not None else _default_region_order(df, dataset_label)
        return dataset_df, all_regions, []

    pairs, gtex_mapped, ahba_raw = _matched_region_pairs(
        df,
        matching_policy=matching_policy,
        matching_policy_hemi_mode=matching_policy_hemi_mode,
        collapse_cerebellum=bool(collapse_cerebellum),
        gtex_rep_mode=gtex_rep_mode,
        gtex_hemi_mode=gtex_hemi_mode,
        repo_root=repo_root,
    )
    matched_gtex_regions = pairs["gtex_region"].astype(str).tolist()
    matched_ahba_regions = pairs["ahba_region"].astype(str).tolist()

    if dataset_label == "GTEx":
        if ordering == REGION_ORDERING_MATCHED:
            return gtex_mapped, matched_gtex_regions, []
        all_ahba_regions = _default_region_order(ahba_raw, "AHBA")
        matched_set = set(matched_ahba_regions)
        superset_regions = matched_ahba_regions + [r for r in all_ahba_regions if r not in matched_set]
        future_imputation_regions = [r for r in all_ahba_regions if r not in matched_set]
        kept_gtex_regions = set(matched_gtex_regions)
        gtex_on_ahba_axis = gtex_mapped.loc[
            gtex_mapped["tissue_or_parcel"].astype(str).isin(kept_gtex_regions)
        ].copy()
        gtex_on_ahba_axis["tissue_or_parcel"] = gtex_on_ahba_axis["mapped_parcel"].astype(str)
        return gtex_on_ahba_axis, superset_regions, future_imputation_regions

    if ordering == REGION_ORDERING_MATCHED:
        return ahba_raw, matched_ahba_regions, []

    all_ahba_regions = _default_region_order(ahba_raw, "AHBA")
    matched_set = set(matched_ahba_regions)
    superset_regions = matched_ahba_regions + [r for r in all_ahba_regions if r not in matched_set]
    return ahba_raw, superset_regions, []


def build_sampled_tensor(
    samples: pd.DataFrame | str | Path,
    *,
    dataset: str,
    region_ordering: str = REGION_ORDERING_DATASET,
    gene_panel: str | Path | Sequence[str] = "richiardi2015",
    n_subjects: int = 20,
    n_regions: int | None = None,
    n_genes: int = 50,
    min_regions_per_subject: int = 5,
    random_seed: int = 42,
    region_order: Sequence[str] | None = None,
    matching_policy: str = MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
    matching_policy_hemi_mode: str = "default",
    collapse_cerebellum: bool = False,
    gtex_rep_mode: str = "centroid",
    gtex_hemi_mode: str = "mirror_left",
    repo_root: str | Path | None = None,
) -> TensorView:
    df = load_gxp_samples_table(samples) if isinstance(samples, (str, Path)) else samples.copy()
    dataset_label = _canonical_dataset_label(dataset)
    dataset_df, all_regions, future_imputation_regions = _resolve_region_ordered_dataset(
        df,
        dataset=dataset_label,
        region_ordering=region_ordering,
        region_order=region_order,
        matching_policy=matching_policy,
        matching_policy_hemi_mode=matching_policy_hemi_mode,
        collapse_cerebellum=bool(collapse_cerebellum),
        gtex_rep_mode=gtex_rep_mode,
        gtex_hemi_mode=gtex_hemi_mode,
        repo_root=repo_root,
    )
    if dataset_df.empty:
        raise ValueError(f"No {dataset_label} rows found in the samples table.")
    gene_cols = [c for c in df.columns if c not in META_COLS]
    panel_genes = load_gene_panel(gene_panel) if isinstance(gene_panel, (str, Path)) else [str(x) for x in gene_panel]
    observed_region_pool = [r for r in all_regions if r not in set(future_imputation_regions)]
    selected_subjects = sample_subjects_by_dataset_region_coverage(
        dataset_df,
        dataset=dataset_label,
        n=n_subjects,
        seed=random_seed,
        min_regions=min_regions_per_subject,
        region_order=observed_region_pool,
    )
    selected_regions = (
        _take_region_prefix(all_regions, n=n_regions)
        if str(region_ordering).strip().lower() != REGION_ORDERING_DATASET
        else sample_regions(all_regions, n=n_regions, seed=random_seed)
    )
    selected_genes = sample_genes(panel_genes, gene_cols, n=n_genes, seed=random_seed)
    region_axis_kind = (
        "ahba_parcel"
        if dataset_label == "GTEx"
        and str(region_ordering).strip().lower() == REGION_ORDERING_MATCHED_SUPERSET
        else _default_region_axis_kind(dataset_label)
    )
    return build_dataset_region_tensor(
        dataset_df,
        dataset=dataset_label,
        subjects=selected_subjects,
        genes=selected_genes,
        region_order=selected_regions,
        region_axis_kind=region_axis_kind,
        future_imputation_regions=future_imputation_regions,
    )


def summarize_tensor_selection(tensor_view: TensorView) -> pd.DataFrame:
    return pd.concat(
        [
            pd.Series(tensor_view.subjects, name="selected_subject"),
            pd.Series(tensor_view.regions, name="selected_region"),
            pd.Series(tensor_view.genes, name="selected_gene"),
        ],
        axis=1,
    )


def _robust_vrange(values: np.ndarray, *, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if int(finite.size) == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, lower))
    vmax = float(np.nanpercentile(finite, upper))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _resolve_box_aspect(
    values: np.ndarray,
    box_aspect: str | Sequence[float] | None,
    axis_assignment: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    if box_aspect is None or (isinstance(box_aspect, str) and box_aspect.lower() == "data"):
        return tuple(float(max(1, int(x))) for x in values.shape)
    if isinstance(box_aspect, str) and box_aspect.lower() == "compressed_data":
        aspect = tuple(float(max(1.0, math.sqrt(max(1, int(x))))) for x in values.shape)
        if axis_assignment is None:
            return aspect
        return tuple(
            float(value) * _COMPRESSED_BOX_ASPECT_FACTORS[str(semantic)]
            for value, semantic in zip(aspect, axis_assignment)
        )
    if isinstance(box_aspect, str) and box_aspect.lower() == "equal":
        return (1.0, 1.0, 1.0)
    if len(box_aspect) != 3:
        raise ValueError("box_aspect must be None, 'data', 'compressed_data', 'equal', or a length-3 sequence.")
    return tuple(float(max(1e-6, float(x))) for x in box_aspect)


def _is_auto(value: object) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _resolve_axis_assignment(axis_assignment: Sequence[str] | None) -> tuple[str, str, str]:
    if axis_assignment is None:
        return DEFAULT_TENSOR_AXIS_ASSIGNMENT
    axis_assignment = tuple(str(x).lower() for x in axis_assignment)
    if len(axis_assignment) != 3 or set(axis_assignment) != set(TENSOR_AXIS_COMPONENTS):
        raise ValueError(
            "axis_assignment must be a permutation of ('gene', 'subject', 'region')."
        )
    return axis_assignment


def _axis_payload(
    tensor_view: TensorView,
    axis_assignment: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]], tuple[str, str, str]]:
    axis_assignment = _resolve_axis_assignment(axis_assignment)
    base_values = np.transpose(tensor_view.values, (2, 0, 1))
    base_observed = np.transpose(
        np.repeat(tensor_view.observed_mask[:, :, None], len(tensor_view.genes), axis=2),
        (2, 0, 1),
    )
    semantic_to_base_axis = {"gene": 0, "subject": 1, "region": 2}
    perm = tuple(semantic_to_base_axis[name] for name in axis_assignment)
    values = np.transpose(base_values, perm)
    observed = np.transpose(base_observed, perm)
    labels = {
        "gene": list(tensor_view.genes),
        "subject": list(tensor_view.subjects),
        "region": list(tensor_view.regions),
    }
    return values, observed, labels, axis_assignment


def _resolve_tick_step(requested: int | str | None) -> int:
    if requested is None:
        return 1
    return max(1, int(requested))


def _format_axis_labels(
    labels: Sequence[str],
    *,
    semantic_name: str,
    mode: str,
    dataset: str | None = None,
) -> list[str]:
    mode = str(mode).lower()
    if semantic_name == "region":
        return [pretty_gtex_label(x) for x in labels]
    if semantic_name == "subject":
        if mode == "index":
            return [f"S{i+1}" for i in range(len(labels))]
        if mode == "suffix":
            return [str(x).split("-")[-1] for x in labels]
        if str(dataset).upper() == "AHBA":
            return [str(x) if str(x).upper().startswith("AHBA-") else f"AHBA-{x}" for x in labels]
        return [str(x) for x in labels]
    if semantic_name == "gene":
        if mode == "index":
            return [f"G{i+1}" for i in range(len(labels))]
        return [str(x) for x in labels]
    return [str(x) for x in labels]


def _axis_count_label(semantic_name: str, count: int) -> str:
    noun = {"gene": "gene", "region": "region", "subject": "subject"}[semantic_name]
    suffix = "" if int(count) == 1 else "s"
    return f"{int(count)} {noun}{suffix}"


def _resolve_tensor_layout(
    *,
    values_shape: Sequence[int],
    axis_assignment: Sequence[str],
    tick_step_map: Mapping[str, int],
    show_ticklabel_map: Mapping[str, bool],
    figsize: tuple[float, float] | str,
    axes_bbox: Sequence[float] | str,
    colorbar_bbox: Sequence[float] | str,
    box_zoom: float | str,
) -> tuple[tuple[float, float], tuple[float, float, float, float], tuple[float, float, float, float], float]:
    if not any(_is_auto(value) for value in (figsize, axes_bbox, colorbar_bbox, box_zoom)):
        return (
            tuple(float(x) for x in figsize),
            tuple(float(x) for x in axes_bbox),
            tuple(float(x) for x in colorbar_bbox),
            float(box_zoom),
        )

    axis_lengths = {semantic: int(values_shape[i]) for i, semantic in enumerate(axis_assignment)}
    visible_ticks = {
        semantic: (
            int(math.ceil(axis_lengths[semantic] / max(1, int(tick_step_map[semantic]))))
            if show_ticklabel_map[semantic]
            else 0
        )
        for semantic in TENSOR_AXIS_COMPONENTS
    }
    n_x = int(values_shape[0])
    n_y_ticks = visible_ticks[axis_assignment[1]]
    n_z_ticks = visible_ticks[axis_assignment[2]]

    auto_width = _clamp(11.0 + 0.006 * max(0, n_x - 100), 9.0, 14.0)
    auto_height = _clamp(
        6.0
        + 0.055 * max(0, n_y_ticks - 12)
        + 0.035 * max(0, n_z_ticks - 10),
        5.2,
        10.0,
    )
    resolved_figsize = (auto_width, auto_height) if _is_auto(figsize) else tuple(float(x) for x in figsize)

    auto_axes_bbox = (0.10, 0.04, 0.76, 0.92)
    resolved_axes_bbox = auto_axes_bbox if _is_auto(axes_bbox) else tuple(float(x) for x in axes_bbox)

    axes_left, axes_bottom, axes_width, axes_height = resolved_axes_bbox
    auto_colorbar_bbox = (
        _clamp(axes_left + axes_width - 0.03, 0.78, 0.92),
        axes_bottom + 0.33 * axes_height,
        0.018,
        0.46 * axes_height,
    )
    resolved_colorbar_bbox = (
        auto_colorbar_bbox if _is_auto(colorbar_bbox) else tuple(float(x) for x in colorbar_bbox)
    )

    max_dim = max(int(x) for x in values_shape)
    auto_box_zoom = _clamp(1.45 - 0.0018 * max(0, max_dim - 100), 0.95, 1.45)
    resolved_box_zoom = auto_box_zoom if _is_auto(box_zoom) else float(box_zoom)
    return resolved_figsize, resolved_axes_bbox, resolved_colorbar_bbox, resolved_box_zoom


def plot_gtex_observation_mask(
    tensor_view: TensorView,
    *,
    figsize: tuple[float, float] = (8.5, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    mask = tensor_view.observed_mask.T.astype(float)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = ListedColormap(["#edf2f7", "#2b6cb0"])
    ax.imshow(mask, aspect="auto", interpolation="none", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("GTEx observation mask by subject and region")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Region")
    ax.set_xticks(np.arange(len(tensor_view.subjects)))
    ax.set_xticklabels(tensor_view.subjects, rotation=90)
    ax.set_yticks(np.arange(len(tensor_view.regions)))
    ax.set_yticklabels([pretty_gtex_label(x) for x in tensor_view.regions])
    for tick, region in zip(ax.get_yticklabels(), tensor_view.regions):
        tick.set_color(GTEX_TENSOR_GROUP_COLOR[GTEX_TENSOR_REGION_GROUP[region]])
    apply_tick_style(ax)
    legend_handles = [
        Patch(facecolor="#2b6cb0", edgecolor="none", label="Observed"),
        Patch(facecolor="#edf2f7", edgecolor="none", label="Missing"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cortical"], edgecolor="none", label="Cortical"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["basal_ganglia"], edgecolor="none", label="Basal ganglia"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["limbic_midbrain"], edgecolor="none", label="Limbic / midbrain"),
        Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cerebellar"], edgecolor="none", label="Cerebellar"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return fig, ax


def plot_gtex_tensor_voxels(
    tensor_view: TensorView,
    *,
    cmap: str = "viridis",
    missing_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0),
    edgecolor: str = "white",
    linewidth: float = 0.18,
    figsize: tuple[float, float] | str = (13.5, 9.5),
    dpi: int | float | None = None,
    axes_bbox: Sequence[float] | str = (0.03, 0.06, 0.78, 0.88),
    colorbar_bbox: Sequence[float] | str = (0.86, 0.16, 0.025, 0.68),
    box_aspect: str | Sequence[float] | None = "compressed_data",
    box_zoom: float | str = 1.0,
    title_pad: float = 2.0,
    axis_assignment: Sequence[str] | None = None,
    elev: float = 20.0,
    azim: float = 24.0,
    subject_tick_step: int | str | None = 1,
    region_tick_step: int | str | None = 1,
    gene_tick_step: int | str | None = 1,
    tick_label_pad: float = 0.0,
    show_axis_labels: bool = False,
    show_x_axis_label: bool | None = True,
    show_y_axis_label: bool | None = None,
    show_z_axis_label: bool | None = None,
    show_subject_ticklabels: bool = True,
    show_region_ticklabels: bool = True,
    show_gene_ticklabels: bool = False,
    subject_label_mode: str = "full",
    gene_label_mode: str = "full",
    show_region_label_colors: bool = False,
    show_axis_lines: bool = True,
    show_tick_lines: bool = True,
    show_grid: bool = False,
    show_legend: bool = False,
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    values, observed, labels_by_semantic, axis_assignment = _axis_payload(tensor_view, axis_assignment)
    filled = np.ones(values.shape, dtype=bool)
    fonts = _resolve_fonts(_VOXEL_TENSOR_FONTS, font_sizes)
    label_pads = _VOXEL_TENSOR_AXIS_LABEL_PADS
    tick_step_map = {
        "gene": _resolve_tick_step(gene_tick_step),
        "subject": _resolve_tick_step(subject_tick_step),
        "region": _resolve_tick_step(region_tick_step),
    }
    show_ticklabel_map = {
        "gene": show_gene_ticklabels,
        "subject": show_subject_ticklabels,
        "region": show_region_ticklabels,
    }
    figsize, axes_bbox, colorbar_bbox, box_zoom = _resolve_tensor_layout(
        values_shape=values.shape,
        axis_assignment=axis_assignment,
        tick_step_map=tick_step_map,
        show_ticklabel_map=show_ticklabel_map,
        figsize=figsize,
        axes_bbox=axes_bbox,
        colorbar_bbox=colorbar_bbox,
        box_zoom=box_zoom,
    )

    vmin, vmax = _robust_vrange(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    facecolors = np.empty(values.shape + (4,), dtype=np.float32)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if not observed[i, j, k] or not np.isfinite(values[i, j, k]):
                    facecolors[i, j, k] = missing_rgba
                else:
                    facecolors[i, j, k] = scalar_map.to_rgba(float(values[i, j, k]))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(axes_bbox, projection="3d")
    ax.voxels(
        filled,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidth=linewidth,
        shade=False,
    )
    resolved_box_aspect = _resolve_box_aspect(values, box_aspect, axis_assignment)
    try:
        ax.set_box_aspect(resolved_box_aspect, zoom=float(box_zoom))
    except TypeError:
        ax.set_box_aspect(resolved_box_aspect)
    ax.view_init(elev=elev, azim=azim)
    axis_label_map = {"gene": "Genes", "subject": "Subjects", "region": "Regions"}
    axis_counts = {
        "gene": len(labels_by_semantic["gene"]),
        "region": len(labels_by_semantic["region"]),
        "subject": len(labels_by_semantic["subject"]),
    }
    title_parts = [
        f"{axis_name}={_axis_count_label(semantic_name, axis_counts[semantic_name])}"
        for axis_name, semantic_name in zip(("x", "y", "z"), axis_assignment)
    ]
    ax.set_title(
        f"{tensor_view.dataset} Input Tensor ({', '.join(title_parts)})",
        fontsize=fonts["title"],
        pad=float(title_pad),
    )
    show_x_axis_label = show_axis_labels if show_x_axis_label is None else show_x_axis_label
    show_y_axis_label = show_axis_labels if show_y_axis_label is None else show_y_axis_label
    show_z_axis_label = show_axis_labels if show_z_axis_label is None else show_z_axis_label

    xlabel_kwargs = {"fontsize": fonts["xlabel"]}
    ylabel_kwargs = {"fontsize": fonts["ylabel"]}
    zlabel_kwargs = {"fontsize": fonts["zlabel"]}

    def _resolve_labelpad(semantic: str) -> float | None:
        if show_ticklabel_map[semantic]:
            return label_pads[semantic]
        return _VOXEL_TENSOR_AXIS_LABEL_ONLY_PADS[semantic]

    if show_x_axis_label:
        pad = _resolve_labelpad(axis_assignment[0])
        if pad is not None:
            xlabel_kwargs["labelpad"] = pad
    if show_y_axis_label:
        pad = _resolve_labelpad(axis_assignment[1])
        if pad is not None:
            ylabel_kwargs["labelpad"] = pad
    if show_z_axis_label:
        pad = _resolve_labelpad(axis_assignment[2])
        if pad is not None:
            zlabel_kwargs["labelpad"] = pad
    ax.set_xlabel(axis_label_map[axis_assignment[0]] if show_x_axis_label else "", **xlabel_kwargs)
    ax.set_ylabel(axis_label_map[axis_assignment[1]] if show_y_axis_label else "", **ylabel_kwargs)
    ax.set_zlabel(axis_label_map[axis_assignment[2]] if show_z_axis_label else "", **zlabel_kwargs)

    x_labels = labels_by_semantic[axis_assignment[0]]
    y_labels = labels_by_semantic[axis_assignment[1]]
    z_labels = labels_by_semantic[axis_assignment[2]]
    x_display = _format_axis_labels(x_labels, semantic_name=axis_assignment[0], mode=subject_label_mode if axis_assignment[0] == "subject" else gene_label_mode if axis_assignment[0] == "gene" else "full", dataset=tensor_view.dataset)
    y_display = _format_axis_labels(y_labels, semantic_name=axis_assignment[1], mode=subject_label_mode if axis_assignment[1] == "subject" else gene_label_mode if axis_assignment[1] == "gene" else "full", dataset=tensor_view.dataset)
    z_display = _format_axis_labels(z_labels, semantic_name=axis_assignment[2], mode=subject_label_mode if axis_assignment[2] == "subject" else gene_label_mode if axis_assignment[2] == "gene" else "full", dataset=tensor_view.dataset)

    x_idx = np.arange(0, len(x_labels), tick_step_map[axis_assignment[0]])
    x_ticks = x_idx + 0.5
    y_idx = np.arange(0, len(y_labels), tick_step_map[axis_assignment[1]])
    y_ticks = y_idx + 0.5
    z_idx = np.arange(0, len(z_labels), tick_step_map[axis_assignment[2]])
    z_ticks = z_idx + 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [x_display[i] for i in x_idx.tolist()] if show_ticklabel_map[axis_assignment[0]] else [],
        fontsize=fonts["xtick"],
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [y_display[i] for i in y_idx.tolist()] if show_ticklabel_map[axis_assignment[1]] else [],
        fontsize=fonts["ytick"],
        rotation=30,
    )
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(
        [z_display[i] for i in z_idx.tolist()] if show_ticklabel_map[axis_assignment[2]] else [],
        fontsize=fonts["ztick"],
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for label in ax.get_zticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="x", pad=float(tick_label_pad))
    ax.tick_params(axis="y", pad=float(tick_label_pad))
    ax.tick_params(axis="z", pad=float(tick_label_pad))

    if show_region_label_colors:
        for tick, semantic_name, region_lookup in (
            (ax.get_xticklabels(), axis_assignment[0], [x_labels[i] for i in x_idx.tolist()]),
            (ax.get_yticklabels(), axis_assignment[1], [y_labels[i] for i in y_idx.tolist()]),
            (ax.get_zticklabels(), axis_assignment[2], [z_labels[i] for i in z_idx.tolist()]),
        ):
            if semantic_name == "region" and show_ticklabel_map[semantic_name]:
                for label, region in zip(tick, region_lookup):
                    group = GTEX_TENSOR_REGION_GROUP.get(region)
                    if group is not None:
                        label.set_color(GTEX_TENSOR_GROUP_COLOR[group])

    ax.grid(bool(show_grid))
    if not show_grid:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.fill = False
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
    if not show_axis_lines:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass

    cax = fig.add_axes(colorbar_bbox)
    cbar = fig.colorbar(scalar_map, cax=cax)
    cbar_label = "Microarray Intensity" if str(tensor_view.dataset).upper() == "AHBA" else "log1p(TPM)"
    cbar.set_label(cbar_label, fontsize=fonts["cbar_label"])
    cbar.ax.tick_params(labelsize=fonts["cbar_label"])

    if show_legend:
        legend_handles = [
            Patch(facecolor=missing_rgba[:3], edgecolor="none", alpha=missing_rgba[3], label="Missing region vector"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cortical"], edgecolor="none", label="Cortical"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["basal_ganglia"], edgecolor="none", label="Basal ganglia"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["limbic_midbrain"], edgecolor="none", label="Limbic / midbrain"),
            Patch(facecolor=GTEX_TENSOR_GROUP_COLOR["cerebellar"], edgecolor="none", label="Cerebellar"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=fonts["legend"],
        )
    return fig, ax


def _per_dataset_minmax_scale(
    values: np.ndarray,
    dataset_slices: Mapping[str, slice],
    *,
    lower: float = 2.0,
    upper: float = 98.0,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    scaled = np.full_like(values, np.nan, dtype=np.float64)
    ranges: dict[str, tuple[float, float]] = {}
    for name, sl in dataset_slices.items():
        block = values[sl]
        vmin, vmax = _robust_vrange(block, lower=lower, upper=upper)
        ranges[name] = (float(vmin), float(vmax))
        denom = max(vmax - vmin, 1e-12)
        finite_mask = np.isfinite(block)
        out_block = np.full_like(block, np.nan, dtype=np.float64)
        out_block[finite_mask] = np.clip((block[finite_mask] - vmin) / denom, 0.0, 1.0)
        scaled[sl] = out_block
    return scaled, ranges


def _joint_axis_payload(
    joint_view: "JointTensorView",
    axis_assignment: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]], tuple[str, str, str]]:
    axis_assignment = _resolve_axis_assignment(axis_assignment)
    base_values = np.transpose(joint_view.values, (2, 0, 1))
    base_observed = np.transpose(
        np.repeat(joint_view.observed_mask[:, :, None], len(joint_view.genes), axis=2),
        (2, 0, 1),
    )
    base_future = np.transpose(
        np.repeat(joint_view.future_imputation_mask[:, :, None], len(joint_view.genes), axis=2),
        (2, 0, 1),
    )
    semantic_to_base_axis = {"gene": 0, "subject": 1, "region": 2}
    perm = tuple(semantic_to_base_axis[name] for name in axis_assignment)
    values = np.transpose(base_values, perm)
    observed = np.transpose(base_observed, perm)
    future = np.transpose(base_future, perm)
    labels = {
        "gene": list(joint_view.genes),
        "subject": list(joint_view.subjects),
        "region": list(joint_view.regions),
    }
    return values, observed, future, labels, axis_assignment


def _format_joint_subject_labels(
    joint_view: "JointTensorView",
    mode: str = "full",
) -> list[str]:
    mode = str(mode).lower()
    labels: list[str] = []
    for idx, (subj, ds) in enumerate(zip(joint_view.subjects, joint_view.subject_datasets)):
        ds_key = str(ds).upper()
        if mode == "index":
            prefix = "G" if ds_key == "GTEX" else "A"
            labels.append(f"{prefix}{idx + 1}")
            continue
        if mode == "suffix":
            labels.append(str(subj).split("-")[-1])
            continue
        s = str(subj)
        if ds_key == "AHBA" and not s.upper().startswith("AHBA-"):
            s = f"AHBA-{s}"
        labels.append(s)
    return labels


def plot_joint_tensor_voxels(
    joint_view: "JointTensorView",
    *,
    cmap: str = "viridis",
    missing_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0),
    future_imputation_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 0.45),
    edgecolor: str = "white",
    linewidth: float = 0.18,
    figsize: tuple[float, float] | str = "auto",
    dpi: int | float | None = None,
    axes_bbox: Sequence[float] | str = "auto",
    colorbar_bbox: Sequence[float] | str = "auto",
    box_aspect: str | Sequence[float] | None = "compressed_data",
    box_zoom: float | str = "auto",
    title_pad: float = 2.0,
    axis_assignment: Sequence[str] | None = None,
    elev: float = 20.0,
    azim: float = 24.0,
    subject_tick_step: int | str | None = 1,
    region_tick_step: int | str | None = 1,
    gene_tick_step: int | str | None = 1,
    tick_label_pad: float = 0.0,
    show_axis_labels: bool = False,
    show_x_axis_label: bool | None = True,
    show_y_axis_label: bool | None = None,
    show_z_axis_label: bool | None = None,
    show_subject_ticklabels: bool = True,
    show_region_ticklabels: bool = True,
    show_gene_ticklabels: bool = False,
    subject_label_mode: str = "full",
    gene_label_mode: str = "full",
    show_region_label_colors: bool = False,
    show_axis_lines: bool = True,
    show_tick_lines: bool = True,
    show_grid: bool = False,
    show_legend: bool = False,
    dataset_gap: int = 1,
    mask_render_mode: str = "two_pass",
    cbar_label: str = "Normalized expression (per-dataset min-max)",
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    scaled_values, ranges = _per_dataset_minmax_scale(joint_view.values, joint_view.dataset_slices)

    joint_for_payload = JointTensorView(
        values=scaled_values,
        observed_mask=joint_view.observed_mask,
        future_imputation_mask=joint_view.future_imputation_mask,
        subjects=joint_view.subjects,
        subject_datasets=joint_view.subject_datasets,
        dataset_slices=joint_view.dataset_slices,
        regions=joint_view.regions,
        genes=joint_view.genes,
        region_axis_kind=joint_view.region_axis_kind,
        matched_region_count=joint_view.matched_region_count,
        source_views=joint_view.source_views,
    )

    values, observed, future, labels_by_semantic, axis_assignment = _joint_axis_payload(
        joint_for_payload, axis_assignment
    )

    fonts = _resolve_fonts(_VOXEL_TENSOR_FONTS, font_sizes)
    label_pads = _VOXEL_TENSOR_AXIS_LABEL_PADS
    tick_step_map = {
        "gene": _resolve_tick_step(gene_tick_step),
        "subject": _resolve_tick_step(subject_tick_step),
        "region": _resolve_tick_step(region_tick_step),
    }
    show_ticklabel_map = {
        "gene": show_gene_ticklabels,
        "subject": show_subject_ticklabels,
        "region": show_region_ticklabels,
    }

    figsize, axes_bbox, colorbar_bbox, box_zoom = _resolve_tensor_layout(
        values_shape=values.shape,
        axis_assignment=axis_assignment,
        tick_step_map=tick_step_map,
        show_ticklabel_map=show_ticklabel_map,
        figsize=figsize,
        axes_bbox=axes_bbox,
        colorbar_bbox=colorbar_bbox,
        box_zoom=box_zoom,
    )

    norm = Normalize(vmin=0.0, vmax=1.0)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    facecolors = np.empty(values.shape + (4,), dtype=np.float32)
    filled = np.ones(values.shape, dtype=bool)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if future[i, j, k]:
                    facecolors[i, j, k] = future_imputation_rgba
                elif not observed[i, j, k] or not np.isfinite(values[i, j, k]):
                    facecolors[i, j, k] = missing_rgba
                else:
                    facecolors[i, j, k] = scalar_map.to_rgba(float(values[i, j, k]))

    if dataset_gap and dataset_gap > 0 and axis_assignment.count("subject") == 1:
        subject_axis = axis_assignment.index("subject")
        gtex_slice = joint_view.dataset_slices.get("GTEx")
        if gtex_slice is not None:
            boundary = int(gtex_slice.stop)
            for offset in range(int(dataset_gap)):
                idx = boundary + offset - int(dataset_gap) // 2
                if 0 <= idx < values.shape[subject_axis]:
                    sl = [slice(None)] * 3
                    sl[subject_axis] = idx
                    filled[tuple(sl)] = False

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(axes_bbox, projection="3d")

    mode = str(mask_render_mode).strip().lower()
    if mode not in ("single_pass", "two_pass"):
        raise ValueError(
            f"mask_render_mode must be 'single_pass' or 'two_pass', got {mask_render_mode!r}"
        )
    if mode == "single_pass":
        ax.voxels(
            filled,
            facecolors=facecolors,
            edgecolors=edgecolor,
            linewidth=linewidth,
            shade=False,
        )
    else:
        filled_opaque = filled & ~future
        filled_future = filled & future
        ax.voxels(
            filled_opaque,
            facecolors=facecolors,
            edgecolors=edgecolor,
            linewidth=linewidth,
            shade=False,
            zorder=1,
        )
        if bool(filled_future.any()):
            ax.voxels(
                filled_future,
                facecolors=facecolors,
                edgecolors="none",
                linewidth=0,
                shade=False,
                zorder=2,
            )

    resolved_box_aspect = _resolve_box_aspect(values, box_aspect, axis_assignment)
    try:
        ax.set_box_aspect(resolved_box_aspect, zoom=float(box_zoom))
    except TypeError:
        ax.set_box_aspect(resolved_box_aspect)
    ax.view_init(elev=elev, azim=azim)

    axis_label_map = {"gene": "Genes", "subject": "Subjects", "region": "Regions"}
    axis_counts = {
        "gene": len(labels_by_semantic["gene"]),
        "region": len(labels_by_semantic["region"]),
        "subject": len(labels_by_semantic["subject"]),
    }
    title_parts = [
        f"{axis_name}={_axis_count_label(semantic_name, axis_counts[semantic_name])}"
        for axis_name, semantic_name in zip(("x", "y", "z"), axis_assignment)
    ]
    ax.set_title(
        f"GTEx + AHBA Joint Tensor ({', '.join(title_parts)})",
        fontsize=fonts["title"],
        pad=float(title_pad),
    )

    show_x_axis_label = show_axis_labels if show_x_axis_label is None else show_x_axis_label
    show_y_axis_label = show_axis_labels if show_y_axis_label is None else show_y_axis_label
    show_z_axis_label = show_axis_labels if show_z_axis_label is None else show_z_axis_label

    xlabel_kwargs = {"fontsize": fonts["xlabel"]}
    ylabel_kwargs = {"fontsize": fonts["ylabel"]}
    zlabel_kwargs = {"fontsize": fonts["zlabel"]}

    def _resolve_labelpad(semantic: str) -> float | None:
        if show_ticklabel_map[semantic]:
            return label_pads[semantic]
        return _VOXEL_TENSOR_AXIS_LABEL_ONLY_PADS[semantic]

    if show_x_axis_label:
        pad = _resolve_labelpad(axis_assignment[0])
        if pad is not None:
            xlabel_kwargs["labelpad"] = pad
    if show_y_axis_label:
        pad = _resolve_labelpad(axis_assignment[1])
        if pad is not None:
            ylabel_kwargs["labelpad"] = pad
    if show_z_axis_label:
        pad = _resolve_labelpad(axis_assignment[2])
        if pad is not None:
            zlabel_kwargs["labelpad"] = pad

    ax.set_xlabel(axis_label_map[axis_assignment[0]] if show_x_axis_label else "", **xlabel_kwargs)
    ax.set_ylabel(axis_label_map[axis_assignment[1]] if show_y_axis_label else "", **ylabel_kwargs)
    ax.set_zlabel(axis_label_map[axis_assignment[2]] if show_z_axis_label else "", **zlabel_kwargs)

    joint_subject_labels = _format_joint_subject_labels(joint_view, mode=subject_label_mode)

    def _labels_for(semantic: str) -> list[str]:
        if semantic == "subject":
            return joint_subject_labels
        if semantic == "gene":
            if str(gene_label_mode).lower() == "index":
                return [f"G{i + 1}" for i in range(len(joint_view.genes))]
            return list(joint_view.genes)
        if semantic == "region":
            return [pretty_gtex_label(x) for x in joint_view.regions]
        return [str(x) for x in labels_by_semantic[semantic]]

    x_display = _labels_for(axis_assignment[0])
    y_display = _labels_for(axis_assignment[1])
    z_display = _labels_for(axis_assignment[2])

    x_labels = labels_by_semantic[axis_assignment[0]]
    y_labels = labels_by_semantic[axis_assignment[1]]
    z_labels = labels_by_semantic[axis_assignment[2]]

    x_idx = np.arange(0, len(x_labels), tick_step_map[axis_assignment[0]])
    y_idx = np.arange(0, len(y_labels), tick_step_map[axis_assignment[1]])
    z_idx = np.arange(0, len(z_labels), tick_step_map[axis_assignment[2]])

    ax.set_xticks(x_idx + 0.5)
    ax.set_xticklabels(
        [x_display[i] for i in x_idx.tolist()] if show_ticklabel_map[axis_assignment[0]] else [],
        fontsize=fonts["xtick"],
    )
    ax.set_yticks(y_idx + 0.5)
    ax.set_yticklabels(
        [y_display[i] for i in y_idx.tolist()] if show_ticklabel_map[axis_assignment[1]] else [],
        fontsize=fonts["ytick"],
        rotation=30,
    )
    ax.set_zticks(z_idx + 0.5)
    ax.set_zticklabels(
        [z_display[i] for i in z_idx.tolist()] if show_ticklabel_map[axis_assignment[2]] else [],
        fontsize=fonts["ztick"],
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for label in ax.get_zticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="x", pad=float(tick_label_pad))
    ax.tick_params(axis="y", pad=float(tick_label_pad))
    ax.tick_params(axis="z", pad=float(tick_label_pad))

    if show_region_label_colors:
        for tick, semantic_name, region_lookup in (
            (ax.get_xticklabels(), axis_assignment[0], [x_labels[i] for i in x_idx.tolist()]),
            (ax.get_yticklabels(), axis_assignment[1], [y_labels[i] for i in y_idx.tolist()]),
            (ax.get_zticklabels(), axis_assignment[2], [z_labels[i] for i in z_idx.tolist()]),
        ):
            if semantic_name == "region" and show_ticklabel_map[semantic_name]:
                for label, region in zip(tick, region_lookup):
                    group = GTEX_TENSOR_REGION_GROUP.get(region)
                    if group is not None:
                        label.set_color(GTEX_TENSOR_GROUP_COLOR[group])

    ax.grid(bool(show_grid))
    if not show_grid:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.fill = False
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
    if not show_axis_lines:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass

    cax = fig.add_axes(colorbar_bbox)
    cbar = fig.colorbar(scalar_map, cax=cax)
    cbar.set_label(cbar_label, fontsize=fonts["cbar_label"])
    cbar.ax.tick_params(labelsize=fonts["cbar_label"])

    if show_legend:
        legend_handles = [
            Patch(facecolor=missing_rgba[:3], edgecolor="none", alpha=missing_rgba[3], label="Missing"),
            Patch(
                facecolor=future_imputation_rgba[:3],
                edgecolor="none",
                alpha=future_imputation_rgba[3],
                label="Future imputation (GTEx)",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=fonts["legend"],
        )

    ax._joint_dataset_ranges = ranges  # for downstream inspection
    return fig, ax


def plot_sampled_tensor(
    samples: pd.DataFrame | str | Path = "data/raw/gxp_samples.csv",
    *,
    dataset: str,
    region_ordering: str = REGION_ORDERING_DATASET,
    gene_panel: str | Path | Sequence[str] = "richiardi2015",
    n_subjects: int = 20,
    n_regions: int | None = None,
    n_genes: int = 50,
    min_regions_per_subject: int = 5,
    random_seed: int = 42,
    region_order: Sequence[str] | None = None,
    matching_policy: str = MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
    matching_policy_hemi_mode: str = "default",
    collapse_cerebellum: bool = False,
    gtex_rep_mode: str = "centroid",
    gtex_hemi_mode: str = "mirror_left",
    repo_root: str | Path | None = None,
    cmap: str = "viridis",
    missing_rgba: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0),
    edgecolor: str = "white",
    linewidth: float = 0.18,
    figsize: tuple[float, float] | str = (13.5, 9.5),
    dpi: int | float | None = None,
    axes_bbox: Sequence[float] | str = (0.03, 0.06, 0.78, 0.88),
    colorbar_bbox: Sequence[float] | str = (0.86, 0.16, 0.025, 0.68),
    box_aspect: str | Sequence[float] | None = "compressed_data",
    box_zoom: float | str = 1.0,
    title_pad: float = 2.0,
    axis_assignment: Sequence[str] | None = None,
    elev: float = 20.0,
    azim: float = 24.0,
    subject_tick_step: int | str | None = 1,
    region_tick_step: int | str | None = 1,
    gene_tick_step: int | str | None = 1,
    tick_label_pad: float = 0.0,
    show_axis_labels: bool = False,
    show_x_axis_label: bool | None = True,
    show_y_axis_label: bool | None = None,
    show_z_axis_label: bool | None = None,
    show_subject_ticklabels: bool = True,
    show_region_ticklabels: bool = True,
    show_gene_ticklabels: bool = False,
    subject_label_mode: str = "full",
    gene_label_mode: str = "full",
    show_region_label_colors: bool = False,
    show_axis_lines: bool = True,
    show_tick_lines: bool = True,
    show_grid: bool = False,
    show_legend: bool = False,
    font_sizes: Mapping[str, int | float | str] | None = None,
) -> tuple[plt.Figure, plt.Axes, TensorView, pd.DataFrame]:
    tensor_view = build_sampled_tensor(
        samples,
        dataset=dataset,
        region_ordering=region_ordering,
        gene_panel=gene_panel,
        n_subjects=n_subjects,
        n_regions=n_regions,
        n_genes=n_genes,
        min_regions_per_subject=min_regions_per_subject,
        random_seed=random_seed,
        region_order=region_order,
        matching_policy=matching_policy,
        matching_policy_hemi_mode=matching_policy_hemi_mode,
        collapse_cerebellum=bool(collapse_cerebellum),
        gtex_rep_mode=gtex_rep_mode,
        gtex_hemi_mode=gtex_hemi_mode,
        repo_root=repo_root,
    )
    fig, ax = plot_gtex_tensor_voxels(
        tensor_view,
        cmap=cmap,
        missing_rgba=missing_rgba,
        edgecolor=edgecolor,
        linewidth=linewidth,
        figsize=figsize,
        dpi=dpi,
        axes_bbox=axes_bbox,
        colorbar_bbox=colorbar_bbox,
        box_aspect=box_aspect,
        box_zoom=box_zoom,
        title_pad=title_pad,
        axis_assignment=axis_assignment,
        elev=elev,
        azim=azim,
        subject_tick_step=subject_tick_step,
        region_tick_step=region_tick_step,
        gene_tick_step=gene_tick_step,
        tick_label_pad=tick_label_pad,
        show_axis_labels=show_axis_labels,
        show_x_axis_label=show_x_axis_label,
        show_y_axis_label=show_y_axis_label,
        show_z_axis_label=show_z_axis_label,
        show_subject_ticklabels=show_subject_ticklabels,
        show_region_ticklabels=show_region_ticklabels,
        show_gene_ticklabels=show_gene_ticklabels,
        subject_label_mode=subject_label_mode,
        gene_label_mode=gene_label_mode,
        show_region_label_colors=show_region_label_colors,
        show_axis_lines=show_axis_lines,
        show_tick_lines=show_tick_lines,
        show_grid=show_grid,
        show_legend=show_legend,
        font_sizes=font_sizes,
    )
    return fig, ax, tensor_view, summarize_tensor_selection(tensor_view)


__all__ = [
    "DEFAULT_TENSOR_AXIS_ASSIGNMENT",
    "GTEX_TENSOR_GROUP_COLOR",
    "GTEX_TENSOR_REGION_GROUP",
    "GTEX_TENSOR_REGION_ORDER",
    "JointTensorView",
    "REGION_ORDERING_DATASET",
    "REGION_ORDERING_MATCHED",
    "REGION_ORDERING_MATCHED_SUPERSET",
    "REGION_ORDERINGS",
    "TensorView",
    "build_dataset_region_tensor",
    "build_joint_tensor_view",
    "build_sampled_tensor",
    "load_gene_panel",
    "load_gxp_samples_table",
    "plot_gtex_observation_mask",
    "plot_gtex_tensor_voxels",
    "plot_joint_tensor_voxels",
    "plot_sampled_tensor",
    "sample_regions",
    "sample_genes",
    "sample_subjects",
    "sample_subjects_by_dataset_region_coverage",
    "sample_subjects_by_region_coverage",
    "summarize_joint_selection",
    "summarize_tensor_selection",
]
