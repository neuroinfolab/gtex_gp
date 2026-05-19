from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


META_COLS = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]
DEFAULT_CEREBELLAR_GTEX_TISSUES = ("brain - cerebellum", "brain - cerebellar hemisphere")
MATCHING_POLICY_CENTROIDS = "centroids"
MATCHING_POLICY_CENTROIDS_AND_VOLUMES = "centroids_and_volumes"
MATCHING_POLICY_HEMI_DEFAULT = "default"
MATCHING_POLICY_HEMI_FORCE_LEFT = "force_left"
DEFAULT_ASSIGNMENT_TABLE_VARIANTS = (
    ("centroid", MATCHING_POLICY_CENTROIDS, MATCHING_POLICY_HEMI_DEFAULT),
    ("centroid_force_left", MATCHING_POLICY_CENTROIDS, MATCHING_POLICY_HEMI_FORCE_LEFT),
    ("centroid_volume", MATCHING_POLICY_CENTROIDS_AND_VOLUMES, MATCHING_POLICY_HEMI_DEFAULT),
    (
        "centroid_volume_force_left",
        MATCHING_POLICY_CENTROIDS_AND_VOLUMES,
        MATCHING_POLICY_HEMI_FORCE_LEFT,
    ),
)
PARCEL_ASSIGNMENT_ROW_COLS = [
    "subject",
    "original_gtex_tissue",
    "normalized_gtex_tissue",
    "coordinate_mapped_parcel",
    "coordinate_mapping_distance",
    "final_mapped_parcel",
    "parcel_idx",
    "mapping_distance",
    "matching_rule",
    "matching_rule_detail",
]
PARCEL_ASSIGNMENT_MAPPING_COLS = [
    "matching_policy",
    "matching_policy_hemi_mode",
    "original_gtex_tissue",
    "normalized_gtex_tissue",
    "final_mapped_parcel",
    "parcel_idx",
    "matching_rule",
    "n_input_rows",
    "n_policy_rows",
    "n_dropped_rows",
    "n_subjects",
    "n_final_parcels_for_source",
    "coordinate_mapped_parcels",
]
EXPECTED_CORTICAL_BA_MATCHES = {
    "brain - frontal cortex (ba9)": "LH_SalVentAttn_PFCl_1",
    "brain - anterior cingulate cortex (ba24)": "LH_SalVentAttn_Med_1",
    "brain - cortex": "RH_Cont_PFCl_1",
}
CEREBELLAR_REGION7_PARCEL = "Cerebellar_Region7"
CEREBELLAR_HEMISPHERE_PARCEL = "Cerebellar_Region4"
CEREBELLUM_PARCEL = "Cerebellar_Region7"
CEREBELLAR_NORMALIZED_LABEL = "Cerebellar hemisphere"


@dataclass(frozen=True)
class AtlasOverlapPaths:
    atlas_info_dir: Path = Path("data/metadata/atlas_info")
    schaefer_nifti: str = "atlas-4S156Parcels_space-MNI152NLin6Asym_dseg.nii.gz"
    schaefer_labels: str = "atlas-4S156Parcels_dseg_reformatted.csv"
    brodmann_nifti: str = "overlay_brodmann_2mm_MNI.nii"
    brodmann_labels: str = "overlay_brodmann_2mm_MNI_reformatted.csv"

    def resolve(self, repo_root: str | Path | None = None) -> dict[str, Path]:
        base = _resolve_atlas_info_dir(self.atlas_info_dir, repo_root=repo_root)
        return {
            "schaefer_nifti": base / self.schaefer_nifti,
            "schaefer_labels": base / self.schaefer_labels,
            "brodmann_nifti": base / self.brodmann_nifti,
            "brodmann_labels": base / self.brodmann_labels,
        }


def _resolve_atlas_info_dir(atlas_info_dir: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Resolve atlas metadata relative to either an explicit repo root or the installed source tree."""
    atlas_dir = Path(atlas_info_dir)
    if atlas_dir.is_absolute():
        return atlas_dir

    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / atlas_dir)

    cwd = Path.cwd().resolve()
    candidates.extend(parent / atlas_dir for parent in (cwd, *cwd.parents))

    module_path = Path(__file__).resolve()
    candidates.extend(parent / atlas_dir for parent in module_path.parents)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: legacy layout used data/raw/atlas_info; current layout is data/metadata/atlas_info.
    swapped: list[Path] = []
    for c in candidates:
        s = str(c)
        if "/data/raw/atlas_info" in s:
            swapped.append(Path(s.replace("/data/raw/atlas_info", "/data/metadata/atlas_info")))
        elif "/data/metadata/atlas_info" in s:
            swapped.append(Path(s.replace("/data/metadata/atlas_info", "/data/raw/atlas_info")))
    for candidate in swapped:
        if candidate.exists():
            return candidate
    return Path(repo_root) / atlas_dir if repo_root is not None else atlas_dir


@dataclass(frozen=True)
class BAPrior:
    source_gtex_tissue: str
    source_description: str
    ba_id: int
    ba_label: str
    preferred_target_hemisphere: str | None = None
    source_hemi_mode: str = "native"

    def to_row(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_CORTICAL_BA_PRIORS: tuple[BAPrior, ...] = (
    BAPrior(
        source_gtex_tissue="brain - frontal cortex (ba9)",
        source_description="frontal cortex",
        ba_id=9,
        ba_label="BA9",
    ),
    BAPrior(
        source_gtex_tissue="brain - anterior cingulate cortex (ba24)",
        source_description="anterior cingulate cortex",
        ba_id=24,
        ba_label="BA24",
    ),
    BAPrior(
        source_gtex_tissue="brain - cortex",
        source_description="right frontal pole / generic cortex",
        ba_id=110,
        ba_label="BA110",
        preferred_target_hemisphere="R",
    ),
)

FORCE_LEFT_CORTICAL_BA_PRIORS: tuple[BAPrior, ...] = (
    BAPrior(
        source_gtex_tissue="brain - frontal cortex (ba9)",
        source_description="frontal cortex",
        ba_id=9,
        ba_label="BA9",
        preferred_target_hemisphere="L",
    ),
    BAPrior(
        source_gtex_tissue="brain - anterior cingulate cortex (ba24)",
        source_description="anterior cingulate cortex",
        ba_id=24,
        ba_label="BA24",
        preferred_target_hemisphere="L",
    ),
    BAPrior(
        source_gtex_tissue="brain - cortex",
        source_description="generic cortex, forced left BA10",
        ba_id=10,
        ba_label="BA10",
        preferred_target_hemisphere="L",
    ),
)


def _require_nifti_stack():
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise ImportError("Brodmann/Schaefer overlap matching requires nibabel.") from exc
    return nib, resample_from_to


def _require_array_stack():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Parcel overlap matching requires numpy and pandas.") from exc
    return np, pd


def _corr_safe(x, y, *, method: str = "pearson") -> float:
    np, pd = _require_array_stack()
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    if int(mask.sum()) < 2:
        return float("nan")
    xs = xs.loc[mask]
    ys = ys.loc[mask]
    if float(xs.std(ddof=0)) < 1e-12 or float(ys.std(ddof=0)) < 1e-12:
        return float("nan")
    return float(xs.corr(ys, method=method))


def _validate_expression_agg(agg: str, *, name: str) -> str:
    agg_l = str(agg).strip().lower()
    if agg_l not in {"mean", "median"}:
        raise ValueError(f"{name} must be one of: mean, median")
    return agg_l


def _group_expression(df, group_col: str, genes: Sequence[str], *, agg: str):
    agg_l = _validate_expression_agg(agg, name="expression aggregation")
    grouped = df.groupby(group_col)[list(genes)]
    if agg_l == "mean":
        return grouped.mean(numeric_only=True)
    return grouped.median(numeric_only=True)


def _coerce_int_labels(arr):
    np, _ = _require_array_stack()
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.int32)
    out[finite] = np.rint(arr[finite]).astype(np.int32)
    return out


def _read_label_table(path: str | Path):
    _, pd = _require_array_stack()
    labels = pd.read_csv(path)
    labels.columns = [str(c).strip() for c in labels.columns]
    return labels


def _label_value_column(labels, candidates: Sequence[str] = ("id", "index", "value", "label_id")) -> str:
    for col in candidates:
        if col in labels.columns:
            return col
    raise ValueError(f"No integer label column found in {labels.columns.tolist()}")


def load_ba_schaefer_volumes(
    paths: AtlasOverlapPaths | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Load Brodmann and Schaefer label volumes on a common Schaefer grid."""
    np, _ = _require_array_stack()
    nib, resample_from_to = _require_nifti_stack()
    resolved = (paths or AtlasOverlapPaths()).resolve(repo_root=repo_root)

    schaefer_img = _as_3d_label_img(nib.load(str(resolved["schaefer_nifti"])), nib=nib)
    ba_img = _as_3d_label_img(nib.load(str(resolved["brodmann_nifti"])), nib=nib)
    same_grid = schaefer_img.shape == ba_img.shape and np.allclose(schaefer_img.affine, ba_img.affine)
    if not same_grid:
        ba_img = resample_from_to(ba_img, schaefer_img, order=0)

    return {
        "schaefer_img": schaefer_img,
        "ba_img": ba_img,
        "schaefer_data": _coerce_int_labels(schaefer_img.get_fdata()),
        "ba_data": _coerce_int_labels(ba_img.get_fdata()),
        "schaefer_labels": _read_label_table(resolved["schaefer_labels"]),
        "ba_labels": _read_label_table(resolved["brodmann_labels"]),
        "same_grid_originally": bool(same_grid),
        "paths": resolved,
    }


def _as_3d_label_img(img, *, nib):
    """Return a 3D NIfTI image, squeezing singleton trailing dimensions from label maps."""
    shape = tuple(img.shape)
    if len(shape) == 3:
        return img
    if len(shape) > 3 and all(int(s) == 1 for s in shape[3:]):
        data = img.get_fdata().reshape(shape[:3])
        return nib.Nifti1Image(data, img.affine)
    raise ValueError(f"Expected a 3D label image or singleton-expanded 3D label image, got shape={shape}")


def _prior_rows(priors: Iterable[BAPrior | Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prior in priors:
        rows.append(prior.to_row() if isinstance(prior, BAPrior) else dict(prior))
    return rows


def compute_ba_schaefer_overlap_table(
    priors: Iterable[BAPrior | Mapping[str, object]] = DEFAULT_CORTICAL_BA_PRIORS,
    top_k: int = 12,
    paths: AtlasOverlapPaths | None = None,
    repo_root: str | Path | None = None,
    relax_preferred_hemisphere_if_empty: bool = True,
):
    """Rank Schaefer parcels by voxel overlap with BA masks for the provided priors."""
    np, pd = _require_array_stack()
    payload = load_ba_schaefer_volumes(paths=paths, repo_root=repo_root)
    schaefer_data = payload["schaefer_data"]
    ba_data = payload["ba_data"]
    ba_img = payload["ba_img"]
    schaefer_labels = payload["schaefer_labels"].copy()
    schaefer_value_col = _label_value_column(schaefer_labels)
    schaefer_labels[schaefer_value_col] = pd.to_numeric(
        schaefer_labels[schaefer_value_col],
        errors="coerce",
    ).astype("Int64")

    rows: list[dict[str, object]] = []
    for prior in _prior_rows(priors):
        ba_id = int(prior["ba_id"])
        ba_mask = ba_data == ba_id
        ba_mask = transform_mask_x_hemi(ba_mask, ba_img, mode=str(prior.get("source_hemi_mode", "native")))
        source_voxels = int(ba_mask.sum())
        if source_voxels == 0:
            rows.append({**prior, "rank": 1, "schaefer_id": np.nan, "schaefer_label": None, "overlap_voxels": 0})
            continue

        candidates = schaefer_labels.copy()
        preferred_hemi = prior.get("preferred_target_hemisphere")
        if preferred_hemi and "hemisphere" in candidates.columns:
            candidates = candidates[candidates["hemisphere"].astype(str).str.upper() == str(preferred_hemi).upper()].copy()

        prior_rows = _overlap_rows_for_prior(
            prior,
            ba_mask,
            source_voxels,
            candidates,
            schaefer_value_col,
            schaefer_data,
            same_grid_originally=bool(payload["same_grid_originally"]),
            preferred_hemisphere_relaxed=False,
        )
        if not prior_rows and preferred_hemi and bool(relax_preferred_hemisphere_if_empty):
            prior_rows = _overlap_rows_for_prior(
                prior,
                ba_mask,
                source_voxels,
                schaefer_labels.copy(),
                schaefer_value_col,
                schaefer_data,
                same_grid_originally=bool(payload["same_grid_originally"]),
                preferred_hemisphere_relaxed=True,
            )
        rows.extend(prior_rows)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values(
        ["source_gtex_tissue", "overlap_voxels", "frac_of_ba", "dice", "schaefer_label"],
        ascending=[True, False, False, False, True],
    ).copy()
    out["rank"] = out.groupby("source_gtex_tissue").cumcount() + 1
    return out[out["rank"] <= int(top_k)].reset_index(drop=True)


def _overlap_rows_for_prior(
    prior: Mapping[str, object],
    ba_mask,
    source_voxels: int,
    candidates,
    schaefer_value_col: str,
    schaefer_data,
    *,
    same_grid_originally: bool,
    preferred_hemisphere_relaxed: bool,
) -> list[dict[str, object]]:
    np, pd = _require_array_stack()
    rows: list[dict[str, object]] = []
    for _, label_row in candidates.iterrows():
        if pd.isna(label_row[schaefer_value_col]):
            continue
        schaefer_id = int(label_row[schaefer_value_col])
        if schaefer_id <= 0:
            continue
        sch_mask = schaefer_data == schaefer_id
        schaefer_voxels = int(sch_mask.sum())
        if schaefer_voxels == 0:
            continue
        overlap = int(np.logical_and(ba_mask, sch_mask).sum())
        if overlap == 0:
            continue
        rows.append(
            {
                **prior,
                "schaefer_id": schaefer_id,
                "schaefer_label": str(label_row.get("label", schaefer_id)),
                "schaefer_7network": str(label_row.get("label_7network", "")),
                "schaefer_17network": str(label_row.get("label_17network", "")),
                "network_label": str(label_row.get("network_label", "")),
                "hemisphere": str(label_row.get("hemisphere", "")),
                "source_voxels": source_voxels,
                "schaefer_voxels": schaefer_voxels,
                "overlap_voxels": overlap,
                "frac_of_ba": overlap / float(source_voxels),
                "frac_of_schaefer": overlap / float(schaefer_voxels),
                "dice": 2.0 * overlap / float(source_voxels + schaefer_voxels),
                "same_grid_originally": bool(same_grid_originally),
                "preferred_hemisphere_relaxed": bool(preferred_hemisphere_relaxed),
            }
        )
    return rows


def transform_mask_x_hemi(mask, img, mode: str = "native"):
    """Mirror a binary mask across MNI x=0 when a BA source prior is hemisphere-specific."""
    np, _ = _require_array_stack()
    mode_l = str(mode).strip().lower()
    mask = np.asarray(mask, dtype=bool)
    if mode_l == "native":
        return mask
    if mode_l not in {"mirror_left", "mirror_right"}:
        raise ValueError("source_hemi_mode must be one of: native, mirror_left, mirror_right")

    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        return mask.copy()
    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    xyz = (img.affine @ np.c_[ijk.astype(np.float64), ones].T).T[:, :3]
    if mode_l == "mirror_left":
        xyz[:, 0] = -np.abs(xyz[:, 0])
    else:
        xyz[:, 0] = np.abs(xyz[:, 0])

    inv_affine = np.linalg.inv(img.affine)
    mirrored_ijk = (inv_affine @ np.c_[xyz, ones].T).T[:, :3]
    mirrored_ijk = np.rint(mirrored_ijk).astype(int)
    valid = np.ones(mirrored_ijk.shape[0], dtype=bool)
    for axis, size in enumerate(mask.shape):
        valid &= (mirrored_ijk[:, axis] >= 0) & (mirrored_ijk[:, axis] < int(size))
    out = np.zeros_like(mask, dtype=bool)
    if valid.any():
        idx = mirrored_ijk[valid]
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def top_overlap_overrides(overlap_df):
    """Return one parcel override per GTEx tissue from a ranked overlap table."""
    _, pd = _require_array_stack()
    if len(overlap_df) == 0:
        return pd.DataFrame(
            columns=[
                "source_gtex_tissue",
                "override_schaefer_id",
                "override_mapped_parcel",
                "override_reason",
            ]
        )
    top = overlap_df.sort_values(["source_gtex_tissue", "rank"]).groupby("source_gtex_tissue", as_index=False).first()
    return top.assign(
        override_schaefer_id=top["schaefer_id"].astype("Int64"),
        override_mapped_parcel=top["schaefer_label"].astype(str),
        override_reason="brodmann_schaefer_voxel_overlap",
    )[
        [
            "source_gtex_tissue",
            "override_schaefer_id",
            "override_mapped_parcel",
            "override_reason",
            "ba_id",
            "ba_label",
            "preferred_target_hemisphere",
            "source_hemi_mode",
            "overlap_voxels",
            "frac_of_ba",
            "frac_of_schaefer",
            "dice",
            "preferred_hemisphere_relaxed",
        ]
    ]


def apply_tissue_parcel_overrides(
    gtex_df,
    target_parcels,
    overrides,
    *,
    tissue_col: str = "tissue_or_parcel",
    target_label_col: str = "tissue_or_parcel",
):
    """Apply explicit tissue-to-parcel overrides to an already coordinate-mapped GTEx dataframe."""
    _, pd = _require_array_stack()
    if len(overrides) == 0:
        return gtex_df.copy()

    target_lookup = target_parcels[[target_label_col, "parcel_idx"]].copy()
    target_lookup[target_label_col] = target_lookup[target_label_col].astype(str)
    override_df = pd.DataFrame(overrides).copy()
    if "override_mapped_parcel" not in override_df.columns:
        raise ValueError("overrides must include override_mapped_parcel")
    if "source_gtex_tissue" not in override_df.columns:
        raise ValueError("overrides must include source_gtex_tissue")

    override_df["override_mapped_parcel"] = override_df["override_mapped_parcel"].astype(str)
    override_df = override_df.merge(
        target_lookup.rename(columns={target_label_col: "override_mapped_parcel", "parcel_idx": "override_parcel_idx"}),
        on="override_mapped_parcel",
        how="left",
    )
    missing = override_df[override_df["override_parcel_idx"].isna()]
    if len(missing):
        labels = missing["override_mapped_parcel"].astype(str).tolist()
        raise ValueError(f"Override parcel labels not found in target_parcels: {labels}")

    out = gtex_df.copy()
    out["_tissue_key"] = out[tissue_col].astype(str)
    lk_idx = dict(zip(override_df["source_gtex_tissue"].astype(str), override_df["override_parcel_idx"].astype(int)))
    lk_label = dict(zip(override_df["source_gtex_tissue"].astype(str), override_df["override_mapped_parcel"].astype(str)))
    if "override_reason" not in override_df.columns:
        override_df["override_reason"] = "manual_override"
    lk_reason = dict(zip(override_df["source_gtex_tissue"].astype(str), override_df["override_reason"].astype(str)))
    mask = out["_tissue_key"].isin(lk_idx)
    if mask.any():
        out.loc[mask, "coordinate_parcel_idx"] = out.loc[mask, "parcel_idx"]
        out.loc[mask, "coordinate_mapped_parcel"] = out.loc[mask, "mapped_parcel"]
        if "mapping_distance" in out.columns:
            out.loc[mask, "coordinate_mapping_distance"] = out.loc[mask, "mapping_distance"]
        out.loc[mask, "parcel_idx"] = out.loc[mask, "_tissue_key"].map(lk_idx).astype("int32")
        out.loc[mask, "mapped_parcel"] = out.loc[mask, "_tissue_key"].map(lk_label)
        out.loc[mask, "mapping_override_reason"] = out.loc[mask, "_tissue_key"].map(lk_reason)
    return out.drop(columns=["_tissue_key"])


def apply_gtex_ahba_matching_policy(
    gtex_df,
    target_parcels,
    *,
    matching_policy: str = MATCHING_POLICY_CENTROIDS,
    matching_policy_hemi_mode: str = MATCHING_POLICY_HEMI_DEFAULT,
    collapse_cerebellum: bool = False,
    atlas_paths: AtlasOverlapPaths | None = None,
    validate_expected: bool = True,
    expected_cortical_matches: Mapping[str, str] | None = EXPECTED_CORTICAL_BA_MATCHES,
    repo_root: str | Path | None = None,
):
    """Apply the GTEx-to-AHBA parcel matching policy after default coordinate matching.

    `gtex_df` must already contain the default nearest-centroid assignment columns
    `parcel_idx`, `mapped_parcel`, and optionally `mapping_distance`.
    """
    np, pd = _require_array_stack()
    policy = str(matching_policy).strip().lower()
    if policy not in {MATCHING_POLICY_CENTROIDS, MATCHING_POLICY_CENTROIDS_AND_VOLUMES}:
        raise ValueError(
            "matching_policy must be one of: "
            f"{MATCHING_POLICY_CENTROIDS!r}, {MATCHING_POLICY_CENTROIDS_AND_VOLUMES!r}"
        )
    hemi_mode = str(matching_policy_hemi_mode).strip().lower()
    if hemi_mode not in {MATCHING_POLICY_HEMI_DEFAULT, MATCHING_POLICY_HEMI_FORCE_LEFT}:
        raise ValueError(
            "matching_policy_hemi_mode must be one of: "
            f"{MATCHING_POLICY_HEMI_DEFAULT!r}, {MATCHING_POLICY_HEMI_FORCE_LEFT!r}"
        )

    base = gtex_df
    if hemi_mode == MATCHING_POLICY_HEMI_FORCE_LEFT:
        base = _apply_force_left_coordinate_match(base, target_parcels)

    out = _initialize_matching_audit_columns(base)
    if policy == MATCHING_POLICY_CENTROIDS:
        if bool(collapse_cerebellum):
            out = _apply_cerebellar_policy(out, target_parcels, collapse_cerebellum=True)
        return out

    priors = (
        FORCE_LEFT_CORTICAL_BA_PRIORS
        if hemi_mode == MATCHING_POLICY_HEMI_FORCE_LEFT
        else DEFAULT_CORTICAL_BA_PRIORS
    )
    overlap_df = compute_ba_schaefer_overlap_table(priors=priors, paths=atlas_paths, repo_root=repo_root)
    cortical_overrides = top_overlap_overrides(overlap_df)
    if bool(validate_expected) and hemi_mode == MATCHING_POLICY_HEMI_DEFAULT:
        _validate_cortical_overrides(cortical_overrides, expected_cortical_matches or {})
    out = _apply_cortical_overlap_policy(out, target_parcels, cortical_overrides)
    out = _apply_cerebellar_policy(out, target_parcels, collapse_cerebellum=bool(collapse_cerebellum))
    return out.reset_index(drop=True)


def build_gtex_ahba_assignment_outputs(
    samples_df,
    *,
    matching_policy: str = MATCHING_POLICY_CENTROIDS,
    matching_policy_hemi_mode: str = MATCHING_POLICY_HEMI_DEFAULT,
    collapse_cerebellum: bool = False,
    rep_mode: str = "centroid",
    hemi_mode: str = "mirror_left",
    atlas_paths: AtlasOverlapPaths | None = None,
    validate_expected: bool = True,
    repo_root: str | Path | None = None,
):
    """Build detailed and collapsed GTEx-to-AHBA assignment tables for one policy variant."""
    np, _ = _require_array_stack()
    from src.io import parse_coordinate_representative
    from src.preprocess import build_target_parcels, map_gtex_to_target

    meta = samples_df[META_COLS].copy()
    xyz = np.vstack(
        [
            parse_coordinate_representative(coord_text, rep_mode=rep_mode, hemi_mode=hemi_mode)
            for coord_text in meta["coordinates"]
        ]
    )
    spatial_df = meta.assign(
        coord_x=xyz[:, 0],
        coord_y=xyz[:, 1],
        coord_z=xyz[:, 2],
        coord_abs_x=np.abs(xyz[:, 0]),
        gtex_rep_mode=str(rep_mode).strip().lower(),
        gtex_hemi_mode=str(hemi_mode).strip().lower(),
    )
    spatial_df = spatial_df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    spatial_df["dataset_upper"] = spatial_df["dataset"].astype(str).str.upper().str.strip()
    spatial_df["subject"] = spatial_df["subject"].astype(str)
    spatial_df["tissue_or_parcel"] = spatial_df["tissue_or_parcel"].astype(str)

    ahba_raw = spatial_df[spatial_df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = spatial_df[spatial_df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    target_parcels = build_target_parcels(ahba_raw)
    gtex_mapped = map_gtex_to_target(gtex_raw, target_parcels)
    input_counts = (
        gtex_mapped.groupby("tissue_or_parcel", as_index=False)
        .agg(n_input_rows=("subject", "size"))
        .rename(columns={"tissue_or_parcel": "original_gtex_tissue"})
    )
    matched = apply_gtex_ahba_matching_policy(
        gtex_mapped,
        target_parcels,
        matching_policy=matching_policy,
        matching_policy_hemi_mode=matching_policy_hemi_mode,
        collapse_cerebellum=bool(collapse_cerebellum),
        atlas_paths=atlas_paths,
        validate_expected=validate_expected,
        repo_root=repo_root,
    )
    row_table = matched[PARCEL_ASSIGNMENT_ROW_COLS].copy()
    row_table = row_table.sort_values(["subject", "original_gtex_tissue", "final_mapped_parcel"]).reset_index(drop=True)
    row_table.insert(0, "matching_policy", str(matching_policy).strip().lower())
    row_table.insert(1, "matching_policy_hemi_mode", str(matching_policy_hemi_mode).strip().lower())
    mapping_table = collapse_assignment_rows_to_mapping(row_table, input_counts)
    return row_table, mapping_table


def build_gtex_ahba_assignment_variant_tables(
    samples_df,
    *,
    variants=DEFAULT_ASSIGNMENT_TABLE_VARIANTS,
    collapse_cerebellum: bool = False,
    rep_mode: str = "centroid",
    hemi_mode: str = "mirror_left",
    atlas_paths: AtlasOverlapPaths | None = None,
    validate_expected: bool = True,
    repo_root: str | Path | None = None,
):
    """Build row and one-to-one mapping tables for named parcel-matching variants."""
    outputs = {}
    for name, matching_policy, matching_policy_hemi_mode in variants:
        row_table, mapping_table = build_gtex_ahba_assignment_outputs(
            samples_df,
            matching_policy=matching_policy,
            matching_policy_hemi_mode=matching_policy_hemi_mode,
            collapse_cerebellum=bool(collapse_cerebellum),
            rep_mode=rep_mode,
            hemi_mode=hemi_mode,
            atlas_paths=atlas_paths,
            validate_expected=validate_expected,
            repo_root=repo_root,
        )
        outputs[str(name)] = {"rows": row_table, "mapping": mapping_table}
    return outputs


def collapse_assignment_rows_to_mapping(row_table, input_counts):
    """Collapse per-row assignment output into GTEx source-tissue to final-parcel mappings."""
    _, pd = _require_array_stack()
    grouped = (
        row_table.groupby(
            [
                "matching_policy",
                "matching_policy_hemi_mode",
                "original_gtex_tissue",
                "normalized_gtex_tissue",
                "final_mapped_parcel",
                "parcel_idx",
                "matching_rule",
            ],
            as_index=False,
        ).agg(
            n_policy_rows=("subject", "size"),
            n_subjects=("subject", "nunique"),
            coordinate_mapped_parcels=("coordinate_mapped_parcel", _join_unique_strings),
            matching_rule_detail=("matching_rule_detail", "first"),
        )
    )
    mapping = grouped.merge(input_counts, on="original_gtex_tissue", how="left")
    mapping["n_input_rows"] = mapping["n_input_rows"].fillna(0).astype(int)
    mapping["n_dropped_rows"] = mapping["n_input_rows"] - mapping["n_policy_rows"]
    mapping["n_final_parcels_for_source"] = (
        mapping.groupby("original_gtex_tissue")["final_mapped_parcel"].transform("nunique").astype(int)
    )
    mapping = mapping.sort_values(["original_gtex_tissue", "final_mapped_parcel"]).reset_index(drop=True)
    return mapping[PARCEL_ASSIGNMENT_MAPPING_COLS + ["matching_rule_detail"]]


def _join_unique_strings(values):
    _, pd = _require_array_stack()
    vals = sorted({str(v) for v in values if pd.notna(v)})
    return " | ".join(vals)


def _apply_force_left_coordinate_match(gtex_df, target_parcels):
    np, _ = _require_array_stack()
    required = {"coord_x", "coord_y", "coord_z"}
    if not required.issubset(gtex_df.columns):
        missing = sorted(required - set(gtex_df.columns))
        raise ValueError(f"force_left matching requires GTEx coordinate columns, missing: {missing}")

    target = target_parcels.copy()
    labels = target["tissue_or_parcel"].astype(str)
    left_mask = labels.str.startswith("LH") | labels.str.startswith("Cerebellar_")
    left_target = target.loc[left_mask].copy()
    if left_target.empty:
        raise ValueError("force_left matching found no LH-prefixed target parcels.")

    txyz = left_target[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=float)
    gxyz = gtex_df[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=float)
    d2 = ((gxyz[:, None, :] - txyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)

    out = gtex_df.copy()
    out["parcel_idx"] = left_target.iloc[idx]["parcel_idx"].to_numpy(dtype=np.int32)
    out["mapped_parcel"] = left_target.iloc[idx]["tissue_or_parcel"].to_numpy()
    out["mapping_distance"] = np.sqrt(d2[np.arange(len(out)), idx]).astype(np.float32)
    out["matching_rule"] = "coordinate_nearest_left_centroid"
    out["matching_rule_detail"] = "nearest-neighbor assignment constrained to LH-prefixed target parcels"
    return out


def _initialize_matching_audit_columns(gtex_df):
    out = gtex_df.copy()
    if "parcel_idx" not in out.columns or "mapped_parcel" not in out.columns:
        raise ValueError("gtex_df must contain coordinate-matched parcel_idx and mapped_parcel columns")
    out["original_gtex_tissue"] = out.get("original_gtex_tissue", out["tissue_or_parcel"]).astype(str)
    out["normalized_gtex_tissue"] = out.get("normalized_gtex_tissue", out["tissue_or_parcel"]).astype(str)
    out["coordinate_parcel_idx"] = out.get("coordinate_parcel_idx", out["parcel_idx"]).astype("int32")
    out["coordinate_mapped_parcel"] = out.get("coordinate_mapped_parcel", out["mapped_parcel"]).astype(str)
    if "mapping_distance" in out.columns:
        out["coordinate_mapping_distance"] = out.get("coordinate_mapping_distance", out["mapping_distance"])
    elif "coordinate_mapping_distance" not in out.columns:
        out["coordinate_mapping_distance"] = float("nan")
    out["final_mapped_parcel"] = out["mapped_parcel"].astype(str)
    out["matching_rule"] = out.get("matching_rule", "coordinate_nearest_centroid")
    out["matching_rule_detail"] = out.get(
        "matching_rule_detail",
        "default representative-coordinate nearest-neighbor assignment",
    )
    return out


def _validate_cortical_overrides(overrides, expected: Mapping[str, str]) -> None:
    if not expected:
        return
    observed = dict(zip(overrides["source_gtex_tissue"].astype(str), overrides["override_mapped_parcel"].astype(str)))
    mismatches = []
    for tissue, expected_parcel in expected.items():
        observed_parcel = observed.get(str(tissue))
        if observed_parcel != str(expected_parcel):
            mismatches.append((str(tissue), str(expected_parcel), observed_parcel))
    if mismatches:
        msg = "; ".join(
            f"{tissue}: expected {expected_parcel!r}, observed {observed_parcel!r}"
            for tissue, expected_parcel, observed_parcel in mismatches
        )
        raise RuntimeError(f"Cortical BA/Schaefer overlap overrides differ from validated defaults: {msg}")


def _target_label_lookup(target_parcels, *, label_col: str = "tissue_or_parcel") -> dict[str, int]:
    lookup = target_parcels[[label_col, "parcel_idx"]].copy()
    lookup[label_col] = lookup[label_col].astype(str)
    return dict(zip(lookup[label_col], lookup["parcel_idx"].astype(int)))


def _apply_cortical_overlap_policy(gtex_df, target_parcels, cortical_overrides):
    np, _ = _require_array_stack()
    out = gtex_df.copy()
    target_lookup = _target_label_lookup(target_parcels)
    for _, row in cortical_overrides.iterrows():
        tissue = str(row["source_gtex_tissue"])
        parcel = str(row["override_mapped_parcel"])
        mask = out["original_gtex_tissue"].astype(str) == tissue
        if not mask.any():
            continue
        if parcel not in target_lookup:
            raise ValueError(f"Cortical override parcel {parcel!r} is not present in target_parcels")
        out.loc[mask, "parcel_idx"] = np.int32(target_lookup[parcel])
        out.loc[mask, "mapped_parcel"] = parcel
        out.loc[mask, "final_mapped_parcel"] = parcel
        if "mapping_distance" in out.columns:
            out.loc[mask, "mapping_distance"] = np.nan
        out.loc[mask, "matching_rule"] = "cortical_ba_schaefer_voxel_overlap"
        out.loc[mask, "matching_rule_detail"] = (
            "computed max voxel overlap "
            f"{row['ba_label']} -> {parcel}; overlap_voxels={int(row['overlap_voxels'])}; "
            f"dice={float(row['dice']):.4f}"
        )
    out["parcel_idx"] = out["parcel_idx"].astype("int32")
    return out


def _apply_cerebellar_policy(gtex_df, target_parcels, *, collapse_cerebellum: bool = False):
    np, _ = _require_array_stack()
    out = gtex_df.copy()
    target_lookup = _target_label_lookup(target_parcels)

    tissue = out["original_gtex_tissue"].astype(str)
    if bool(collapse_cerebellum):
        has_hemi = (
            out[tissue == "brain - cerebellar hemisphere"]
            .groupby("subject")
            .size()
            .index.astype(str)
        )
        drop_mask = tissue.eq("brain - cerebellum") & out["subject"].astype(str).isin(set(has_hemi))
        out = out.loc[~drop_mask].copy()

        tissue = out["original_gtex_tissue"].astype(str)
        keep_mask = tissue.isin(DEFAULT_CEREBELLAR_GTEX_TISSUES)
        if not keep_mask.any():
            out["parcel_idx"] = out["parcel_idx"].astype("int32")
            return out
        if CEREBELLAR_REGION7_PARCEL not in target_lookup:
            raise ValueError(f"{CEREBELLAR_REGION7_PARCEL!r} is not present in target_parcels")
        parcel_idx = np.int32(target_lookup[CEREBELLAR_REGION7_PARCEL])
        out.loc[keep_mask, "parcel_idx"] = parcel_idx
        out.loc[keep_mask, "mapped_parcel"] = CEREBELLAR_REGION7_PARCEL
        out.loc[keep_mask, "final_mapped_parcel"] = CEREBELLAR_REGION7_PARCEL
        out.loc[keep_mask, "tissue_or_parcel"] = CEREBELLAR_NORMALIZED_LABEL
        out.loc[keep_mask, "normalized_gtex_tissue"] = CEREBELLAR_NORMALIZED_LABEL
        if "mapping_distance" in out.columns:
            out.loc[keep_mask, "mapping_distance"] = np.nan
        out.loc[keep_mask, "matching_rule"] = "cerebellar_region7_expression_policy"
        out.loc[keep_mask, "matching_rule_detail"] = (
            "prefer brain - cerebellar hemisphere; fallback brain - cerebellum; "
            f"assign {CEREBELLAR_REGION7_PARCEL}"
        )
    else:
        assignments = {
            "brain - cerebellar hemisphere": CEREBELLAR_HEMISPHERE_PARCEL,
            "brain - cerebellum": CEREBELLUM_PARCEL,
        }
        for source_tissue, parcel in assignments.items():
            mask = tissue.eq(source_tissue)
            if not mask.any():
                continue
            if parcel not in target_lookup:
                raise ValueError(f"Cerebellar override parcel {parcel!r} is not present in target_parcels")
            out.loc[mask, "parcel_idx"] = np.int32(target_lookup[parcel])
            out.loc[mask, "mapped_parcel"] = parcel
            out.loc[mask, "final_mapped_parcel"] = parcel
            if "mapping_distance" in out.columns:
                out.loc[mask, "mapping_distance"] = np.nan
            out.loc[mask, "matching_rule"] = "cerebellar_manual_region_policy"
            out.loc[mask, "matching_rule_detail"] = f"manual cerebellar assignment {source_tissue} -> {parcel}"
    out["parcel_idx"] = out["parcel_idx"].astype("int32")
    return out


def _center_voxel(mask):
    np, _ = _require_array_stack()
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        return None
    return np.rint(ijk.mean(axis=0)).astype(int)


def plot_ba_schaefer_overlap_slices(
    overlap_df,
    volume_payload: Mapping[str, object],
    source_gtex_tissue: str,
    *,
    top_n: int = 4,
    alpha: float = 0.62,
    figsize: tuple[float, float] = (18.0, 6.8),
    crop_pad: int = 10,
    crop: bool = True,
    show_context: bool = True,
):
    """Visualize one BA source mask with the top overlapping Schaefer parcels on three slices."""
    np, _ = _require_array_stack()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, to_rgba
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError("BA/Schaefer slice visualization requires matplotlib.") from exc

    if len(overlap_df) == 0:
        raise ValueError("No overlap rows available.")
    sub = overlap_df[overlap_df["source_gtex_tissue"].astype(str) == str(source_gtex_tissue)].sort_values("rank").head(int(top_n))
    if len(sub) == 0:
        raise ValueError(f"No overlap rows for {source_gtex_tissue!r}")

    ba_id = int(sub.iloc[0]["ba_id"])
    ba_data = volume_payload["ba_data"]
    ba_img = volume_payload["ba_img"]
    schaefer_img = volume_payload["schaefer_img"]
    schaefer_data = volume_payload["schaefer_data"]
    ba_mask = ba_data == ba_id
    ba_mask = transform_mask_x_hemi(ba_mask, ba_img, mode=str(sub.iloc[0].get("source_hemi_mode", "native")))
    center = _center_voxel(ba_mask)
    if center is None:
        raise ValueError(f"BA{ba_id} has no voxels in the resampled grid")

    slices = [(0, int(center[0]), "sagittal"), (1, int(center[1]), "coronal"), (2, int(center[2]), "axial")]
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.25, wspace=0.18)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
    ba_cmap = ListedColormap([(1, 1, 1, 0), to_rgba("#111111", 0.26)])
    sub_ranked = sub.reset_index(drop=True)
    legend_handles = [
        Patch(facecolor="#111111", edgecolor="#111111", alpha=0.34, label=f"BA{ba_id} source mask")
    ]
    for i, row in sub_ranked.iterrows():
        legend_handles.append(
            Patch(
                facecolor=colors[i % len(colors)],
                edgecolor=colors[i % len(colors)],
                alpha=alpha,
                label=f"#{int(row['rank'])} {row['schaefer_label']} ({int(row['overlap_voxels'])} vox)",
            )
        )

    for ax, (axis, idx, title) in zip(axes, slices):
        if axis == 0:
            ba_slice = ba_mask[idx, :, :].T
            sch_slice = schaefer_data[idx, :, :].T
        elif axis == 1:
            ba_slice = ba_mask[:, idx, :].T
            sch_slice = schaefer_data[:, idx, :].T
        else:
            ba_slice = ba_mask[:, :, idx].T
            sch_slice = schaefer_data[:, :, idx].T

        target_slice_mask = np.zeros_like(ba_slice, dtype=bool)
        for _, row in sub_ranked.iterrows():
            target_slice_mask |= sch_slice == int(row["schaefer_id"])
        crop_bounds = _slice_crop_bounds(ba_slice | target_slice_mask, pad=int(crop_pad)) if crop else None

        ba_plot = ba_slice[crop_bounds] if crop_bounds is not None else ba_slice
        sch_plot = sch_slice[crop_bounds] if crop_bounds is not None else sch_slice
        extent = _slice_extent_for_display(schaefer_img, axis=axis, crop_bounds=crop_bounds)
        target_plot = target_slice_mask[crop_bounds] if crop_bounds is not None else target_slice_mask
        if show_context:
            context = np.ma.masked_where((sch_plot <= 0) | ba_plot | target_plot, sch_plot)
            ax.imshow(
                context,
                origin="lower",
                cmap=ListedColormap([(0.82, 0.82, 0.82, 0.14)]),
                interpolation="nearest",
                extent=extent,
            )
        ax.imshow(ba_plot, origin="lower", cmap=ba_cmap, interpolation="nearest", extent=extent)
        ax.contour(
            ba_plot.astype(float),
            levels=[0.5],
            colors=["#111111"],
            linewidths=1.4,
            origin="lower",
            extent=extent,
        )

        for i, row in sub_ranked.iterrows():
            schaefer_id = int(row["schaefer_id"])
            overlay = np.ma.masked_where(sch_plot != schaefer_id, sch_plot)
            ax.imshow(
                overlay,
                origin="lower",
                cmap=ListedColormap([colors[i % len(colors)]]),
                alpha=alpha,
                interpolation="nearest",
                extent=extent,
            )
        mni_coord = _slice_mni_coordinate(schaefer_img, axis=axis, idx=idx)
        ax.set_title(f"{title} slice {idx} ({mni_coord})", fontsize=16, weight="bold")
        ax.set_xlabel(_display_axis_labels(axis)[0], fontsize=13)
        ax.set_ylabel(_display_axis_labels(axis)[1], fontsize=13)
        ax.tick_params(labelsize=11)

    fig.suptitle(
        f"{source_gtex_tissue}: BA{ba_id} source mask with top Schaefer voxel-overlap candidates",
        fontsize=16,
        weight="bold",
        y=0.95,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    return fig


def plot_top_ba_schaefer_matches_brainspace(
    overlap_df,
    volume_payload: Mapping[str, object],
    *,
    rank: int = 1,
    hemi_view: str = "native",
    max_points_per_mask: int = 1800,
    draw_match_lines: bool = True,
    match_line_width: float = 5.0,
    random_state: int = 0,
    title: str | None = None,
):
    """Plot the selected BA-to-Schaefer voxel-overlap matches together in MNI 3D space."""
    np, _ = _require_array_stack()
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("BA/Schaefer brain-space visualization requires plotly.") from exc

    if len(overlap_df) == 0:
        raise ValueError("No overlap rows available.")
    selected = (
        overlap_df[overlap_df["rank"].astype(int) == int(rank)]
        .sort_values(["source_gtex_tissue", "rank"])
        .copy()
    )
    if len(selected) == 0:
        raise ValueError(f"No overlap rows found for rank={rank}")
    hemi_view_l = str(hemi_view).strip().lower()
    if hemi_view_l not in {"native", "mirror_left"}:
        raise ValueError("hemi_view must be one of: native, mirror_left")

    ba_data = volume_payload["ba_data"]
    ba_img = volume_payload["ba_img"]
    schaefer_img = volume_payload["schaefer_img"]
    schaefer_data = volume_payload["schaefer_data"]
    rng = np.random.default_rng(int(random_state))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
    fig = go.Figure()

    all_xyz: list[np.ndarray] = []
    for color_idx, (_, row) in enumerate(selected.iterrows()):
        color = colors[color_idx % len(colors)]
        ba_id = int(row["ba_id"])
        schaefer_id = int(row["schaefer_id"])
        ba_mask = ba_data == ba_id
        ba_mask = transform_mask_x_hemi(ba_mask, ba_img, mode=str(row.get("source_hemi_mode", "native")))
        sch_mask = schaefer_data == schaefer_id

        ba_centroid = _transform_xyz_for_brainspace_view(
            _mask_mni_centroid(ba_mask, schaefer_img)[None, :],
            hemi_view=hemi_view_l,
        )[0]
        sch_centroid = _transform_xyz_for_brainspace_view(
            _mask_mni_centroid(sch_mask, schaefer_img)[None, :],
            hemi_view=hemi_view_l,
        )[0]
        all_xyz.extend([ba_centroid[None, :], sch_centroid[None, :]])

        for mask_name, mask, opacity, marker_size, symbol in [
            ("BA source", ba_mask, 0.16, 2.2, "circle"),
            ("selected Schaefer parcel", sch_mask, 0.72, 2.8, "diamond"),
        ]:
            xyz = _mask_mni_points(
                mask,
                schaefer_img,
                max_points=int(max_points_per_mask),
                rng=rng,
            )
            xyz = _transform_xyz_for_brainspace_view(xyz, hemi_view=hemi_view_l)
            if len(xyz) == 0:
                continue
            all_xyz.append(xyz)
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    name=(
                        f"{row['source_gtex_tissue']} | {mask_name}: "
                        f"{row['ba_label'] if mask_name == 'BA source' else row['schaefer_label']}"
                    ),
                    marker={
                        "size": marker_size,
                        "color": color,
                        "opacity": opacity,
                        "symbol": symbol,
                    },
                    customdata=np.repeat(
                        np.array([[
                            str(row["source_gtex_tissue"]),
                            str(row["ba_label"]),
                            str(row["schaefer_label"]),
                            int(row["overlap_voxels"]),
                            float(row["frac_of_ba"]),
                            float(row["dice"]),
                            hemi_view_l,
                        ]], dtype=object),
                        repeats=len(xyz),
                        axis=0,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{mask_name}<br>"
                        "BA=%{customdata[1]}<br>"
                        "Schaefer=%{customdata[2]}<br>"
                        "overlap_voxels=%{customdata[3]}<br>"
                        "frac_of_ba=%{customdata[4]:.3f}<br>"
                        "dice=%{customdata[5]:.3f}<br>"
                        "hemi_view=%{customdata[6]}<br>"
                        "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
                    ),
                )
            )

        if bool(draw_match_lines):
            fig.add_trace(
                go.Scatter3d(
                    x=[ba_centroid[0], sch_centroid[0]],
                    y=[ba_centroid[1], sch_centroid[1]],
                    z=[ba_centroid[2], sch_centroid[2]],
                    mode="lines",
                    name=f"Centroid match outline | {row['ba_label']} -> {row['schaefer_label']}",
                    line={"color": "#111111", "width": float(match_line_width) + 2.0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[ba_centroid[0], sch_centroid[0]],
                    y=[ba_centroid[1], sch_centroid[1]],
                    z=[ba_centroid[2], sch_centroid[2]],
                    mode="lines+markers",
                    name=f"Centroid match | {row['ba_label']} -> {row['schaefer_label']}",
                    line={"color": color, "width": float(match_line_width)},
                    marker={
                        "size": [8.5, 9.5],
                        "color": [color, color],
                        "symbol": ["circle", "diamond"],
                        "opacity": 1.0,
                        "line": {"width": 2.6, "color": "#111111"},
                    },
                    customdata=np.array(
                        [
                            [
                                str(row["source_gtex_tissue"]),
                                str(row["ba_label"]),
                                str(row["schaefer_label"]),
                                "BA centroid",
                                float(row["overlap_voxels"]),
                                float(row["frac_of_ba"]),
                                float(row["dice"]),
                                hemi_view_l,
                            ],
                            [
                                str(row["source_gtex_tissue"]),
                                str(row["ba_label"]),
                                str(row["schaefer_label"]),
                                "Schaefer centroid",
                                float(row["overlap_voxels"]),
                                float(row["frac_of_ba"]),
                                float(row["dice"]),
                                hemi_view_l,
                            ],
                        ],
                        dtype=object,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[3]}<br>"
                        "BA=%{customdata[1]}<br>"
                        "Schaefer=%{customdata[2]}<br>"
                        "overlap_voxels=%{customdata[4]:.0f}<br>"
                        "frac_of_ba=%{customdata[5]:.3f}<br>"
                        "dice=%{customdata[6]:.3f}<br>"
                        "hemi_view=%{customdata[7]}<br>"
                        "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
                    ),
                )
            )

    ranges = _mni_axis_ranges(all_xyz)
    fig.update_layout(
        title=title
        or (
            f"Top BA-to-Schaefer voxel-overlap matches in MNI space (rank {int(rank)})"
            + (" | mirrored to left hemisphere" if hemi_view_l == "mirror_left" else "")
        ),
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "range": list(ranges["x"]), "backgroundcolor": "#f8fafc"},
            "yaxis": {"title": "MNI y", "range": list(ranges["y"]), "backgroundcolor": "#f8fafc"},
            "zaxis": {"title": "MNI z", "range": list(ranges["z"]), "backgroundcolor": "#f8fafc"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": -1.8, "z": 1.05}},
        },
    )
    return fig


def _transform_xyz_for_brainspace_view(xyz, *, hemi_view: str):
    np, _ = _require_array_stack()
    out = np.asarray(xyz, dtype=float).copy()
    if len(out) == 0:
        return out
    view = str(hemi_view).strip().lower()
    if view == "native":
        return out
    if view == "mirror_left":
        out[:, 0] = -np.abs(out[:, 0])
        return out
    raise ValueError("hemi_view must be one of: native, mirror_left")


def _mask_mni_centroid(mask, img):
    np, _ = _require_array_stack()
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    ijk_center = ijk.astype(float).mean(axis=0)
    xyz = img.affine @ np.r_[ijk_center, 1.0]
    return np.asarray(xyz[:3], dtype=float)


def _mask_mni_points(mask, img, *, max_points: int, rng):
    np, _ = _require_array_stack()
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        return np.empty((0, 3), dtype=float)
    if int(max_points) > 0 and len(ijk) > int(max_points):
        keep = rng.choice(len(ijk), size=int(max_points), replace=False)
        ijk = ijk[np.sort(keep)]
    ones = np.ones((ijk.shape[0], 1), dtype=float)
    return (img.affine @ np.c_[ijk.astype(float), ones].T).T[:, :3]


def _mni_axis_ranges(xyz_arrays: Sequence[object]) -> dict[str, tuple[float, float]]:
    np, _ = _require_array_stack()
    if len(xyz_arrays) == 0:
        return {"x": (-90.0, 90.0), "y": (-126.0, 90.0), "z": (-72.0, 108.0)}
    xyz = np.vstack([np.asarray(arr, dtype=float) for arr in xyz_arrays if len(arr)])
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    widths = np.maximum(maxs - mins, 1.0)
    max_width = float(np.max(widths))
    mids = (mins + maxs) / 2.0
    pad = 0.12 * max_width
    half = 0.5 * max_width + pad
    return {
        "x": (float(mids[0] - half), float(mids[0] + half)),
        "y": (float(mids[1] - half), float(mids[1] + half)),
        "z": (float(mids[2] - half), float(mids[2] + half)),
    }


def _slice_crop_bounds(mask_2d, pad: int = 10):
    np, _ = _require_array_stack()
    ij = np.argwhere(mask_2d)
    if len(ij) == 0:
        return None
    lo = np.maximum(ij.min(axis=0) - int(pad), 0)
    hi = np.minimum(ij.max(axis=0) + int(pad) + 1, mask_2d.shape)
    return (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])))


def _axis_world_ranges(img, axis: int, crop_bounds):
    np, _ = _require_array_stack()
    shape = tuple(int(x) for x in img.shape[:3])
    axes_3d = [a for a in range(3) if a != int(axis)]
    if crop_bounds is None:
        starts = [0, 0]
        stops = [shape[axes_3d[0]], shape[axes_3d[1]]]
    else:
        starts = [int(crop_bounds[0].start), int(crop_bounds[1].start)]
        stops = [int(crop_bounds[0].stop), int(crop_bounds[1].stop)]
    coords = []
    for ax3d, start, stop in zip(axes_3d, starts, stops):
        pts = np.zeros((2, 4), dtype=float)
        pts[:, 3] = 1.0
        pts[:, ax3d] = [start, max(start, stop - 1)]
        xyz = (img.affine @ pts.T).T[:, ax3d]
        coords.append((float(xyz[0]), float(xyz[1])))
    return coords


def _slice_extent_for_display(img, axis: int, crop_bounds=None):
    ranges = _axis_world_ranges(img, axis=axis, crop_bounds=crop_bounds)
    # Slices are transposed before display, so x corresponds to the second displayed axis.
    y_range, x_range = ranges[0], ranges[1]
    return [x_range[0], x_range[1], y_range[0], y_range[1]]


def _slice_mni_coordinate(img, axis: int, idx: int) -> str:
    np, _ = _require_array_stack()
    ijk = np.zeros(4, dtype=float)
    ijk[3] = 1.0
    ijk[int(axis)] = float(idx)
    xyz = img.affine @ ijk
    label = ["x", "y", "z"][int(axis)]
    return f"MNI {label}={xyz[int(axis)]:.1f}"


def _display_axis_labels(axis: int) -> tuple[str, str]:
    if int(axis) == 0:
        return "MNI y", "MNI z"
    if int(axis) == 1:
        return "MNI x", "MNI z"
    return "MNI x", "MNI y"


def available_expression_genes(samples_df, meta_cols: Sequence[str] = META_COLS) -> list[str]:
    """Return expression columns from a combined AHBA/GTEx sample table."""
    meta = set(str(c) for c in meta_cols)
    return [str(c) for c in samples_df.columns if str(c) not in meta]


def load_gene_sets_for_matching(
    gene_list_paths: Mapping[str, str | Path] | None,
    available_genes: Sequence[str],
    *,
    include_all_shared: bool = True,
    min_genes: int = 5,
) -> dict[str, list[str]]:
    """Load named gene lists and intersect each with available expression columns."""
    available = [str(g) for g in available_genes]
    available_set = set(available)
    gene_sets: dict[str, list[str]] = {}
    if include_all_shared:
        gene_sets["all_shared_genes"] = list(available)

    for name, path in (gene_list_paths or {}).items():
        p = Path(path)
        if not p.exists():
            continue
        requested = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        genes = [g for g in requested if g in available_set]
        if len(genes) >= int(min_genes):
            gene_sets[str(name)] = genes
    return gene_sets


def default_cerebellar_gene_list_paths(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    # Gene lists migrated from data/raw/gene_lists -> data/metadata/gene_lists.
    # Resolve each name against whichever directory currently has the file.
    def _pick(name: str) -> Path:
        meta = root / "data/metadata/gene_lists" / name
        legacy = root / "data/raw/gene_lists" / name
        if not meta.exists() and legacy.exists():
            return legacy
        return meta
    return {
        "gtex_100hvg": _pick("gtex_100hvg.txt"),
        "gtex_100hvg_demeaned": _pick("gtex_100hvg_demeaned.txt"),
        "ahba_100hvg": _pick("ahba_100hvg.txt"),
        "ahba_250hvg": _pick("ahba_250hvg.txt"),
        "ahba_500hvg": _pick("ahba_500hvg.txt"),
        "gtex_10deg": _pick("gtex_10deg.txt"),
        "gtex_25deg": _pick("gtex_25deg.txt"),
        "richiardi2015": _pick("richiardi2015.txt"),
        "syngo": _pick("syngo.txt"),
    }


def compute_cerebellar_expression_rankings(
    samples_df,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    source_tissues: Sequence[str] = DEFAULT_CEREBELLAR_GTEX_TISSUES,
    cerebellar_prefix: str = "Cerebellar_",
    gtex_agg: str = "median",
    ahba_agg: str = "mean",
):
    """Rank AHBA cerebellar parcels for GTEx cerebellar tissues by expression correlation."""
    np, pd = _require_array_stack()
    rows: list[dict[str, object]] = []
    df = samples_df.copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    ahba = df[df["dataset_upper"] == "AHBA"].copy()
    gtex = df[df["dataset_upper"] == "GTEX"].copy()
    ahba_cereb = ahba[ahba["tissue_or_parcel"].astype(str).str.startswith(str(cerebellar_prefix))].copy()
    if len(ahba_cereb) == 0:
        raise ValueError(f"No AHBA parcels found with prefix {cerebellar_prefix!r}")

    for gene_set_name, genes_in in gene_sets.items():
        genes = [str(g) for g in genes_in if str(g) in df.columns]
        if len(genes) < 2:
            continue
        gtex_expr = _group_expression(
            gtex[gtex["tissue_or_parcel"].isin(source_tissues)],
            "tissue_or_parcel",
            genes,
            agg=gtex_agg,
        )
        ahba_expr = _group_expression(ahba_cereb, "tissue_or_parcel", genes, agg=ahba_agg)
        for source_tissue, source_vec in gtex_expr.iterrows():
            for parcel_name, parcel_vec in ahba_expr.iterrows():
                rows.append(
                    {
                        "gene_set": str(gene_set_name),
                        "source_gtex_tissue": str(source_tissue),
                        "ahba_cerebellar_parcel": str(parcel_name),
                        "gtex_agg": _validate_expression_agg(gtex_agg, name="gtex_agg"),
                        "ahba_agg": _validate_expression_agg(ahba_agg, name="ahba_agg"),
                        "n_genes": int(len(genes)),
                        "pearson_r": _corr_safe(source_vec, parcel_vec, method="pearson"),
                        "spearman_r": _corr_safe(source_vec, parcel_vec, method="spearman"),
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values(
        ["gene_set", "source_gtex_tissue", "pearson_r", "spearman_r", "ahba_cerebellar_parcel"],
        ascending=[True, True, False, False, True],
    ).copy()
    out["rank_pearson"] = out.groupby(["gene_set", "source_gtex_tissue"]).cumcount() + 1

    out = out.sort_values(
        ["gene_set", "source_gtex_tissue", "spearman_r", "pearson_r", "ahba_cerebellar_parcel"],
        ascending=[True, True, False, False, True],
    ).copy()
    out["rank_spearman"] = out.groupby(["gene_set", "source_gtex_tissue"]).cumcount() + 1
    return out.sort_values(["source_gtex_tissue", "gene_set", "rank_pearson"]).reset_index(drop=True)


def compute_expression_parcel_rankings(
    samples_df,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    source_tissues: Sequence[str],
    target_parcels: Sequence[str] | None = None,
    target_prefix: str | None = None,
    target_col_name: str = "ahba_target_parcel",
    gtex_agg: str = "median",
    ahba_agg: str = "mean",
):
    """Rank selected AHBA target parcels for selected GTEx source tissues by expression correlation."""
    np, pd = _require_array_stack()
    rows: list[dict[str, object]] = []
    df = samples_df.copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    ahba = df[df["dataset_upper"] == "AHBA"].copy()
    gtex = df[df["dataset_upper"] == "GTEX"].copy()

    if target_parcels is not None:
        target_set = set(str(p) for p in target_parcels)
        ahba_target = ahba[ahba["tissue_or_parcel"].isin(target_set)].copy()
    elif target_prefix is not None:
        ahba_target = ahba[ahba["tissue_or_parcel"].astype(str).str.startswith(str(target_prefix))].copy()
    else:
        raise ValueError("Provide target_parcels or target_prefix")
    if len(ahba_target) == 0:
        raise ValueError("No AHBA target parcels found for expression ranking")

    for gene_set_name, genes_in in gene_sets.items():
        genes = [str(g) for g in genes_in if str(g) in df.columns]
        if len(genes) < 2:
            continue
        gtex_expr = _group_expression(
            gtex[gtex["tissue_or_parcel"].isin(source_tissues)],
            "tissue_or_parcel",
            genes,
            agg=gtex_agg,
        )
        ahba_expr = _group_expression(ahba_target, "tissue_or_parcel", genes, agg=ahba_agg)
        for source_tissue, source_vec in gtex_expr.iterrows():
            for parcel_name, parcel_vec in ahba_expr.iterrows():
                rows.append(
                    {
                        "gene_set": str(gene_set_name),
                        "source_gtex_tissue": str(source_tissue),
                        target_col_name: str(parcel_name),
                        "gtex_agg": _validate_expression_agg(gtex_agg, name="gtex_agg"),
                        "ahba_agg": _validate_expression_agg(ahba_agg, name="ahba_agg"),
                        "n_genes": int(len(genes)),
                        "pearson_r": _corr_safe(source_vec, parcel_vec, method="pearson"),
                        "spearman_r": _corr_safe(source_vec, parcel_vec, method="spearman"),
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values(
        ["gene_set", "source_gtex_tissue", "pearson_r", "spearman_r", target_col_name],
        ascending=[True, True, False, False, True],
    ).copy()
    out["rank_pearson"] = out.groupby(["gene_set", "source_gtex_tissue"]).cumcount() + 1
    out = out.sort_values(
        ["gene_set", "source_gtex_tissue", "spearman_r", "pearson_r", target_col_name],
        ascending=[True, True, False, False, True],
    ).copy()
    out["rank_spearman"] = out.groupby(["gene_set", "source_gtex_tissue"]).cumcount() + 1
    return out.sort_values(["source_gtex_tissue", "gene_set", "rank_pearson"]).reset_index(drop=True)


def run_expression_pair_matching(
    samples_df,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    source_tissues: Sequence[str],
    target_parcels: Sequence[str] | None = None,
    target_prefix: str | None = None,
    target_col_name: str = "ahba_target_parcel",
    consensus_top_n: int = 3,
    score_col: str = "pearson_r",
    gtex_agg: str = "median",
    ahba_agg: str = "mean",
) -> dict[str, object]:
    """Run expression-based target-parcel ranking plus consensus/profile summaries for two source labels."""
    rankings = compute_expression_parcel_rankings(
        samples_df,
        gene_sets,
        source_tissues=source_tissues,
        target_parcels=target_parcels,
        target_prefix=target_prefix,
        target_col_name=target_col_name,
        gtex_agg=gtex_agg,
        ahba_agg=ahba_agg,
    )
    consensus = summarize_expression_parcel_rankings(
        rankings,
        target_col_name=target_col_name,
        top_n=consensus_top_n,
    )
    profile_similarity = (
        compare_expression_rank_profiles(
            rankings,
            tissues=source_tissues,
            target_col_name=target_col_name,
            score_col=score_col,
        )
        if len(source_tissues) == 2
        else None
    )
    return {
        "rankings": rankings,
        "consensus": consensus,
        "profile_similarity": profile_similarity,
    }


def build_expression_pair_score_pivot(
    rankings_df,
    *,
    source_tissues: Sequence[str],
    target_col_name: str = "ahba_target_parcel",
    score_col: str = "pearson_r",
):
    """Return one row per gene-set/target parcel for direct scatter comparison between two source labels."""
    if len(source_tissues) != 2:
        raise ValueError("score-pivot comparison currently expects exactly two source tissues")
    pivot = rankings_df.pivot_table(
        index=["gene_set", target_col_name],
        columns="source_gtex_tissue",
        values=score_col,
        aggfunc="mean",
    ).dropna(subset=list(source_tissues))
    return pivot.reset_index()


def summarize_expression_parcel_rankings(
    rankings_df,
    *,
    target_col_name: str = "ahba_target_parcel",
    top_n: int = 3,
):
    """Aggregate expression parcel ranks across gene lists."""
    _, pd = _require_array_stack()
    if len(rankings_df) == 0:
        return pd.DataFrame()
    grouped = (
        rankings_df.groupby(["source_gtex_tissue", target_col_name], as_index=False)
        .agg(
            mean_rank_pearson=("rank_pearson", "mean"),
            median_rank_pearson=("rank_pearson", "median"),
            mean_rank_spearman=("rank_spearman", "mean"),
            median_rank_spearman=("rank_spearman", "median"),
            n_rank1_pearson=("rank_pearson", lambda s: int((s == 1).sum())),
            n_top3_pearson=("rank_pearson", lambda s: int((s <= int(top_n)).sum())),
            n_rank1_spearman=("rank_spearman", lambda s: int((s == 1).sum())),
            n_top3_spearman=("rank_spearman", lambda s: int((s <= int(top_n)).sum())),
            mean_pearson_r=("pearson_r", "mean"),
            mean_spearman_r=("spearman_r", "mean"),
            gene_sets_observed=("gene_set", "nunique"),
            min_n_genes=("n_genes", "min"),
        )
        .sort_values(
            ["source_gtex_tissue", "n_rank1_pearson", "n_top3_pearson", "mean_rank_pearson", "mean_pearson_r"],
            ascending=[True, False, False, True, False],
        )
        .reset_index(drop=True)
    )
    grouped["consensus_rank"] = grouped.groupby("source_gtex_tissue").cumcount() + 1
    return grouped


def compare_expression_rank_profiles(
    rankings_df,
    *,
    tissues: Sequence[str],
    target_col_name: str = "ahba_target_parcel",
    score_col: str = "pearson_r",
):
    """Compare whether two GTEx labels point to similar target-parcel profiles."""
    if len(tissues) != 2:
        raise ValueError("rank-profile comparison currently expects exactly two tissues")
    sub = rankings_df[rankings_df["source_gtex_tissue"].isin(tissues)].copy()
    pivot = sub.pivot_table(
        index=["gene_set", target_col_name],
        columns="source_gtex_tissue",
        values=score_col,
        aggfunc="mean",
    )
    if any(t not in pivot.columns for t in tissues):
        return {"score_col": score_col, "n_common_profiles": 0, "profile_correlation": float("nan")}
    common = pivot.dropna(subset=list(tissues))
    corr = common[tissues[0]].corr(common[tissues[1]], method="pearson") if len(common) >= 2 else float("nan")
    return {
        "score_col": score_col,
        "tissue_a": str(tissues[0]),
        "tissue_b": str(tissues[1]),
        "n_common_profiles": int(len(common)),
        "profile_correlation": float(corr),
    }


def summarize_cerebellar_expression_rankings(rankings_df, *, top_n: int = 3):
    """Aggregate cerebellar parcel ranks across gene lists."""
    return summarize_expression_parcel_rankings(rankings_df, target_col_name="ahba_cerebellar_parcel", top_n=top_n)


def compare_cerebellar_tissue_rank_profiles(
    rankings_df,
    *,
    tissues: Sequence[str] = DEFAULT_CEREBELLAR_GTEX_TISSUES,
    score_col: str = "pearson_r",
):
    """Compare whether two GTEx cerebellar labels point to similar AHBA cerebellar parcels."""
    return compare_expression_rank_profiles(
        rankings_df,
        tissues=tissues,
        target_col_name="ahba_cerebellar_parcel",
        score_col=score_col,
    )
