from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .ahba import list_ahba_subject_files, load_ahba_metadata, load_ahba_subject_parcellation
from .coordinates import build_tissue_coordinate_map
from .gtex import (
    compute_gtex_expression_threshold_masks,
    list_brain_gct_files,
    load_gct_as_subject_genes,
    load_gtex_metadata,
    tissue_label_from_gct,
)
from .schema import META_COLS, ordered_columns, validate_gxp_samples


def _csv_has_columns(path: Path, required_columns: set[str]) -> bool:
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except OSError:
        return False
    return required_columns.issubset({str(col).strip() for col in header})


def _atlas_file(atlas_dir: Path, name: str, *, required_columns: set[str] | None = None) -> Path:
    candidates = [
        atlas_dir / name,
        atlas_dir / "AtlasMaps" / name,
        Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/atlas_info") / name,
        Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/atlas_info/AtlasMaps") / name,
    ]
    for p in candidates:
        if p.exists() and (required_columns is None or _csv_has_columns(p, required_columns)):
            return p
    raise FileNotFoundError(f"Could not resolve atlas file {name!r} from {atlas_dir}")


def _resolve_gtex_metadata_path(gtex_root: str | Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit)
    root = Path(gtex_root)
    candidates = [
        root / "metadata" / "annotations_v8_GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
        root.parent / "metadata" / "annotations_v8_GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
        root.parent.parent / "metadata" / "annotations_v8_GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_ahba_metadata_path(ahba_root: str | Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit)
    path = Path(ahba_root) / "AHBA_metadata.csv"
    return path if path.exists() else None


def _merge_metadata(df: pd.DataFrame, meta: pd.DataFrame | None, *, dataset_name: str) -> pd.DataFrame:
    if meta is None or meta.empty:
        out = df.copy()
        out.insert(1, "age", pd.NA)
        out.insert(2, "sex", pd.NA)
        return out
    merged = df.merge(meta, on="subject", how="left")
    missing = merged[merged["age"].isna() | merged["sex"].isna()]["subject"].astype(str).unique().tolist()
    if missing:
        raise ValueError(
            f"{dataset_name} metadata did not provide age/sex for {len(missing)} subject(s); "
            f"examples: {missing[:10]}"
        )
    rest = [c for c in merged.columns if c not in {"subject", "age", "sex"}]
    return merged[["subject", "age", "sex"] + rest]


def _read_gene_panel(path: list[str] | str | Path | None) -> list[str] | None:
    if path is None:
        return None
    if isinstance(path, list):
        return [str(x) for x in path]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "label" in df.columns and len(df.columns) > 1:
            cols = [str(c) for c in df.columns if str(c).strip().lower() != "label"]
            if cols:
                return cols
        for col in ("gene", "gene_symbol", "symbol", "gene_name"):
            if col in df.columns:
                return [str(x) for x in df[col].dropna().tolist()]
        return [str(x) for x in df.iloc[:, 0].dropna().tolist()]
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _ahba_available_genes(
    ahba_root: str | Path,
    *,
    parcellation: str = "S156",
    processing: str = "raw",
) -> set[str]:
    files = list_ahba_subject_files(ahba_root, parcellation=parcellation, processing=processing)
    if not files:
        raise RuntimeError(f"No AHBA subject files found under {ahba_root}")
    common: set[str] | None = None
    for path in files:
        header = pd.read_csv(path, nrows=0)
        cols = {
            str(c)
            for c in header.columns
            if str(c) != "label" and not str(c).startswith("Unnamed:")
        }
        common = cols if common is None else (common & cols)
    return common or set()


def compute_gtex_ahba_overlap_report(
    *,
    gtex_root: str | Path,
    sample_attributes_path: str | Path,
    ahba_root: str | Path,
    ahba_processing: str = "raw",
    ahba_parcellation: str = "S156",
    rin_threshold: float = 6.0,
    tpm_threshold: float = 0.1,
    reads_threshold: float = 6.0,
    min_fraction: float = 0.2,
    gene_id_style: str = "symbol",
    gene_panel: list[str] | str | Path | None = None,
    assay_freeze: str | None = None,
    chunksize: int = 4096,
) -> dict[str, object]:
    gtex_info = compute_gtex_expression_threshold_masks(
        gtex_root=gtex_root,
        sample_attributes_path=sample_attributes_path,
        rin_threshold=rin_threshold,
        tpm_threshold=tpm_threshold,
        reads_threshold=reads_threshold,
        min_fraction=min_fraction,
        gene_id_style=gene_id_style,
        assay_freeze=assay_freeze,
        chunksize=chunksize,
    )
    ahba_genes = _ahba_available_genes(
        ahba_root,
        parcellation=ahba_parcellation,
        processing=ahba_processing,
    )
    gtex_universe = set(gtex_info["tpm_universe_genes"])
    tpm_pass = set(gtex_info["tpm_genes"])
    reads_pass = set(gtex_info["reads_genes"])
    gtex_pass = set(gtex_info["passing_genes"])
    overlap = gtex_pass & ahba_genes
    ahba_in_gtex_universe = ahba_genes & gtex_universe
    ahba_not_in_gtex_tpm_universe = ahba_genes - gtex_universe
    ahba_fail_tpm_only = (ahba_in_gtex_universe & reads_pass) - tpm_pass
    ahba_fail_reads_only = (ahba_in_gtex_universe & tpm_pass) - reads_pass
    ahba_fail_both = ahba_in_gtex_universe - (tpm_pass | reads_pass)
    panel = _read_gene_panel(gene_panel)
    final_genes = sorted(overlap)
    report_rows = [
        ("gtex", "total_gtex_genes_in_tpm", gtex_info["report"]["total_gtex_genes_in_tpm"]),
        ("gtex", "retained_brain_samples_after_rin", gtex_info["report"]["retained_brain_samples_after_rin"]),
        ("gtex", "genes_passing_tpm_threshold", gtex_info["report"]["genes_passing_tpm_threshold"]),
        ("gtex", "genes_passing_reads_threshold", gtex_info["report"]["genes_passing_reads_threshold"]),
        ("gtex", "gtex_genes_passing_both_thresholds", gtex_info["report"]["gtex_genes_passing_both_thresholds"]),
        ("ahba", "ahba_genes_available_in_chosen_preprocessing", int(len(ahba_genes))),
        ("overlap", "final_gtex_ahba_overlap_size", int(len(overlap))),
        ("overlap_loss", "ahba_not_in_gtex_tpm_universe", int(len(ahba_not_in_gtex_tpm_universe))),
        ("overlap_loss", "ahba_fail_tpm_only_after_rin", int(len(ahba_fail_tpm_only))),
        ("overlap_loss", "ahba_fail_reads_only_after_rin", int(len(ahba_fail_reads_only))),
        ("overlap_loss", "ahba_fail_both_thresholds_after_rin", int(len(ahba_fail_both))),
    ]
    if panel:
        panel_lookup = {g.upper() for g in panel}
        final_genes = [g for g in final_genes if g.upper() in panel_lookup]
        report_rows.extend(
            [
                ("panel", "active_gene_panel_size", int(len(panel_lookup))),
                ("panel", "final_overlap_after_active_gene_panel", int(len(final_genes))),
            ]
        )
    report_df = pd.DataFrame(report_rows, columns=["group", "step", "size"])
    return {
        "report_df": report_df,
        "report": {step: size for _, step, size in report_rows},
        "filtered_genes": final_genes,
        "gtex_retained_samples": gtex_info["retained_samples"],
        "gtex_passing_genes": sorted(gtex_info["passing_genes"]),
        "ahba_genes": sorted(ahba_genes),
        "ahba_not_in_gtex_tpm_universe": sorted(ahba_not_in_gtex_tpm_universe),
        "ahba_fail_tpm_only_after_rin": sorted(ahba_fail_tpm_only),
        "ahba_fail_reads_only_after_rin": sorted(ahba_fail_reads_only),
        "ahba_fail_both_thresholds_after_rin": sorted(ahba_fail_both),
    }


def _align_and_concat(gtex_df: pd.DataFrame, ahba_df: pd.DataFrame, gene_panel: list[str] | None) -> pd.DataFrame:
    if not gtex_df.columns.is_unique:
        dupes = gtex_df.columns[gtex_df.columns.duplicated()].unique().tolist()
        raise ValueError(
            "GTEx build table has duplicate columns before alignment. "
            f"Sample duplicates: {dupes[:10]}"
        )
    if not ahba_df.columns.is_unique:
        dupes = ahba_df.columns[ahba_df.columns.duplicated()].unique().tolist()
        raise ValueError(
            "AHBA build table has duplicate columns before alignment. "
            f"Sample duplicates: {dupes[:10]}"
        )
    gtex_genes = [c for c in gtex_df.columns if c not in META_COLS]
    ahba_genes = [c for c in ahba_df.columns if c not in META_COLS]
    common = sorted(set(gtex_genes) & set(ahba_genes))
    if gene_panel:
        panel = {g.upper(): g for g in gene_panel}
        common = [g for g in common if g.upper() in panel]
    out = pd.concat(
        [gtex_df[ordered_columns(common)], ahba_df[ordered_columns(common)]],
        ignore_index=True,
    )
    validate_gxp_samples(out)
    return out


def build_gxp_samples(
    *,
    gtex_root: str | Path,
    ahba_root: str | Path,
    atlas_info_dir: str | Path,
    gene_panel: list[str] | str | Path | None = None,
    gtex_sample_attributes_path: str | Path | None = None,
    gtex_rin_threshold: float = 6.0,
    gtex_tpm_threshold: float = 0.1,
    gtex_reads_threshold: float = 6.0,
    gtex_min_fraction: float = 0.2,
    gtex_assay_freeze: str | None = None,
    output: str | Path | None = None,
    expression_kind: str = "tpm",
    gtex_transform: str = "log1p",
    gene_id_style: str = "symbol",
    parcellation: str = "S156",
    ahba_processing: str = "raw",
    gtex_metadata_path: str | Path | None = None,
    ahba_metadata_path: str | Path | None = None,
    return_overlap_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    atlas_dir = Path(atlas_info_dir)
    overlap_info: dict[str, object] | None = None
    effective_gene_panel = _read_gene_panel(gene_panel)
    if gtex_sample_attributes_path is not None:
        overlap_info = compute_gtex_ahba_overlap_report(
            gtex_root=gtex_root,
            sample_attributes_path=gtex_sample_attributes_path,
            ahba_root=ahba_root,
            ahba_processing=ahba_processing,
            ahba_parcellation=parcellation,
            rin_threshold=gtex_rin_threshold,
            tpm_threshold=gtex_tpm_threshold,
            reads_threshold=gtex_reads_threshold,
            min_fraction=gtex_min_fraction,
            gene_id_style=gene_id_style,
            gene_panel=effective_gene_panel,
            assay_freeze=gtex_assay_freeze,
        )
        effective_gene_panel = overlap_info["filtered_genes"]

    coord_map = build_tissue_coordinate_map(
        ba_atlas_map=_atlas_file(atlas_dir, "MaptoBA.csv"),
        s156_atlas_map=_atlas_file(atlas_dir, "MaptoS156.csv"),
        brodmann_coords_path=_atlas_file(
            atlas_dir,
            "overlay_brodmann_2mm_MNI_reformatted.csv",
            required_columns={"mni_x", "mni_y", "mni_z"},
        ),
        s156_coords_path=_atlas_file(
            atlas_dir,
            "atlas-4S156Parcels_dseg_reformatted.csv",
            required_columns={"mni_x", "mni_y", "mni_z"},
        ),
    )

    gtex_parts = []
    for gct in list_brain_gct_files(gtex_root, expression_kind=expression_kind):
        tissue = tissue_label_from_gct(gct)
        gtex_parts.append(
            load_gct_as_subject_genes(
                gct,
                tissue,
                coord_map[tissue],
                transform=gtex_transform,
                gene_id_style=gene_id_style,
            )
        )
    if not gtex_parts:
        raise RuntimeError(f"No GTEx brain GCT files found under {gtex_root}")
    gtex_df = pd.concat(gtex_parts, ignore_index=True)
    resolved_gtex_metadata_path = _resolve_gtex_metadata_path(gtex_root, gtex_metadata_path)
    gtex_meta = load_gtex_metadata(resolved_gtex_metadata_path) if resolved_gtex_metadata_path else None
    gtex_df = _merge_metadata(gtex_df, gtex_meta, dataset_name="GTEx")

    s156_coords = pd.read_csv(_atlas_file(atlas_dir, "atlas-4S156Parcels_dseg_reformatted.csv"))
    ahba_parts = []
    for csv_path in list_ahba_subject_files(ahba_root, parcellation=parcellation, processing=ahba_processing):
        subject = csv_path.name.replace("AHBA_schaefer156_", "").replace(".csv", "")
        ahba_parts.append(load_ahba_subject_parcellation(csv_path, subject, s156_coords))
    if not ahba_parts:
        raise RuntimeError(f"No AHBA subject files found under {ahba_root}")
    ahba_df = pd.concat(ahba_parts, ignore_index=True)
    resolved_ahba_metadata_path = _resolve_ahba_metadata_path(ahba_root, ahba_metadata_path)
    ahba_meta = load_ahba_metadata(resolved_ahba_metadata_path) if resolved_ahba_metadata_path else None
    ahba_df = _merge_metadata(ahba_df, ahba_meta, dataset_name="AHBA")

    out = _align_and_concat(gtex_df, ahba_df, effective_gene_panel)
    if output:
        save = out.copy()
        save["coordinates"] = save["coordinates"].apply(str)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        save.to_csv(output, index=False)
    if return_overlap_info:
        return out, (overlap_info or {})
    return out
