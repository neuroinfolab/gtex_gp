from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BRAIN_TISSUE_NAME_MAP = {
    "Brain_Anterior_cingulate_cortex_BA24": "brain - anterior cingulate cortex (ba24)",
    "Brain_Cortex": "brain - cortex",
    "Brain_Frontal_Cortex_BA9": "brain - frontal cortex (ba9)",
    "Brain_Nucleus_accumbens_basal_ganglia": "brain - nucleus accumbens (basal ganglia)",
    "Brain_Caudate_basal_ganglia": "brain - caudate (basal ganglia)",
    "Brain_Putamen_basal_ganglia": "brain - putamen (basal ganglia)",
    "Brain_Hippocampus": "brain - hippocampus",
    "Brain_Amygdala": "brain - amygdala",
    "Brain_Hypothalamus": "brain - hypothalamus",
    "Brain_Substantia_nigra": "brain - substantia nigra",
    "Brain_Cerebellar_Hemisphere": "brain - cerebellar hemisphere",
    "Brain_Cerebellum": "brain - cerebellum",
}


def _unique_positions(values: list[str]) -> list[int]:
    seen: set[str] = set()
    keep: list[int] = []
    for i, value in enumerate(values):
        if value in seen:
            continue
        seen.add(value)
        keep.append(i)
    return keep


def gtex_subject_id(sample_or_subject_id: str) -> str:
    parts = str(sample_or_subject_id).strip().split("-")
    if len(parts) >= 2 and parts[0].upper() == "GTEX":
        return "-".join(parts[:2])
    return str(sample_or_subject_id).strip()


def _normalize_tissue_label(label: object) -> str:
    return str(label).strip().lower()


def tissue_label_from_gct(path: str | Path) -> str:
    stem = Path(path).name
    stem = stem.replace(".gct.gz", "").replace(".gct", "")
    stem = re.sub(r"^gene_(tpm|reads)_v\d+_", "", stem)
    stem = re.sub(r"^gene_tpm_v10_", "", stem)
    stem = re.sub(r"^gene_reads_v10_", "", stem)
    for key, label in BRAIN_TISSUE_NAME_MAP.items():
        expected = "brain_" + key.replace("Brain_", "").lower()
        if stem == expected:
            return label
    return ""


def resolve_brain_expression_dir(gtex_root: str | Path, expression_kind: str = "tpm") -> Path:
    root = Path(gtex_root)
    kind = str(expression_kind).lower()
    candidates: list[Path]
    if kind in {"tpm", "median_tpm"}:
        candidates = [
            root / "brain_RNA-seq" / "tpm",
            root / "brain RNA-seq" / "tpm",
            root / "brainRNA-seq",
        ]
    elif kind in {"counts", "read_counts", "reads"}:
        candidates = [
            root / "brain_RNA-seq" / "read_counts",
            root / "brain RNA-seq" / "read_counts",
            root / "brainRNA-seq",
        ]
    else:
        raise ValueError("expression_kind must be one of: tpm, median_tpm, read_counts")
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"No GTEx brain expression directory found under {root}")


def resolve_cohortwide_gct_path(gtex_root: str | Path, expression_kind: str = "tpm") -> Path:
    brain_dir = resolve_brain_expression_dir(gtex_root, expression_kind=expression_kind)
    kind = str(expression_kind).lower()
    if kind in {"counts", "read_counts", "reads"}:
        pattern = "*gene_reads.gct*"
    else:
        pattern = "*gene_tpm.gct*"
    matches = sorted(p for p in brain_dir.glob(pattern) if "median" not in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"No cohort-wide GTEx {expression_kind} GCT found under {brain_dir}")
    for p in matches:
        if p.suffix != ".gz":
            return p
    return matches[0]


def list_brain_gct_files(gtex_root: str | Path, expression_kind: str = "tpm") -> list[Path]:
    brain_dir = resolve_brain_expression_dir(gtex_root, expression_kind=expression_kind)
    kind = str(expression_kind).lower()
    if kind in {"counts", "read_counts", "reads"}:
        prefixes = ("gene_reads_",)
    else:
        prefixes = ("gene_tpm_",)
    by_tissue: dict[str, Path] = {}
    for p in brain_dir.glob("*.gct*"):
        name = p.name
        if not name.endswith((".gct", ".gct.gz")):
            continue
        if "spinal_cord" in name:
            continue
        if not name.startswith(prefixes):
            continue
        tissue = tissue_label_from_gct(p)
        if not tissue:
            continue
        prev = by_tissue.get(tissue)
        if prev is None:
            by_tissue[tissue] = p
            continue
        # Prefer the uncompressed .gct when both forms are present.
        if prev.suffix == ".gz" and p.suffix != ".gz":
            by_tissue[tissue] = p
    return sorted(by_tissue.values())


def load_gtex_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"SUBJID": "subject"})
    sex_map = {1: "M", 2: "F", "1": "M", "2": "F"}
    df["sex"] = df["SEX"].map(sex_map)

    def mid_age(value: object) -> float:
        text = str(value).strip()
        if "-" not in text:
            return np.nan
        lo, hi = text.split("-", 1)
        try:
            return float((int(lo) + int(hi)) // 2)
        except ValueError:
            return np.nan

    df["age"] = df["AGE"].apply(mid_age)
    return df[["subject", "age", "sex"]]


def load_gtex_subject_phenotypes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"SUBJID": "subject"})
    sex_map = {1: "M", 2: "F", "1": "M", "2": "F"}
    df["sex"] = df["SEX"].map(sex_map)

    def mid_age(value: object) -> float:
        text = str(value).strip()
        if "-" not in text:
            return np.nan
        lo, hi = text.split("-", 1)
        try:
            return float((int(lo) + int(hi)) // 2)
        except ValueError:
            return np.nan

    df["age"] = df["AGE"].apply(mid_age)
    cols = ["subject", "age", "sex"]
    if "DTHHRDY" in df.columns:
        cols.append("DTHHRDY")
    return df[cols]


def load_gtex_sample_attributes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"SAMPID": "sample"})
    df["subject"] = df["sample"].map(gtex_subject_id)
    keep = ["sample", "subject"]
    for col in ("SMTS", "SMTSD", "SMUBRID", "SMRIN", "SMAFRZE"):
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def filter_gtex_sample_attributes(
    sample_attributes_df: pd.DataFrame,
    *,
    rin_threshold: float | None = 6.0,
    assay_freeze: str | None = None,
) -> pd.DataFrame:
    df = sample_attributes_df.copy()
    allowed_tissues = {_normalize_tissue_label(v) for v in BRAIN_TISSUE_NAME_MAP.values()}
    if "SMTSD" in df.columns:
        df = df[df["SMTSD"].map(_normalize_tissue_label).isin(allowed_tissues)]
    elif "SMTS" in df.columns:
        df = df[df["SMTS"].map(_normalize_tissue_label) == "brain"]
    if rin_threshold is not None and "SMRIN" in df.columns:
        rin = pd.to_numeric(df["SMRIN"], errors="coerce")
        df = df[rin > float(rin_threshold)]
    if assay_freeze is not None and "SMAFRZE" in df.columns:
        df = df[df["SMAFRZE"].astype(str).str.strip().str.upper() == str(assay_freeze).strip().upper()]
    return df.copy()


def _parse_gct_shape(gct_path: str | Path) -> tuple[int, int]:
    with Path(gct_path).open("r") as f:
        _ = f.readline()
        dims = f.readline().strip().split("\t")
    if len(dims) != 2:
        raise ValueError(f"Could not parse GCT shape from {gct_path}")
    return int(dims[0]), int(dims[1])


def _gct_sample_columns(gct_path: str | Path, retained_samples: list[str]) -> list[str]:
    header = pd.read_csv(gct_path, sep="\t", skiprows=2, nrows=0)
    sample_cols = [c for c in header.columns if c in retained_samples]
    if retained_samples and not sample_cols:
        raise RuntimeError(
            f"No retained sample IDs from metadata matched cohort-wide GCT columns in {gct_path}"
        )
    return sample_cols


def _compute_threshold_mask_from_gct(
    gct_path: str | Path,
    *,
    sample_cols: list[str],
    value_threshold: float,
    min_fraction: float,
    gene_id_style: str,
    chunksize: int = 4096,
    collect_universe: bool = False,
) -> tuple[set[str], int, set[str] | None]:
    min_count = max(1, math.ceil(len(sample_cols) * float(min_fraction))) if sample_cols else 1
    usecols = ["Name", "Description"] + sample_cols
    passing: set[str] = set()
    universe: set[str] | None = set() if collect_universe else None
    total_rows = 0
    for chunk in pd.read_csv(gct_path, sep="\t", skiprows=2, usecols=usecols, chunksize=chunksize):
        total_rows += len(chunk)
        gene_ids = (
            chunk["Description"].astype(str)
            if gene_id_style == "symbol"
            else chunk["Name"].astype(str).str.split(".").str[0]
        )
        if universe is not None:
            universe.update(gene_ids.tolist())
        if sample_cols:
            numeric = chunk[sample_cols].apply(pd.to_numeric, errors="coerce")
            counts = (numeric >= value_threshold).sum(axis=1)
            keep = counts >= min_count
            passing.update(gene_ids[keep].tolist())
    return passing, total_rows, universe


def compute_gtex_expression_threshold_masks(
    *,
    gtex_root: str | Path,
    sample_attributes_path: str | Path,
    rin_threshold: float = 6.0,
    tpm_threshold: float = 0.1,
    reads_threshold: float = 6.0,
    min_fraction: float = 0.2,
    gene_id_style: str = "symbol",
    assay_freeze: str | None = None,
    chunksize: int = 4096,
) -> dict[str, object]:
    sample_attrs = load_gtex_sample_attributes(sample_attributes_path)
    retained_attrs = filter_gtex_sample_attributes(
        sample_attrs,
        rin_threshold=rin_threshold,
        assay_freeze=assay_freeze,
    )
    retained_samples = retained_attrs["sample"].astype(str).tolist()
    tpm_gct = resolve_cohortwide_gct_path(gtex_root, expression_kind="tpm")
    reads_gct = resolve_cohortwide_gct_path(gtex_root, expression_kind="read_counts")
    tpm_sample_cols = _gct_sample_columns(tpm_gct, retained_samples)
    reads_sample_cols = _gct_sample_columns(reads_gct, retained_samples)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tpm_future = pool.submit(
            _compute_threshold_mask_from_gct,
            tpm_gct,
            sample_cols=tpm_sample_cols,
            value_threshold=tpm_threshold,
            min_fraction=min_fraction,
            gene_id_style=gene_id_style,
            chunksize=chunksize,
            collect_universe=True,
        )
        reads_future = pool.submit(
            _compute_threshold_mask_from_gct,
            reads_gct,
            sample_cols=reads_sample_cols,
            value_threshold=reads_threshold,
            min_fraction=min_fraction,
            gene_id_style=gene_id_style,
            chunksize=chunksize,
            collect_universe=False,
        )
        tpm_mask, tpm_rows, tpm_universe = tpm_future.result()
        reads_mask, reads_rows, _ = reads_future.result()

    passing_both = tpm_mask & reads_mask
    report = {
        "total_gtex_genes_in_tpm": int(tpm_rows),
        "retained_brain_samples_after_rin": int(len(retained_samples)),
        "genes_passing_tpm_threshold": int(len(tpm_mask)),
        "genes_passing_reads_threshold": int(len(reads_mask)),
        "gtex_genes_passing_both_thresholds": int(len(passing_both)),
    }
    return {
        "report": report,
        "sample_attributes": retained_attrs,
        "retained_samples": retained_samples,
        "tpm_universe_genes": tpm_universe or set(),
        "tpm_genes": tpm_mask,
        "reads_genes": reads_mask,
        "passing_genes": passing_both,
        "tpm_total_rows": int(tpm_rows),
        "reads_total_rows": int(reads_rows),
        "tpm_gct_path": tpm_gct,
        "reads_gct_path": reads_gct,
    }


def load_gct_as_subject_genes(
    gct_path: str | Path,
    tissue_label: str,
    coordinates_list: list[tuple[float, float, float]],
    *,
    transform: str = "log1p",
    gene_id_style: str = "symbol",
) -> pd.DataFrame:
    df = pd.read_csv(gct_path, sep="\t", skiprows=2)
    if gene_id_style == "symbol":
        gene_ids = df["Description"].astype(str).tolist()
    elif gene_id_style == "ensembl":
        gene_ids = df["Name"].astype(str).str.split(".").str[0].tolist()
    else:
        raise ValueError("gene_id_style must be 'symbol' or 'ensembl'")
    keep_gene_positions = _unique_positions(gene_ids)
    if len(keep_gene_positions) != len(gene_ids):
        df = df.iloc[keep_gene_positions].reset_index(drop=True)
        gene_ids = [gene_ids[i] for i in keep_gene_positions]
    sample_ids = [c for c in df.columns if c not in ("Name", "Description")]
    if not sample_ids:
        return pd.DataFrame()

    expr = df.set_index("Name")[sample_ids].T.astype(float)
    expr.index = [gtex_subject_id(sid) for sid in sample_ids]
    dupes = expr.index[expr.index.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(
            "GTEx subject IDs are not unique within a tissue file after subject parsing; "
            f"refusing to average duplicate subjects: {dupes[:10]}"
        )
    if transform == "log1p":
        expr = np.log1p(expr)
    elif transform not in {"none", None}:
        raise ValueError("transform must be 'log1p' or 'none'")
    expr.columns = gene_ids
    expr.insert(0, "coordinates", [coordinates_list] * len(expr))
    expr.insert(0, "tissue_or_parcel", tissue_label)
    expr.insert(0, "dataset", "GTEx")
    expr.insert(0, "subject", expr.index.astype(str))
    return expr.reset_index(drop=True)
