from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

META_COLS = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]


def gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def validate_gxp_samples(df: pd.DataFrame, *, require_genes: bool = True) -> None:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"gxp_samples table is missing required columns: {missing}")
    genes = gene_columns(df)
    if require_genes and not genes:
        raise ValueError("gxp_samples table has no gene columns")
    seen = set()
    dupes = []
    for c in df.columns:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        raise ValueError(f"gxp_samples table has duplicate columns: {dupes}")


def ordered_columns(gene_cols: Sequence[str]) -> list[str]:
    return META_COLS + list(gene_cols)
