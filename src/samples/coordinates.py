from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .gtex import BRAIN_TISSUE_NAME_MAP

GTEX_ATLAS_COL = "Genotype-Tissue Expression (GTEx v8) project"
BA_ATLAS_KEY_COL = "Brodmann ROI value in nifti"
S156_ATLAS_KEY_COL = "index"


def atlas_gtex_to_tissue_label(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    return "brain - " + str(value).strip().lower()


def normalize_atlas_key(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and value == int(value):
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_tissue_coordinate_map(
    *,
    ba_atlas_map: str | Path,
    s156_atlas_map: str | Path,
    brodmann_coords_path: str | Path,
    s156_coords_path: str | Path,
) -> dict[str, list[tuple[float, float, float]]]:
    ba_atlas = pd.read_csv(ba_atlas_map)
    s156_atlas = pd.read_csv(s156_atlas_map)
    brodmann_coords = pd.read_csv(brodmann_coords_path)
    s156_coords = pd.read_csv(s156_coords_path)
    brodmann_coords["id_str"] = brodmann_coords["id"].astype(str)
    s156_coords["index_str"] = s156_coords["index"].astype(str)

    tissue_to_ba: dict[str, list[str]] = {}
    for _, row in ba_atlas.iterrows():
        tissue = atlas_gtex_to_tissue_label(row.get(GTEX_ATLAS_COL, ""))
        key = normalize_atlas_key(row.get(BA_ATLAS_KEY_COL))
        if tissue and key:
            tissue_to_ba.setdefault(tissue, []).append(key)

    tissue_to_s156: dict[str, list[str]] = {}
    for _, row in s156_atlas.iterrows():
        tissue = atlas_gtex_to_tissue_label(row.get(GTEX_ATLAS_COL, ""))
        key = normalize_atlas_key(row.get(S156_ATLAS_KEY_COL))
        if tissue and key:
            tissue_to_s156.setdefault(tissue, []).append(key)

    cortical = {
        "brain - anterior cingulate cortex (ba24)",
        "brain - cortex",
        "brain - frontal cortex (ba9)",
    }
    out: dict[str, list[tuple[float, float, float]]] = {}
    for tissue in BRAIN_TISSUE_NAME_MAP.values():
        if tissue in cortical and tissue in tissue_to_ba:
            block = brodmann_coords[brodmann_coords["id_str"].isin(tissue_to_ba[tissue])]
        else:
            block = s156_coords[s156_coords["index_str"].isin(tissue_to_s156.get(tissue, []))]
        if len(block):
            out[tissue] = [
                (float(row["mni_x"]), float(row["mni_y"]), float(row["mni_z"]))
                for _, row in block.iterrows()
            ]
        else:
            out[tissue] = [(float("nan"), float("nan"), float("nan"))]
    return out
