from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

COORD_PATTERN = re.compile(r"\(\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)")
META_COLS = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]


def normalize_gene_name(name: str) -> str:
    return str(name).upper().replace("-", "_").replace(".", "_")


def parse_coordinate_centroid(coord_text: str) -> Tuple[float, float, float]:
    toks = COORD_PATTERN.findall(str(coord_text))
    if not toks:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray([[float(a), float(b), float(c)] for a, b, c in toks], dtype=np.float64)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def load_gene_header_and_hvg(csv_path: Path, hvg_path: Path) -> Dict[str, List[str]]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    genes_all = header[6:]
    hvg_requested = [ln.strip() for ln in hvg_path.read_text().splitlines() if ln.strip()]
    norm_map = {normalize_gene_name(g): g for g in genes_all}
    genes_hvg = [norm_map[normalize_gene_name(g)] for g in hvg_requested if normalize_gene_name(g) in norm_map]
    missing_hvg = [g for g in hvg_requested if normalize_gene_name(g) not in norm_map]
    return {
        "genes_all": genes_all,
        "hvg_requested": hvg_requested,
        "genes_hvg": genes_hvg,
        "missing_hvg": missing_hvg,
    }


def read_expression_subset(csv_path: Path, gene_cols: Iterable[str]) -> pd.DataFrame:
    gene_cols = list(gene_cols)
    usecols = META_COLS + gene_cols
    dtype_map = {
        "subject": "string",
        "age": "string",
        "sex": "string",
        "dataset": "string",
        "tissue_or_parcel": "string",
        "coordinates": "string",
    }
    dtype_map.update({g: np.float32 for g in gene_cols})
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, low_memory=False)
    xyz = np.vstack([parse_coordinate_centroid(c) for c in df["coordinates"]])
    df["coord_x"] = xyz[:, 0]
    df["coord_y"] = xyz[:, 1]
    df["coord_z"] = xyz[:, 2]
    df["coord_abs_x"] = np.abs(df["coord_x"])
    df = df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    df["subject"] = df["subject"].astype(str)
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    return df


def hash_config(d: Dict[str, object]) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def dump_yaml(path: Path, obj: Dict[str, object]) -> None:
    """Best-effort YAML writer without hard dependency on pyyaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(obj, sort_keys=True))
        return
    except Exception:
        pass

    lines: List[str] = []
    for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
        if isinstance(v, (int, float, bool)):
            lines.append(f"{k}: {v}")
        elif v is None:
            lines.append(f"{k}: null")
        else:
            lines.append(f"{k}: '{str(v)}'")
    path.write_text("\n".join(lines) + "\n")
