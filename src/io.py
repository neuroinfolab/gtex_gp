from __future__ import annotations

import csv
import os
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


def parse_coordinate_points(coord_text: str) -> np.ndarray:
    toks = COORD_PATTERN.findall(str(coord_text))
    if not toks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[float(a), float(b), float(c)] for a, b, c in toks], dtype=np.float64)


def transform_coordinate_points(points: np.ndarray, hemi_mode: str = "native") -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    mode = str(hemi_mode).strip().lower()
    if mode == "native":
        return pts
    if mode == "mirror_left":
        pts[:, 0] = -np.abs(pts[:, 0])
        return pts
    raise ValueError(f"Unsupported hemi_mode={hemi_mode!r}; expected 'native' or 'mirror_left'")


def parse_coordinate_representative(
    coord_text: str,
    rep_mode: str = "medoid",
    hemi_mode: str = "native",
) -> Tuple[float, float, float]:
    pts = parse_coordinate_points(coord_text)
    if len(pts) == 0:
        return (np.nan, np.nan, np.nan)
    pts = transform_coordinate_points(pts, hemi_mode=hemi_mode)
    mode = str(rep_mode).strip().lower()
    if mode == "centroid":
        rep = pts.mean(axis=0)
    elif mode == "medoid":
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        rep = pts[int(np.argmin(d2.sum(axis=1))), :]
    else:
        raise ValueError(f"Unsupported rep_mode={rep_mode!r}; expected 'centroid' or 'medoid'")
    return (float(rep[0]), float(rep[1]), float(rep[2]))


def _resolve_gtex_spatial_option(value: str | None, env_name: str, default: str) -> str:
    if value is None:
        value = os.environ.get(env_name, default)
    return str(value).strip().lower()


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


def read_expression_subset(
    csv_path: Path,
    gene_cols: Iterable[str],
    rep_mode: str | None = None,
    hemi_mode: str | None = None,
) -> pd.DataFrame:
    gene_cols = list(gene_cols)
    rep_mode_resolved = _resolve_gtex_spatial_option(rep_mode, "GTEX_REP_MODE", "centroid")
    hemi_mode_resolved = _resolve_gtex_spatial_option(hemi_mode, "GTEX_HEMI_MODE", "mirror_left")
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
    xyz = np.vstack(
        [
            parse_coordinate_representative(c, rep_mode=rep_mode_resolved, hemi_mode=hemi_mode_resolved)
            for c in df["coordinates"]
        ]
    )
    df["coord_x"] = xyz[:, 0]
    df["coord_y"] = xyz[:, 1]
    df["coord_z"] = xyz[:, 2]
    df["coord_abs_x"] = np.abs(df["coord_x"])
    df["gtex_rep_mode"] = rep_mode_resolved
    df["gtex_hemi_mode"] = hemi_mode_resolved
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
