from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Eligibility:
    subject: str
    n_samples: int
    n_obs_parcels: int
    eligible: bool
    reason: str


def build_target_parcels(ahba_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        ahba_df.groupby("tissue_or_parcel")[["coord_x", "coord_y", "coord_z"]]
        .mean()
        .reset_index()
        .sort_values("tissue_or_parcel")
        .reset_index(drop=True)
    )
    grp["parcel_idx"] = np.arange(len(grp), dtype=np.int32)
    return grp[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]]


def map_gtex_to_target(gtex_df: pd.DataFrame, target_parcels: pd.DataFrame) -> pd.DataFrame:
    out = gtex_df.copy()
    txyz = target_parcels[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    gxyz = out[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    d2 = ((gxyz[:, None, :] - txyz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    out["parcel_idx"] = idx.astype(np.int32)
    out["mapped_parcel"] = target_parcels.loc[idx, "tissue_or_parcel"].to_numpy()
    out["mapping_distance"] = np.sqrt(d2[np.arange(len(out)), idx]).astype(np.float32)
    return out


def macro_system(parcel_name: str) -> str:
    s = str(parcel_name)
    if s.startswith("Cerebellar_"):
        return "cerebellar"
    if s.startswith("LH-") or s.startswith("RH-"):
        return "subcortical"
    if "Amygdala" in s or "Hippocampus" in s:
        return "subcortical"
    if s.startswith("LH_Vis") or s.startswith("RH_Vis") or s.startswith("LH_SomMot") or s.startswith("RH_SomMot"):
        return "visual_somatomotor"
    return "cortical_association"


def hemisphere(parcel_name: str) -> str:
    s = str(parcel_name)
    if s.startswith("LH"):
        return "L"
    if s.startswith("RH"):
        return "R"
    return "M"


def add_target_meta(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    out["macro_system"] = out["tissue_or_parcel"].map(macro_system)
    out["hemisphere"] = out["tissue_or_parcel"].map(hemisphere)
    return out


def add_sample_groups(df: pd.DataFrame, target_meta: pd.DataFrame) -> pd.DataFrame:
    lk_sys = dict(zip(target_meta["parcel_idx"].tolist(), target_meta["macro_system"].tolist()))
    lk_hemi = dict(zip(target_meta["parcel_idx"].tolist(), target_meta["hemisphere"].tolist()))
    out = df.copy()
    out["macro_system"] = out["parcel_idx"].map(lk_sys)
    out["hemisphere"] = out["parcel_idx"].map(lk_hemi)
    return out


def build_region_matrix(
    df: pd.DataFrame,
    gene_cols: List[str],
    target_parcels: pd.DataFrame,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    r = len(target_parcels)
    out = np.full((r, len(gene_cols)), np.nan, dtype=np.float64)
    obs = np.zeros(r, dtype=bool)
    for i in range(r):
        sub = df[df["parcel_idx"] == i]
        if len(sub) == 0:
            continue
        x = sub[gene_cols].to_numpy(dtype=np.float64)
        out[i, :] = np.nanmean(x, axis=0) if agg == "mean" else np.nanmedian(x, axis=0)
        obs[i] = True
    return out, obs


def build_subject_observed_matrices(
    subj_h: pd.DataFrame,
    subj_raw: pd.DataFrame,
    gene_cols: List[str],
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if agg not in {"mean", "median"}:
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    reducer = "mean" if agg == "mean" else "median"
    grp_h = getattr(subj_h.groupby("parcel_idx")[gene_cols], reducer)()
    grp_raw = getattr(subj_raw.groupby("parcel_idx")[gene_cols], reducer)()
    common = sorted(set(grp_h.index.tolist()) & set(grp_raw.index.tolist()))
    if len(common) == 0:
        return np.array([], dtype=np.int32), np.zeros((0, len(gene_cols))), np.zeros((0, len(gene_cols)))
    obs_idx = np.asarray(common, dtype=np.int32)
    xh = grp_h.loc[common, gene_cols].to_numpy(dtype=np.float64)
    xr = grp_raw.loc[common, gene_cols].to_numpy(dtype=np.float64)
    return obs_idx, xh, xr


def build_subject_eligibility(gtex_raw: pd.DataFrame, min_observed_parcels: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    subjects = sorted(gtex_raw["subject"].dropna().astype(str).unique().tolist())
    for sid in subjects:
        sub = gtex_raw[gtex_raw["subject"] == sid].copy()
        n_samples = int(len(sub))
        n_obs = int(sub["parcel_idx"].nunique())
        eligible = bool(n_obs >= min_observed_parcels)
        rows.append(
            {
                "subject": sid,
                "n_samples": n_samples,
                "n_obs_parcels": n_obs,
                "eligible": eligible,
                "reason": "eligible" if eligible else f"insufficient_observed_parcels<{min_observed_parcels}",
            }
        )
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def compute_global_distance_table(target_meta: pd.DataFrame, global_obs_mask: np.ndarray) -> pd.DataFrame:
    coords = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    obs_idx = np.where(global_obs_mask.astype(bool))[0]
    obs_coords = coords[obs_idx, :]
    d2 = ((coords[:, None, :] - obs_coords[None, :, :]) ** 2).sum(axis=2)
    nn = np.argmin(d2, axis=1)
    nearest_obs_idx = obs_idx[nn]
    nearest_name = target_meta.iloc[nearest_obs_idx]["tissue_or_parcel"].astype(str).to_numpy()
    dist = np.sqrt(d2[np.arange(len(coords)), nn])
    out = pd.DataFrame(
        {
            "parcel_idx": target_meta["parcel_idx"].to_numpy(dtype=np.int32),
            "parcel_name": target_meta["tissue_or_parcel"].astype(str).to_numpy(),
            "coord_x": target_meta["coord_x"].to_numpy(dtype=np.float64),
            "coord_y": target_meta["coord_y"].to_numpy(dtype=np.float64),
            "coord_z": target_meta["coord_z"].to_numpy(dtype=np.float64),
            "macro_system": target_meta["macro_system"].astype(str).to_numpy(),
            "hemisphere": target_meta["hemisphere"].astype(str).to_numpy(),
            "is_observed_gtex": global_obs_mask.astype(bool),
            "dist_to_nearest_observed": dist.astype(np.float64),
            "nearest_observed_parcel": nearest_name,
        }
    )
    return out
