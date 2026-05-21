"""
Shared data-loading helpers for 2D visualisation scripts.

Exposes get_panel_data(subject, model, gene) → dict with keys:
  coords       – (N, 3) MNI coordinates for every parcel that has *any* value
  labels       – list[str] parcel labels, same length as coords
  v1           – array (N,) GTEx input   (NaN where parcel absent)
  v2           – array (N,) Reconstruction
  v3           – array (N,) Whole-Brain Fit (fullfit)
  v4           – array (N,) AHBA reference  (naive fullfit, fixed)
  vmin / vmax  – shared colour limits across v1-v4
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── path setup ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
APP_DIR = HERE.parent
sys.path.insert(0, str(APP_DIR))
import data_loader

SUBJECTS = list(data_loader.SUBJECT_BUNDLES.keys())
MODELS   = ['naive', 'dlam', 'plam']

# ── load atlas once ────────────────────────────────────────────────────────────
labels, atlas_aligned = data_loader.load_alignment()
rh_to_lh_map  = data_loader.build_rh_to_lh_map(atlas_aligned)
excluded_map  = data_loader.build_excluded_parcel_map()

# Build a coord lookup: label → (x, y, z)
_all_atlas = pd.read_csv(
    APP_DIR.parent / 'subject_bundle' / 'atlas_info' /
    'atlas-4S156Parcels_dseg_reformatted.csv'
)
_coord_lookup: dict[str, np.ndarray] = {
    row['label']: np.array([row['mni_x'], row['mni_y'], row['mni_z']])
    for _, row in _all_atlas.iterrows()
}


def _dict_to_array(d: dict, parcel_list: list) -> np.ndarray:
    return np.array([d.get(p, np.nan) for p in parcel_list], dtype=float)


def get_panel_data(subject: str, model: str, gene: str) -> dict:
    npz       = data_loader.load_subject(model, subject)
    naive_npz = data_loader.load_subject('naive', subject)

    # V1 — harmonized ground truth at eval parcels (same combat space as V2)
    # v1_gtex_input returns raw counts which are on a completely different scale
    # to loro_fused_subject_h; using the harmonized version makes comparison valid.
    c1, s1 = data_loader.v1_gtex_harmonized(npz, atlas_aligned, gene)
    c1, s1 = data_loader.remap_rh_to_lh_nearest(c1, s1, rh_to_lh_map)

    # V2 — Reconstruction
    c2, s2 = data_loader.v2_reconstruction(npz, atlas_aligned, gene)
    c2, s2 = data_loader.remap_rh_to_lh_nearest(c2, s2, rh_to_lh_map)

    # V3 — Whole-brain fit (fullfit, mirrored)
    c3r, s3r = data_loader.v4_fullfit(npz, atlas_aligned, gene)
    c2r, s2r = data_loader.v2_reconstruction(npz, atlas_aligned, gene)
    c2l, s2l = data_loader.remap_rh_to_lh_nearest(c2r, s2r, rh_to_lh_map)
    c3, s3 = data_loader.mirror_lh_to_rh(
        {**c3r, **c2l}, {**s3r, **s2l}, rh_to_lh_map)
    c3, s3 = data_loader.fill_excluded_parcels(c3, s3, excluded_map)

    # V4 — AHBA reference (naive model, same mirror logic)
    c4r, s4r = data_loader.v4_fullfit(naive_npz, atlas_aligned, gene)
    c2nr, s2nr = data_loader.v2_reconstruction(naive_npz, atlas_aligned, gene)
    c2nl, s2nl = data_loader.remap_rh_to_lh_nearest(c2nr, s2nr, rh_to_lh_map)
    c4, s4 = data_loader.mirror_lh_to_rh(
        {**c4r, **c2nl}, {**s4r, **s2nl}, rh_to_lh_map)
    c4, s4 = data_loader.fill_excluded_parcels(c4, s4, excluded_map)

    # Union of all parcels that appear in any view
    all_parcels = sorted(
        set(c1) | set(s1) | set(c2) | set(s2) |
        set(c3) | set(s3) | set(c4) | set(s4)
    )
    # Keep only parcels with known coords
    all_parcels = [p for p in all_parcels if p in _coord_lookup]

    coords = np.stack([_coord_lookup[p] for p in all_parcels])  # (N, 3)

    merged = lambda c, s: {**c, **s}
    v1 = _dict_to_array(merged(c1, s1), all_parcels)
    v2 = _dict_to_array(merged(c2, s2), all_parcels)
    v3 = _dict_to_array(merged(c3, s3), all_parcels)
    v4 = _dict_to_array(merged(c4, s4), all_parcels)

    all_vals = np.concatenate([v for v in [v1, v2, v3, v4] if not np.all(np.isnan(v))])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    return dict(
        coords=coords, labels=all_parcels,
        v1=v1, v2=v2, v3=v3, v4=v4,
        vmin=vmin, vmax=vmax,
    )
