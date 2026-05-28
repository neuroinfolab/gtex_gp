from __future__ import annotations

import numpy as np


def should_fold_hemispheres(gtex_hemi_mode: object = None, matching_policy_hemi_mode: object = None) -> bool:
    """Whether model-space spatial features should collapse hemispheric sign."""
    return str(gtex_hemi_mode).lower() == "mirror_left" or str(matching_policy_hemi_mode).lower() == "force_left"


def model_spatial_coords(coords_full: np.ndarray, *, fold_hemispheres: bool = True) -> np.ndarray:
    """Return model-space coordinates from raw ``[x, y, z]`` parcel coordinates.

    For mirrored / force-left pipelines, the spatial model should see folded
    coordinates in ``[abs(x), y, z]`` order. Raw coordinates should remain the
    source for anatomical metadata, plotting, and exports.
    """
    coords = np.asarray(coords_full, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_full must have shape (n, 3), got {coords.shape}")
    if fold_hemispheres:
        return np.column_stack([np.abs(coords[:, 0]), coords[:, 1], coords[:, 2]]).astype(np.float64)
    return coords.copy().astype(np.float64)
