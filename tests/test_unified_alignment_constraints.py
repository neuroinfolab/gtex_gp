from __future__ import annotations

import numpy as np

from src.models.unified_generative import update_alignment_R


def test_alignment_is_orthogonal():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(20, 3))
    Z = rng.normal(size=(20, 3))
    R = update_alignment_R(U, Z)
    err = np.linalg.norm(R.T @ R - np.eye(3))
    det = np.linalg.det(R)
    assert err < 1e-6
    assert np.isfinite(det)
    assert abs(abs(det) - 1.0) < 1e-6
