from __future__ import annotations

import numpy as np

from src.models.unified_generative import _robust_weights


def test_robust_weights_monotonic():
    r = np.array([0.0, 0.5, 1.0, 3.0, 6.0])
    w_h = _robust_weights(r, "huber", 1.0, 4.0)
    w_t = _robust_weights(r, "student_t", 1.0, 4.0)
    assert np.isfinite(w_h).all() and np.isfinite(w_t).all()
    assert np.all(np.diff(w_h) <= 1e-10)
    assert np.all(np.diff(w_t) <= 1e-10)
