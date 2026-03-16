from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .procrustes import fit_o3, stabilize_invertible


@dataclass
class BasisMap:
    model: str
    M: np.ndarray
    b: np.ndarray
    invM: np.ndarray
    diagnostics: Dict[str, float]


def fit_basis_map(T_sub: np.ndarray, T_ref: np.ndarray, model: str = "o3") -> BasisMap:
    model = str(model)
    k = T_sub.shape[1]
    z = np.zeros(k, dtype=np.float64)

    if model == "o3":
        R = fit_o3(T_sub, T_ref).astype(np.float64)
        M = R
        b = z
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(np.linalg.det(M)),
            "orthogonality_error": float(np.linalg.norm(M.T @ M - np.eye(k), ord="fro")),
            "condition_number": float(np.linalg.cond(M)),
            "regularization_lambda": 0.0,
            "basis_rank": float(np.linalg.matrix_rank(M)),
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    if model == "affine_gl3":
        X = np.c_[T_sub, np.ones(T_sub.shape[0], dtype=np.float64)]
        beta, *_ = np.linalg.lstsq(X, T_ref, rcond=None)
        M0 = beta[:k, :].astype(np.float64)
        b = beta[k, :].astype(np.float64)
        M, lam, det, cond = stabilize_invertible(M0)
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(det),
            "orthogonality_error": float(np.linalg.norm(M.T @ M - np.eye(k), ord="fro")),
            "condition_number": float(cond),
            "regularization_lambda": float(lam),
            "basis_rank": float(np.linalg.matrix_rank(M)),
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    if model == "o3_diagscale":
        R = fit_o3(T_sub, T_ref).astype(np.float64)
        Z = T_sub @ R
        d = np.ones(k, dtype=np.float64)
        b = np.zeros(k, dtype=np.float64)
        for j in range(k):
            x = Z[:, j]
            y = T_ref[:, j]
            if np.nanstd(x) > 1e-8:
                s, c = np.polyfit(x, y, deg=1)
                d[j] = max(abs(float(s)), 1e-6)
                b[j] = float(c)
            else:
                d[j] = 1.0
                b[j] = float(np.nanmean(y - x))
        M0 = R @ np.diag(d)
        M, lam, det, cond = stabilize_invertible(M0)
        invM = np.linalg.inv(M)
        diag = {
            "det_linear": float(det),
            "orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(k), ord="fro")),
            "condition_number": float(cond),
            "regularization_lambda": float(lam),
            "basis_rank": float(np.linalg.matrix_rank(M)),
            "diag_scale_1": float(d[0]) if k >= 1 else np.nan,
            "diag_scale_2": float(d[1]) if k >= 2 else np.nan,
            "diag_scale_3": float(d[2]) if k >= 3 else np.nan,
        }
        return BasisMap(model=model, M=M, b=b, invM=invM, diagnostics=diag)

    raise ValueError(f"Unknown basis model: {model}")


def apply_scores(T: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return T @ basis_map.M + basis_map.b[None, :]


def apply_linear(U: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return U @ basis_map.M


def invert_scores(Tp: np.ndarray, basis_map: BasisMap) -> np.ndarray:
    return (Tp - basis_map.b[None, :]) @ basis_map.invM
