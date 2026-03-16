from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from .metrics import aggregate_subject_metrics, metrics_from_vectors


def run_loro(
    subject_bundle: Dict[str, object],
    pipeline_callable: Callable[[int, np.ndarray], Tuple[np.ndarray, Dict[str, object]]],
    cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    subject = str(subject_bundle["subject"])
    obs_idx = np.asarray(subject_bundle["obs_idx"], dtype=np.int32)
    X_obs_h = np.asarray(subject_bundle["X_obs_h"], dtype=np.float64)
    baseline_h_full = np.asarray(subject_bundle["baseline_h_full"], dtype=np.float64)
    config_hash = str(subject_bundle.get("config_hash", cfg.get("config_hash", "na")))
    min_train_obs = int(cfg.get("min_train_obs_loro", 2))

    idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}

    fold_rows = []
    y_true_all = []
    y_pred_all = []
    y_base_all = []
    skipped = 0

    for fold_id, hold in enumerate(obs_idx.tolist()):
        hold = int(hold)
        train_idx = np.asarray([p for p in obs_idx.tolist() if int(p) != hold], dtype=np.int32)
        if len(train_idx) < min_train_obs:
            skipped += 1
            fold_rows.append(
                {
                    "subject": subject,
                    "fold_id": int(fold_id),
                    "held_out_parcel": hold,
                    "n_train_obs": int(len(train_idx)),
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "medae": np.nan,
                    "baseline_rmse": np.nan,
                    "status": "skipped_too_few_train",
                    "config_hash": config_hash,
                }
            )
            continue

        hold_pos = idx_to_pos[hold]
        x_true = X_obs_h[hold_pos, :]
        x_base = baseline_h_full[hold, :]
        try:
            x_pred, diag = pipeline_callable(hold, train_idx)
            met = metrics_from_vectors(x_true, x_pred)
            base_rmse = float(np.sqrt(np.mean((x_true - x_base) ** 2)))
            leak_flag = bool(diag.get("leak_flag", False))
            row = {
                "subject": subject,
                "fold_id": int(fold_id),
                "held_out_parcel": hold,
                "n_train_obs": int(len(train_idx)),
                **met,
                "baseline_rmse": base_rmse,
                "hold_pred_std": float(diag.get("hold_pred_std", np.nan)),
                "hold_abs_error_mean": float(np.mean(np.abs(x_true - x_pred))),
                "status": "failed_leakage" if leak_flag else "ok",
                "config_hash": config_hash,
            }
            for k, v in diag.items():
                if k in {"leak_flag", "hold_pred_std"}:
                    continue
                if np.isscalar(v):
                    row[f"diag_{k}"] = v
            fold_rows.append(row)
            if leak_flag:
                skipped += 1
                continue
            y_true_all.append(x_true)
            y_pred_all.append(np.asarray(x_pred, dtype=np.float64))
            y_base_all.append(x_base)
        except Exception as e:
            skipped += 1
            fold_rows.append(
                {
                    "subject": subject,
                    "fold_id": int(fold_id),
                    "held_out_parcel": hold,
                    "n_train_obs": int(len(train_idx)),
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "medae": np.nan,
                    "baseline_rmse": np.nan,
                    "status": f"failed:{str(e)[:120]}",
                    "config_hash": config_hash,
                }
            )

    folds_df = pd.DataFrame(fold_rows)
    agg = aggregate_subject_metrics(y_true_all, y_pred_all, y_base_all)
    summary = {
        "subject": subject,
        "n_obs_parcels": int(len(obs_idx)),
        "n_folds": int(len(y_true_all)),
        "n_skipped_folds": int(skipped),
        **agg,
        "config_hash": config_hash,
    }
    return folds_df, summary
