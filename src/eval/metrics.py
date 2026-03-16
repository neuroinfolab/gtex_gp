from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def metrics_from_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"pearson_r": np.nan, "spearman_rho": np.nan, "rmse": np.nan, "mae": np.nan, "medae": np.nan}
    pear = np.nan
    if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pear = float(stats.pearsonr(y_true, y_pred).statistic)
    spear = float(stats.spearmanr(y_true, y_pred).statistic)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    medae = float(np.median(np.abs(y_true - y_pred)))
    return {"pearson_r": pear, "spearman_rho": spear, "rmse": rmse, "mae": mae, "medae": medae}


def aggregate_subject_metrics(true_list: list[np.ndarray], pred_list: list[np.ndarray], baseline_list: list[np.ndarray]) -> Dict[str, float]:
    if len(true_list) == 0:
        return {
            "n_points": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
            "baseline_rmse": np.nan,
            "better_than_baseline_rmse": False,
        }
    yt = np.concatenate(true_list)
    yp = np.concatenate(pred_list)
    yb = np.concatenate(baseline_list)
    met = metrics_from_vectors(yt, yp)
    b_rmse = float(np.sqrt(np.mean((yt - yb) ** 2)))
    return {
        "n_points": int(len(yt)),
        **met,
        "baseline_rmse": b_rmse,
        "better_than_baseline_rmse": bool(np.isfinite(met["rmse"]) and met["rmse"] < b_rmse),
    }
