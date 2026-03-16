#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.eval.loro import run_loro
from src.harmonize import fit_harmonizer
from src.latent.pls import fit_subject_pls
from src.models.baseline_pipeline import run_subject
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_eligibility,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)


DEFAULT_COMBOS = [
    "combat__affine_gl3__distance_shrink__rbf",
    "hier_affine__affine_gl3__distance_shrink__rbf",
    "combat__affine_gl3__constrained_anchor__rbf",
    "hier_affine__affine_gl3__constrained_anchor__rbf",
    "robustz_affine__affine_gl3__distance_shrink__rbf",
    "zscore_affine__affine_gl3__distance_shrink__rbf",
]


@dataclass
class Config:
    run_root: str = "out/vnext_alignment"
    csv_path: str = "gxp_samples.csv"
    out_root: str = "out/vnext_alignment_allgenes"
    chunk_size: int = 500
    seed: int = 123
    min_observed_parcels: int = 5
    c_min: int = 8
    n_jobs: int = 1
    smoke_subjects: int = 0
    combos: str = ""
    n_comp_target: int = 3
    ridge_alpha_bridge: float = 1e-2
    rbf_smoothing: float = 0.10
    gp_rbf_length: float = 25.0
    distance_d0: float = 45.0
    distance_tau: float = 10.0
    whiten_eps: float = 1e-4
    min_group_samples: int = 200
    combat_use_covariates: bool = True
    hier_lambda_a: float = 10.0
    hier_lambda_b: float = 10.0
    hier_base_method: str = "robustz_affine"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="All-gene validation on gate-passing vNext combos.")
    p.add_argument("--run-root", default=Config.run_root)
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--min-observed-parcels", type=int, default=Config.min_observed_parcels)
    p.add_argument("--c-min", type=int, default=Config.c_min)
    p.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    p.add_argument("--smoke-subjects", type=int, default=Config.smoke_subjects)
    p.add_argument("--combos", default=Config.combos)
    p.add_argument("--n-comp-target", type=int, default=Config.n_comp_target)
    p.add_argument("--ridge-alpha-bridge", type=float, default=Config.ridge_alpha_bridge)
    p.add_argument("--rbf-smoothing", type=float, default=Config.rbf_smoothing)
    p.add_argument("--gp-rbf-length", type=float, default=Config.gp_rbf_length)
    p.add_argument("--distance-d0", type=float, default=Config.distance_d0)
    p.add_argument("--distance-tau", type=float, default=Config.distance_tau)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--min-group-samples", type=int, default=Config.min_group_samples)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.combat_use_covariates)
    p.add_argument("--hier-lambda-a", type=float, default=Config.hier_lambda_a)
    p.add_argument("--hier-lambda-b", type=float, default=Config.hier_lambda_b)
    p.add_argument("--hier-base-method", default=Config.hier_base_method)
    a = p.parse_args()
    return Config(**vars(a))


def _parse_combo(combo: str) -> Dict[str, str]:
    parts = str(combo).split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid combo id: {combo}")
    return {
        "model_combo": combo,
        "harmonization": parts[0],
        "basis_model": parts[1],
        "strategy": parts[2],
        "spatial_method": parts[3],
    }


def _read_passing_combos(run_root: Path, override: str) -> List[str]:
    if str(override).strip():
        return [x.strip() for x in str(override).split(",") if x.strip()]
    gate_path = run_root / "tables" / "rollout_gate_summary_consolidated.csv"
    if gate_path.exists():
        gate = pd.read_csv(gate_path)
        gate = gate[gate["gate_pass"] == True].copy()  # noqa: E712
        combos = gate.sort_values("mean_pearson", ascending=False)["model_combo"].astype(str).tolist()
        if len(combos) >= 6:
            return combos[:6]
        if len(combos) > 0:
            return combos
    return list(DEFAULT_COMBOS)


def _fit_harmonizer_for_strategy(
    cfg: Config,
    hm: str,
    strategy: str,
    ahba_train: pd.DataFrame,
    gtex_train: pd.DataFrame,
    genes: List[str],
    subject_subset: List[str] | None = None,
):
    method = "piecewise_robustz" if strategy == "piecewise_harmonization" else hm
    cfg_use = cfg
    if method == "hier_affine" and subject_subset is not None:
        cfg_use = SimpleNamespace(**asdict(cfg), hier_subject_subset=[str(s) for s in subject_subset])
    return fit_harmonizer(ahba_train, gtex_train, genes, method=method, cfg=cfg_use)


def _write_report(out_root: Path, summary: Dict[str, object], best_combo: str | None) -> None:
    lines = [
        "# All-Gene Gate-Pass Validation Report",
        "",
        f"- Timestamp (UTC): {summary.get('timestamp_utc')}",
        f"- Input run root: `{summary.get('run_root')}`",
        f"- Data source: `{summary.get('csv_path')}`",
        f"- Genes (all): {summary.get('n_genes_all')}",
        f"- Eligible subjects: {summary.get('n_subjects_eligible')}",
        f"- Evaluated combos: {summary.get('n_combos')}",
        "",
        "## Best combo (all-gene)",
        "",
        f"- `{best_combo}`" if best_combo else "- none",
        "",
        "## Next Steps After A-D (from prior instructions)",
        "",
        "1. Milestone E: implement unified probabilistic generative model in `src/models/unified_generative.py` with MAP coordinate-ascent and posterior uncertainty outputs.",
        "2. Milestone F: add robust likelihood (Huber/Student-t) and optional heteroscedastic noise, then evaluate with uncertainty reliability and tail-error diagnostics.",
    ]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)

    root = Path(".").resolve()
    run_root = (root / cfg.run_root).resolve()
    csv_path = (root / cfg.csv_path).resolve()
    out_root = (root / cfg.out_root).resolve()
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    config_dir = out_root / "configs"
    for d in [out_root, table_dir, fig_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    header = io_utils.load_gene_header_and_hvg(csv_path, root / "ahba_100hvg.txt")
    genes_all = header["genes_all"]
    if len(genes_all) == 0:
        raise RuntimeError("No gene columns found in csv header.")

    combos = _read_passing_combos(run_root, cfg.combos)
    combo_specs = [_parse_combo(c) for c in combos]
    pd.DataFrame(combo_specs).to_csv(table_dir / "selected_combos.csv", index=False)

    df = io_utils.read_expression_subset(csv_path, genes_all)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEX subset empty.")

    target = build_target_parcels(ahba_raw)
    parcel_lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(parcel_lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)

    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    y_full = np.c_[coords_full[:, 1], coords_full[:, 2], np.abs(coords_full[:, 0])]

    eligibility_df = build_subject_eligibility(gtex_raw, cfg.min_observed_parcels)
    if cfg.smoke_subjects > 0:
        keep = eligibility_df[eligibility_df["eligible"]].head(int(cfg.smoke_subjects))["subject"].tolist()
        eligibility_df["eligible"] = eligibility_df["subject"].isin(set(keep))
    eligibility_df.to_csv(table_dir / "subject_eligibility.csv", index=False)
    eligible_subjects = eligibility_df[eligibility_df["eligible"]]["subject"].astype(str).tolist()
    if len(eligible_subjects) == 0:
        raise RuntimeError("No eligible subjects.")

    cfg_dict = asdict(cfg)
    io_utils.dump_yaml(config_dir / "allgenes_validation.yaml", cfg_dict)
    io_utils.dump_json(config_dir / "allgenes_validation.json", cfg_dict)
    cfg_hash = io_utils.hash_config(cfg_dict)

    fold_rows_all: List[pd.DataFrame] = []
    summary_rows_all: List[Dict[str, object]] = []

    for spec in combo_specs:
        hm = spec["harmonization"]
        bm = spec["basis_model"]
        st = spec["strategy"]
        sm = spec["spatial_method"]
        combo = spec["model_combo"]
        print(f"[all-genes combo] {combo}")

        harmonizer = _fit_harmonizer_for_strategy(cfg, hm, st, ahba_raw, gtex_raw, genes_all)
        ahba_h = harmonizer.transform(ahba_raw, "AHBA")
        gtex_h = harmonizer.transform(gtex_raw, "GTEX")

        ahba_h_full, _ = build_region_matrix(ahba_h, genes_all, target_meta, agg="mean")
        ahba_pls = fit_subject_pls(ahba_h_full, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)
        ahba_ref_T = ahba_pls["T"]

        stack_h = []
        stack_raw = []

        for sid in eligible_subjects:
            subj_h = gtex_h[gtex_h["subject"] == sid].copy()
            subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()
            obs_idx, X_obs_h, X_obs_raw = build_subject_observed_matrices(subj_h, subj_raw, genes_all)
            if len(obs_idx) < cfg.min_observed_parcels:
                continue

            idx_to_pos = {int(p): i for i, p in enumerate(obs_idx.tolist())}
            subject_bundle = {
                "subject": sid,
                "obs_idx": obs_idx,
                "X_obs_h": X_obs_h,
                "X_obs_raw": X_obs_raw,
                "coords_full": coords_full,
                "target_meta": target_meta,
            }
            atlas_bundle = {"ahba_h_full": ahba_h_full, "ahba_ref_T": ahba_ref_T}
            method_bundle = {
                "harmonizer": harmonizer,
                "basis_model": bm,
                "strategy": st,
                "spatial_method": sm,
                "n_comp_target": cfg.n_comp_target,
                "ridge_alpha_bridge": cfg.ridge_alpha_bridge,
                "rbf_smoothing": cfg.rbf_smoothing,
                "gp_rbf_length": cfg.gp_rbf_length,
                "seed": cfg.seed,
                "c_min": cfg.c_min,
                "distance_d0": cfg.distance_d0,
                "distance_tau": cfg.distance_tau,
                "uncertainty_shrink": bool(st == "gp_uncertainty"),
            }

            full_pred, _ = run_subject(subject_bundle, atlas_bundle, method_bundle, cfg_dict)
            stack_h.append(full_pred["X_full_h"])
            stack_raw.append(full_pred["X_full_raw"])

            def _pipeline_for_fold(hold: int, train_idx: np.ndarray):
                train_mask = ~((gtex_raw["subject"] == sid) & (gtex_raw["parcel_idx"] == int(hold)))
                gtex_train_fold = add_sample_groups(gtex_raw[train_mask].copy(), target_meta)
                ahba_train_fold = ahba_raw.copy()

                fold_harm = _fit_harmonizer_for_strategy(
                    cfg,
                    hm,
                    st,
                    ahba_train_fold,
                    gtex_train_fold,
                    genes_all,
                    subject_subset=[sid],
                )
                ahba_h_fold = fold_harm.transform(ahba_train_fold, "AHBA")
                gtex_h_fold = fold_harm.transform(gtex_train_fold, "GTEX")
                ahba_h_full_fold, _ = build_region_matrix(ahba_h_fold, genes_all, target_meta, agg="mean")
                ahba_pls_fold = fit_subject_pls(ahba_h_full_fold, y_full, n_comp_target=cfg.n_comp_target, adaptive=True)

                subj_h_fold = gtex_h_fold[gtex_h_fold["subject"] == sid].copy()
                subj_raw_fold = gtex_train_fold[gtex_train_fold["subject"] == sid].copy()
                obs_idx_fold, xh_fold, xr_fold = build_subject_observed_matrices(subj_h_fold, subj_raw_fold, genes_all)

                leak_flag = bool(int(hold) in set(obs_idx_fold.tolist()))
                if len(obs_idx_fold) < 3:
                    raise RuntimeError("insufficient train parcels in fold")

                sb_fold = {
                    "subject": sid,
                    "obs_idx": obs_idx_fold,
                    "X_obs_h": xh_fold,
                    "X_obs_raw": xr_fold,
                    "coords_full": coords_full,
                    "target_meta": target_meta,
                }
                ab_fold = {"ahba_h_full": ahba_h_full_fold, "ahba_ref_T": ahba_pls_fold["T"]}
                mb_fold = dict(method_bundle)
                mb_fold["harmonizer"] = fold_harm
                pred_fold, diag_fold = run_subject(sb_fold, ab_fold, mb_fold, cfg_dict)
                hold_pred = pred_fold["X_full_h"][int(hold), :]
                hold_pos = idx_to_pos[int(hold)]
                hold_std = float(np.sqrt(max(float(diag_fold.get("uvar", np.zeros(coords_full.shape[0]))[int(hold)]), 0.0)))
                return hold_pred, {"leak_flag": leak_flag, "hold_pred_std": hold_std}

            eval_bundle = {
                "subject": sid,
                "obs_idx": obs_idx,
                "X_obs_h": X_obs_h,
                "baseline_h_full": ahba_h_full,
                "config_hash": cfg_hash,
            }
            folds_df, summary = run_loro(eval_bundle, _pipeline_for_fold, {"config_hash": cfg_hash})
            folds_df["model_combo"] = combo
            folds_df["harmonization"] = hm
            folds_df["basis_model"] = bm
            folds_df["strategy"] = st
            folds_df["spatial_method"] = sm
            fold_rows_all.append(folds_df)

            summary.update(
                {
                    "model_combo": combo,
                    "harmonization": hm,
                    "basis_model": bm,
                    "strategy": st,
                    "spatial_method": sm,
                    "coverage_tier": "high" if int(summary["n_obs_parcels"]) >= cfg.c_min else "low",
                    "deployment_mode": "model" if int(summary["n_obs_parcels"]) >= cfg.c_min else "atlas_prior",
                }
            )
            summary_rows_all.append(summary)

        if len(stack_h) > 0:
            mean_h = np.nanmean(np.stack(stack_h, axis=0), axis=0)
            mean_raw = np.nanmean(np.stack(stack_raw, axis=0), axis=0)
            gtex_h_tbl = pd.concat(
                [
                    target_meta[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].rename(columns={"tissue_or_parcel": "parcel_name"}).reset_index(drop=True),
                    pd.DataFrame(mean_h, columns=genes_all),
                ],
                axis=1,
            )
            gtex_raw_tbl = pd.concat(
                [
                    target_meta[["parcel_idx", "tissue_or_parcel", "coord_x", "coord_y", "coord_z"]].rename(columns={"tissue_or_parcel": "parcel_name"}).reset_index(drop=True),
                    pd.DataFrame(mean_raw, columns=genes_all),
                ],
                axis=1,
            )
            gtex_h_tbl.to_csv(table_dir / f"aggregate_allgenes_{combo}_mean_harmonized.csv", index=False)
            gtex_raw_tbl.to_csv(table_dir / f"aggregate_allgenes_{combo}_mean_raw.csv", index=False)

    folds_all = pd.concat(fold_rows_all, ignore_index=True) if len(fold_rows_all) else pd.DataFrame()
    summary_all = pd.DataFrame(summary_rows_all)
    folds_all.to_csv(table_dir / "subject_loro_folds_allgenes.csv", index=False)
    summary_all.to_csv(table_dir / "subject_loro_summary_allgenes.csv", index=False)

    if len(summary_all) > 0:
        group_cols = ["model_combo", "harmonization", "basis_model", "strategy", "spatial_method"]
        model_grid = (
            summary_all.groupby(group_cols, as_index=False)
            .agg(
                n_subjects=("subject", "nunique"),
                mean_pearson=("pearson_r", "mean"),
                median_pearson=("pearson_r", "median"),
                mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
                mean_baseline_rmse=("baseline_rmse", "mean"),
                frac_better_baseline_rmse=("better_than_baseline_rmse", "mean"),
            )
            .sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        model_grid = pd.DataFrame()
    model_grid.to_csv(table_dir / "model_grid_summary_allgenes.csv", index=False)

    gate = (
        summary_all.groupby(["model_combo"], as_index=False)
        .agg(mean_pearson=("pearson_r", "mean"), mean_rmse=("rmse", "mean"), mean_baseline_rmse=("baseline_rmse", "mean"), n_subjects=("subject", "nunique"))
        .sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True])
    ) if len(summary_all) else pd.DataFrame(columns=["model_combo", "mean_pearson", "mean_rmse", "mean_baseline_rmse", "n_subjects"])
    if len(gate):
        gate["gate_pass"] = (gate["mean_pearson"] >= 0.50) & (gate["mean_rmse"] < gate["mean_baseline_rmse"])
    gate.to_csv(table_dir / "rollout_gate_summary_allgenes.csv", index=False)

    if len(summary_all):
        tier = (
            summary_all.groupby(["model_combo", "coverage_tier"], as_index=False)
            .agg(mean_pearson=("pearson_r", "mean"), mean_rmse=("rmse", "mean"), mean_baseline_rmse=("baseline_rmse", "mean"), n_subjects=("subject", "nunique"))
        )
        tier["gate_pass"] = (tier["mean_pearson"] >= 0.50) & (tier["mean_rmse"] < tier["mean_baseline_rmse"])
    else:
        tier = pd.DataFrame(columns=["model_combo", "coverage_tier", "mean_pearson", "mean_rmse", "mean_baseline_rmse", "n_subjects", "gate_pass"])
    tier.to_csv(table_dir / "coverage_tier_gate_summary_allgenes.csv", index=False)

    hvg_path = run_root / "tables" / "model_grid_summary_hvg.csv"
    if hvg_path.exists() and len(model_grid):
        hvg = pd.read_csv(hvg_path)[["model_combo", "mean_pearson", "mean_rmse", "frac_better_baseline_rmse"]].rename(
            columns={
                "mean_pearson": "hvg_mean_pearson",
                "mean_rmse": "hvg_mean_rmse",
                "frac_better_baseline_rmse": "hvg_frac_better_baseline_rmse",
            }
        )
        cmp = model_grid[["model_combo", "mean_pearson", "mean_rmse", "frac_better_baseline_rmse"]].merge(hvg, on="model_combo", how="left")
        cmp["delta_mean_pearson_allgenes_minus_hvg"] = cmp["mean_pearson"] - cmp["hvg_mean_pearson"]
        cmp["delta_mean_rmse_allgenes_minus_hvg"] = cmp["mean_rmse"] - cmp["hvg_mean_rmse"]
        cmp["delta_frac_better_baseline"] = cmp["frac_better_baseline_rmse"] - cmp["hvg_frac_better_baseline_rmse"]
    else:
        cmp = pd.DataFrame()
    cmp.to_csv(table_dir / "allgenes_vs_hvg_comparison.csv", index=False)

    if len(model_grid):
        plt.figure(figsize=(10, 4.5))
        d = model_grid.sort_values("mean_pearson", ascending=False)
        sns.barplot(data=d, x="model_combo", y="mean_pearson", color="#3274A1")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean Pearson (LORO)")
        plt.title("All-gene model ranking")
        plt.tight_layout()
        plt.savefig(fig_dir / "allgenes_model_ranking.png", dpi=220)
        plt.close()

    if len(summary_all):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        sns.scatterplot(data=summary_all, x="n_obs_parcels", y="pearson_r", hue="model_combo", s=16, alpha=0.6, ax=axes[0], legend=False)
        axes[0].set_title("All-gene: coverage vs Pearson")
        axes[0].grid(True, alpha=0.2)
        sns.scatterplot(data=summary_all, x="n_obs_parcels", y="rmse", hue="model_combo", s=16, alpha=0.6, ax=axes[1], legend=False)
        axes[1].set_title("All-gene: coverage vs RMSE")
        axes[1].grid(True, alpha=0.2)
        fig.savefig(fig_dir / "allgenes_coverage_vs_performance.png", dpi=220)
        plt.close(fig)

    if len(summary_all):
        if "robustz_affine__affine_gl3__distance_shrink__rbf" in set(summary_all["model_combo"].tolist()):
            baseline_combo = "robustz_affine__affine_gl3__distance_shrink__rbf"
        else:
            baseline_combo = str(model_grid.iloc[0]["model_combo"]) if len(model_grid) else ""
        base = summary_all[summary_all["model_combo"] == baseline_combo][["subject", "rmse"]].rename(columns={"rmse": "baseline_rmse_subject"})
        delta = summary_all.merge(base, on="subject", how="left")
        delta = delta[delta["model_combo"] != baseline_combo].copy()
        if len(delta):
            delta["delta_rmse_vs_baseline"] = delta["rmse"] - delta["baseline_rmse_subject"]
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=delta, x="baseline_rmse_subject", y="delta_rmse_vs_baseline", hue="model_combo", s=18, alpha=0.65)
            plt.axhline(0, color="black", lw=1)
            plt.xlabel(f"Subject RMSE baseline ({baseline_combo})")
            plt.ylabel("RMSE delta vs baseline")
            plt.title("All-gene baseline-vs-model RMSE deltas")
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(fig_dir / "allgenes_baseline_vs_model_scatter.png", dpi=220)
            plt.close()

    best_combo = str(model_grid.iloc[0]["model_combo"]) if len(model_grid) else None
    summary = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "run_root": str(run_root),
        "csv_path": str(csv_path),
        "config_hash": cfg_hash,
        "n_genes_all": int(len(genes_all)),
        "n_subjects_total": int(len(eligibility_df)),
        "n_subjects_eligible": int(eligibility_df["eligible"].sum()),
        "n_combos": int(len(combo_specs)),
        "combos": [c["model_combo"] for c in combo_specs],
        "best_combo": best_combo,
        "best_mean_pearson": float(model_grid.iloc[0]["mean_pearson"]) if len(model_grid) else np.nan,
        "best_mean_rmse": float(model_grid.iloc[0]["mean_rmse"]) if len(model_grid) else np.nan,
    }
    io_utils.dump_json(out_root / "summary.json", summary)
    _write_report(out_root, summary, best_combo)

    print("All-gene gate-pass validation complete.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
