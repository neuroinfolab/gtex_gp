#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.models.unified_generative import UnifiedGenerativeConfig, fit_global_atlas_unified, infer_subject_unified
from src.preprocess import (
    add_sample_groups,
    add_target_meta,
    build_region_matrix,
    build_subject_observed_matrices,
    build_target_parcels,
    map_gtex_to_target,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export unified winner all-gene completed harmonized dataset with imputation flag")
    p.add_argument("--csv-path", default="/Users/erdem/Documents/github/gtex_gp/gxp_samples.csv")
    p.add_argument("--winner-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified")
    p.add_argument("--out-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_final")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=True)
    return p.parse_args()


def _read_all_genes(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header[6:]


def _coord_to_str(x: float, y: float, z: float) -> str:
    return f"[({x:.6f}, {y:.6f}, {z:.6f})]"


def _winner_from_summary(winner_root: Path) -> str:
    summary = json.loads((winner_root / "summary.json").read_text())
    return str(summary["winner_model"])


def _extract_winner_config(winner_model: str) -> dict:
    # Example:
    # unified__harm=combat__cal=hier_affine_map__robust=student_t__hetero=gene_var__uncshrink=false
    parts = winner_model.split("__")
    cfg = {
        "harm": "combat",
        "cal": "hier_affine_map",
        "robust": "student_t",
        "hetero": "gene_var",
        "uncshrink": "false",
    }
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            cfg[k] = v
    return cfg


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    csv_path = Path(args.csv_path).resolve()
    winner_root = Path(args.winner_root).resolve()
    out_root = Path(args.out_root).resolve()
    tab_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    winner_model = _winner_from_summary(winner_root)
    winner_cfg = _extract_winner_config(winner_model)

    genes_all = _read_all_genes(csv_path)
    df = io_utils.read_expression_subset(csv_path, genes_all)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)
    if len(ahba_raw) == 0 or len(gtex_raw) == 0:
        raise RuntimeError("AHBA or GTEx subset empty")

    target = build_target_parcels(ahba_raw)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    target_meta = add_target_meta(target)
    ahba_raw = add_sample_groups(ahba_raw, target_meta)
    gtex_raw = add_sample_groups(gtex_raw, target_meta)

    cfg_ns = SimpleNamespace(
        combat_use_covariates=bool(args.combat_use_covariates),
        whiten_eps=1e-4,
        hier_lambda_a=10.0,
        hier_lambda_b=10.0,
        hier_base_method="robustz_affine",
    )
    harmonizer = fit_harmonizer(ahba_raw, gtex_raw, genes_all, method=str(winner_cfg.get("harm", "combat")), cfg=cfg_ns)
    ahba_h = harmonizer.transform(ahba_raw, "AHBA")
    gtex_h = harmonizer.transform(gtex_raw, "GTEX")

    coords_full = target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    ahba_h_full, _ = build_region_matrix(ahba_h, genes_all, target_meta, agg="mean")

    ucfg = UnifiedGenerativeConfig(
        latent_dim=3,
        max_iters=5,
        lambda_w=1.0,
        lambda_z=1.0,
        lambda_cal_a=10.0,
        lambda_cal_b=10.0,
        gp_length_scale=25.0,
        gp_noise=1e-3,
        robust_loss=str(winner_cfg.get("robust", "student_t")),
        heteroscedastic=str(winner_cfg.get("hetero", "none")).lower() != "none",
        calibration_mode=str(winner_cfg.get("cal", "hier_affine_map")),
        uncertainty_shrink=str(winner_cfg.get("uncshrink", "false")).lower() == "true",
        unc_alpha=0.5,
        unc_beta=0.5,
        unc_m0=0.5,
        unc_tau=0.2,
        random_state=int(args.seed),
    )
    atlas = fit_global_atlas_unified(ahba_h_full, coords_full, ucfg)

    # Build observed parcel mask per subject from source GTEx rows.
    observed_map = gtex_raw.groupby("subject")["parcel_idx"].apply(lambda s: set(int(x) for x in s.unique())).to_dict()

    # Prepare outputs
    meta_cols = ["subject", "age", "sex", "dataset", "tissue_or_parcel", "coordinates"]
    export_cols = meta_cols + ["is_imputed"] + genes_all

    gtex_out_path = out_root / "gxp_completed_harmonized_gtex_only.csv"
    combined_out_path = out_root / "gxp_completed_harmonized_combined.csv"

    subjects = sorted(gtex_raw["subject"].astype(str).unique().tolist())
    imputation_rows = []
    n_rows_gtex_actual = 0
    n_rows_imputed_1 = 0
    n_rows_imputed_0_gtex = 0
    missing_gene_cells_gtex = 0

    # Reset outputs if rerun.
    if gtex_out_path.exists():
        gtex_out_path.unlink()
    if combined_out_path.exists():
        combined_out_path.unlink()

    # Write AHBA harmonized rows into combined first.
    ahba_export = ahba_h[meta_cols + genes_all].copy()
    ahba_export.insert(6, "is_imputed", 0)
    ahba_export = ahba_export[export_cols]
    ahba_export.to_csv(combined_out_path, index=False)

    first_gtex_write = True
    for sid in subjects:
        subj_h = gtex_h[gtex_h["subject"] == sid].copy()
        subj_raw = gtex_raw[gtex_raw["subject"] == sid].copy()

        obs_idx, xh, _ = build_subject_observed_matrices(subj_h, subj_raw, genes_all)
        obs_set = set(int(x) for x in obs_idx.tolist())

        if len(obs_idx) == 0:
            x_pred_h = ahba_h_full.copy()
        else:
            res = infer_subject_unified(
                {"subject": sid, "obs_idx": obs_idx, "X_obs_h": xh},
                atlas,
                ucfg,
                fold_ctx={"prior_h": ahba_h_full, "obs_idx": obs_idx},
            )
            x_pred_h = np.asarray(res["x_hat_h_full"], dtype=np.float64)
            x_pred_h[obs_idx, :] = xh

        # Metadata defaults from raw subject rows.
        age = str(subj_raw["age"].dropna().iloc[0]) if len(subj_raw["age"].dropna()) else ""
        sex = str(subj_raw["sex"].dropna().iloc[0]) if len(subj_raw["sex"].dropna()) else ""

        n_r = len(target_meta)
        meta_df = pd.DataFrame(
            {
                "subject": np.repeat(sid, n_r),
                "age": np.repeat(age, n_r),
                "sex": np.repeat(sex, n_r),
                "dataset": np.repeat("GTEx", n_r),
                "tissue_or_parcel": target_meta["tissue_or_parcel"].astype(str).to_numpy(),
                "coordinates": [
                    _coord_to_str(float(cx), float(cy), float(cz))
                    for cx, cy, cz in target_meta[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
                ],
                "is_imputed": np.asarray([0 if r in obs_set else 1 for r in range(n_r)], dtype=np.int32),
            }
        )
        gene_df = pd.DataFrame(x_pred_h, columns=genes_all)
        gtex_export_df = pd.concat([meta_df, gene_df], axis=1)[export_cols]
        gtex_export_df.to_csv(gtex_out_path, mode="w" if first_gtex_write else "a", header=first_gtex_write, index=False)
        gtex_export_df.to_csv(combined_out_path, mode="a", header=False, index=False)
        first_gtex_write = False

        n_rows_gtex_actual += int(len(gtex_export_df))
        n_rows_imputed_1 += int((gtex_export_df["is_imputed"] == 1).sum())
        n_rows_imputed_0_gtex += int((gtex_export_df["is_imputed"] == 0).sum())
        missing_gene_cells_gtex += int(np.isnan(x_pred_h).sum())

        n_obs = int(len(observed_map.get(sid, set())))
        n_imp = int(len(target_meta) - n_obs)
        imputation_rows.append(
            {
                "subject": sid,
                "n_total_parcels": int(len(target_meta)),
                "n_observed": n_obs,
                "n_imputed": n_imp,
                "imputed_fraction": float(n_imp / max(len(target_meta), 1)),
            }
        )

    imputation_df = pd.DataFrame(imputation_rows).sort_values("subject")
    imputation_df.to_csv(tab_dir / "imputation_breakdown.csv", index=False)
    imputation_df.to_csv(tab_dir / "subject_parcel_coverage_after_fill.csv", index=False)

    # QC (stream-safe; avoid reloading giant all-gene CSVs).
    g_counts = imputation_df.set_index("subject")["n_total_parcels"].astype(int)
    n_rows_ahba_actual = int(len(ahba_export))
    n_rows_combined_actual = int(n_rows_ahba_actual + n_rows_gtex_actual)
    # Lightweight schema check from header only.
    with open(combined_out_path, newline="") as f:
        header_row = next(csv.reader(f))
    is_imputed_column_present = ("is_imputed" in header_row)
    qc = {
        "winner_model": winner_model,
        "n_genes": int(len(genes_all)),
        "n_subjects_gtex": int(g_counts.shape[0]),
        "n_parcels_expected": int(len(target_meta)),
        "n_rows_gtex_expected": int(g_counts.shape[0] * len(target_meta)),
        "n_rows_gtex_actual": int(n_rows_gtex_actual),
        "n_rows_ahba_actual": int(n_rows_ahba_actual),
        "n_rows_combined_actual": int(n_rows_combined_actual),
        "is_imputed_column_present": bool(is_imputed_column_present),
        "n_rows_is_imputed_1": int(n_rows_imputed_1),
        "n_rows_is_imputed_0": int(n_rows_ahba_actual + n_rows_imputed_0_gtex),
        "gtex_imputed_rows_expected": int(imputation_df["n_imputed"].sum()),
        "gtex_observed_rows_expected": int(imputation_df["n_observed"].sum()),
        "gtex_imputed_rows_actual": int(n_rows_imputed_1),
        "gtex_observed_rows_actual": int(n_rows_imputed_0_gtex),
        "missing_gene_cells_gtex": int(missing_gene_cells_gtex),
        "missing_gene_cells_combined": int(missing_gene_cells_gtex + int(np.isnan(ahba_export[genes_all].to_numpy(dtype=np.float64)).sum())),
        "all_subjects_have_150_rows": bool((g_counts == len(target_meta)).all()),
    }
    pd.DataFrame([qc]).to_csv(tab_dir / "export_qc_summary.csv", index=False)

    # Figure: observed vs imputed counts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    totals = {
        "Observed GTEx parcels": int(imputation_df["n_observed"].sum()),
        "Imputed GTEx parcels": int(imputation_df["n_imputed"].sum()),
    }
    axes[0].bar(list(totals.keys()), list(totals.values()), color=["#2a9d8f", "#e76f51"])
    axes[0].set_ylabel("Row count")
    axes[0].set_title("GTEx observed vs imputed rows")
    axes[0].tick_params(axis="x", rotation=20)

    sns.histplot(imputation_df["imputed_fraction"], bins=25, kde=True, ax=axes[1], color="#264653")
    axes[1].set_xlabel("Imputed fraction per subject")
    axes[1].set_title("Subject imputation-rate distribution")
    fig.savefig(fig_dir / "imputed_vs_observed_counts.png", dpi=220)
    plt.close(fig)

    # Build report.
    grid = pd.read_csv(winner_root / "tables" / "unified_model_grid_summary_hvg.csv")
    subj = pd.read_csv(winner_root / "tables" / "unified_subject_loro_summary_hvg.csv")
    wrow = grid[grid["model_combo"] == winner_model].iloc[0]

    lines = [
        "# Unified Winner Final Export Report",
        "",
        f"Generated: {io_utils.utc_timestamp()}",
        "",
        "## Objective",
        "Create final harmonized-space GTEx completion to 150 parcels per subject using the unified E/F winner,",
        "and package with AHBA in original CSV schema plus `is_imputed` flag.",
        "",
        "## Winner model",
        f"- `{winner_model}`",
        f"- Subjects evaluated (HVG LORO): `{int(wrow['n_subjects'])}`",
        f"- Mean Pearson: `{float(wrow['mean_pearson']):.4f}`",
        f"- Mean RMSE: `{float(wrow['mean_rmse']):.4f}`",
        f"- Gate pass: `{bool(wrow['gate_pass'])}`",
        "",
        "## Atlas-level filling",
        "- Used AHBA-derived 150 canonical parcels as target atlas.",
        "- Completed GTEx subject maps in harmonized space for all genes.",
        "",
        "## Subject-level performance context",
        f"- Subject-level rows in source evaluation: `{len(subj)}`",
        f"- Mean subject Pearson (winner): `{float(subj[subj['model_combo']==winner_model]['pearson_r'].mean()):.4f}`",
        f"- Mean subject RMSE (winner): `{float(subj[subj['model_combo']==winner_model]['rmse'].mean()):.4f}`",
        "",
        "## Export outputs",
        f"- Combined: `{combined_out_path}`",
        f"- GTEx-only: `{gtex_out_path}`",
        "",
        "## Imputation accounting",
        f"- GTEx rows observed: `{qc['gtex_observed_rows_actual']}`",
        f"- GTEx rows imputed: `{qc['gtex_imputed_rows_actual']}`",
        f"- Subjects with exactly 150 rows: `{qc['all_subjects_have_150_rows']}`",
        "",
        "## Schema",
        "- Original columns preserved with one insertion: `is_imputed` after `coordinates`.",
        "- `is_imputed=1` only for GTEx unobserved parcels filled by model.",
        "- `is_imputed=0` for observed GTEx parcels and all AHBA rows.",
        "",
        "## QC tables",
        "- `tables/export_qc_summary.csv`",
        "- `tables/subject_parcel_coverage_after_fill.csv`",
        "- `tables/imputation_breakdown.csv`",
        "",
        "## Figure",
        "- `figures/imputed_vs_observed_counts.png`",
    ]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")

    # PDF render best effort.
    try:
        subprocess.run(
            [
                "pandoc",
                str(out_root / "report.md"),
                "-o",
                str(out_root / "report.pdf"),
                "--pdf-engine=tectonic",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        pass

    manifest = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "winner_model": winner_model,
        "source_csv": str(csv_path),
        "winner_root": str(winner_root),
        "outputs": {
            "combined_csv": str(combined_out_path),
            "gtex_only_csv": str(gtex_out_path),
            "report_md": str(out_root / "report.md"),
            "report_pdf": str(out_root / "report.pdf"),
            "qc_table": str(tab_dir / "export_qc_summary.csv"),
        },
        "qc": qc,
    }
    io_utils.dump_json(out_root / "report_manifest.json", manifest)

    print(f"Completed unified winner all-gene export: {out_root}")


if __name__ == "__main__":
    main()
