#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified phase2 report")
    p.add_argument("--phase2-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_phase2")
    p.add_argument("--winner-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified")
    p.add_argument("--baseline-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment")
    return p.parse_args()


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    args = parse_args()
    p2 = Path(args.phase2_root).resolve()
    wr = Path(args.winner_root).resolve()
    br = Path(args.baseline_root).resolve()

    tab = p2 / "tables"
    fig = p2 / "figures"

    model = _safe_read(tab / "allgenes_model_summary.csv")
    subj = _safe_read(tab / "allgenes_subject_loro_summary.csv")
    unc = _safe_read(tab / "allgenes_uncertainty_calibration.csv")
    unc_cov = _safe_read(tab / "allgenes_uncertainty_by_coverage.csv")
    marker = _safe_read(tab / "marker_region_correlation.csv")
    sym = _safe_read(tab / "hemisphere_symmetry_checks.csv")
    outlier = _safe_read(tab / "regional_outlier_scan.csv")
    shift = _safe_read(tab / "imputed_observed_shift.csv")

    winner_sum = json.loads((wr / "summary.json").read_text()) if (wr / "summary.json").exists() else {}
    baseline_ref = _safe_read(br / "tables" / "model_grid_summary_hvg.csv")
    ref_row = baseline_ref[baseline_ref.get("model_combo", pd.Series(dtype=str)) == "combat__affine_gl3__constrained_anchor__rbf"] if len(baseline_ref) else pd.DataFrame()

    m = model.iloc[0].to_dict() if len(model) else {}

    # recommendation
    gate_pass = bool(m.get("gate_pass", False)) if len(model) else False
    rec_ge8 = "deploy_winner_with_uncertainty" if gate_pass else "hold"
    rec_lt8 = "conservative_abstain_or_low_confidence"

    lines = [
        "# Unified Winner Phase-II Report",
        "",
        f"Generated: {io_utils.utc_timestamp()}",
        "",
        "## Scope",
        "- Winner-only, all-gene subject-level LORO validation.",
        "- Uncertainty calibration diagnostics.",
        "- Biological plausibility audits.",
        "",
        "## All-gene LORO Summary",
    ]

    if len(model):
        lines += [
            f"- n_subjects: `{int(m['n_subjects'])}`",
            f"- mean_pearson: `{float(m['mean_pearson']):.4f}`",
            f"- median_pearson: `{float(m['median_pearson']):.4f}`",
            f"- mean_rmse: `{float(m['mean_rmse']):.4f}`",
            f"- mean_baseline_rmse: `{float(m['mean_baseline_rmse']):.4f}`",
            f"- frac_better_baseline_rmse: `{float(m['frac_better_baseline_rmse']):.4f}`",
            f"- gate_pass: `{bool(m['gate_pass'])}`",
            "",
        ]

    if len(ref_row):
        rr = ref_row.iloc[0]
        lines += [
            "## Context vs A–D Reference (HVG reference)",
            f"- Reference combo: `combat__affine_gl3__constrained_anchor__rbf`",
            f"- Reference mean_pearson: `{float(rr['mean_pearson']):.4f}`",
            f"- Reference mean_rmse: `{float(rr['mean_rmse']):.4f}`",
            "",
        ]

    lines += [
        "## Uncertainty Calibration",
        f"- calibration bins: `{len(unc)}`",
        f"- coverage bins: `{len(unc_cov)}`",
        "",
        "## Biological Audits",
        f"- marker genes evaluated: `{len(marker)}`",
        f"- marker pass fraction: `{float(marker['pass_marker'].mean()):.4f}`" if len(marker) else "- marker pass fraction: `nan`",
        f"- hemisphere symmetry pass fraction: `{float(sym['pass_symmetry'].mean()):.4f}`" if len(sym) else "- hemisphere symmetry pass fraction: `nan`",
        f"- outlier scan pass fraction: `{float(outlier['pass_outlier'].mean()):.4f}`" if len(outlier) else "- outlier scan pass fraction: `nan`",
        f"- imputed/observed shift pass fraction: `{float(shift['pass_shift'].mean()):.4f}`" if len(shift) else "- imputed/observed shift pass fraction: `nan`",
        "",
        "## Deployment Recommendation",
        f"- `n_obs >= 8`: `{rec_ge8}`",
        f"- `n_obs < 8`: `{rec_lt8}`",
        "",
        "## Artifacts",
        "### Tables",
        "- tables/allgenes_subject_loro_folds.csv",
        "- tables/allgenes_subject_loro_summary.csv",
        "- tables/allgenes_model_summary.csv",
        "- tables/allgenes_uncertainty_calibration.csv",
        "- tables/allgenes_uncertainty_by_coverage.csv",
        "- tables/marker_region_correlation.csv",
        "- tables/hemisphere_symmetry_checks.csv",
        "- tables/regional_outlier_scan.csv",
        "- tables/imputed_observed_shift.csv",
        "",
        "### Figures",
        "- figures/allgenes_model_ranking.png",
        "- figures/allgenes_coverage_vs_performance.png",
        "- figures/allgenes_uncertainty_reliability.png",
        "- figures/allgenes_uvar_vs_error.png",
        "- figures/marker_heatmap_ahba_vs_gtex.png",
        "- figures/hemisphere_symmetry_scatter.png",
        "- figures/imputed_observed_shift_violin.png",
    ]

    (p2 / "report_phase2.md").write_text("\n".join(lines) + "\n")

    try:
        subprocess.run(
            [
                "pandoc",
                str(p2 / "report_phase2.md"),
                "-o",
                str(p2 / "report_phase2.pdf"),
                "--pdf-engine=tectonic",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        pass

    io_utils.dump_json(
        p2 / "summary_phase2.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "winner_model": winner_sum.get("winner_model"),
            "allgenes_model_summary": model.iloc[0].to_dict() if len(model) else {},
            "bio_audit": {
                "marker_pass_fraction": float(marker["pass_marker"].mean()) if len(marker) else np.nan,
                "symmetry_pass_fraction": float(sym["pass_symmetry"].mean()) if len(sym) else np.nan,
                "outlier_pass_fraction": float(outlier["pass_outlier"].mean()) if len(outlier) else np.nan,
                "shift_pass_fraction": float(shift["pass_shift"].mean()) if len(shift) else np.nan,
            },
            "recommendation": {"ge8": rec_ge8, "lt8": rec_lt8},
        },
    )

    io_utils.dump_json(
        p2 / "report_phase2_manifest.json",
        {
            "timestamp_utc": io_utils.utc_timestamp(),
            "tables": sorted([p.name for p in (p2 / "tables").glob("*.csv")]) if (p2 / "tables").exists() else [],
            "figures": sorted([p.name for p in (p2 / "figures").glob("*.png")]) if (p2 / "figures").exists() else [],
            "report_md": str(p2 / "report_phase2.md"),
            "report_pdf": str(p2 / "report_phase2.pdf"),
        },
    )

    print(f"Built phase2 report: {p2}")


if __name__ == "__main__":
    main()
