#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Build data/raw/gxp_samples.csv from AHBA and GTEx source files.")
    p.add_argument("--gtex-root", required=True)
    p.add_argument("--ahba-root", required=True)
    p.add_argument("--atlas-info-dir", default="data/metadata/atlas_info")
    p.add_argument("--gene-panel", default="data/metadata/gene_lists/allgenes_stability_dk/allgenes_stable_r0.2.csv")
    p.add_argument("--output", default="data/raw/gxp_samples.csv")
    p.add_argument("--expression-kind", default="tpm", choices=["tpm", "read_counts"])
    p.add_argument("--gtex-transform", default="log1p", choices=["log1p", "none"])
    p.add_argument("--gene-id-style", default="symbol", choices=["symbol", "ensembl"])
    p.add_argument("--parcellation", default="S156")
    p.add_argument("--ahba-processing", default="raw")
    p.add_argument("--gtex-sample-attributes-path")
    p.add_argument("--gtex-rin-threshold", type=float, default=6.0)
    p.add_argument("--gtex-tpm-threshold", type=float, default=0.1)
    p.add_argument("--gtex-reads-threshold", type=float, default=6.0)
    p.add_argument("--gtex-min-fraction", type=float, default=0.2)
    p.add_argument("--gtex-assay-freeze")
    p.add_argument("--gtex-metadata-path")
    p.add_argument("--ahba-metadata-path")
    args = p.parse_args()

    from src.samples import build_gxp_samples

    df = build_gxp_samples(
        gtex_root=args.gtex_root,
        ahba_root=args.ahba_root,
        atlas_info_dir=args.atlas_info_dir,
        gene_panel=args.gene_panel,
        gtex_sample_attributes_path=args.gtex_sample_attributes_path,
        gtex_rin_threshold=float(args.gtex_rin_threshold),
        gtex_tpm_threshold=float(args.gtex_tpm_threshold),
        gtex_reads_threshold=float(args.gtex_reads_threshold),
        gtex_min_fraction=float(args.gtex_min_fraction),
        gtex_assay_freeze=args.gtex_assay_freeze,
        output=args.output,
        expression_kind=args.expression_kind,
        gtex_transform=args.gtex_transform,
        gene_id_style=args.gene_id_style,
        parcellation=args.parcellation,
        ahba_processing=args.ahba_processing,
        gtex_metadata_path=args.gtex_metadata_path,
        ahba_metadata_path=args.ahba_metadata_path,
    )
    print(f"Wrote {Path(args.output)} with shape={df.shape}")


if __name__ == "__main__":
    main()
