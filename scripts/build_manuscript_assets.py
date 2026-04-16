#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflows import (
    build_manuscript_assets,
    load_dataset_bundle,
    load_workflow_config,
    run_allgene_loro,
    run_dlam,
    run_naive_fill,
    run_plam,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manuscript assets from the notebook-first workflow outputs.")
    p.add_argument("--config", default="configs/notebook_full.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_workflow_config(Path(args.config))
    bundle = load_dataset_bundle(
        str(cfg.get("csv_path", "data/raw/gxp_samples.csv")),
        str(cfg.get("hvg_path", "out/raw/gene_lists/ahba_100hvg.txt")),
        cfg,
    )
    model_results = {
        "naive": run_naive_fill(bundle, cfg),
        "dlam": run_dlam(bundle, cfg),
        "plam": run_plam(bundle, cfg),
    }
    eval_results = {
        "naive": run_allgene_loro(bundle, "naive", None, cfg),
        "dlam": run_allgene_loro(bundle, "dlam", None, cfg),
        "plam": run_allgene_loro(bundle, "plam", None, cfg),
    }
    assets = build_manuscript_assets(bundle, model_results, eval_results, cfg)
    print(f"Built manuscript assets under {assets.asset_root}")


if __name__ == "__main__":
    main()
