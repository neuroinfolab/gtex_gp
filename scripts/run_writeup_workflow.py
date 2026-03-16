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

from src.workflows import load_workflow_config, run_writeup_workflow


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notebook-first end-to-end workflow for AHBA-GTEx write-up assets.")
    p.add_argument("--config", default="configs/notebook_full.yaml")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--hvg-path", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--manuscript-root", default=None)
    p.add_argument("--smoke-subjects", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--write-manuscript-assets", default=None)
    p.add_argument("--force-stage", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_workflow_config(Path(args.config))
    if args.csv_path is not None:
        cfg["csv_path"] = args.csv_path
    if args.hvg_path is not None:
        cfg["hvg_path"] = args.hvg_path
    if args.out_root is not None:
        cfg["out_root"] = args.out_root
    if args.manuscript_root is not None:
        cfg["manuscript_root"] = args.manuscript_root
    if args.smoke_subjects is not None:
        cfg["smoke_subjects"] = int(args.smoke_subjects)
    if args.chunk_size is not None:
        cfg["chunk_size"] = int(args.chunk_size)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.write_manuscript_assets is not None:
        cfg["write_manuscript_assets"] = _parse_bool(args.write_manuscript_assets)
    if args.force_stage:
        cfg["force_rerun_stages"] = list(dict.fromkeys([*(cfg.get("force_rerun_stages", []) or []), *args.force_stage]))

    manifest = run_writeup_workflow(cfg)
    print(f"Completed notebook-first workflow. Manifest: {Path(cfg.get('out_root', 'out/notebook_writeup')) / 'manifests' / 'run_manifest.json'}")
    print(f"Representative subject: {manifest.get('representative_subject', '')}")


if __name__ == "__main__":
    main()
