# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AHBA-GTEx End-to-End Write-up Workflow
#
# This notebook is the canonical interactive entrypoint for the current repository.
# It starts from a raw `gxp_samples.csv` file, runs the shared preprocessing and the
# three manuscript comparators (naive fill, DLAM, PLAM), performs the all-gene
# evaluation workflow, and builds manuscript-facing tables and figures.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    cwd = Path.cwd().resolve()
    REPO_ROOT = cwd.parent if cwd.name == "notebooks" else cwd

import sys
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
    select_representative_subject,
    summarize_shared_harmonization,
    write_run_manifest,
)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "notebook_full.yaml"))
    return p.parse_known_args()[0]


ARGS = _parse_cli() if "__file__" in globals() else argparse.Namespace(config=str(REPO_ROOT / "configs" / "notebook_full.yaml"))
CONFIG_PATH = Path(ARGS.config)
CFG = load_workflow_config(CONFIG_PATH)
if not Path(str(CFG.get("csv_path", ""))).is_absolute():
    CFG["csv_path"] = str((REPO_ROOT / str(CFG.get("csv_path"))).resolve())
if not Path(str(CFG.get("hvg_path", ""))).is_absolute():
    CFG["hvg_path"] = str((REPO_ROOT / str(CFG.get("hvg_path"))).resolve())
if not Path(str(CFG.get("out_root", ""))).is_absolute():
    CFG["out_root"] = str((REPO_ROOT / str(CFG.get("out_root"))).resolve())
if not Path(str(CFG.get("manuscript_root", ""))).is_absolute():
    CFG["manuscript_root"] = str((REPO_ROOT / str(CFG.get("manuscript_root"))).resolve())

print("Config:")
print(json.dumps({k: v for k, v in CFG.items() if k != "force_rerun_stages"}, indent=2))

# %% [markdown]
# ## 1. Load and validate the dataset bundle
#
# This stage loads the HVG slice used for manuscript visual diagnostics, builds the
# canonical parcel geometry, and determines the eligible GTEx subject set.

# %%
BUNDLE = load_dataset_bundle(CFG["csv_path"], CFG["hvg_path"], CFG)
summary_df = pd.DataFrame([BUNDLE.counts])
summary_df

# %% [markdown]
# ## 2. Shared harmonization summary
#
# Both DLAM and PLAM operate on the same harmonized AHBA-GTEx domain before their
# model-specific alignment stages diverge.

# %%
HARMONIZATION = summarize_shared_harmonization(BUNDLE, CFG)
pd.DataFrame([HARMONIZATION["summary"]])

# %% [markdown]
# ## 3. Naive atlas-fill baseline
#
# The naive baseline preserves observed GTEx parcels and copies the harmonized AHBA
# atlas value into unobserved parcels.

# %%
NAIVE_RESULT = run_naive_fill(BUNDLE, CFG)
pd.DataFrame([NAIVE_RESULT.summary])

# %% [markdown]
# ## 4. DLAM
#
# DLAM fits a subject-specific deterministic latent alignment and then interpolates
# aligned latent spatial scores before decoding back to gene space.

# %%
DLAM_RESULT = run_dlam(BUNDLE, CFG)
pd.DataFrame([DLAM_RESULT.summary])

# %% [markdown]
# ## 5. PLAM
#
# PLAM fits a global atlas latent geometry and infers subject-specific aligned latent
# deviation fields under the probabilistic model.

# %%
PLAM_RESULT = run_plam(BUNDLE, CFG)
pd.DataFrame([PLAM_RESULT.summary])

# %% [markdown]
# ## 6. All-gene LORO evaluation
#
# The manuscript-facing quantitative evaluation is all-gene and compares the naive
# baseline, DLAM, and PLAM directly.

# %%
EVAL_NAIVE = run_allgene_loro(BUNDLE, "naive", None, CFG)
EVAL_DLAM = run_allgene_loro(BUNDLE, "dlam", None, CFG)
EVAL_PLAM = run_allgene_loro(BUNDLE, "plam", None, CFG)
EVAL_RESULTS = {"naive": EVAL_NAIVE, "dlam": EVAL_DLAM, "plam": EVAL_PLAM}

summary_table = pd.read_csv(EVAL_PLAM.tables["summary_csv"])
summary_table

# %% [markdown]
# ## 7. Representative subject
#
# The same representative subject is used across DLAM, PLAM, and naive-fill subject
# figures so the comparison stays controlled.

# %%
REPRESENTATIVE_SUBJECT = select_representative_subject(EVAL_RESULTS, CFG)
pd.DataFrame([
    {
        "subject": REPRESENTATIVE_SUBJECT.subject,
        "n_obs_parcels": REPRESENTATIVE_SUBJECT.n_obs_parcels,
        "selection_rule": REPRESENTATIVE_SUBJECT.selection_rule,
        **REPRESENTATIVE_SUBJECT.metadata,
    }
])

# %% [markdown]
# ## 8. Manuscript assets
#
# This stage reuses the current manuscript figure builders after materializing the
# compatibility artifacts expected by those builders.

# %%
ASSETS = build_manuscript_assets(
    BUNDLE,
    {"naive": NAIVE_RESULT, "dlam": DLAM_RESULT, "plam": PLAM_RESULT},
    EVAL_RESULTS,
    CFG,
)
pd.DataFrame(
    {
        "kind": ["figures", "tables"],
        "count": [len(ASSETS.figure_paths), len(ASSETS.table_paths)],
        "target": [ASSETS.manuscript_fig_dir, ASSETS.manuscript_table_dir],
    }
)

# %% [markdown]
# ## 9. Final manifest
#
# The manifest records the config, stage outputs, and the generated manuscript asset
# inventory.

# %%
RUN_MANIFEST = write_run_manifest(
    BUNDLE,
    {"naive": NAIVE_RESULT, "dlam": DLAM_RESULT, "plam": PLAM_RESULT},
    EVAL_RESULTS,
    ASSETS,
    CFG,
)
pd.DataFrame([
    {
        "config_hash": RUN_MANIFEST["config_hash"],
        "timestamp": RUN_MANIFEST["timestamp"],
        "n_subjects_eligible": RUN_MANIFEST["n_subjects_eligible"],
        "n_genes_all": RUN_MANIFEST["n_genes_all"],
        "n_genes_hvg": RUN_MANIFEST["n_genes_hvg"],
        "representative_subject": REPRESENTATIVE_SUBJECT.subject,
    }
])

# %% [markdown]
# ## 10. Key output locations

# %%
print("Run manifest:", Path(BUNDLE.out_root) / "manifests" / "run_manifest.json")
print("Evaluation figures:", Path(EVAL_PLAM.asset_root) / "figures")
print("Evaluation tables:", Path(EVAL_PLAM.asset_root) / "tables")
print("Manuscript figure dir:", ASSETS.manuscript_fig_dir)
print("Manuscript table dir:", ASSETS.manuscript_table_dir)
