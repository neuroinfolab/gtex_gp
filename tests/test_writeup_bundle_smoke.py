from __future__ import annotations

from pathlib import Path

from src.workflows import load_dataset_bundle, load_workflow_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_writeup_bundle_smoke(tmp_path: Path) -> None:
    cfg = load_workflow_config(REPO_ROOT / 'configs' / 'notebook_smoke.yaml')
    cfg['out_root'] = str(tmp_path / 'out')
    bundle = load_dataset_bundle(cfg['csv_path'], cfg['hvg_path'], cfg)
    assert bundle.counts['n_subjects_eligible'] == 2
    assert bundle.counts['n_genes_hvg'] == 5
    assert bundle.target_meta.shape[0] == 6
    assert len(bundle.eligible_subjects) == 2
