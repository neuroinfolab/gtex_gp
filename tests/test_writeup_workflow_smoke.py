from __future__ import annotations

from pathlib import Path

from src.workflows import load_workflow_config, run_writeup_workflow


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_writeup_workflow_smoke(tmp_path: Path) -> None:
    cfg = load_workflow_config(REPO_ROOT / 'configs' / 'notebook_smoke.yaml')
    cfg['out_root'] = str(tmp_path / 'out')
    cfg['manuscript_root'] = str(tmp_path / 'manuscript')
    manifest = run_writeup_workflow(cfg)
    assert manifest['n_subjects_eligible'] == 2
    assert (tmp_path / 'out' / 'manifests' / 'run_manifest.json').exists()
