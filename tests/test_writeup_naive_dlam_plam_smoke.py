from __future__ import annotations

from pathlib import Path

from src.workflows import load_dataset_bundle, load_workflow_config, run_dlam, run_naive_fill, run_plam


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_naive_dlam_plam_smoke(tmp_path: Path) -> None:
    cfg = load_workflow_config(REPO_ROOT / 'configs' / 'notebook_smoke.yaml')
    cfg['out_root'] = str(tmp_path / 'out')
    bundle = load_dataset_bundle(cfg['csv_path'], cfg['hvg_path'], cfg)
    naive = run_naive_fill(bundle, cfg)
    dlam = run_dlam(bundle, cfg)
    plam = run_plam(bundle, cfg)
    for result in [naive, dlam, plam]:
        for path in result.outputs.values():
            assert Path(path).exists()
