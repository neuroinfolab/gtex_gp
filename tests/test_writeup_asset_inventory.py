from __future__ import annotations

from pathlib import Path

from src.workflows import (
    build_manuscript_assets,
    load_dataset_bundle,
    load_workflow_config,
    run_allgene_loro,
    run_dlam,
    run_naive_fill,
    run_plam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_asset_inventory(tmp_path: Path) -> None:
    cfg = load_workflow_config(REPO_ROOT / 'configs' / 'notebook_smoke.yaml')
    cfg['out_root'] = str(tmp_path / 'out')
    cfg['manuscript_root'] = str(tmp_path / 'manuscript')
    bundle = load_dataset_bundle(cfg['csv_path'], cfg['hvg_path'], cfg)
    results = {
        'naive': run_naive_fill(bundle, cfg),
        'dlam': run_dlam(bundle, cfg),
        'plam': run_plam(bundle, cfg),
    }
    evals = {
        'naive': run_allgene_loro(bundle, 'naive', None, cfg),
        'dlam': run_allgene_loro(bundle, 'dlam', None, cfg),
        'plam': run_allgene_loro(bundle, 'plam', None, cfg),
    }
    assets = build_manuscript_assets(bundle, results, evals, cfg)
    assert assets.figure_paths
    assert assets.table_paths
    assert (Path(assets.manuscript_fig_dir) / 'atlas_deterministic_heatmap_demo.pdf').exists()
    assert (Path(assets.manuscript_table_dir) / 'parcel_legend_top10.csv').exists()
