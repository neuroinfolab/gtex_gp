from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_smoke_execution(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'smoke.yaml'
    cfg_path.write_text((REPO_ROOT / 'configs' / 'notebook_smoke.yaml').read_text().replace('out/notebook_writeup_smoke', str(tmp_path / 'out')).replace('docs/manuscript', str(tmp_path / 'manuscript')))
    cmd = ['python3', str(REPO_ROOT / 'notebooks' / 'ahba_gtex_writeup_end_to_end.py'), '--config', str(cfg_path)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    assert (tmp_path / 'out' / 'manifests' / 'run_manifest.json').exists()
