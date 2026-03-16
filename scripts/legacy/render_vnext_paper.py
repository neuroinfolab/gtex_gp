#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils


def parse_args():
    p = argparse.ArgumentParser(description="Render vNext paper markdown to PDF and TeX with citations.")
    p.add_argument("--paper-root", default="out/vnext_alignment_allgenes/paper_vnext")
    p.add_argument("--markdown", default="manuscript_vnext.md")
    p.add_argument("--bib", default="references.bib")
    p.add_argument("--pdf", default="manuscript_vnext.pdf")
    p.add_argument("--tex", default="manuscript_vnext.tex")
    p.add_argument("--csl", default="vancouver.csl")
    return p.parse_args()


def _ensure_csl(csl_path: Path) -> None:
    if csl_path.exists():
        return
    url = "https://www.zotero.org/styles/vancouver"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read()
    csl_path.write_bytes(data)


def _run(cmd, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> None:
    a = parse_args()
    root = Path(".").resolve()
    paper_root = (root / a.paper_root).resolve()
    md = paper_root / a.markdown
    bib = paper_root / a.bib
    pdf = paper_root / a.pdf
    tex = paper_root / a.tex
    csl = paper_root / a.csl

    if not md.exists():
        raise FileNotFoundError(f"Markdown not found: {md}")
    if not bib.exists():
        raise FileNotFoundError(f"Bibliography not found: {bib}")

    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found in PATH")
    if shutil.which("tectonic") is None:
        raise RuntimeError("tectonic not found in PATH")

    _ensure_csl(csl)

    base = [
        "pandoc",
        str(md.name),
        "--standalone",
        "--from",
        "markdown+tex_math_dollars",
        "--citeproc",
        "--bibliography",
        str(bib.name),
        "--csl",
        str(csl.name),
        "--resource-path",
        str(paper_root),
    ]

    _run(base + ["-t", "latex", "-o", str(tex.name)], cwd=paper_root)
    _run(base + ["--pdf-engine=tectonic", "-o", str(pdf.name)], cwd=paper_root)

    manifest_path = paper_root / "render_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
    manifest.update(
        {
            "render_timestamp_utc": io_utils.utc_timestamp(),
            "render_command_pdf": f"pandoc {md.name} --standalone --from markdown+tex_math_dollars --citeproc --bibliography {bib.name} --csl {csl.name} --pdf-engine=tectonic -o {pdf.name}",
            "render_command_tex": f"pandoc {md.name} --standalone --from markdown+tex_math_dollars --citeproc --bibliography {bib.name} --csl {csl.name} -t latex -o {tex.name}",
            "render_outputs": {"pdf": str(pdf), "tex": str(tex), "csl": str(csl)},
        }
    )
    io_utils.dump_json(manifest_path, manifest)
    print(f"Rendered PDF: {pdf}")


if __name__ == "__main__":
    main()
