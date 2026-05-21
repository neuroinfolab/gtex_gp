"""
nilearn glass-brain markers — GTEx Input vs Reconstruction side-by-side.

Layout: 3 rows × 2 columns
  Rows:  Sagittal (x) | Coronal (y) | Axial (z)
  Cols:  GTEx Input   | Reconstruction
  Scale: shared between both columns so colours are directly comparable.

Output: outputs/v1_nilearn_folder/<subject>_<model>_<gene>.png
Run:    python viz_nilearn.py [--gene GENE] [--subjects S1 S2 ...] [--models M1 M2 ...]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from nilearn import plotting

import viz_data

OUT_DIR = Path(__file__).parent / 'outputs' / 'v1_nilearn_folder'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS       = [('x', 'Sagittal'), ('y', 'Coronal'), ('z', 'Axial')]
COL_TITLES  = ['GTEx Input', 'Reconstruction']
CMAP        = 'viridis'
MARKER_SIZE = 60


def _make_figure(data: dict, subject: str, model: str, gene: str) -> plt.Figure:
    coords = data['coords']
    panels = [data['v1'], data['v2']]

    # Shared scale across both panels
    all_vals = np.concatenate([v[~np.isnan(v)] for v in panels])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[CMAP]

    fig, axes = plt.subplots(
        3, 2, figsize=(10, 11),
        gridspec_kw={'hspace': 0.08, 'wspace': 0.06},
    )
    fig.patch.set_facecolor('white')

    for row, (display_mode, row_label) in enumerate(VIEWS):
        for col, vals in enumerate(panels):
            ax = axes[row, col]

            valid  = ~np.isnan(vals)
            c_plot = coords[valid]
            v_plot = vals[valid]
            colors = [mcolors.to_hex(cmap(norm(v))) for v in v_plot]

            display = plotting.plot_glass_brain(
                None,
                display_mode=display_mode,
                axes=ax,
                colorbar=False,
                alpha=0.15,
                annotate=False,
            )
            display.add_markers(c_plot, marker_color=colors, marker_size=MARKER_SIZE)

            if row == 0:
                n = int(valid.sum())
                ax.set_title(f'{COL_TITLES[col]}  (n={n})',
                             fontsize=11, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=4)

    # Shared colorbar
    sm = matplotlib.cm.ScalarMappable(
        cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label('log₂ expression', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    n_v1 = int((~np.isnan(data['v1'])).sum())
    n_v2 = int((~np.isnan(data['v2'])).sum())
    fig.suptitle(
        f'{subject}  ·  model={model}  ·  gene={gene}\n'
        f'shared scale  [{vmin:.2f}, {vmax:.2f}]  '
        f'(GTEx n={n_v1}, Recon n={n_v2})',
        fontsize=11, fontweight='bold', y=1.02,
    )
    return fig


def run(gene: str, subjects: list[str], models: list[str]):
    for subject in subjects:
        for model in models:
            print(f'  nilearn  {subject} / {model} / {gene}')
            data = viz_data.get_panel_data(subject, model, gene)
            fig  = _make_figure(data, subject, model, gene)
            out  = OUT_DIR / f'{subject}_{model}_{gene}.png'
            fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f'    → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gene',     default='DPM1')
    p.add_argument('--subjects', nargs='+', default=viz_data.SUBJECTS)
    p.add_argument('--models',   nargs='+', default=viz_data.MODELS)
    args = p.parse_args()
    run(args.gene, args.subjects, args.models)
