"""
Approach 3 — Pure matplotlib with bubble-map style and brain silhouette.

Uses MNI coordinate projections + proportional bubble size (all equal here)
+ a hand-drawn oval silhouette to give brain context. The silhouette is drawn
as an ellipse patch; no external atlas image needed.

Same 3×4 layout:
  Rows:    Sagittal (x-view) | Coronal (y-view) | Axial (z-view)
  Cols:    GTEx Input | Reconstruction | Whole-Brain Fit | AHBA Reference

Output: outputs/matplotlib/<subject>_<model>_<gene>.png
Run:    python viz_matplotlib.py [--gene GENE] [--subjects S1 S2 ...] [--models M1 M2 ...]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

import viz_data

OUT_DIR = Path(__file__).parent / 'outputs' / 'matplotlib'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (view_name, x_col, y_col, x_label, y_label, silhouette_wh)
# silhouette_wh: (half-width, half-height) in mm for the oval brain outline
VIEWS = [
    ('Sagittal', 1, 2, 'Y (mm)', 'Z (mm)', (90, 80)),
    ('Coronal',  0, 2, 'X (mm)', 'Z (mm)', (80, 80)),
    ('Axial',    0, 1, 'X (mm)', 'Y (mm)', (80, 75)),
]
COL_TITLES = ['GTEx Input', 'Reconstruction', 'Whole-Brain Fit', 'AHBA Reference']
CMAP       = 'viridis'
SCATTER_S  = 110
ALPHA      = 0.88
EC         = (0.2, 0.2, 0.2, 0.5)
LINEWIDTH  = 0.5


def _draw_silhouette(ax, hw: float, hh: float):
    """Draw a simple elliptical brain outline."""
    ell = mpatches.Ellipse(
        (0, 0), width=2 * hw, height=2 * hh,
        linewidth=1.2, edgecolor='#bbb', facecolor='#f8f8f8',
        zorder=0,
    )
    ax.add_patch(ell)
    ax.set_xlim(-hw * 1.15, hw * 1.15)
    ax.set_ylim(-hh * 1.15, hh * 1.15)
    ax.set_aspect('equal', adjustable='box')


def _make_figure(data: dict, subject: str, model: str, gene: str) -> plt.Figure:
    coords = data['coords']
    panels = [data['v1'], data['v2'], data['v3'], data['v4']]
    vmin, vmax = data['vmin'], data['vmax']
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_ = matplotlib.colormaps[CMAP]

    fig, axes = plt.subplots(
        3, 4,
        figsize=(18, 12),
        gridspec_kw={'hspace': 0.18, 'wspace': 0.12},
    )
    fig.patch.set_facecolor('white')

    for row, (view_name, xi, yi, xl, yl, sil) in enumerate(VIEWS):
        for col, vals in enumerate(panels):
            ax = axes[row, col]

            _draw_silhouette(ax, *sil)

            valid = ~np.isnan(vals)
            x_plot = coords[valid, xi]
            y_plot = coords[valid, yi]
            v_plot = vals[valid]
            colors = cmap_(norm(v_plot))

            sc = ax.scatter(
                x_plot, y_plot,
                c=colors,
                s=SCATTER_S,
                alpha=ALPHA,
                edgecolors=EC,
                linewidths=LINEWIDTH,
                zorder=2,
            )

            ax.set_xlabel(xl, fontsize=7.5)
            ax.set_ylabel(yl, fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.grid(True, alpha=0.25, zorder=1)

            n_shown = int(valid.sum())
            ax.set_title(
                f'{COL_TITLES[col]}  (n={n_shown})' if row == 0
                else f'n={n_shown}',
                fontsize=8.5 if row == 0 else 7,
                fontweight='bold' if row == 0 else 'normal',
                pad=4,
            )

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(f'{view_name}\n{yl}', fontsize=8.5, labelpad=4)

    # Shared colour bar on the right
    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.68])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('log₂ expression', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'{subject}  ·  model={model}  ·  gene={gene}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    return fig


def run(gene: str, subjects: list[str], models: list[str]):
    for subject in subjects:
        for model in models:
            print(f'  matplotlib  {subject} / {model} / {gene}')
            data = viz_data.get_panel_data(subject, model, gene)
            fig  = _make_figure(data, subject, model, gene)
            out  = OUT_DIR / f'{subject}_{model}_{gene}.png'
            fig.savefig(out, dpi=130, bbox_inches='tight',
                        facecolor='white', pad_inches=0.15)
            plt.close(fig)
            print(f'    → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gene',     default='DPM1')
    p.add_argument('--subjects', nargs='+', default=viz_data.SUBJECTS)
    p.add_argument('--models',   nargs='+', default=viz_data.MODELS)
    args = p.parse_args()
    run(args.gene, args.subjects, args.models)
