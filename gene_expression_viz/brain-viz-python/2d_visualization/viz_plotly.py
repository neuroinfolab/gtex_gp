"""
Approach 2 — Plotly static scatter panels (exported as PNG via kaleido).

Same 3×4 layout as viz_nilearn.py but rendered with Plotly's make_subplots.
Each panel projects parcel MNI centroids onto the 2D plane for that view:
  sagittal (x-view) → (y, z)
  coronal  (y-view) → (x, z)
  axial    (z-view) → (x, y)

Output: outputs/plotly/<subject>_<model>_<gene>.png
Run:    python viz_plotly.py [--gene GENE] [--subjects S1 S2 ...] [--models M1 M2 ...]

Requires: pip install kaleido
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import viz_data

OUT_DIR = Path(__file__).parent / 'outputs' / 'plotly'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (axis_name, x_col, y_col, x_label, y_label)
VIEWS = [
    ('Sagittal', 1, 2, 'Y (mm)', 'Z (mm)'),
    ('Coronal',  0, 2, 'X (mm)', 'Z (mm)'),
    ('Axial',    0, 1, 'X (mm)', 'Y (mm)'),
]
COL_TITLES = ['GTEx Input', 'Reconstruction', 'Whole-Brain Fit', 'AHBA Reference']
CMAP = 'Viridis'
MARKER_SIZE = 9


def _make_figure(data: dict, subject: str, model: str, gene: str) -> go.Figure:
    coords = data['coords']
    panels = [data['v1'], data['v2'], data['v3'], data['v4']]
    vmin, vmax = data['vmin'], data['vmax']

    subplot_titles = [
        f'<b>{ct}</b>' for ct in COL_TITLES
    ] * 3  # repeated for each row — plotly flattens row-major

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=subplot_titles[:4],   # only top row gets titles via this param
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    # Add column headers manually (plotly only supports one row of titles)
    for col_i, ct in enumerate(COL_TITLES):
        fig.add_annotation(
            text=f'<b>{ct}</b>',
            xref='paper', yref='paper',
            x=(col_i + 0.5) / 4, y=1.04,
            showarrow=False, font=dict(size=12),
        )

    for row, (view_name, xi, yi, xl, yl) in enumerate(VIEWS):
        for col, vals in enumerate(panels):
            valid = ~np.isnan(vals)
            x = coords[valid, xi]
            y = coords[valid, yi]
            v = vals[valid]

            trace = go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    size=MARKER_SIZE,
                    color=v,
                    colorscale=CMAP,
                    cmin=vmin, cmax=vmax,
                    showscale=(col == 3 and row == 1),
                    colorbar=dict(
                        title='log₂ expr',
                        thickness=14,
                        len=0.5,
                        y=0.5,
                        titlefont=dict(size=10),
                        tickfont=dict(size=9),
                    ) if (col == 3 and row == 1) else None,
                    line=dict(width=0.4, color='rgba(0,0,0,0.25)'),
                ),
                showlegend=False,
                hovertemplate='%{text}<br>val=%{marker.color:.3f}',
                text=[data['labels'][i] for i, ok in enumerate(valid) if ok],
            )
            fig.add_trace(trace, row=row + 1, col=col + 1)

            # Row label on leftmost column
            if col == 0:
                fig.update_yaxes(title_text=f'<b>{view_name}</b><br>{yl}',
                                 title_font=dict(size=10),
                                 row=row + 1, col=1)
            if row == 2:
                fig.update_xaxes(title_text=xl, title_font=dict(size=9),
                                 row=row + 1, col=col + 1)

    fig.update_layout(
        width=1600, height=1000,
        paper_bgcolor='white',
        plot_bgcolor='#f5f5f5',
        title=dict(
            text=f'{subject}  ·  model={model}  ·  gene={gene}',
            font=dict(size=14),
            x=0.5,
        ),
        margin=dict(t=80, b=40, l=60, r=80),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#ddd', zeroline=True,
                     zerolinecolor='#aaa', zerolinewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='#ddd', zeroline=True,
                     zerolinecolor='#aaa', zerolinewidth=1,
                     scaleanchor=None)
    return fig


def run(gene: str, subjects: list[str], models: list[str]):
    for subject in subjects:
        for model in models:
            print(f'  plotly  {subject} / {model} / {gene}')
            data = viz_data.get_panel_data(subject, model, gene)
            fig  = _make_figure(data, subject, model, gene)
            out  = OUT_DIR / f'{subject}_{model}_{gene}.png'
            fig.write_image(str(out), scale=1.8)
            print(f'    → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gene',     default='DPM1')
    p.add_argument('--subjects', nargs='+', default=viz_data.SUBJECTS)
    p.add_argument('--models',   nargs='+', default=viz_data.MODELS)
    args = p.parse_args()
    run(args.gene, args.subjects, args.models)
