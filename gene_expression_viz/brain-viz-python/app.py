"""
Brain Gene Expression Explorer — V1 | V2 | V3  (side-by-side, shared camera)
yabplot + PyVista + trame interactive prototype

Run:  python app.py
Open: http://localhost:1234

Prerequisites:
  - Run build_atlas.py at least once (subcortical works without wb_command;
    cortical requires wb_command — see build_atlas.py for install instructions).
"""

import base64
import glob
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as matplotlib_cm
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk

# ── Python 3.13 shutdown fix ───────────────────────────────────────────────────
# During interpreter teardown, Python sets module globals to None before the GC
# runs __del__ on remaining PolyData objects.  PyVista's __del__ then crashes
# accessing _vtk_utilities attributes.  Python already ignores __del__ exceptions
# (prints them), so catching them here is equivalent — just without the spam.
if hasattr(pv.PolyData, '__del__'):
    _pv_polydata_orig_del = pv.PolyData.__del__
    def _pv_polydata_safe_del(self):
        try:
            _pv_polydata_orig_del(self)
        except Exception:
            pass
    pv.PolyData.__del__ = _pv_polydata_safe_del
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, html
from pyvista.trame.ui import plotter_ui

from yabplot.data import get_surface_paths
from yabplot.utils import load_gii, parse_lut
from yabplot.mesh import map_values_to_surface

import data_loader

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE            = Path(__file__).parent
ATLAS_CACHE     = HERE / 'atlas_cache'
CORTICAL_DIR    = ATLAS_CACHE / 'cortical'
SUBCORTICAL_DIR = ATLAS_CACHE / 'subcortical'

# ── Rendering config ───────────────────────────────────────────────────────────
CMAP            = 'viridis'
CMAP_DIV        = 'PuOr'
NAN_COLOR       = (0.55, 0.55, 0.55)
CORTEX_ALPHA    = 0.92
SUBCORTEX_ALPHA = 1.0
ZOOM_SENSITIVITY = 0.1

# ── 1. Load alignment ──────────────────────────────────────────────────────────
print("Loading atlas alignment...")
labels, atlas_aligned = data_loader.load_alignment()
rh_to_lh_map   = data_loader.build_rh_to_lh_map(atlas_aligned)
excluded_map   = data_loader.build_excluded_parcel_map()
print(f"  RH→LH parcel map: {len(rh_to_lh_map)} entries")
print(f"  Excluded parcels with counterparts: {len(excluded_map)} ({list(excluded_map)})")

# MNI centroid for every parcel — used to place 3D region labels in the scene
_centroid_lookup: dict = (
    atlas_aligned.set_index('label')[['mni_x', 'mni_y', 'mni_z']]
    .to_dict('index')
)


# ── 2. Load subject data ───────────────────────────────────────────────────────
INIT_MODEL   = 'naive'
INIT_SUBJECT = 'GTEX-13OW8'
SUBJECT_LIST = list(data_loader.SUBJECT_BUNDLES.keys())
print(f"Loading subject data ({INIT_MODEL}/{INIT_SUBJECT})...")
_MODEL_LIST = ['naive', 'dlam', 'plam']
_all_npz: dict = {m: data_loader.load_subject(m, INIT_SUBJECT) for m in _MODEL_LIST}
npz_data = _all_npz[INIT_MODEL]
gene_names_all = npz_data['gene_names'].tolist()

# ── 3. Gene list ───────────────────────────────────────────────────────────────
try:
    gene_list = data_loader.load_gene_list('ahba_100hvg')
    gene_list = [g for g in gene_list if g in gene_names_all]
except FileNotFoundError:
    gene_list = gene_names_all[:50]
if not gene_list:
    gene_list = gene_names_all[:20]
INIT_GENE = gene_list[0]
print(f"Gene list: {len(gene_list)} genes, starting with '{INIT_GENE}'")

# ── 4. Cortical surfaces (fsLR32k midthickness) ────────────────────────────────
print("Loading cortical surfaces from yabplot...")
lh_path, rh_path = get_surface_paths('midthickness', 'bmesh')
lh_v, lh_f = load_gii(lh_path)
rh_v, rh_f = load_gii(rh_path)


def _to_pv_faces(f: np.ndarray) -> np.ndarray:
    return np.column_stack([np.full(len(f), 3), f]).astype(np.intp).flatten()


LH_FACES = _to_pv_faces(lh_f)
RH_FACES = _to_pv_faces(rh_f)

# ── 5. Cortical atlas vertex map ───────────────────────────────────────────────
cortical_atlas_available = (
    (CORTICAL_DIR / 'atlas.csv').exists() and
    (CORTICAL_DIR / 'atlas.txt').exists()
)
if cortical_atlas_available:
    print("Loading cortical atlas vertex map...")
    tar_labels = np.loadtxt(str(CORTICAL_DIR / 'atlas.csv'), dtype=int)
    lut_ids, _, lut_names, _ = parse_lut(str(CORTICAL_DIR / 'atlas.txt'))
else:
    print(
        "WARNING: Cortical atlas not built yet.\n"
        "  Run: python build_atlas.py (needs wb_command installed first).\n"
        "  Cortex will render grey until then."
    )
    tar_labels = lut_ids = lut_names = None

# ── 6. Subcortical VTK meshes ──────────────────────────────────────────────────
print("Loading subcortical meshes...")
vtk_files = sorted(glob.glob(str(SUBCORTICAL_DIR / '*.vtk')))
subcortical_meshes: dict[str, pv.PolyData] = {}
for vf in vtk_files:
    name = Path(vf).stem
    subcortical_meshes[name] = pv.read(vf)
print(f"  Loaded {len(subcortical_meshes)} subcortical meshes.")

# ── 7. Colourmap helpers ───────────────────────────────────────────────────────
_cmap_fn = matplotlib_cm.get_cmap(CMAP)
_GRADIENT_CSS = 'linear-gradient(to right, ' + ', '.join(
    '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    for r, g, b, _ in [_cmap_fn(i / 9) for i in range(10)]
) + ')'
_LOW_HEX  = '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in _cmap_fn(0.0)[:3]])
_HIGH_HEX = '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in _cmap_fn(1.0)[:3]])

_cmap_div_fn = matplotlib_cm.get_cmap(CMAP_DIV)
_GRADIENT_DIV_CSS = 'linear-gradient(to right, ' + ', '.join(
    '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    for r, g, b, _ in [_cmap_div_fn(i / 9) for i in range(10)]
) + ')'
_DIV_LOW_HEX  = '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in _cmap_div_fn(0.0)[:3]])
_DIV_HIGH_HEX = '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in _cmap_div_fn(1.0)[:3]])


def _build_vtk_lut(cmap_name: str, n: int = 256) -> vtk.vtkLookupTable:
    fn = matplotlib_cm.get_cmap(cmap_name)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.Build()
    for i in range(n):
        r, g, b, a = fn(i / (n - 1))
        lut.SetTableValue(i, r, g, b, a)
    # Match PyVista's nan_color so unmapped/NaN parcels stay grey, not VTK's
    # default dark-red NaN color (0.5, 0.0, 0.0).
    lut.SetNanColor(*NAN_COLOR, 1.0)
    return lut


def _set_viewport_lut(lh_actor, rh_actor, sub_actors: dict, lut: vtk.vtkLookupTable):
    for actor in [lh_actor, rh_actor] + [a for _, a in sub_actors.values()]:
        actor.GetMapper().SetLookupTable(lut)


_lut_seq = _build_vtk_lut(CMAP)
_lut_div = _build_vtk_lut(CMAP_DIV)


def _clim_from_dicts(*dict_pairs) -> list:
    """Compute shared [vmin, vmax] across one or more (cortex_dict, subcortex_dict) pairs."""
    all_vals = []
    for c, s in dict_pairs:
        all_vals.extend(c.values())
        all_vals.extend(s.values())
    if not all_vals:
        return [0.0, 1.0]
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return [vmin, vmax]


def _symmetric_clim(c_dict: dict, s_dict: dict) -> list:
    """Return [-max_abs, max_abs] so that 0 is always the colormap midpoint."""
    all_vals = list(c_dict.values()) + list(s_dict.values())
    if not all_vals:
        return [-1.0, 1.0]
    max_abs = max((abs(v) for v in all_vals if not np.isnan(v)), default=1.0) or 1.0
    return [-max_abs, max_abs]


def _shared_clim(gene: str) -> list:
    """Joint V3 clim across all three models — stable when switching models."""
    pairs = []
    for npz in _all_npz.values():
        c3, s3 = data_loader.v4_fullfit(npz, atlas_aligned, gene)
        c3, s3 = data_loader.mirror_lh_to_rh(c3, s3, rh_to_lh_map)
        c3, s3 = data_loader.fill_excluded_parcels(c3, s3, excluded_map)
        pairs.append((c3, s3))
    return _clim_from_dicts(*pairs)


def _val_to_hex(val: float, vmin: float, vmax: float) -> str:
    norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    r, g, b, _ = _cmap_fn(max(0.0, min(1.0, norm)))
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def _make_legend_html(c_dict: dict, s_dict: dict, clim: list) -> str:
    """Build an HTML snippet: one colored swatch + name per active V1 parcel."""
    vmin, vmax = clim
    items = sorted({**c_dict, **s_dict}.items(), key=lambda x: x[1], reverse=True)
    parts = []
    for lbl, val in items:
        display = lbl[3:] if lbl[:3] in ('LH_', 'LH-') else lbl
        color = _val_to_hex(val, vmin, vmax)
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin:2px 14px 2px 0;white-space:nowrap;">'
            f'<span style="width:13px;height:13px;background:{color};'
            f'border-radius:2px;flex-shrink:0;border:1px solid #bbb;"></span>'
            f'<span style="font-size:11px;color:#333;">{display}</span>'
            f'</span>'
        )
    return ''.join(parts)


def _make_cbar_html(label: str, vmin: float, vmax: float, width: str = '280px',
                    gradient_css: str = None, low_hex: str = None, high_hex: str = None,
                    unit_label: str = 'log₂ expr') -> str:
    grad = gradient_css or _GRADIENT_CSS
    lo   = low_hex   or _LOW_HEX
    hi   = high_hex  or _HIGH_HEX
    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:stretch;'
        f'width:{width};margin:0 8px;">'
        f'<span style="font-size:10px;color:#555;margin-bottom:4px;text-align:center;">'
        f'{label}</span>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:10px;font-weight:600;color:{lo};">▼ Low</span>'
        f'<span style="font-size:10px;font-weight:600;color:{hi};">High ▼</span>'
        f'</div>'
        f'<div style="height:14px;border-radius:3px;background:{grad};'
        f'border:1px solid #ccc;"></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#777;margin-top:2px;">'
        f'<span>{vmin:.2f}</span>'
        f'<span style="color:#999;font-style:italic;">{unit_label}</span>'
        f'<span>{vmax:.2f}</span></div>'
        f'</div>'
    )


def _make_cbar_section_html(clim1, clim2, clim3, clim4, mode: str,
                             v2_is_residual: bool = False,
                             v3_is_residual: bool = False) -> str:
    if mode == 'shared':
        bar = _make_cbar_html('All panels · joint scale across models', *clim1, width='380px')
        return f'<div style="display:flex;justify-content:center;">{bar}</div>'
    if mode == 'wbf':
        bar = _make_cbar_html('All panels · Whole-Brain Fit scale', *clim3, width='380px')
        return f'<div style="display:flex;justify-content:center;">{bar}</div>'
    bar1 = _make_cbar_html('GTEx Ground Truth', *clim1, width='180px')
    if v2_is_residual:
        bar2 = _make_cbar_html(
            'Residual (Pred − Truth)', *clim2, width='180px',
            gradient_css=_GRADIENT_DIV_CSS,
            low_hex=_DIV_LOW_HEX, high_hex=_DIV_HIGH_HEX,
            unit_label='log₂ Δexpr',
        )
    else:
        bar2 = _make_cbar_html('Reconstruction', *clim2, width='180px')
    if v3_is_residual:
        bar3 = _make_cbar_html(
            'Residual (WBF − AHBA Ref)', *clim3, width='180px',
            gradient_css=_GRADIENT_DIV_CSS,
            low_hex=_DIV_LOW_HEX, high_hex=_DIV_HIGH_HEX,
            unit_label='log₂ Δexpr',
        )
    else:
        bar3 = _make_cbar_html('Whole-Brain Fit', *clim3, width='180px')
    bar4 = _make_cbar_html('AHBA Reference', *clim4, width='180px')
    return f'<div style="display:flex;justify-content:space-evenly;">{bar1}{bar2}{bar3}{bar4}</div>'

_gene_stats: dict[str, tuple[float, float]] = {}  # gene → (r, rmse)

def _compute_gene_stats() -> None:
    """Precompute r, RMSE, mean truth, mean pred for every gene (harmonized LORO)."""
    global _gene_stats
    mask    = npz_data['loro_eval_mask'].astype(bool)
    truth_h = npz_data['loro_truth_subject_h']
    pred_h  = npz_data['loro_fused_subject_h']
    stats: dict[str, tuple[float, float, float, float]] = {}
    for gene in gene_list:
        g_idx = data_loader.get_gene_idx(npz_data, gene)
        th = truth_h[mask, g_idx]
        ph = pred_h[mask, g_idx]
        valid = ~(np.isnan(th) | np.isnan(ph))
        if valid.sum() < 2:
            stats[gene] = (float('nan'),) * 4
            continue
        th_v, ph_v = th[valid], ph[valid]
        r    = float(np.corrcoef(th_v, ph_v)[0, 1])
        rmse = float(np.sqrt(np.mean((ph_v - th_v) ** 2)))
        stats[gene] = (r, rmse, float(th_v.mean()), float(ph_v.mean()))
    _gene_stats = stats


def _make_v5_chart(gene: str) -> str:
    """Two-row V5 analytics panel (all harmonized scale).
    Row 1 (top):    color-coded parcel scatter + parcel bar chart  (current gene)
    Row 2 (bottom): per-gene mean scatter      + per-gene r bars   (all genes)
    """
    try:
        g_idx   = data_loader.get_gene_idx(npz_data, gene)
        mask    = npz_data['loro_eval_mask'].astype(bool)
        truth_h = npz_data['loro_truth_subject_h'][:, g_idx]
        pred_h  = npz_data['loro_fused_subject_h'][:, g_idx]

        parcels, th_vals, ph_vals = [], [], []
        for i, row in atlas_aligned.iterrows():
            if not mask[i]:
                continue
            t, p = float(truth_h[i]), float(pred_h[i])
            if np.isnan(t) or np.isnan(p):
                continue
            parcels.append(row['label'])
            th_vals.append(t)
            ph_vals.append(p)

        if len(parcels) < 2:
            return ''

        th  = np.array(th_vals)
        ph  = np.array(ph_vals)
        lbl = [p[3:] if p[:3] in ('LH_', 'LH-') else p for p in parcels]

        r_val = float(np.corrcoef(th, ph)[0, 1])
        rmse  = float(np.sqrt(np.mean((ph - th) ** 2)))

        genes_sorted = sorted(
            [(g, _gene_stats[g][0]) for g in gene_list
             if not np.isnan(_gene_stats.get(g, (float('nan'),))[0])],
            key=lambda x: x[1],
        )
        g_names    = [x[0] for x in genes_sorted]
        g_r        = [x[1] for x in genes_sorted]
        bar_colors = ['#e07b39' if g == gene else '#5b8dee' for g in g_names]

        g_mt = np.array([_gene_stats[g][2] for g in g_names])
        g_mp = np.array([_gene_stats[g][3] for g in g_names])

        fig = plt.figure(figsize=(13, 6.5), facecolor='white')
        gs  = fig.add_gridspec(2, 2, width_ratios=[1.5, 2.8], hspace=0.55, wspace=0.50)
        ax_tl = fig.add_subplot(gs[1, 0])
        ax_tr = fig.add_subplot(gs[1, 1])
        ax_bl = fig.add_subplot(gs[0, 0])
        ax_br = fig.add_subplot(gs[0, 1])

        all_v = np.concatenate([th, ph])
        mn, mx = all_v.min(), all_v.max()
        pad = (mx - mn) * 0.12

        # ── top-left: color-coded parcel scatter ─────────────────────────
        pt_colors = plt.cm.viridis(np.linspace(0.15, 0.90, len(parcels)))
        ax_bl.scatter(th, ph, c=pt_colors, s=55, zorder=3,
                      edgecolors='white', linewidths=0.5)
        ax_bl.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                   '--', color='#bbb', lw=1.0, zorder=2)
        for name, x, y in zip(lbl, th, ph):
            ax_bl.annotate(name, (x, y), fontsize=5.5,
                           xytext=(3, 2), textcoords='offset points', color='#444')
        ax_bl.set_xlabel('Harmonized Truth', fontsize=7.5)
        ax_bl.set_ylabel('LORO Prediction', fontsize=7.5)
        ax_bl.set_title(f'{gene} — held-out parcels  r={r_val:.3f}  RMSE={rmse:.3f}',
                        fontsize=7.5, fontweight='bold')
        ax_bl.set_xlim(mn - pad, mx + pad)
        ax_bl.set_ylim(mn - pad, mx + pad)
        ax_bl.tick_params(labelsize=6.5)
        ax_bl.set_facecolor('#f8f8f8')
        ax_bl.grid(True, alpha=0.3)

        # ── top-right: parcel bar chart ───────────────────────────────────
        order = np.argsort(th)
        ypos  = np.arange(len(order))
        bh    = 0.38
        ax_br.barh(ypos + bh / 2, th[order], bh,
                   label='Harm. Truth', color='#5b8dee', alpha=0.88)
        ax_br.barh(ypos - bh / 2, ph[order], bh,
                   label='LORO Pred.', color='#f4845e', alpha=0.88)
        ax_br.set_yticks(ypos)
        ax_br.set_yticklabels([lbl[j] for j in order], fontsize=6.5)
        ax_br.set_xlabel('log₂ expression', fontsize=7.5)
        ax_br.set_title('Parcel values — sorted by truth', fontsize=7.5, fontweight='bold')
        ax_br.legend(fontsize=7, loc='lower right')
        ax_br.tick_params(labelsize=6.5)
        ax_br.set_facecolor('#f8f8f8')
        ax_br.grid(True, axis='x', alpha=0.3)

        # ── bottom-left: global scatter, one point per gene ───────────────
        ga_all  = np.concatenate([g_mt, g_mp])
        ga_mn, ga_mx = np.nanmin(ga_all), np.nanmax(ga_all)
        ga_pad  = (ga_mx - ga_mn) * 0.12
        ax_tl.scatter(g_mt, g_mp, color='#5b8dee', s=28, zorder=3,
                      edgecolors='white', linewidths=0.3, alpha=0.75)
        cur_mt = _gene_stats.get(gene, (float('nan'),)*4)[2]
        cur_mp = _gene_stats.get(gene, (float('nan'),)*4)[3]
        if not np.isnan(cur_mt):
            ax_tl.scatter([cur_mt], [cur_mp], color='#e07b39', s=55, zorder=5,
                          edgecolors='white', linewidths=0.5)
            ax_tl.annotate(gene, (cur_mt, cur_mp), fontsize=5.5,
                           xytext=(4, 3), textcoords='offset points',
                           color='#e07b39', fontweight='bold')
        ax_tl.plot([ga_mn - ga_pad, ga_mx + ga_pad],
                   [ga_mn - ga_pad, ga_mx + ga_pad],
                   '--', color='#aaa', lw=0.9, zorder=2)
        ax_tl.set_xlabel('Mean Harm. Truth', fontsize=7.5)
        ax_tl.set_ylabel('Mean LORO Pred.', fontsize=7.5)
        ax_tl.set_title('All genes — mean expression  (orange = current)',
                        fontsize=7.5, fontweight='bold')
        ax_tl.set_xlim(ga_mn - ga_pad, ga_mx + ga_pad)
        ax_tl.set_ylim(ga_mn - ga_pad, ga_mx + ga_pad)
        ax_tl.tick_params(labelsize=6.5)
        ax_tl.set_facecolor('#f8f8f8')
        ax_tl.grid(True, alpha=0.3)

        # ── bottom-right: per-gene r bar chart ───────────────────────────
        xpos = np.arange(len(g_names))
        ax_tr.bar(xpos, g_r, color=bar_colors, width=0.8, zorder=3)
        ax_tr.axhline(0, color='#888', lw=0.6, zorder=2)
        ax_tr.set_xticks(xpos)
        ax_tr.set_xticklabels(g_names, rotation=90, fontsize=4.0)
        ax_tr.set_ylabel('Pearson r', fontsize=7.5)
        ax_tr.set_title('Per-gene LORO r  (sorted; orange = current)',
                        fontsize=7.5, fontweight='bold')
        ax_tr.tick_params(axis='y', labelsize=6.5)
        ax_tr.set_facecolor('#f8f8f8')
        ax_tr.grid(True, axis='y', alpha=0.3)
        ax_tr.set_xlim(-0.5, len(g_names) - 0.5)

        # ── row separator + row labels ────────────────────────────────────
        pos_top = ax_bl.get_position()
        pos_bot = ax_tl.get_position()
        mid_y   = (pos_top.y0 + pos_bot.y1) / 2
        from matplotlib.lines import Line2D
        fig.add_artist(Line2D([0.02, 0.98], [mid_y, mid_y],
                               transform=fig.transFigure,
                               color='#d0d0d0', linewidth=0.8, linestyle='-'))
        fig.text(0.005, (pos_top.y0 + pos_top.y1) / 2, 'Current gene',
                 fontsize=6.5, color='#aaa', va='center', ha='left',
                 rotation=90, style='italic')
        fig.text(0.005, (pos_bot.y0 + pos_bot.y1) / 2, 'All genes',
                 fontsize=6.5, color='#aaa', va='center', ha='left',
                 rotation=90, style='italic')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;height:auto;display:block;" />'
        )
    except Exception as e:
        print(f'[v5] {e}')
        return ''


# Precompute per-gene stats for the initial subject/model
_compute_gene_stats()

# ── 8. PyVista plotter — three side-by-side viewports ─────────────────────────
pv.global_theme.trame.default_mode = 'server'
pv.global_theme.background = 'white'
plotter = pv.Plotter(shape=(1, 4), notebook=False)

SURF_KW_BASE = dict(
    scalars='Data', cmap=CMAP,
    smooth_shading=True, lighting=True,
    ambient=0.55, diffuse=0.45, specular=0.05,
    show_scalar_bar=False,
)

# ── 9. VTK clipping plane (keeps x ≤ 0, i.e. left hemisphere) ─────────────────
_clip_plane = vtk.vtkPlane()
_clip_plane.SetNormal(-1.0, 0.0, 0.0)
_clip_plane.SetOrigin(0.0, 0.0, 0.0)


def _make_viewport(col: int, title: str, subtitle: str = ''):
    """Set up one viewport: background, label, cortex + subcortex meshes.

    Returns (lh_m, rh_m, lh_a, rh_a, sub, title_actor, subtitle_actor).
    title_actor and subtitle_actor can be updated later via SetInput().
    """
    plotter.subplot(0, col)
    plotter.set_background('white')
    # Use raw vtkTextActor (not plotter.add_text) so we can call SetInput() later.
    # plotter.add_text(position='upper_edge') returns a CornerAnnotation which lacks SetInput.
    title_actor = vtk.vtkTextActor()
    title_actor.SetInput(title)
    title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    title_actor.SetPosition(0.5, 0.92)
    title_actor.GetTextProperty().SetJustificationToCentered()
    title_actor.GetTextProperty().SetFontSize(14)
    title_actor.GetTextProperty().SetColor(0.2, 0.2, 0.2)
    plotter.renderer.AddActor2D(title_actor)
    subtitle_actor = None
    if subtitle:
        subtitle_actor = vtk.vtkTextActor()
        subtitle_actor.SetInput(subtitle)
        subtitle_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        subtitle_actor.SetPosition(0.5, 0.93)
        subtitle_actor.GetTextProperty().SetJustificationToCentered()
        subtitle_actor.GetTextProperty().SetFontSize(11)
        subtitle_actor.GetTextProperty().SetColor(0.45, 0.45, 0.45)
        plotter.renderer.AddActor2D(subtitle_actor)

    lh_m = pv.PolyData(lh_v.astype(np.float32), LH_FACES)
    rh_m = pv.PolyData(rh_v.astype(np.float32), RH_FACES)
    lh_m['Data'] = np.full(len(lh_v), np.nan, dtype=np.float32)
    rh_m['Data'] = np.full(len(rh_v), np.nan, dtype=np.float32)
    lh_a = plotter.add_mesh(lh_m, nan_color=NAN_COLOR, clim=[0, 1],
                            opacity=CORTEX_ALPHA, **SURF_KW_BASE)
    rh_a = plotter.add_mesh(rh_m, nan_color=NAN_COLOR, clim=[0, 1],
                            opacity=CORTEX_ALPHA, **SURF_KW_BASE)

    sub: dict[str, tuple] = {}
    for name, raw in subcortical_meshes.items():
        m = raw.copy()
        m['Data'] = np.full(m.n_points, np.nan, dtype=np.float32)
        a = plotter.add_mesh(m, nan_color=NAN_COLOR, clim=[0, 1],
                             opacity=SUBCORTEX_ALPHA, **SURF_KW_BASE)
        sub[name] = (m, a)

    return lh_m, rh_m, lh_a, rh_a, sub, title_actor, subtitle_actor


# ── 10-13. Build all four viewports ───────────────────────────────────────────
print("Pre-building meshes for all four viewports...")
lh_v1, rh_v1, lh_a1, rh_a1, sub_v1, _, _ = _make_viewport(
    0, 'GTEx Ground Truth')
lh_v2, rh_v2, lh_a2, rh_a2, sub_v2, _v2_title_actor, _v2_sub_actor = _make_viewport(
    1, 'Reconstruction')
lh_v3, rh_v3, lh_a3, rh_a3, sub_v3, _v3_title_actor, _v3_sub_actor = _make_viewport(
    2, 'Whole-Brain Fit')
lh_v4, rh_v4, lh_a4, rh_a4, sub_v4, _, _ = _make_viewport(
    3, 'AHBA Reference')

# ── 14. Shared camera — rotating any viewport moves all four ──────────────────
_shared_cam = plotter.renderers[0].GetActiveCamera()
plotter.renderers[1].SetActiveCamera(_shared_cam)
plotter.renderers[2].SetActiveCamera(_shared_cam)
plotter.renderers[3].SetActiveCamera(_shared_cam)

_all_actors = (
    [lh_a1, rh_a1] + [a for _, a in sub_v1.values()] +
    [lh_a2, rh_a2] + [a for _, a in sub_v2.values()] +
    [lh_a3, rh_a3] + [a for _, a in sub_v3.values()] +
    [lh_a4, rh_a4] + [a for _, a in sub_v4.values()]
)

# ── 14. Update function (in-place scalar writes, no actor rebuild) ─────────────
def _make_cortex_scalars(cortex_dict: dict):
    if not cortical_atlas_available:
        return (np.full(len(lh_v), np.nan, np.float32),
                np.full(len(rh_v), np.nan, np.float32))
    all_vals = map_values_to_surface(cortex_dict, tar_labels, lut_ids, lut_names)
    return (all_vals[:len(lh_v)].astype(np.float32),
            all_vals[len(lh_v):].astype(np.float32))


def _update_viewport(lh_mesh, rh_mesh, lh_actor, rh_actor, sub_actors,
                     c_dict, s_dict, clim):
    lh_s, rh_s = _make_cortex_scalars(c_dict)
    lh_mesh['Data'] = lh_s
    rh_mesh['Data'] = rh_s
    lh_actor.GetMapper().SetScalarRange(*clim)
    rh_actor.GetMapper().SetScalarRange(*clim)
    for name, (m, actor) in sub_actors.items():
        val = float(s_dict.get(name, np.nan))
        m['Data'] = np.full(m.n_points, val, dtype=np.float32)
        actor.GetMapper().SetScalarRange(*clim)


def update_scene(c1, s1, c2, s2, c3, s3, c4, s4, gene: str, cbar_mode: str = 'local',
                 shared_clim=None, v2_is_residual: bool = False,
                 v3_is_residual: bool = False) -> tuple:
    if cbar_mode == 'shared':
        clim = shared_clim if shared_clim is not None else _clim_from_dicts((c1, s1), (c2, s2), (c3, s3), (c4, s4))
        clim1 = clim2 = clim3 = clim4 = clim
    elif cbar_mode == 'wbf':
        clim = _clim_from_dicts((c3, s3))
        clim1 = clim2 = clim3 = clim4 = clim
    else:
        clim1 = _clim_from_dicts((c1, s1))
        clim2 = _clim_from_dicts((c2, s2))
        clim3 = _clim_from_dicts((c3, s3))
        clim4 = _clim_from_dicts((c4, s4))
    # Residual panels must be symmetric around 0 regardless of cbar_mode.
    if v2_is_residual:
        clim2 = _symmetric_clim(c2, s2)
    if v3_is_residual:
        clim3 = _symmetric_clim(c3, s3)
    _update_viewport(lh_v1, rh_v1, lh_a1, rh_a1, sub_v1, c1, s1, clim1)
    _update_viewport(lh_v2, rh_v2, lh_a2, rh_a2, sub_v2, c2, s2, clim2)
    _update_viewport(lh_v3, rh_v3, lh_a3, rh_a3, sub_v3, c3, s3, clim3)
    _update_viewport(lh_v4, rh_v4, lh_a4, rh_a4, sub_v4, c4, s4, clim4)
    plotter.render()
    return clim1, clim2, clim3, clim4


def _set_clip(half: bool):
    for actor in _all_actors:
        mapper = actor.GetMapper()
        mapper.RemoveAllClippingPlanes()
        if half:
            mapper.AddClippingPlane(_clip_plane)


# ── 15. Data helpers ───────────────────────────────────────────────────────────
def _get_all(gene: str, cbar_mode: str = 'local', v2_mode: str = 'recon'):
    if cbar_mode in ('shared', 'wbf'):
        c1, s1 = data_loader.v1_gtex_harmonized(npz_data, atlas_aligned, gene)
    else:
        c1, s1 = data_loader.v1_gtex_input(npz_data, atlas_aligned, gene)
    c1, s1 = data_loader.remap_rh_to_lh_nearest(c1, s1, rh_to_lh_map)
    if v2_mode == 'residual':
        c2, s2 = data_loader.v2_residual(npz_data, atlas_aligned, gene)
    else:
        c2, s2 = data_loader.v2_reconstruction(npz_data, atlas_aligned, gene)
    c2, s2 = data_loader.remap_rh_to_lh_nearest(c2, s2, rh_to_lh_map)
    # V3: fullfit at imputed parcels, but V2 reconstruction at GTEx parcels.
    # Remap V2 to LH keys before merging so that mirror_lh_to_rh uses the
    # reconstruction value (not fullfit) when it copies LH → RH.
    c3_raw, s3_raw = data_loader.v4_fullfit(npz_data, atlas_aligned, gene)
    c2_raw, s2_raw = data_loader.v2_reconstruction(npz_data, atlas_aligned, gene)
    c2_lh, s2_lh = data_loader.remap_rh_to_lh_nearest(c2_raw, s2_raw, rh_to_lh_map)
    c3, s3 = data_loader.mirror_lh_to_rh({**c3_raw, **c2_lh}, {**s3_raw, **s2_lh}, rh_to_lh_map)
    c3, s3 = data_loader.fill_excluded_parcels(c3, s3, excluded_map)
    return c1, s1, c2, s2, c3, s3


def _get_ahba_ref(gene: str):
    """Naive whole-brain fit with LORO reconstruction at sampled regions — fixed reference."""
    naive_npz = _all_npz['naive']
    c3_raw, s3_raw = data_loader.v4_fullfit(naive_npz, atlas_aligned, gene)
    c2_raw, s2_raw = data_loader.v2_reconstruction(naive_npz, atlas_aligned, gene)
    c2_lh, s2_lh = data_loader.remap_rh_to_lh_nearest(c2_raw, s2_raw, rh_to_lh_map)
    c4, s4 = data_loader.mirror_lh_to_rh({**c3_raw, **c2_lh}, {**s3_raw, **s2_lh}, rh_to_lh_map)
    c4, s4 = data_loader.fill_excluded_parcels(c4, s4, excluded_map)
    return c4, s4


# ── 16. Initial render ─────────────────────────────────────────────────────────
print(f"Rendering initial scene for gene '{INIT_GENE}'...")
c1_i, s1_i, c2_i, s2_i, c3_i, s3_i = _get_all(INIT_GENE)
c4_i, s4_i = _get_ahba_ref(INIT_GENE)
print(f"  V1: {len(c1_i)}cx+{len(s1_i)}sub  "
      f"V2: {len(c2_i)}cx+{len(s2_i)}sub  "
      f"V3: {len(c3_i)}cx+{len(s3_i)}sub  "
      f"V4: {len(c4_i)}cx+{len(s4_i)}sub")
_excl_check = ['RH-EXA', 'RH-STH', 'RH-VeP']
for _n in _excl_check:
    print(f"  Excluded {_n}: in s3={_n in s3_i}, val={s3_i.get(_n, 'MISSING'):.4f}" if _n in s3_i else f"  Excluded {_n}: MISSING from s3")
_init_clims = update_scene(c1_i, s1_i, c2_i, s2_i, c3_i, s3_i, c4_i, s4_i, INIT_GENE)

plotter.subplot(0, 0)
plotter.reset_camera()
_cam = plotter.camera_position
_CAM_FULL_BASE = (tuple(_cam[0]), tuple(_cam[1]), (0.0, 1.0, 0.0))
_focus = _CAM_FULL_BASE[1]
_CAM_HALF_BASE = (
    (_focus[0] + 600.0, _focus[1], _focus[2]),
    _focus,
    (0.0, 0.0, 1.0),
)


def _apply_camera(half: bool) -> None:
    """Switch camera to the full-brain or half-brain base position."""
    base = _CAM_HALF_BASE if half else _CAM_FULL_BASE
    _shared_cam.SetPosition(*base[0])
    _shared_cam.SetFocalPoint(*base[1])
    _shared_cam.SetViewUp(*base[2])
    for renderer in plotter.renderers:
        renderer.ResetCameraClippingRange()


_apply_camera(False)
plotter.camera.zoom(1.5)
plotter.enable_trackball_style()
try:
    # Lower values make scroll-wheel zoom feel less jumpy.
    plotter.iren.GetInteractorStyle().SetMouseWheelMotionFactor(ZOOM_SENSITIVITY)
except Exception:
    pass

# ── Click tooltip — actor-address lookup ──────────────────────────────────────
# Key by actor address, not dataset address: smooth_shading=True causes PyVista
# to run vtkPolyDataNormals internally, producing a new vtkPolyData copy whose
# address differs from the original mesh.  The actor object is unchanged by that
# pipeline step, so actor-address comparison is reliable.
_VP_DISPLAY = {
    'V1': 'GTEx Truth',
    'V2': 'Reconstruction',
    'V3': 'Whole-Brain Fit',
    'V4': 'AHBA Ref',
}
_actor_lookup: dict[str, tuple] = {}
# Subcortical: one actor per named region per viewport
for _vi, _sub_vp in enumerate([sub_v1, sub_v2, sub_v3, sub_v4]):
    _vp_id = f'V{_vi + 1}'
    for _name, (_mesh, _actor) in _sub_vp.items():
        _disp = _name[3:] if _name[:3] in ('LH_', 'LH-', 'RH_', 'RH-') else _name
        try:
            _actor_lookup[_actor.GetAddressAsString('')] = (_name, _disp, _vp_id, _mesh, 'sub')
        except Exception:
            pass
# Cortical: one LH + one RH actor per viewport
if cortical_atlas_available:
    _lut_id_to_name = {int(lid): str(lname) for lid, lname in zip(lut_ids, lut_names)}
    for _vi, (_la, _ra, _lm, _rm) in enumerate([
        (lh_a1, rh_a1, lh_v1, rh_v1),
        (lh_a2, rh_a2, lh_v2, rh_v2),
        (lh_a3, rh_a3, lh_v3, rh_v3),
        (lh_a4, rh_a4, lh_v4, rh_v4),
    ]):
        _vp = f'V{_vi + 1}'
        try:
            _actor_lookup[_la.GetAddressAsString('')] = (None, None, _vp, _lm, 'lh')
            _actor_lookup[_ra.GetAddressAsString('')] = (None, None, _vp, _rm, 'rh')
        except Exception:
            pass

print(f"Click lookup: {len(_actor_lookup)} actors registered "
      f"({sum(1 for v in _actor_lookup.values() if v[4]=='sub')} sub, "
      f"{sum(1 for v in _actor_lookup.values() if v[4] in ('lh','rh'))} cortex)")

_hover_picker = vtk.vtkCellPicker()
_hover_picker.SetTolerance(0.01)
_hover_last: list = [None]


# ── 17. Trame server + state ───────────────────────────────────────────────────
server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.gene         = INIT_GENE
state.gene_list    = gene_list
state.model        = INIT_MODEL
state.model_list   = ['naive', 'dlam', 'plam']
state.subject      = INIT_SUBJECT
state.subject_list = SUBJECT_LIST
state.n_v1           = f"{len(c1_i)}cx+{len(s1_i)}sub"
state.n_v2           = f"{len(c2_i)}cx+{len(s2_i)}sub"
state.n_v3           = f"{len(c3_i)}cx+{len(s3_i)}sub"
state.n_v4           = f"{len(c4_i)}cx+{len(s4_i)}sub"
_ic1, _ic2, _ic3, _ic4 = _init_clims
state.v1_legend_html  = _make_legend_html(c1_i, s1_i, _ic1)
state.cbar_html       = _make_cbar_section_html(_ic1, _ic2, _ic3, _ic4, 'local')
state.v5_chart_html   = _make_v5_chart(INIT_GENE)
state.show_v5         = False
state.drawer          = True
state.tooltip_visible     = False
state.tooltip_parcel      = ''
state.tooltip_value       = ''
state.tooltip_is_residual = False
state.tooltip_row1        = ''
state.tooltip_row2        = ''
state.tooltip_row3        = ''
state.cbar_mode       = 'local'
state.v2_mode         = 'recon'
state.half_brain      = 'full'
state.cortex_alpha    = CORTEX_ALPHA
state.subcortex_alpha = SUBCORTEX_ALPHA
state.cerebellum_alpha = SUBCORTEX_ALPHA
def _push():
    try:
        ctrl.view_update()
    except Exception:
        pass


# Parcel-level lookup dicts for residual tooltips — populated in _reload_scene
_residual_truth: dict   = {}   # V2: harmonized ground truth per parcel
_residual_pred: dict    = {}   # V2: LORO prediction per parcel
_v3_residual_ref: dict  = {}   # V3: AHBA reference (naive) per parcel


def _update_cbars(c1, s1, clim1, clim2, clim3, clim4, cbar_mode: str,
                  v2_is_residual: bool = False, v3_is_residual: bool = False):
    state.v1_legend_html = _make_legend_html(c1, s1, clim1)
    state.cbar_html = _make_cbar_section_html(
        clim1, clim2, clim3, clim4, cbar_mode, v2_is_residual, v3_is_residual)


def _reload_scene(c1, s1, c2, s2, c3, s3, gene, v2_is_residual: bool = False):
    global _residual_truth, _residual_pred, _v3_residual_ref
    c4, s4 = _get_ahba_ref(gene)

    v3_is_residual = v2_is_residual and state.model != 'naive'

    # Compute V3 display values — residual (WBF − AHBA ref) for plam/dlam only
    if v3_is_residual:
        c3_disp = {k: v - c4[k] for k, v in c3.items() if k in c4}
        s3_disp = {k: v - s4[k] for k, v in s3.items() if k in s4}
        _v3_residual_ref = {**c4, **s4}
    else:
        c3_disp, s3_disp = c3, s3
        _v3_residual_ref = {}

    # Update V3 LUT and title to match its current mode
    _set_viewport_lut(lh_a3, rh_a3, sub_v3, _lut_div if v3_is_residual else _lut_seq)
    _v3_title_actor.SetInput('Residual (WBF − AHBA Ref)' if v3_is_residual else 'Whole-Brain Fit')
    if _v3_sub_actor:
        _v3_sub_actor.SetInput(
            'plam/dlam WBF minus naive AHBA reference' if v3_is_residual
            else 'Ground truth data combined with imputed data over whole brain'
        )

    sc = _shared_clim(gene) if state.cbar_mode == 'shared' else None
    clim1, clim2, clim3, clim4 = update_scene(
        c1, s1, c2, s2, c3_disp, s3_disp, c4, s4, gene,
        state.cbar_mode, sc, v2_is_residual, v3_is_residual)
    _update_cbars(c1, s1, clim1, clim2, clim3, clim4, state.cbar_mode, v2_is_residual, v3_is_residual)

    state.v5_chart_html = _make_v5_chart(gene)

    # Store per-parcel lookup dicts for residual tooltips
    if v2_is_residual:
        ct, st = data_loader.v1_gtex_harmonized(npz_data, atlas_aligned, gene)
        ct, st = data_loader.remap_rh_to_lh_nearest(ct, st, rh_to_lh_map)
        _residual_truth = {**ct, **st}
        cp, sp = data_loader.v2_reconstruction(npz_data, atlas_aligned, gene)
        cp, sp = data_loader.remap_rh_to_lh_nearest(cp, sp, rh_to_lh_map)
        _residual_pred = {**cp, **sp}
    else:
        _residual_truth = {}
        _residual_pred  = {}


@state.change('gene')
def on_gene(gene, **_):
    is_residual = (state.v2_mode == 'residual')
    c1, s1, c2, s2, c3, s3 = _get_all(gene, state.cbar_mode, state.v2_mode)
    c4, s4 = _get_ahba_ref(gene)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    state.n_v4 = f"{len(c4)}cx+{len(s4)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, gene, is_residual)
    _push()


@state.change('model')
def on_model(model, **_):
    global npz_data
    is_residual = (state.v2_mode == 'residual')
    npz_data = _all_npz[model]
    _compute_gene_stats()
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode, state.v2_mode)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene, is_residual)
    _push()


@state.change('subject')
def on_subject(subject, **_):
    global npz_data, _all_npz
    is_residual = (state.v2_mode == 'residual')
    _all_npz = {m: data_loader.load_subject(m, subject) for m in _MODEL_LIST}
    npz_data = _all_npz[state.model]
    _compute_gene_stats()
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode, state.v2_mode)
    c4, s4 = _get_ahba_ref(state.gene)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    state.n_v4 = f"{len(c4)}cx+{len(s4)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene, is_residual)
    _push()


@state.change('cbar_mode')
def on_cbar_mode(**_):
    is_residual = (state.v2_mode == 'residual')
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode, state.v2_mode)
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene, is_residual)
    _push()


@state.change('v2_mode')
def on_v2_mode(v2_mode, **_):
    is_residual = (v2_mode == 'residual')
    _set_viewport_lut(lh_a2, rh_a2, sub_v2, _lut_div if is_residual else _lut_seq)
    _v2_title_actor.SetInput('Residual (Pred − Truth)' if is_residual else 'Reconstruction')
    if _v2_sub_actor:
        _v2_sub_actor.SetInput(
            'Model prediction minus harmonized ground truth' if is_residual
            else 'LORO predicted values for sampled regions'
        )
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode, v2_mode)
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene, is_residual)
    _push()


_cortex_actors    = [lh_a1, rh_a1, lh_a2, rh_a2, lh_a3, rh_a3, lh_a4, rh_a4]
_subcortex_actors = []
_cerebellum_actors = []
for _sub_vp in [sub_v1, sub_v2, sub_v3, sub_v4]:
    for _name, (_, _actor) in _sub_vp.items():
        if _name.startswith('Cerebellar_'):
            _cerebellum_actors.append(_actor)
        else:
            _subcortex_actors.append(_actor)


@state.change('cortex_alpha')
def on_cortex_alpha(cortex_alpha, **_):
    alpha = float(cortex_alpha)
    for actor in _cortex_actors:
        actor.GetProperty().SetOpacity(alpha)
    plotter.render()
    _push()


@state.change('subcortex_alpha')
def on_subcortex_alpha(subcortex_alpha, **_):
    alpha = float(subcortex_alpha)
    for actor in _subcortex_actors:
        actor.GetProperty().SetOpacity(alpha)
    plotter.render()
    _push()


@state.change('cerebellum_alpha')
def on_cerebellum_alpha(cerebellum_alpha, **_):
    alpha = float(cerebellum_alpha)
    for actor in _cerebellum_actors:
        actor.GetProperty().SetOpacity(alpha)
    plotter.render()
    _push()


@state.change('half_brain')
def on_half_brain(half_brain, **_):
    is_half = (half_brain == 'half')
    _set_clip(is_half)
    _apply_camera(is_half)
    plotter.render()
    _push()


# ── Click tooltip callback ─────────────────────────────────────────────────────
def _on_click(obj, event):
    """Show tooltip when clicking a parcel (sub- or cortical) in any viewport."""
    try:
        # Use raw VTK obj (vtkRenderWindowInteractor) — most direct way to get
        # the current event position from within a VTK observer callback.
        x, y = obj.GetEventPosition()
        hit = False
        for renderer in plotter.renderers:
            if _hover_picker.Pick(x, y, 0, renderer) == 0:
                continue
            actor = _hover_picker.GetActor()
            if actor is None:
                continue
            addr = actor.GetAddressAsString('')
            if addr not in _actor_lookup:
                continue
            name, disp, vp_id, mesh, kind = _actor_lookup[addr]

            if kind == 'sub':
                try:
                    val = float(mesh['Data'][0])
                except Exception:
                    val = float('nan')
            elif kind in ('lh', 'rh'):
                pt_id = _hover_picker.GetPointId()
                if pt_id < 0:
                    continue
                offset = 0 if kind == 'lh' else len(lh_v)
                tar_idx = offset + pt_id
                if tar_idx >= len(tar_labels):
                    continue
                parcel_id = int(tar_labels[tar_idx])
                name = _lut_id_to_name.get(parcel_id, '')
                if not name:
                    continue
                disp = name[3:] if name[:3] in ('LH_', 'LH-', 'RH_', 'RH-') else name
                try:
                    val = float(mesh['Data'][pt_id])
                except Exception:
                    val = float('nan')
            else:
                continue

            key = (name, vp_id)
            if _hover_last[0] == key:
                _hover_last[0] = None
                state.tooltip_visible = False
            else:
                _hover_last[0] = key
                vp_label = _VP_DISPLAY.get(vp_id, vp_id)
                state.tooltip_visible = True
                state.tooltip_parcel = f'{vp_label} · {disp}'
                def _fmt(v): return 'N/A' if np.isnan(v) else f'{v:.3f}'
                if vp_id == 'V2' and state.v2_mode == 'residual' and name in _residual_truth:
                    truth_v = _residual_truth.get(name, float('nan'))
                    pred_v  = _residual_pred.get(name, float('nan'))
                    state.tooltip_is_residual = True
                    state.tooltip_row1 = f'Truth:  {_fmt(truth_v)}'
                    state.tooltip_row2 = f'Pred:   {_fmt(pred_v)}'
                    state.tooltip_row3 = f'Δ:      {_fmt(val)}'
                elif vp_id == 'V3' and state.v2_mode == 'residual' and state.model != 'naive' and name in _v3_residual_ref:
                    ref_v  = _v3_residual_ref.get(name, float('nan'))
                    wbf_v  = (ref_v + val) if not (np.isnan(ref_v) or np.isnan(val)) else float('nan')
                    state.tooltip_is_residual = True
                    state.tooltip_row1 = f'AHBA:   {_fmt(ref_v)}'
                    state.tooltip_row2 = f'WBF:    {_fmt(wbf_v)}'
                    state.tooltip_row3 = f'Δ:      {_fmt(val)}'
                else:
                    state.tooltip_is_residual = False
                    state.tooltip_value = 'N/A' if np.isnan(val) else f'{val:.3f}'
            state.flush()
            _push()
            hit = True
            break

        if not hit and _hover_last[0] is not None:
            _hover_last[0] = None
            state.tooltip_visible = False
            state.flush()
            _push()
    except Exception as e:
        print(f'[click] {e}')


plotter.iren.add_observer('LeftButtonPressEvent', _on_click)


# ── 18. Trame UI ───────────────────────────────────────────────────────────────
_SEC  = ('font-size:10px; font-weight:700; color:#999; letter-spacing:0.10em;'
         ' text-transform:uppercase; margin:0 0 8px 0;')
_LBL  = 'font-size:12px; color:#555; margin-bottom:4px;'
_GAP  = 'height:14px;'
_DIV  = 'margin:16px 0;'
_FULL = 'width:100%;'
_BTN  = 'font-size:11px; text-transform:none; flex:1;'

with SinglePageLayout(server) as layout:
    layout.title.set_text('Brain Gene Expression Explorer')

    layout.icon.hide()  # remove non-functional default left hamburger

    # ── Toolbar ────────────────────────────────────────────────────────────────
    with layout.toolbar:
        v3.VAppBarNavIcon(click='drawer = !drawer')

    # ── Side drawer ────────────────────────────────────────────────────────────
    with v3.VNavigationDrawer(
        v_model=('drawer', True),
        location='left',
        width=270,
        style='background:#fafafa;',
    ):
        with html.Div(style='padding:20px 16px; display:flex; flex-direction:column;'):

            # ── Data ──────────────────────────────────────────────────────────
            html.P('Data', style=_SEC)

            html.P('Gene', style=_LBL)
            v3.VSelect(
                v_model=('gene',), items=('gene_list',),
                density='compact', hide_details=True, variant='outlined',
            )
            html.Div(style=_GAP)

            html.P('Subject', style=_LBL)
            v3.VSelect(
                v_model=('subject',), items=('subject_list',),
                density='compact', hide_details=True, variant='outlined',
            )
            html.Div(style=_GAP)

            html.P('Model', style=_LBL)
            v3.VSelect(
                v_model=('model',), items=('model_list',),
                density='compact', hide_details=True, variant='outlined',
            )

            v3.VDivider(style=_DIV)

            # ── Error Analysis ────────────────────────────────────────────────
            html.P('Error Analysis', style=_SEC)
            with v3.VBtnToggle(
                v_model=('v2_mode',), mandatory=True,
                density='compact', variant='outlined', divided=True,
                style=_FULL,
            ):
                v3.VBtn('Reconstruction', value='recon',   size='small', style=_BTN)
                v3.VBtn('Residual',       value='residual', size='small', style=_BTN)

            v3.VDivider(style=_DIV)

            # ── View ──────────────────────────────────────────────────────────
            html.P('View', style=_SEC)

            html.P('Brain clip', style=_LBL)
            with v3.VBtnToggle(
                v_model=('half_brain',), mandatory=True,
                density='compact', variant='outlined', divided=True,
                style=_FULL,
            ):
                v3.VBtn('Full', value='full', size='small', style=_BTN)
                v3.VBtn('Half', value='half', size='small', style=_BTN)
            html.Div(style=_GAP)

            html.P('Color scale', style=_LBL)
            with v3.VBtnToggle(
                v_model=('cbar_mode',), mandatory=True,
                density='compact', variant='outlined', divided=True,
                style=_FULL,
            ):
                v3.VBtn('Local',  value='local',  size='small', style=_BTN)
                v3.VBtn('Shared', value='shared', size='small', style=_BTN)
                v3.VBtn('WBF',    value='wbf',    size='small', style=_BTN)

            v3.VDivider(style=_DIV)

            # ── Opacity ───────────────────────────────────────────────────────
            html.P('Opacity', style=_SEC)

            html.P('Cortex', style=_LBL)
            v3.VSlider(
                v_model=('cortex_alpha',), min=0.0, max=1.0, step=0.05,
                density='compact', hide_details=True,
            )
            html.Div(style=_GAP)

            html.P('Subcortex', style=_LBL)
            v3.VSlider(
                v_model=('subcortex_alpha',), min=0.0, max=1.0, step=0.05,
                density='compact', hide_details=True,
            )
            html.Div(style=_GAP)

            html.P('Cerebellum', style=_LBL)
            v3.VSlider(
                v_model=('cerebellum_alpha',), min=0.0, max=1.0, step=0.05,
                density='compact', hide_details=True,
            )

            v3.VDivider(style=_DIV)

            # ── Analytics ─────────────────────────────────────────────────
            html.P('Analytics', style=_SEC)
            v3.VSwitch(
                v_model=('show_v5',),
                label='Show V5 Analytics',
                density='compact', hide_details=True,
                color='primary',
            )

    # ── Hover tooltip card (fixed, bottom-center, above colorbar) ────────────
    with html.Div(
        v_show=('tooltip_visible',),
        style=(
            'position:fixed; bottom:190px; left:50%; transform:translateX(-50%); z-index:300;'
            'background:rgba(255,255,255,0.97);'
            'border:1px solid #e0e0e0; border-radius:8px;'
            'padding:10px 14px; pointer-events:none;'
            'box-shadow:0 3px 14px rgba(0,0,0,0.14);'
            'min-width:190px; text-align:center;'
        ),
    ):
        html.Div(
            v_text=('tooltip_parcel',),
            style='font-size:13px; font-weight:600; color:#1a1a1a; margin-bottom:5px;',
        )
        html.Div(
            v_if='!tooltip_is_residual',
            v_text=('tooltip_value',),
            style='font-size:12px; color:#666; font-family:monospace;',
        )
        with html.Div(v_else=True, style='font-size:12px; font-family:monospace; text-align:left;'):
            html.Div(v_text='tooltip_row1', style='color:#555;')
            html.Div(v_text='tooltip_row2', style='color:#555;')
            html.Div(v_text='tooltip_row3', style='color:#d32f2f; font-weight:700;')

    # ── Main content ───────────────────────────────────────────────────────────
    with layout.content:
        with html.Div(style='display:flex; flex-direction:column; height:100%; background:white;'):
            with html.Div(style='flex:1; min-height:0;'):
                with plotter_ui(plotter, style='width:100%; height:100%; background:white;'):
                    pass
            with html.Div(style='flex-shrink:0; border-top:1px solid #e0e0e0; background:#fafafa;'):
                with html.Div(style='padding:8px 14px 4px;'):
                    html.Div(v_html=('cbar_html',))
                with html.Div(style='padding:2px 14px 6px; display:flex; align-items:center; flex-wrap:wrap;'):
                    html.Span(
                        'GTEx Regions — ',
                        style='font-size:11px; color:#888; margin-right:6px; white-space:nowrap;',
                    )
                    html.Span(v_html=('v1_legend_html',))
            with html.Div(
                v_show=('show_v5',),
                style='flex-shrink:0; border-top:1px solid #e0e0e0; background:white;',
            ):
                with html.Div(style='padding:4px 14px 2px;'):
                    html.Span(
                        'V5 — Model Analytics',
                        style='font-size:10px; font-weight:700; color:#999; letter-spacing:0.10em; text-transform:uppercase;',
                    )
                with html.Div(style='padding:0 14px 8px;'):
                    html.Div(v_html=('v5_chart_html',))


if __name__ == '__main__':
    server.start(exec_mode='main', port=1234)
