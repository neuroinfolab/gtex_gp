"""
Brain Gene Expression Explorer — V1 | V2 | V3  (side-by-side, shared camera)
yabplot + PyVista + trame interactive prototype

Run:  python app.py
Open: http://localhost:1234

Prerequisites:
  - Run build_atlas.py at least once (subcortical works without wb_command;
    cortical requires wb_command — see build_atlas.py for install instructions).
"""

import glob
from pathlib import Path

import matplotlib.cm as matplotlib_cm
import numpy as np
import pyvista as pv
import vtk
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
NAN_COLOR       = (0.80, 0.80, 0.80)
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


def _make_cbar_html(label: str, vmin: float, vmax: float, width: str = '280px') -> str:
    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:stretch;'
        f'width:{width};margin:0 8px;">'
        f'<span style="font-size:10px;color:#555;margin-bottom:4px;text-align:center;">'
        f'{label}</span>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:10px;font-weight:600;color:{_LOW_HEX};">▼ Low</span>'
        f'<span style="font-size:10px;font-weight:600;color:{_HIGH_HEX};">High ▼</span>'
        f'</div>'
        f'<div style="height:14px;border-radius:3px;background:{_GRADIENT_CSS};'
        f'border:1px solid #ccc;"></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#777;margin-top:2px;">'
        f'<span>{vmin:.2f}</span>'
        f'<span style="color:#999;font-style:italic;">log₂ expr</span>'
        f'<span>{vmax:.2f}</span></div>'
        f'</div>'
    )


def _make_cbar_section_html(clim1, clim2, clim3, clim4, mode: str) -> str:
    if mode == 'shared':
        bar = _make_cbar_html('All panels · joint scale across models', *clim1, width='380px')
        return f'<div style="display:flex;justify-content:center;">{bar}</div>'
    if mode == 'wbf':
        bar = _make_cbar_html('All panels · Whole-Brain Fit scale', *clim3, width='380px')
        return f'<div style="display:flex;justify-content:center;">{bar}</div>'
    bar1 = _make_cbar_html('GTEx Ground Truth', *clim1, width='180px')
    bar2 = _make_cbar_html('Reconstruction', *clim2, width='180px')
    bar3 = _make_cbar_html('Whole-Brain Fit', *clim3, width='180px')
    bar4 = _make_cbar_html('AHBA Reference', *clim4, width='180px')
    return f'<div style="display:flex;justify-content:space-evenly;">{bar1}{bar2}{bar3}{bar4}</div>'

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
    """Set up one viewport: background, label, cortex + subcortex meshes."""
    plotter.subplot(0, col)
    plotter.set_background('white')
    plotter.add_text(title, position='upper_edge', font_size=10, color='gray')
    if subtitle:
        _ta = vtk.vtkTextActor()
        _ta.SetInput(subtitle)
        _ta.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        _ta.SetPosition(0.5, 0.93)
        _ta.GetTextProperty().SetJustificationToCentered()
        _ta.GetTextProperty().SetFontSize(11)
        _ta.GetTextProperty().SetColor(0.45, 0.45, 0.45)
        plotter.renderer.AddActor2D(_ta)

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

    return lh_m, rh_m, lh_a, rh_a, sub


# ── 10-13. Build all four viewports ───────────────────────────────────────────
print("Pre-building meshes for all four viewports...")
lh_v1, rh_v1, lh_a1, rh_a1, sub_v1 = _make_viewport(
    0, 'GTEx Ground Truth',
    'Combat harmonized ground truth data used to fit and evaluate model')
lh_v2, rh_v2, lh_a2, rh_a2, sub_v2 = _make_viewport(
    1, 'Reconstruction',
    'LORO predicted values for sampled regions')
lh_v3, rh_v3, lh_a3, rh_a3, sub_v3 = _make_viewport(
    2, 'Whole-Brain Fit',
    'Ground truth data combined with imputed data over whole brain')
lh_v4, rh_v4, lh_a4, rh_a4, sub_v4 = _make_viewport(
    3, 'AHBA Reference',
    'Naive whole-brain fit with LORO reconstruction at sampled regions')

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
                 shared_clim=None) -> tuple:
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
def _get_all(gene: str, cbar_mode: str = 'local'):
    if cbar_mode in ('shared', 'wbf'):
        c1, s1 = data_loader.v1_gtex_harmonized(npz_data, atlas_aligned, gene)
    else:
        c1, s1 = data_loader.v1_gtex_input(npz_data, atlas_aligned, gene)
    c1, s1 = data_loader.remap_rh_to_lh_nearest(c1, s1, rh_to_lh_map)
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
state.cbar_mode       = 'local'
state.half_brain      = 'full'
state.cortex_alpha    = CORTEX_ALPHA
state.subcortex_alpha = SUBCORTEX_ALPHA
state.cerebellum_alpha = SUBCORTEX_ALPHA
def _push():
    try:
        ctrl.view_update()
    except Exception:
        pass


def _update_cbars(c1, s1, clim1, clim2, clim3, clim4, cbar_mode: str):
    state.v1_legend_html = _make_legend_html(c1, s1, clim1)
    state.cbar_html = _make_cbar_section_html(clim1, clim2, clim3, clim4, cbar_mode)


def _reload_scene(c1, s1, c2, s2, c3, s3, gene):
    c4, s4 = _get_ahba_ref(gene)
    sc = _shared_clim(gene) if state.cbar_mode == 'shared' else None
    clim1, clim2, clim3, clim4 = update_scene(c1, s1, c2, s2, c3, s3, c4, s4, gene, state.cbar_mode, sc)
    _update_cbars(c1, s1, clim1, clim2, clim3, clim4, state.cbar_mode)


@state.change('gene')
def on_gene(gene, **_):
    c1, s1, c2, s2, c3, s3 = _get_all(gene, state.cbar_mode)
    c4, s4 = _get_ahba_ref(gene)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    state.n_v4 = f"{len(c4)}cx+{len(s4)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, gene)
    _push()


@state.change('model')
def on_model(model, **_):
    global npz_data
    npz_data = _all_npz[model]
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene)
    _push()


@state.change('subject')
def on_subject(subject, **_):
    global npz_data, _all_npz
    _all_npz = {m: data_loader.load_subject(m, subject) for m in _MODEL_LIST}
    npz_data = _all_npz[state.model]
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode)
    c4, s4 = _get_ahba_ref(state.gene)
    state.n_v1 = f"{len(c1)}cx+{len(s1)}sub"
    state.n_v2 = f"{len(c2)}cx+{len(s2)}sub"
    state.n_v3 = f"{len(c3)}cx+{len(s3)}sub"
    state.n_v4 = f"{len(c4)}cx+{len(s4)}sub"
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene)
    _push()


@state.change('cbar_mode')
def on_cbar_mode(**_):
    c1, s1, c2, s2, c3, s3 = _get_all(state.gene, state.cbar_mode)
    _reload_scene(c1, s1, c2, s2, c3, s3, state.gene)
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


# ── 18. Trame UI ───────────────────────────────────────────────────────────────
LABEL_STYLE = 'color:#555; font-size:12px; margin-right:6px; white-space:nowrap;'
SEP_STYLE   = 'margin:0 18px; opacity:.3;'

with SinglePageLayout(server) as layout:
    layout.title.set_text('Brain Gene Expression Explorer')

    with layout.toolbar:
        v3.VSpacer()

        html.Span('Gene', style=LABEL_STYLE)
        v3.VSelect(
            v_model=('gene',),
            items=('gene_list',),
            density='compact',
            hide_details=True,
            variant='outlined',
            style='max-width:140px;',
        )

        v3.VDivider(vertical=True, style=SEP_STYLE)

        html.Span('Subject', style=LABEL_STYLE)
        v3.VSelect(
            v_model=('subject',),
            items=('subject_list',),
            density='compact',
            hide_details=True,
            variant='outlined',
            style='max-width:150px;',
        )

        v3.VDivider(vertical=True, style=SEP_STYLE)

        html.Span('Model', style=LABEL_STYLE)
        v3.VSelect(
            v_model=('model',),
            items=('model_list',),
            density='compact',
            hide_details=True,
            variant='outlined',
            style='max-width:110px;',
        )

        v3.VDivider(vertical=True, style=SEP_STYLE)

        with v3.VBtnToggle(
            v_model=('half_brain',),
            mandatory=True,
            density='compact',
            variant='outlined',
            divided=True,
        ):
            v3.VBtn('Full', value='full', size='small',
                    style='font-size:11px; text-transform:none; min-width:54px;')
            v3.VBtn('Half', value='half', size='small',
                    style='font-size:11px; text-transform:none; min-width:54px;')

        v3.VDivider(vertical=True, style=SEP_STYLE)

        html.Span('Scale', style=LABEL_STYLE)
        with v3.VBtnToggle(
            v_model=('cbar_mode',),
            mandatory=True,
            density='compact',
            variant='outlined',
            divided=True,
        ):
            v3.VBtn('Local',  value='local',  size='small',
                    style='font-size:11px; text-transform:none; min-width:54px;')
            v3.VBtn('Shared', value='shared', size='small',
                    style='font-size:11px; text-transform:none; min-width:60px;')
            v3.VBtn('WBF', value='wbf', size='small',
                    style='font-size:11px; text-transform:none; min-width:54px;')

        v3.VDivider(vertical=True, style=SEP_STYLE)

        _SLIM = 'color:#666; font-size:11px; margin:0 3px 0 6px; white-space:nowrap;'
        html.Span('Opacity', style=LABEL_STYLE)
        html.Span('Ctx', style=_SLIM)
        v3.VSlider(
            v_model=('cortex_alpha',),
            min=0.0, max=1.0, step=0.05,
            density='compact', hide_details=True,
            style='width:80px; margin:0 2px;',
        )
        html.Span('Sub', style=_SLIM)
        v3.VSlider(
            v_model=('subcortex_alpha',),
            min=0.0, max=1.0, step=0.05,
            density='compact', hide_details=True,
            style='width:80px; margin:0 2px;',
        )
        html.Span('Cer', style=_SLIM)
        v3.VSlider(
            v_model=('cerebellum_alpha',),
            min=0.0, max=1.0, step=0.05,
            density='compact', hide_details=True,
            style='width:80px; margin:0 2px;',
        )

        v3.VSpacer()

    with layout.content:
        with html.Div(style='display:flex; flex-direction:column; height:100%; background:white;'):
            # 3D viewports fill the available space
            with html.Div(style='flex:1; min-height:0;'):
                with plotter_ui(plotter, style='width:100%; height:100%; background:white;'):
                    pass
            # Bottom strip — colourbar gradients + V1 region swatches
            with html.Div(style='flex-shrink:0; border-top:1px solid #e0e0e0; background:#fafafa;'):
                # Row 1: colorbar(s) — layout adapts to Local / Shared mode
                with html.Div(style='padding:8px 14px 4px;'):
                    html.Div(v_html=('cbar_html',))
                # Row 2: V1 active region swatches
                with html.Div(style='padding:2px 14px 6px; display:flex; align-items:center; flex-wrap:wrap;'):
                    html.Span(
                        'GTEx Regions — ',
                        style='font-size:11px; color:#888; margin-right:6px; white-space:nowrap;',
                    )
                    html.Span(v_html=('v1_legend_html',))


if __name__ == '__main__':
    server.start(exec_mode='main', port=1234)
