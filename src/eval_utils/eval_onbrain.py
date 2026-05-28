"""On-brain rendering of TensorView slices.

nilearn glass-brain dot plots + yabplot cortical / subcortical surface renders for
single ``(subject, gene)`` slices of the gtex_gp TensorViews. This is the helper
home for ``eval_gxp_onbrain_views_copy.ipynb`` so that notebook stays config-first
(toggles + one-line calls).

Heavy deps (nilearn, pyvista, yabplot) are imported lazily inside the functions /
renderer that use them, so importing this module is cheap.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from .eval_style import model_label

__all__ = [
    "subject_gene_values", "slice_value_dict", "shared_clim", "region_coords", "search",
    "build_panels", "render_nilearn_row", "BrainSurfaceRenderer", "render_scope_row",
    "format_demo_caption", "render_subject_brain_grid",
]


# --------------------------------------------------------------------- slicing
def subject_gene_values(view, subject_id, gene):
    """1-D ``(n_regions,)`` values for one subject & gene; unobserved parcels -> NaN."""
    si = view.subjects.index(subject_id)
    gi = view.genes.index(gene)
    vals = view.values[si, :, gi].astype(float).copy()
    vals[~view.observed_mask[si, :]] = np.nan
    return vals


def slice_value_dict(view, subject_id, gene):
    """``{parcel_label: value}`` for one (subject, gene); observed parcels only."""
    return {lab: float(x)
            for lab, x in zip(view.regions, subject_gene_values(view, subject_id, gene))
            if np.isfinite(x)}


def shared_clim(value_arrays, pct=(2.0, 98.0)):
    """Robust ``(vmin, vmax)`` over all finite values — one scale for every panel."""
    allv = np.concatenate([v[np.isfinite(v)] for v in value_arrays]) if value_arrays else np.array([])
    if allv.size == 0:
        return (None, None)
    lo, hi = np.nanpercentile(allv, pct)
    if lo == hi:
        lo, hi = float(allv.min()), float(allv.max())
    return float(lo), float(hi)


def region_coords(prepost, regions):
    """``(n_regions, 3)`` MNI centroids (coord_x/y/z) reindexed to ``regions`` order."""
    lut = (prepost["target_meta"]
           .assign(_k=lambda d: d["tissue_or_parcel"].astype(str))
           .set_index("_k")[["coord_x", "coord_y", "coord_z"]])
    co = lut.reindex(list(regions)).to_numpy(float)
    if np.isnan(co).any():
        raise ValueError("some regions are missing MNI coords in target_meta")
    return co


def search(items, substr):
    """Case-insensitive substring filter — handy for finding gene / subject ids."""
    s = str(substr).lower()
    return [x for x in items if s in str(x).lower()]


# --------------------------------------------------------------------- panels
def build_panels(model_name, truth, recon, fused, ahba, gtex_subject, ahba_subject):
    """Standard 4-panel spec ``[(title, view, subject), ...]`` shared by both rows."""
    ml = model_label(model_name)
    return [
        (f"{ml} — held-out truth\n{gtex_subject}", truth, gtex_subject),
        (f"{ml} — LORO recon\n{gtex_subject}",     recon, gtex_subject),
        (f"{ml} — LORO fused\n{gtex_subject}",     fused, gtex_subject),
        (f"AHBA reference\n{ahba_subject}",        ahba,  ahba_subject),
    ]


def _shared_colorbar(fig, axes, vmin, vmax, cmap, label):
    sm = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.012, pad=0.01)
    cbar.set_label(label, fontsize=9)


# --------------------------------------------------------------------- nilearn
def render_nilearn_row(panels, gene, coords, *, vmin, vmax, cmap="viridis",
                       display_mode="x", node_size=45, figsize=(16, 3.6)):
    """One row of nilearn ``plot_markers`` glass-brain dots, shared scale + colorbar."""
    from nilearn import plotting
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (title, view, subject) in zip(axes, panels):
        vals = subject_gene_values(view, subject, gene)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            ax.set_title(f"{title}\n(no sampled parcels)", fontsize=9)
            ax.axis("off")
            continue
        plotting.plot_markers(vals[finite], coords[finite], node_size=node_size, node_cmap=cmap,
                              node_vmin=vmin, node_vmax=vmax, display_mode=display_mode, axes=ax,
                              colorbar=False, annotate=False, title=None)
        ax.set_title(title, fontsize=9)
    _shared_colorbar(fig, axes, vmin, vmax, cmap, f"{gene} — harmonized expression")
    fig.suptitle(f"{gene}: nilearn dots — truth / recon / fused vs AHBA (shared scale)",
                 fontsize=11, y=1.06)
    plt.show()
    plt.close(fig)   # return nothing: avoids Jupyter double-rendering the returned Figure


# --------------------------------------------------------------- yabplot surfaces
class BrainSurfaceRenderer:
    """Single-PyVista-scene cortex+subcortex renderer over the vendored 4S156 atlas.

    yabplot's ``plot_cortical`` / ``plot_subcortical`` each build their own figure and
    cannot overlay cortex-data with subcortex-data, so this ports the collaborator's
    ``BrainRenderer`` pattern: one ``pv.Plotter`` with the LH cortical surface
    (per-vertex parcel values) plus each subcortical VTK as a uniform-colored mesh.
    """

    # scope -> (camera, cortex_alpha, cortex_carries_data, show_subcortex, unsampled_cortex_alpha)
    # unsampled_cortex_alpha (5th, optional): when set + cortex carries data, sampled parcels render
    # at cortex_alpha while *unsampled* cortex fades to this lower alpha (translucent shell).
    SCOPE_CFG = {
        "cortical":               ("left_lateral", 0.88, True,  False, None),
        "subcortical_cerebellar": ("left_lateral", 0.12, False, True,  None),  # cortex = translucent shell
        "joint":                  ("left_lateral", 0.95, True,  True,  0.12),   # LH lateral; sampled clear, rest faded
    }

    def __init__(self, atlas_cache="data/brain_atlas_4s",
                 nan_color=(0.55, 0.55, 0.55), sub_alpha=1.0):
        import os
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        import pyvista as pv
        from yabplot.data import get_surface_paths
        from yabplot.utils import load_gii, parse_lut
        from yabplot.mesh import map_values_to_surface
        pv.OFF_SCREEN = True
        self._pv = pv
        self._map = map_values_to_surface
        self.nan_color = nan_color
        self.sub_alpha = sub_alpha

        cache = Path(atlas_cache)
        lh_path, rh_path = get_surface_paths("midthickness", "bmesh")   # fsLR32k; downloads once
        self.lh_v, lh_f = load_gii(lh_path)
        self.rh_v, rh_f = load_gii(rh_path)
        self.lh_faces = self._faces(lh_f)
        self.rh_faces = self._faces(rh_f)
        self.n_lh = int(len(self.lh_v))
        self.tar_labels = np.loadtxt(str(cache / "cortical" / "atlas.csv"), dtype=int)  # LH then RH
        self.lut_ids, _, self.lut_names, _ = parse_lut(str(cache / "cortical" / "atlas.txt"))
        # All subcortical meshes (LH-/RH-/Cerebellar-); render() filters by hemisphere.
        self.sub_meshes = {Path(p).stem: pv.read(p)
                           for p in sorted(glob.glob(str(cache / "subcortical" / "*.vtk")))}

    @staticmethod
    def _faces(f):
        f = np.asarray(f)
        n = f.shape[0]
        return np.hstack([np.full((n, 1), 3, np.int64), f]).ravel()

    def _camera(self, p, view, zoom):
        p.reset_camera()
        b = p.bounds
        cx, cy, cz = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])
        d = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.2
        pos = {"left_lateral":  (cx - d, cy, cz),   # LH lateral surface (camera at -x)
               "left_medial":   (cx + d, cy, cz),   # LH medial wall (camera at +x)
               "right_lateral": (cx + d, cy, cz),   # RH lateral surface (camera at +x)
               "right_medial":  (cx - d, cy, cz),   # RH medial wall (camera at -x)
               "superior":      (cx, cy, cz + d),
               "anterior":      (cx, cy + d, cz)}[view]
        up = (0, 1, 0) if view == "superior" else (0, 0, 1)
        p.camera_position = [pos, (cx, cy, cz), up]
        p.camera.zoom(zoom)

    def render(self, value_by_label, scope, *, vmin, vmax, cmap="viridis",
               window=(800, 800), zoom=1.6, camera=None, hemispheres="left"):
        """Render one scope to an RGB screenshot (for ``imshow``).

        ``hemispheres`` ∈ {'left','right','both'} selects which cortical surface(s) and
        which subcortical meshes are drawn. The atlas labels (LH_/RH_) place each parcel
        on its correct side, so 'both' is the anatomically-faithful full-brain view.
        """
        view0, cortex_alpha, cortex_data, show_sub, *rest = self.SCOPE_CFG[scope]
        unsampled_alpha = rest[0] if rest else None
        view = camera or view0
        pv = self._pv
        p = pv.Plotter(off_screen=True, window_size=list(window))
        p.set_background("white")
        light_kw = dict(show_scalar_bar=False, smooth_shading=True, lighting=True,
                        ambient=0.55, diffuse=0.45, specular=0.05)
        kw = dict(cmap=cmap, clim=(vmin, vmax), nan_color=self.nan_color, **light_kw)

        sides = {"left": ["L"], "right": ["R"], "both": ["L", "R"]}[hemispheres]
        vtx_vals = (self._map(value_by_label, self.tar_labels, self.lut_ids, self.lut_names)
                    if cortex_data else None)
        for side in sides:
            verts, faces, sl = ((self.lh_v, self.lh_faces, slice(0, self.n_lh)) if side == "L"
                                else (self.rh_v, self.rh_faces, slice(self.n_lh, None)))
            mesh = pv.PolyData(verts.astype(np.float32), faces)
            if cortex_data:
                vv = vtx_vals[sl].astype(np.float32)
                if unsampled_alpha is not None:
                    # explicit per-vertex RGBA: sampled parcels clear (cortex_alpha), unsampled
                    # cortex faded to a translucent shell (unsampled_alpha) + greyed nan_color.
                    finite = np.isfinite(vv)
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    rgba = plt.get_cmap(cmap)(np.clip(norm(np.where(finite, vv, vmin)), 0, 1)).astype(np.float32)
                    rgba[~finite, :3] = self.nan_color
                    rgba[finite, 3] = cortex_alpha
                    rgba[~finite, 3] = unsampled_alpha
                    mesh["rgba"] = (rgba * 255).astype(np.uint8)
                    p.add_mesh(mesh, scalars="rgba", rgba=True, **light_kw)
                else:
                    mesh["Data"] = vv
                    p.add_mesh(mesh, scalars="Data", opacity=cortex_alpha, **kw)
            else:
                p.add_mesh(mesh, color=self.nan_color, opacity=cortex_alpha,
                           smooth_shading=True, lighting=True)   # translucent context shell
        if show_sub:
            for name, raw in self.sub_meshes.items():
                if hemispheres == "left" and name.startswith("RH"):
                    continue
                if hemispheres == "right" and name.startswith("LH"):
                    continue
                m = raw.copy()
                m["Data"] = np.full(m.n_points, float(value_by_label.get(name, np.nan)), np.float32)
                p.add_mesh(m, scalars="Data", opacity=self.sub_alpha, **kw)
        self._camera(p, view, zoom)
        return p.screenshot(return_img=True, transparent_background=False)


def render_scope_row(renderer, scope, panels, gene, *, vmin, vmax, cmap="viridis",
                     zoom=1.6, camera=None, hemispheres="both", export_dir=None, figsize=(16, 4)):
    """One row of yabplot surface renders for ``scope``; optionally save the PNG.

    ``hemispheres`` ∈ {'both','left','right'} selects which cortical surface(s) +
    subcortical meshes are drawn (default 'both' = anatomically-faithful full brain,
    matching the PLS-gradient grid; LH_/RH_ atlas labels place each parcel on its side).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (title, view, subject) in zip(axes, panels):
        vbl = slice_value_dict(view, subject, gene)
        ax.imshow(renderer.render(vbl, scope, vmin=vmin, vmax=vmax, cmap=cmap, zoom=zoom,
                                  camera=camera, hemispheres=hemispheres))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    _shared_colorbar(fig, axes, vmin, vmax, cmap, f"{gene} — harmonized expression")
    cam = camera or renderer.SCOPE_CFG[scope][0]
    side = next((s for s in ("lateral", "medial", "superior", "anterior") if s in cam), cam)
    fig.suptitle(f"{gene}: yabplot [{scope}] (left {side}) — shared scale", fontsize=11, y=1.06)
    if export_dir:
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        gtex_subject = panels[0][2]
        fp = out / f"{scope}__{gene}__{gtex_subject}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print("saved", fp)
    plt.show()
    plt.close(fig)   # return nothing: avoids Jupyter double-rendering the returned Figure


# --------------------------------------------------- multi-subject brain grid
def format_demo_caption(meta_df, subject, *, prefix=None):
    """``{id}\\nage={age} · sex={sex}`` from a metadata frame (``subject/age/sex`` cols).

    Works for both ``prepost['gtex_raw']`` and ``prepost['ahba_raw']`` (one row per
    donor-tissue/parcel sample, so first non-null age/sex is taken). ``prefix``
    overrides the displayed id line.
    """
    sid = str(subject)
    age = sex = None
    if meta_df is not None:
        rows = meta_df[meta_df["subject"].astype(str) == sid]
        if len(rows):
            a = rows["age"].dropna()
            if len(a):
                try:
                    age = f"{float(a.iloc[0]):.0f}"
                except (ValueError, TypeError):
                    age = str(a.iloc[0])
            s = rows["sex"].dropna()
            if len(s):
                sex = str(s.iloc[0])
    return f"{prefix or sid}\nage={age or '?'} · sex={sex or '?'}"


def render_subject_brain_grid(renderer, view, gene, subjects, *, ncols=6, nrows=None,
                              scope="joint", cmap="viridis", hemispheres="left",
                              captions=None, clim_pct=(2.0, 98.0), zoom=1.0,
                              caption_fontscale=1.0, title=None, title_fontscale=None,
                              title_gap=0.08, wspace=None, hspace=None,
                              cbar_label=None, export_path=None, figsize_per=(2.6, 3.0)):
    """Grid of per-subject on-brain slices for one ``gene`` (4c ``joint`` styling).

    Renders ``subjects`` as a surface-brain grid (default ``joint`` scope, LH only:
    sampled parcels clear, unsampled cortex faded to a translucent shell — so coverage
    diversity is visible). The grid shape is adjustable: set ``ncols`` and/or ``nrows``
    (e.g. ``ncols=6, nrows=3`` for the 18-brain GTEx panel). If ``nrows`` is ``None`` it
    is derived from ``ncols``; if both are given the grid is exactly ``nrows x ncols``
    (extra cells are left blank, only the first ``nrows*ncols`` subjects are drawn).
    All panels share one robust color scale (``clim_pct``, shared across every brain) and
    one common colorbar (``cbar_label`` — pass e.g. ``f'{gene} expression'``).

    ``captions`` (``{subject: text}``, e.g. :func:`format_demo_caption`) is drawn above
    each brain; ``caption_fontscale`` sizes it (smaller avoids label-to-label collision
    when brains are packed). ``title_fontscale`` sizes the suptitle (defaults to
    ``caption_fontscale``); ``title_gap`` is the fraction of figure height reserved under
    it. ``wspace``/``hspace`` override the auto inter-panel spacing (raise ``wspace`` to
    separate brains). Ends with ``plt.show()`` (no return) to avoid double-rendering.
    """
    subjects = list(subjects)
    ncols = int(ncols) or 1
    if nrows is None:
        ncols = min(ncols, len(subjects)) or 1
        nrows = int(math.ceil(len(subjects) / ncols))
    else:
        nrows = int(nrows)
    subjects = subjects[: nrows * ncols]   # cap to the requested grid
    n = len(subjects)
    captions = captions or {}
    arrs = [subject_gene_values(view, s, gene) for s in subjects]
    vmin, vmax = shared_clim(arrs, pct=clim_pct)
    cap_fs = 11.0 * caption_fontscale

    # Auto spacing grows with caption size; either axis is overridable.
    extra = max(0.0, caption_fontscale - 1.0)
    if wspace is None:
        wspace = 0.04 + 0.22 * extra
    if hspace is None:
        hspace = 0.22 + 0.18 * extra
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(figsize_per[0] * (1 + 0.18 * extra) * ncols,
                                      figsize_per[1] * nrows),
                             gridspec_kw={"wspace": wspace, "hspace": hspace})
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")
        if idx >= n:
            continue
        s = subjects[idx]
        vbl = {lab: float(x) for lab, x in zip(view.regions, arrs[idx]) if np.isfinite(x)}
        ax.imshow(renderer.render(vbl, scope, vmin=vmin, vmax=vmax, cmap=cmap,
                                  zoom=zoom, hemispheres=hemispheres))
        ax.set_title(captions.get(s, str(s)), fontsize=cap_fs, linespacing=1.2)

    # Reserve a top band for the suptitle BEFORE the colorbar so the cbar spans the
    # adjusted axes height (keeps clear space under the main title).
    if title:
        fig.subplots_adjust(top=max(0.55, 1.0 - title_gap))

    sm = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    cbar.ax.tick_params(labelsize=cap_fs * 0.7)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=cap_fs * 0.95)
    cbar.outline.set_visible(False)

    if title:
        title_fs = 11.0 * (caption_fontscale if title_fontscale is None else title_fontscale) * 1.4
        fig.suptitle(title, fontsize=title_fs, fontweight="bold", y=1.0 - title_gap * 0.30)

    if export_path:
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, dpi=200, bbox_inches="tight")
        print("saved", export_path)
    plt.show()
    plt.close(fig)
