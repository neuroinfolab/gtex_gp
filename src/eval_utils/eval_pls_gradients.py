"""On-brain PLS gradient views (Vogel-style projection onto a frozen AHBA basis).

The reverse of the DLAM alignment evaluation: instead of learning a per-subject
basis-map that rotates a subject onto AHBA, we freeze the AHBA latent axes (LV1-3)
and project each GTEx full-brain reconstruction (``loro_fused``) onto them, asking how
naturally that brain falls onto the canonical AHBA gradient.

Pipeline
--------
1. ``fit_ahba_basis`` — donor-average the harmonized AHBA reference (parcels x genes),
   fit the two-block PLS (X=genes, Y=[|coord_x|, coord_y, coord_z]) via
   :func:`src.latent.pls.fit_pls_basis`, and return the frozen projector + per-parcel
   reference scores ``T_ahba``. Optionally pickle-cached.
2. ``project_subject`` — project a subject's ``loro_fused`` slice onto that basis -> ``T_gtex``.
3. ``select_subject_by_recon_percentile`` — pick a subject at a recon-quality percentile.
4. ``render_pls_gradient_grid`` — LV rows x [GTEx sag, GTEx cor, AHBA sag, AHBA cor, scatter].

Brain panels reuse :class:`eval_onbrain.BrainSurfaceRenderer` (``joint`` scope).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..latent.pls import fit_pls_basis, project_pls_scores
from ..spatial.model_coords import model_spatial_coords
from .eval_style import model_label
from .eval_onbrain import BrainSurfaceRenderer  # noqa: F401  (re-exported for convenience)

__all__ = [
    "build_cfg_from_cache", "pls_spatial_Y", "fit_ahba_basis", "project_subject",
    "select_subject_by_recon_percentile", "select_subjects_by_recon_percentiles",
    "format_subject_header", "format_subject_caption",
    "render_pls_gradient_grid", "render_pls_gradient_many_brains",
]


# --------------------------------------------------------------- config from cache
def build_cfg_from_cache(cache_root, model, gene_scope):
    """Reconstruct ``EDAConfig`` from the *cache's own* stored config.

    Reads a per-subject JSON sidecar under ``{cache_root}/{gene_scope}/{model}/`` and
    pulls the matching / harmonization hparams that were actually used to fit the cache,
    so the views/basis built here can't drift from the model fit. Returns
    ``(cfg, cfg_dict)`` — ``cfg_dict`` is the raw stored config (has ``n_comp_target`` etc.).
    Asserts all subjects in the dir share one ``cfg_hash``.
    """
    import json
    from .results_eda import EDAConfig

    d = Path(cache_root) / gene_scope / model
    sidecars = sorted(d.glob("*.json"))
    if not sidecars:
        raise FileNotFoundError(f"no cache JSON sidecars under {d}")
    # cfg_hash carries per-subject bits, so check the matching/harmonization fields that
    # must agree across the whole cache for the views/basis to line up with the fit.
    fit_keys = ("matching_policy", "matching_policy_hemi_mode", "gtex_rep_mode", "gtex_hemi_mode",
                "collapse_cerebellum", "gene_scope", "min_observed_parcels",
                "combat_use_covariates", "csv_path", "n_comp_target")
    configs = [json.loads(p.read_text())["config"] for p in sidecars]
    subsets = {tuple((k, c.get(k)) for k in fit_keys) for c in configs}
    if len(subsets) > 1:
        raise RuntimeError(f"cache {d} mixes {len(subsets)} distinct fit configs across subjects")
    c = configs[0]
    cfg = EDAConfig(
        csv_path=c["csv_path"],
        cache_root=str(cache_root),
        gene_scope=c.get("gene_scope", gene_scope),
        min_observed_parcels=c["min_observed_parcels"],
        combat_use_covariates=c["combat_use_covariates"],
        drop_macro_system_covariate=c.get("drop_macro_system_covariate", False),
        gtex_rep_mode=c["gtex_rep_mode"],
        gtex_hemi_mode=c["gtex_hemi_mode"],
        matching_policy=c["matching_policy"],
        matching_policy_hemi_mode=c["matching_policy_hemi_mode"],
        collapse_cerebellum=c["collapse_cerebellum"],
    )
    return cfg, c


# ------------------------------------------------------------------ AHBA basis
def pls_spatial_Y(xyz):
    """Folded model spatial target ``[|x|, y, z]`` from raw ``[x, y, z]``."""
    return model_spatial_coords(xyz, fold_hemispheres=True)


def fit_ahba_basis(prepost, *, genes=None, n_comp=3, cache_path=None):
    """Freeze the AHBA latent basis — the *same reference DLAM uses*.

    Mirrors ``dlam_diagnostics._fit_dlam_fold_payload`` exactly: the harmonized AHBA
    region matrix via ``build_region_matrix(prepost['ahba_h'], genes, target_meta)`` and
    the spatial target ``[|coord_x|, coord_y, coord_z]``, fit with the two-block PLS.
    Returns ``(basis, T_ahba, regions)`` in ``target_meta.parcel_idx`` order;
    ``T_ahba`` (== DLAM's ``t_ref_full``) is ``(n_parcels, n_comp)``, NaN where AHBA is
    unsampled. ``cache_path`` (if set) pickles the result and is reused when regions match.
    """
    from ..preprocess import build_region_matrix
    tm = prepost["target_meta"].sort_values("parcel_idx")
    regions = tm["tissue_or_parcel"].astype(str).tolist()
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("regions") == regions:
            return payload["basis"], payload["T_ahba"], regions

    genes = list(genes if genes is not None else prepost["genes"])
    X, _obs = build_region_matrix(prepost["ahba_h"], genes, prepost["target_meta"], agg="mean")  # (R, G)
    Y = pls_spatial_Y(tm[["coord_x", "coord_y", "coord_z"]].to_numpy(float))                     # (R, 3)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    basis = fit_pls_basis(X[ok], Y[ok], n_comp_target=n_comp, adaptive=True)
    T_ahba = project_pls_scores(basis, X)            # (R, n_comp); NaN where AHBA unsampled

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"basis": basis, "T_ahba": T_ahba, "regions": regions}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return basis, T_ahba, regions


def project_subject(basis, gtex_view, subject_id):
    """Project one subject's full-brain reconstruction onto the AHBA basis.

    Returns ``(n_regions, n_comp)`` scores aligned to ``gtex_view.regions``.
    """
    si = gtex_view.subjects.index(subject_id)
    Xg = gtex_view.values[si].astype(float)          # (R, G), dense for loro_fused
    return project_pls_scores(basis, Xg)


# --------------------------------------------------------------- subject pick
def select_subject_by_recon_percentile(cfg, model, percentile=50.0, metric="pearson_r"):
    """Pick the subject at ``percentile`` of recon quality (default Pearson r).

    Returns ``(subject_id, metric_value, metrics_df_sorted)``. Higher Pearson r / R² is
    better; for RMSE the percentile is taken on the ascending (worse->better is reversed)
    scale, i.e. percentile 100 = best.
    """
    from .results_eda import compute_subject_metrics_from_cache
    df = compute_subject_metrics_from_cache(cfg, model=str(model)).dropna(subset=[metric]).copy()
    if len(df) == 0:
        raise RuntimeError(f"no valid {model} subject metrics for {metric!r}")
    ascending = (metric != "rmse")  # so that higher percentile == better recon
    df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    idx = int(round((float(percentile) / 100.0) * (len(df) - 1)))
    idx = max(0, min(idx, len(df) - 1))
    row = df.iloc[idx]
    return str(row["subject"]), float(row[metric]), df


def _subject_ages(prepost):
    """Map ``subject_id -> float age`` from ``prepost['gtex_raw']`` (drops non-numeric)."""
    g = prepost.get("gtex_raw") if hasattr(prepost, "get") else None
    if g is None:
        return {}
    ages = {}
    for sid, age in zip(g["subject"].astype(str), g["age"]):
        try:
            ages[sid] = float(age)
        except (ValueError, TypeError):
            continue
    return ages


def select_subjects_by_recon_percentiles(cfg, model, n=5, metric="pearson_r",
                                          mode="percentile", seed=0, prepost=None):
    """Pick ``n`` subjects spanning a chosen axis, ordered left->right along it.

    - ``mode='percentile'`` (default): evenly spaced recon-quality percentiles
      (n=5 -> p10/30/50/70/90: centers of ``n`` equal-width bins, avoiding the tails),
      ordered worst->best.
    - ``mode='random'``: ``n`` subjects drawn uniformly at random (seeded); columns
      appear in the random draw order (not recon-sorted).
    - ``mode='age'``: ``n`` subjects whose ages are nearest to evenly-spaced targets
      across the observed age range (``linspace(min_age, max_age, n)``), ordered
      young->old. Requires ``prepost`` (ages from ``prepost['gtex_raw']``).

    Returns ``(subjects, scores, metrics_df_sorted)`` — ``scores`` is the recon
    ``metric`` value for each chosen subject (so captions still show r), in column order.
    """
    from .results_eda import compute_subject_metrics_from_cache
    df = compute_subject_metrics_from_cache(cfg, model=str(model)).dropna(subset=[metric]).copy()
    if len(df) == 0:
        raise RuntimeError(f"no valid {model} subject metrics for {metric!r}")
    ascending = (metric != "rmse")  # higher row index == better recon
    df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    n = max(1, min(int(n), len(df)))
    mode = str(mode).lower()

    if mode == "age":
        if prepost is None:
            raise ValueError("mode='age' requires prepost (for subject ages)")
        ages = _subject_ages(prepost)
        pool = [(sid, ages[sid]) for sid in df["subject"].astype(str) if sid in ages]
        if not pool:
            raise RuntimeError("no subjects with numeric age in the metric pool")
        n = min(n, len(pool))
        sids = np.array([p[0] for p in pool])
        av = np.array([p[1] for p in pool], dtype=float)
        targets = np.linspace(av.min(), av.max(), n)
        chosen, used = [], set()
        for t in targets:
            order = np.argsort(np.abs(av - t))            # nearest age to target
            for k in order:
                if int(k) not in used:
                    used.add(int(k)); chosen.append(int(k)); break
        chosen = sorted(chosen, key=lambda k: av[k])       # young -> old
        subjects = [str(sids[k]) for k in chosen]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(df), size=n, replace=False)  # draw order = column order (seeded)
        subjects = [str(s) for s in df.iloc[sel]["subject"]]
    else:
        # centers of n equal-width percentile bins: (i+0.5)/n for i in 0..n-1
        pct = (np.arange(n) + 0.5) / n
        sel = np.unique(np.clip(np.round(pct * (len(df) - 1)).astype(int), 0, len(df) - 1))
        if len(sel) < n:  # de-dup collisions at small N by filling nearest free index
            free = [i for i in range(len(df)) if i not in set(sel)]
            sel = np.sort(np.concatenate([sel, np.array(free[: n - len(sel)], dtype=int)]))
        subjects = [str(s) for s in df.iloc[sel]["subject"]]

    mval = dict(zip(df["subject"].astype(str), df[metric].astype(float)))
    scores = [float(mval.get(s, np.nan)) for s in subjects]
    return (subjects, scores, df)


# -------------------------------------------------------------------- plotting
def format_subject_header(prepost, metrics_df, subject, model_name, metric="pearson_r", value=None):
    """Build the GTEx-column header block: ``{MODEL} — {subject}\\n(age=..; sex=..; coverage=..; recon ..)``.

    Age/sex come from ``prepost['gtex_raw']``; coverage and (if not given) the recon
    metric value come from ``metrics_df`` (the DataFrame returned by
    :func:`select_subject_by_recon_percentile`).
    """
    sid = str(subject)
    age = sex = cov = None
    g = prepost.get("gtex_raw") if hasattr(prepost, "get") else None
    if g is not None:
        row = g[g["subject"].astype(str) == sid]
        if len(row):
            try:
                age = f"{float(row['age'].iloc[0]):.0f}"
            except (ValueError, TypeError):
                age = str(row["age"].iloc[0])
            sex = str(row["sex"].iloc[0])
    if metrics_df is not None:
        mr = metrics_df[metrics_df["subject"].astype(str) == sid]
        if len(mr):
            if "coverage" in mr:
                cov = int(mr["coverage"].iloc[0])
            if value is None and metric in mr:
                value = float(mr[metric].iloc[0])
    val_s = f"{value:.2f}" if value is not None else "?"
    return (f"{model_label(model_name)} — {sid}\n"
            f"age={age or '?'} ; sex={sex or '?'}\n"
            f"cov={cov if cov is not None else '?'} ; r={val_s}")


def format_subject_caption(prepost, metrics_df, subject, metric="pearson_r", value=None):
    """Two-line compact per-subject caption: ``{subject}\\n{age}{sex} · r={val}``.

    For the many-brain grid where the full :func:`format_subject_header` block is too
    heavy across many columns. Age/sex from ``prepost['gtex_raw']``; recon value from
    ``metrics_df`` if not given.
    """
    sid = str(subject)
    age = sex = None
    g = prepost.get("gtex_raw") if hasattr(prepost, "get") else None
    if g is not None:
        row = g[g["subject"].astype(str) == sid]
        if len(row):
            try:
                age = f"{float(row['age'].iloc[0]):.0f}"
            except (ValueError, TypeError):
                age = str(row["age"].iloc[0])
            sex = str(row["sex"].iloc[0])
    if value is None and metrics_df is not None:
        mr = metrics_df[metrics_df["subject"].astype(str) == sid]
        if len(mr) and metric in mr:
            value = float(mr[metric].iloc[0])
    demo = f"{age or '?'}{sex or ''}"
    val_s = f"{value:.2f}" if value is not None else "?"
    return f"{sid}\n{demo} · r={val_s}"


def _value_dict(regions, scores_1d):
    return {regions[r]: float(scores_1d[r]) for r in range(len(regions)) if np.isfinite(scores_1d[r])}


# DLAM diagnostics' per-LV latent palette (LV1=cividis, LV2=plasma, LV3=viridis).
_COMPONENT_CMAPS = ("cividis", "plasma", "viridis")


def _lv_cmap(cmap, lv):
    """Resolve the colormap for LV ``lv``.

    ``cmap='auto'`` (or 'component') follows DLAM diagnostics' sequential per-LV
    palette; any other value is a fixed cmap (e.g. diverging 'RdBu_r') used for all LVs.
    Returns ``(cmap_name, diverging)`` — diverging maps get symmetric color limits.
    """
    if str(cmap).lower() in ("auto", "component"):
        return _COMPONENT_CMAPS[lv % len(_COMPONENT_CMAPS)], False
    return str(cmap), True


def render_pls_gradient_grid(renderer, T_gtex, T_ahba, regions, subject_id, *,
                             model_name="dlam", gtex_header=None, n_lv=3, cmap="RdBu_r",
                             sagittal_face="left_lateral", hemispheres="both", zoom=1.9,
                             clim_pct=98.0, font_scale=1.8, export_dir=None, figsize_per=(4.2, 4.0)):
    """LV rows x [GTEx sagittal, AHBA sagittal, AHBA-vs-GTEx scatter].

    ``gtex_header`` is the title over the GTEx column (e.g. the subject info block from
    :func:`format_subject_header`); defaults to ``"{model} — {subject}"``. ``cmap`` is a
    fixed diverging map ('RdBu_r', symmetric limits) or ``'auto'`` to follow DLAM
    diagnostics' per-LV palette (cividis/plasma/viridis, percentile limits). Scatter
    points colored by the GTEx score. ``font_scale`` scales all text.
    """
    ncomp = int(min(n_lv, T_gtex.shape[1], T_ahba.shape[1]))
    fs_head = 14.0 * font_scale    # subject block + column headers (unified, formal)
    fs_label = 12.0 * font_scale
    fs_row = 17.0 * font_scale
    fs_tick = 10.0 * font_scale
    if gtex_header is None:
        gtex_header = f"{model_label(model_name)} — {subject_id}"

    # Two subfigures: brains tight on the left, scatters (with their own y-axis room) on the right.
    fig = plt.figure(figsize=(figsize_per[0] * 3.4, figsize_per[1] * ncomp))
    sf_brain, sf_scatter = fig.subfigures(1, 2, width_ratios=[2.0, 1.2], wspace=0.0)
    axb = np.atleast_2d(sf_brain.subplots(ncomp, 2, gridspec_kw={"wspace": 0.02, "hspace": 0.10}))
    axs = np.atleast_1d(sf_scatter.subplots(ncomp, 1, gridspec_kw={"hspace": 0.40}))

    for lv in range(ncomp):
        g, a = T_gtex[:, lv], T_ahba[:, lv]
        cmap_lv, diverging = _lv_cmap(cmap, lv)
        both = np.concatenate([g[np.isfinite(g)], a[np.isfinite(a)]])
        if both.size and diverging:
            m = float(np.nanpercentile(np.abs(both), clim_pct)) or 1.0
            vmin, vmax = -m, m
        elif both.size:
            vmin = float(np.nanpercentile(both, 100.0 - clim_pct))
            vmax = float(np.nanpercentile(both, clim_pct))
        else:
            vmin, vmax = -1.0, 1.0
        gdict, adict = _value_dict(regions, g), _value_dict(regions, a)

        # brain panels — GTEx (col 0), AHBA (col 1), drawn close together
        for col, vd in enumerate((gdict, adict)):
            axb[lv, col].imshow(renderer.render(vd, "joint", vmin=vmin, vmax=vmax,
                                                cmap=cmap_lv, zoom=zoom, camera=sagittal_face,
                                                hemispheres=hemispheres))
            axb[lv, col].axis("off")
        if lv == 0:
            axb[0, 0].set_title(gtex_header, fontsize=fs_head, fontweight="bold", linespacing=1.4)
            axb[0, 1].set_title("AHBA ref", fontsize=fs_head, fontweight="bold")
        axb[lv, 0].text(-0.10, 0.5, f"LV{lv + 1}", transform=axb[lv, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=fs_row, fontweight="bold")

        # scatter: AHBA (x) vs GTEx (y), colored by GTEx score
        ax = axs[lv]
        fin = np.isfinite(g) & np.isfinite(a)
        sc = ax.scatter(a[fin], g[fin], c=g[fin], cmap=cmap_lv, vmin=vmin, vmax=vmax,
                        s=34, edgecolor="k", linewidth=0.3, zorder=3)
        if int(fin.sum()) > 2:
            r = float(np.corrcoef(a[fin], g[fin])[0, 1])
            lo = float(min(a[fin].min(), g[fin].min())); hi = float(max(a[fin].max(), g[fin].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1.0, zorder=1)
        else:
            r = np.nan
        if lv == 0:
            ax.set_title("Molecular scores", fontsize=fs_head, fontweight="bold")
        ax.set_xlabel("AHBA score", fontsize=fs_label)
        ax.set_ylabel("GTEx score", fontsize=fs_label)
        ax.set_box_aspect(1)   # square panel (box, not data-equal)
        ax.set_anchor("W")     # left-justify the square so it sits close to the brains
        ax.tick_params(labelsize=fs_tick)
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.05, 0.95, f"r = {r:.2f}\nn = {int(fin.sum())}", transform=ax.transAxes,
                va="top", ha="left", fontsize=fs_label,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.8))
        cbar = sf_scatter.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(labelsize=fs_tick)
        cbar.outline.set_visible(False)

    if export_dir:
        out = Path(export_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / f"pls_gradients__{model_name}__{subject_id}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight"); print("saved", fp)
    plt.show()
    plt.close(fig)


def render_pls_gradient_many_brains(renderer, T_ahba, subject_Ts, regions, *,
                                    model_name="dlam", subject_captions=None, n_lv=3,
                                    cmap="RdBu_r", sagittal_face="left_lateral",
                                    hemispheres="both", zoom=1.9, clim_pct=98.0,
                                    clim_pad=1.05, font_scale=1.5, export_dir=None,
                                    figsize_per=(2.7, 2.7)):
    """Many-brain comparison: LV rows x [AHBA ref | subj_1 ... subj_n].

    A minimally-labeled extension of :func:`render_pls_gradient_grid` focused entirely
    on the brains (no scatter). ``subject_Ts`` is an ordered mapping ``{subject_id: T}``
    (each ``T`` is parcels x LV, from :func:`project_subject`); columns are drawn
    left->right in that order, after the AHBA ref column on the far left. ``cmap``
    defaults to diverging ``'RdBu_r'``; per-LV color limits are **anchored to the AHBA
    ref** (symmetric ``±clim_pad·percentile(|ref|, clim_pct)``) so every subject is read
    relative to the reference. One slim colorbar per LV row at the right.
    ``subject_captions`` (``{subject_id: text}``) gives the compact per-subject titles
    (see :func:`format_subject_caption`); falls back to the bare id.
    """
    subjects = list(subject_Ts.keys())
    ncomp = int(min(n_lv, T_ahba.shape[1], *(T.shape[1] for T in subject_Ts.values())))
    n_sub = len(subjects)
    subject_captions = subject_captions or {}
    fs_head = 11.0 * font_scale
    fs_cap = 8.5 * font_scale
    fs_row = 14.0 * font_scale
    fs_tick = 7.5 * font_scale

    # Columns: 0=AHBA ref | 1=divider spacer | 2..2+n_sub-1=subjects | last=colorbar.
    C_REF, C_GAP, C_SUB0 = 0, 1, 2
    C_CBAR = C_SUB0 + n_sub
    n_col = C_CBAR + 1
    gap, cbar_w = 0.18, 0.10
    wr = [1.0, gap] + [1.0] * n_sub + [cbar_w]

    fig = plt.figure(figsize=(figsize_per[0] * (n_sub + 1.4), figsize_per[1] * ncomp))
    axg = np.atleast_2d(fig.subplots(ncomp, n_col,
                                     gridspec_kw={"width_ratios": wr,
                                                  "wspace": 0.04, "hspace": 0.10}))

    for lv in range(ncomp):
        a = T_ahba[:, lv]
        cmap_lv, _ = _lv_cmap(cmap, lv)
        ref = a[np.isfinite(a)]
        m = (float(np.nanpercentile(np.abs(ref), clim_pct)) * clim_pad) if ref.size else 1.0
        m = m or 1.0
        vmin, vmax = -m, m

        # divider spacer column: a single dashed vertical rule between ref and subjects.
        axg[lv, C_GAP].axis("off")
        axg[lv, C_GAP].axvline(0.5, color="0.6", lw=1.0, ls="--")

        col_specs = [(C_REF, a)] + [(C_SUB0 + j, subject_Ts[s][:, lv]) for j, s in enumerate(subjects)]
        for col, vals in col_specs:
            axg[lv, col].imshow(
                renderer.render(_value_dict(regions, vals), "joint", vmin=vmin, vmax=vmax,
                                cmap=cmap_lv, zoom=zoom, camera=sagittal_face,
                                hemispheres=hemispheres))
            axg[lv, col].axis("off")

        # column titles (top row only): ref + per-subject compact captions
        if lv == 0:
            axg[0, C_REF].set_title("AHBA ref", fontsize=fs_head, fontweight="bold")
            for j, s in enumerate(subjects):
                axg[0, C_SUB0 + j].set_title(subject_captions.get(s, s),
                                             fontsize=fs_cap, linespacing=1.3)
        # LV row label on the far-left edge
        axg[lv, C_REF].text(-0.14, 0.5, f"LV{lv + 1}", transform=axg[lv, C_REF].transAxes,
                            rotation=90, va="center", ha="right", fontsize=fs_row, fontweight="bold")

        # one slim shared colorbar per row (anchored to the ref-derived clim).
        # Brains are pre-rendered RGBA, so build an explicit ScalarMappable for the scale.
        axg[lv, C_CBAR].axis("off")
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_lv)
        cbar = fig.colorbar(sm, ax=axg[lv, C_CBAR], fraction=0.9, pad=0.02)
        cbar.ax.tick_params(labelsize=fs_tick)
        cbar.outline.set_visible(False)

    if export_dir:
        out = Path(export_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / f"pls_gradients_many__{model_name}__n{len(subjects)}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight"); print("saved", fp)
    plt.show()
    plt.close(fig)
