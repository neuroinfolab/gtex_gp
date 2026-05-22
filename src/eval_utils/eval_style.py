#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["naive", "dlam", "plam"]
MODEL_COLORS = {"naive": "#7f7f7f", "dlam": "#1f77b4", "plam": "#d62728"}
MODEL_LABELS = {"naive": "Naive", "dlam": "DLAM", "plam": "PLAM"}

# Legacy element-keyed font map; preserved for plotters that still use the
# `FONT["title"] + N` pattern. New code should use the token system below
# (FONT_TOKENS, font_size, _resolve_fonts) instead.
FONT = {"title": 11, "label": 10, "tick": 9, "legend": 9, "small": 8}

# ---------------------------------------------------------------------------
# Font-size token system
#
# Plotters declare a `_DEFAULT_FONT_SIZES` dict mapping element names to
# tokens (e.g. "title": "xl"). Each plotter accepts an optional
# `font_sizes={…}` override that merges over its defaults. Tokens are
# resolved through `font_size()`, which accepts:
#   - a token string ("xs", "s", "m", "l", "xl", "xxl")
#   - a token + offset string ("m+1", "xl-2")
#   - an absolute integer (passes through)
# A global `set_font_scale(scale)` multiplies every resolved size, so a
# single knob rescales every plot at once.
#
# Standard element keys (use these names so muscle memory transfers across
# plotters): title, xlabel, ylabel, tick, legend, legend_title, annotation,
# footer, cbar_label, cbar_tick.
# ---------------------------------------------------------------------------

FONT_TOKENS: dict[str, float] = {
    "xs":  7.0,
    "s":   9.0,
    "m":  10.0,
    "l":  12.0,
    "xl": 14.0,
    "xxl": 16.0,
}

_FONT_SCALE: float = 1.0
_TOKEN_RE = re.compile(r"^\s*([a-zA-Z]+)\s*([+\-]\s*\d+(?:\.\d+)?)?\s*$")


def font_size(spec: int | float | str) -> int:
    """Resolve a font-size spec to an integer point size.

    Accepts:
      - int/float: returned as int (multiplied by the global font scale).
      - "<token>" or "<token>±<offset>" string: resolved via FONT_TOKENS,
        offset added before scaling.

    Examples: ``font_size("m") == 10`` (at scale 1.0), ``font_size("l+2") == 14``,
    ``font_size("xl-1") == 13``, ``font_size(13) == 13``.
    """
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return max(1, int(round(float(spec) * _FONT_SCALE)))
    if not isinstance(spec, str):
        raise TypeError(f"font_size: unsupported spec type {type(spec).__name__}")
    m = _TOKEN_RE.match(spec)
    if not m:
        raise ValueError(f"font_size: cannot parse spec {spec!r} (use 'token', 'token+N', or int)")
    token = m.group(1).lower()
    if token not in FONT_TOKENS:
        raise ValueError(f"font_size: unknown token {token!r} (known: {sorted(FONT_TOKENS)})")
    base = FONT_TOKENS[token]
    offset_txt = (m.group(2) or "").replace(" ", "")
    offset = float(offset_txt) if offset_txt else 0.0
    return max(1, int(round((base + offset) * _FONT_SCALE)))


def set_font_scale(scale: float) -> None:
    """Globally scale every token-resolved font size by `scale`.

    Call from a notebook to scale every plot at once (e.g. for figure
    presentations). Re-runs `set_academic_style()` so matplotlib's rcParam
    defaults pick up the new sizes too.
    """
    global _FONT_SCALE
    _FONT_SCALE = max(0.1, float(scale))
    set_academic_style()


def _resolve_fonts(
    defaults: Mapping[str, int | float | str],
    override: Mapping[str, int | float | str] | None = None,
) -> dict[str, int]:
    """Merge override over defaults and resolve every entry to an int pt size."""
    merged: dict[str, int | float | str] = dict(defaults)
    if override:
        for key, value in override.items():
            merged[key] = value
    return {key: font_size(value) for key, value in merged.items()}
DISPLAY_LABEL_PREFIXES_TO_STRIP = ("Brain - ",)
GTEX_DISPLAY_LABEL_OVERRIDES = {
    "cortex": "Cortex (BA10)",
}
PARCEL_GROUP_ORDER = ["cortical", "basal_ganglia", "limbic_midbrain", "cerebellar", "other"]
PARCEL_GROUP_COLORS = {
    "cortical": ["#2166ac", "#4393c3", "#92c5de"],
    "basal_ganglia": ["#b35806", "#f1a340", "#fdb863"],
    "limbic_midbrain": ["#762a83", "#c51b7d", "#8c510a", "#bf812d"],
    "cerebellar": ["#1b7837", "#5aae61", "#a6dba0"],
    "other": ["#6b6b6b", "#969696", "#bdbdbd"],
}

# Region-stratified evaluation colors. These are the manuscript-facing color
# families used by population scatters and embedding diagnostics: cortical
# orange/brown, subcortical blue/purple, cerebellar green, other gray.
REGION_SCATTER_GROUP_ORDER = ["cortical", "subcortical", "cerebellar", "other"]
REGION_SCATTER_GROUP_COLORS = {
    "cortical": ["#b35806", "#e08214", "#f1a340", "#fdb863", "#7f3b08"],
    "subcortical": ["#2166ac", "#4393c3", "#92c5de", "#762a83", "#9970ab", "#c2a5cf"],
    "subcortical_basal_ganglia": ["#762a83", "#9970ab", "#c2a5cf", "#40004b", "#8e0152"],
    "subcortical_other": ["#2166ac", "#4393c3", "#92c5de", "#053061", "#67a9cf"],
    "cerebellar": ["#1b7837", "#5aae61", "#a6dba0", "#00441b", "#7fbf7b"],
    "other": ["#6b6b6b", "#969696", "#bdbdbd", "#525252"],
}
REGION_SCATTER_GROUP_BASE = {
    "cortical": "#e08214",
    "subcortical": "#2166ac",
    "cerebellar": "#1b7837",
    "other": "#6b6b6b",
}


def strip_display_label_prefixes(name: str) -> str:
    s = str(name)
    for prefix in DISPLAY_LABEL_PREFIXES_TO_STRIP:
        if s.lower().startswith(prefix.lower()):
            return s[len(prefix) :]
    return s


def collapse_macro_system_to_region_group(macro_system: object) -> str:
    s = str(macro_system).strip().lower()
    if s == "cerebellar":
        return "cerebellar"
    if s in {"subcortical", "basal_ganglia", "limbic_midbrain"}:
        return "subcortical"
    if s in {"unknown", "nan", "none", ""}:
        return "other"
    return "cortical"


def scatter_subcortical_palette_key(region: str) -> str:
    s = str(region).lower()
    if any(k in s for k in ("caudate", "putamen", "accumbens", "pallid")):
        return "subcortical_basal_ganglia"
    return "subcortical_other"


def region_scatter_palette(
    regions: Sequence[str],
    *,
    region_group_lookup: Mapping[str, str] | None = None,
    weights: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Palette for parcel/region labels matching region-stratified eval plots."""
    rset = list({str(r) for r in regions})
    if region_group_lookup is None:
        group_lookup = {
            r: collapse_macro_system_to_region_group(parcel_label_group(r))
            for r in rset
        }
    else:
        group_lookup = {r: str(region_group_lookup.get(r, "other")) for r in rset}

    palette: dict[str, object] = {}
    counters: dict[str, int] = {}
    for group in REGION_SCATTER_GROUP_ORDER:
        group_regions = [r for r in rset if group_lookup.get(r, "other") == group]
        if weights is not None:
            group_regions = sorted(group_regions, key=lambda r: (-int(weights.get(r, 0)), format_legend_label(r)))
        else:
            group_regions = sorted(group_regions, key=format_legend_label)
        for region in group_regions:
            palette_key = scatter_subcortical_palette_key(region) if group == "subcortical" else group
            colors = REGION_SCATTER_GROUP_COLORS.get(palette_key, REGION_SCATTER_GROUP_COLORS["other"])
            j = counters.get(palette_key, 0)
            palette[region] = colors[j % len(colors)]
            counters[palette_key] = j + 1

    leftover = [r for r in rset if r not in palette]
    for region in sorted(leftover, key=format_legend_label):
        colors = REGION_SCATTER_GROUP_COLORS["other"]
        j = counters.get("other", 0)
        palette[region] = colors[j % len(colors)]
        counters["other"] = j + 1
    return palette


def ordered_region_scatter_values(
    regions: Sequence[str],
    *,
    region_group_lookup: Mapping[str, str] | None = None,
    weights: Mapping[str, int] | None = None,
) -> list[str]:
    raw = list({str(r) for r in regions})
    if region_group_lookup is None:
        group_lookup = {
            r: collapse_macro_system_to_region_group(parcel_label_group(r))
            for r in raw
        }
    else:
        group_lookup = {r: str(region_group_lookup.get(r, "other")) for r in raw}
    out: list[str] = []
    for group in REGION_SCATTER_GROUP_ORDER:
        vals = [r for r in raw if group_lookup.get(r, "other") == group]
        if weights is not None:
            vals = sorted(vals, key=lambda r: (-int(weights.get(r, 0)), format_legend_label(r)))
        else:
            vals = sorted(vals, key=format_legend_label)
        out.extend(vals)
    out.extend(sorted([r for r in raw if r not in out], key=format_legend_label))
    return out


_TICK_RC = {
    # whitegrid otherwise hides tick marks. Force them on globally so
    # every plot in the workspace has axis ticks by default.
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.major.pad": 3.0,
    "ytick.major.pad": 3.0,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.0,
}


# Apply at import so ticks are on even before set_academic_style() is called.
plt.rcParams.update(_TICK_RC)


def set_academic_style() -> None:
    # Pass tick overrides through `sns.set_theme(rc=...)` so they are part of
    # the resolved style and won't be clobbered by seaborn's whitegrid defaults.
    sns.set_theme(style="whitegrid", context="paper", rc=dict(_TICK_RC))
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            # Token-resolved defaults; respect the global font scale.
            "axes.titlesize": font_size("xl"),
            "axes.labelsize": font_size("m"),
            "xtick.labelsize": font_size("s"),
            "ytick.labelsize": font_size("s"),
            "legend.fontsize": font_size("s"),
            "legend.title_fontsize": font_size("s+1"),
            **_TICK_RC,
        }
    )


def apply_tick_style(ax, *, label_fontsize: int | None = None) -> None:
    """Force tick marks on a given axes regardless of upstream rcParams state.

    Use this in plotters as a safety net — `set_academic_style()` should have
    already turned ticks on globally, but this guarantees any axes built on a
    stale style still ends up with visible ticks.
    """
    if ax is None:
        return
    ax.tick_params(
        axis="both", which="major",
        bottom=True, left=True, top=False, right=False,
        labelbottom=True, labelleft=True,
        labeltop=False, labelright=False,
        length=_TICK_RC["xtick.major.size"],
        width=_TICK_RC["xtick.major.width"],
        direction=_TICK_RC["xtick.direction"],
        color=_TICK_RC["xtick.color"],
        labelsize=(label_fontsize if label_fontsize is not None else FONT["tick"]),
    )
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(_TICK_RC["axes.edgecolor"])
        ax.spines[spine].set_linewidth(_TICK_RC["axes.linewidth"])


def pretty_gtex_label(name: str) -> str:
    s = strip_display_label_prefixes(str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    override = GTEX_DISPLAY_LABEL_OVERRIDES.get(s.lower())
    if override is not None:
        return override
    s = re.sub(r"\(\s*ba\s*(\d+)\s*\)", r"(BA\1)", s, flags=re.IGNORECASE)
    if s:
        s = s[0].upper() + s[1:]
    return s


# Canonical typeset forms for the four supported metrics. Use mathtext so they
# render correctly in matplotlib titles, axis labels, and legends.
METRIC_LABELS = {
    "pearson_r":  r"Pearson $r$",
    "pearson":    r"Pearson $r$",
    "spearman_r": r"Spearman $\rho$",
    "spearman":   r"Spearman $\rho$",
    "r2":         r"$R^2$",
    "rmse":       "RMSE",
    "kendall_tau": r"Kendall $\tau$",
    "kendall":    r"Kendall $\tau$",
}

# Same set with a `mean_` prefix, used by fold-combo summaries.
METRIC_LABELS_MEAN = {
    f"mean_{k}": f"Mean fold {v}"
    for k, v in METRIC_LABELS.items()
}


def format_metric_label(metric: str) -> str:
    """Return the canonical typeset label for a known metric name.

    Falls back to :func:`format_legend_label` for unrecognized inputs.
    Centralizes Pearson r, Spearman ρ, R², RMSE formatting so every plot
    in the workspace agrees.
    """
    key = str(metric).lower().strip()
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    if key in METRIC_LABELS_MEAN:
        return METRIC_LABELS_MEAN[key]
    return format_legend_label(metric)


def format_legend_label(name: str) -> str:
    key = str(name).lower().strip()
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    if key in METRIC_LABELS_MEAN:
        return METRIC_LABELS_MEAN[key]
    pieces = []
    for piece in strip_display_label_prefixes(str(name)).replace("_", " ").split(";"):
        words = []
        for word in piece.strip().split():
            low = word.lower()
            if low in {"ba9", "ba24", "ahba", "gtex", "hvg", "deg", "dlam", "plam", "rmse"}:
                words.append(low.upper())
            elif low.endswith("hvg") and low[:-3].isdigit():
                words.append(f"{low[:-3]}HVG")
            elif low in {"nuc"}:
                words.append("Nuc")
            elif low == "r2":
                words.append("R2")
            else:
                words.append(low.capitalize())
        if words:
            pieces.append(" ".join(words))
    return "; ".join(pieces)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), format_legend_label(str(model)))


def ordered_models(models) -> list[str]:
    vals = [str(m).lower() for m in models]
    seen = set()
    out = []
    for model in MODEL_ORDER + vals:
        if model in vals and model not in seen:
            out.append(model)
            seen.add(model)
    return out


def parcel_label_group(label: str) -> str:
    s = str(label).lower()
    if "cerebell" in s:
        return "cerebellar"
    if any(k in s for k in ["cortex", "ba9", "ba24"]):
        return "cortical"
    if any(k in s for k in ["nuc acc", "nucleus accumbens", "put", "caudate"]):
        return "basal_ganglia"
    if any(k in s for k in ["hippocampus", "hypothalamus", "amygdala", "substantia nigra", "substanti nigra"]):
        return "limbic_midbrain"
    return "other"


def parcel_group_sort_key(label: str) -> tuple[int, str]:
    group = parcel_label_group(label)
    try:
        group_i = PARCEL_GROUP_ORDER.index(group)
    except ValueError:
        group_i = len(PARCEL_GROUP_ORDER)
    return group_i, format_legend_label(label)


def build_parcel_label_table(
    prepost: dict[str, object],
    eligible_only: bool = True,
) -> pd.DataFrame:
    target = prepost["target_meta"][["parcel_idx", "tissue_or_parcel"]].copy()
    target["parcel_idx"] = target["parcel_idx"].astype(int)
    target["ahba_parcel"] = target["tissue_or_parcel"].astype(str)

    gtex_key = "gtex_eligible_raw" if bool(eligible_only) else "gtex_raw"
    gtex = prepost[gtex_key].copy()
    gtex["parcel_idx"] = gtex["parcel_idx"].astype(int)
    native = (
        gtex.groupby("parcel_idx")["tissue_or_parcel"]
        .apply(lambda s: "; ".join(sorted(set(pretty_gtex_label(str(x)) for x in s.dropna().tolist()))))
        .rename("gtex_native")
        .reset_index()
    )
    obs = (
        gtex.groupby("parcel_idx")
        .agg(
            n_subjects_observed=("subject", lambda s: int(pd.Series(s).astype(str).nunique())),
            n_samples=("subject", "size"),
        )
        .reset_index()
    )
    out = target.merge(native, on="parcel_idx", how="left").merge(obs, on="parcel_idx", how="left")
    out["gtex_native"] = out["gtex_native"].fillna("")
    out["n_subjects_observed"] = out["n_subjects_observed"].fillna(0).astype(int)
    out["n_samples"] = out["n_samples"].fillna(0).astype(int)
    out["label_ahba"] = out["ahba_parcel"].astype(str)
    out["label_gtex"] = np.where(out["gtex_native"].astype(str).str.len() > 0, out["gtex_native"], out["ahba_parcel"])
    out["label_both"] = out["ahba_parcel"].astype(str) + ": " + out["label_gtex"].astype(str)
    out["label"] = out["label_both"]
    return out.sort_values("parcel_idx").reset_index(drop=True)


def parcel_label_lookup(
    parcel_label_df: pd.DataFrame | None,
    label_mode: str = "both",
) -> dict[int, str]:
    if parcel_label_df is None:
        return {}
    mode = str(label_mode).lower()
    col = {
        "ahba": "label_ahba",
        "gtex": "label_gtex",
        "both": "label_both",
    }.get(mode)
    if col is None:
        raise ValueError("label_mode must be one of: ahba, gtex, both")
    return dict(zip(parcel_label_df["parcel_idx"].astype(int), parcel_label_df[col].astype(str)))


def parcel_color_map(
    parcel_label_df: pd.DataFrame | None,
    palette: str = "tab20",
) -> dict[int, str]:
    if parcel_label_df is None:
        return {}
    if str(palette).lower() in {"functional", "parcel_group", "grouped"}:
        colors: dict[int, str] = {}
        counters = {k: 0 for k in PARCEL_GROUP_COLORS}
        label_col = "label_gtex" if "label_gtex" in parcel_label_df.columns else "label"
        df = parcel_label_df.copy()
        df["_group_order"] = df[label_col].map(lambda s: parcel_group_sort_key(str(s))[0])
        df["_pretty_label"] = df[label_col].map(format_legend_label)
        for _, row in df.sort_values(["_group_order", "_pretty_label", "parcel_idx"]).iterrows():
            pid = int(row["parcel_idx"])
            group = parcel_label_group(str(row[label_col]))
            group_palette = PARCEL_GROUP_COLORS[group]
            colors[pid] = group_palette[counters[group] % len(group_palette)]
            counters[group] += 1
        return colors
    ids = parcel_label_df["parcel_idx"].astype(int).tolist()
    colors = sns.color_palette(str(palette), n_colors=max(1, len(ids)))
    return {pid: to_hex(color) for pid, color in zip(ids, colors)}


__all__ = [
    "DISPLAY_LABEL_PREFIXES_TO_STRIP",
    "FONT",
    "FONT_TOKENS",
    "GTEX_DISPLAY_LABEL_OVERRIDES",
    "METRIC_LABELS",
    "METRIC_LABELS_MEAN",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "apply_tick_style",
    "font_size",
    "format_metric_label",
    "model_label",
    "ordered_models",
    "PARCEL_GROUP_COLORS",
    "PARCEL_GROUP_ORDER",
    "build_parcel_label_table",
    "format_legend_label",
    "parcel_color_map",
    "parcel_group_sort_key",
    "parcel_label_group",
    "parcel_label_lookup",
    "pretty_gtex_label",
    "set_academic_style",
    "set_font_scale",
    "strip_display_label_prefixes",
]
