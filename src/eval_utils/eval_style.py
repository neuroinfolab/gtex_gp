#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["naive", "dlam", "plam"]
MODEL_COLORS = {"naive": "#7f7f7f", "dlam": "#1f77b4", "plam": "#d62728"}
MODEL_LABELS = {"naive": "Naive", "dlam": "DLAM", "plam": "PLAM"}
FONT = {"title": 11, "label": 10, "tick": 9, "legend": 9, "small": 8}
PARCEL_GROUP_ORDER = ["cortical", "basal_ganglia", "limbic_midbrain", "cerebellar", "other"]
PARCEL_GROUP_COLORS = {
    "cortical": ["#2166ac", "#4393c3", "#92c5de"],
    "basal_ganglia": ["#b35806", "#f1a340", "#fdb863"],
    "limbic_midbrain": ["#762a83", "#c51b7d", "#8c510a", "#bf812d"],
    "cerebellar": ["#1b7837", "#5aae61", "#a6dba0"],
    "other": ["#6b6b6b", "#969696", "#bdbdbd"],
}


def set_academic_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT["label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
        }
    )


def pretty_gtex_label(name: str) -> str:
    s = str(name)
    pref = "brain - "
    if s.lower().startswith(pref):
        return s[len(pref) :]
    return s


def format_legend_label(name: str) -> str:
    pieces = []
    for piece in str(name).replace("_", " ").split(";"):
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
    "FONT",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_ORDER",
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
]
