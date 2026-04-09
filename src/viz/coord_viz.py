from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src import io as io_utils
from src.preprocess import add_target_meta, build_target_parcels, map_gtex_to_target
from src.workflows.writeup_pipeline import _resolve_optional_path


def _load_metadata(csv_path: str = "data/raw/gxp_samples.csv") -> pd.DataFrame:
    csv_resolved = _resolve_optional_path(csv_path, ["gxp_samples.csv"])
    dtype_map = {col: "string" for col in io_utils.META_COLS}
    df = pd.read_csv(csv_resolved, usecols=io_utils.META_COLS, dtype=dtype_map, low_memory=False)
    xyz = np.vstack([io_utils.parse_coordinate_centroid(c) for c in df["coordinates"]])
    df["coord_x"] = xyz[:, 0]
    df["coord_y"] = xyz[:, 1]
    df["coord_z"] = xyz[:, 2]
    df = df.dropna(subset=["coord_x", "coord_y", "coord_z"]).copy()
    df["dataset_upper"] = df["dataset"].astype(str).str.upper().str.strip()
    df["subject"] = df["subject"].astype(str)
    df["tissue_or_parcel"] = df["tissue_or_parcel"].astype(str)
    return df.reset_index(drop=True)


def _parse_all_coordinates(coord_text: object) -> List[Tuple[float, float, float]]:
    toks = io_utils.COORD_PATTERN.findall(str(coord_text))
    return [(float(a), float(b), float(c)) for a, b, c in toks]


def _transform_xyz(xyz: np.ndarray, hemi_mode: str) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    mode = str(hemi_mode).lower()
    if mode == "native":
        return out
    if mode == "mirror_left":
        out[:, 0] = -np.abs(out[:, 0])
        return out
    raise ValueError("hemi_mode must be one of: native, mirror_left")


def _representative_xyz(coords: List[Tuple[float, float, float]], rep_mode: str, hemi_mode: str) -> np.ndarray:
    xyz = np.asarray(coords, dtype=np.float64)
    xyz = _transform_xyz(xyz, hemi_mode=hemi_mode)
    mode = str(rep_mode).lower()
    if mode == "centroid":
        return xyz.mean(axis=0)
    if mode == "medoid":
        d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
        idx = int(np.argmin(d2.sum(axis=1)))
        return xyz[idx, :]
    raise ValueError("rep_mode must be one of: centroid, medoid")


def load_coordinate_overlay_data(
    csv_path: str = "data/raw/gxp_samples.csv",
    rep_mode: str = "centroid",
    hemi_mode: str = "native",
) -> Dict[str, pd.DataFrame]:
    df = _load_metadata(csv_path=csv_path)
    ahba_raw = df[df["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df[df["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)

    target = build_target_parcels(ahba_raw)
    target_meta = add_target_meta(target)
    gtex_rep_rows: List[Dict[str, object]] = []
    full_rows: List[Dict[str, object]] = []
    for sample_idx, row in gtex_raw.reset_index(drop=True).iterrows():
        coords = _parse_all_coordinates(row["coordinates"])
        if not coords:
            continue
        rep_xyz = _representative_xyz(coords, rep_mode=rep_mode, hemi_mode=hemi_mode)
        gtex_rep_rows.append(
            {
                "sample_index": int(sample_idx),
                "subject": str(row["subject"]),
                "tissue_or_parcel": str(row["tissue_or_parcel"]),
                "coord_x": float(rep_xyz[0]),
                "coord_y": float(rep_xyz[1]),
                "coord_z": float(rep_xyz[2]),
                "rep_mode": str(rep_mode).lower(),
                "hemi_mode": str(hemi_mode).lower(),
            }
        )
        full_xyz = _transform_xyz(np.asarray(coords, dtype=np.float64), hemi_mode=hemi_mode)
        for coord_idx, (x, y, z) in enumerate(full_xyz):
            full_rows.append(
                {
                    "sample_index": int(sample_idx),
                    "coord_index": int(coord_idx),
                    "subject": str(row["subject"]),
                    "tissue_or_parcel": str(row["tissue_or_parcel"]),
                    "coord_x": float(x),
                    "coord_y": float(y),
                    "coord_z": float(z),
                    "rep_mode": str(rep_mode).lower(),
                    "hemi_mode": str(hemi_mode).lower(),
                    "point_kind": "gtex_full",
                }
            )
    gtex_rep = pd.DataFrame(gtex_rep_rows)
    gtex_mapped = map_gtex_to_target(gtex_rep, target_meta)

    gtex_representatives = gtex_mapped[
        [
            "sample_index",
            "subject",
            "tissue_or_parcel",
            "coord_x",
            "coord_y",
            "coord_z",
            "mapped_parcel",
            "mapping_distance",
            "parcel_idx",
            "rep_mode",
            "hemi_mode",
        ]
    ].copy()
    gtex_representatives["point_kind"] = "gtex_representative"

    gtex_full = pd.DataFrame(full_rows)
    if len(gtex_full):
        rep_meta = gtex_representatives[
            ["sample_index", "mapped_parcel", "mapping_distance", "parcel_idx", "rep_mode", "hemi_mode"]
        ].copy()
        gtex_full = gtex_full.merge(rep_meta, on=["sample_index", "rep_mode", "hemi_mode"], how="left")

    ahba_points = target_meta.copy()
    ahba_points["point_kind"] = "ahba_parcel"
    ahba_points["rep_mode"] = str(rep_mode).lower()
    ahba_points["hemi_mode"] = str(hemi_mode).lower()

    return {
        "ahba_parcels": ahba_points.reset_index(drop=True),
        "gtex_representatives": gtex_representatives.reset_index(drop=True),
        "gtex_centroids": gtex_representatives.reset_index(drop=True),
        "gtex_full": gtex_full.reset_index(drop=True),
    }


def _plotly_import():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive coordinate overlays. Install `plotly` in the notebook environment."
        ) from exc
    return go, px


def _qualitative_colors(n: int, palette_name: str = "Safe") -> List[str]:
    _, px = _plotly_import()
    palette = list(getattr(px.colors.qualitative, palette_name, px.colors.qualitative.Safe))
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def _axis_ranges(ahba: pd.DataFrame, gtex: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    xs = np.r_[ahba["coord_x"].to_numpy(dtype=np.float64), gtex["coord_x"].to_numpy(dtype=np.float64)]
    ys = np.r_[ahba["coord_y"].to_numpy(dtype=np.float64), gtex["coord_y"].to_numpy(dtype=np.float64)]
    zs = np.r_[ahba["coord_z"].to_numpy(dtype=np.float64), gtex["coord_z"].to_numpy(dtype=np.float64)]

    x_range = float(np.max(xs) - np.min(xs))
    y_range = float(np.max(ys) - np.min(ys))
    z_range = float(np.max(zs) - np.min(zs))
    max_range = max(x_range, y_range, z_range, 1.0)

    x_mid = float(np.mean([np.min(xs), np.max(xs)]))
    y_mid = float(np.mean([np.min(ys), np.max(ys)]))
    z_mid = float(np.mean([np.min(zs), np.max(zs)]))
    half = 0.5 * max_range
    return {
        "x": (x_mid - half, x_mid + half),
        "y": (y_mid - half, y_mid + half),
        "z": (z_mid - half, z_mid + half),
    }


def _mirror_left(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["coord_x_original"] = out["coord_x"].to_numpy(dtype=np.float64)
    out["coord_x"] = -np.abs(out["coord_x"].to_numpy(dtype=np.float64))
    out["hemi_transform"] = "mirror_left"
    return out


def _symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["coord_x_original"] = base["coord_x"].to_numpy(dtype=np.float64)
    base["hemi_transform"] = "original"

    dup = df.copy()
    dup["coord_x_original"] = dup["coord_x"].to_numpy(dtype=np.float64)
    dup["coord_x"] = -dup["coord_x"].to_numpy(dtype=np.float64)
    dup["hemi_transform"] = "mirrored_copy"

    return pd.concat([base, dup], ignore_index=True)


def _apply_hemi_view(ahba: pd.DataFrame, gtex: pd.DataFrame, hemi_view: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    view = str(hemi_view).lower()
    if view == "native":
        ah = ahba.copy()
        gt = gtex.copy()
        ah["coord_x_original"] = ah["coord_x"].to_numpy(dtype=np.float64)
        gt["coord_x_original"] = gt["coord_x"].to_numpy(dtype=np.float64)
        ah["hemi_transform"] = "native"
        gt["hemi_transform"] = "native"
        return ah, gt
    if view == "mirror_left":
        return _mirror_left(ahba), _mirror_left(gtex)
    if view == "symmetrize":
        return _symmetrize(ahba), _symmetrize(gtex)
    raise ValueError("hemi_view must be one of: native, mirror_left, symmetrize")


def plot_ahba_gtex_overlay(
    mode: str = "centroid",
    csv_path: str = "data/raw/gxp_samples.csv",
    hemi_view: str = "native",
    rep_mode: str = "centroid",
    color_by: str = "mapped_parcel",
    palette_name: str = "Safe",
    ahba_color: str = "#9ca3af",
    ahba_size: float = 4.8,
    gtex_size: float = 4.0,
    ahba_opacity: float = 0.40,
    gtex_opacity: float = 0.78,
    ahba_symbol: str = "circle",
    gtex_symbol: str = "circle",
    draw_match_lines: bool = False,
    match_line_color: str = "#1e3a8a",
    match_line_width: float = 1.0,
    title: str | None = None,
):
    go, _ = _plotly_import()

    hemi_view_l = str(hemi_view).lower()
    rep_mode_l = str(rep_mode).lower()
    data = load_coordinate_overlay_data(csv_path=csv_path, rep_mode=rep_mode_l, hemi_mode=hemi_view_l if hemi_view_l in {"native", "mirror_left"} else "native")
    ahba = data["ahba_parcels"]
    mode_l = str(mode).lower()
    if mode_l not in {"centroid", "full"}:
        raise ValueError("mode must be one of: centroid, full")
    gtex = data["gtex_representatives"] if mode_l == "centroid" else data["gtex_full"]
    ahba, gtex = _apply_hemi_view(ahba, gtex, hemi_view=hemi_view)

    color_col = str(color_by)
    if color_col not in gtex.columns:
        raise ValueError(f"color_by={color_col!r} is not available in GTEx overlay columns")

    fig = go.Figure()

    ahba_custom = np.stack(
        [
            ahba["tissue_or_parcel"].astype(str).to_numpy(),
            ahba["parcel_idx"].to_numpy(dtype=np.int32),
            ahba["macro_system"].astype(str).to_numpy(),
            ahba["hemisphere"].astype(str).to_numpy(),
            ahba["hemi_transform"].astype(str).to_numpy(),
            ahba["coord_x_original"].to_numpy(dtype=np.float64),
            ahba["rep_mode"].astype(str).to_numpy(),
            ahba["hemi_mode"].astype(str).to_numpy(),
        ],
        axis=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=ahba["coord_x"],
            y=ahba["coord_y"],
            z=ahba["coord_z"],
            mode="markers",
            name="AHBA parcels",
            marker={
                "size": ahba_size,
                "color": ahba_color,
                "opacity": ahba_opacity,
                "symbol": ahba_symbol,
                "line": {"width": 0.4, "color": "#374151"},
            },
            customdata=ahba_custom,
            hovertemplate=(
                "<b>AHBA parcel</b><br>"
                "parcel=%{customdata[0]}<br>"
                "parcel_idx=%{customdata[1]}<br>"
                "macro_system=%{customdata[2]}<br>"
                "hemisphere=%{customdata[3]}<br>"
                "hemi_transform=%{customdata[4]}<br>"
                "x_original=%{customdata[5]:.2f}<br>"
                "rep_mode=%{customdata[6]}<br>"
                "assignment_hemi=%{customdata[7]}<br>"
                "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
            ),
        )
    )

    if bool(draw_match_lines):
        ahba_lookup = (
            ahba.drop_duplicates(subset=["parcel_idx"])
            .set_index("parcel_idx")[["coord_x", "coord_y", "coord_z"]]
            .to_dict("index")
        )
        line_x: List[float | None] = []
        line_y: List[float | None] = []
        line_z: List[float | None] = []
        for _, row in gtex.iterrows():
            parcel_idx = int(row["parcel_idx"])
            if parcel_idx not in ahba_lookup:
                continue
            target = ahba_lookup[parcel_idx]
            line_x.extend([float(row["coord_x"]), float(target["coord_x"]), None])
            line_y.extend([float(row["coord_y"]), float(target["coord_y"]), None])
            line_z.extend([float(row["coord_z"]), float(target["coord_z"]), None])
        if line_x:
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    name="GTEx to AHBA match",
                    line={"color": match_line_color, "width": match_line_width},
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    groups = sorted(gtex[color_col].astype(str).unique().tolist())
    color_map = dict(zip(groups, _qualitative_colors(len(groups), palette_name=palette_name)))

    for group in groups:
        sub = gtex[gtex[color_col].astype(str) == group].copy()
        custom_cols = [
            sub["subject"].astype(str).to_numpy(),
            sub["tissue_or_parcel"].astype(str).to_numpy(),
            sub["mapped_parcel"].astype(str).to_numpy(),
            sub["parcel_idx"].to_numpy(dtype=np.int32),
            sub["mapping_distance"].to_numpy(dtype=np.float64),
            sub["hemi_transform"].astype(str).to_numpy(),
            sub["coord_x_original"].to_numpy(dtype=np.float64),
            sub["rep_mode"].astype(str).to_numpy(),
            sub["hemi_mode"].astype(str).to_numpy(),
        ]
        if "coord_index" in sub.columns:
            custom_cols.append(sub["coord_index"].to_numpy(dtype=np.int32))
        custom = np.stack(custom_cols, axis=1)

        hover = (
            "<b>GTEx point</b><br>"
            "subject=%{customdata[0]}<br>"
            "tissue=%{customdata[1]}<br>"
            "mapped_parcel=%{customdata[2]}<br>"
            "parcel_idx=%{customdata[3]}<br>"
            "mapping_distance=%{customdata[4]:.2f}<br>"
            "hemi_transform=%{customdata[5]}<br>"
            "x_original=%{customdata[6]:.2f}<br>"
            "rep_mode=%{customdata[7]}<br>"
            "assignment_hemi=%{customdata[8]}<br>"
        )
        if "coord_index" in sub.columns:
            hover += "coord_index=%{customdata[9]}<br>"
        hover += "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"

        fig.add_trace(
            go.Scatter3d(
                x=sub["coord_x"],
                y=sub["coord_y"],
                z=sub["coord_z"],
                mode="markers",
                name=f"GTEx {group}",
                marker={
                    "size": gtex_size,
                    "color": color_map[group],
                    "opacity": gtex_opacity,
                    "symbol": gtex_symbol,
                },
                customdata=custom,
                hovertemplate=hover,
            )
        )

    ranges = _axis_ranges(ahba, gtex)
    fig.update_layout(
        title=title
        or (
            (
                "AHBA parcel coordinates with GTEx centroids"
                if mode_l == "centroid"
                else "AHBA parcel coordinates with all GTEx coordinates"
            )
            + {
                "native": "",
                "mirror_left": " (mirrored to left hemisphere)",
                "symmetrize": " (symmetrized across hemispheres)",
            }[str(hemi_view).lower()]
            + f" | assignment={rep_mode_l}"
        ),
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "range": list(ranges["x"]), "backgroundcolor": "#f8fafc"},
            "yaxis": {"title": "MNI y", "range": list(ranges["y"]), "backgroundcolor": "#f8fafc"},
            "zaxis": {"title": "MNI z", "range": list(ranges["z"]), "backgroundcolor": "#f8fafc"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.0}},
        },
    )
    return fig, {"ahba_parcels": ahba, "gtex_points": gtex, "color_map": color_map}


def plot_ahba_vs_gtex_centroids(
    csv_path: str = "data/raw/gxp_samples.csv",
    mode: str = "centroid",
    hemi_view: str = "native",
    rep_mode: str = "centroid",
    color_by: str = "mapped_parcel",
    palette_name: str = "Safe",
    ahba_symbol: str = "circle",
    gtex_symbol: str = "circle",
):
    if str(mode).lower() != "centroid":
        raise ValueError("plot_ahba_vs_gtex_centroids only supports mode='centroid'")
    return plot_ahba_gtex_overlay(
        mode="centroid",
        csv_path=csv_path,
        hemi_view=hemi_view,
        rep_mode=rep_mode,
        color_by=color_by,
        palette_name=palette_name,
        ahba_symbol=ahba_symbol,
        gtex_symbol=gtex_symbol,
    )


def plot_ahba_vs_gtex_full_coordinates(
    csv_path: str = "data/raw/gxp_samples.csv",
    mode: str = "full",
    hemi_view: str = "native",
    rep_mode: str = "centroid",
    color_by: str = "mapped_parcel",
    palette_name: str = "Safe",
    ahba_symbol: str = "circle",
    gtex_symbol: str = "circle",
):
    if str(mode).lower() != "full":
        raise ValueError("plot_ahba_vs_gtex_full_coordinates only supports mode='full'")
    return plot_ahba_gtex_overlay(
        mode="full",
        csv_path=csv_path,
        hemi_view=hemi_view,
        rep_mode=rep_mode,
        color_by=color_by,
        palette_name=palette_name,
        ahba_symbol=ahba_symbol,
        gtex_symbol=gtex_symbol,
    )


def plot_ahba_gtex_mirrored_left(
    mode: str = "centroid",
    csv_path: str = "data/raw/gxp_samples.csv",
    rep_mode: str = "centroid",
    color_by: str = "mapped_parcel",
    palette_name: str = "Safe",
    ahba_symbol: str = "circle",
    gtex_symbol: str = "circle",
):
    return plot_ahba_gtex_overlay(
        mode=mode,
        csv_path=csv_path,
        hemi_view="mirror_left",
        rep_mode=rep_mode,
        color_by=color_by,
        palette_name=palette_name,
        ahba_symbol=ahba_symbol,
        gtex_symbol=gtex_symbol,
    )


def plot_ahba_gtex_symmetrized(
    mode: str = "centroid",
    csv_path: str = "data/raw/gxp_samples.csv",
    rep_mode: str = "centroid",
    color_by: str = "mapped_parcel",
    palette_name: str = "Safe",
    ahba_symbol: str = "circle",
    gtex_symbol: str = "circle",
):
    return plot_ahba_gtex_overlay(
        mode=mode,
        csv_path=csv_path,
        hemi_view="symmetrize",
        rep_mode=rep_mode,
        color_by=color_by,
        palette_name=palette_name,
        ahba_symbol=ahba_symbol,
        gtex_symbol=gtex_symbol,
    )
