from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import META_COLS


def load_ahba_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["uid"])
    df["subject"] = df["uid"].astype(str)
    sex = df["sex"].astype(str).str.upper().str.strip()
    df["sex"] = sex.map({"M": "M", "F": "F", "MALE": "M", "FEMALE": "F"}).fillna(sex.str[:1])
    return df[["subject", "age", "sex"]]


def load_ahba_subject_parcellation(
    csv_path: str | Path,
    subject_id: str,
    atlas_coords: pd.DataFrame,
    *,
    label_col: str = "label",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, keep_default_na=True)
    gene_cols = [c for c in df.columns if c != label_col] if label_col in df.columns else list(df.columns)
    rows = []
    for i, (_, values) in enumerate(df[gene_cols].iterrows()):
        if values.isna().all():
            continue
        atlas_row = atlas_coords.iloc[i]
        label = atlas_row[label_col] if label_col in atlas_row.index else str(i + 1)
        rec = {
            "subject": str(subject_id),
            "dataset": "AHBA",
            "tissue_or_parcel": label,
            "coordinates": [(float(atlas_row["mni_x"]), float(atlas_row["mni_y"]), float(atlas_row["mni_z"]))],
        }
        rec.update(values.to_dict())
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=META_COLS)
    return pd.DataFrame(rows)


def list_ahba_subject_files(root: str | Path, *, parcellation: str = "S156", processing: str = "raw") -> list[Path]:
    base = Path(root) / parcellation / "microarray" / processing
    if parcellation.upper() == "S156":
        prefix = "AHBA_schaefer156_"
    else:
        prefix = f"AHBA_{parcellation}_"
    files = [
        p for p in base.glob(f"{prefix}*.csv")
        if "mean" not in p.name and "median" not in p.name
    ]
    return sorted(files)
