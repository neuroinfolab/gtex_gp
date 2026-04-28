"""
One-time atlas setup for the 4S156Parcels custom yabplot atlas.

Run:
    python build_atlas.py

Prerequisites:
    - Subcortical: no extra deps (uses marching cubes).
    - Cortical:    wb_command must be on PATH.
                   Install Connectome Workbench from the DMG in ~/Downloads,
                   then add its bin dir to PATH, e.g.:
                       export PATH="/Applications/workbench/bin_macosx64:$PATH"

Outputs (in atlas_cache/):
    atlas_cache/subcortical/   — one .vtk per subcortical parcel + atlas_LUT.txt
    atlas_cache/cortical/      — atlas.csv (vertex map) + atlas.txt (LUT)
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

from yabplot.atlas_builder import build_cortical_atlas, build_subcortical_atlas

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
BUNDLE       = HERE.parent / 'subject_bundle'
ATLAS_INFO   = BUNDLE / 'atlas_info'
NII_ORIG     = ATLAS_INFO / 'atlas-4S156Parcels_space-MNI152NLin6Asym_dseg.nii.gz'
NII_3D       = ATLAS_INFO / 'atlas-4S156Parcels_3d.nii.gz'   # squeezed version
CSV_PATH     = ATLAS_INFO / 'atlas-4S156Parcels_dseg_reformatted.csv'
CACHE_DIR    = HERE / 'atlas_cache'
CORTICAL_DIR    = CACHE_DIR / 'cortical'
SUBCORTICAL_DIR = CACHE_DIR / 'subcortical'
WB_LABEL_FILE   = CACHE_DIR / 'wb_labels.txt'

# ── Step 0: Squeeze 4D NIfTI to 3D (required by yabplot/scikit-image) ─────────
if not NII_3D.exists():
    print("Squeezing NIfTI from 4D to 3D...")
    img = nib.load(str(NII_ORIG))
    data3d = img.get_fdata().squeeze()
    img3d = nib.Nifti1Image(data3d, img.affine, img.header)
    img3d.header.set_data_shape(data3d.shape)
    nib.save(img3d, str(NII_3D))
    print(f"  saved: {NII_3D}")

NII_PATH = str(NII_3D)

# ── Load atlas metadata ────────────────────────────────────────────────────────
df            = pd.read_csv(CSV_PATH)
cortical_df   = df[df['structure'] == 'cortex']
subcortical_df = df[df['structure'] == 'subcortex']
cortical_labels   = cortical_df['label'].tolist()
subcortical_labels = subcortical_df['label'].tolist()

# ── Step 1: Subcortical atlas (no wb_command needed) ──────────────────────────
print("\n=== Building subcortical atlas ===")
if not (SUBCORTICAL_DIR / 'atlas_LUT.txt').exists():
    labels_dict = {int(row['index']): row['label'] for _, row in subcortical_df.iterrows()}
    build_subcortical_atlas(
        nii_path=NII_PATH,
        labels_dict=labels_dict,
        out_dir=str(SUBCORTICAL_DIR),
    )
    print("✓ Subcortical atlas built.")
else:
    print("  (already built, skipping)")

# ── Step 2: Generate wb_command label list file ────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
if not WB_LABEL_FILE.exists():
    print("\nGenerating wb label file...")
    with open(WB_LABEL_FILE, 'w') as f:
        for _, row in df.iterrows():
            np.random.seed(int(row['index']))
            r, g, b = np.random.randint(50, 255, 3)
            f.write(f"{row['label']}\n")
            f.write(f"{int(row['index'])} {r} {g} {b} 255\n")
    print(f"  saved: {WB_LABEL_FILE}")

# ── Step 3: Cortical atlas (requires wb_command on PATH) ──────────────────────
print("\n=== Building cortical atlas (requires wb_command) ===")
if (CORTICAL_DIR / 'atlas.csv').exists() and (CORTICAL_DIR / 'atlas.txt').exists():
    print("  (already built, skipping)")
else:
    try:
        build_cortical_atlas(
            nii_path=NII_PATH,
            wb_txt_path=str(WB_LABEL_FILE),
            out_dir=str(CORTICAL_DIR),
            include_list=cortical_labels,
            atlasname='atlas',
        )
        print("✓ Cortical atlas built.")
    except FileNotFoundError as e:
        print(f"✗ wb_command not found: {e}")
        print()
        print("  To install Connectome Workbench on macOS:")
        print("  1. Open ~/Downloads/workbench-macub-v2.1.0.dmg")
        print("  2. Drag the workbench app to /Applications/")
        print("  3. Add to PATH in your shell config (~/.zshrc):")
        print("       export PATH=\"/Applications/workbench/bin_macosx64:$PATH\"")
        print("  4. Re-run: python build_atlas.py")
    except Exception as e:
        print(f"✗ Cortical atlas FAILED: {e}")

print("\nDone. Next step: python app.py")
