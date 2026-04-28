"""
verify_subject.py — pre-flight data check for a GTEx subject.

Usage:
    python verify_subject.py GTEX-13OW8
    python verify_subject.py GTEX-1117F

Checks:
  Step 1 — Data shape: gtex_mask count, loro_eval_mask split, per-parcel
            array coverage across all three models.
  Step 2 — Hemisphere verification: label prefix vs. MNI centroid x,
            flags RH parcels and midline/bilateral structures.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
import data_loader


def step1(subject_id: str):
    print(f"\n{'='*70}")
    print(f"STEP 1 — Data shape: {subject_id}")
    print(f"{'='*70}")

    labels, atlas = data_loader.load_alignment()
    arrays = ['loro_truth_subject_raw', 'loro_truth_subject_h',
              'loro_fused_subject_h', 'fullfit_subject_h']

    for model in ['naive', 'dlam', 'plam']:
        try:
            npz = data_loader.load_subject(model, subject_id)
        except FileNotFoundError as e:
            print(f"  [{model}] MISSING: {e}")
            continue

        gtex = npz['gtex_mask'].astype(bool)
        loro = npz['loro_eval_mask'].astype(bool)
        input_parcels = gtex & loro
        pred_parcels  = gtex & ~loro

        print(f"\n  Model: {model}")
        print(f"    gtex_mask total:            {gtex.sum()}")
        print(f"    loro_eval_mask (inputs):    {input_parcels.sum()}")
        print(f"    gtex & ~loro (predictions): {pred_parcels.sum()}")
        print(f"    {'Idx':>4}  {'Tag':5}  {'Label':38}  {'Struct':10}  {'H':2}  "
              + "  ".join(f"{a.split('_',1)[-1][:8]:8}" for a in arrays))

        for i, row in atlas.iterrows():
            if not gtex[i]:
                continue
            tag = 'INPUT' if loro[i] else 'PRED'
            arr_str = "  ".join(
                ('ok      ' if not np.isnan(npz[a][i, 0]) else 'NaN     ')
                if a in npz else 'missing '
                for a in arrays
            )
            print(f"    {i:4d}  {tag:5}  {row['label']:38}  "
                  f"{row['structure']:10}  {row['hemisphere']:2}  {arr_str}")

        # V3 coverage check
        if 'loro_truth_subject_h' in npz:
            truth_h = npz['loro_truth_subject_h'][:, 0]
            non_nan = (~np.isnan(truth_h[gtex])).sum()
            status = 'OK' if non_nan == gtex.sum() else f'WARN: only {non_nan}/{gtex.sum()} non-NaN'
            print(f"    loro_truth_subject_h coverage over gtex_mask: {status}")


def step2(subject_id: str):
    print(f"\n{'='*70}")
    print(f"STEP 2 — Hemisphere verification: {subject_id}")
    print(f"{'='*70}")

    labels, atlas = data_loader.load_alignment()
    bundle = data_loader._bundle_for(subject_id)
    atlas_csv = pd.read_csv(bundle / 'atlas_info' / 'atlas-4S156Parcels_dseg_reformatted.csv')
    atlas_csv = atlas_csv.set_index('label')

    npz = data_loader.load_subject('naive', subject_id)
    gtex = npz['gtex_mask'].astype(bool)

    rh_flags, midline_flags, mismatch_flags = [], [], []

    print(f"\n  {'Idx':>4}  {'Label':38}  {'H':2}  {'MNI x':7}  Verdict")
    print(f"  {'-'*80}")

    for i, row in atlas.iterrows():
        if not gtex[i]:
            continue
        lbl = row['label']
        hemi_csv = row['hemisphere']
        mni_x = float(atlas_csv.loc[lbl, 'mni_x'])

        if lbl.startswith('LH_') or lbl.startswith('LH-'):
            label_hemi = 'L'
        elif lbl.startswith('RH_') or lbl.startswith('RH-'):
            label_hemi = 'R'
        else:
            label_hemi = 'B'

        if label_hemi == 'B':
            flag = 'midline' if abs(mni_x) < 5 else 'bilateral-offset'
            verdict = f'BILATERAL/MIDLINE  x={mni_x:.1f}mm  (flag for collaborator)'
            midline_flags.append((lbl, mni_x))
        elif label_hemi == 'R':
            verdict = f'FLAG RH  x={mni_x:.1f}mm'
            rh_flags.append((lbl, mni_x))
        elif label_hemi == 'L' and mni_x > 0:
            verdict = f'FLAG MISMATCH: LH label, x={mni_x:.1f}mm > 0'
            mismatch_flags.append((lbl, mni_x))
        else:
            verdict = 'clean LH'

        print(f"  {i:4d}  {lbl:38}  {hemi_csv:2}  {mni_x:7.1f}  {verdict}")

    print(f"\n  {'='*40}")
    if not (rh_flags or mismatch_flags):
        hemi_verdict = 'PASS: no clear RH parcels.'
    else:
        hemi_verdict = 'STOP: RH parcels found — resolve before proceeding.'
    print(f"  Verdict: {hemi_verdict}")
    if rh_flags:
        print(f"  RH parcels ({len(rh_flags)}):")
        for lbl, x in rh_flags:
            print(f"    {lbl}  x={x:.1f}mm")
    if midline_flags:
        print(f"  Bilateral/midline parcels ({len(midline_flags)}) — flag for collaborator:")
        for lbl, x in midline_flags:
            print(f"    {lbl}  x={x:.1f}mm")
    if mismatch_flags:
        print(f"  Label/centroid mismatches ({len(mismatch_flags)}):")
        for lbl, x in mismatch_flags:
            print(f"    {lbl}  x={x:.1f}mm")

    return not bool(rh_flags or mismatch_flags)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_subject.py <SUBJECT_ID>")
        sys.exit(1)

    subject_id = sys.argv[1]
    step1(subject_id)
    ok = step2(subject_id)
    sys.exit(0 if ok else 1)
