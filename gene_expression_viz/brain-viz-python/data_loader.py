"""
Data loading utilities for the gene expression brain visualizer.

All functions are stateless and operate on the subject_bundle/ directory.

Hemisphere handling
-------------------
The upstream pipeline forces all MNI x-coordinates negative (LH) before model
fitting.  As a result some GTEx samples end up matched to an RH-labelled AHBA
parcel (the mirrored RH copy was the nearest neighbour), but the model still
treats that parcel as LH.

Display strategy (per collaborator, 2026-04-25):
  V1/V2/V3 — remap_rh_to_lh_nearest():
    For each RH parcel that carries data, flip its centroid x → -x, find the
    nearest same-structure LH parcel, and display the value there.  Everything
    ends up rendered on the left hemisphere.
  V4      — render raw: the NPZ already contains values for all 150 parcels
    (both LH- and RH-labelled) and they are already roughly symmetric from the
    mirroring procedure.  No additional remap or mirror needed.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BUNDLE = Path(__file__).parent.parent / 'subject_bundle'

_HERE = Path(__file__).parent
SUBJECT_BUNDLES: dict[str, Path] = {
    'GTEX-1117F':  _HERE.parent / 'subject_bundle',
    'GTEX-13OW8':  _HERE.parent / 'subject_bundle_GTEX-13OW8',
    'GTEX-11DZ1':  _HERE.parent / 'subject_bundle_GTEX-11DZ1',
    'GTEX-1B996':  _HERE.parent / 'subject_bundle_GTEX-1B996',
    'GTEX-1JMPZ':  _HERE.parent / 'subject_bundle_GTEX-1JMPZ',
}


def _bundle_for(subject_id: str) -> Path:
    return SUBJECT_BUNDLES.get(subject_id, BUNDLE)


def load_alignment():
    """
    Returns (labels, atlas_aligned).

    labels:        list[str], 150 parcel names in alphabetical AHBA order.
                   This is the row order for every (150, G) array in every .npz.
    atlas_aligned: DataFrame with 150 rows, one per label in that same order.
                   Columns include: label, structure, hemisphere, mni_x/y/z, etc.
    """
    parcel_order = json.loads((BUNDLE / 'parcel_order.json').read_text())
    labels = parcel_order['ahba_labels_alphabetical']

    atlas = pd.read_csv(BUNDLE / 'atlas_info' / 'atlas-4S156Parcels_dseg_reformatted.csv')
    atlas_aligned = atlas.set_index('label').loc[labels].reset_index()
    return labels, atlas_aligned


def load_subject(model: str = 'naive', subject_id: str = 'GTEX-1117F') -> dict:
    """
    Load all arrays for one subject/model combination from its .npz file.

    Returns a dict-like object (numpy NpzFile) with keys:
        gene_names, gtex_mask, loro_truth_subject_raw,
        loro_eval_mask, loro_fused_subject_h, loro_truth_subject_h,
        fullfit_subject_h, ...
    """
    path = _bundle_for(subject_id) / 'cache' / model / f'{subject_id}.npz'
    if not path.exists():
        raise FileNotFoundError(f"No NPZ found at: {path}")
    return np.load(str(path), allow_pickle=True)


def get_gene_idx(npz_data, gene_name: str) -> int:
    """Return the column index for a gene name. Raises ValueError if not found."""
    hits = np.where(npz_data['gene_names'] == gene_name)[0]
    if len(hits) == 0:
        raise ValueError(f"Gene '{gene_name}' not found in dataset.")
    return int(hits[0])


def build_value_dict(npz_data, atlas_aligned: pd.DataFrame,
                     array_key: str, mask_key: str, gene_name: str):
    """
    Build {label: float} dicts for cortex and subcortex.

    Parcels outside the mask, or with NaN values, are omitted —
    yabplot renders those as nan_color (grey) automatically.

    Parameters
    ----------
    npz_data    : NpzFile returned by load_subject()
    atlas_aligned: DataFrame returned by load_alignment()
    array_key   : key in npz_data for the (150, G) expression array
                  e.g. 'loro_truth_subject_raw', 'loro_fused_subject_h', ...
    mask_key    : key in npz_data for the (150,) boolean mask
                  e.g. 'gtex_mask', 'loro_eval_mask'
    gene_name   : gene symbol, e.g. 'DPM1'

    Returns
    -------
    (cortex_dict, subcortex_dict) : dict[str, float], dict[str, float]
    """
    g_idx = get_gene_idx(npz_data, gene_name)
    mask  = npz_data[mask_key].astype(bool)      # (150,)
    vals  = npz_data[array_key][:, g_idx]         # (150,)

    cortex_dict   = {}
    subcortex_dict = {}

    for i, row in atlas_aligned.iterrows():
        if not mask[i]:
            continue
        v = float(vals[i])
        if np.isnan(v):
            continue
        if row['structure'] == 'cortex':
            cortex_dict[row['label']] = v
        else:
            subcortex_dict[row['label']] = v

    return cortex_dict, subcortex_dict


# ── Convenience wrappers for each view ────────────────────────────────────────

def v1_gtex_input(npz_data, atlas_aligned, gene_name):
    """V1: Raw GTEx measurements at LORO held-out parcels (loro_eval_mask)."""
    return build_value_dict(npz_data, atlas_aligned,
                            'loro_truth_subject_raw', 'loro_eval_mask', gene_name)


def v1_gtex_harmonized(npz_data, atlas_aligned, gene_name):
    """V1 harmonized: ground truth from fullfit at LORO held-out parcels.

    fullfit_subject_h at GTEx parcel positions equals loro_truth_subject_h —
    both contain the combat-harmonized ground truth used to fit the model.
    """
    return build_value_dict(npz_data, atlas_aligned,
                            'fullfit_subject_h', 'loro_eval_mask', gene_name)


def v2_reconstruction(npz_data, atlas_aligned, gene_name):
    """V2: Model reconstruction at LORO-evaluated GTEx parcels (loro_fused_subject_h, loro_eval_mask).

    Uses loro_eval_mask (not gtex_mask) so that V2 and V3 cover exactly the
    same parcel set. For subjects with skipped holds, gtex_mask includes parcels
    where AHBA had no data and loro_truth_subject_h is NaN; restricting to
    loro_eval_mask keeps both views comparable.
    """
    return build_value_dict(npz_data, atlas_aligned,
                            'loro_fused_subject_h', 'loro_eval_mask', gene_name)


def v2_residual(npz_data, atlas_aligned, gene_name):
    """V2 residual: LORO prediction minus harmonized ground truth at eval parcels.

    Residual = loro_fused_subject_h − loro_truth_subject_h.
    Both arrays are in the same combat-harmonized space, so the difference
    is meaningful.  Positive values = model over-predicts; negative = under-predicts.
    """
    g_idx = get_gene_idx(npz_data, gene_name)
    mask  = npz_data['loro_eval_mask'].astype(bool)
    pred  = npz_data['loro_fused_subject_h'][:, g_idx]
    truth = npz_data['loro_truth_subject_h'][:, g_idx]
    resid = pred - truth

    cortex_dict    = {}
    subcortex_dict = {}
    for i, row in atlas_aligned.iterrows():
        if not mask[i]:
            continue
        v = float(resid[i])
        if np.isnan(v):
            continue
        if row['structure'] == 'cortex':
            cortex_dict[row['label']] = v
        else:
            subcortex_dict[row['label']] = v
    return cortex_dict, subcortex_dict


def v3_ground_truth(npz_data, atlas_aligned, gene_name):
    """V3: AHBA ground truth at LORO-evaluated GTEx parcels (loro_truth_subject_h, loro_eval_mask).

    Directly comparable to V2 — same parcels, AHBA truth vs. model output.
    loro_eval_mask excludes skipped holds where loro_truth_subject_h is NaN.
    """
    return build_value_dict(npz_data, atlas_aligned,
                            'loro_truth_subject_h', 'loro_eval_mask', gene_name)


def v4_fullfit(npz_data, atlas_aligned, gene_name):
    """V4: Dense whole-brain fit — all 150 parcels from fullfit_subject_h.

    imputed_mask (138) + gtex_mask (12) partition all 150 parcels with no
    overlap. fullfit_subject_h has values for all 150, so we use the union.
    Rendered as-is (bilateral); the NPZ predictions are already symmetric.
    """
    g_idx = get_gene_idx(npz_data, gene_name)
    imp_mask  = npz_data['imputed_mask'].astype(bool) if 'imputed_mask' in npz_data else np.zeros(150, bool)
    gtex_mask = npz_data['gtex_mask'].astype(bool)
    full_mask = imp_mask | gtex_mask
    vals = npz_data['fullfit_subject_h'][:, g_idx]

    cortex_dict = {}
    subcortex_dict = {}
    for i, row in atlas_aligned.iterrows():
        if not full_mask[i]:
            continue
        v = float(vals[i])
        if np.isnan(v):
            continue
        if row['structure'] == 'cortex':
            cortex_dict[row['label']] = v
        else:
            subcortex_dict[row['label']] = v
    return cortex_dict, subcortex_dict


def build_excluded_parcel_map() -> dict[str, str]:
    """
    Build {excluded_label: counterpart_label} for the parcels absent from the
    NPZ (no AHBA coverage) whose opposite-hemisphere counterpart IS present.
    Bilateral parcels (Cerebellar_*) are skipped — no counterpart exists.
    Call once at startup.
    """
    parcel_order = json.loads((BUNDLE / 'parcel_order.json').read_text())
    labels_150 = set(parcel_order['ahba_labels_alphabetical'])
    atlas = pd.read_csv(BUNDLE / 'atlas_info' / 'atlas-4S156Parcels_dseg_reformatted.csv')
    excluded = set(atlas['label']) - labels_150

    mapping: dict[str, str] = {}
    for label in excluded:
        if label.startswith('RH-'):
            counterpart = 'LH-' + label[3:]
        elif label.startswith('RH_'):
            counterpart = 'LH_' + label[3:]
        elif label.startswith('LH-'):
            counterpart = 'RH-' + label[3:]
        elif label.startswith('LH_'):
            counterpart = 'RH_' + label[3:]
        else:
            continue  # bilateral — handled separately below
        if counterpart in labels_150:
            mapping[label] = counterpart

    # Cerebellar_Region9 has no LH/RH counterpart; fill from nearest cerebellar
    # region in the NPZ (Cerebellar_Region4, ~21 mm away by MNI centroid).
    if 'Cerebellar_Region9' in excluded and 'Cerebellar_Region4' in labels_150:
        mapping['Cerebellar_Region9'] = 'Cerebellar_Region4'

    return mapping


def fill_excluded_parcels(cortex_dict: dict, subcortex_dict: dict,
                           excluded_map: dict) -> tuple:
    """
    For each excluded parcel that has a counterpart with a value, copy that
    value so the parcel renders instead of showing as NaN (grey).
    Checks both cortex and subcortex dicts; skips if already populated.
    """
    for excl, counterpart in excluded_map.items():
        if excl in cortex_dict or excl in subcortex_dict:
            continue
        if counterpart in cortex_dict:
            cortex_dict[excl] = cortex_dict[counterpart]
        elif counterpart in subcortex_dict:
            subcortex_dict[excl] = subcortex_dict[counterpart]
    return cortex_dict, subcortex_dict


def build_rh_to_lh_map(atlas_aligned: pd.DataFrame) -> dict:
    """
    Pre-compute a mapping from every RH parcel label to its nearest LH parcel.

    For each RH parcel centroid the x coordinate is flipped (x → -x) and the
    closest LH parcel centroid in the same structure (cortex or subcortex) is
    found by Euclidean distance.  Bilateral parcels (Cerebellar_*) are skipped.

    Call once at startup; pass the result to remap_rh_to_lh_nearest().
    """
    rh_to_lh: dict[str, str] = {}
    for structure in ('cortex', 'subcortex'):
        lh = atlas_aligned[
            (atlas_aligned['hemisphere'] == 'L') &
            (atlas_aligned['structure'] == structure)
        ]
        rh = atlas_aligned[
            (atlas_aligned['hemisphere'] == 'R') &
            (atlas_aligned['structure'] == structure)
        ]
        if lh.empty or rh.empty:
            continue
        lh_coords = lh[['mni_x', 'mni_y', 'mni_z']].values
        lh_labels = lh['label'].values
        for _, row in rh.iterrows():
            flipped = np.array([-row['mni_x'], row['mni_y'], row['mni_z']])
            dists = np.linalg.norm(lh_coords - flipped, axis=1)
            rh_to_lh[row['label']] = lh_labels[int(np.argmin(dists))]
    return rh_to_lh


def remap_rh_to_lh_nearest(cortex_dict: dict, subcortex_dict: dict,
                            rh_to_lh: dict) -> tuple:
    """
    Remap RH-labelled parcel values to their nearest LH parcel for display.

    Used for V1/V2/V3 so that everything renders in the left hemisphere,
    consistent with how the upstream pipeline treats all data as LH.

    If a direct LH value already exists for the target parcel it takes
    precedence (the LH measurement is more accurate than a remapped RH one).
    """
    def _remap(d: dict) -> dict:
        out = {k: v for k, v in d.items() if not (k.startswith('RH_') or k.startswith('RH-'))}
        for label, val in d.items():
            if (label.startswith('RH_') or label.startswith('RH-')) and label in rh_to_lh:
                lh_label = rh_to_lh[label]
                if lh_label not in out:
                    out[lh_label] = val
        return out

    return _remap(cortex_dict), _remap(subcortex_dict)


def mirror_lh_to_rh(cortex_dict: dict, subcortex_dict: dict,
                    rh_to_lh: dict) -> tuple:
    """
    Produce a symmetric bilateral display for V3 (fullfit).

    The upstream pipeline forces all x-coordinates negative before fitting, so
    the model only ever sees LH data.  Raw RH predictions in fullfit_subject_h
    are unreliable (naive: spatially inconsistent; dlam: collapses to a
    near-constant).  This function discards the raw RH values and replaces each
    RH parcel with its nearest LH counterpart's value, giving a symmetric view
    backed by the actual LH predictions.
    """
    def _mirror(d: dict) -> dict:
        out = {k: v for k, v in d.items()
               if not (k.startswith('RH_') or k.startswith('RH-'))}
        for rh_label, lh_label in rh_to_lh.items():
            if lh_label in out:
                out[rh_label] = out[lh_label]
        return out

    return _mirror(cortex_dict), _mirror(subcortex_dict)


def load_gene_list(preset: str = 'ahba_100hvg') -> list[str]:
    """Load a gene preset from gene_lists/. Returns a list of gene symbols."""
    path = BUNDLE / 'gene_lists' / f'{preset}.txt'
    if not path.exists():
        raise FileNotFoundError(f"Gene list not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
