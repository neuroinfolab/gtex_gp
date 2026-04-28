"""
inspect_npz.py — print GTEx-masked rows from an NPZ as a DataFrame.

Usage:
    python inspect_npz.py GTEX-13OW8
    python inspect_npz.py GTEX-13OW8 --model plam
    python inspect_npz.py GTEX-13OW8 --model dlam --genes BDNF,DPM1,SNAP25
    python inspect_npz.py GTEX-13OW8 --view v1   # filter to a specific view's mask
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import data_loader

VIEW_MASKS = {
    'v1': ('loro_truth_subject_raw', 'loro_eval_mask'),
    'v2': ('loro_fused_subject_h',   'gtex_mask'),
    'v3': ('loro_truth_subject_h',   'gtex_mask'),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='Subject ID, e.g. GTEX-13OW8')
    parser.add_argument('--model', default='naive', choices=['naive', 'dlam', 'plam'])
    parser.add_argument('--genes', default=None,
                        help='Comma-separated gene symbols (default: first 5 in gene list)')
    parser.add_argument('--view', default=None, choices=['v1', 'v2', 'v3'],
                        help='Restrict to a specific view mask (default: gtex_mask)')
    parser.add_argument('--all-genes', action='store_true',
                        help='Include all genes (wide output)')
    parser.add_argument('--out', default=None,
                        help='CSV output path. If --view is set and --out is omitted, '
                             'auto-saves to <subject>_<view>_<model>.csv')
    args = parser.parse_args()

    labels, atlas = data_loader.load_alignment()
    npz = data_loader.load_subject(args.model, args.subject)
    gene_names = npz['gene_names']

    # --- Determine gene columns ---
    if args.all_genes:
        gene_cols = gene_names.tolist()
    elif args.genes:
        gene_cols = [g.strip() for g in args.genes.split(',')]
        missing = [g for g in gene_cols if g not in gene_names]
        if missing:
            print(f"WARNING: genes not found in dataset: {missing}", file=sys.stderr)
            gene_cols = [g for g in gene_cols if g in gene_names]
    else:
        try:
            preset = data_loader.load_gene_list('ahba_100hvg')
            preset = [g for g in preset if g in gene_names]
            gene_cols = preset[:5]
        except FileNotFoundError:
            gene_cols = gene_names[:5].tolist()

    gene_indices = {g: int(np.where(gene_names == g)[0][0]) for g in gene_cols}

    # --- Determine row mask ---
    if args.view:
        array_key, mask_key = VIEW_MASKS[args.view]
        row_mask = npz[mask_key].astype(bool)
        mask_label = f"{args.view} ({mask_key})"
    else:
        row_mask = npz['gtex_mask'].astype(bool)
        mask_label = 'gtex_mask'

    # --- Build DataFrame ---
    arrays_to_show = [
        ('loro_truth_subject_raw', 'raw'),
        ('loro_truth_subject_h',   'truth_h'),
        ('loro_fused_subject_h',   'fused_h'),
        ('fullfit_subject_h',      'fullfit_h'),
    ]

    rows = []
    for i, row in atlas.iterrows():
        if not row_mask[i]:
            continue
        entry = {
            'parcel':     row['label'],
            'structure':  row['structure'],
            'hemi':       row['hemisphere'],
            'gtex':       bool(npz['gtex_mask'][i]),
            'loro_eval':  bool(npz['loro_eval_mask'][i]),
        }
        for arr_key, col_prefix in arrays_to_show:
            if arr_key not in npz:
                continue
            for gene, g_idx in gene_indices.items():
                val = npz[arr_key][i, g_idx]
                entry[f'{col_prefix}:{gene}'] = round(float(val), 4) if not np.isnan(val) else None
        rows.append(entry)

    df = pd.DataFrame(rows)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # --- CSV output ---
    csv_path = None
    if args.out:
        csv_path = Path(args.out)
    elif args.view:
        csv_path = Path(f"{args.subject}_{args.view}_{args.model}.csv")

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path.resolve()}")

    # --- Console output ---
    print(f"\nSubject: {args.subject}  |  Model: {args.model}  |  Mask: {mask_label}")
    print(f"Genes shown: {gene_cols}")
    print(f"Rows: {len(df)}\n")
    print(df.to_string(index=False))
    print()

if __name__ == '__main__':
    main()
