"""
Run all three 2D visualisation approaches for one gene across all subjects/models.

Usage:
  python generate_all.py                      # gene=DPM1, all subjects & models
  python generate_all.py --gene MAPT          # different gene
  python generate_all.py --gene DPM1 --subjects GTEX-1117F --models naive
  python generate_all.py --approaches nilearn matplotlib   # skip plotly
"""

import argparse
import viz_data

APPROACHES = ['nilearn', 'plotly', 'matplotlib']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gene',       default='DPM1')
    p.add_argument('--subjects',   nargs='+', default=viz_data.SUBJECTS)
    p.add_argument('--models',     nargs='+', default=viz_data.MODELS)
    p.add_argument('--approaches', nargs='+', default=APPROACHES,
                   choices=APPROACHES)
    args = p.parse_args()

    print(f'Gene: {args.gene}')
    print(f'Subjects ({len(args.subjects)}): {args.subjects}')
    print(f'Models: {args.models}')
    print(f'Approaches: {args.approaches}')
    print()

    if 'nilearn' in args.approaches:
        print('=== nilearn ===')
        import viz_nilearn
        viz_nilearn.run(args.gene, args.subjects, args.models)

    if 'plotly' in args.approaches:
        print('=== plotly ===')
        import viz_plotly
        viz_plotly.run(args.gene, args.subjects, args.models)

    if 'matplotlib' in args.approaches:
        print('=== matplotlib ===')
        import viz_matplotlib
        viz_matplotlib.run(args.gene, args.subjects, args.models)

    print('\nDone. Outputs in outputs/nilearn/, outputs/plotly/, outputs/matplotlib/')


if __name__ == '__main__':
    main()
