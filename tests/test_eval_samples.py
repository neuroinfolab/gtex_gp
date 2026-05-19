from pathlib import Path

import numpy as np
import pandas as pd

from src.eval_utils.eval_samples import (
    GTEX_TENSOR_REGION_ORDER,
    build_dataset_region_tensor,
    build_sampled_tensor,
)


def test_build_dataset_region_tensor_respects_region_order_and_mask():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "age": [40, 40, 50],
            "sex": ["F", "F", "M"],
            "dataset": ["GTEx", "GTEx", "GTEx"],
            "tissue_or_parcel": [
                "brain - frontal cortex (ba9)",
                "brain - cerebellum",
                "brain - frontal cortex (ba9)",
            ],
            "coordinates": ["[]", "[]", "[]"],
            "GENE1": [1.0, 2.0, 3.0],
            "GENE2": [4.0, 5.0, 6.0],
        }
    )

    tensor = build_dataset_region_tensor(
        df,
        dataset="GTEx",
        subjects=["S1", "S2"],
        genes=["GENE1", "GENE2"],
        region_order=GTEX_TENSOR_REGION_ORDER,
        region_axis_kind="gtex_tissue",
    )

    assert tensor.values.shape == (2, len(GTEX_TENSOR_REGION_ORDER), 2)
    assert tensor.regions[0] == "brain - frontal cortex (ba9)"
    assert tensor.regions[-1] == "brain - cerebellum"

    frontal_idx = tensor.regions.index("brain - frontal cortex (ba9)")
    cereb_idx = tensor.regions.index("brain - cerebellum")

    assert tensor.observed_mask[0, frontal_idx]
    assert tensor.observed_mask[0, cereb_idx]
    assert tensor.observed_mask[1, frontal_idx]
    assert not tensor.observed_mask[1, cereb_idx]

    assert np.isclose(tensor.values[0, frontal_idx, 0], 1.0)
    assert np.isclose(tensor.values[0, cereb_idx, 1], 5.0)
    assert np.isnan(tensor.values[1, cereb_idx, 0])


def test_build_sampled_tensor_region_matched_uses_shared_order():
    rows = [
        ("A1", "AHBA", "Parcel_A", "(0, 0, 0)", 10.0),
        ("A1", "AHBA", "Parcel_B", "(10, 0, 0)", 20.0),
        ("G1", "GTEx", "brain - frontal cortex (ba9)", "(0.1, 0, 0)", 1.0),
        ("G1", "GTEx", "brain - hippocampus", "(9.9, 0, 0)", 2.0),
    ]
    df = pd.DataFrame(
        {
            "subject": [r[0] for r in rows],
            "age": [40] * len(rows),
            "sex": ["F"] * len(rows),
            "dataset": [r[1] for r in rows],
            "tissue_or_parcel": [r[2] for r in rows],
            "coordinates": [r[3] for r in rows],
            "GENE1": [r[4] for r in rows],
        }
    )

    gtex_tensor = build_sampled_tensor(
        df,
        dataset="GTEx",
        region_ordering="region_matched",
        matching_policy="centroids",
        gene_panel=["GENE1"],
        n_subjects=1,
        n_genes=1,
        min_regions_per_subject=1,
    )
    ahba_tensor = build_sampled_tensor(
        df,
        dataset="AHBA",
        region_ordering="region_matched",
        matching_policy="centroids",
        gene_panel=["GENE1"],
        n_subjects=1,
        n_genes=1,
        min_regions_per_subject=1,
    )

    assert gtex_tensor.regions == ["brain - frontal cortex (ba9)", "brain - hippocampus"]
    assert ahba_tensor.regions == ["Parcel_A", "Parcel_B"]
