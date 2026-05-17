from pathlib import Path

import numpy as np
import pandas as pd

from src.eval_utils.eval_samples import (
    GTEX_TENSOR_REGION_ORDER,
    build_gtex_tissue_tensor,
)


def test_build_gtex_tissue_tensor_respects_region_order_and_mask():
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

    tensor = build_gtex_tissue_tensor(
        df,
        subjects=["S1", "S2"],
        genes=["GENE1", "GENE2"],
        region_order=GTEX_TENSOR_REGION_ORDER,
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
