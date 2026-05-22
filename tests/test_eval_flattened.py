from __future__ import annotations

import numpy as np

from src.eval_utils.eval_flattened import (
    build_paired_flattened_tensor_matrices,
    gene_correlation_table,
    sample_correlation_table,
)
from src.eval_utils.eval_samples import TensorView


def _view(values, *, stage):
    return TensorView(
        values=np.asarray(values, dtype=np.float32),
        observed_mask=np.asarray([[True, False], [True, True]], dtype=bool),
        subjects=["S1", "S2"],
        regions=["LH_Vis_1", "Cerebellar_Region7"],
        genes=["G1", "G2", "G3"],
        dataset="GTEx",
        region_axis_kind="ahba_parcel",
        matching={
            "gtex_to_ahba": {
                "brain - cortex": "LH_Vis_1",
                "brain - cerebellum": "Cerebellar_Region7",
            }
        },
        pipeline_stage=stage,
    )


def test_build_paired_flattened_tensor_matrices_uses_observed_intersection_and_tissue_order():
    truth = _view(
        [
            [[1, 2, 3], [10, 10, 10]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        stage="loro_truth",
    )
    recon = _view(
        [
            [[1, 2, 3], [11, 11, 11]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        stage="loro_recon",
    )

    flat_truth, flat_recon = build_paired_flattened_tensor_matrices(truth, recon)

    assert flat_truth.X.shape == (3, 3)
    assert flat_recon.X.shape == (3, 3)
    assert flat_truth.row_metadata["subject"].tolist() == ["S1", "S2", "S2"]
    assert flat_truth.row_metadata["group"].tolist() == [
        "brain - cortex",
        "brain - cortex",
        "brain - cerebellum",
    ]
    assert list(flat_truth.group_slices) == ["brain - cortex", "brain - cerebellum"]


def test_flattened_correlation_tables_return_gene_and_sample_axes():
    truth = _view(
        [
            [[1, 2, 3], [10, 10, 10]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        stage="loro_truth",
    )
    recon = _view(
        [
            [[1, 2, 3], [11, 11, 11]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        stage="loro_recon",
    )
    flat_truth, flat_recon = build_paired_flattened_tensor_matrices(truth, recon)

    gene_corr = gene_correlation_table(flat_truth, flat_recon)
    sample_corr = sample_correlation_table(flat_truth, flat_recon)

    assert gene_corr["gene"].tolist() == ["G1", "G2", "G3"]
    assert np.allclose(gene_corr["pearson_r"], 1.0)
    assert sample_corr["subject"].tolist() == ["S1", "S2", "S2"]
    assert np.allclose(sample_corr["pearson_r"], 1.0)
