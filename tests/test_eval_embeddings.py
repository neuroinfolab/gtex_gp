import numpy as np
import pandas as pd

from src.eval_utils.eval_embeddings import (
    add_gtex_native_region_labels,
    add_parcel_network_labels,
    build_gtex_region_embedding_matrix,
    build_prepost_region_embedding_matrix,
    build_region_embedding_matrix_from_joint_tensor_view,
    build_region_embedding_matrix_from_tensor_view,
    embedding_color_spec,
    preprocess_expression_features,
)
from src.eval_utils.eval_samples import TensorView, build_joint_tensor_view


def _toy_prepost():
    raw_cube = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [np.nan, np.nan]],
        ],
        dtype=float,
    )
    harm_cube = raw_cube + 10.0
    return {
        "subjects": ["S1", "S2"],
        "genes": ["G1", "G2"],
        "raw_cube": raw_cube,
        "harm_cube": harm_cube,
        "obs_mask": np.array([[True, True], [True, False]], dtype=bool),
        "target_meta": pd.DataFrame(
            {
                "parcel_idx": [0, 1],
                "tissue_or_parcel": ["LH_Vis_1", "Cerebellar_Region7"],
                "macro_system": ["visual_somatomotor", "cerebellar"],
                "hemisphere": ["L", "B"],
            }
        ),
        "gtex_eligible_raw": pd.DataFrame(
            {
                "subject": ["S1", "S1", "S2"],
                "parcel_idx": [0, 1, 0],
                "age": [40, 40, 50],
                "sex": ["F", "F", "M"],
                "tissue_or_parcel": ["Brain - Cortex", "Brain - Cerebellum", "Brain - Cortex"],
            }
        ),
        "ahba_raw": pd.DataFrame(
            {
                "subject": ["A1", "A1", "A2"],
                "parcel_idx": [0, 1, 0],
                "G1": [1.0, 3.0, 5.0],
                "G2": [2.0, 4.0, 6.0],
            }
        ),
        "ahba_h": pd.DataFrame(
            {
                "subject": ["A1", "A1", "A2"],
                "parcel_idx": [0, 1, 0],
                "G1": [11.0, 13.0, 15.0],
                "G2": [12.0, 14.0, 16.0],
            }
        ),
    }


def test_build_gtex_region_embedding_matrix_flattens_observed_rows():
    mat = build_gtex_region_embedding_matrix(_toy_prepost(), stage="raw")

    assert mat.stage == "raw_matched"
    assert mat.X.shape == (3, 2)
    assert mat.genes == ["G1", "G2"]
    assert mat.dropped_rows == 0
    assert mat.metadata["region"].tolist() == ["LH_Vis_1", "Cerebellar_Region7", "LH_Vis_1"]
    assert mat.metadata["region_group"].tolist() == ["cortical", "cerebellar", "cortical"]
    assert mat.metadata["age"].tolist() == [40, 40, 50]


def test_build_prepost_region_embedding_matrix_supports_ahba_dataframe_source():
    mat = build_prepost_region_embedding_matrix(_toy_prepost(), dataset="AHBA", stage="harmonized")

    assert mat.X.shape == (3, 2)
    assert mat.dataset == "AHBA"
    assert mat.stage == "harmonized"
    assert mat.metadata["region"].tolist() == ["LH_Vis_1", "Cerebellar_Region7", "LH_Vis_1"]


def test_build_region_embedding_matrix_from_tensor_view_is_core_adapter():
    view = TensorView(
        values=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [np.nan, np.nan]],
            ],
            dtype=float,
        ),
        observed_mask=np.array([[True, True], [True, False]], dtype=bool),
        subjects=["S1", "S2"],
        regions=["LH_Vis_1", "Cerebellar_Region7"],
        genes=["G1", "G2"],
        dataset="GTEx",
        region_axis_kind="ahba_parcel",
        future_imputation_mask=np.array([[False, False], [False, True]], dtype=bool),
        pipeline_stage="harmonized",
    )

    mat = build_region_embedding_matrix_from_tensor_view(view)

    assert mat.X.shape == (3, 2)
    assert mat.dataset == "GTEx"
    assert mat.stage == "harmonized"
    assert mat.region_axis_kind == "ahba_parcel"
    assert mat.metadata["region"].tolist() == ["LH_Vis_1", "Cerebellar_Region7", "LH_Vis_1"]
    assert mat.metadata["macro_system"].tolist() == ["visual_somatomotor", "cerebellar", "visual_somatomotor"]


def test_build_region_embedding_matrix_from_joint_tensor_view_keeps_dataset_labels():
    gtex = TensorView(
        values=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
        observed_mask=np.array([[True, True]], dtype=bool),
        subjects=["G1"],
        regions=["LH_Vis_1", "Cerebellar_Region7"],
        genes=["G1", "G2"],
        dataset="GTEx",
        region_axis_kind="ahba_parcel",
        pipeline_stage="harmonized",
    )
    ahba = TensorView(
        values=np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=float),
        observed_mask=np.array([[True, True]], dtype=bool),
        subjects=["A1"],
        regions=["LH_Vis_1", "Cerebellar_Region7"],
        genes=["G1", "G2"],
        dataset="AHBA",
        region_axis_kind="ahba_parcel",
        pipeline_stage="harmonized",
    )
    joint = build_joint_tensor_view(gtex, ahba)
    mat = build_region_embedding_matrix_from_joint_tensor_view(joint)

    assert mat.X.shape == (4, 2)
    assert mat.dataset == "joint"
    assert mat.stage == "harmonized"
    assert mat.metadata["dataset"].tolist() == ["GTEx", "GTEx", "AHBA", "AHBA"]
    assert mat.metadata["region"].tolist() == ["LH_Vis_1", "Cerebellar_Region7", "LH_Vis_1", "Cerebellar_Region7"]


def test_preprocess_expression_features_modes():
    X = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])

    centered, centered_stats = preprocess_expression_features(X, mode="center")
    assert centered_stats.mode == "center"
    assert np.allclose(centered.mean(axis=0), [0.0, 0.0])

    standardized, std_stats = preprocess_expression_features(X, mode="standardize")
    assert std_stats.mode == "standardize"
    assert std_stats.zero_scale_features == 1
    assert np.allclose(standardized[:, 0].std(ddof=0), 1.0)
    assert np.allclose(standardized[:, 1], 0.0)

    unchanged, none_stats = preprocess_expression_features(X, mode="none")
    assert none_stats.mode == "none"
    assert np.allclose(unchanged, X)


def test_embedding_color_spec_supports_region_and_macro_system():
    mat = build_gtex_region_embedding_matrix(_toy_prepost(), stage="harmonized")
    df = mat.metadata.copy()
    df["embed_x"] = [0.0, 1.0, 2.0]
    df["embed_y"] = [0.0, 1.0, 2.0]

    key, order, palette = embedding_color_spec(df, color_by="region", max_legend_items=None)
    assert key == "region"
    assert set(order) == {"LH_Vis_1", "Cerebellar_Region7"}
    assert set(order).issubset(palette)

    key, order, palette = embedding_color_spec(df, color_by="macro_system")
    assert key == "macro_system"
    assert set(order) == {"visual_somatomotor", "cerebellar"}
    assert set(order).issubset(palette)

    key, order, palette = embedding_color_spec(df, color_by="dataset")
    assert key == "dataset"
    assert order == ["GTEx"]
    assert set(order).issubset(palette)

    key, order, palette = embedding_color_spec(df, color_by="sex")
    assert key == "sex"
    assert set(order) == {"F", "M"}
    assert set(order).issubset(palette)

    key, order, palette = embedding_color_spec(df, color_by="age")
    assert key == "age"
    assert order == []
    assert palette == {}


def test_macro_system_palette_separates_visual_somatomotor_and_association():
    df = pd.DataFrame(
        {
            "macro_system": ["Visual/Somatomotor", "Cortical Association"],
            "region_group": ["cortical", "cortical"],
            "embed_x": [0.0, 1.0],
            "embed_y": [0.0, 1.0],
        }
    )

    key, order, palette = embedding_color_spec(df, color_by="macro_system")

    assert key == "macro_system"
    assert set(order) == {"Visual/Somatomotor", "Cortical Association"}
    assert palette["Visual/Somatomotor"] != palette["Cortical Association"]


def test_add_parcel_network_labels_enables_network_color_key(tmp_path):
    atlas = tmp_path / "atlas.csv"
    atlas.write_text(
        "label,network_label,network_label_17network\n"
        "LH_Vis_1,Vis,VisCent\n"
        "LH_DorsAttn_Post_1,DorsAttn,DorsAttnA\n"
    )
    mat = build_gtex_region_embedding_matrix(_toy_prepost(), stage="harmonized")

    labeled = add_parcel_network_labels(mat, atlas)

    assert labeled.metadata["network_label"].tolist() == ["Vis", "Cerebellar", "Vis"]
    key, order, palette = embedding_color_spec(labeled.metadata, color_by="network_label")
    assert key == "network_label"
    assert {"Vis", "Cerebellar"}.issubset(order)
    assert palette["Vis"] != palette["Cerebellar"]


def test_add_gtex_native_region_labels_enables_native_color_key():
    mat = build_region_embedding_matrix_from_tensor_view(
        TensorView(
            values=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
            observed_mask=np.array([[True, True]], dtype=bool),
            subjects=["S1"],
            regions=["LH_Vis_1", "Cerebellar_Region7"],
            genes=["G1", "G2"],
            dataset="GTEx",
            region_axis_kind="ahba_parcel",
            pipeline_stage="loro_recon",
        )
    )

    labeled = add_gtex_native_region_labels(mat, _toy_prepost())
    assert labeled.metadata["gtex_native_region"].tolist() == ["Brain - Cortex", "Brain - Cerebellum"]
    assert labeled.metadata["age"].tolist() == [40, 40]
    assert labeled.metadata["sex"].tolist() == ["F", "F"]

    key, order, palette = embedding_color_spec(labeled.metadata, color_by="gtex_native_region")
    assert key == "gtex_native_region"
    assert set(order) == {"Brain - Cortex", "Brain - Cerebellum"}
    assert set(order).issubset(palette)
