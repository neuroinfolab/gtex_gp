from __future__ import annotations

import pandas as pd

from src.spatial.parcel_matching import (
    AtlasOverlapPaths,
    apply_gtex_ahba_matching_policy,
    apply_tissue_parcel_overrides,
    build_gtex_ahba_assignment_variant_tables,
    build_expression_pair_score_pivot,
    compute_cerebellar_expression_rankings,
    compute_ba_schaefer_overlap_table,
    compute_expression_parcel_rankings,
    run_expression_pair_matching,
    summarize_expression_parcel_rankings,
    summarize_cerebellar_expression_rankings,
    top_overlap_overrides,
)


def _target(labels):
    return pd.DataFrame(
        {
            "parcel_idx": list(range(len(labels))),
            "tissue_or_parcel": labels,
        }
    )


def test_cortical_ba_overlap_defaults_find_expected_top_parcels():
    df = compute_ba_schaefer_overlap_table(paths=AtlasOverlapPaths(), top_k=3)
    overrides = top_overlap_overrides(df)
    top = dict(zip(overrides["source_gtex_tissue"], overrides["override_mapped_parcel"]))

    assert top["brain - frontal cortex (ba9)"] == "LH_SalVentAttn_PFCl_1"
    assert top["brain - anterior cingulate cortex (ba24)"] == "LH_SalVentAttn_Med_1"
    assert top["brain - cortex"] == "RH_Cont_PFCl_1"

    cortex = overrides[overrides["source_gtex_tissue"] == "brain - cortex"].iloc[0]
    assert int(cortex["ba_id"]) == 110
    assert cortex["source_hemi_mode"] == "native"
    assert bool(cortex["preferred_hemisphere_relaxed"]) is False


def test_apply_tissue_parcel_overrides_preserves_coordinate_assignment_columns():
    gtex = pd.DataFrame(
        {
            "tissue_or_parcel": ["brain - cortex", "brain - hippocampus"],
            "parcel_idx": [1, 2],
            "mapped_parcel": ["old_coordinate_match", "keep"],
            "mapping_distance": [10.0, 2.0],
        }
    )
    target = pd.DataFrame(
        {
            "parcel_idx": [83, 2],
            "tissue_or_parcel": ["RH_Cont_PFCl_1", "keep"],
        }
    )
    overrides = pd.DataFrame(
        {
            "source_gtex_tissue": ["brain - cortex"],
            "override_mapped_parcel": ["RH_Cont_PFCl_1"],
            "override_reason": ["test_override"],
        }
    )

    out = apply_tissue_parcel_overrides(gtex, target, overrides)

    assert int(out.loc[0, "parcel_idx"]) == 83
    assert out.loc[0, "mapped_parcel"] == "RH_Cont_PFCl_1"
    assert out.loc[0, "coordinate_mapped_parcel"] == "old_coordinate_match"
    assert out.loc[0, "mapping_override_reason"] == "test_override"
    assert int(out.loc[1, "parcel_idx"]) == 2


def test_matching_policy_centroids_adds_audit_columns_without_changing_assignments():
    gtex = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "tissue_or_parcel": ["brain - hippocampus", "brain - putamen (basal ganglia)"],
            "parcel_idx": [1, 2],
            "mapped_parcel": ["LH_Hippocampus", "LH-Pu"],
            "mapping_distance": [3.0, 4.0],
        }
    )
    out = apply_gtex_ahba_matching_policy(
        gtex,
        _target(["unused", "LH_Hippocampus", "LH-Pu"]),
        matching_policy="centroids",
    )

    assert out["parcel_idx"].tolist() == [1, 2]
    assert out["mapped_parcel"].tolist() == ["LH_Hippocampus", "LH-Pu"]
    assert out["coordinate_mapped_parcel"].tolist() == ["LH_Hippocampus", "LH-Pu"]
    assert set(out["matching_rule"]) == {"coordinate_nearest_centroid"}


def test_matching_policy_force_left_centroids_restricts_coordinate_candidates():
    target = pd.DataFrame(
        {
            "parcel_idx": [0, 1],
            "tissue_or_parcel": ["RH_TestParcel", "LH_TestParcel"],
            "coord_x": [0.0, 10.0],
            "coord_y": [0.0, 0.0],
            "coord_z": [0.0, 0.0],
        }
    )
    gtex = pd.DataFrame(
        {
            "subject": ["S1"],
            "tissue_or_parcel": ["brain - hippocampus"],
            "coord_x": [0.1],
            "coord_y": [0.0],
            "coord_z": [0.0],
            "parcel_idx": [0],
            "mapped_parcel": ["RH_TestParcel"],
            "mapping_distance": [0.1],
        }
    )

    out = apply_gtex_ahba_matching_policy(
        gtex,
        target,
        matching_policy="centroids",
        matching_policy_hemi_mode="force_left",
    )

    assert out["mapped_parcel"].tolist() == ["LH_TestParcel"]
    assert out["parcel_idx"].tolist() == [1]
    assert set(out["matching_rule"]) == {"coordinate_nearest_left_centroid"}


def test_assignment_variant_tables_collapse_to_one_to_one_mappings():
    samples = pd.DataFrame(
        {
            "subject": ["A1", "A2", "A3", "G1", "G2"],
            "age": ["", "", "", "", ""],
            "sex": ["", "", "", "", ""],
            "dataset": ["AHBA", "AHBA", "AHBA", "GTEx", "GTEx"],
            "tissue_or_parcel": [
                "LH_TestParcel",
                "RH_TestParcel",
                "Cerebellar_Region7",
                "brain - test",
                "brain - test",
            ],
            "coordinates": [
                "(10, 0, 0)",
                "(0, 0, 0)",
                "(0, 10, 0)",
                "(0.1, 0, 0)",
                "(0.2, 0, 0)",
            ],
        }
    )

    out = build_gtex_ahba_assignment_variant_tables(
        samples,
        variants=(
            ("centroid", "centroids", "default"),
            ("centroid_force_left", "centroids", "force_left"),
        ),
    )
    centroid = out["centroid"]["mapping"].set_index("original_gtex_tissue")
    force_left = out["centroid_force_left"]["mapping"].set_index("original_gtex_tissue")

    assert centroid.loc["brain - test", "final_mapped_parcel"] == "RH_TestParcel"
    assert centroid.loc["brain - test", "matching_rule"] == "coordinate_nearest_centroid"
    assert int(centroid.loc["brain - test", "n_input_rows"]) == 2
    assert int(centroid.loc["brain - test", "n_policy_rows"]) == 2
    assert int(centroid.loc["brain - test", "n_final_parcels_for_source"]) == 1

    assert force_left.loc["brain - test", "final_mapped_parcel"] == "LH_TestParcel"
    assert force_left.loc["brain - test", "matching_rule"] == "coordinate_nearest_left_centroid"
    assert int(force_left.loc["brain - test", "n_dropped_rows"]) == 0


def test_matching_policy_centroids_and_volumes_cortex_from_computed_ba_overlap():
    target = _target(
        [
            "old_coordinate_match",
            "LH_SalVentAttn_PFCl_1",
            "LH_SalVentAttn_Med_1",
            "RH_Cont_PFCl_1",
            "Cerebellar_Region7",
        ]
    )
    gtex = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1"],
            "tissue_or_parcel": [
                "brain - frontal cortex (ba9)",
                "brain - anterior cingulate cortex (ba24)",
                "brain - cortex",
                "brain - hippocampus",
            ],
            "parcel_idx": [0, 0, 0, 0],
            "mapped_parcel": ["old_coordinate_match"] * 4,
            "mapping_distance": [10.0, 11.0, 12.0, 2.0],
        }
    )

    out = apply_gtex_ahba_matching_policy(
        gtex,
        target,
        matching_policy="centroids_and_volumes",
        atlas_paths=AtlasOverlapPaths(),
        validate_expected=True,
    )
    by_tissue = out.set_index("original_gtex_tissue")

    assert by_tissue.loc["brain - frontal cortex (ba9)", "mapped_parcel"] == "LH_SalVentAttn_PFCl_1"
    assert by_tissue.loc["brain - anterior cingulate cortex (ba24)", "mapped_parcel"] == "LH_SalVentAttn_Med_1"
    assert by_tissue.loc["brain - cortex", "mapped_parcel"] == "RH_Cont_PFCl_1"
    assert by_tissue.loc["brain - hippocampus", "mapped_parcel"] == "old_coordinate_match"
    assert by_tissue.loc["brain - cortex", "coordinate_mapped_parcel"] == "old_coordinate_match"
    assert by_tissue.loc["brain - cortex", "matching_rule"] == "cortical_ba_schaefer_voxel_overlap"


def test_matching_policy_centroids_and_volumes_cerebellar_manual_assignments_by_default():
    target = _target(["old_coordinate_match", "Cerebellar_Region4", "Cerebellar_Region7"])
    gtex = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S3"],
            "tissue_or_parcel": [
                "brain - cerebellum",
                "brain - cerebellar hemisphere",
                "brain - cerebellum",
                "brain - hippocampus",
            ],
            "parcel_idx": [0, 0, 0, 0],
            "mapped_parcel": ["old_coordinate_match"] * 4,
            "mapping_distance": [10.0, 11.0, 12.0, 2.0],
        }
    )

    out = apply_gtex_ahba_matching_policy(
        gtex,
        target,
        matching_policy="centroids_and_volumes",
        atlas_paths=AtlasOverlapPaths(),
        validate_expected=True,
    )

    assert len(out) == 4
    by_tissue = out.set_index("original_gtex_tissue")
    assert by_tissue.loc["brain - cerebellar hemisphere", "mapped_parcel"] == "Cerebellar_Region4"
    assert set(out[out["original_gtex_tissue"] == "brain - cerebellum"]["mapped_parcel"]) == {"Cerebellar_Region7"}
    assert set(out[out["original_gtex_tissue"].isin(["brain - cerebellum", "brain - cerebellar hemisphere"])]["matching_rule"]) == {
        "cerebellar_manual_region_policy"
    }


def test_matching_policy_centroids_and_volumes_cerebellar_collapse_is_opt_in():
    target = _target(["old_coordinate_match", "Cerebellar_Region4", "Cerebellar_Region7"])
    gtex = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S3"],
            "tissue_or_parcel": [
                "brain - cerebellum",
                "brain - cerebellar hemisphere",
                "brain - cerebellum",
                "brain - hippocampus",
            ],
            "parcel_idx": [0, 0, 0, 0],
            "mapped_parcel": ["old_coordinate_match"] * 4,
            "mapping_distance": [10.0, 11.0, 12.0, 2.0],
        }
    )

    out = apply_gtex_ahba_matching_policy(
        gtex,
        target,
        matching_policy="centroids_and_volumes",
        collapse_cerebellum=True,
        atlas_paths=AtlasOverlapPaths(),
        validate_expected=True,
    )

    assert len(out) == 3
    assert not ((out["subject"] == "S1") & (out["original_gtex_tissue"] == "brain - cerebellum")).any()
    cereb = out[out["normalized_gtex_tissue"] == "Cerebellar hemisphere"].sort_values("subject")
    assert cereb["subject"].tolist() == ["S1", "S2"]
    assert set(cereb["mapped_parcel"]) == {"Cerebellar_Region7"}
    assert set(cereb["tissue_or_parcel"]) == {"Cerebellar hemisphere"}
    assert set(cereb["matching_rule"]) == {"cerebellar_region7_expression_policy"}


def test_cerebellar_expression_rankings_and_consensus():
    samples = pd.DataFrame(
        {
            "subject": ["A1", "A2", "A3", "G1", "G2"],
            "age": ["", "", "", "", ""],
            "sex": ["", "", "", "", ""],
            "dataset": ["AHBA", "AHBA", "AHBA", "GTEx", "GTEx"],
            "tissue_or_parcel": [
                "Cerebellar_Region1",
                "Cerebellar_Region2",
                "LH_Default_PFC_1",
                "brain - cerebellum",
                "brain - cerebellar hemisphere",
            ],
            "coordinates": [""] * 5,
            "G1": [1.0, 4.0, 8.0, 1.1, 4.2],
            "G2": [2.0, 3.0, 8.0, 2.1, 3.2],
            "G3": [3.0, 2.0, 8.0, 3.1, 2.2],
            "G4": [4.0, 1.0, 8.0, 4.1, 1.2],
        }
    )
    rankings = compute_cerebellar_expression_rankings(
        samples,
        {"toy": ["G1", "G2", "G3", "G4"]},
    )
    top = rankings[rankings["rank_pearson"] == 1]
    top_by_tissue = dict(zip(top["source_gtex_tissue"], top["ahba_cerebellar_parcel"]))

    assert top_by_tissue["brain - cerebellum"] == "Cerebellar_Region1"
    assert top_by_tissue["brain - cerebellar hemisphere"] == "Cerebellar_Region2"

    consensus = summarize_cerebellar_expression_rankings(rankings)
    assert set(consensus["source_gtex_tissue"]) == {"brain - cerebellum", "brain - cerebellar hemisphere"}


def test_generic_expression_rankings_support_selected_target_parcels():
    samples = pd.DataFrame(
        {
            "subject": ["A1", "A2", "A3", "G1", "G2"],
            "age": ["", "", "", "", ""],
            "sex": ["", "", "", "", ""],
            "dataset": ["AHBA", "AHBA", "AHBA", "GTEx", "GTEx"],
            "tissue_or_parcel": [
                "LH_SalVentAttn_PFCl_1",
                "RH_Cont_PFCl_1",
                "Cerebellar_Region1",
                "brain - frontal cortex (ba9)",
                "brain - cortex",
            ],
            "coordinates": [""] * 5,
            "G1": [1.0, 4.0, 8.0, 1.1, 4.2],
            "G2": [2.0, 3.0, 8.0, 2.1, 3.2],
            "G3": [3.0, 2.0, 8.0, 3.1, 2.2],
            "G4": [4.0, 1.0, 8.0, 4.1, 1.2],
        }
    )
    rankings = compute_expression_parcel_rankings(
        samples,
        {"toy": ["G1", "G2", "G3", "G4"]},
        source_tissues=["brain - frontal cortex (ba9)", "brain - cortex"],
        target_parcels=["LH_SalVentAttn_PFCl_1", "RH_Cont_PFCl_1"],
        target_col_name="ahba_candidate_parcel",
    )
    top = rankings[rankings["rank_pearson"] == 1]
    top_by_tissue = dict(zip(top["source_gtex_tissue"], top["ahba_candidate_parcel"]))

    assert top_by_tissue["brain - frontal cortex (ba9)"] == "LH_SalVentAttn_PFCl_1"
    assert top_by_tissue["brain - cortex"] == "RH_Cont_PFCl_1"

    consensus = summarize_expression_parcel_rankings(rankings, target_col_name="ahba_candidate_parcel")
    assert set(consensus["source_gtex_tissue"]) == {"brain - frontal cortex (ba9)", "brain - cortex"}

    result = run_expression_pair_matching(
        samples,
        {"toy": ["G1", "G2", "G3", "G4"]},
        source_tissues=["brain - frontal cortex (ba9)", "brain - cortex"],
        target_parcels=["LH_SalVentAttn_PFCl_1", "RH_Cont_PFCl_1"],
        target_col_name="ahba_candidate_parcel",
    )
    pivot = build_expression_pair_score_pivot(
        result["rankings"],
        source_tissues=["brain - frontal cortex (ba9)", "brain - cortex"],
        target_col_name="ahba_candidate_parcel",
    )
    assert {"rankings", "consensus", "profile_similarity"} <= set(result)
    assert set(pivot["ahba_candidate_parcel"]) == {"LH_SalVentAttn_PFCl_1", "RH_Cont_PFCl_1"}
