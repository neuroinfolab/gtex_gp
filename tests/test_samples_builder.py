from pathlib import Path

import numpy as np
import pandas as pd

from src.samples.build import compute_gtex_ahba_overlap_report
from src.samples.gtex import list_brain_gct_files, tissue_label_from_gct
from src.samples.legacy_alignment import aggregate_parcels_to_tissues_by_key, evaluate_tissue_gene_alignment, read_gene_list
from src.samples.schema import validate_gxp_samples


def test_gtex_tissue_file_parsing_and_layout_resolution(tmp_path: Path):
    root = tmp_path / "GTEx_v11" / "brain_RNA-seq" / "tpm"
    root.mkdir(parents=True)
    keep = root / "gene_tpm_v11_brain_cortex.gct"
    keep.write_text("")
    (root / "gene_tpm_v11_brain_cortex.gct.gz").write_text("")
    (root / "gene_tpm_v11_brain_spinal_cord_cervical_c-1.gct").write_text("")
    assert tissue_label_from_gct(keep) == "brain - cortex"
    assert list_brain_gct_files(tmp_path / "GTEx_v11", expression_kind="tpm") == [keep]


def test_legacy_alignment_aggregates_ba_or_s156_keys_to_gtex_tissues():
    ahba = pd.DataFrame(
        {
            "label": ["1", "2", "3"],
            "GENE1": [1.0, 3.0, 100.0],
            "GENE2": [2.0, 4.0, 200.0],
        }
    )
    atlas = pd.DataFrame(
        {
            "Brodmann ROI value in nifti": [1, 2],
            "Genotype-Tissue Expression (GTEx v8) project": ["Cortex", "Cortex"],
        }
    )
    out = aggregate_parcels_to_tissues_by_key(
        ahba,
        atlas,
        atlas_key_col="Brodmann ROI value in nifti",
        tissue_aggregation_style="mean",
    )
    assert out.index.tolist() == ["brain - cortex"]
    assert np.isclose(out.loc["brain - cortex", "GENE1"], 2.0)
    assert np.isclose(out.loc["brain - cortex", "GENE2"], 3.0)


def test_legacy_alignment_metrics_shape():
    source = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 4.0, 8.0]],
        index=["brain - cortex", "brain - amygdala", "brain - cerebellum"],
        columns=["A", "B", "C"],
    )
    comp = source.copy()
    metrics = evaluate_tissue_gene_alignment(source, comp)
    assert set(metrics) == {
        "n_common_tissues",
        "n_overlap_genes",
        "mean_gene_r",
        "mean_tissue_r",
        "rsa_pearson",
        "rsa_spearman",
    }
    assert metrics["mean_gene_r"] > 0.99


def test_schema_validation_accepts_current_columns():
    df = pd.DataFrame(
        {
            "subject": ["s1"],
            "age": [40],
            "sex": ["F"],
            "dataset": ["GTEx"],
            "tissue_or_parcel": ["brain - cortex"],
            "coordinates": ["[(0, 0, 0)]"],
            "GENE1": [1.0],
        }
    )
    validate_gxp_samples(df)


def test_read_gene_list_accepts_region_by_gene_stability_csv(tmp_path: Path):
    p = tmp_path / "stability_matrix.csv"
    p.write_text(
        "label,A1BG,A1BG-AS1,AAAS\n"
        "L_bankssts,0.4,0.5,0.6\n"
        "L_cuneus,0.3,0.2,0.1\n"
    )
    genes = read_gene_list(p)
    assert genes == ["A1BG", "A1BG-AS1", "AAAS"]


def test_compute_gtex_ahba_overlap_report(tmp_path: Path):
    gtex_root = tmp_path / "GTEx_v10"
    tpm_dir = gtex_root / "brain_RNA-seq" / "tpm"
    reads_dir = gtex_root / "brain_RNA-seq" / "read_counts"
    tpm_dir.mkdir(parents=True)
    reads_dir.mkdir(parents=True)

    tpm_gct = tpm_dir / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct"
    tpm_gct.write_text(
        "#1.2\n"
        "3\t3\n"
        "Name\tDescription\tGTEX-A-0001-SM-1\tGTEX-B-0002-SM-2\tGTEX-C-0003-SM-3\n"
        "ENSG1.1\tG1\t1.0\t0.0\t1.0\n"
        "ENSG2.1\tG2\t0.2\t0.0\t0.0\n"
        "ENSG3.1\tG3\t0.0\t0.0\t0.0\n"
    )
    reads_gct = reads_dir / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct"
    reads_gct.write_text(
        "#1.2\n"
        "3\t3\n"
        "Name\tDescription\tGTEX-A-0001-SM-1\tGTEX-B-0002-SM-2\tGTEX-C-0003-SM-3\n"
        "ENSG1.1\tG1\t10\t0\t8\n"
        "ENSG2.1\tG2\t0\t0\t0\n"
        "ENSG3.1\tG3\t7\t0\t0\n"
    )

    sample_attrs = tmp_path / "SampleAttributesDS.txt"
    sample_attrs.write_text(
        "SAMPID\tSMTSD\tSMRIN\n"
        "GTEX-A-0001-SM-1\tBrain - Cortex\t7.0\n"
        "GTEX-B-0002-SM-2\tBrain - Cortex\t7.5\n"
        "GTEX-C-0003-SM-3\tWhole Blood\t8.0\n"
    )

    ahba_root = tmp_path / "AHBA"
    ahba_dir = ahba_root / "S156" / "microarray" / "raw"
    ahba_dir.mkdir(parents=True)
    ahba_csv = ahba_dir / "AHBA_schaefer156_1.csv"
    ahba_csv.write_text(
        ",label,G1,G3\n"
        "0,P1,1,2\n"
    )

    info = compute_gtex_ahba_overlap_report(
        gtex_root=gtex_root,
        sample_attributes_path=sample_attrs,
        ahba_root=ahba_root,
        ahba_processing="raw",
        gene_id_style="symbol",
    )
    report = info["report"]
    assert report["total_gtex_genes_in_tpm"] == 3
    assert report["retained_brain_samples_after_rin"] == 2
    assert report["genes_passing_tpm_threshold"] == 2
    assert report["genes_passing_reads_threshold"] == 2
    assert report["gtex_genes_passing_both_thresholds"] == 1
    assert report["ahba_genes_available_in_chosen_preprocessing"] == 2
    assert report["final_gtex_ahba_overlap_size"] == 1
    assert info["filtered_genes"] == ["G1"]
