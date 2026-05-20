# gxp_samples.csv Builder

This document defines the reproducible path for building `data/raw/gxp_samples.csv`.
The builder lives in `src/samples` and keeps the output schema compatible with
the current modeling pipeline:

```text
subject, age, sex, dataset, tissue_or_parcel, coordinates, <gene columns...>
```

The CSV itself stays unmapped: GTEx rows retain native sampled tissue labels and
their manually assigned coordinate lists. Downstream model code may later reduce
those coordinate lists to representative points and map them to AHBA parcels.

## Inputs

### AHBA

AHBA data are prepared outside HPC from abagen-derived outputs and loaded from
`GeneEx2Conn_data/AHBA`. The default target remains S156, but the builder keeps
the parcellation path explicit so later parcellations can be added without
changing the output schema.

The ENIGMA Toolbox provides microarray expression data collected from six human
donor brains and released by the Allen Human Brain Atlas. These microarray
expression data were first generated using `abagen`, a toolbox that provides
reproducible workflows for processing and preparing gene co-expression data
according to previously established recommendations:

- ENIGMA Toolbox gene maps:
  `https://enigma-toolbox.readthedocs.io/en/latest/pages/10.genemaps/index.html`
- `abagen` recommendations:
  Arnatkeviciute et al., 2019, NeuroImage
- ENIGMA extra data release:
  `https://github.com/saratheriver/enigma-extra`

Preprocessing included intensity-based filtering of microarray probes,
selection of a representative probe for each gene across both hemispheres,
matching of microarray samples to brain parcels from the Desikan-Killiany,
Glasser, and Schaefer parcellations, normalization, and aggregation within
parcels and across donors.

Moreover, genes whose similarity across donors fell below a threshold
(`r < 0.2`) were removed, leaving a total of `12,668` genes for analysis when
using the Desikan-Killiany atlas. To accommodate users, the release also
provides unthresholded or progressively thresholded datasets with varying
stability thresholds for every parcellation.

The shipped ENIGMA/abagen-derived DK stability panels in this repo range from
`r-1` (effectively unthresholded / all genes) through `r0.2`, `r0.4`, `r0.6`,
and `r0.8`. Increasing the donor-stability threshold retains fewer genes. For
the DK atlas, `r0.2` contains `12,668` genes and is the default panel in this
builder.

The default gene panel is:

```text
data/metadata/gene_lists/allgenes_stability_dk/allgenes_stable_r0.2.csv
```

### GTEx

GTEx expression data are loaded from the reuploaded neuroVformer GTEx folders.
The builder accepts both layouts:

```text
data/GTEx/GTEx_v10/brain_RNA-seq/tpm/gene_tpm_v10_brain_<tissue>.gct
data/GTEx/GTEx_v10/brain_RNA-seq/read_counts/gene_reads_v10_brain_<tissue>.gct
data/GTEx/GTEx_v11/brain_RNA-seq/tpm/gene_tpm_v11_brain_<tissue>.gct
data/GTEx/GTEx_v11/brain_RNA-seq/read_counts/gene_reads_v11_brain_<tissue>.gct
```

The active checkout now stores GTEx under `neuroVformer/data/GTEx/`.
The current directory name is `brain_RNA-seq`. The resolver still accepts the
older `brain RNA-seq` layout for compatibility with earlier reuploads. V11
updates the annotation to GENCODE 47 but does not add new GTEx donors or
samples relative to v10.

GTEx metadata currently available in this workspace lives under:

```text
data/GTEx/metadata/
  annotations_v8_GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
  annotations_v8_GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
```

`SampleAttributesDS.txt` is sample-level and includes `SMRIN`; it should be the
source of any RIN-based filtering. `SubjectPhenotypesDS.txt` is subject-level
and supplies `SEX`, `AGE`, and `DTHHRDY`.

AHBA donor metadata lives beside the AHBA source tree:

```text
GeneEx2Conn_data/AHBA/AHBA_metadata.csv
```

This file supplies `uid`, `age`, and `sex`; `uid` is the donor identifier that
matches the per-donor AHBA expression CSV suffixes.

The default row-level GTEx transform used when writing `gxp_samples.csv` is
still `log1p(TPM)`. Before `gxp_samples.csv` is built, however, the repo now
supports an explicit GTEx gene-filtering audit path based on sample-level QC
and expression thresholds.

### GTEx Filtering Before `gxp_samples.csv`

This repo's current filtering design is intentionally brain-scoped and
pre-build:

1. restrict to brain samples using GTEx sample metadata,
2. apply `SMRIN > 6` at the sample level,
3. compute gene thresholds on the retained brain samples only,
4. intersect the retained GTEx genes with the AHBA gene universe for the chosen
   AHBA preprocessing,
5. optionally apply a smaller analysis panel such as `richiardi2015` on top of
   that overlap.

This design was chosen for three reasons:

- the downstream task is AHBA-to-brain GTEx alignment, not whole-body GTEx
  modeling;
- the `>= 20% of samples` denominator should reflect only the samples that are
  actually eligible for the build;
- separating GTEx-side filtering from the final AHBA overlap makes the loss at
  each stage auditable instead of hiding everything inside the final shared
  panel.

The current threshold helper reports the following steps:

- total GTEx genes in the cohort-wide TPM GCT,
- retained brain sample count after `SMRIN > 6`,
- genes passing `TPM >= 0.1` in `>= 20%` of retained samples,
- genes passing `reads >= 6` in `>= 20%` of retained samples,
- GTEx genes passing both thresholds,
- AHBA genes available in the selected AHBA preprocessing,
- final GTEx ∩ AHBA overlap size.

It also breaks AHBA-side overlap loss into:

- genes absent from the GTEx TPM universe,
- genes failing only the TPM threshold after RIN filtering,
- genes failing only the read-count threshold after RIN filtering,
- genes failing both thresholds after RIN filtering.

#### Workflow Details

The implementation uses the cohort-wide brain GTEx matrices rather than the
median TPM file:

- TPM universe / TPM threshold source:
  `GTEx_Analysis_*_gene_tpm.gct`
- read-count threshold source:
  `GTEx_Analysis_*_gene_reads.gct`
- sample QC / tissue filter source:
  `annotations_v8_GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`

The filtering order is important. `SMRIN > 6` is applied before the gene
thresholds because it changes the retained sample set and therefore the
`>= 20%` denominator. The GTEx side is filtered first, then intersected with
the AHBA gene universe by exact gene identifier match (symbol by default,
Ensembl-without-version when requested).

#### Effects and Interpretation

The main expected effects are:

- a substantial GTEx-side reduction from the full cohort-wide gene universe to
  a thresholded brain-only set,
- a smaller but still meaningful loss from exact GTEx-to-AHBA universe overlap,
- a potentially very large additional reduction if a small manuscript gene
  panel is layered on top of the shared set.

Interpretation should stay asymmetric:

- `SMRIN > 6`, TPM thresholding, and read-count thresholding are GTEx-side
  filters;
- AHBA genes are not "failing RIN" themselves;
- AHBA-side loss means those genes are absent from the final GTEx filtered set
  or do not match the GTEx universe under the current identifier policy.

#### References and Provenance

The GTEx-side threshold choices here are motivated by the preprocessing choices
described in related prior work:

- VariantFormer: uses GTEx v10 TPM with an RNA integrity screen (`RIN > 6`)
  before downstream modeling.
- HYFA: applies GTEx eQTL-style filtering, including expression thresholds of
  `TPM >= 0.1` in `>= 20%` of samples and read counts `>= 6` in `>= 20%` of
  samples, after restricting the sample set.

This repo does not currently reproduce the full HYFA normalization stack before
`gxp_samples.csv` (for example, TMM normalization and inverse normal
transformation are not part of the `gxp_samples.csv` builder path). The current
builder uses those references to justify the sample-level RIN gate and the
cohort-wide GTEx gene-eligibility audit, then writes the final sample table
with the repo's standard `log1p(TPM)` row transform.

The library interface now supports these GTEx-side pre-build controls directly
in `build_gxp_samples(...)`:

- `gtex_sample_attributes_path`
- `gtex_metadata_path`
- `ahba_metadata_path`
- `gtex_rin_threshold`
- `gtex_tpm_threshold`
- `gtex_reads_threshold`
- `gtex_min_fraction`
- `gtex_assay_freeze`
- `return_overlap_info`

When `gtex_sample_attributes_path` is provided, the builder computes the
GTEx-side threshold report internally, restricts the final build to the
resulting GTEx ∩ AHBA overlap (and any optional smaller panel layered on top),
and can return that overlap report alongside the dataframe via
`return_overlap_info=True`.

`gtex_metadata_path` should point to `SubjectPhenotypesDS.txt` for GTEx age/sex.
`ahba_metadata_path` should point to `AHBA_metadata.csv` for AHBA age/sex. If
these are omitted, the builder now tries the standard workspace locations
relative to `gtex_root` and `ahba_root`.

In practice this means the notebook can be split cleanly into:

1. an exploratory phase that calls `compute_gtex_ahba_overlap_report(...)`
   directly on a subsetted gene list, and
2. a formal build phase that passes the same GTEx filtering arguments into
   `build_gxp_samples(...)`.

## Gene Matching

The default matching key is gene symbol because AHBA tables are symbol-facing.
For GTEx GCT files, the builder uses `Description` as the gene column by
default and can use Ensembl IDs with `--gene-id-style ensembl`. If Ensembl IDs
or GENCODE 47 annotation are needed for manuscript auditing, they should be
written as a sidecar gene map rather than replacing the default CSV gene
columns.

VariantFormer filtering is optional. The default builder should maximize
AHBA-GTEx overlap under the selected gene panel rather than requiring
VariantFormer availability.

## GTEx Coordinates

GTEx tissues do not carry subject-specific measured coordinates. The builder
uses manually curated atlas maps to assign each sampled GTEx brain tissue a
list of plausible MNI coordinates. Every donor row for a tissue inherits the
same coordinate list.

The current mapping is:

- cortical GTEx tissues: `MaptoBA.csv` plus Brodmann coordinates
- subcortical and cerebellar GTEx tissues: `MaptoS156.csv` plus S156 coordinates

These coordinate lists are stored in `coordinates` as text. They are not the
same as the downstream `parcel_idx` assignment.

The builder resolves the coordinate-bearing atlas CSVs by schema, not only by
filename. This matters because the repo-local `data/metadata/atlas_info`
mirror may contain a Brodmann label table without `mni_x`, `mni_y`, `mni_z`,
while the full coordinate-bearing file lives in the upstream
`GeneEx2Conn_data/atlas_info` source tree.

## Duplicate GTEx Subjects Within a Tissue File

GTEx source GCTs are sample-level, not subject-level. After parsing sample IDs
down to subject IDs, duplicates can appear within a tissue file. The original
neuroVformer path averaged those duplicated subject rows before `log1p`. The
current builder does not. It raises instead.

This is intentional:

- the canonical CSV is meant to represent one GTEx row per subject per tissue,
- silent averaging changes both row count and expression values,
- technical replicate handling should be an explicit decision, not an implicit
  side effect of the builder.

Separately, GTEx symbol-mode loading can encounter duplicated gene symbols in
the GCT `Description` column. The current builder resolves that by keeping the
first occurrence of each symbol when constructing the row-level GTEx table, so
the final CSV always has unique gene columns. This is stricter and more stable
than letting duplicate symbol columns survive until final concatenation.

## Original Mean/Median Alignment Option

For reproducibility, the original neuroVformer notebook alignment remains
available as `src.samples.legacy_alignment` and is surfaced in
`notebooks/preprocessing/build_gxp_samples.ipynb`.

The original method did not map GTEx donor rows to AHBA parcel centroids. It
worked at the GTEx tissue level:

1. Load GTEx median TPM as a `tissue x gene` table.
2. Load AHBA BA and S156 aggregate CSVs.
3. Use `MaptoBA.csv` to assign BA rows to cortical GTEx tissues.
4. Use `MaptoS156.csv` to assign S156 rows to subcortical/cerebellar GTEx
   tissues.
5. Aggregate matched AHBA rows per GTEx tissue by mean or median.
6. Align genes by symbol or Ensembl-without-version.
7. Compare the GTEx tissue matrix to the AHBA-derived tissue matrix.

The original alignment grid varied:

```text
processing_style: raw, minmax, srs
subject_aggregation_style: mean, median
tissue_aggregation_style: mean, median
log1p: true, false
mirror_interpolate: true, false
```

Metrics were mean gene-wise Pearson r, mean tissue-wise Pearson r, and
region-region RSA for Pearson and Spearman tissue-correlation matrices. The
notebook visualizations were heatmaps of the two tissue-by-gene matrices plus
source/comparison tissue-correlation matrices.

The intended notebook flow is:

1. select a gene list such as all stable genes or an HVG list,
2. load the GTEx median TPM tissue-by-gene matrix,
3. run `evaluate_ahba_combination_grid(...)` across preprocessing strategies,
4. select one ranked preprocessing configuration,
5. visualize it with `evaluate_gtex_tpm_df(..., silence_plotting=False)`,
6. build `gxp_samples.csv` with the selected gene panel.

## Command

Minimal CLI:

```bash
python scripts/build_gxp_samples.py \
  --gtex-root /scratch/asr655/neuroinformatics/Seq2GeneEx/neuroVformer/data/GTEx/GTEx_v11 \
  --ahba-root /scratch/asr655/neuroinformatics/GeneEx2Conn_data/AHBA \
  --gene-panel data/metadata/gene_lists/allgenes_stability_dk/allgenes_stable_r0.2.csv \
  --gtex-sample-attributes-path /scratch/asr655/neuroinformatics/Seq2GeneEx/neuroVformer/data/GTEx/metadata/annotations_v8_GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt \
  --gtex-metadata-path /scratch/asr655/neuroinformatics/Seq2GeneEx/neuroVformer/data/GTEx/metadata/annotations_v8_GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt \
  --ahba-metadata-path /scratch/asr655/neuroinformatics/GeneEx2Conn_data/AHBA/AHBA_metadata.csv \
  --output data/raw/gxp_samples.csv
```

The script assumes source files already exist. It does not download AHBA or
GTEx data.
