# GTEx-to-AHBA Parcel Matching Procedure

This document specifies the manuscript-facing policy for assigning GTEx brain tissue samples to AHBA-derived target parcels in `gtex_gp`. It separates the default coordinate-based assignment from anatomically motivated overrides used for cortical and cerebellar GTEx labels.

Companion notebook: [`parcel_assignment.ipynb`](../../parcel_assignment.ipynb). Use the notebook for exploratory coordinate plots, Brodmann/Schaefer voxel-overlap figures, and expression-similarity diagnostics. Use this document as the stable written procedure and implementation reference.

## Scope

The target parcel axis is defined by AHBA labels in `data/raw/gxp_samples.csv`. Downstream model matrices use the `parcel_idx` order constructed by sorting unique AHBA `tissue_or_parcel` labels alphabetically and assigning contiguous indices.

GTEx rows contain native GTEx tissue labels and atlas-derived coordinate lists. The modeling pipeline reduces each GTEx coordinate list to a representative coordinate and maps it to the AHBA parcel axis.

## Input Resources

Primary local resources:

- `data/raw/gxp_samples.csv`: combined AHBA and GTEx expression table.
- `data/raw/atlas_info/atlas-4S156Parcels_dseg_reformatted.csv`: 4S156 label table used for Schaefer, subcortical, thalamic, hippocampal/amygdala, and cerebellar target labels.
- `data/raw/atlas_info/atlas-4S156Parcels_dseg.json`: atlas provenance and source-atlas metadata.
- `data/raw/atlas_info/overlay_brodmann_2mm_MNI_reformatted.csv`: Brodmann label table used for cortical BA priors.
- `data/raw/atlas_info/atlas-4S156Parcels_space-MNI152NLin6Asym_dseg.nii.gz`: 4S156 label volume.
- `data/raw/atlas_info/overlay_brodmann_2mm_MNI.nii`: Brodmann label volume.

External anatomical references:

- GTEx Portal sampling site page: <https://gtexportal.org/home/samplingSitePage>
- NCI GTEx SOP library, including `BBRB-PR-0004-W1-G4 Brain Autopsy Normal Tissue Collection`: <https://dctd.cancer.gov/data-tools-biospecimens/biospecimens-biobanks/resources/sops/gtex>
- UBERON cortex (`UBERON:0001870`): <https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0001870>
- UBERON frontal cortex / BA9 GTEx reference (`UBERON:0009834`): <https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0009834>
- UBERON anterior cingulate cortex / BA24 GTEx reference (`UBERON:0009835`): <https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0009835>
- UBERON cerebellum (`UBERON:0002037`): <https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0002037>

## Default Coordinate Matching

The default assignment remains coordinate based.

1. GTEx and AHBA coordinate strings are parsed from `gxp_samples.csv`.
2. Each coordinate list is reduced to one representative point (`centroid` by default in the active LORO cache workflow).
3. Coordinates are represented in a common hemisphere space for modeling (`mirror_left` by default in the active LORO cache workflow).
4. AHBA target parcels are represented by their parcel centroids.
5. Each GTEx representative point is assigned to the nearest AHBA target centroid by Euclidean distance.

This procedure is retained as the default for labels where the anatomical label maps clearly to one 4S156 parcel family.

## Subcortical Matching Policy

Subcortical GTEx labels map cleanly to named subcortical structures in the 4S156 parcellation. For these labels, nearest-centroid matching is expected to be stable and manually validatable because the GTEx tissue name and the 4S156 label name refer to the same anatomical structure.

Manual validation uses:

- `atlas-4S156Parcels_dseg.json`, which identifies the non-cortical source atlases.
- `atlas-4S156Parcels_dseg_reformatted.csv`, which contains parcel labels, source atlas names, structure class, hemisphere, and MNI centroids.

Suggested manual checks:

| GTEx label | Expected 4S156/AHBA structural target family | Source atlas in 4S156 metadata | Notes |
|---|---|---|---|
| `brain - amygdala` | `LH_Amygdala`, `RH_Amygdala` | `SubcorticalHCP` / HCP subcortical parcellation | Direct anatomical label match. |
| `brain - hippocampus` | `LH_Hippocampus`, `RH_Hippocampus` | `SubcorticalHCP` / HCP subcortical parcellation | Direct anatomical label match. |
| `brain - caudate (basal ganglia)` | `LH-Ca`, `RH-Ca` | `CIT168Subcortical` | `Ca` is defined as caudate nucleus in the JSON `RegionNames`. |
| `brain - putamen (basal ganglia)` | `LH-Pu`, `RH-Pu` | `CIT168Subcortical` | `Pu` is defined as putamen in the JSON `RegionNames`. |
| `brain - nucleus accumbens (basal ganglia)` | `LH-NAC`, `RH-NAC` | `CIT168Subcortical` | `NAC` is defined as nucleus accumbens in the JSON `RegionNames`. |
| `brain - substantia nigra` | `LH-SNc_PBP_VTA`, `RH-SNc_PBP_VTA`; nearby `LH-SNr`, `RH-SNr` | `CIT168Subcortical` | CIT168 separates pars compacta/PBP/VTA and pars reticulata labels. This should be manually checked against the final coordinate assignment. |
| `brain - hypothalamus` | `LH-HTH`, `RH-HTH` | `CIT168Subcortical` | `HTH` is defined as hypothalamus in the JSON `RegionNames`. |

Because the active model space is hemisphere-normalized, bilateral native labels may be represented in a single hemisphere coordinate system after assignment. The manual check should therefore validate the structure identity first and the hemisphere-transformed coordinate position second.

## Cortical Override Policy

The broad cortical GTEx labels are not sufficiently resolved by centroid matching alone. These labels are therefore assigned by a Brodmann-to-Schaefer voxel-overlap override.

External GTEx and UBERON references provide the anatomical priors:

- `brain - frontal cortex (ba9)` is treated as BA9.
- `brain - anterior cingulate cortex (ba24)` is treated as BA24.
- `brain - cortex` is treated as right frontal pole / anterior prefrontal cortex; the right-hemisphere BA110 Brodmann label is used as the practical right hemisphere BA10-like proxy available in the local Brodmann atlas.

Override computation:

1. Load Brodmann and 4S156/Schaefer NIfTI label volumes.
2. Evaluate the BA source masks in native hemisphere space for BA9, BA24, and BA110.
3. Compute voxel-wise overlaps between each BA mask and candidate Schaefer cortical parcels.
4. Select the Schaefer parcel with maximal overlap for each GTEx cortical label.
5. Derive the GTEx-to-AHBA cortical assignment from these voxel-overlap winners when the volume-aware matching policy is selected.
6. Validate the derived winners against the expected defaults below during smoke tests and production runs.
7. Represent all assigned parcels in the common hemisphere-normalized modeling space after the assignment is made.

Current voxel-overlap winners from the local atlas resources:

| GTEx label | Anatomical prior | Override AHBA/Schaefer parcel |
|---|---|---|
| `brain - frontal cortex (ba9)` | BA9 | `LH_SalVentAttn_PFCl_1` |
| `brain - anterior cingulate cortex (ba24)` | BA24 | `LH_SalVentAttn_Med_1` |
| `brain - cortex` | BA110, right anterior prefrontal proxy for right frontal pole / BA10-like cortex | `RH_Cont_PFCl_1` |

Gene-expression similarity is used only as a secondary diagnostic for ties or ambiguous cases. It is not the primary cortical assignment rule.

## Cerebellar Override Policy

GTEx includes both `brain - cerebellum` and `brain - cerebellar hemisphere`. Empirically, these labels have highly similar expression-based AHBA parcel profiles and both point most strongly to `Cerebellar_Region7` in the current gene-list diagnostics.

The production policy is:

1. Prefer `brain - cerebellar hemisphere` when a subject has both cerebellar labels.
2. Discard `brain - cerebellum` for that subject when `brain - cerebellar hemisphere` is present, because the two samples behave as duplicates for parcel-matching purposes.
3. If a subject lacks `brain - cerebellar hemisphere` but has `brain - cerebellum`, use the `brain - cerebellum` expression row as a fallback.
4. Assign either accepted row to `Cerebellar_Region7`.
5. Normalize the output GTEx label to `Cerebellar hemisphere` for downstream reporting, regardless of whether the input row was `brain - cerebellar hemisphere` or fallback `brain - cerebellum`.

This policy is intentionally conservative: it avoids double-counting two near-duplicate cerebellar measurements while preserving one cerebellar observation when only the broader cerebellum label is available.

## Assignment Precedence

When implemented in preprocessing, matching should be applied in this order:

1. Build the default coordinate-based GTEx-to-AHBA assignment.
2. Apply cortical Brodmann/Schaefer hard overrides for BA9, BA24, and BA110 labels.
3. Apply cerebellar duplicate resolution and `Cerebellar_Region7` assignment.
4. Retain default coordinate matching for subcortical labels, with the manual validation table above as the audit reference.
5. Preserve the original GTEx label in an audit column and expose the final normalized label used for modeling/reporting.

Recommended audit columns:

- `original_gtex_tissue`
- `normalized_gtex_tissue`
- `coordinate_mapped_parcel`
- `coordinate_mapping_distance`
- `final_mapped_parcel`
- `matching_rule`
- `matching_rule_detail`

## Status

The active LORO cache pipeline exposes this as a matching policy choice:

- `centroids`: default coordinate/centroid nearest-neighbor matching.
- `centroids_and_volumes`: coordinate/centroid matching followed by derived cortical BA/Schaefer voxel-overlap assignment and cerebellar `Cerebellar_Region7` duplicate/fallback handling.

The exploratory notebook does not write diagnostic CSVs by default; set `SAVE_NOTEBOOK_TABLES = True` in [`parcel_assignment.ipynb`](../../parcel_assignment.ipynb) only when refreshing notebook-derived tables under `out/`.
