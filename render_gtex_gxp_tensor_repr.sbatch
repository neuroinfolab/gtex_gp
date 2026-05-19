#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --mem=12G
#SBATCH --job-name=gtex_tensor_repr
#SBATCH --output=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/render_gtex_gxp_tensor_repr_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/render_gtex_gxp_tensor_repr_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp}"
CSV_PATH="${CSV_PATH:-data/raw/gxp_samples.csv}"
ACTIVE_GENE_LIST="${ACTIVE_GENE_LIST:-richiardi2015}"
REGION_ORDERING="${REGION_ORDERING:-region_matched_superset}"
MATCHING_POLICY="${MATCHING_POLICY:-centroids_and_volumes}"
MATCHING_POLICY_HEMI_MODE="${MATCHING_POLICY_HEMI_MODE:-force_left}"
COLLAPSE_CEREBELLUM="${COLLAPSE_CEREBELLUM:-false}"
GTEX_REP_MODE="${GTEX_REP_MODE:-centroid}"
GTEX_HEMI_MODE="${GTEX_HEMI_MODE:-mirror_left}"
N_SUBJECTS="${N_SUBJECTS:-75}"
N_GENES="${N_GENES:-100}"
N_REGIONS="${N_REGIONS:-none}"
MIN_REGIONS_PER_SUBJECT="${MIN_REGIONS_PER_SUBJECT:-5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MISSING_ALPHA="${MISSING_ALPHA:-1.0}"
FIGSIZE="${FIGSIZE:-auto}"
AXES_BBOX="${AXES_BBOX:-auto}"
COLORBAR_BBOX="${COLORBAR_BBOX:-auto}"
BOX_ASPECT="${BOX_ASPECT:-compressed_data}"
BOX_ZOOM="${BOX_ZOOM:-1}"
DPI="${DPI:-400}"
TITLE_PAD="${TITLE_PAD:-0.0}"
AXIS_ASSIGNMENT="${AXIS_ASSIGNMENT:-gene,region,subject}"
SHOW_GRID="${SHOW_GRID:-false}"
ELEV="${ELEV:-20.0}"
AZIM="${AZIM:-24.0}"
SUBJECT_TICK_STEP="${SUBJECT_TICK_STEP:-3}"
GENE_TICK_STEP="${GENE_TICK_STEP:-5}"
REGION_TICK_STEP="${REGION_TICK_STEP:-1}"
TICK_LABEL_PAD="${TICK_LABEL_PAD:--3.0}"
SUBJECT_LABEL_MODE="${SUBJECT_LABEL_MODE:-full}"
GENE_LABEL_MODE="${GENE_LABEL_MODE:-full}"
SHOW_X_AXIS_LABEL="${SHOW_X_AXIS_LABEL:-true}"
SHOW_Y_AXIS_LABEL="${SHOW_Y_AXIS_LABEL:-none}"
SHOW_Z_AXIS_LABEL="${SHOW_Z_AXIS_LABEL:-none}"
SHOW_AXIS_LABELS="${SHOW_AXIS_LABELS:-false}"
SHOW_SUBJECT_TICKLABELS="${SHOW_SUBJECT_TICKLABELS:-true}"
SHOW_REGION_TICKLABELS="${SHOW_REGION_TICKLABELS:-true}"
SHOW_REGION_LABEL_COLORS="${SHOW_REGION_LABEL_COLORS:-false}"
SHOW_GENE_TICKLABELS="${SHOW_GENE_TICKLABELS:-false}"
SHOW_AXIS_LINES="${SHOW_AXIS_LINES:-true}"
SHOW_TICK_LINES="${SHOW_TICK_LINES:-true}"
SHOW_LEGEND="${SHOW_LEGEND:-false}"
OUT_PATH="${OUT_PATH:-out/tensor_renders/gtex_gxp_tensor_repr.png}"

cd "${REPO_ROOT}"
mkdir -p out/slurm "$(dirname "${OUT_PATH}")"

echo "[gtex-tensor-repr] repo=${REPO_ROOT} out=${OUT_PATH}"

SINGULARITY_CMD="source /ext3/env.sh && python3 scripts/render_gtex_gxp_tensor.py \
  --csv-path \"${CSV_PATH}\" \
  --active-gene-list \"${ACTIVE_GENE_LIST}\" \
  --region-ordering \"${REGION_ORDERING}\" \
  --matching-policy \"${MATCHING_POLICY}\" \
  --matching-policy-hemi-mode \"${MATCHING_POLICY_HEMI_MODE}\" \
  --collapse-cerebellum \"${COLLAPSE_CEREBELLUM}\" \
  --gtex-rep-mode \"${GTEX_REP_MODE}\" \
  --gtex-hemi-mode \"${GTEX_HEMI_MODE}\" \
  --n-subjects \"${N_SUBJECTS}\" \
  --n-genes \"${N_GENES}\" \
  --n-regions \"${N_REGIONS}\" \
  --min-regions-per-subject \"${MIN_REGIONS_PER_SUBJECT}\" \
  --random-seed \"${RANDOM_SEED}\" \
  --missing-alpha \"${MISSING_ALPHA}\" \
  --figsize \"${FIGSIZE}\" \
  --axes-bbox \"${AXES_BBOX}\" \
  --colorbar-bbox \"${COLORBAR_BBOX}\" \
  --box-aspect \"${BOX_ASPECT}\" \
  --box-zoom \"${BOX_ZOOM}\" \
  --dpi \"${DPI}\" \
  --title-pad \"${TITLE_PAD}\" \
  --axis-assignment \"${AXIS_ASSIGNMENT}\" \
  --show-grid \"${SHOW_GRID}\" \
  --elev \"${ELEV}\" \
  --azim \"${AZIM}\" \
  --subject-tick-step \"${SUBJECT_TICK_STEP}\" \
  --gene-tick-step \"${GENE_TICK_STEP}\" \
  --region-tick-step \"${REGION_TICK_STEP}\" \
  --tick-label-pad \"${TICK_LABEL_PAD}\" \
  --subject-label-mode \"${SUBJECT_LABEL_MODE}\" \
  --gene-label-mode \"${GENE_LABEL_MODE}\" \
  --show-x-axis-label \"${SHOW_X_AXIS_LABEL}\" \
  --show-y-axis-label \"${SHOW_Y_AXIS_LABEL}\" \
  --show-z-axis-label \"${SHOW_Z_AXIS_LABEL}\" \
  --show-axis-labels \"${SHOW_AXIS_LABELS}\" \
  --show-subject-ticklabels \"${SHOW_SUBJECT_TICKLABELS}\" \
  --show-region-ticklabels \"${SHOW_REGION_TICKLABELS}\" \
  --show-region-label-colors \"${SHOW_REGION_LABEL_COLORS}\" \
  --show-gene-ticklabels \"${SHOW_GENE_TICKLABELS}\" \
  --show-axis-lines \"${SHOW_AXIS_LINES}\" \
  --show-tick-lines \"${SHOW_TICK_LINES}\" \
  --show-legend \"${SHOW_LEGEND}\" \
  --out-path \"${OUT_PATH}\""

singularity exec --fakeroot \
  --overlay "/scratch/$USER/envs/vformer_env/overlay-25GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "${SINGULARITY_CMD}"

echo "[gtex-tensor-repr] done: ${OUT_PATH}"
