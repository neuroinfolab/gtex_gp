#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --job-name=loro_subj_test
#SBATCH --output=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/loro_subj_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/loro_subj_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp}"
SUBJECT_ID="${SUBJECT_ID:-GTEX-14ASI}"
MODEL_NAME="${MODEL_NAME:-all}"
GENE_SCOPE="${GENE_SCOPE:-hvg}"
HVG_PATH="${HVG_PATH:-data/raw/gene_lists/ahba_100hvg.txt}"
USE_CACHE="${USE_CACHE:-true}"
DROP_MACRO_SYSTEM_COVARIATE="${DROP_MACRO_SYSTEM_COVARIATE:-false}"
OUT_ROOT="${OUT_ROOT:-out/loro_subject_cache_smoke}"
DLAM_STRATEGY="${DLAM_STRATEGY:-baseline}"
DLAM_SPATIAL_METHOD="${DLAM_SPATIAL_METHOD:-rbf}"
LATENT_DIM="${LATENT_DIM:-3}"
DYNAMIC_RANK="${DYNAMIC_RANK:-false}"
PLAM_MAX_RANK="${PLAM_MAX_RANK:-10}"
ATLAS_AGG="${ATLAS_AGG:-mean}"
GTEX_REP_MODE="${GTEX_REP_MODE:-centroid}"
GTEX_HEMI_MODE="${GTEX_HEMI_MODE:-mirror_left}"
# Parcel assignment policy:
# - centroids: keep nearest-centroid coordinate matching for all GTEx regions.
# - centroids_and_volumes: start from centroids, then derive cortical BA9/BA24/BA110
#   matches from BA-to-Schaefer voxel overlap and map cerebellar labels explicitly.
MATCHING_POLICY="${MATCHING_POLICY:-centroids_and_volumes}"
MATCHING_POLICY_HEMI_MODE="${MATCHING_POLICY_HEMI_MODE:-default}"
COLLAPSE_CEREBELLUM="${COLLAPSE_CEREBELLUM:-false}"

cd "${REPO_ROOT}"
mkdir -p out/slurm

echo "[single-subject] repo=${REPO_ROOT} subject=${SUBJECT_ID} model=${MODEL_NAME} scope=${GENE_SCOPE} dlam_strategy=${DLAM_STRATEGY} dlam_spatial_method=${DLAM_SPATIAL_METHOD} drop_macro_system_covariate=${DROP_MACRO_SYSTEM_COVARIATE} matching_policy=${MATCHING_POLICY} matching_policy_hemi_mode=${MATCHING_POLICY_HEMI_MODE} collapse_cerebellum=${COLLAPSE_CEREBELLUM} out_root=${OUT_ROOT}"

SINGULARITY_CMD="source /ext3/env.sh && python3 scripts/run_loro_cache_batch.py \
  --subject \"${SUBJECT_ID}\" \
  --model \"${MODEL_NAME}\" \
  --gene-scope \"${GENE_SCOPE}\" \
  --hvg-path \"${HVG_PATH}\" \
  --out-root \"${OUT_ROOT}\" \
  --use-cache \"${USE_CACHE}\" \
  --dlam-strategy \"${DLAM_STRATEGY}\" \
  --dlam-spatial-method \"${DLAM_SPATIAL_METHOD}\" \
  --drop-macro-system-covariate \"${DROP_MACRO_SYSTEM_COVARIATE}\" \
  --latent-dim \"${LATENT_DIM}\" \
  --dynamic-rank \"${DYNAMIC_RANK}\" \
  --plam-latent-dim-max \"${PLAM_MAX_RANK}\" \
  --atlas-agg \"${ATLAS_AGG}\" \
  --gtex-rep-mode \"${GTEX_REP_MODE}\" \
  --gtex-hemi-mode \"${GTEX_HEMI_MODE}\" \
  --matching-policy \"${MATCHING_POLICY}\" \
  --matching-policy-hemi-mode \"${MATCHING_POLICY_HEMI_MODE}\" \
  --collapse-cerebellum \"${COLLAPSE_CEREBELLUM}\""

singularity exec --fakeroot \
  --overlay "/scratch/$USER/envs/vformer_env/overlay-25GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "${SINGULARITY_CMD}"

echo "[single-subject] done"
