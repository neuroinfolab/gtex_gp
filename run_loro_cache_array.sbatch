#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --job-name=loro_cache
#SBATCH --output=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/loro_array_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp/out/slurm/loro_array_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu
#SBATCH --array=1-318%64

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp}"
MODEL_NAME="${MODEL_NAME:-all}"
GENE_SCOPE="${GENE_SCOPE:-hvg}"
HVG_PATH="${HVG_PATH:-data/raw/gene_lists/ahba_100hvg.txt}"
USE_CACHE="${USE_CACHE:-true}"
OUT_ROOT="${OUT_ROOT:-out/loro_subject_cache}"
LATENT_DIM="${LATENT_DIM:-3}"
DYNAMIC_RANK="${DYNAMIC_RANK:-false}"
PLAM_MAX_RANK="${PLAM_MAX_RANK:-10}"
ATLAS_AGG="${ATLAS_AGG:-mean}"
GTEX_REP_MODE="${GTEX_REP_MODE:-centroid}"
GTEX_HEMI_MODE="${GTEX_HEMI_MODE:-mirror_left}"
# Parcel assignment policy:
# - centroids: keep nearest-centroid coordinate matching for all GTEx regions.
# - centroids_and_volumes: start from centroids, then derive cortical BA9/BA24/BA110
#   matches from BA-to-Schaefer voxel overlap and map cerebellar samples to Cerebellar_Region7.
MATCHING_POLICY="${MATCHING_POLICY:-centroids}"

cd "${REPO_ROOT}"
mkdir -p out/slurm

echo "[array] repo=${REPO_ROOT} task=${SLURM_ARRAY_TASK_ID} model=${MODEL_NAME} scope=${GENE_SCOPE} matching_policy=${MATCHING_POLICY} out_root=${OUT_ROOT}"

SINGULARITY_CMD="source /ext3/env.sh && python3 scripts/run_loro_cache_batch.py \
  --from-sbatch-array \
  --model \"${MODEL_NAME}\" \
  --gene-scope \"${GENE_SCOPE}\" \
  --hvg-path \"${HVG_PATH}\" \
  --out-root \"${OUT_ROOT}\" \
  --use-cache \"${USE_CACHE}\" \
  --latent-dim \"${LATENT_DIM}\" \
  --dynamic-rank \"${DYNAMIC_RANK}\" \
  --plam-latent-dim-max \"${PLAM_MAX_RANK}\" \
  --atlas-agg \"${ATLAS_AGG}\" \
  --gtex-rep-mode \"${GTEX_REP_MODE}\" \
  --gtex-hemi-mode \"${GTEX_HEMI_MODE}\" \
  --matching-policy \"${MATCHING_POLICY}\""

singularity exec --fakeroot \
  --overlay "/scratch/$USER/envs/vformer_env/overlay-25GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "${SINGULARITY_CMD}"

echo "[array] done task=${SLURM_ARRAY_TASK_ID}"
