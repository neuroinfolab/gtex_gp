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
#SBATCH --array=1-385%128
# 385 = full GTEx subject pool (absolute upper bound). Tasks whose index
# exceeds the policy's eligible-subject count exit cleanly via
# scripts/run_loro_cache_batch.py (logged as `[skip-out-of-range] ...`).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/asr655/neuroinformatics/Seq2GeneEx/gtex_gp}"
MODEL_NAME="${MODEL_NAME:-all}"
GENE_SCOPE="${GENE_SCOPE:-hvg}"
HVG_PATH="${HVG_PATH:-data/metadata/gene_lists/ahba_100hvg.txt}"
USE_CACHE="${USE_CACHE:-true}"
COMBAT_USE_COVARIATES="${COMBAT_USE_COVARIATES:-true}"
DROP_MACRO_SYSTEM_COVARIATE="${DROP_MACRO_SYSTEM_COVARIATE:-false}"
# ComBat covariate batch-handling mode: mixed | pooled | per_batch
# 'mixed' (default): age+sex per-batch coefficients, macro_system pooled.
COV_BATCH_MODE="${COV_BATCH_MODE:-mixed}"
OUT_ROOT="${OUT_ROOT:-out/loro_subject_cache_c}"  # convention: _c=centroids, _cv=centroids_and_volumes
DLAM_STRATEGY="${DLAM_STRATEGY:-baseline}" # baseline, reference_residual, constrained_anchor, pls_linear_projection, distance_shrink, gp_uncertainty, moe_basis
DLAM_SPATIAL_METHOD="${DLAM_SPATIAL_METHOD:-rbf}"
ANCHOR_DISTANCE_SHRINK="${ANCHOR_DISTANCE_SHRINK:-false}"
ANCHOR_DISTANCE_D0="${ANCHOR_DISTANCE_D0:-30.0}"
ANCHOR_DISTANCE_TAU="${ANCHOR_DISTANCE_TAU:-8.0}"
GP_OPTIMIZE="${GP_OPTIMIZE:-true}"
GP_N_RESTARTS="${GP_N_RESTARTS:-0}"
# t_prior_residual interpolator knobs (only used when DLAM_STRATEGY=t_prior_residual).
# The atlas-scale floor is intentionally NOT exposed here: it is a code-level
# default (True). Flip it in code for ablations, not via sbatch.
T_PRIOR_INTERP="${T_PRIOR_INTERP:-imq}"               # imq | gaussian | tps | gp
T_PRIOR_LENGTH_SCALE="${T_PRIOR_LENGTH_SCALE:-25.0}"
T_PRIOR_GP_NOISE="${T_PRIOR_GP_NOISE:-0.01}"
T_PRIOR_GP_OPTIMIZE="${T_PRIOR_GP_OPTIMIZE:-false}"
# More aggressive constrained-anchor shrinkage: ANCHOR_DISTANCE_D0=25.0 ANCHOR_DISTANCE_TAU=6.0
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

echo "[array] repo=${REPO_ROOT} task=${SLURM_ARRAY_TASK_ID} model=${MODEL_NAME} scope=${GENE_SCOPE} dlam_strategy=${DLAM_STRATEGY} dlam_spatial_method=${DLAM_SPATIAL_METHOD} anchor_distance_shrink=${ANCHOR_DISTANCE_SHRINK} anchor_distance_d0=${ANCHOR_DISTANCE_D0} anchor_distance_tau=${ANCHOR_DISTANCE_TAU} gp_optimize=${GP_OPTIMIZE} gp_n_restarts=${GP_N_RESTARTS} t_prior_interp=${T_PRIOR_INTERP} t_prior_length_scale=${T_PRIOR_LENGTH_SCALE} t_prior_gp_noise=${T_PRIOR_GP_NOISE} t_prior_gp_optimize=${T_PRIOR_GP_OPTIMIZE} combat_use_covariates=${COMBAT_USE_COVARIATES} drop_macro_system_covariate=${DROP_MACRO_SYSTEM_COVARIATE} matching_policy=${MATCHING_POLICY} matching_policy_hemi_mode=${MATCHING_POLICY_HEMI_MODE} collapse_cerebellum=${COLLAPSE_CEREBELLUM} out_root=${OUT_ROOT}"

SINGULARITY_CMD="source /ext3/env.sh && python3 scripts/run_loro_cache_batch.py \
  --from-sbatch-array \
  --model \"${MODEL_NAME}\" \
  --gene-scope \"${GENE_SCOPE}\" \
  --hvg-path \"${HVG_PATH}\" \
  --out-root \"${OUT_ROOT}\" \
  --use-cache \"${USE_CACHE}\" \
  --dlam-strategy \"${DLAM_STRATEGY}\" \
  --dlam-spatial-method \"${DLAM_SPATIAL_METHOD}\" \
  --anchor-distance-shrink \"${ANCHOR_DISTANCE_SHRINK}\" \
  --anchor-distance-d0 \"${ANCHOR_DISTANCE_D0}\" \
  --anchor-distance-tau \"${ANCHOR_DISTANCE_TAU}\" \
  --gp-optimize \"${GP_OPTIMIZE}\" \
  --gp-n-restarts \"${GP_N_RESTARTS}\" \
  --t-prior-interp \"${T_PRIOR_INTERP}\" \
  --t-prior-length-scale \"${T_PRIOR_LENGTH_SCALE}\" \
  --t-prior-gp-noise \"${T_PRIOR_GP_NOISE}\" \
  --t-prior-gp-optimize \"${T_PRIOR_GP_OPTIMIZE}\" \
  --combat-use-covariates \"${COMBAT_USE_COVARIATES}\" \
  --drop-macro-system-covariate \"${DROP_MACRO_SYSTEM_COVARIATE}\" \
  --cov-batch-mode \"${COV_BATCH_MODE}\" \
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

echo "[array] done task=${SLURM_ARRAY_TASK_ID}"
