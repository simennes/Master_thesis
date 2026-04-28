#!/bin/bash

# Finalize each outer fold from saved best-trial checkpoints:
#   sbatch src/SLURM/nested_cv_avggrm_weighted_finalize_from_best.slurm
#
# Merge once all array jobs are finished:
#   python -m src.nested_cv_avggrm_weighted_unified --mode merge --config config/nested_cv_avggrm_weighted_bpcrr_config.json

#SBATCH --account=share-ie-imf
#SBATCH --job-name=ncv_best_final
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-16%16
#SBATCH --output=logs/ncv_best_final_%A_%a.out
#SBATCH --error=logs/ncv_best_final_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-<not-array>}"
echo "Array task count: ${SLURM_ARRAY_TASK_COUNT:-<unknown>}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

CONFIG_PATH=${CONFIG_PATH:-config/nested_cv_avggrm_weighted_bpcrr_config.json}
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID:-1}
export FINALIZE_SPLIT_INDEX="${SPLIT_INDEX}"

echo "Config path: ${CONFIG_PATH}"
echo "Selected outer split: ${SPLIT_INDEX}"

ENV_PREFIX=$(python -c "import sys; print(sys.prefix)")
echo "Resolved env prefix: ${ENV_PREFIX}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Required for BPCRR-INLA.
export INLA_NUM_THREADS=${SLURM_CPUS_PER_TASK}:1
export R_HOME="${ENV_PREFIX}/lib/R"
export R_LIBS_USER="${ENV_PREFIX}/lib/R/library"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib"
fi
export R_PROFILE_USER=""
export R_ENVIRON_USER=""

python -m src.nested_cv_avggrm_weighted_finalize_from_best --config "${CONFIG_PATH}" --selected_splits "${SPLIT_INDEX}"
