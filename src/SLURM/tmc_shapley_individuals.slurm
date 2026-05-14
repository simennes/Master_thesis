#!/bin/bash
#
# Default: 5 array tasks for the 5 configured target islands.
# For 10 shards, submit with:
#   sbatch --array=0-9 src/SLURM/tmc_shapley_individuals.slurm
#
# Merge target-level uncertainty files after all array tasks finish:
#   python -m scripts.run_tmc_shapley_individuals --mode merge --config config/tmc_shapley_individuals.json

#SBATCH --account=share-ie-imf
#SBATCH --job-name=shap_ind
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-9
#SBATCH --output=logs/tmc_shapley_individuals_%A_%a.out
#SBATCH --error=logs/tmc_shapley_individuals_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
DEFAULT_NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
NUM_SHARDS=${TMC_SHAPLEY_NUM_SHARDS:-${SWEEP_NUM_SHARDS:-${DEFAULT_NUM_SHARDS}}}
CONFIG_PATH=${CONFIG_PATH:-config/tmc_shapley_individuals.json}

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-<not-array>}"
echo "Array task count: ${SLURM_ARRAY_TASK_COUNT:-<unknown>}"
echo "Selected shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Config path: ${CONFIG_PATH}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m scripts.run_tmc_shapley_individuals \
  --mode worker \
  --config "${CONFIG_PATH}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${SHARD_INDEX}"
