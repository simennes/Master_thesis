#!/bin/bash

# Merge once all array jobs are finished:
#   python -m scripts.run_avggrm_diversity_greedy --mode merge --config config/avggrm_diversity_greedy_config.json
#
# To force the model without editing the config:
#   sbatch --export=ALL,MODEL_TYPE=bpcrr src/SLURM/avggrm_diversity_greedy.slurm

#SBATCH --account=share-ie-imf
#SBATCH --job-name=avggrm_div
#SBATCH --time=1-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-19
#SBATCH --output=logs/avggrm_div_%A_%a.log
#SBATCH --error=logs/avggrm_div_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

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

CONFIG_PATH=${CONFIG_PATH:-config/avggrm_diversity_greedy_config.json}
DEFAULT_NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-20}
NUM_SHARDS=${SWEEP_NUM_SHARDS:-${DEFAULT_NUM_SHARDS}}

echo "Config path: ${CONFIG_PATH}"
echo "Selected shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Model override: ${MODEL_TYPE:-<config>}"

ENV_PREFIX=$(python -c "import sys; print(sys.prefix)")
echo "Resolved env prefix: ${ENV_PREFIX}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONNOUSERSITE=1

# Required when running BPCRR-INLA. Harmless for Ridge.
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

MODEL_ARGS=()
if [ -n "${MODEL_TYPE:-}" ]; then
  MODEL_ARGS=(--model_type "${MODEL_TYPE}")
fi

python -m scripts.run_avggrm_diversity_greedy \
  --mode worker \
  --config "${CONFIG_PATH}" \
  "${MODEL_ARGS[@]}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${SHARD_INDEX}"
