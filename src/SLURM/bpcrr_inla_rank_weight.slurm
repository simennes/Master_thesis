#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=bpcrr_rw
#SBATCH --time=0-08:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-4
#SBATCH --output=logs/bpcrr_rw_%A_%a.log
#SBATCH --error=logs/bpcrr_rw_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-<not-array>}"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1

ENV_PREFIX=$(conda run -n master_thesis --no-capture-output python -c "import sys; print(sys.prefix)")
echo "Resolved env prefix: ${ENV_PREFIX}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export INLA_NUM_THREADS=1:${SLURM_CPUS_PER_TASK}

export PYTHONNOUSERSITE=1
export R_HOME="${ENV_PREFIX}/lib/R"
export R_LIBS_USER="${ENV_PREFIX}/lib/R/library"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib"
fi

export R_PROFILE_USER=""
export R_ENVIRON_USER=""

CONFIG_PATH="config/bpcrr_inla_rank_weight_config.json"
DEFAULT_NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-5}
NUM_SHARDS=${SWEEP_NUM_SHARDS:-${DEFAULT_NUM_SHARDS}}

conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_weight \
  --mode worker \
  --config "${CONFIG_PATH}" \
  --num_shards ${NUM_SHARDS} \
  --shard_index ${SHARD_INDEX}

# Merge once all array jobs are finished:
# python -m scripts.run_bpcrr_inla_rank_weight --mode merge --config "${CONFIG_PATH}"
