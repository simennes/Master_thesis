#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=bpcrr_inla
#SBATCH --time=3-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/bpcrr_inla_%A_%a.log
#SBATCH --error=logs/bpcrr_inla_%A_%a.err
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

echo "Conda binary: $(command -v conda)"
conda run -n master_thesis --no-capture-output python -c "import sys; print('Python in env:', sys.executable)"

ENV_PREFIX=$(conda run -n master_thesis --no-capture-output python -c "import sys; print(sys.prefix)")
echo "Resolved env prefix: ${ENV_PREFIX}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export INLA_NUM_THREADS=1:${SLURM_CPUS_PER_TASK}

# Keep Python and R package resolution inside the conda env.
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

CONFIG_PATH="config/bpcrr_inla_config.json"
NUM_SHARDS=1

# Run a single-node preflight once from array task 0.
if [ "${SHARD_INDEX}" -eq 0 ]; then
  conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_select \
    --mode preflight \
    --config "${CONFIG_PATH}"
fi

conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_select \
  --mode merge \
  --config "${CONFIG_PATH}" \
  --num_shards ${NUM_SHARDS} \
  --shard_index ${SHARD_INDEX}

# Merge once all array jobs are finished:
# python -m scripts.run_bpcrr_inla_rank_select --mode merge --config "${CONFIG_PATH}"
