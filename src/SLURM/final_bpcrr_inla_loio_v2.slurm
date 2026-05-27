#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=bpcrr_v2
#SBATCH --time=2-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/bpcrr_v2_%j.log
#SBATCH --error=logs/bpcrr_v2_%j.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
SHARD_INDEX=0

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Running as a single non-array job"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

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

CONFIG_PATH=${CONFIG_PATH:-config/final_bpcrr_inla_loio_v2_config.json}
NUM_SHARDS=1

conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_select \
  --mode preflight \
  --config "${CONFIG_PATH}"

conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_select \
  --mode worker \
  --config "${CONFIG_PATH}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${SHARD_INDEX}"

# With NUM_SHARDS=1, outputs are written directly to the final trait folders.
# No merge job is needed.
