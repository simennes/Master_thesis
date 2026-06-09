#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=within_pevmean
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-29
#SBATCH --output=logs/within_pevmean_%A_%a.log
#SBATCH --error=logs/within_pevmean_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

# Default array is 3 traits x 10 shards per trait.
# Run:
#   sbatch src/SLURM/within_pop_pevmean_subsets.slurm
#
# Merge after all shards finish:
#   python -m scripts.run_within_pop_pevmean_subsets --config config/within_pop/subset_pevmean_config.json --mode merge

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

TRAITS=(body_mass thr_tarsus thr_wing)
NUM_SHARDS_PER_TRAIT=${NUM_SHARDS_PER_TRAIT:-10}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TRAIT_INDEX=$((TASK_ID / NUM_SHARDS_PER_TRAIT))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS_PER_TRAIT))

if (( TRAIT_INDEX >= ${#TRAITS[@]} )); then
  echo "Array task ${TASK_ID} maps to invalid trait index ${TRAIT_INDEX}" >&2
  exit 1
fi

TRAIT=${TRAIT:-${TRAITS[${TRAIT_INDEX}]}}

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Array task ID: ${TASK_ID}"
echo "Trait: ${TRAIT}"
echo "Trait index: ${TRAIT_INDEX}"
echo "Shard: ${SHARD_INDEX}/${NUM_SHARDS_PER_TRAIT}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1

echo "Conda binary: $(command -v conda)"
conda run -n master_thesis --no-capture-output python -c "import sys; print('Python in env:', sys.executable)"

# PEVmean uses joblib workers; keep BLAS single-threaded inside each worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SWEEP_NUM_SHARDS=${NUM_SHARDS_PER_TRAIT}

CONFIG_PATH=${CONFIG_PATH:-config/within_pop/subset_pevmean_config.json}
echo "Config: ${CONFIG_PATH}"

conda run -n master_thesis --no-capture-output python -m scripts.run_within_pop_pevmean_subsets \
  --config "${CONFIG_PATH}" \
  --trait "${TRAIT}" \
  --shard_index "${SHARD_INDEX}" \
  --num_shards "${NUM_SHARDS_PER_TRAIT}"
