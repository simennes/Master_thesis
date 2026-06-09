#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=within_pc_ridge
#SBATCH --time=0-08:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=logs/within_pc_ridge_%A_%a.log
#SBATCH --error=logs/within_pc_ridge_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

# Run all traits in parallel:
#   sbatch src/SLURM/within_pop_pc_ridge_10fold.slurm
#
# Merge the per-trait outputs after the array job finishes:
#   python -m scripts.run_pc_ridge_nested_kfold --config config/within_pop/pc_ridge_nested_10fold_config.json --mode merge

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

TRAITS=(body_mass thr_tarsus thr_wing)
TRAIT=${TRAIT:-${TRAITS[${SLURM_ARRAY_TASK_ID:-0}]}}

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Trait: ${TRAIT}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1

echo "Conda binary: $(command -v conda)"
conda run -n master_thesis --no-capture-output python -c "import sys; print('Python in env:', sys.executable)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG_PATH=${CONFIG_PATH:-config/within_pop/pc_ridge_nested_10fold_config.json}
echo "Config: ${CONFIG_PATH}"

conda run -n master_thesis --no-capture-output python -m scripts.run_pc_ridge_nested_kfold \
  --config "${CONFIG_PATH}" \
  --trait "${TRAIT}"
