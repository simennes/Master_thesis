#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=ncv_ridge_iw
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-16
#SBATCH --output=logs/ncv_ridge_iw_%A_%a.out
#SBATCH --error=logs/ncv_ridge_iw_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-<not-array>}"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG_PATH="config/nested_cv_importance_weighted_ridge_config.json"
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}

python -m src.nested_cv_importance_weighted_ridge \
  --config "${CONFIG_PATH}" \
  --selected_splits "${SPLIT_INDEX}"

# Merge once all array jobs are finished:
# python -m src.nested_cv_importance_weighted_ridge --mode merge --config "${CONFIG_PATH}"
