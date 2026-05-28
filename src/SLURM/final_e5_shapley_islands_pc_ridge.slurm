#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=e5_shapley_islands
#SBATCH --time=12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=logs/e5_shapley_islands_%A_%a.log
#SBATCH --error=logs/e5_shapley_islands_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

# One array task per trait (3 shards, run in parallel):
#   sbatch src/SLURM/final_e5_shapley_islands_pc_ridge.slurm

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

# Each utility eval is a single PC-ridge fit; let BLAS use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG_PATH=config/final_e5_shapley_islands_pc_ridge_config.json

TRAITS=(body_mass thr_tarsus thr_wing)
TRAIT=${TRAITS[$SLURM_ARRAY_TASK_ID]}
echo "Config: ${CONFIG_PATH}"
echo "Trait (shard ${SLURM_ARRAY_TASK_ID}): ${TRAIT}"

python -m scripts.run_e5_shapley_islands_pc_ridge \
  --config "${CONFIG_PATH}" \
  --trait "${TRAIT}"
