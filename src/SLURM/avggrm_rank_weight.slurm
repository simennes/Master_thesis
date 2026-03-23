#!/bin/sh
#SBATCH --account=share-ie-imf
#SBATCH --job-name=avggrm_rw
#SBATCH --time=1-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-19
#SBATCH --output=logs/avggrm_rw_%A_%a.log
#SBATCH --error=logs/avggrm_rw_%A_%a.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
conda activate master_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG_PATH="config/avggrm_rank_weight_config.json"
NUM_SHARDS=20
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

python -m scripts.run_avggrm_rank_weight \
  --mode worker \
  --config "${CONFIG_PATH}" \
  --num_shards ${NUM_SHARDS} \
  --shard_index ${SHARD_INDEX}

# Merge once all array jobs are finished:
# python -m scripts.run_avggrm_rank_weight --mode merge --config "${CONFIG_PATH}"
