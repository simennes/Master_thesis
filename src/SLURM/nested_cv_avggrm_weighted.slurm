#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=ncv_avggrm
#SBATCH --time=2-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:2
#SBATCH --constraint="v100|a100"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/ncv_avggrm_%j.out
#SBATCH --error=logs/ncv_avggrm_%j.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

CONFIG_PATH="config/nested_cv_avggrm_weighted_config.json"

python -m src.nested_cv_avggrm_weighted --config "${CONFIG_PATH}"
