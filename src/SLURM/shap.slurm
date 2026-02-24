#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=shap
#SBATCH --time=1-12:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:2              # Setting the number of GPUs to 1
#SBATCH --constraint="a100"        # Requesting A100 GPUs
#SBATCH --mem=32G                 # Asking for 32GB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # Request 8 CPU cores per task
#SBATCH --ntasks-per-node=1                  # Number of tasks (keep 1 if training a single model)
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err

#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1
conda activate master_thesis
python -m scripts.run_tmc_shapley_islands --config config/tmc_shapley_islands.json