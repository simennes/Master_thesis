#!/bin/sh
#SBATCH --account=share-ie-imf
#SBATCH --job-name=shap_sweep
#SBATCH --time=4-00:00:00

#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/shap_sweep2.log
#SBATCH --error=logs/shap_sweep2.err

#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
conda activate master_thesis

python -m scripts.run_tmc_shapley_sweep --config config/tmc_shapley_sweep.json
