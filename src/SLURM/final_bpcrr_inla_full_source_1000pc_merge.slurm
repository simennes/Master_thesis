#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=bpcrr_full1000_merge
#SBATCH --time=0-01:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/bpcrr_full1000_merge_%j.log
#SBATCH --error=logs/bpcrr_full1000_merge_%j.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

echo "Running merge from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"

module purge
module load Anaconda3/2024.02-1

CONFIG_PATH=${CONFIG_PATH:-config/final_bpcrr_inla_full_source_1000pc_config.json}

conda run -n master_thesis --no-capture-output python -m scripts.run_bpcrr_inla_rank_select \
  --mode merge \
  --config "${CONFIG_PATH}"
