#!/bin/bash

# Post-hoc re-selection of E6 importance-weighted PC-ridge configs.
# Re-selects the winning trial per outer fold under a grid of selection rules
# (top_m inner islands x lambda in `mean - lambda*SE`), refits each chosen
# config on the outer training set, evaluates on the held-out island, and writes
# a long CSV + boxplot figures.
#
# Requires E6 importance runs produced AFTER per-island logging was added
# (re-run final_e6_pc_ridge_nested_loio.slurm for the importance configs first).
#
# Submit:
#   sbatch src/SLURM/posthoc_reselect_importance.slurm
# Override the rule grid:
#   sbatch --export=ALL,TOP_M=2,4,6,8,all,LAMBDAS=0,0.25,0.5,1,2 src/SLURM/posthoc_reselect_importance.slurm

#SBATCH --account=share-ie-idi
#SBATCH --job-name=e6_reselect
#SBATCH --time=04:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e6_reselect_%j.out
#SBATCH --error=logs/e6_reselect_%j.err
#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"
mkdir -p logs

module purge
module load Anaconda3/2024.02-1
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TOP_M=${TOP_M:-2,4,6,8,all}
LAMBDAS=${LAMBDAS:-0,0.5,1,2}

python -m scripts.posthoc_reselect_importance \
  --top-m "${TOP_M}" \
  --lambda "${LAMBDAS}" \
  --plot
