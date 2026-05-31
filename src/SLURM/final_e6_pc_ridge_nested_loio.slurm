#!/bin/bash

# E6: Nested LOIO for PC ridge with AvgGRM or importance weighting.
#
# Pick a config and submit:
#   sbatch --export=ALL,CONFIG_PATH=config/final_e6_pc_ridge_avggrm_body_mass_config.json     src/SLURM/final_e6_pc_ridge_nested_loio.slurm
#   sbatch --export=ALL,CONFIG_PATH=config/final_e6_pc_ridge_importance_thr_wing_config.json   src/SLURM/final_e6_pc_ridge_nested_loio.slurm
#
# When all array tasks finish, merge with:
#   python -m src.nested_cv_avggrm_weighted_ridge       --mode merge --config <CONFIG_PATH>   # avgGRM variants
#   python -m src.nested_cv_importance_weighted_ridge  --mode merge --config <CONFIG_PATH>    # importance variants

#SBATCH --account=share-ie-imf
#SBATCH --job-name=e6_pc_ridge
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-15
#SBATCH --output=logs/e6_pc_ridge_%A_%a.out
#SBATCH --error=logs/e6_pc_ridge_%A_%a.err
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

CONFIG_PATH=${CONFIG_PATH:-config/final_e6_pc_ridge_avggrm_body_mass_config.json}
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID:-1}

echo "Config path: ${CONFIG_PATH}"
echo "Selected outer split: ${SPLIT_INDEX}"

# Choose the runner module based on the filename: 'avggrm' vs 'importance'.
if [[ "${CONFIG_PATH}" == *avggrm* ]]; then
  MODULE="src.nested_cv_avggrm_weighted_ridge"
elif [[ "${CONFIG_PATH}" == *importance* ]]; then
  MODULE="src.nested_cv_importance_weighted_ridge"
else
  echo "ERROR: cannot infer weighting (avggrm|importance) from CONFIG_PATH" >&2
  exit 2
fi
echo "Runner module: ${MODULE}"

python -m "${MODULE}" \
  --config "${CONFIG_PATH}" \
  --selected_splits "${SPLIT_INDEX}"
