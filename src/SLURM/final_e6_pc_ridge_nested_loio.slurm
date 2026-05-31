#!/bin/bash

# E6: Nested LOIO for PC ridge with AvgGRM or importance weighting.
# Runs all 15 outer folds sequentially in a single job. The runner persists
# results incrementally after each completed outer fold, so a crash later
# still leaves the prior folds' results on disk.
#
# Pick a config and submit (one job per trait-weighting combination):
#   sbatch --export=ALL,CONFIG_PATH=config/e6/final_e6_pc_ridge_avggrm_body_mass_config.json    src/SLURM/final_e6_pc_ridge_nested_loio.slurm
#   sbatch --export=ALL,CONFIG_PATH=config/e6/final_e6_pc_ridge_importance_thr_wing_config.json src/SLURM/final_e6_pc_ridge_nested_loio.slurm
#
# Or all 6 at once:
#   for cfg in config/e6/final_e6_pc_ridge_*_config.json; do
#     sbatch --export=ALL,CONFIG_PATH=$cfg src/SLURM/final_e6_pc_ridge_nested_loio.slurm
#   done

#SBATCH --account=share-ie-idi
#SBATCH --job-name=e6_pc_ridge
#SBATCH --time=1-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e6_pc_ridge_%j.out
#SBATCH --error=logs/e6_pc_ridge_%j.err
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

CONFIG_PATH=${CONFIG_PATH:-config/e6/final_e6_pc_ridge_avggrm_body_mass_config.json}

echo "Config path: ${CONFIG_PATH}"

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

# No --selected_splits: runner iterates all outer folds and saves after each.
python -m "${MODULE}" --config "${CONFIG_PATH}"
