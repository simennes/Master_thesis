#!/bin/bash

# E8: Nested LOIO for MLP (or PC-MLP) with weighting_mode = uniform | avggrm | importance.
#
# Single SLURM job per CONFIG_PATH; the runner uses parallel_outer_splits internally
# to distribute outer folds across visible GPUs.
#
#   sbatch --export=ALL,CONFIG_PATH=config/e8/final_e8_mlp_uniform_body_mass_config.json     src/SLURM/final_e8_mlp_nested_loio.slurm
#   sbatch --export=ALL,CONFIG_PATH=config/e8/final_e8_pc_mlp_avggrm_thr_wing_config.json    src/SLURM/final_e8_mlp_nested_loio.slurm

#SBATCH --account=share-ie-imf
#SBATCH --job-name=e8_mlp
#SBATCH --time=2-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e8_mlp_%j.out
#SBATCH --error=logs/e8_mlp_%j.err
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
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONFIG_PATH=${CONFIG_PATH:-config/e8/final_e8_mlp_uniform_body_mass_config.json}
echo "Config path: ${CONFIG_PATH}"

python -m src.nested_cv_importance_weighted_mlp --config "${CONFIG_PATH}"
