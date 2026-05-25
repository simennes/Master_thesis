#!/bin/sh
#SBATCH --account=share-ie-imf
#SBATCH --job-name=e1_pc_ridge
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e1_pc_ridge_%j.log
#SBATCH --error=logs/e1_pc_ridge_%j.err
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Submit one job per trait, e.g.:
#   sbatch --export=CONFIG_PATH=config/final_e1_pc_ridge_nested_loio_body_mass_config.json src/SLURM/final_e1_pc_ridge_nested_loio.slurm
#   sbatch --export=CONFIG_PATH=config/final_e1_pc_ridge_nested_loio_thr_tarsus_config.json src/SLURM/final_e1_pc_ridge_nested_loio.slurm
#   sbatch --export=CONFIG_PATH=config/final_e1_pc_ridge_nested_loio_thr_wing_config.json src/SLURM/final_e1_pc_ridge_nested_loio.slurm
# Or run all traits sequentially in one job using the combined config:
#   sbatch --export=CONFIG_PATH=config/final_e1_pc_ridge_nested_loio_config.json src/SLURM/final_e1_pc_ridge_nested_loio.slurm

CONFIG_PATH=${CONFIG_PATH:-config/final_e1_pc_ridge_nested_loio_config.json}
echo "Config: ${CONFIG_PATH}"

python -m scripts.run_pc_ridge_nested_loio \
  --config "${CONFIG_PATH}"
