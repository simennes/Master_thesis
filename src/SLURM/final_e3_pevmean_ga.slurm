#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=e3_pevmean_ga
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e3_pevmean_ga_%j.log
#SBATCH --error=logs/e3_pevmean_ga_%j.err
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
source /cluster/apps/eb/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate master_thesis

# Marker-form PEVmean uses small (k+1) x (k+1) Cholesky per subset, so BLAS
# threads in each worker matter less than process-level parallelism. Run
# n_jobs workers with 1 BLAS thread each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Submit one job per trait in parallel:
#   sbatch --export=ALL,CONFIG_PATH=config/final_e3_pevmean_ga_body_mass_config.json src/SLURM/final_e3_pevmean_ga.slurm
#   sbatch --export=ALL,CONFIG_PATH=config/final_e3_pevmean_ga_thr_tarsus_config.json src/SLURM/final_e3_pevmean_ga.slurm
#   sbatch --export=ALL,CONFIG_PATH=config/final_e3_pevmean_ga_thr_wing_config.json   src/SLURM/final_e3_pevmean_ga.slurm

CONFIG_PATH=${CONFIG_PATH:-config/final_e3_pevmean_ga_body_mass_config.json}
echo "Config: ${CONFIG_PATH}"

python -m scripts.run_pevmean_ga \
  --mode worker \
  --config "${CONFIG_PATH}"
