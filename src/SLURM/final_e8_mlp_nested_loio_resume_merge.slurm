#!/bin/bash

# Resume selected E8 MLP outer folds, then merge shard results back into the
# canonical output file for the config.
#
# Example:
#   sbatch --export=ALL,CONFIG_PATH=config/e8/final_e8_mlp_uniform_thr_wing_config.json,SELECTED_SPLITS=12_13_14_15 src/SLURM/final_e8_mlp_nested_loio_resume_merge.slurm

#SBATCH --account=share-ie-idi
#SBATCH --job-name=e8_mlp_resume
#SBATCH --time=2-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/e8_mlp_resume_%j.out
#SBATCH --error=logs/e8_mlp_resume_%j.err
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

CONFIG_PATH=${CONFIG_PATH:-config/e8/final_e8_mlp_uniform_thr_wing_config.json}
RAW_SELECTED_SPLITS=${SELECTED_SPLITS:-}
SELECTED_SPLITS=${RAW_SELECTED_SPLITS//_/,}

if [[ -z "${SELECTED_SPLITS}" ]]; then
  echo "ERROR: SELECTED_SPLITS must be set, e.g. SELECTED_SPLITS=12_13_14_15" >&2
  exit 2
fi

echo "Config path: ${CONFIG_PATH}"
echo "Raw selected splits: ${RAW_SELECTED_SPLITS}"
echo "Selected splits: ${SELECTED_SPLITS}"

# Preserve the completed folds from the canonical result as a shard so the
# final merge can rebuild the same canonical result file after the resume run.
python - "${CONFIG_PATH}" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

paths = cfg["base_train"]["paths"]
out_dir = paths.get("output_dir", "outputs/nested_cv")
out_name = paths.get("output_name", "nested_cv_importance_weighted_mlp")
main_path = os.path.join(out_dir, f"{out_name}_results.json")

if not os.path.exists(main_path):
    print(f"No existing canonical result found at {main_path}; nothing to shard yet.")
    raise SystemExit(0)

with open(main_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

folds = sorted({int(item["fold"]) for item in payload.get("per_fold_metrics", [])})
if not folds:
    print(f"Existing canonical result at {main_path} has no completed folds.")
    raise SystemExit(0)

suffix = "_".join(str(fold) for fold in folds)
shard_path = os.path.join(out_dir, f"{out_name}_splits_{suffix}_results.json")
with open(shard_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Preserved completed folds {folds} to {shard_path}")
PY

python -m src.nested_cv_importance_weighted_mlp \
  --config "${CONFIG_PATH}" \
  --selected_splits "${SELECTED_SPLITS}"

python -m src.nested_cv_importance_weighted_mlp \
  --mode merge \
  --config "${CONFIG_PATH}"
