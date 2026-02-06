#!/bin/bash
#SBATCH --job-name=yb_{ee.identifier}
#SBATCH --output=yolobattle_train.out
#SBATCH --error=yolobattle_train.out
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
{slurm.sbatch}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
  module -q avail apptainer 2>/dev/null && module load apptainer || true
fi

PROFILE="{experiment.profile}"
VAL_FRAC="{experiment.val_frac}"
NUM_GPUS="{experiment.num_gpus}"
ULTRA_MODEL="{experiment.ultra_model}"
COLOR_PRESET="{experiment.color_preset}"
DATASET_ROOT="{system.dataset_root}"
EE_REPEAT="{experiment.repeat}"

TRAIN_ARGS=(--no-sweep)

if [[ -n "$VAL_FRAC" && "$VAL_FRAC" != "None" && "$VAL_FRAC" != "null" ]]; then
  TRAIN_ARGS+=(--val-frac "$VAL_FRAC")
fi

if [[ -n "$NUM_GPUS" && "$NUM_GPUS" != "None" && "$NUM_GPUS" != "null" ]]; then
  TRAIN_ARGS+=(--num-gpus "$NUM_GPUS")
fi

if [[ -n "$ULTRA_MODEL" && "$ULTRA_MODEL" != "None" && "$ULTRA_MODEL" != "null" ]]; then
  TRAIN_ARGS+=(--ultra-model "$ULTRA_MODEL")
fi

if [[ -n "$COLOR_PRESET" && "$COLOR_PRESET" != "None" && "$COLOR_PRESET" != "null" ]]; then
  TRAIN_ARGS+=(--color-preset "$COLOR_PRESET")
fi

if [[ -n "$DATASET_ROOT" && "$DATASET_ROOT" != "None" && "$DATASET_ROOT" != "null" ]]; then
  TRAIN_ARGS+=(--dataset-root "$DATASET_ROOT")
fi

if [[ -n "$EE_REPEAT" && "$EE_REPEAT" != "None" && "$EE_REPEAT" != "null" ]]; then
  export EE_REPEAT="$EE_REPEAT"
fi

python -u -m yolobattle apptainer run --backend ultralytics --profile "$PROFILE" -- "${TRAIN_ARGS[@]}"
