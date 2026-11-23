#!/bin/bash -l
# Submit from project_a/cli (create logs/ BEFORE qsub):
#   mkdir -p logs
#   qsub compare_masks_job.sh /path/refiner.safetensors /path/h0.safetensors [Train|Val|Both] [K]
# Uses the current submission/ model code, which must match the checkpoint.
#$ -P ec523kb
#$ -N compare_masks
#$ -cwd
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 2
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

set -eo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "Usage: qsub $0 REFINER.safetensors H0.safetensors [Train|Val|Both] [K=4]" >&2
  exit 2
fi

cd "${SGE_O_WORKDIR:-$(pwd)}"
CHECKPOINT=$1
H0=$2
SPLIT=${3:-Train}
STEPS=${4:-4}
DATASET=/projectnb/ec523kb/projects/RoadSeg/dataset

case "$SPLIT" in
  Train|Val) SPLITS=("$SPLIT") ;;
  Both) SPLITS=(Train Val) ;;
  *) echo "Split must be Train, Val, or Both" >&2; exit 2 ;;
esac
case "$STEPS" in
  [0-8]) ;;
  *) echo "K must be an integer from 0 to 8" >&2; exit 2 ;;
esac
for FILE in "$CHECKPOINT" "$H0" compare_masks.py; do
  if [ ! -f "$FILE" ]; then
    echo "Missing $FILE (submit from the repo's project_a/cli folder)" >&2
    exit 1
  fi
done

module load miniconda
conda activate dl_env

export OMP_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export PYTHONUNBUFFERED=1
export LC_ALL=C

RUN_NAME=$(basename "$CHECKPOINT" .safetensors)
OUTPUT_ROOT="comparisons/${RUN_NAME}_k${STEPS}_${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
echo "Checkpoint: $CHECKPOINT"
echo "H0: $H0"
echo "Output: $PWD/$OUTPUT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv

for CURRENT_SPLIT in "${SPLITS[@]}"; do
  python -u compare_masks.py \
    --data-root "$DATASET/$CURRENT_SPLIT" \
    --checkpoint "$CHECKPOINT" \
    --standin-baseline "$H0" \
    --first-per-domain 50 \
    --steps "$STEPS" \
    --device cuda \
    --output "$OUTPUT_ROOT/$CURRENT_SPLIT"
done

echo "Finished. Comparisons and predicted masks: $PWD/$OUTPUT_ROOT"
