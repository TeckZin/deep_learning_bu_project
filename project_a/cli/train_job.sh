#!/bin/bash -l
# Batch training job for the BU SCC. Submit from the cli folder:
#     mkdir -p logs && qsub train_job.sh
# Re-submitting the same script continues from the last finished epoch.
#$ -P ec523kb
#$ -N road_refiner
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

OUT=weights/refiner.safetensors
RESUME=""
[ -f weights/refiner.resume.pt ] && RESUME="--resume weights/refiner.resume.pt"

# Swap --demo-baseline for --baseline <staff checkpoint> once staff release it.
python train.py \
  --data-root /projectnb/ec523kb/projects/RoadSeg/dataset/Train \
  --demo-baseline \
  --epochs 40 --steps 4 --batch-size 8 --crop 512 \
  --lr 3e-4 --warmup 500 --pos-weight 1 --stress-aug 0.3 \
  --val-manifest val_small/manifest.json \
  --val-stress-manifest val_small_stress/manifest.json \
  --workers 6 --output "$OUT" $RESUME
