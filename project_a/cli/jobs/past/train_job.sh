#!/bin/bash -l
# Batch training job for the BU SCC. Submit from the cli folder:
#     mkdir -p logs && qsub train_job.sh
# Re-submitting the same script continues run "$RUN" from its last finished epoch.
#
# CPU rule (the SCC kills jobs that use more cores than requested):
#   -pe omp 8 reserves 8 cores; we use 1 main thread + (NSLOTS - 2) loader workers.
#$ -P ec523kb
#$ -N attempt1
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

# Run name: weights/<RUN>.safetensors, weights/<RUN>_best.safetensors, weights/<RUN>.resume.pt
# For a new run, change RUN here AND the -N line above.
RUN=attempt1

# Stay inside the reserved cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORKERS=$(( NSLOTS > 2 ? NSLOTS - 2 : 0 ))
echo "run=$RUN slots=$NSLOTS workers=$WORKERS gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUT=weights/$RUN.safetensors
RESUME=""
[ -f weights/$RUN.resume.pt ] && RESUME="--resume weights/$RUN.resume.pt"

# Swap --demo-baseline for --baseline <staff checkpoint> once staff release it.
python train.py \
  --data-root /projectnb/ec523kb/projects/RoadSeg/dataset/Train \
  --demo-baseline \
  --epochs 40 --steps 4 --batch-size 8 --crop 512 \
  --lr 3e-4 --warmup 500 --pos-weight 1 --stress-aug 0.3 \
  --val-manifest val_small/manifest.json \
  --val-stress-manifest val_small_stress/manifest.json \
  --workers $WORKERS --output "$OUT" $RESUME
