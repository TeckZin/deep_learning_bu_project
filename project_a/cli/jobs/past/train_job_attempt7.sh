#!/bin/bash -l
# attempt7 = attempt6 data setup (H0 on half A, refiner on half B, --y0-aug 0.5) plus
# refiner changes in submission/parts.py and the loss:
#   - LayerNorm on the frozen DINOv2 features before the tokenizer MLP
#   - uncertainty channel 4p(1-p) in the answer tokens
#   - head: core-output context channels (16) into the full-res stack + gated update
#   - soft-clDice loss (--cldice-weight 0.5)
#   - --val-every 0: validate only before training and after the last epoch
#   - --detach-y: no gradient through the fed-back answer (each step learns to improve its input)
# Reuses weights/h0_attempt6_best.safetensors (no need to retrain H0).
# Submit from the cli folder:  mkdir -p logs && qsub train_job_attempt7.sh
# Re-submitting continues run "$RUN" from its last finished epoch.
#$ -P ec523kb
#$ -N attempt7
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

RUN=attempt7
H0=weights/h0_attempt6_best.safetensors

if [ ! -f "$H0" ]; then
  echo "Missing $H0: run train_h0_job_attempt6.sh first"; exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORKERS=$(( NSLOTS > 2 ? NSLOTS - 2 : 0 ))
echo "run=$RUN h0=$H0 slots=$NSLOTS workers=$WORKERS gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUT=weights/$RUN.safetensors
RESUME=""
[ -f weights/$RUN.resume.pt ] && RESUME="--resume weights/$RUN.resume.pt"

python train.py \
  --data-root /projectnb/ec523kb/projects/RoadSeg/dataset/Train \
  --standin-baseline "$H0" --part b \
  --epochs 40 --steps 4 --batch-size 8 --crop 512 \
  --lr 3e-4 --warmup 500 --pos-weight 1 --edge-weight 4 --stress-aug 0.3 --y0-aug 0.5 \
  --cldice-weight 0.5 --detach-y --val-every 0 \
  --val-manifest val_small/manifest.json \
  --val-stress-manifest val_small_stress/manifest.json \
  --workers $WORKERS --output "$OUT" $RESUME
