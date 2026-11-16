#!/bin/bash -l
# attempt6, step 2: refiner on HALF B of Train (--part b), using the H0 from
# train_h0_job_attempt6.sh (trained on half A), plus:
#   --y0-aug 0.5   degrade y0 (refiner input only) so there are real errors to fix
#   --lr 3e-4      (attempt4/5 used 1e-4)
#   --crop 512 --batch-size 8
# Submit from the cli folder AFTER h0_attempt6 has finished:
#     mkdir -p logs && qsub train_job_attempt6.sh
# or chain it:  qsub -hold_jid h0_attempt6 train_job_attempt6.sh
# Re-submitting continues run "$RUN" from its last finished epoch.
#$ -P ec523kb
#$ -N attempt6
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

RUN=attempt6
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
  --val-manifest val_small/manifest.json \
  --val-stress-manifest val_small_stress/manifest.json \
  --workers $WORKERS --output "$OUT" $RESUME
