#!/bin/bash -l
# attempt4: same as attempt3_dino, plus two edge changes:
#   - stronger full-resolution refine head (submission/parts.py: 64 ch, dilations 1,2,4,8,1)
#   - edge-weighted BCE (--edge-weight 4: pixels within 3 px of a road edge count 4x)
# Log also prints the training loss of y0 and of each step y1..y4.
# Submit from the cli folder:  mkdir -p logs && qsub train_job_attempt4.sh
# Re-submitting continues run "$RUN" from its last finished epoch.
#$ -P ec523kb
#$ -N attempt4_edges
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

# For a new run, change RUN here AND the -N line above.
RUN=attempt4_edges
H0=weights/h0_dino_best.safetensors

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
  --standin-baseline "$H0" \
  --epochs 40 --steps 4 --batch-size 8 --crop 512 \
  --lr 1e-4 --warmup 500 --pos-weight 1 --edge-weight 4 --stress-aug 0.3 \
  --val-manifest val_small/manifest.json \
  --val-stress-manifest val_small_stress/manifest.json \
  --workers $WORKERS --output "$OUT" $RESUME
