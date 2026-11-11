#!/bin/bash -l
# attempt6, step 1: H0 on frozen DINOv2, trained on HALF A of Train only (--part a).
# The refiner (train_job_attempt6.sh) trains on half B, so its y0 has realistic errors.
# Submit from the cli folder:  mkdir -p logs && qsub train_h0_job_attempt6.sh
# Writes weights/h0_attempt6.safetensors and weights/h0_attempt6_best.safetensors
#$ -P ec523kb
#$ -N h0_attempt6
#$ -l h_rt=04:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

module load miniconda
conda activate dl_env
cd "$SGE_O_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORKERS=$(( NSLOTS > 2 ? NSLOTS - 2 : 0 ))
echo "slots=$NSLOTS workers=$WORKERS gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train_h0.py \
  --data-root /projectnb/ec523kb/projects/RoadSeg/dataset/Train \
  --part a \
  --val-manifest val_small/manifest.json \
  --dino dinov2_vits14 --epochs 10 --batch-size 8 --lr 1e-3 \
  --workers $WORKERS --output weights/h0_attempt6.safetensors
