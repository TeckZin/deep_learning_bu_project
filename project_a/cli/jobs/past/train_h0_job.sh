#!/bin/bash -l
# Train our own H0 on frozen DINOv2 (stand-in baseline). Submit from the cli folder:
#     mkdir -p logs && qsub train_h0_job.sh
# First download DINOv2 once on the login node (needs internet):
#     python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"
#$ -P ec523kb
#$ -N h0_dino
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
  --val-manifest val_small/manifest.json \
  --dino dinov2_vits14 --epochs 10 --batch-size 8 --lr 1e-3 \
  --workers $WORKERS --output weights/h0_dino.safetensors
