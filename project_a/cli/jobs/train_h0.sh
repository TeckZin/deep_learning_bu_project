#!/bin/bash -l
# H0 heads for one validation fold (all heads of the scheme in one pass).
# From the cli folder:  mkdir -p logs && qsub jobs/train_h0.sh FOLD [SCHEME=oof] [extra train_h0.py flags]
# Writes weights/h0/<encoder>_<scheme>/fold<FOLD>/h0_*.safetensors + plan.json (encoder: --encoder, default dinov2_vits14)
#$ -P ec523kb
#$ -N h0
#$ -l h_rt=06:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
source jobs/common.sh
FOLD="${1:?usage: qsub jobs/train_h0.sh FOLD [SCHEME] [flags]}"; SCHEME="${2:-oof}"; shift $(( $# < 2 ? $# : 2 ))
python train_h0.py --val-fold "$FOLD" --h0-scheme "$SCHEME" --workers $WORKERS "$@"
