#!/bin/bash -l
# Example panels (examples.py) for a trained run: best / worst / improves / no_gain / late / uses_z.
# From the cli folder:  mkdir -p logs && qsub jobs/examples.sh RUN FOLD [extra examples.py flags]
#   e.g. qsub jobs/examples.sh fdzeolla-attempt10 0
# Reads weights/RUN/foldFOLD/refiner_best.safetensors and the H0 head from its run.json (eval_h0).
# Writes weights/RUN/foldFOLD/examples/ (scores.csv, one folder of PNGs per category).
# Add  --share DIR --title "..."  for a small gallery to commit (README.md + JPEGs); it can also be
# built later without a GPU:  python examples.py --out DIR/examples --share ../../docs/experiments/NAME
#$ -P ec523kb
#$ -N examples
#$ -l h_rt=03:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

source jobs/common.sh
RUN="${1:?usage: qsub jobs/examples.sh RUN FOLD [flags]}"; FOLD="${2:?fold}"; shift 2
DIR="weights/$RUN/fold$FOLD"
H0=$(python -c "import json; print(json.load(open('$DIR/run.json'))['eval_h0'])")
python examples.py --weights "$DIR/refiner_best.safetensors" --h0 "$H0" --val-fold "$FOLD" \
    --out "$DIR/examples" "$@"
