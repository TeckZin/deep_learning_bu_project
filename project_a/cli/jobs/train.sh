#!/bin/bash -l
# Refiner for one experiment and one validation fold. Needs jobs/train_h0.sh for that fold first.
# From the cli folder:  mkdir -p logs && qsub jobs/train.sh EXPERIMENT FOLD [extra train.py flags]
#   EXPERIMENT = name of experiments/<name>.json; extra flags override the file (e.g. --set core.layers=6)
# Outputs: weights/<login>-<run>/fold<FOLD>/ (or --run NAME). Re-submitting continues from the last epoch.
#$ -P ec523kb
#$ -N refiner
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
source jobs/common.sh
EXP="${1:?usage: qsub jobs/train.sh EXPERIMENT FOLD [flags]}"; FOLD="${2:?fold}"; shift 2
# Run name: --run in the flags, else <login>-<"run" in the experiment file>, so runs are unique on runs/.
RUN="$USER-$(python -c "import json; print(json.load(open('experiments/$EXP.json')).get('run', '$EXP'))")"
ARGS=("$@")
for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--run" ] && RUN="${ARGS[$((i + 1))]}"; done
RESUME=""
[ -f "weights/$RUN/fold$FOLD/refiner.resume.pt" ] && RESUME="--resume weights/$RUN/fold$FOLD/refiner.resume.pt"
# H0 heads: weights/h0/<encoder>_<scheme>/fold<FOLD> from the experiment file (or --h0-dir in the flags)
python train.py --config "experiments/$EXP.json" --val-fold "$FOLD" --run "$RUN" --workers $WORKERS $RESUME "$@"
