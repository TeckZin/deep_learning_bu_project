#!/bin/bash
# Submit an experiment on several folds: H0 for each fold, then the refiner after it.
# From the cli folder (login node):  bash jobs/cv.sh EXPERIMENT [FOLDS="0 1 2 3 4"] [extra train.py flags]
# Default use is ONE fold:            bash jobs/cv.sh attempt8 0
# Skips the H0 job when weights/h0/<encoder>_<scheme>/fold<F>/plan.json already exists (shared H0 heads).
# Runs land in weights/<login>-<run>/fold<F>/ (jobs/train.sh), so members never share a run folder.
set -euo pipefail
EXP="${1:?usage: bash jobs/cv.sh EXPERIMENT [FOLDS] [flags]}"; FOLDS="${2:-0 1 2 3 4}"; shift $(( $# < 2 ? $# : 2 ))
SCHEME=$(python3 -c "import json; print(json.load(open('experiments/$EXP.json')).get('h0_scheme', 'oof'))")
ENCODER=$(python3 -c "import json; print(json.load(open('experiments/$EXP.json')).get('encoder', 'dinov2_vits14'))")
mkdir -p logs
for F in $FOLDS; do
  HOLD=""
  if [ ! -f "weights/h0/${ENCODER}_$SCHEME/fold$F/plan.json" ]; then
    JID=$(qsub -terse -N "h0_${ENCODER}_${SCHEME}_f$F" jobs/train_h0.sh "$F" "$SCHEME" --encoder "$ENCODER")
    HOLD="-hold_jid $JID"
    echo "fold $F: H0 job $JID"
  fi
  JID=$(qsub -terse $HOLD -N "${EXP}_f$F" jobs/train.sh "$EXP" "$F" "$@")
  echo "fold $F: refiner job $JID"
done
