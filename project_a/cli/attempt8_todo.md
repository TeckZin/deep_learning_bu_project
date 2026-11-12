# attempt8 TODO

Goal: keep attempt7's connectivity gain (clDice 71.2) and win back the edges.

## Setup
- [ ] Commit attempt7 and tag it: `git add -A && git commit -m "attempt7" && git tag attempt7`
- [ ] Create the job script:
      sed -e 's/attempt7/attempt8/g' -e 's/--cldice-weight 0.5/--cldice-weight 0.2/' \
        -e 's/--val-every 0/--val-every 5/' train_job_attempt7.sh > train_job_attempt8.sh
- [ ] Check the script: `grep -n "attempt8\|cldice\|val-every" train_job_attempt8.sh`
- [ ] Submit: `qsub train_job_attempt8.sh`
- [ ] Watch: `tail -F logs/attempt8.*.log`

## Changes vs attempt7 (only these two)
- [ ] `--cldice-weight 0.5` -> `0.2`
- [ ] `--val-every 0` -> `5` (validate every 5 epochs, keep the best)

## Check when it finishes
- [ ] K=4 quality above attempt7's 54.01
- [ ] BIoU at K=4 back above the baseline's (about 8), up from 4.78
- [ ] clDice still well above the baseline's (about 62)
- [ ] Which epoch was best? If not the last one, attempt7 was probably overtrained
- [ ] Look at 5 predicted masks next to their labels: are roads too thick?

## If edges are still bad
- [ ] attempt9: `--edge-weight 8` (from 4), everything else as attempt8
- [ ] attempt10: `--step-weights 0,0,0,1` to test whether steps 2-4 can contribute

## For the report
- [ ] Record K=0/1/2/4 + stress, and IoU / clDice / BIoU, for attempts 6, 7, 8
- [ ] Retrain on the staff baseline once it is released

## Notes: the boundary problem
- Boundary IoU is 20% of quality. It compares only a 3-pixel band along road edges,
  so an edge shifted by a few pixels scores close to zero.
- Baseline alone: BIoU about 8. Low because y0 is predicted on a 16-pixel grid.
- attempt7: BIoU fell from 8.44 (before training) to 4.78, while clDice rose
  from 61.81 to 71.20. All of the gain came from connectivity, none from edges.
- Suspected cause (not confirmed): the clDice loss only looks at centrelines, so
  making roads thicker costs it nothing, and thicker roads move every edge.
- The refiner cannot see the image (handout rule), only 1/16-resolution features,
  so edge gains will be limited. Getting back to 8 is realistic; far above is not.
- Validation has 100 images: differences under about 1 quality point are noise.
