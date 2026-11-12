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

## If edges are still bad (training changes)
- [ ] `--edge-weight 8` (from 4), everything else as attempt8
- [ ] `--step-weights 0,0,0,1` to test whether steps 2-4 can contribute

## After attempt8 (architecture changes, one per attempt)
- [ ] Spatial state: carry the core's grid output between steps (`model.py`)
- [ ] Staged upsampling head: 1/16 -> 1/8 -> 1/4 -> full (`parts.py`)
- [ ] Keep the training settings fixed while changing the architecture
- [ ] Each one changes the checkpoint layout: commit + tag the old code first, then retrain
- [ ] Run `python -m submission.model` and the unit tests after each change

## For the report
- [ ] Record K=0/1/2/4 + stress, and IoU / clDice / BIoU, for attempts 6, 7, 8
- [ ] Retrain on the staff baseline once it is released

## Notes: attempt7, what is good
- First run where the refiner beats the baseline on validation:
  K=0 52.96 -> K=1 53.96, K=2 54.03, K=4 54.01 (about +1 point).
- The gain holds on the stress set: stress K=4 went from 52.05 to 53.72.
- Roads are better connected: clDice rose from 61.81 to 71.20 (+9.4).
- It learns on the training tiles: y4 ends about 0.14 below y0 (0.60 vs 0.74),
  where attempt6 never got below y0.
- It repairs damaged inputs: input loss about 0.98 is brought down to 0.60.
- More steps no longer hurt: K=4 is not below K=1 (in attempt6 it was).
- Within budget: 4,024,658 parameters (core 3,159,552), peak memory 16.6 GiB.

## Notes: attempt7, what is bad
- Edges got worse: BIoU at K=4 fell from 8.44 to 4.78.
- Road area got slightly worse: IoU at K=4 fell from 64.29 to 63.38.
- The loop is really one step: K=1, 2 and 4 are within 0.07 of each other,
  and in training y3 and y4 equal y2.
- The whole gain comes from one metric (clDice), so it depends on the clDice loss.
- The +1 point is the same size as the validation noise (100 images).
- No validation during training, so we do not know if epoch 40 was the best
  or if the late epochs only memorized the training tiles.
- Six things changed at once, so we cannot say which change produced the gain.
- Trained on the DINOv2 stand-in, so it cannot be the leaderboard entry.

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

## Notes: architecture, what can still change
It is not all down to training. Two parts of the design hold back the two weak
points (the loop and the edges). Neither change is tested yet; the expected
effects are reasoning from the logs, not measurements.

### 1. Spatial state (do this first; targets the loop)
- Now: only the 32 memory tokens and the answer y are carried between steps.
  The grid tokens (one per patch) are rebuilt from the features and y at every
  step, and the core's per-patch output is thrown away after the head reads it.
- Problem: step 2 starts from almost the same inputs as step 1, so it produces
  almost the same thing. This matches the log: steps 2-4 add nothing.
- Change: keep the core's grid output as part of the state, so each step builds
  on the previous step's computation.
- Allowed: the handout says a spatial state may keep one token per coarse patch,
  and lists a low-resolution feature grid as the reasonable starting state.
- Where: `submission/model.py` (`initialize` and `step`). Almost no new parameters.
- Success looks like: K=4 clearly above K=1 on validation.

### 2. Staged upsampling head (targets the edges)
- Now: each token becomes its 16x16 pixel block through one linear layer,
  independently of its neighbours; a small conv stack tidies up afterwards.
- Change: upsample in stages, 1/16 -> 1/8 -> 1/4 -> full, with a small conv at
  each stage that also sees the current answer at that scale.
- Allowed: convolutions for upsampling and state-conditioned detail recovery are
  permitted, as long as the input is the core's output and not the image.
- Where: replaces `Head` in `submission/parts.py`.
- Success looks like: BIoU at K=4 above the baseline's (about 8).

### 3. Smaller options (lower priority)
- Bigger core: 3.16M of the 5M cap is used, so 6 layers instead of 4 would fit.
  Little gain expected while steps 2-4 are idle.
- Finer answer tokens: patch 8 instead of 16 gives 4x the tokens and much slower
  attention on full tiles. Not a starting point.
- Leave alone: the core itself (4 self-attention layers, d=256) shows no fault.

### Order
1. attempt8 first (training only, about 45 minutes): shows how much of the edge
   loss came from the clDice weight.
2. Spatial state, alone, with training settings fixed.
3. Staged upsampling head, alone.
