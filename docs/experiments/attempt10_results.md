# attempt10: results

Architecture and decisions: `attempt10_design.md`. Per-epoch numbers: `attempt10_fold0_history.csv`.

## Run

| | |
|---|---|
| Job | 8805405, 1x H200, about 2h15m, peak memory 20.7 GiB |
| Code | commit `6f51784`, `experiments/attempt10.json`, no `--set` overrides |
| Data | CV fold 0 held out (839 validation images); out-of-fold H0 heads `weights/h0/dinov2_vits14_oof/fold0` |
| Recipe | 40 epochs, 4 unrolled steps, batch 8, crop 512, AdamW lr 3e-4 (500 warmup, cosine to 1.5e-5), wd 0.05, clip 1.0, edge weight 4, clDice weight 0.2, y0_aug 0.5, stress_aug 0, detach_y |
| Params | 5,668,210 (core 4,739,072) |
| Outputs (SCC) | `runs/fdzeolla-attempt10/fold0/`: `refiner_best.safetensors` (= epoch 40), `refiner.safetensors`, `history.csv`, `run.json`; log `runs/logs/attempt10_f0.8805405.log` |

Quality = 0.5 IoU + 0.3 clDice + 0.2 Boundary IoU. Validation every 5 epochs on all of fold 0.

## Validation (fold 0, 839 images)

| Epoch | K=0 | K=1 | K=2 | K=4 | K=4 IoU | K=4 clDice | K=4 BIoU |
|---|---|---|---|---|---|---|---|
| 0 (untrained) | 55.64 | 54.87 | 54.59 | 53.90 | 63.62 | 66.59 | 10.58 |
| 5 | 55.64 | 57.62 | 57.58 | 57.46 | 63.72 | 79.73 | 8.41 |
| 10 | 55.64 | 57.81 | 57.72 | 57.58 | 63.61 | 80.42 | 8.22 |
| 15 | 55.64 | 58.12 | 58.10 | 58.01 | 64.38 | 79.95 | 9.17 |
| 20 | 55.64 | 58.41 | 58.42 | 58.36 | 64.45 | 80.99 | 9.18 |
| 25 | 55.64 | 59.15 | 59.30 | 59.27 | 65.47 | 81.71 | 10.14 |
| 30 | 55.64 | 59.37 | 59.39 | 59.30 | 65.73 | 80.74 | 11.09 |
| 35 | 55.64 | 59.50 | 59.76 | 59.78 | 65.87 | 82.41 | 10.61 |
| **40** | 55.64 | **59.59** | **59.84** | **59.86** | **65.90** | **82.66** | **10.55** |

H0 alone (K=0): IoU 63.82, clDice 72.40, BIoU 10.06.

Validation loss by step (epoch 40): y0 0.7817, y1 0.6881, y2 0.6787, y4 0.6760.

## Training

| Epoch | Loss | y0 | y0 after aug (input) | y1 | y2 | y4 | Grad norm | Clipped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.697 | 0.560 | 0.801 | 0.708 | 0.697 | 0.694 | 0.31 | 5% |
| 10 | 0.536 | 0.563 | 0.809 | 0.537 | 0.534 | 0.536 | 0.22 | 0% |
| 20 | 0.488 | 0.557 | 0.800 | 0.492 | 0.487 | 0.487 | 0.30 | 2% |
| 30 | 0.427 | 0.547 | 0.801 | 0.435 | 0.426 | 0.425 | 0.53 | 9% |
| 40 | 0.393 | 0.540 | 0.782 | 0.400 | 0.392 | 0.391 | 0.57 | 12% |

Gradient norm by module at epoch 40: tokenizers 0.47, head 0.19, memory init 0.18, core 0.10, image CNN 0.09.

## Loop diagnostics (first 50 validation tiles, epoch 40, 8 steps)

| Step t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mean \|dy\| | 2.58 | 1.20 | 0.43 | 0.22 | 0.15 | 0.13 | 0.13 | 0.13 |
| mean \|dz\| | 0.379 | 0.251 | 0.163 | 0.111 | 0.075 | 0.051 | 0.036 | 0.026 |
| GRU gate u | 0.45 | 0.48 | 0.45 | 0.45 | 0.45 | 0.46 | 0.47 | 0.47 |
| loss (t0 = 0.812) | 0.670 | 0.656 | 0.652 | 0.649 | 0.649 | 0.649 | 0.650 | 0.651 |

Quality on these 50 tiles: K=4 63.06, K=8 62.59, K=4 reset-state 60.12, K=4 no-mask-feedback 60.75.

## What it shows

- **+2.16 over the same recipe on the attempt7 architecture** (see Control below), and +4.22 over H0.
- **Large gain over H0: +4.22 at K=4.** By metric: clDice +10.3 (contributes 3.08), IoU +2.1 (1.04),
  BIoU +0.5 (0.10). Most of the gain is still connectivity.
- **Edges no longer get worse.** BIoU ends 0.5 above H0 (attempt7 lost about 3.7). It first dropped to about
  8.2 (epochs 5-10) and then recovered.
- **The loop is still mostly one step.** K=1 gives +3.95. K=2 adds 0.25 and K=4 adds 0.02. The same holds
  in training (y1 0.400, y4 0.391). The corrections do shrink smoothly (|dy| 2.6 -> 0.2 by t=4), and z keeps
  changing at every step. Steps 2-4 are not idle, but they add little quality.
- **Stable past the trained K.** z settles (|dz| 0.026 at t=8, gate about 0.47), and K=8 is only 0.5 below K=4.
- **No bypass.** Resetting z before each step costs 2.9 points, and feeding y0 back instead of y_t costs 2.3.
  The recurrent state is used, but less than K=0 -> K=1 suggests.
- **Not converged.** The best epoch is the last one, and validation was still rising (+0.1 per 5 epochs at
  the end) while the LR reached its floor. Train loss keeps falling. Gradient norm and clip rate rise after
  epoch 20 (0.30 -> 0.57, 2% -> 12%), mostly in the tokenizers.

## Stress set (fold 0)

Job 8928429: the 839 fold-0 validation images corrupted with the staff generator (`splits.py manifest
--fold 0 --stress --seed 523`, about 280 each of blur, occlusion and shadow), scored with the epoch-40
weights. The stress copy is at `runs/eval/val_fold0_stress/` on the SCC, usable as `--val-stress-manifest`.
Results: `weights/fdzeolla-attempt10/fold0/stress_eval.json`.

| | K=0 | K=1 | K=2 | K=4 |
|---|---|---|---|---|
| Ordinary | 55.64 | 59.59 | 59.84 | 59.85 |
| Stress (all) | 55.13 | 58.90 | 59.15 | 59.18 |
| Stress: blur (279) | 54.80 | 58.89 | 59.18 | 59.21 |
| Stress: occlusion (279) | 55.23 | 59.31 | 59.51 | 59.53 |
| Stress: shadow (281) | 55.38 | 58.50 | 58.76 | 58.79 |

- Stress costs 0.67 at K=4 (0.51 at K=0), although attempt10 trained without stress augmentation.
  The refiner's gain over H0 holds on stress images (+4.05 vs +4.21).
- Leaderboard-style score on fold 0: 0.2 x 59.59 + 0.2 x 59.84 + 0.4 x 59.85 + 0.2 x 59.18 = **59.66**.
- Shadow costs the most. Each kind is a different third of the images, so the per-kind rows are not
  exact before/after pairs.
- Ordinary K=4 is 59.85 here vs 59.86 in training (different GPU, L40S vs H200).

## Control: same recipe, attempt7 architecture

Job 8900326: `qsub -N a10ctl_f0 jobs/train.sh attempt10 0 --arch attempt7 --run fdzeolla-attempt10-arch7`.
Only the architecture differs. Per-epoch numbers: `attempt10_arch7_fold0_history.csv`.

| Fold 0, epoch 40 | K=0 | K=1 | K=2 | K=4 | K=4 IoU | K=4 clDice | K=4 BIoU | Params | Peak mem |
|---|---|---|---|---|---|---|---|---|---|
| attempt7 architecture | 55.64 | 57.73 | 57.73 | 57.70 | 63.47 | 80.44 | 9.19 | 4.02M | 16.6 GiB |
| attempt10 | 55.64 | 59.59 | 59.84 | 59.86 | 65.90 | 82.66 | 10.55 | 5.67M | 20.7 GiB |
| Difference | | +1.86 | +2.11 | **+2.16** | +2.43 | +2.22 | +1.36 | | |

- **The architecture is worth +2.2 at K=4.** By metric: IoU contributes 1.22, clDice 0.67 and BIoU 0.27.
  The gain is spread over all three metrics. The recipe alone (control vs H0) gives +2.1, mostly clDice.
- **Only attempt10 gets anything from steps 2-4.** In the control, K=1 to K=4 is -0.03; in attempt10 it is +0.27.
- **The control plateaus; attempt10 does not.** The control's best epoch was 35 (K=4 57.73) and it gained
  0.2 after epoch 25. attempt10 was still rising at epoch 40.
- **The loop settles better.** At t=8, |dy| is 0.48 in the control and 0.13 in attempt10.
- **Both runs use their state.** On 50 tiles, resetting the state costs 5.9 in the control (memory slots) and
  2.9 in attempt10. Feeding y0 back costs 2.7 and 2.3.
- **Training fit:** final train loss at y4 is 0.463 in the control and 0.391 in attempt10.
- **Rising gradient norms come from the recipe.** The control's tokenizer gradient also rises about 4x
  after epoch 10 (0.08 -> 0.31). attempt10 runs at higher norms (total 0.57 vs 0.36; 12% vs 2% of
  iterations clipped).
- **Confound:** attempt10 is also larger (6 core layers vs 4, 5.67M vs 4.02M parameters). The +2.2 is
  architecture plus capacity.

## Per-image examples

Made with `examples.py` (job 8900318). It scores each of the 839 images at K=0/1/2/4 and at K=4 with z
reset before every step, then draws panels for the 6 worst, best, most improved, most worsened, most
helped by steps 2-4, and most z-dependent images. Rankings skip images with under 1% road.

    qsub jobs/examples.sh fdzeolla-attempt10 0        # -> weights/fdzeolla-attempt10/fold0/examples/

Gallery to browse on GitHub (JPEG copies of the panels, score tables):
[`attempt10_examples/`](attempt10_examples/README.md). Built with
`python examples.py --out <examples dir> --share ../../docs/experiments/attempt10_examples --title "attempt10, fold 0"`.

Of the 527 images with at least 1% road:
- K=0 -> K=4: 368 improve by more than 1 point and 72 get worse; the median change is +3.8.
- K=1 -> K=4: 104 improve and 39 get worse.
- z (K=4 vs z reset): helps 321 and hurts 59.
- Gain by road density: +9.8 on thin-road tiles (1-3% road), +6.0 at 3-8%, +2.4 above 8%.

What the panels show:
- Worst: all 9 images below 10 quality already failed at K=0. They are thin rural roads H0 never
  finds, and the refiner never recovers them.
- Improves: H0 misses a road, step 1 finds it, and steps 2-4 clear false patches (urban/2378: 1.5 -> 59.7).
  Predicted roads are wider than the truth, which keeps BIoU low.
- Worse: often a thin road that H0 found and step 1 erases (rural/3306: 46.6 -> 18.1). With z reset it
  survives (52.3), so z can also push in the wrong direction.
- Uses z: resetting z adds false road on bare ground and construction sites (urban/3868: 57.2 -> 39.7).
- Predicted roads show small holes on a regular grid, which looks like a 16-pixel patch artifact.

## Submission

attempt10 is the active preset (`ACTIVE = "attempt10"`) and the current submission.

| | |
|---|---|
| Bundle (SCC) | `submissions/attempt10_fold0.zip` and `submissions/attempt10_fold0/` in the team space |
| Contents | `infer.py`, `frozen_encoder.py`, `submission/*.py`, one weights file (H0 + refiner), README |
| Encoder | DINOv2 ViT-S/14 from torch.hub, frozen |
| H0 | `weights/h0/dinov2_vits14_oof/fold0/h0_all.safetensors` (trained on folds 1-4) |
| Refiner | `weights/fdzeolla-attempt10/fold0/refiner_best.safetensors` (epoch 40, folds 1-4) |
| Leaderboard entry | `OUT/k4/` from `python infer.py --input IMAGES --output OUT` (`--input` may also be one image file) |
| Environment | `requirements.txt`, pinned to the SCC environment: Python 3.12, numpy 2.3.5, Pillow 12.3.0, safetensors 0.5.3, torch 2.6.0 |
| Encoder download | DINOv2 from torch.hub on first run, as in earlier submissions (one-line command in the bundle README) |
| Expected score (fold 0) | 59.66 leaderboard-style (K=4 59.86, stress K=4 59.18) |

Built and checked by job 8938187 (the first build, job 8937525, is kept as `submissions/attempt10_fold0_v1`;
it had unpinned requirements and no single-image input):

    python export_submission.py --refiner weights/fdzeolla-attempt10/fold0/refiner_best.safetensors \
        --out /projectnb/ec523kb/projects/CowenZeollaZinTan/submissions/attempt10_fold0 --zip \
        --check runs/eval/export_check_images

`--check` unzipped a copy into an empty folder and ran `infer.py` on one urban and one rural validation
image, then on a single image file; both wrote masks for K = 0, 1, 2, 4, 6, 8.

- The bundle runs the trained model with the image CNN on: `infer.py` rebuilds it from the config stored
  in the weights and packs the image into f.
- Fold 0 was held out, so H0 and the refiner saw 4 of the 5 folds. A final bundle could retrain both on all
  folds (after the 80-epoch run settles the training length) and be exported the same way.

## Not measured yet
- How much of the +2.2 over the control comes from size alone (e.g. attempt7 with `--set core.layers=6`).
- Other folds (variance). The stress set is measured above.
- Which part does the work: image CNN, spatial z, RoPE, 6 layers.
