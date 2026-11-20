# Modular refactor (Project A refiner)

Status: implemented on branch `refactor/modules` (2026-11-17), off `main` at the `attempt7` tag.
Diagrams, entry points and file map: [CodeStructure.md](CodeStructure.md).
Companion to [ModulesAndDesignDecisions.md](ModulesAndDesignDecisions.md), which lists the stages
mapped to modules here. Paths are relative to `project_a/cli/`.

## Goal
`main` is a stable base where every stage of the pipeline is a module, several architectures
coexist as named presets, and ablations are a config change, not a code edit.

## Constraints that shaped the design
- **Leaderboard = our test set; there is no staff baseline in practice.** Our frozen DINOv2 +
  our own H0 head is the frozen encoder stage (handout: frozen segmenter E + H0). The code submission is a zip with its own folder-in /
  folder-out inference CLI, one weights file, and no training or metric code.
- **Leaderboard runner advice:** stage 1 encodes the whole folder into an intermediate folder,
  stage 2 refines from it (with checks that stage 1 finished); output masks are 0/1, the size of
  the input, one folder per number of refinement rounds K.
- **Handout contract still holds** (staff `predict.py`, `tests/`): `build_refiner(feature_channels)`
  with no other argument, `initialize()` never calls `model.core`, `step()` calls it exactly once,
  prefix-consistent, no K. Staff files stay unedited.
- **Image only inside the loop** (TA, 2026-11-18): a CNN inside `step()` may receive the image and
  features (so doc item 12a, feeding the image to the head's CNN, is allowed); using the image on the
  final prediction after the loop is not. The current head still sees only core outputs and y_t.
  Details: [ChangingTheArchitecture.md](ChangingTheArchitecture.md#where-the-image-may-go-ta-ruling).
- **CV (team decision):** Train + Val pooled into 5 folds stratified by urban / rural; a run trains
  on 4 and validates on 1. Default is one run (fold 0); 5 runs, with per-fold hyperparameters, on demand.
- **H0 vs refiner data (team decision):** out-of-fold y0 by default, disjoint halves as an option.

## Module map (doc stages → files)

| Doc stage | File | Notes |
|---|---|---|
| 1 Data loader, 1a augmentation | `data.py` | `TileDataset` over split rows, `StressAugment`, flips/rotations, aligned crops, `degrade_logits` (y0 aug) |
| 2 Encoder | `frozen_encoder.py` | `ENCODERS`: DINOv2 S/B/L (frozen, torch.hub) and `pool` (no download; tests) |
| 3 CV split | `splits.py` → `splits/folds.csv` | Committed fold file; `fold_plan()` decides H0 vs refiner data |
| 4 Prediction head (H0) | `frozen_encoder.py` (`H0Head`), `train_h0.py` | `FrozenEncoder(encoder, heads)`; per-image head choice for OOF |
| 4b Peek-in eval | `evaluation.py` | Same `validate()` as stage 14, at K=0; `validate_heads()` scores all H0s in one encoder pass |
| 7a Cache | `infer.py` stage 1 (intermediate folder) | Training cache deferred: stress augmentation changes the image, hence f |
| 5 Patchify, 7 Tokenizer, 8 Fuse x,y | `submission/tokenizers.py` | One module; `fuse` = `sum` / `concat`; `feat_norm`, `uncertainty` flags |
| 6 z initialization | `submission/memory.py` | `memory` (learned slots read the grid once) |
| 9 QKV, 10 positions, 11 attention / FFN / norm / layers | `submission/core.py` | One module: `mode`, `layers`, `heads`, `ff_mult`, `dropout` |
| (positions) | `submission/pos.py` | Additive 2D sin/cos with crop offset |
| 12 CNN, 13 head | `submission/output_heads.py` | `pixelshuffle` head: `hidden`, `dilations`, `ctx`, `gate` |
| 14 Evaluation | `evaluation.py`, staff `evaluate.py` / `score.py` | Visual comparison tool (14a) not built yet |
| Loss | `losses.py` | BCE (edge-weighted) + Dice + soft clDice |
| Leaderboard inference / zip | `infer.py`, `export_submission.py` | See below |

`submission/model.py` wires tokenizer → core → head generically (`initialize` / `step`) and is the
only file that knows the step order.

## Choosing architectures and running ablations
- `submission/config.py`: one frozen dataclass per module, `RefinerConfig` holding them, a
  `PRESETS` dict (`attempt7`, `attempt6`) and `ACTIVE` (what `build_refiner(c)` builds). Each
  module file has a `kind → class` registry, so a new variant is a new class + a registry entry +
  a preset. Never edit a preset that has trained weights; add a new one.
- `attempt7` reproduces the attempt7 code bit-for-bit (old checkpoints load with `strict=True`,
  identical outputs); `attempt6` matches attempt6's recorded parameter count (3,944,449).
- `train.py --arch NAME --set core.layers=6 --set tokenizer.fuse=concat` for ablations.
- `train.py --config experiments/<name>.json` holds a training recipe (any flag, by dest name),
  with `"fold_overrides": {"<fold>": {...}}` for per-fold settings. Command line > fold override >
  file > defaults.
- Every checkpoint stores its architecture (`refiner_cfg`) in the safetensors metadata;
  `submission.model.load_refiner()` rebuilds any checkpoint whatever `ACTIVE` says. `run.json`
  next to it records every setting and the git commit.

## Folds and H0 schemes (`splits.fold_plan`)
- **oof (default):** one H0 per training fold j (`h0_oof<j>`, trained on the other training folds)
  plus `h0_all` (all 4 training folds). The refiner trains on all 4 training folds; each image's y0
  comes from the H0 that left its fold out. Validation and inference use `h0_all`.
- **halves:** training folds split 50/50 per domain; `h0_a` on half a, refiner on half b; `h0_a`
  everywhere.
- `train_h0.py` trains all heads of a scheme in ONE pass (the encoder runs once per batch; each head
  takes a loss only on its own images) and writes `weights/h0/<encoder>_<scheme>/fold<F>/plan.json`, which
  `train.py` checks so heads and fold can't be mismatched.

## Leaderboard submission
- `infer.py --input IMAGES --output OUT [--stage encode|refine]`: stage 1 writes
  `OUT_intermediate/<id>.safetensors` + `index.json` (written last); stage 2 asserts it is complete,
  then writes `OUT/k<K>/<id>.png` (uint8, 1 = road) for K = 0, 1, 2, 4, 6, 8 from one 8-step run.
  One image at a time, so RAM does not grow with the folder.
- `export_submission.py --refiner CKPT [--h0 H0] --out dist/submission --zip --check SAMPLES`:
  copies `infer.py`, `frozen_encoder.py`, `submission/`, bundles H0 + refiner + metadata into
  `weights/submission.safetensors`, writes README / requirements, and `--check` runs the export
  from an empty temp folder.

## Jobs (BU SCC)
`jobs/train_h0.sh FOLD [SCHEME]`, `jobs/train.sh EXPERIMENT FOLD [flags]`, and
`bash jobs/cv.sh EXPERIMENT [FOLDS] [flags]` (submits H0 then refiner per fold, skipping H0 when
already trained). Data root from `$ROADSEG_DATA`. Old per-attempt scripts are in `jobs/past/`
(they run only at tag `attempt7`).

## Verification done
- Staff tests + `tests/test_project.py`: 35 passed. Ours cover every preset under the staff
  runner's checks (core calls, parameter identity, prefix consistency) and the caps, config
  overrides / metadata round trip, stratified folds, OOF routing never using an H0 that saw the
  image, halves disjointness, and train_h0 → train → export → infer end to end on synthetic tiles.
- `smoke_test.py` passes; staff `predict.py` runs on a checkpoint from the new `train.py`.
- `infer.py` masks match a direct encoder → H0 → refiner computation pixel for pixel.

## Still to do
1. On SCC: `python splits.py make`, commit `splits/folds.csv`; first real run `bash jobs/cv.sh attempt7 0`
   to get a fold-0 reference for attempt8 and later variants.
2. New variants as presets (separate PRs): RoPE in the core, spatial state (memory + grid tokens,
   attempt8 note 1), staged upsampling head (attempt8 note 2).
3. Visual comparison tool (doc 14a: image / y0 / yK / label grid).
4. Training-time feature cache (only for runs without stress augmentation).
5. Handout controls: non-recurrent one-step model and untied stack as presets.
