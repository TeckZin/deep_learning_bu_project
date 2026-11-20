# Changing a stage of the pipeline

How to change any stage 1-14 of [ModulesAndDesignDecisions.md](ModulesAndDesignDecisions.md).
Where the files sit and how they connect: [CodeStructure.md](CodeStructure.md). Paths are relative to `project_a/cli/`.

## Two kinds of change

**Refiner (stages 5-13, in `submission/`)**
1. Edit or add a class in the module's file; register a new variant in that file's registry
   (`TOKENIZERS`, `MEMORIES`, `HEADS`).
2. Expose it as a field of that module's config dataclass in `submission/config.py`, with a default that
   keeps the current behaviour (existing presets and trained weights must not change).
3. Add a preset to `PRESETS`, or for a quick try use `--set section.field=value` (no code).
4. Check: `python -m submission.model` (shapes, prefix consistency, parameter counts) and
   `python -m unittest discover -s tests` (every preset against the staff runner's rules and the caps).
5. Train: `bash jobs/cv.sh <experiment> 0 --arch <preset>`; outputs go to `weights/<login>-<run>/fold0/`.

Rules that cannot change: `initialize()` never calls the core; `step()` calls `model.core` exactly once;
the model never sees K; the image may be used anywhere **inside** the loop (inside `step()`), never on the
final prediction **after** the loop (TA ruling, below).

**Pipeline (stages 1-4 and 14, outside `submission/`)**
Edit the file; if it is a choice, add a flag to `train.py` / `train_h0.py` (any `train.py` flag can then be
set in `experiments/<name>.json`); run the tests.

## Where each stage lives

| Stage | Where | Knob / how |
|---|---|---|
| 1 Data loader, augmentation | `data.py`: `TileDataset`, `augment_tiles` (flips / rotations), `StressAugment`, `degrade_logits` (y0 corruption) | `--stress-aug`, `--y0-aug`; add a flag for anything new |
| 2 Encoder | `frozen_encoder.py`: add a class to `ENCODERS` with `.channels` and `forward(image, out_hw)` → features at H/16 | `train_h0.py --encoder NAME` (and `"encoder"` in the experiment file). Its H0 heads get their own folder `weights/h0/<encoder>_<scheme>/`. Retrain H0 and the refiner. |
| 3 CV split | `splits.py`: `make_folds` (folds), `fold_plan` (which data trains H0 vs the refiner) | `--h0-scheme oof\|halves`; a new scheme goes in `fold_plan` + both scripts' `choices`. Don't regenerate the shared `splits/folds.csv` casually. |
| 4 Prediction head (H0) | `frozen_encoder.py: H0Head`, trained by `train_h0.py` | `--hidden`; a different class would also need a `kind` in its saved metadata for `load_frozen_encoder` to rebuild it |
| 4b Peek-in eval | `evaluation.py`: `validate` at K=0, `validate_heads` | Automatic: the "frozen encoder + H0 alone (K=0)" line in `train.py`, per-head lines in `train_h0.py` |
| 5 Patchify | `RefinerConfig.patch` | `--set patch=8`; tokenizer, head and crop alignment follow it. Attention cost grows 4×. |
| 6 z initialization | `submission/memory.py`, `MemoryConfig` (`kind`, `nz`) | `--set memory.nz=64`; changing what is carried between steps (e.g. grid tokens) also means editing `step` in `submission/model.py` |
| 7 Tokenizer | `submission/tokenizers.py`, `TokenizerConfig` (`feat_norm`, `uncertainty`) | `--set tokenizer.feat_norm=false` |
| 7a Cache | Inference: `infer.py` stage 1 writes `<OUTPUT>_intermediate/` | Training cache not built: stress augmentation changes the image and so the features |
| 8 Fuse x, y, z | `tokenizers.py` (`fuse`: `sum` / `concat`); z joins as extra tokens in `model.py: step` | `--set tokenizer.fuse=concat`; a new fusion is a branch in `FusedTokenizer`. An image CNN stem would fuse here too. |
| 9-11 QKV, positional rotation, attention, heads, norms, FFN, layers | `submission/core.py`, `CoreConfig` (`mode`, `layers`, `heads`, `ff_mult`, `dropout`); positions are additive today (`pos.py`, added in the tokenizer) | `--set core.layers=6`, `--set core.heads=4`; custom Q/K/V or RoPE needs a hand-written attention block (`nn.TransformerEncoderLayer` does not expose Q and K) |
| 12 CNN | `submission/output_heads.py: PixelShuffleHead`, `HeadConfig` (`hidden`, `dilations`, `ctx`, `gate`) | `--set head.dilations=1,2,4`. 12a: the image (and features) **may** enter this CNN, because it runs inside the loop (TA, below); today it sees only state. 12b: z into the CNN needs `model.py` to pass the state to the head. |
| 13 Head, memory creation | Prediction: `output_heads.py`; memory = the core's memory-token output, split off in `model.py: step` | A new head class goes in `HEADS`, selected with `head.kind` |
| 14 Evaluation | `evaluation.py` (uses the staff `metrics.py`, don't edit it) | 14a visual tool not built yet: a new script reading `infer.py`'s `k<K>/` masks |

## Where the image may go (TA ruling)

The full-resolution image counts as a representation (like the features). Current rule (2026-11-18):

> "Basically if the CNN comes within the loop, even only receiving images and features, not numbers,
> it's fine. However if it is seeing the images after the loop, it's not acceptable."

| Where the image is used | Allowed? | Example in our code |
|---|---|---|
| Inside the loop: any CNN that runs in every `step()` | Yes | An image stem in the tokenizer (stages 7-8), or the image as extra input to the head's CNN (stage 12) |
| Before the loop, feeding every step | Yes | Image features computed once, like `f` |
| After the loop: on the final prediction after step K | **No** | Post-processing `y_K` with the image (e.g. a guided filter or edge snap in `infer.py`, after the refiner) |

This replaces an earlier answer that ruled out "transformer → CNN that sees the image"; inside the
loop it is now fine. The handout's rule still applies: no **independent** CNN segmentation bypass. Keep
the recurrent state in charge of the decision and show it with the runner's ablations
(`predict.py --ablation reset-state` / `no-mask-feedback` should still change the output a lot).

To get the image into `step(f, z, y)` (built for attempt10, `submission/image.py`): set `image.use=true`.
`train.py` then wraps the frozen base in `ImagePackedBase`, which packs the pixel-unshuffled RGB channels
next to the encoder features; `infer.py` stores the raw image in the intermediate folder (stage 1) and packs
it in stage 2. The refiner unpacks it with `unpack_image`. Only `SpatialRefiner` (memory.kind=spatial) uses it.

## Gotcha: changing H0 without changing the encoder

`jobs/cv.sh` skips H0 training when `weights/h0/<encoder>_<scheme>/fold<F>/plan.json` exists, and those
heads are shared by the whole team. A different encoder gets its own folder automatically, but a change
to H0 itself (e.g. `--hidden`) with the same encoder would reuse the old heads. Train the new heads into
their own folder and point the refiner at it:
```bash
python train_h0.py --val-fold 0 --hidden 256 --out-dir weights/h0/dinov2_vits14_oof_h256/fold0
bash jobs/cv.sh attempt8 0 --h0-dir weights/h0/dinov2_vits14_oof_h256/fold0 --run <login>-attempt8-h256
```
