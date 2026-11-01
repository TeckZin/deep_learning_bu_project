# Midway training and submission runbook

This implements the temporary starting segmenter needed before staff release
their encoder/head. The objective is a runnable, trained submission. No minimum
accuracy, refinement gain, or monotonic improvement is assumed.

The pipeline is ImageNet ResNet18 → U-Net decoder/head → frozen features and
initial road logits → Teck's `Tx` tokenizer plus mask tokens → shared recurrent
transformer → corrected road logits. We train the U-Net decoder first with the
encoder frozen, then freeze the whole baseline and train the refiner. The
staff-owned baseline loader and evaluator are unchanged. Our results explicitly
identify the temporary baseline and are local results, not an official entry.

## What you need on SCC

- Your own clone with this implementation branch pulled.
- Your actual scheduler project name for `qsub -P`, GPU access, and the course's
  permitted GPU type if one is required. Check available resources with `qgpus`.
- Python 3.12 and a GPU driver compatible with the selected PyTorch CUDA wheel.
- Read access to `/projectnb/ec523kb/projects/RoadSeg/dataset`.
- Write access to `/projectnb/ec523kb/projects/CowenZeollaZinTan` or an alternative
  output/environment/cache directory you configure.
- Network access for package installation and one ImageNet weight download, or
  the corresponding packages/weight file transferred from another machine.

Read data in place. Do not copy the whole dataset into your home directory.
Train in scheduler jobs; setup and weight downloading do not train a model.

## First SCC run

From your SCC clone, after fetching and switching to the implementation branch:

```bash
cp scripts/scc_config.example.sh scripts/scc_config.local.sh
module avail python3
qgpus
```

Edit `scripts/scc_config.local.sh`. Fill in `SCC_PROJECT`, check `SCC_REPO`, and
set `SCC_PYTHON_MODULE`/`SCC_PYTHON` so Python 3.12 is selected. Adjust environment,
output and cache locations if necessary. The local settings are gitignored.
The default CUDA wheel index is `cu124`; change it if your available driver
requires a different supported PyTorch 2.6.0 wheel build. CUDA runtime libraries
come with the wheel; a working compatible NVIDIA driver is still required.

```bash
bash scripts/scc_setup.sh
bash scripts/scc_submit.sh smoke
```

Setup creates your private environment, installs the project, downloads encoder
weights if absent, and checks selected image/mask pairs. The smoke submission
requests one GPU, four CPU slots and 30 minutes. It trains 10 updates per stage
on two tiles per domain with 128×128 crops, exports a bundle, runs image
inference, and writes a local evaluation report. The job must finish these
checks before you move on; no score threshold is required.

The submission command prints a run directory and SGE job ID. Monitor with:

```bash
qstat -u "$USER"
tail -f /path/printed/by/the/submission/training.log
```

If the smoke job completes, submit the default midway run:

```bash
bash scripts/scc_submit.sh train
```

This requests four hours as a wall-time limit, not an estimated training time.
It uses the first 10 sorted Train images per domain (20 tiles total), first two
Val tiles per domain, 256×256 crops, batch size two, and 200 updates per stage.
Checkpoints and full-tile validation are saved every 50 updates and at completion.
If you need a faster first artifact, lower the two update counts in your local
SCC config. The smoke run also produces real trained weights and can serve as
the first working artifact; report its limited training honestly.

Every scheduler job ends by printing the exported bundle path. Logs and results
stay in the run directory; the complete inference bundle is `bundle-<job-id>`.

## Manual environment and training commands

Use a new Python 3.12 environment, not a shared environment. CPU development:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_midway.txt
python -m pip install --no-deps -e .
```

For GPU training replace the wheel index with a compatible CUDA build, such as
`https://download.pytorch.org/whl/cu124`. Keep torch and torchvision versions
matched. The expected version pair is documented by
[PyTorch](https://pytorch.org/get-started/previous-versions/).

Prepare the encoder once, outside the GPU job:

```bash
export TORCH_HOME=/path/to/project/cache/torch
python -m road_refinement prepare --encoder-weights /path/to/project/cache/resnet18-imagenet.safetensors
python -m road_refinement check --data-root /projectnb/ec523kb/projects/RoadSeg/dataset --encoder-weights /path/to/project/cache/resnet18-imagenet.safetensors
```

Inside an allocated GPU job:

```bash
python -m road_refinement check --require-cuda
python -m road_refinement train \
  --stage both --device cuda \
  --encoder-weights /path/to/project/cache/resnet18-imagenet.safetensors \
  --run-dir /path/to/project/runs/my-first-run
```

Run baseline and refiner separately with `--stage baseline` and `--stage refiner`
using the same run directory. A refiner-only run can use an existing complete
baseline via `--baseline /path/to/baseline.safetensors`. The actual baseline used
by the refiner is saved in its run directory for export.

Training uses AdamW, masked BCE, and gradient clipping. Baseline learning rate
is 1e-3; refiner learning rate is 1e-4. Refiner K is sampled from 1, 2 and 4, with
equal supervision weights across the selected unroll. CUDA mixed precision is
enabled unless `--no-amp` is passed. The frozen encoder's BatchNorm statistics
remain fixed even while the decoder trains.

Use `--train-per-domain 0 --val-per-domain 0` for all allowed Train/Val tiles.
Full validation at every checkpoint can be expensive; the default deliberately
uses four validation tiles for the first submission. Crop size must be a
multiple of 32. Evaluation and inference use original-resolution tiles; there
is no internal image tiling. No labels are resized.

If CUDA is requested but unavailable, the command fails instead of silently
training on CPU. `--device cpu --workers 0` is available for small local checks.
`--synthetic` is only for testing without pretrained weights: it marks the run
and exported bundle as synthetic and initializes the encoder randomly. Do not
use that flag for the course-data training run.

## Checkpoints and resume

The run directory contains:

```text
config.json                       # architecture, initial targets, seed, selected tiles, source/environment
invocations.jsonl                 # requested targets/options for each invocation, including resume
encoder.json                      # pretrained weight source/checksum
refiner_baseline.json              # exact baseline checkpoint used
metrics.jsonl                     # loss, validation, timing and training memory
baseline_validation.json
refiner_validation.json
checkpoints/
  baseline_latest.safetensors
  baseline_best.safetensors
  refiner_latest.safetensors
  refiner_best.safetensors
  refiner_baseline.safetensors
  baseline_trainer.pt              # private optimizer/RNG state; not submitted
  refiner_trainer.pt
```

Best means highest pooled validation road IoU among saved checkpoints, including
poor results. Latest optimizer states allow continuation after interruption.
Resume uses deterministic crop indices and restores optimizer/scaler/RNG state.
Keep the same source, data selection, crop size, batch size and seed. Increasing
the update target is allowed. Only load trainer files created by your own run;
the inference bundle contains safetensors weights instead of these trainer files.

```bash
python -m road_refinement train --stage both --device cuda \
  --run-dir /path/to/existing/run \
  --encoder-weights /path/to/project/cache/resnet18-imagenet.safetensors --resume
```

For scheduler resume, set `SCC_RUN_DIR` in your local config to the existing run
and run `bash scripts/scc_submit.sh resume`. Saved crop/subset settings are read
automatically, as are the original update targets. Set
`SCC_RESUME_BASELINE_UPDATES` and/or `SCC_RESUME_REFINER_UPDATES` explicitly to
extend training. If extending baseline training changes its best checkpoint,
start a new refiner run against that new baseline; a previously trained refiner
cannot silently switch its frozen baseline.

## Export, inference and local scoring

```bash
python -m road_refinement export --run-dir /path/to/run --output /path/to/new-bundle
python -m road_refinement predict --bundle /path/to/new-bundle \
  --input /path/to/image.png --output /path/to/new-results --steps 4 --device cpu
python -m road_refinement predict --bundle /path/to/new-bundle \
  --input /path/to/images-only-folder --output /path/to/new-folder-results --steps 4 --device cuda --save-trace
```

All inference weights load from the bundle. It includes source and installation
instructions so staff can run it without the original checkout. You can archive
the complete bundle directory for submission; include both weight files.
Packages must already be installed for fully offline execution. Inference does
not need the original ImageNet download, training data, masks, or SCC paths.

Folders are searched recursively for PNG/JPEG images. Do **not** pass the dataset
root, which also contains label PNGs; pass an images-only directory. Relative
IDs are preserved and duplicate IDs fail validation. An input file gets its own
filename stem as its ID. Output directories must be new or empty.

Outputs are `probabilities/<id>.npy` (float32 H×W), `masks/<id>.png` (0/255,
threshold >=0.5), `run.json`, and optional `trace/k0` through `trace/kK` arrays.
Images are padded once to a multiple of 32 and cropped back to their original
dimensions. K=0 returns the baseline unchanged; K=1..8 calls the same core
exactly K times. Prediction state and previous logits both feed the next update.

```bash
python -m road_refinement evaluate --bundle /path/to/new-bundle \
  --data-root /projectnb/ec523kb/projects/RoadSeg/dataset \
  --val-per-domain 2 --output /path/to/local-score.json --device cuda
```

This reports ordinary K=0,1,2,4 and stress K=4, pooled Road IoU/clDice/Boundary
IoU, domain/corruption breakdowns, and the handbook's four-component score. The
stress assignment and metrics call the original course implementations. Small
validation subsets need not contain all three stress families. Reports clearly
say `official_baseline: false`; this does not claim eligibility for final ranking.

## Checks before handing it in

```bash
python -m unittest discover -s project_a/cli/tests -v
python -m unittest discover -s tests -v
python project_a/cli/smoke_test.py --output /tmp/ec523-course-smoke
```

The new tests cover image/mask alignment, ignored pixels, frozen normalization,
tokenizer/core gradients, exact recurrence and prefixes, resume equivalence,
single-image/folder inference, and an exported bundle executed in a separate
process with network connections blocked. They use synthetic fixtures, not course
performance results. GPU training and the real dataset must still pass the SCC
smoke job. See the actual validation log in `docs/midway_validation.md` for checks
performed during implementation.

The team owns its report, research claims, and interpretation. Use these tools
to collect honest initial results and document the temporary baseline, training
subset, update counts, and code assistance.
