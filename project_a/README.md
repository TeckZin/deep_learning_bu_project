# EC523 Project A: Road Segmentation with a Recurrent Refiner

Inference code for our Project A submission. Given a folder of RGB images, it
predicts a road mask for each one (1 = road, 0 = non-road) at the same height
and width as the input image.



![workflwo](docs/workflow.svg)

## How it works

1. **Baseline (frozen, staff-provided)**: produces image features and an initial
   road-mask guess `y_0`.
2. **Refiner (ours, `submission/model.py`)**: improves the mask over K rounds.
   Each round:
   - **Tx** turns the baseline features into tokens (with 2D position encoding).
   - **Ty** turns the current mask `y_t` into tokens (probability + uncertainty).
   - **Encoder** fuses the state `z_t` with both token sets.
   - **Core** (one shared transformer, reused every round) updates the state to `z_(t+1)`.
   - **Head** decodes a full-resolution correction: `y_(t+1) = y_t + correction`.
3. `predict.py` (staff runner) saves the mask after exactly K rounds.

## Folder contents

```
predict.py            Inference CLI (staff runner): folder in, masks out
baseline.py           Loads the frozen baseline
common.py             Shared helpers (hashing, output folder checks, JSON)
requirements.txt      Python dependencies
submission/model.py   Our refiner: build_refiner(), initialize(), step()
weights/
  refiner.safetensors   Our trained refiner checkpoint (the only one)
  baseline.safetensors  Frozen baseline checkpoint (staff-provided)
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From this folder (all paths are relative):

```bash
python predict.py \
  --checkpoint weights/refiner.safetensors \
  --baseline   weights/baseline.safetensors \
  --input      path/to/images \
  --output     results_k4 \
  --steps      4 \
  --device     cuda
```

- `--input`: folder of `.png` / `.jpg` / `.jpeg` images (subfolders are searched).
- `--output`: must be a new or empty folder. We put K in the name (`results_k4`).
- `--steps`: number of refinement rounds K (0 to 8). **Our final results use K = 4.**
- `--device`: `cuda` or `cpu`.
- `--save-trace` (optional): also saves the mask after every round.

## Output

```
results_k4/
  masks/<image_id>.png           Final binary mask after K rounds, same H x W as input
  probabilities/<image_id>.npy   Road probability per pixel (float32, 0 to 1)
  trace/k0 ... kK/               Only with --save-trace: mask after each round (k0 = baseline)
  run.json                       Run record: K, seed, device, checkpoint hashes, timing
```

**Which round is which:** the folder name and `run.json` (`"steps"`) give K.
`masks/` and `probabilities/` are always the result after round K.
With `--save-trace`, `trace/k<i>/` is the result after round i.

**Mask values:** `masks/*.png` are written by the staff runner as 0 (non-road)
and 255 (road). To get 0/1, divide by 255, or threshold `probabilities/*.npy` at 0.5.

## Hardware

Runs on one GPU (well under 48 GB) or on CPU. Images are processed one at a time
and attention runs on at most a 32x32 token grid, so memory stays small even for
large images.

## Quick check without real data

This uses synthetic images, an untrained model and a fake baseline. It only checks
that the code runs:

```bash
python make_demo.py --output demo
python -c "
from baseline import load_baseline
from submission.model import build_refiner
from safetensors.torch import save_file
base = load_baseline(None, 'cpu', demo=True)
save_file(build_refiner(base.feature_channels).state_dict(), 'demo_ckpt.safetensors')
"
python predict.py --demo --checkpoint demo_ckpt.safetensors --input demo --output demo_out --steps 3
```

Expected output: `Saved N images after exactly 3 steps to demo_out`.

## Data

Course staff supply the data package and official splits. This submission does not
include any data, labels, or our own prediction results.

# EC523 Project A: student materials

- `handout/project_a.tex`: editable LaTeX handout, with vector diagrams,
  equations, bibliography, design choices, and grading expectations.
- `handout/assets/course_imagery.png`: attributed imagery/annotation figure.
- `cli/`: submission runner, untrained example, evaluator, stress generator,
  synthetic demo generator, tests, and usage instructions.

Course staff will supply the versioned data package and official split
manifests. Do not obtain or reconstruct evaluation labels from other sources.
The source credit for the borrowed figure remains in the handout; staff will
provide full research-provenance instructions before any paper submission.

Compile from the `handout` directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error project_a.tex
```

The bibliography is embedded in the TeX source, so no BibTeX step is needed.
The PDF was built with TeX Live and checked by rendering every page. See
`VALIDATION.md` for the actual checks performed and untested items.

The separate instructor package contains release decisions, data provenance,
and hosting notes. Those notes are not part of the student distribution.

This is a review draft: the trained frozen baseline and final course resource
allocation are intentionally not invented. The source `.tex` and the full
bundle should travel with the PDF so figures and code remain reproducible.
