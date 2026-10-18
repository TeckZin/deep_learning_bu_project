# EC523 Project A: submission CLI and reference evaluator

This is an instructor-review starter kit. It includes a working exact-K runner,
an untrained recurrent-transformer example, synthetic smoke-test data generation,
the reference metric implementation, and a disclosed RGB stress generator.
It does **not** include a trained course-data baseline, training pipeline, final data
split, leaderboard website, or a validated L40S compute budget.

## Setup

Use Python 3.12 and a fresh virtual environment. From this `cli` directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python smoke_test.py --output smoke_outputs
```

The pinned versions were tested on CPU. On the cluster, staff must supply a
CUDA-compatible PyTorch environment and validate the same protocol before
release. Never install over a shared cluster environment. Smoke outputs are
synthetic and untrained; their numerical scores are not scientific results.
No network is used by the runner, tests, generator, or evaluator after setup.

## Files students change versus files staff own

Students change `submission/`, provide their training scripts/configuration,
and export `submission/model.safetensors`. All parameters must be constructed
and registered before loading. The checkpoint is loaded with safetensors;
no Python pickle is accepted. Cite any reused code and explain every change.

Staff own `baseline.py`, `predict.py`, `common.py`, `metrics.py`, `evaluate.py`,
`score.py`, and `make_stress.py`. Staff replace only the official-baseline hook
before release, then freeze the runner and evaluation environment. The demo
baseline is a brightness heuristic, not a learned segmentation model.

The baseline adapter returns `(features, initial_logits)` and declares
`feature_channels`. The included demo features are `[B,5,h,w]`: coarse RGB
and two positional coordinates. Real features are a staff release decision.
The adapter must normalize RGB as required by its checkpoint, keep weights
and normalization statistics frozen, and return original-resolution logits.

Student interface:

```python
model = build_refiner(feature_channels)
state = model.initialize(features, initial_logits)
state, next_logits = model.step(features, state, previous_logits)
```

In handout Section 3, `state` is **z_t**, a latent token bank; `previous_logits`
is **y_t**, the dense current road-logit map. `initial_logits` is y_0 and
`next_logits` is y_(t+1). Ground truth is g and training loss is L. We retain
the descriptive `*_logits` Python names to make the interface unambiguous.
The example state is [B,N,d], with one d-dimensional token per coarse spatial
location. Image and prediction features are embedded/fused before the shared
core; the output head decodes back to [B,1,H,W] logits. A logit map is not
itself a bank of d-dimensional transformer tokens.

`model.core` must be the same registered transformer module at every update.
The staff runner owns the outer loop and never passes the requested total K
to the student interface. Features are copied before a student call so in-place
editing cannot accidentally contaminate the next step. State must carry the
previous computation. Returned logits have shape `[B,1,H,W]`, matching input.
Output must depend on the loop, not on an independent segmentation bypass.

The example contains one transformer layer, mask feedback, and recurrent state.
Its random weights are solely an interface demonstration. It is not a baseline
that students should copy unchanged and present as their contribution.

## Inference

After staff release the official baseline:

```sh
python predict.py --checkpoint submission/model.safetensors \
  --baseline staff/baseline.safetensors --input evaluation/images \
  --output predictions/ordinary4 --steps 4 --device cuda --save-trace
```

Before that, smoke-test the contract:

```sh
python make_demo.py --output demo
python predict.py --demo --checkpoint demo/model.safetensors \
  --input demo/images --output predictions/demo4 --steps 4 --save-trace
```

Supported exact step counts are integers 0 through 8; 1, 2, and 4 are scored.
At K=0 the runner returns the staff baseline directly. At other K, it enforces
one call to `model.core` per update and checks that parameter identity/storage
does not change. It saves all steps on CPU; this is intentional for audit, not
a latency-optimized implementation. Memory-heavy full-resolution attention is
the student's responsibility to avoid. Core size is fixed independently of K.

Diagnostic options `--ablation reset-state` and `--ablation no-mask-feedback`
reset the hidden state at each step or always feed the initial logits,
respectively. They are rejected by the final score combiner. In the residual
example, replacing previous logits also changes the residual path; discuss
this confound and use retrained controls for stronger claims.

Input is a directory of RGB PNG/JPEG images, recursively searched. A path
`images/urban/001.png` becomes ID `urban/001`. Duplicate stems, missing images,
nonfinite/invalid outputs, and nonempty output directories are rejected.
Every output directory has:

```
probabilities/urban/001.npy   # float32 HxW in [0,1]
masks/urban/001.png           # uint8 0/255, threshold >= 0.5
trace/k0/urban/001.npy        # if --save-trace; all 0..K
run.json                     # hashes, core calls, counts, diagnostic timing
```

The evaluator reads probabilities, not visualization PNGs. Official evaluation
does not accept custom per-image thresholds, retraining between K, extra
models, ensembling, or test-time augmentation. For an internal tiling proposal,
obtain approval before training; count all tiles in time/compute. Arbitrary
tiling per K is not allowed. The current runner processes one full image at a
time; it does not secretly resize images to save memory.

## Evaluation manifest

The manifest is a nonempty JSON list. Image and label paths are resolved
relative to the manifest. IDs must exactly match the inference directory.
Supplied course labels: road=3, ignore=0, other IDs 1,2,4,5,6,7 are non-road.
Do not use already remapped labels or RGB color masks.

```json
[
  {"id": "urban/001", "image": "images/urban/001.png",
   "label": "labels/urban/001.png", "domain": "urban", "condition": "ordinary"},
  {"id": "rural/002", "image": "images/rural/002.png",
   "label": "labels/rural/002.png", "domain": "rural", "condition": "ordinary"}
]
```

One manifest has one condition. Staff run inference with only the input image
directory mounted; labels and evaluation manifests are not visible to student
code. The evaluator runs separately with labels. Download only the versioned
data package supplied by course staff and use its split manifests. Do not
search for matching external images or annotations, reconstruct evaluation
labels from outside sources, or train/tune on course holdouts.

```sh
python evaluate.py --manifest evaluation/ordinary.json \
  --predictions predictions/ordinary4 --output reports/ordinary4.json
python make_stress.py --manifest evaluation/ordinary.json \
  --output evaluation/stress --seed 523
```

The stress generator never reads labels to choose locations or categories.
It preserves source IDs/domains/label paths, balances three corruption types
within domain, and saves the exact corruption parameters. Its generated label
paths are absolute for local evaluation; regenerate manifests after moving
data to another machine. Do not mount these manifests into the inference job.
For private evaluation, staff keep generated images/seeds private until use.

## Four-component score

Run the same checkpoint on ordinary images at K=1,2,4 and stress images at K=4.
Evaluate each run, then:

```sh
python score.py --ordinary-1 reports/ordinary1.json \
  --ordinary-2 reports/ordinary2.json --ordinary-4 reports/ordinary4.json \
  --stress-4 reports/stress4.json --output reports/score.json
```

The combiner verifies matching model/baseline/source hashes and source tiles,
correct K/condition, no ablation, and no demo flag. Demo runs require the
explicit `--allow-demo` option and stay marked demo. Staff recompute reports;
student-authored JSON is not trusted evidence.

For any set D and K:

`A = 0.50 RoadIoU + 0.30 clDice + 0.20 BoundaryIoU`

`Score = 100*(0.20 Aordinary1 + 0.20 Aordinary2 + 0.40 Aordinary4 + 0.20 Astress4)`

Counts are pooled across all images before ratios. Domains are reported
separately but are not equally reweighted in the overall score. Boundary IoU
uses three 3x3 erosions and excludes the image frame and ignored-pixel margins.
Zhang skeletonization is pinned to scikit-image 0.25.2. See the handout for
equations and all empty-set conventions. The fixed hard metric is different
from a soft-clDice training loss. `tests/test_metrics.py` covers basic
analytical examples, empty/ignored masks, exact threshold, boundary width,
input validation, and the numerical leaderboard example.

## Limits of this kit and staff security responsibilities

No GPU benchmark, model training, private evaluation service, or website has
been implemented here. Runtime hooks catch common implementation mistakes;
they cannot enforce honesty in arbitrary Python code. Use a disposable,
offline container, nonprivileged user, read-only image/weight mounts, writable
output only, memory/CPU/GPU/time limits, no secrets, and source review.
Deterministic settings may reject some GPU operators; staff must test the
release environment and document any approved numerical tolerance.

`run.json` timing includes transfer of every step to CPU and has no warmup.
It is diagnostic, not an official efficiency result. Official profiling must
warm up, synchronize CUDA, run batch size one on identical hardware/input
policy, and report end-to-end time including baseline and all tiles. Measure
training peak memory separately; inference memory does not establish training
feasibility. Check K-prefix consistency, freezing of the real baseline,
parameter sharing, mask/state dependence, and bypasses during code review.
