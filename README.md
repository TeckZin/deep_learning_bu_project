# BU Deep Learning Project #1

# EC523 Project A — Team To-Do List

![Looped refiner workflow](docs/workflow.svg)
![Transfoermer core](docs/transformer_core.png)

## Setup & Unblocking
- [ ] Ask staff: release of trained baseline + data package, feature contract (C, grid stride), tile policy, final resource caps, GPU access
- [ ] Set up Python 3.12 env, install `cli/requirements.txt`
- [ ] Run `python -m unittest discover -s tests -v` and `python smoke_test.py --output smoke_outputs`
- [ ] Every member reads and can explain the example `submission/model.py`
- [ ] Read the three papers: Universal Transformers, TRM (Less is More), Recurrent Scene Parsing
- [ ] Start a contribution log; assign a primary owner + reviewer for each major piece

## Data & Baseline
- [ ] Download the official course data package; verify checksums and split manifests
- [ ] Verify raw labels (3 = road, 0 = ignore, 1/2/4–7 = non-road); no double remapping
- [ ] Keep all crops from a tile in that tile's split
- [ ] Load the frozen staff baseline; confirm weights + normalization stats stay frozen
- [ ] Reproduce the baseline (K = 0) on development data
- [ ] Save examples of baseline successes and failures

## Model (`submission/model.py`)
- [ ] `build_refiner(feature_channels)` — build all modules once; expose shared transformer as `model.core`
- [ ] Tx: features → image tokens (read C and grid size at run time, LayerNorm, position embeddings)
- [ ] Ty: logits → answer tokens on the same grid as Tx
- [ ] Shared transformer core (≤ 5M params; ≤ 10M total trainable)
- [ ] Output head: tokens → `[B,1,H,W]` logits (upsampling + detail recovery convs allowed)
- [ ] `initialize(f, y0)` → z0, with no core call
- [ ] `step(f, z, y)` → (z_next, y_next), exactly one core call, no labels or total K
- [ ] No independent CNN/U-Net segmentation bypass
- [ ] Passes `predict.py` checks at K = 0 through 8
- [ ] Prefix consistency: steps 0–2 of a K = 4 run match a K = 2 run
- [ ] Export weights to `submission/model.safetensors`

## Training Pipeline
- [ ] Crop-based data loader (no full-resolution global attention)
- [ ] Run frozen baseline to get f and y0 (optionally cache features)
- [ ] Unroll K steps; BCE on logits with ignore mask; skip all-ignore crops
- [ ] Intermediate supervision with weights w_t (sum to 1)
- [ ] Train with mixed K so one checkpoint works at every K
- [ ] Mixed precision / grad accumulation / activation checkpointing as needed
- [ ] Overfit a tiny subset first to verify gradients, ignore mask, and feedback
- [ ] Log peak training memory (target ≤ 32 GiB) and GPU-hours
- [ ] Select checkpoints on development data only — never evaluation labels

## Evaluation
- [ ] Run `evaluate.py` on dev data for every K
- [ ] Generate stress set with `make_stress.py`; evaluate at K = 4
- [ ] Compute the four-component leaderboard score with `score.py`
- [ ] Report urban / rural and per-corruption breakdowns
- [ ] Measure latency, inference memory, parameter counts (core vs total)

## Required Experiments
- [ ] **1. Iteration curve:** baseline + same checkpoint at K = 1, 2, 4, 6, 8; plot each metric vs K and vs time; state which K were trained
- [ ] **2. State/feedback interventions:** `--ablation reset-state` and `--ablation no-mask-feedback`; plus a retrained no-feedback control for strong claims
- [ ] **3. Non-recurrent control:** one-step model at matched parameters; untied stack at matched compute if feasible; report mismatches
- [ ] **4. Design ablation:** one hypothesis-driven change, with predicted outcome written before running
- [ ] **5. Failures & uncertainty:** unselected successes/failures (false connections, vanishing roads, later-step degradation); 3 seeds or paired bootstrap intervals

## Final Submission
- [ ] One nominated checkpoint + source code
- [ ] Pinned environment and training instructions
- [ ] Run manifest linking results to seeds and checkpoints
- [ ] Fresh offline environment runs the exact `predict.py` command at all K, no downloads, no labels
- [ ] Method diagram and parameter table
- [ ] K-sweep plots and latency/memory measurements
- [ ] All required controls and representative intermediate masks + failures

## Report & Presentation
- [ ] Specific, falsifiable research claim
- [ ] Method, fair controls, ablations, uncertainty, error analysis, limitations
- [ ] Report written by the team (AI only for grammar); course template and page limit
- [ ] Citations for code, data, figures, ideas; AI/code-use disclosure
- [ ] Contribution statement
- [ ] Slides in course template (no AI for the presentation)
- [ ] Every member can explain one update, state vs parameters, tensor shapes, and what state-reset / feedback removal should change


