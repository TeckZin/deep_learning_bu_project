# attempt7: recurrent refiner on a DINOv2 stand-in baseline

DEVELOPMENT ONLY. The baseline here is our own stand-in (frozen DINOv2 ViT-S/14 + a small
H0 head we trained), not the staff baseline. These refiner weights only work on top of
that stand-in and are not a leaderboard entry.

## Files

| File | What it is |
|---|---|
| `submission/model.py`, `core.py`, `parts.py` | Refiner code (shared transformer core, tokenizer, head) |
| `submission/model.safetensors` | Refiner weights, same file as `weights/attempt7_best.safetensors` |
| `weights/attempt7.safetensors` | Refiner weights after the last epoch (epoch 40; identical to best in this run) |
| `weights/h0_attempt6_best.safetensors` | H0 head weights for the stand-in baseline |
| `train_h0.py`, `train_h0_job_attempt6.sh` | Trains the H0 head |
| `train.py`, `train_job_attempt7.sh` | Trains the refiner |
| `standin_baseline.py` | Builds the stand-in baseline (DINOv2 encoder + H0 head) |
| `environment.yml` | Conda environment |
| `logs/attempt7.*.log` | Training log with the validation results below |

The DINOv2 encoder weights are NOT in this folder. They are the public
`facebookresearch/dinov2` `dinov2_vits14` weights, loaded through `torch.hub` and kept frozen.
Download them once on a machine with internet (the login node):

    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"

## Loading the weights

    import torch
    from safetensors.torch import load_file
    from standin_baseline import load_standin
    from submission.model import build_refiner

    device = torch.device("cuda")
    base = load_standin("weights/h0_attempt6_best.safetensors", device).eval()   # DINOv2 + H0, frozen
    model = build_refiner(base.feature_channels).to(device).eval()
    model.load_state_dict(load_file("submission/model.safetensors"))

    # image: float32 RGB / 255, shape [1, 3, H, W]
    with torch.no_grad():
        features, y = base(image)                 # y = y0, the baseline logits (K = 0)
        state = model.initialize(features, y)
        for _ in range(4):                        # K refinement steps
            state, y = model.step(features, state, y)
    prob = y.sigmoid()                            # road probability, threshold 0.5

## Reproducing the run

Run from the `cli` folder on the BU SCC. Seed 0 (the default) for both jobs.

    mkdir -p logs
    qsub train_h0_job_attempt6.sh                          # H0 on half A of Train (--part a)
    qsub -hold_jid h0_attempt6 train_job_attempt7.sh       # refiner on half B (--part b)

Refiner settings: 40 epochs, 4 unrolled steps, batch 8, crop 512, lr 3e-4, warmup 500,
edge weight 4, clDice weight 0.5, stress-aug 0.3, y0-aug 0.5, detach-y, validation only
before and after training.

## Results (val_small, 100 images; val_small_stress, 100 images)

Quality = 0.5 IoU + 0.3 clDice + 0.2 Boundary IoU.

| | K=0 | K=1 | K=2 | K=4 | stress K=4 |
|---|---|---|---|---|---|
| Before training | 52.96 | 53.17 | 53.06 | 52.38 | 52.05 |
| After epoch 40 | 52.96 | 53.96 | 54.03 | 54.01 | 53.72 |

At K=4 after training: IoU 63.38, clDice 71.20, Boundary IoU 4.78. Leaderboard-style score 53.94.

Known limits: the gain over K=0 comes from clDice; Boundary IoU is lower than the baseline's,
and K=2 and K=4 add almost nothing over K=1. With 100 validation images, differences under
about 1 point are within noise.

## Parameters

Refiner 3,944,449 total, core 3,159,552, measured on attempt6. attempt7 adds a small number
of parameters in `parts.py`; the exact count is printed at the top of `logs/attempt7.*.log`.

## Assistance disclosure

Parts of the training scripts and of `submission/parts.py` were written with AI assistance (Claude).
