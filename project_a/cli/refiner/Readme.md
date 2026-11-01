# `refiner/`: the looped refiner Rθ

This folder holds the trainable part of the project: the purple, dashed **Rθ** box in the workflow diagram. It takes the frozen baseline's guess `y0` and improves it over K steps.

```
image x ─▶ Encoder E (frozen) ─▶ f ─┐
                                    ├─▶ Rθ (this folder), K steps ─▶ yK ─▶ σ ─▶ threshold 0.5 ─▶ road mask
image x ─▶ E + head H0 (frozen) ─▶ y0 ─┘
```

## Files

| File | Class | Diagram box | Job |
|---|---|---|---|
| `refiner.py` | `Refiner` | the whole Rθ box | Runs the K-step loop and wires every other piece together |
| `memory.py` | `Initializer` | Initializer Iφ | Builds the starting memory `z0` from `f` and `y0`, once, before the loop |
| `head.py` | `Head` | Output head | Turns answer tokens back into a full-size update for the mask |
| `__init__.py` | | | Empty; makes this folder importable as `refiner` |

### `refiner.py`: `Refiner(c_in, d=256, patch=16, nz=32, nhead=8, num_layers=4)`

`forward(f, y0, K, return_all=False)` runs:

```python
img = Tx(f)                                   # image tokens, computed once
zt  = Initializer(img, Ty(y0))                # z0
yt  = y0
for each of K steps:
    ans = Ty(yt)                              # answer tokens from the current mask
    out = Core(cat[img, ans, zt])             # ONE core call per step
    _, ans_out, zt = split(out)               # zt+1 is the new memory
    yt  = yt + Head(ans_out)                  # yt+1, fed back as the next yt
return yt                                     # yK  (or every step's yt if return_all)
```

Rules it follows (the project's interface tests check these):

- **K = 0** returns `y0` unchanged, with zero core calls.
- **Exactly K core calls** for K steps.
- **Feedback**: each step's `yt+1` and `zt+1` are the next step's inputs.
- **Prefix consistency**: in `eval()` mode, the first 2 steps of a K=4 run equal a K=2 run.
- **Grid check**: `Tx(f)` and `Ty(yt)` must give the same number of tokens, so `patch` must equal `H / h` (image size ÷ encoder grid size). The assert explains the fix if it fails.

### `memory.py`: `Initializer(d=256, nz=32, nhead=8)`

`nz` learned "memory slot" vectors read the image tokens and the starting answer tokens through cross-attention, and the result is `z0` with shape `(B, nz, d)`. After that, the memory is updated by the core each step: the last `nz` tokens of the core's output become `zt+1`. The memory lets the refiner carry information from one step to the next beyond the mask itself.

### `head.py`: `Head(d=256, patch=16)`

`forward(tokens, h, w, out_size)` maps each answer token `(d,)` to a `patch × patch` block of pixels, then rearranges the blocks into a full map (`pixel_shuffle`): `(B, N, d)` → `(B, 1, H, W)`. This is the reverse of `Ty`'s patch conv.

Its output is a **correction** that the refiner adds to `yt`. The final layer starts at zero, so an untrained refiner returns `y0` unchanged and training starts from the baseline instead of from noise.

## Shapes (example: `d` = 256, `patch` = 8, `nz` = 32, 64×80 images)

| Tensor | Shape | Example |
|---|---|---|
| `f` (from the frozen encoder) | `(B, c_in, h, w)` | `(2, 64, 8, 10)` |
| `y0`, `yt`, `yK` (logits) | `(B, 1, H, W)` | `(2, 1, 64, 80)` |
| image / answer tokens | `(B, N, d)`, `N = h·w` | `(2, 80, 256)` |
| memory `zt` | `(B, nz, d)` | `(2, 32, 256)` |
| core input / output | `(B, 2N + nz, d)` | `(2, 192, 256)` |

## How it connects to the rest of the project

| Uses | From | For |
|---|---|---|
| `Tx`, `Ty`, `pos2d` | `tokenizer/` | Turning `f` and `yt` into tokens |
| `Core` | `transformer/` | The per-step attention |
| `f`, `y0` | `baseline.py` (frozen E and H0) | The refiner's inputs |

| Used by | How |
|---|---|
| `train.py` | Trains `Refiner` against ground truth `g`; the baseline stays frozen |
| `submission/model.py` | Exposes `Refiner` in the form the course interface requires (still to be wired) |
| `predict.py` → `evaluate.py` → `score.py` | Run the submitted model, then measure IoU, clDice and BIoU |

## Test

From the `cli` folder:

```bash
python -m refiner.memory     # z0: (2, 32, 256)
python -m refiner.head       # dy: (2, 1, 64, 80)
python -m refiner.refiner    # K=0 identity True, prefix K2==K4 True
```
