# `transformer/`: the shared transformer core

This folder holds the **shared transformer core**, the central box of the refiner Rθ in the workflow diagram. It's where the refiner's reasoning happens each step: every token can look at every other token.

```
Tx(f)   image tokens  ─┐
Ty(yt)  answer tokens ─┼─ cat ─▶ Core ─▶ split ─▶ answer tokens ─▶ Head ─▶ yt+1
zt      memory tokens ─┘                    └──▶ memory tokens ─▶ zt+1
```

## Files

| File | Class | Job |
|---|---|---|
| `core.py` | `Core` | A stack of pre-norm transformer encoder layers plus a final LayerNorm |
| `__init__.py` | | Empty; makes this folder importable as `transformer` |

### `core.py`: `Core(d=256, nhead=8, num_layers=4, dim_ff=4·d, dropout=0.1)`

Built from PyTorch's own `nn.TransformerEncoderLayer` and `nn.TransformerEncoder`; no extra library needed.

| Setting | Value | Why |
|---|---|---|
| `batch_first=True` | tokens are `(B, tokens, d)` | Matches `Tx`, `Ty` and the memory |
| `norm_first=True` | pre-norm | More stable when the same layers run K times in a row |
| `activation="gelu"` | GELU | Standard in modern transformers |
| `d` | 256 | Must equal the `d` used by `Tx`, `Ty`, `Initializer` and `Head` |
| `nhead` | 8 | `d` must be divisible by it (here 32 numbers per head) |

`forward(tokens)` takes `(B, T, d)` and returns `(B, T, d)`: the same shape, with every token updated. It doesn't know which tokens are image, answer or memory; the refiner joins them before the call and splits them after.

## What "shared" means

The refiner creates **one** `Core` and calls it once per step, so all K steps use the same weights. The refiner is one block applied repeatedly, not K different blocks. The interface tests count these calls: K steps must give exactly K core calls.

## Cost to watch

Each layer compares every token with every other one, so memory and time grow with `T²`, where `T = 2N + nz` and `N = h·w` is the encoder's grid size. This is paid K times per image.

| Encoder grid | `T` (with `nz` = 32) | Full attention |
|---|---|---|
| 8×10 | 192 | Easy |
| 16×16 | 544 | Fine |
| 32×32 | 2,080 | Heavy, times K |
| 64×64 | 8,224 | Too big: pool `f` and raise `patch` |

## Parameters (`d` = 256, `dim_ff` = 1024)

| Part | Count |
|---|---|
| One encoder layer (attention 263,168 + feed-forward 525,568 + two LayerNorms 1,024) | 789,760 |
| 4 layers | 3,159,040 |
| Final LayerNorm | 512 |
| **`Core` total** | **3,159,552** |

## How it connects to the rest of the project

| Direction | File | Relationship |
|---|---|---|
| Used by | `refiner/refiner.py` | Calls `Core` once per step on `cat[Tx(f), Ty(yt), zt]` |
| Receives tokens from | `tokenizer/Tx.py`, `tokenizer/Ty.py`, `refiner/memory.py` | All must use the same `d` |
| Sends tokens to | `refiner/head.py` (answer part), next step's memory (memory part) | Split by the refiner |

`Core` is unrelated to `encoder/` (Encoder E). That one is the frozen backbone that makes `f` from the image, before any tokens exist.

## Test

From the `cli` folder:

```bash
python -m transformer.core   # input (2, 192, 256) → output (2, 192, 256)
```
