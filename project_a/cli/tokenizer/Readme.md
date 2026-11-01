# `Tx` and `Ty`: tensor shapes, step by step

Both are **tokenizers** for the looped refiner `Rθ`. They turn 2D grids into token sequences on the **same h×w grid**, so the shared transformer core can line them up:

- `Tx(f)` turns the frozen encoder's features `f` into **image tokens** (computed from the same `f` every step).
- `Ty(yt)` turns the current answer mask `yt` into **answer tokens** (changes every step as `yt` is refined).

```
f  (B, C, h, w) ──Tx──▶ image tokens  (B, N, d) ─┐
yt (B, 1, H, W) ──Ty──▶ answer tokens (B, N, d) ─┼─ cat ─▶ (B, 2N + Nz, d) ─▶ shared transformer core
zt (B, Nz, d)   ─────── memory tokens (B, Nz, d) ─┘
```

Attention happens only in the core, not inside `Tx` or `Ty`.

## Notation

| Symbol | Meaning | Example |
|---|---|---|
| `B` | batch size | 2 |
| `C` (`c_in`) | encoder output channels | 64 |
| `H`, `W` | full image / mask size | 64, 80 |
| `h`, `w` | encoder feature-map size | 8, 10 |
| `patch` | `H / h` = `W / w`, the encoder's downsampling factor | 8 |
| `d` | token size; divisible by 4 (and by the core's `nhead`) | 256 |
| `N = h * w` | tokens per grid (one per grid position) | 80 |
| `Nz` | number of memory tokens | e.g. 32 |

The rule that ties them together: **`patch` must equal `H / h`**, so that `Ty` produces the same `N` tokens on the same grid as `Tx`.

---

## `Tx.forward(f)`: image tokens

| Step | Code | Shape | Example |
|---|---|---|---|
| Input | `f` | `(B, C, h, w)` | `(2, 64, 8, 10)` |
| 1. Project channels | `self.proj(f)` (1×1 conv, `C` → `d`) | `(B, d, h, w)` | `(2, 256, 8, 10)` |
| 2. Flatten the grid | `x.flatten(2)` | `(B, d, h*w)` | `(2, 256, 80)` |
| 3. Tokens first | `.transpose(1, 2)` | `(B, N, d)` | `(2, 80, 256)` |
| 4. Add positions | `x + pos2d(h, w, d)` | `(B, N, d)` | `(2, 80, 256)` |
| 5. Normalize | `self.norm(x)` (LayerNorm over `d`) | `(B, N, d)` | `(2, 80, 256)` |
| Output | `return x` | `(B, N, d)` | `(2, 80, 256)` |

- **Step 1** changes only the channel count; a 1×1 conv mixes channels at each pixel and keeps `h`, `w`.
- **Step 2** merges `h` and `w` into one axis of length `h*w`, row by row (row-major).
- **Step 3** makes each grid position one token of size `d`: the `(batch, tokens, features)` layout transformers expect.
- **Step 4** broadcasts the `(N, d)` position codes over the batch: `(N, d)` → `(1, N, d)` → added to `(B, N, d)`.
- **Step 5** normalizes each token's `d` numbers on its own.

Since `f` is frozen and the same every step, `Tx(f)` can be computed once and reused for all K steps.

## `Ty.forward(yt)`: answer tokens

| Step | Code | Shape | Example |
|---|---|---|---|
| Input | `yt` | `(B, 1, H, W)` | `(2, 1, 64, 80)` |
| 1. Patchify | `self.proj(y)` (conv, kernel = stride = `patch`, 1 → `d`) | `(B, d, H/patch, W/patch)` = `(B, d, h, w)` | `(2, 256, 8, 10)` |
| 2. Flatten the grid | `x.flatten(2)` | `(B, d, h*w)` | `(2, 256, 80)` |
| 3. Tokens first | `.transpose(1, 2)` | `(B, N, d)` | `(2, 80, 256)` |
| 4. Add positions | `x + pos2d(h, w, d)` | `(B, N, d)` | `(2, 80, 256)` |
| 5. Normalize | `self.norm(x)` | `(B, N, d)` | `(2, 80, 256)` |
| Output | `return x` | `(B, N, d)` | `(2, 80, 256)` |

- **Step 1** is the only difference from `Tx`. The full-resolution mask is cut into non-overlapping `patch × patch` squares, and each square (`patch²` pixels) becomes one `d`-number token. Output size by the conv formula: `(64 − 8)/8 + 1 = 8` and `(80 − 8)/8 + 1 = 10`.
- **Steps 2–5** are identical to `Tx`, including the **same `pos2d` codes**, so image token `i` and answer token `i` describe the same location.

## `pos2d(h, w, d)`

Gives every grid position `(row y, column x)` a unique `d`-number code. Shared by `Tx` and `Ty`.

| Step | Code | Shape | Example |
|---|---|---|---|
| Row index of each token | `arange(h).repeat_interleave(w)` | `(N,)` | `(80,)`: 0,0,…,0,1,1,… |
| Column index of each token | `arange(w).repeat(h)` | `(N,)` | `(80,)`: 0,1,…,9,0,1,… |
| Frequencies | `omega` | `(d/4,)` | `(64,)` |
| Scaled columns | `x[:, None] * omega[None]` | `(N, d/4)` | `(80, 64)` |
| Scaled rows | `y[:, None] * omega[None]` | `(N, d/4)` | `(80, 64)` |
| Concatenate | `cat([sin x, cos x, sin y, cos y], dim=1)` | `(N, d)` | `(80, 256)` |

```
| sin(x·ω) | cos(x·ω) | sin(y·ω) | cos(y·ω) |
   d/4        d/4        d/4        d/4
```

This is why `d` must be divisible by 4. The token order matches the row-major flatten, so code `i` belongs to token `i`.
Frequencies: `ω_k = 1 / 10000^(k / (d/4))`, `k = 0 … d/4 − 1`, from fast-changing (fine position) to slow-changing (coarse position).

---

## Into the shared core

Each step, the core receives all three token sets joined along the token axis:

| Tokens | Shape | Example (`Nz` = 32) |
|---|---|---|
| `Tx(f)` image tokens | `(B, N, d)` | `(2, 80, 256)` |
| `Ty(yt)` answer tokens | `(B, N, d)` | `(2, 80, 256)` |
| `zt` memory tokens | `(B, Nz, d)` | `(2, 32, 256)` |
| `cat(..., dim=1)` | `(B, 2N + Nz, d)` | `(2, 192, 256)` |

Attention cost grows with `(2N + Nz)²` and is paid K times per image. A 16×16 encoder grid gives 544 tokens (fine); a 64×64 grid gives over 8,000 (too many for full attention). Pool `f` and raise `patch` together if the grid is large.

---

## Parameter count (`C` = 64, `d` = 256, `patch` = 8)

| Part | Formula | Count |
|---|---|---|
| `Tx.proj` (1×1 conv) | `C·d + d` | 16,640 |
| `Tx.norm` | `2d` | 512 |
| **`Tx` total** | | **17,152** |
| `Ty.proj` (patch conv) | `patch²·1·d + d` | 16,640 |
| `Ty.norm` | `2d` | 512 |
| **`Ty` total** | | **17,152** |
| `pos2d` | fixed, not learned | 0 |

(The two totals match only by coincidence: `C = 64` and `patch² = 64`.)

---

## Running the tests

```bash
conda activate dl_env
python Tx.py
python Ty.py
```

`Ty.py` imports `pos2d` from `Tx.py`, so keep both files in the same folder.

Expected output of `Tx.py` (verified):

```
input : (2, 64, 8, 10)
output: (2, 80, 256)
pos2d : (80, 256)
```

Expected output of `Ty.py`:

```
input : (2, 1, 64, 80)
output: (2, 80, 256)
params: 17152
```
