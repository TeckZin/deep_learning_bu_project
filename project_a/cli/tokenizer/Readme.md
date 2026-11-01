# `Tx` and `Ty`: tensor shapes, step by step

`Tx` turns a CNN feature map into tokens with 2D positional encoding.
`Ty` runs a transformer encoder (self-attention) over those tokens and turns them back into a feature map.

```
f (B, c_in, h, w) ──Tx──▶ tokens (B, N, d) ──transformer──▶ (B, N, d) ──reshape──▶ (B, d, h, w)
                   └──────────────────────────── Ty ─────────────────────────────────┘
```

## Notation

| Symbol | Meaning | Example |
|---|---|---|
| `B` | batch size | 2 |
| `c_in` | input channels (from the CNN) | 64 |
| `h`, `w` | height and width of the feature map | 8, 10 |
| `d` | token size (model dimension); divisible by 4 and by `nhead` | 256 |
| `N = h * w` | number of tokens (one per grid position) | 80 |
| `nhead` | attention heads | 8 |
| `d_head = d / nhead` | size of each head | 32 |
| `dim_ff` | feed-forward hidden size, default `4 * d` | 1024 |
| `num_layers` | stacked encoder layers | 2 |

---

## `Tx.forward(f)`

| Step | Code | Shape | Example |
|---|---|---|---|
| Input | `f` | `(B, c_in, h, w)` | `(2, 64, 8, 10)` |
| 1. Project channels | `self.proj(f)` (1×1 conv, `c_in` → `d`) | `(B, d, h, w)` | `(2, 256, 8, 10)` |
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

## `pos2d(h, w, d)`

Gives every grid position `(row y, column x)` a unique `d`-number code.

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

## `Ty.forward(f)`

| Step | Code | Shape | Example |
|---|---|---|---|
| Input | `f` | `(B, c_in, h, w)` | `(2, 64, 8, 10)` |
| 1. Tokenize | `self.tx(f)` | `(B, N, d)` | `(2, 80, 256)` |
| 2. Self-attention layers | `self.encoder(x)` | `(B, N, d)` | `(2, 80, 256)` |
| 3. Final normalize | `self.norm(x)` | `(B, N, d)` | `(2, 80, 256)` |
| 4. Channels first | `x.transpose(1, 2)` | `(B, d, N)` | `(2, 256, 80)` |
| 5. Back to a grid | `.reshape(B, -1, h, w)` | `(B, d, h, w)` | `(2, 256, 8, 10)` |
| Output | | `(B, d, h, w)` | `(2, 256, 8, 10)` |

`h` and `w` are read from `f` at the start because `Tx` flattens them away; step 5 undoes `Tx`'s steps 2–3.

### Inside one encoder layer (pre-norm)

Every sub-step keeps the shape `(B, N, d)`; the residual additions require it.

| Sub-step | Operation | Shape inside | Example |
|---|---|---|---|
| a. Norm | LayerNorm | `(B, N, d)` | `(2, 80, 256)` |
| b. Q, K, V | three linear maps `d → d` | `(B, N, d)` each | `(2, 80, 256)` |
| c. Split heads | reshape | `(B, nhead, N, d_head)` | `(2, 8, 80, 32)` |
| d. Attention scores | `Q Kᵀ / √d_head` | `(B, nhead, N, N)` | `(2, 8, 80, 80)` |
| e. Weighted sum | `softmax(scores) V` | `(B, nhead, N, d_head)` | `(2, 8, 80, 32)` |
| f. Merge heads + output linear | reshape, `d → d` | `(B, N, d)` | `(2, 80, 256)` |
| g. Residual | `x + attn` | `(B, N, d)` | `(2, 80, 256)` |
| h. Norm | LayerNorm | `(B, N, d)` | `(2, 80, 256)` |
| i. Feed-forward up | linear `d → dim_ff`, GELU | `(B, N, dim_ff)` | `(2, 80, 1024)` |
| j. Feed-forward down | linear `dim_ff → d` | `(B, N, d)` | `(2, 80, 256)` |
| k. Residual | `x + ff` | `(B, N, d)` | `(2, 80, 256)` |

The `(N, N)` score matrix in step d is where every grid position looks at every other one. Its memory grows with `N² = (h·w)²`, so large feature maps get expensive.

---

## Parameter count (`c_in` = 64, `d` = 256, `dim_ff` = 1024, 2 layers)

| Part | Formula | Count |
|---|---|---|
| `Tx.proj` (1×1 conv) | `c_in·d + d` | 16,640 |
| `Tx.norm` | `2d` | 512 |
| `pos2d` | fixed, not learned | 0 |
| **`Tx` total** | | **17,152** |
| Attention Q, K, V | `3(d·d + d)` | 197,376 |
| Attention output | `d·d + d` | 65,792 |
| Feed-forward up | `d·dim_ff + dim_ff` | 263,168 |
| Feed-forward down | `dim_ff·d + d` | 262,400 |
| Two LayerNorms | `2 · 2d` | 1,024 |
| **One encoder layer** | | **789,760** |
| Encoder (2 layers) | | 1,579,520 |
| `Ty.norm` | `2d` | 512 |
| **`Ty` total (includes `Tx`)** | | **1,597,184** |

---

## Running the tests

```bash
conda activate dl_env
python Tx.py
python Ty.py
```

Expected output of `Tx.py` (verified):

```
input : (2, 64, 8, 10)
output: (2, 80, 256)
pos2d : (80, 256)
```

Expected output of `Ty.py`:

```
input : (2, 64, 8, 10)
tokens: (2, 80, 256)
output: (2, 256, 8, 10)
params: 1597184
```
