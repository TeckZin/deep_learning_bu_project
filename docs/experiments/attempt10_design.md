# attempt10: design decisions

Status: **trained on fold 0** (2026-11-20, commit `6f51784`). Results: `attempt10_results.md`.

- Architecture: the `attempt10` preset in `project_a/cli/submission/config.py` (class `SpatialRefiner` in
  `submission/model.py`).
- Training recipe: `experiments/attempt10.json`. This is attempt8's recipe with `arch: attempt10` and `stress_aug` 0.
  `y0_aug` stays at 0.5 so later steps have damage to repair, and flips and rotations stay on.
- The `attempt7` and `attempt6` presets and their weights do not change.
- Submitting: `ACTIVE = "attempt10"` in `config.py` (since 2026-11-25). The submission bundle is in
  `attempt10_results.md` (Submission).
  
Goals: win back the edges (attempt7 BIoU 4.78) by giving the loop the full-resolution image, and make
steps 2-4 useful by carrying a spatial memory between steps (the attempt8 TODO "spatial state" idea).

## One step

```
initialize(f, y0):   z0 = tanh(Conv3x3([up(Conv1x1(LN(x))), pool(p0)]))   # [B, 16, H/4, W/4], no core call
step(f, z_t, y_t):
    x, img     = unpack(f)                      # DINO features + RGB image packed in f
    Tx, Ty, Tz = TokX(LN(pool(x)), pool(E)), TokY(y_t), TokZ(z_t)       # [B, N, d] each
    tokens     = LN(Linear([LN(Tx) ; LN(Ty) ; LN(Tz)]))                 # one fused token per patch
    tokens     = core(tokens, rope)             # ONE call: 6 x pre-norm [self-attn + 2D RoPE, FFN], final LN
    E          = ImageCNN(img)                  # [B, 24, H, W] full-res edge features (CNN inside the loop)
    dy, z_t+1  = Head(tokens, y_t, z_t, E)      # full-res dilated CNN; dy gated, z GRU-updated
    return z_t+1, y_t + dy
```

## Decisions

### 1. Core: encoder (self-attention only), not a decoder
One stream of tokens. Each token holds x, y and z for its patch, and every patch attends to every patch.

Why not a decoder (state tokens cross-attending to image tokens Tx):
- x, y and z all sit on the same 64x64 patch grid. Fusing them per token gives each patch its own image
  evidence directly. A decoder has to learn to find it through attention.
- Cross-attention costs a second N x N attention per layer (N = 4096 on a full tile) and about
  0.26M parameters per layer, so only 5 layers would fit instead of 6.
- References: ViT and DiT are encoder-only. In DiT's own comparison (`docs/Papers/DiT.pdf`, Fig. 5),
  cross-attention conditioning lost to adaLN-Zero and added about 15% FLOPs. RAFT (`docs/Papers/RAFT.pdf`)
  concatenates its fixed context features into the update at every iteration and credits that with better
  motion boundaries. SAM adds its dense mask prompt element-wise to the image embedding. Cross-attention is
  normally used for tokens that are not lined up with the image (text, a few query tokens).
- What a decoder would have offered: x re-injected at every layer, and image evidence kept apart from the
  answer. If attempt10 shows the image evidence getting washed out, try these two ablations:
  - `attempt10-ada`: x modulates every layer through a per-patch scale, shift and gate (spatial adaLN-Zero).
  - `attempt10-xattn`: the decoder.

### 2. Core size: d=256, 8 heads, 6 layers, ff_mult 4
| | params |
|---|---|
| Core (6 x `RopeEncoderLayer(256, 8, 1024)` + final LayerNorm) | 4,739,072 (cap 5,000,000) |
| Rest (image CNN, tokenizers, memory init, head) | 929,138 |
| Total | 5,668,210 (cap 10,000,000) |

- Head dim 32: divisible by 4 for 2D RoPE, and works with fused attention.
- With 6 layers, ff_mult 4 still fits (ff 3 would be 3.95M). 4 is the standard ViT ratio.
- Layers are pre-norm with GELU and dropout 0.05. `RopeEncoderLayer` has the same parameters as
  `nn.TransformerEncoderLayer`, but its attention is written by hand on `F.scaled_dot_product_attention`
  so RoPE can rotate Q and K. RoPE has no parameters.

### 3. 2D positional encoding: RoPE on Q and K
- The additive 2D sinusoidal code (`pos.py`) stays available (`core.pos = add | rope | both`). attempt10
  uses `rope`.
- RoPE: half of each head's dimensions rotate by row, half by column. Attention then depends on the
  relative position of two patches. Base 100 instead of 10000, because the grid is at most 64 cells.
- With `rope`, no additive position code is added to the tokens.
- It uses `model.pos_offset` like the additive code, so training crops keep their true tile positions.

### 4. The original image inside the loop

**The problem.** The refiner's interface is fixed by the staff runner:
`initialize(f, y0)` and `step(f, z, y)`. The only per-image input is `f`, the frozen encoder's features
[B, 384, H/16, W/16]. There is no image argument, so the image has to travel inside `f`.

**Packing the image into f** (`submission/image.py`, `pack_image`). On a 1024 x 1024 tile:

| | Shape | What happens |
|---|---|---|
| RGB image | [B, 3, 1024, 1024] | ImageNet-normalized (same mean/std as DINOv2) |
| `pixel_unshuffle(image, 16)` | [B, 768, 64, 64] | Each 16 x 16 x 3 pixel block becomes the 768 channels of its grid cell. Nothing is lost or resampled: it only rearranges the pixels. |
| DINO features | [B, 384, 64, 64] | Unchanged |
| Packed `f` = concat | [B, 1152, 64, 64] | What `step()` receives |

Inside `step()`, `unpack_image` splits `f` back into the DINO features (the first 384 channels) and the
image (`pixel_shuffle` of the last 768 channels gives [B, 3, 1024, 1024] again, exactly the original pixels).

**Why it stays aligned in training.** Training crops are cut from `f` on the 16-pixel grid. Cell (i, j)
of the packed channels holds exactly the pixels of cell (i, j) of the crop, so the image is cropped along
with the features.

**Who packs it.** No flag: the refiner config decides (`image.use = true` in the attempt10 preset).
- `train.py`: when `image.use`, it wraps the frozen encoder + H0 in `ImagePackedBase`, so
  `base(image)` returns the packed `f` and the same y0. H0 still sees only the DINO features, so the
  existing H0 heads are reused without retraining.
- `infer.py` stage 1 (encode) saves the raw image as uint8 in each intermediate file, as the TA asked
  (about +3 MB per 1024 tile). Stage 2 (refine) packs it into `f` before the loop.
- `build_refiner(feature_channels)` gets 1152 and knows from the config that the last 768 are the image.

**What uses the image: the image CNN.** 3 conv layers (3x3, 24 channels, GELU) at full resolution, no
downsampling, give edge features `E` [B, 24, H, W]. They run on the ImageNet-normalized image inside every
`step()`. The image is the same each step, so this recomputes the same `E`. `E` is used twice:
- in the head CNN, next to the coarse `dy` and `p`: this is where road boundaries are drawn, so it is
  the main use;
- average-pooled to the patch grid and concatenated to the DINO channels before the TokX MLP, so `Tx`
  carries both the semantic features and the patch's edge summary.

**TA ruling (2026-11-18).** A CNN inside the loop may see the image; nothing after the loop may. The
image CNN runs inside `step()`, and nothing touches `y_K` after the last step. Bypass check: the head only
adds a gated correction to `y_t`, and `predict.py --ablation reset-state` and `no-mask-feedback` must
still change the output clearly.

### 5. Memory z: spatial maps, "parallel to y"
- z_t is `[B, Cz, H/s, W/s]` with **Cz = 16, s = 4** (256x256 on a full tile). It replaces the 32 memory
  slots. `s = 1` (full resolution) is one config field away. It costs about 16x the memory when
  backpropagating through K steps.
- z0 comes from the frozen encoder output plus H0's answer (above), like RAFT's context-initialized hidden
  state. It is kept in float32 under autocast, like y.

### 5b. Head: one full-resolution CNN writes dy and z
`SpatialHead` (`submission/output_heads.py`):
1. Coarse dy: each core token becomes its 16x16 block of logits through `Linear(256, 256)` and
   `pixel_shuffle`. 16 more channels of each token (`ctx`) are upsampled bilinearly.
2. Stack: 5 dilated 3x3 convs (dilations 1, 2, 4, 8, 1; 64 channels; GELU) at full resolution. The input is
   `[coarse dy, p_t, ctx (16), E (24), up(z_t) (16)]`, 58 channels in all.
3. dy: a 3x3 conv on the stack features gives `r, g`, and `dy = (coarse + r) * sigmoid(g + 2)` (gated, as attempt7).
4. z: the stack features are average-pooled to z's grid (H/4) and concatenated with `z_t`. A 3x3 conv on
   that gives `u, c` (Cz channels each). The update is ConvGRU style (as in RAFT):
   `z_t+1 = (1 - u) * z_t + u * tanh(c)`, with `u = sigmoid(.)`. This keeps z bounded for any K.

Initialization: the token projection starts small (std 0.01), and the dy conv starts at zero (r = 0,
gate about 0.88). So the untrained refiner stays near y0 but every step still changes the mask. The z conv
uses PyTorch's default initialization, so z changes from the start. That is why `reset-state` has an
effect even before training.
- Global context now comes only from self-attention over all patches. If long-range connectivity
  (clDice) drops compared with attempt7, add a few global register tokens (`memory.registers`).

### 6. Tokenizers: one per input, shared d and positions
| Input | What it is | Tokenizer |
|---|---|---|
| x | 384 ch DINO at H/16 (+ pooled E), same every step | LayerNorm, then 1x1 conv MLP (as attempt7) |
| y | 1 ch logits, full resolution, changes every step | Patch embed of [p, logit/10, 4p(1-p)] (as attempt7) |
| z | Cz ch learned maps at H/4 | Strided conv to the patch grid |

No weight sharing: the three inputs have different channel counts, resolutions and scales, and sharing
would save only about 0.1M parameters.

### 7. Fusion: LayerNorm per source, then concat and Linear (or MLP)
`tokens = LN(Linear([LN(Tx) ; LN(Ty) ; LN(Tz)]))` (`tokenizer.fuse = "concat"`, the default).

- Correction to the first plan, `Linear([Tx ; Ty ; Tz])` without the per-source norms: that equals
  `Wx Tx + Wy Ty + Wz Tz + b`, a sum of three projected vectors. Each tokenizer already ends in a linear
  layer, so it has no more expressive power than plain `sum` fusion.
- What the per-source LayerNorm adds: Tx, Ty and Tz start on an equal footing. Raw DINO tokens have a
  much larger norm than the answer tokens (the reason for attempt7's `feat_norm`), and without it one
  source can dominate. The fusion layer then learns how much weight each source gets, per channel.
- `tokenizer.fuse = "mlp"` (Linear, GELU, Linear) is an option: x, y and z can then interact before the
  first attention (e.g. "x says road AND y says not road", an error signal). With linear fusion the first
  layer's queries can only use a linear mix; later layers can form such interactions after the first FFN.
  Try it with `--set tokenizer.fuse=mlp`.
- x is fused again at every step, because `step()` tokenizes it again.

## Interpretation
- A token is one patch's case file: what the image says here (x), what I currently answer here (y),
  and what I noted here on earlier steps (z).
- Attention: each patch asks every other patch a question, e.g. a patch whose road ends with low
  confidence asks "who nearby looks like road continuing in my direction?". RoPE adds where the other
  patch is, so some heads can specialize in direction and distance, others in content anywhere.
  8 heads = 8 such questions in parallel. The FFN updates each patch's case file with what it gathered.
- 6 layers = 6 rounds of message passing per step (evidence can hop along a road behind a tree).
  K steps = the same procedure again with the updated y and z; x stays fixed.
- z at H/4 with Cz = 16 holds 4 x 4 x 16 = 256 numbers per patch, exactly one token (d = 256): a
  token-sized memory per patch, laid out spatially. TokZ reads it as a token; the head CNN reads it as
  maps with detail finer than a patch. The finer layout is worth it because the head CNN, which sees
  the image edges, writes z. If the trained z is smooth within each patch, carrying the core's output
  tokens instead would be the simpler equivalent (an ablation).
- Risk: z is never supervised directly, so the model could ignore it. The `reset-state` ablation shows it.

## Alternatives considered (not in attempt10)
| Option | Why not now |
|---|---|
| Decoder: state tokens cross-attend to Tx | See decision 1. Ablation `attempt10-xattn` |
| Spatial adaLN-Zero (x modulates each layer) | Best in DiT, but DiT's condition is global, not per patch; untested here. Costs about 0.39M per layer (5 layers at ff 2). Ablation `attempt10-ada` |
| Shared tokenizer for x, y, z | See decision 6 |
| Global memory slots (attempt7) | Replaced by spatial z. Can come back as registers |
| z at full resolution | Memory cost through K steps. A config change if needed |

## Code (as built)
- `submission/image.py`: `pack_image` / `unpack_image`, `ImagePackedBase` (wraps the frozen base so
  `base(image)` returns f with the image packed in), `ImageCNN`.
- `submission/pos.py`: `rope2d`, `apply_rope` (crop-aware, base 100).
- `submission/core.py`: `RopeEncoderLayer` (same parameters as `nn.TransformerEncoderLayer`), `core.pos`.
- `submission/tokenizers.py`: `SplitTokenizer` (`fuse = concat | mlp`).
- `submission/memory.py`: `SpatialMemory` (z0 from x and p0).
- `submission/output_heads.py`: `SpatialHead` (returns dy and z_t+1).
- `submission/model.py`: `SpatialRefiner`; `build_refiner` picks it when `memory.kind = "spatial"`.
- `submission/config.py`: new fields `tokenizer.kind = split`, `memory.kind/channels/stride`, `core.pos`,
  `head.kind = spatial`, section `image` (defaults keep attempt7/6 unchanged); the `attempt10` preset.
- The image is packed from the refiner config, not an encoder flag: `train.py` wraps the bases in
  `ImagePackedBase` when `image.use`; `infer.py` stores the raw uint8 image in the intermediate files
  and packs it in the refine stage. H0 heads are unchanged.
- Tests: attempt10 passes the staff runner checks and caps; pack/unpack round trip; RoPE scores depend
  only on relative position; train -> export -> infer end to end for attempt7 and attempt10.

Parameters (`python -m submission.model`, 384 DINO channels + 768 packed image channels):
core 4,739,072, total 5,668,210.

## Checks after the fold-0 run (details in `attempt10_results.md`)
- [ ] K=4 clearly above K=1: **no.** K=1 59.59, K=4 59.86 (+0.27). Almost all of the gain still comes
  from step 1.
- [x] BIoU at K=4 above the baseline's (about 8): **yes.** 10.55, compared with H0's 10.06 on this fold.
- [x] clDice still near attempt7's 71: **yes.** 82.66 (H0 72.40).
- [~] `reset-state` and `no-mask-feedback` change the output: **moderately.** On 50 tiles K=4 scores 63.06,
  reset-state 60.12 and no-mask-feedback 60.75. The state matters, but the drop is not large.
