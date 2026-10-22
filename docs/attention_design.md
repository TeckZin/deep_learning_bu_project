# Attention in the looped refiner: options and intuitions

Design notes for the shared core `R_θ`. Notation follows the handout: `f` encoder features,
`y_t` road logits after `t` steps, `z_t` memory (latent) tokens, `J` = keys (so `K` stays the
loop count). Shapes are for one full 1024×1024 tile at stride 16 → a 64×64 grid, `N = 4096`
tokens per stream; 512×512 training crops give a 32×32 grid, `N = 1024`.

## 1. Attention in one paragraph

Each token produces three vectors with learned matrices:

| Vector | Formula | Role |
|---|---|---|
| Query | `Q = X·W_Q` | "what am I looking for?" |
| Key | `J = X·W_J` | "how to find me" (used only for matching) |
| Value | `V = X·W_V` | "what I hand over if chosen" |

Each query is scored against every key (dot product, scaled by `√d_h`), the scores are turned
into weights by a row-wise softmax, and the token receives the weighted average of the values:

```
Attention = softmax_row(Q·Jᵀ / √d_h) · V
```

**Keys and values come from the same tokens, through different projections. Neither is a
label.** A value is whatever information is useful to pass on (road evidence, direction,
width, texture...), learned end to end. The only "road / not road" number is produced later
by the output head.

*Library analogy:* the key is a book's catalog entry, the value is the book's contents.
Attention is a soft lookup: it returns a blend of contents, weighted by how well each
catalog entry matched the request.

*Road example:* a token at a shadow-covered gap asks "who near me looks like road pointing my
way?" Tokens on either side of the gap have keys saying "confident road segment, running
east–west", so they match strongly; their values (road evidence, direction) flow into the
gap token, and the head can then raise the road logits there.

## 2. Where attention lives in the model

| Component | Attention? |
|---|---|
| Encoder `E` | CNN: no. ViT (e.g. DINOv3): self-attention over image patches, frozen, run once |
| Tokenizers `T_x`, `T_y` | No: linear / patch-conv projections |
| Initializer `I_φ` | Optional (e.g. learned queries reading `X` and `y_0`) |
| **Core `R_θ`, every step** | **Yes. This is the attention we design** |
| FFN inside the core | No: a separate MLP on each token |
| Output head | No: readout, upsampling, convolutions |

## 3. Three ways to combine image, answer, and memory in the core

Inputs to every step: image tokens `X = T_x(f)` (fixed), answer tokens `A_t = T_y(y_t)`
(change each step), memory `z_t` (changes each step). All three share the same grid
positions: token `k` in each stream is the same 16×16 patch.

### Option 1: Add, then self-attention (the starter: `model.py`, `core(state + context)`)

```
h = z_t + X + A_t                    [N, d]
Q = h·W_Q, J = h·W_J, V = h·W_V      [N, d_h] each   (projection)
O = softmax(Q·Jᵀ/√d_h)·V             [N, d_h]        scores: N × N
z_{t+1} = h + O·W_O → + FFN          [N, d]
```
- **Intuition:** every location talks to every other location, but each location first
  blends what it sees, what it predicted, and what it remembers into one vector.
- **Relation to TRM:** this is how TRM itself combines its input, answer, and latent: it
  adds their embeddings before its shared network (TRM, Fig. 1; cited in handout §12).
- **Pros:** cheapest; trivially grid-aligned; simplest to implement.
- **Cons:** the model can't query the streams separately (e.g. "give me image evidence,
  not your old answer"); the image signal is re-added every step and can dominate or wash
  out the memory.

### Option 2: Concatenate, then self-attention

```
H = [X ; A_t ; z_t]                  [3N, d]
Q = H·W_Q, J = H·W_J, V = H·W_V      [3N, d_h] each  (projection)
O = softmax(Q·Jᵀ/√d_h)·V             [3N, d_h]       scores: 3N × 3N
z_{t+1} = z_t + O[memory rows]·W_O   [N, d]
```
- **Intuition:** one big conversation in which every token can pick whom to listen to,
  whether image, answer, or memory.
- **Pros:** streams stay distinct; maximally flexible.
- **Cons:** 9× the attention cost; image and answer tokens also get updated, and those
  outputs are thrown away; streams need type embeddings so the model can tell them apart.

### Option 3: Cross-attention, where memory reads the evidence

```
M = [X ; A_t]                        [2N, d]         image + current answer
Q = z_t·W_Q                          [N, d_h]        memory asks       (projection)
J = M·W_J, V = M·W_V                 [2N, d_h] each  evidence responds (projection)
O = softmax(Q·Jᵀ/√d_h)·V             [N, d_h]        scores: N × 2N
z' = z_t + O·W_O                     [N, d]
z'' = z' + SelfAttn(z')              (optional: memory ↔ memory talk)
z_{t+1} = z'' + FFN(z'')
```
- **Intuition:** memory is the thing being updated; the image and the current answer are
  evidence it consults. Roles are explicit.
- **Where it comes from:** the input / answer / latent separation is from TRM
  (Jolicoeur-Martineau, 2025), the recursive-reasoning paper cited in the project handout
  (§12). TRM itself sums the three (see option 1); giving memory its own queries over the
  input via cross-attention is the Perceiver mechanism (Jaegle et al., 2021).
- **Pros:** clean separation; image keys/values can be cached across steps (see §4).
- **Cons:** more pieces than option 1; 2× option 1's score matrix if done globally.

### The linear algebra (single head)

$N$ tokens per stream, width $d$, head width $d_h$; $W_Q, W_J, W_V \in \mathbb{R}^{d\times d_h}$
are learned and shared across loop steps; softmax is row-wise.

**Option 1: add, then self-attention**

$$
h = z_t + X + A_t \in \mathbb{R}^{N\times d}
$$
$$
Q = hW_Q,\quad J = hW_J,\quad V = hW_V \in \mathbb{R}^{N\times d_h}
$$
$$
S = \frac{QJ^\top}{\sqrt{d_h}} \in \mathbb{R}^{N\times N},\qquad
O = \operatorname{softmax}(S)\,V \in \mathbb{R}^{N\times d_h}
$$
$$
z_{t+1} = h + O\,W_O \;(\to +\,\text{FFN})
$$

A single score is $S_{ij} = h_i^\top (W_Q W_J^\top)\, h_j / \sqrt{d_h}$. Since each
$h_i = z_i + x_i + a_i$, it expands into 9 cross terms ($z\cdot z$, $z\cdot x$, $x\cdot a$, …)
that all share one weight matrix. That is the precise sense in which adding first "blends"
the streams: the model can't weight image-to-memory matches differently from
answer-to-memory matches.

**Option 2: concatenate (stack rows), then self-attention**

$$
H = \begin{bmatrix} X \\ A_t \\ z_t \end{bmatrix} \in \mathbb{R}^{3N\times d}
\quad(\text{+ a learned type embedding per stream})
$$
$$
Q = HW_Q,\quad J = HW_J,\quad V = HW_V \in \mathbb{R}^{3N\times d_h}
$$
$$
S = \frac{QJ^\top}{\sqrt{d_h}} =
\begin{bmatrix}
S_{XX} & S_{XA} & S_{Xz}\\
S_{AX} & S_{AA} & S_{Az}\\
S_{zX} & S_{zA} & S_{zz}
\end{bmatrix} \in \mathbb{R}^{3N\times 3N},\quad \text{each block } N\times N
$$
$$
O = \operatorname{softmax}(S)\,V \in \mathbb{R}^{3N\times d_h},\qquad
z_{t+1} = z_t + O_{[2N:3N]}\,W_O
$$

Only the memory rows of $O$ are kept, and they depend only on the bottom row of blocks
$[S_{zX}\; S_{zA}\; S_{zz}]$. That is exactly cross-attention from $z$ to all three streams
($X$, $A$, $z$). The other two rows of blocks are wasted compute in a one-layer core; they
only matter in a multi-layer core, where the updated $X$ and $A$ feed the next layer.

**Option 3: cross-attention (memory queries, evidence keys and values)**

$$
M = \begin{bmatrix} X \\ A_t \end{bmatrix} \in \mathbb{R}^{2N\times d}
$$
$$
Q = z_t W_Q \in \mathbb{R}^{N\times d_h},\qquad
J = MW_J,\quad V = MW_V \in \mathbb{R}^{2N\times d_h}
$$
$$
S = \frac{1}{\sqrt{d_h}}\big[\, QJ_X^\top \;\big|\; QJ_A^\top \,\big] \in \mathbb{R}^{N\times 2N}
$$
$$
P = \operatorname{softmax}(S) = [\,P_X \mid P_A\,],\qquad
O = PV = P_X V_X + P_A V_A \in \mathbb{R}^{N\times d_h}
$$
$$
z_{t+1} = z_t + O\,W_O \;(\to \text{optional self-attention on } z \to +\,\text{FFN})
$$

- **Shared budget:** the softmax runs over all $2N$ keys together, so image and answer
  evidence compete for the same attention (each row of $P$ sums to 1 overall).
- **The output splits cleanly:** $P_X V_X$ is "what I took from the image";
  $P_A V_A$ is "what I took from the current mask".
- **Cacheable:** $J_X = XW_J$ and $V_X = XW_V$ are the same every step, so compute them
  once; each step recomputes only $Q$, $J_A$, $V_A$.

| | Queries from | Keys/values from | Score matrix | Output rows |
|---|---|---|---|---|
| Option 1 | $h$ ($N$) | $h$ ($N$) | $N\times N$ | $N$ |
| Option 2 | $H$ ($3N$) | $H$ ($3N$) | $3N\times 3N$ | $3N$, keep $N$ |
| Option 3 | $z$ ($N$) | $M$ ($2N$) | $N\times 2N$ | $N$ |

## 4. Cross-attention in detail

### Who plays which role
- **Queries:** one per memory token (one per grid cell), computed from that token's
  *current* state: where it is, what it believes, what happened in earlier steps. So the
  questions are specific and can change across steps:
  - gap under a shadow: "is there road leading into me from either side?"
  - confident road interior: asks little and keeps its state
  - predicted road without image support: "does any image evidence back this up?"
    (how false bridges can get removed)
- **Keys and values:** every image token *and* every answer token contributes both. "The
  current mask says road here" is therefore something memory can look up, as a learned
  vector rather than a raw label.

### Shapes and output dimensionality
```
Q  = z_t · W_Q            [4096, d]
J  = [X ; A_t] · W_J      [8192, d]
V  = [X ; A_t] · W_V      [8192, d]
S  = Q · Jᵀ / √d_h        [4096, 8192]
W  = softmax_row(S)       [4096, 8192]
O  = W · V                [4096, d]     ← rows = number of queries; 8192 cancels out
z' = z_t + O · W_O        [4096, d]
```
**Output size = number of queries.** Because queries are grid-aligned memory tokens, the
output is grid-aligned automatically. With `h` heads, `d` is split into `h` slices of
`d/h`, each head runs the same computation, and `W_O` mixes the concatenated result back
to `d`.

### What changes across loop steps

| | Fixed across steps | Changes every step |
|---|---|---|
| `W_Q, W_J, W_V, W_O`, FFN | ✅ (shared weights: the loop) | |
| Image tokens `X` and their `J`, `V` | ✅ (**compute once, reuse for all K**) | |
| Answer tokens `A_t` and their `J`, `V` | | ✅ (mask feedback) |
| Memory `z_t` and its queries | | ✅ |
| Attention weights | | ✅ (recomputed each step) |

How the required ablations read in these terms:
- **Reset state** (`z ← z_0` each step): queries can't change, so memory can't ask new
  questions based on what it learned.
- **No mask feedback** (`y_t ← y_0`): the answer keys and values are frozen at `y_0`, so
  memory can't see its own corrections.
- **Diagnostic:** plot one gap token's attention map at step 1 vs step 4. If the maps are
  nearly identical, the loop is mostly repeating the same computation.

### Position in cross-attention
Queries and keys both carry grid positions, so position goes into the `Q·Jᵀ` score:
- 2D RoPE: rotate `Q` by the query cell's (row, col) and `J` by the key cell's; the score
  then depends on the offset. Image and answer keys use the same coordinates as memory.
- Relative bias: add `B[Δrow, Δcol]` to the scores (natural with windows).
- Apply position inside every attention call, so it doesn't wash out over K steps.
- Prefer relative encodings: training crops (32×32) and full tiles (64×64) then share the
  same notion of "3 cells to the north-east". See the experiment ideas in the README.

### Windowing
Global attention scores every query against every key, and training stores this for every
unrolled step. **Windowed cross-attention** (each memory token attends only to
image/answer tokens in a local window, e.g. 8×8 cells) cuts cost to `N × 2·64` and adds a
locality prior. Long-range links then come from shifted windows, a few global tokens, or
the loop itself: information can travel one window further per step, which is itself a
testable hypothesis.

## 5. From tokens back to pixels

```
z_{t+1}               [4096, d]
→ readout d → 1       [4096, 1]
→ reshape             [1, 64, 64]       token k → cell (k // 64, k % 64)
→ upsample ×16        [1, 1024, 1024]
y_{t+1} = y_t + Δ     [1, 1024, 1024]   = required [B, 1, H, W]
```
Better for thin roads: **unpatchify**. Read out `d → 256` per token, rearrange each to a
16×16 block (pixel shuffle), optionally followed by a few smoothing convs. Each token then
places the road edge inside its own patch (helps Boundary IoU).

**Constraint:** the reshape works only if memory tokens *are* the grid (`N_z = 64·64`, fixed
order). With free-floating latents (e.g. 256 Perceiver-style tokens), you need an extra
decoder step where grid queries cross-attend back to the latents. Start grid-aligned.

## 6. Recommended starting point

Option 3 with grid-aligned memory: windowed cross-attention from `z_t` to `[X ; A_t]`, a
memory self-attention block, an FFN, 2D RoPE or relative bias in every attention call,
image keys/values cached across steps, and an unpatchify head with a residual logit update.
Keep option 1 (the starter) as the simple baseline to compare against.

## Further reading
- Perceiver (Jaegle et al., 2021), arXiv:2103.03206: latents cross-attending to inputs with
  shared weights
- DETR (Carion et al., 2020), arXiv:2005.12872: learned query tokens, positions added in
  every attention layer
- Swin Transformer (Liu et al., 2021), arXiv:2103.14030: window attention, relative bias
- TRM (Jolicoeur-Martineau, 2025), arXiv:2510.04871 (cited in the handout): separate input / answer / latent

(arXiv IDs from memory; verify before citing.)
