# Multi-Head Attention (MHA) and Grouped Query Attention (GQA)

## Notation and dimensions

| Symbol | Meaning | Code variable | Default value |
|--------|---------|---------------|---------------|
| $T$ | Sequence length (number of token positions) | `seqlen` (line 584), configured by `args.train_seq_len` | 1024 |
| $d_\text{model}$ | Model hidden dimension | `dim` (line 584), configured by `args.model_dim` | 512 |
| $h$ | Number of query/attention heads | `self.num_heads` (line 569) | 8 |
| $h_\text{kv}$ | Number of key-value heads (GQA) | `self.num_kv_heads` (line 570) | 4 |
| $d_k$ | Per-head dimension = $d_\text{model} / h$ | `self.head_dim` (line 571) | 64 |
| $B$ | Batch size | `bsz` (line 584) | varies |

## Single-head attention (the building block)

Given input $X \in \mathbb{R}^{T \times d_\text{model}}$ (one sequence, batch omitted for clarity):

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_k}$.

The attention score matrix:

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

The output:

$$Y = AV \in \mathbb{R}^{T \times d_k}$$

**Important:** it's $Y = AV$, not $Y = AV^\top$. $A$ is $T \times T$ and $V$ is $T \times d_k$, so $AV$ gives $T \times d_k$. Transposing $V$ would give a dimension mismatch: $(T \times T) \cdot (d_k \times T)$ doesn't conform.

### What $A_{ij}$ means

$A \in \mathbb{R}^{T \times T}$ is the attention weight matrix. Entry $A_{ij}$ is "how much position $i$ attends to position $j$." Each row sums to 1 (from softmax). For causal (autoregressive) attention, $A_{ij} = 0$ for $j > i$ — position $i$ can only look at past and present positions.

### What $T$ means, concretely

$T$ is the number of token positions in the sequence. If you feed in the sentence "The cat sat" tokenized as 3 tokens, $T = 3$. In this codebase, $T = 1024$ (the `TRAIN_SEQ_LEN`). The $T \times T$ attention matrix is therefore $1024 \times 1024$ — each of the 1024 positions computes a weight over all 1024 positions.

## Standard multi-head attention (MHA)

Instead of one set of $W_Q, W_K, W_V$, we have $h$ independent sets (called "heads"):

For head $i \in \{1, \ldots, h\}$:

$$Q_i = XW_Q^{(i)}, \quad K_i = XW_K^{(i)}, \quad V_i = XW_V^{(i)}$$

where $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_\text{model} \times d_k}$.

Each head computes its own attention independently:

$$A_i = \text{softmax}\!\left(\frac{Q_i {K_i}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

$$Y_i = A_i V_i \in \mathbb{R}^{T \times d_k}$$

The heads are concatenated and projected:

$$\text{MultiHead}(X) = \text{Concat}(Y_1, Y_2, \ldots, Y_h) \, W_O$$

where $\text{Concat}(Y_1, \ldots, Y_h) \in \mathbb{R}^{T \times (h \cdot d_k)}$ and $W_O \in \mathbb{R}^{(h \cdot d_k) \times d_\text{model}}$.

Since $h \cdot d_k = d_\text{model}$ (by construction), the output is $\in \mathbb{R}^{T \times d_\text{model}}$ — same shape as the input.

### Total parameter count for MHA

$$\underbrace{h \cdot d_\text{model} \cdot d_k}_{W_Q \text{ total}} + \underbrace{h \cdot d_\text{model} \cdot d_k}_{W_K \text{ total}} + \underbrace{h \cdot d_\text{model} \cdot d_k}_{W_V \text{ total}} + \underbrace{d_\text{model}^2}_{W_O}$$

Since $h \cdot d_k = d_\text{model}$, this simplifies to $4 \cdot d_\text{model}^2$. With $d_\text{model} = 512$: $4 \times 512^2 = 1{,}048{,}576$ params per attention layer.

## Grouped Query Attention (GQA) — what this code actually uses

Standard MHA has $h$ independent K and V projections. GQA reduces this: $h_\text{kv} < h$ KV heads are shared among groups of query heads. Each group of $h / h_\text{kv}$ query heads shares one K head and one V head.

In this codebase: $h = 8$ query heads, $h_\text{kv} = 4$ KV heads. Each pair of query heads shares one KV head.

For query head $i$, define $g(i) = \lfloor i \cdot h_\text{kv} / h \rfloor$ as the KV group index. Then:

$$Q_i = XW_Q^{(i)} \in \mathbb{R}^{T \times d_k} \quad \text{(8 independent Q projections)}$$

$$K_{g(i)} = XW_K^{(g(i))} \in \mathbb{R}^{T \times d_k} \quad \text{(4 shared K projections)}$$

$$V_{g(i)} = XW_V^{(g(i))} \in \mathbb{R}^{T \times d_k} \quad \text{(4 shared V projections)}$$

$$A_i = \text{softmax}\!\left(\frac{Q_i {K_{g(i)}}^\top}{\sqrt{d_k}}\right), \quad Y_i = A_i V_{g(i)}$$

The concatenation and output projection are the same as standard MHA:

$$\text{GQA}(X) = \text{Concat}(Y_1, \ldots, Y_h) \, W_O$$

### GQA parameter savings

$$\underbrace{h \cdot d_\text{model} \cdot d_k}_{W_Q} + \underbrace{h_\text{kv} \cdot d_\text{model} \cdot d_k}_{W_K} + \underbrace{h_\text{kv} \cdot d_\text{model} \cdot d_k}_{W_V} + \underbrace{d_\text{model}^2}_{W_O}$$

$= d_\text{model}^2 + h_\text{kv} \cdot d_\text{model} \cdot d_k + h_\text{kv} \cdot d_\text{model} \cdot d_k + d_\text{model}^2$

$= 2 \cdot d_\text{model}^2 + 2 \cdot h_\text{kv} \cdot d_\text{model} \cdot d_k$

With $d_\text{model} = 512$, $h_\text{kv} = 4$, $d_k = 64$:

$2 \times 512^2 + 2 \times 4 \times 512 \times 64 = 524{,}288 + 262{,}144 = 786{,}432$ params

vs. MHA's $1{,}048{,}576$. That's a 25% reduction in attention parameters.

## Code-to-math mapping (train_gpt.py, CausalSelfAttention)

The per-head projections $W_Q^{(1)}, \ldots, W_Q^{(h)}$ are **not** stored as $h$ separate matrices. They're packed into one big matrix, then the output is reshaped into heads.

### Packed projection matrices (lines 575-578)

| Code | Math | Shape |
|------|------|-------|
| `self.c_q` | $W_Q = [W_Q^{(1)} \| \cdots \| W_Q^{(h)}]$ | $\mathbb{R}^{d_\text{model} \times d_\text{model}}$ = $512 \times 512$ |
| `self.c_k` | $W_K = [W_K^{(1)} \| \cdots \| W_K^{(h_\text{kv})}]$ | $\mathbb{R}^{d_\text{model} \times (h_\text{kv} \cdot d_k)}$ = $512 \times 256$ |
| `self.c_v` | $W_V = [W_V^{(1)} \| \cdots \| W_V^{(h_\text{kv})}]$ | $\mathbb{R}^{d_\text{model} \times (h_\text{kv} \cdot d_k)}$ = $512 \times 256$ |
| `self.proj` | $W_O$ | $\mathbb{R}^{d_\text{model} \times d_\text{model}}$ = $512 \times 512$ |

### Forward pass step-by-step (lines 583-603)

```
x: (B, T, d_model)                       # Input, e.g. (B, 1024, 512)

# 1. Project Q, K, V using packed matrices, then reshape into per-head views
q = c_q(x)                               # (B, T, d_model) → (B, T, h·d_k)
  .reshape(B, T, h, d_k)                 # → (B, T, 8, 64)
  .transpose(1,2)                         # → (B, 8, T, 64)  = (B, h, T, d_k)

k = c_k(x)                               # (B, T, d_model) → (B, T, h_kv·d_k)
  .reshape(B, T, h_kv, d_k)              # → (B, T, 4, 64)
  .transpose(1,2)                         # → (B, 4, T, 64)  = (B, h_kv, T, d_k)

v = c_v(x)                               # same as k: → (B, 4, T, 64)

# 2. QK-norm (RMSNorm on last dim, per head)
q = rms_norm(q)                           # normalize each head's queries
k = rms_norm(k)                           # normalize each head's keys

# 3. Rotary positional embeddings
q = apply_rotary_emb(q, cos, sin)         # inject position info
k = apply_rotary_emb(k, cos, sin)

# 4. Per-head learned gain on queries
q = q * q_gain[None, :, None, None]       # q_gain shape: (h,) = (8,)

# 5. Scaled dot-product attention (with GQA broadcasting)
#    PyTorch's F.scaled_dot_product_attention handles the h_kv→h broadcast:
#    each pair of Q heads (0,1), (2,3), ... shares one KV head
y = sdpa(q, k, v, is_causal=True)         # → (B, h, T, d_k) = (B, 8, T, 64)

# 6. Concatenate heads and project (the W_O step)
y = y.transpose(1,2)                      # → (B, T, h, d_k) = (B, T, 8, 64)
  .reshape(B, T, d_model)                 # → (B, T, 512) = Concat(Y_1,...,Y_h)
y = proj(y)                               # → (B, T, 512) = Concat(...) @ W_O
```

The reshape from `(B, T, h, d_k)` to `(B, T, d_model)` **is** the concatenation. Since the per-head outputs are contiguous in the last dimension, reshape just reinterprets the memory layout — zero compute.

### Additional operations not in vanilla attention math

This implementation adds several things on top of the standard $\text{softmax}(QK^\top/\sqrt{d_k})V$ formula:

1. **QK-Norm** (line 588-589): $Q_i \leftarrow \text{RMSNorm}(Q_i)$, $K_j \leftarrow \text{RMSNorm}(K_j)$. Stabilizes attention logits by ensuring $\|q\| \approx 1$ and $\|k\| \approx 1$ before the dot product.

2. **RoPE** (lines 590-592): Rotary Positional Embedding multiplies complex-number-encoded positions into Q and K so that $q_i^\top k_j$ naturally decays with $|i-j|$. This is how the model knows token order (pure attention is permutation-invariant without positional info).

3. **q_gain** (line 593): A learned per-head scalar $\gamma_i$ that scales $Q_i \leftarrow \gamma_i Q_i$. Effectively a learned temperature: $A_i = \text{softmax}(\gamma_i^2 \cdot Q_i K_i^\top / \sqrt{d_k})$. Initialized to 1.5 (`qk_gain_init`), so attention starts sharper than default.

4. **Causal mask** (`is_causal=True`): $A_{ij} = 0$ for $j > i$, enforcing autoregressive property. Implemented inside Flash Attention without materializing the $T \times T$ mask.

## Why multi-head attention?

### The standard argument

A single attention head computes one set of attention weights $A \in \mathbb{R}^{T \times T}$. Each position can only express one "attention pattern" — one distribution over which other positions to attend to. With $h$ heads, each position simultaneously maintains $h$ independent patterns.

Empirically, different heads learn qualitatively different roles:
- Some heads attend to the immediately preceding token (local/bigram patterns)
- Some attend to syntactically related tokens (subject-verb agreement)
- Some attend to semantically related tokens far away
- Some heads in early layers attend to punctuation/delimiters
- Some heads learn "induction heads" — copying patterns from earlier in the sequence

### How is specialization possible without explicit instruction?

This is the core of the question. There is no loss term that says "head 3 should learn syntax." The specialization is **emergent**, arising from four mechanisms:

**1. Random initialization breaks symmetry**

Each $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ is initialized randomly. At step 0, each head projects $X$ into a different random $d_k$-dimensional subspace. These different starting points mean each head sees a different "view" of the data from the very first gradient step, and gradient descent moves them in different directions.

If all heads were initialized identically, they would receive identical gradients and remain identical forever (symmetry is never broken). Random init is sufficient to break this symmetry.

**2. The bottleneck forces complementarity**

Each head can only produce a $d_k$-dimensional output ($d_k = 64$), but the model needs $d_\text{model} = 512$ dimensions to represent the full residual stream. The concatenation $\text{Concat}(Y_1, \ldots, Y_h) \in \mathbb{R}^{T \times d_\text{model}}$ means each head "owns" a $d_k$-wide slice of the output space.

If two heads learned the same attention pattern and extracted the same information, the $W_O$ projection couldn't use their combined $2 d_k$ dimensions any better than a single $d_k$. The gradient signal through $W_O$ therefore rewards heads that capture **different** information — redundant heads receive weaker gradient because $W_O$ learns to downweight duplicated signals.

Formally: let $Y_{\text{cat}} = [Y_1; Y_2; \ldots; Y_h]$. The output is $Y_{\text{cat}} W_O$. If $Y_i \approx Y_j$, then columns $i$ and $j$ of $Y_{\text{cat}}$ are near-duplicates. The effective rank of $Y_{\text{cat}}$ drops by one, and $W_O$ has one fewer useful dimension to work with. The loss gradient pushes $Y_i$ and $Y_j$ apart to restore rank.

**3. The loss landscape has many local optima**

The loss $L$ as a function of all heads' parameters has a complex landscape with many local minima. Different random initializations land in different basins of attraction. Within a basin, gradient descent refines the head toward one particular "skill." The landscape structure — shaped by the statistics of natural language — ensures that useful skills (local attention, copying, syntax tracking) correspond to low-loss basins.

This is the same mechanism by which convolutional filters in vision models learn edge detectors, texture detectors, and color detectors without being told to. The structure of the data + the architecture's inductive biases + gradient descent = emergent specialization.

**4. Gradient signal from different data aspects**

The cross-entropy loss aggregates prediction errors across all positions and all vocabulary items. Different types of prediction errors (failing to predict a repeated word, failing to match verb tense, failing to predict punctuation) create different gradient signals. Each head, starting from a different random point, is differently positioned to reduce different error types. The head that happens to be closest (in parameter space) to capturing bigram statistics gets pulled toward that role; the head closest to capturing long-range dependencies gets pulled toward that.

### Analogy: division of labor in a team

Imagine 8 people (heads) working on a group essay, each writing a paragraph (their $d_k$ contribution). They start by writing random drafts. After reading the full essay (the loss evaluates the concatenated output), each person gets feedback. The person whose random draft happened to be closest to a good introduction refines toward introductions. The person whose draft sounded like a conclusion refines that way. Nobody assigned roles — the feedback loop (gradient) + different starting points (random init) + the constraint that each person can only write one paragraph (bottleneck) drives specialization.

### Pros and cons of multi-head attention

**Pros:**
- Captures multiple relationship types simultaneously per position
- Each head's $d_k \times d_k$ attention is cheaper than one large $d_\text{model} \times d_\text{model}$ attention ($h$ heads of dimension $d_k$ costs $O(T^2 d_k h) = O(T^2 d_\text{model})$, same as single-head with $d_\text{model}$, but with more expressivity)
- Easier to parallelize — heads are independent, can be computed simultaneously on GPU
- Different heads can attend at different distances (some local, some global)
- Provides interpretability: you can visualize per-head attention patterns

**Cons:**
- Each head only sees a $d_k$-dimensional projection — can't attend based on full $d_\text{model}$ information. Information must be "compressed" into $d_k$ dimensions
- No guarantee that heads won't be redundant in practice (some heads empirically learn near-duplicate patterns and can be pruned)
- More hyperparameters ($h$, and now $h_\text{kv}$ with GQA)
- GQA reduces parameters but further limits KV expressivity — shared K,V means grouped query heads can't attend to information that requires different key representations

### GQA specifically: why share KV heads?

The hypothesis: the "what to look for" (Q) needs more diversity than "what's available to look at" (K) or "what to retrieve" (V). Two query heads sharing one KV head can still compute different attention patterns because their different $W_Q^{(i)}$ projections weight the shared key-space differently. Empirically, GQA with $h_\text{kv} = h/2$ loses very little quality vs. full MHA while saving 25% of attention parameters — valuable when you're constrained to 16MB.

### GQA implementation: how head sharing works in practice

Q has shape $(B, h, T, d_k) = (B, 8, T, 64)$, K and V have $(B, h_\text{kv}, T, d_k) = (B, 4, T, 64)$. The head counts don't match.

`F.scaled_dot_product_attention` with `enable_gqa=True` groups query heads in chunks of $h / h_\text{kv} = 2$:

    Q heads 0, 1   share   K/V head 0
    Q heads 2, 3   share   K/V head 1
    Q heads 4, 5   share   K/V head 2
    Q heads 6, 7   share   K/V head 3

Inside the Flash Attention CUDA kernel, this is **stride-based** — when computing attention for Q head $i$, the kernel reads K/V from head index $\lfloor i \cdot h_\text{kv} / h \rfloor$. No memory is duplicated; the kernel just indexes K/V differently per query head.

Before PyTorch added `enable_gqa`, the manual approach was to explicitly copy K/V:

    k = k.repeat_interleave(h // h_kv, dim=1)   # (B, 4, T, 64) → (B, 8, T, 64)
    v = v.repeat_interleave(h // h_kv, dim=1)   # wasteful memory copy

The `enable_gqa` flag achieves the same result without materializing the repeated tensors.

---

## Tensor shapes: how `bsz` arises and flows through the model

### Data loader creates the batch dimension (train_gpt.py, lines 486-494)

The token stream is a flat 1D array. `next_batch` carves out a chunk and reshapes:

    local_tokens = train_batch_tokens / (world_size × grad_accum_steps)
                 = 524,288 / (1 × 8)  = 65,536   (single GPU)

    local: shape (65,537,)                      # flat 1D, +1 for the shift
    x = local[:-1].reshape(-1, seq_len)         # (65,536,).reshape(-1, 1024) → (64, 1024)
    y = local[1:].reshape(-1, seq_len)          # same → (64, 1024)

Nobody explicitly sets $B = 64$. It falls out of:

$$B = \frac{\text{local\_tokens}}{\text{seq\_len}} = \frac{65{,}536}{1024} = 64$$

`x` and `y` are both `torch.Tensor` with dtype `int64` and shape $(B, T)$.

### Embedding adds the model dimension (train_gpt.py, line 701)

    input_ids: (64, 1024)        int64 — 2D tensor of token IDs
    tok_emb(input_ids): (64, 1024, 512)  float — each int replaced by its 512-dim vector

This is how $(B, T)$ of integers becomes $(B, T, d_\text{model})$ of floats. The shape $(B, T, d_\text{model})$ propagates unchanged through every Block until the final norm.

### Type system: Tensor vs nn.Parameter

Everything in the model is a `Tensor`. `nn.Parameter` is a subclass of `Tensor` with one distinction: it registers itself with the module so the optimizer can find it via `.parameters()`.

    nn.Parameter ⊂ Tensor

- **nn.Parameter**: all learned weights (embedding table, linear weights, `q_gain`, `attn_scale`, etc.)
- **Plain Tensor**: all intermediate activations (matmul outputs, attention scores, residuals)

Both participate in the computation graph. The difference is purely whether the optimizer updates them.

---

## Why `transpose(1, 2)` in attention

After projection and reshape:

    c_q(x)                          →  (B, T, h·d_k)   = (64, 1024, 512)
      .reshape(B, T, h, d_k)       →  (B, T, h, d_k)   = (64, 1024, 8, 64)
      .transpose(1, 2)              →  (B, h, T, d_k)   = (64, 8, 1024, 64)

`F.scaled_dot_product_attention` expects `(B, h, T, d_k)`. With T and $d_k$ as the last two dimensions, the per-head attention $Q_i K_i^\top$ is a matmul on the trailing dims:

$$(T, d_k) \times (d_k, T) \to (T, T)$$

PyTorch batches this matmul over both $B$ and $h$ simultaneously in one fused kernel call (512 independent $(1024 \times 64) \times (64 \times 1024)$ matmuls: $64 \text{ seqs} \times 8 \text{ heads}$).

If the shape were left as $(B, T, h, d_k)$, the matmul dims would be $(h, d_k) \times (h, d_k)^\top$ — semantically wrong.

---

## Why `.contiguous()` before reshape

`transpose()` in PyTorch doesn't move data — it changes **stride** metadata that tells PyTorch how to index memory. After transpose, the memory layout doesn't match the logical axis order. `reshape` requires contiguous memory (or compatible strides), so:

    y: (B, h, T, d_k)              # after SDPA, memory in (B, h, T, d_k) order
      .transpose(1, 2)             # (B, T, h, d_k) — logical, but memory still h-before-T
      .contiguous()                 # actual memory copy: rearranges data to match new axis order
      .reshape(B, T, d_model)       # merge last two dims: (B, T, 8, 64) → (B, T, 512)

Without `.contiguous()`, `reshape` would throw `RuntimeError: view size is not compatible with input tensor's size and stride`.

---

## What `detach().to("cpu")` means

Two separate operations:

- **`detach()`**: disconnects the tensor from the autograd graph. Every forward op records itself so `backward()` can trace gradients. `detach()` says "drop the graph — I only need the value." Prevents wasting memory on graph history for tensors used only for logging or saving.

- **`.to("cpu")`**: moves the tensor from GPU VRAM to CPU RAM. Equivalent to `.cpu()`.

Example from warmup snapshot (line 938):

    {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}

This saves an independent CPU copy of all weights: `detach` (drop graph) → `cpu` (move to RAM) → `clone` (make independent copy so later in-place GPU updates don't affect the snapshot).

---

## Sequence length: training vs evaluation vs production LLMs

### In this codebase: $T$ is fixed at eval time too

The eval function (lines 256-257) uses the same `train_seq_len`:

    x = local[:-1].reshape(-1, args.train_seq_len)   # same 1024
    y = local[1:].reshape(-1, args.train_seq_len)

The validation set is simply chunked into 1024-token sequences. No long-context handling.

### Why $T = 1024$ and not longer

Training efficiency. With a 10-minute wall-clock cap:

- Short sequences ($T = 1024$) → large batch ($B = 64$) → high GPU utilization (many parallel matmuls)
- Long sequences ($T = 200{,}000$) → tiny batch ($B = 1$) → GPU underutilized, fewer total tokens processed

The competition metric is BPB on fixed validation data, not long-range coherence. Short-context training maximizes tokens/second within the budget.

### Why production models claim 200K+ context

1. **They train on long sequences.** Often progressively: $4\text{K} \to 32\text{K} \to 128\text{K}$ during training stages.

2. **Nothing in the math fixes $T$.** The attention equations work for any $T$. RoPE computes positional frequencies on-the-fly: `torch.arange(seq_len, ...)`, so it handles any length natively.

3. **Memory is the constraint, not architecture.** Naive attention materializes the $T \times T$ score matrix: for $T = 200\text{K}$, that's $4 \times 10^{10}$ entries per head per layer. Flash Attention computes in $O(T)$ memory by tiling, making long context feasible. Compute remains $O(T^2)$.

4. **Additional techniques** for very long context: RoPE frequency scaling (NTK-aware, YaRN), sliding window attention, KV-cache compression, sparse attention. This codebase uses none — it's a minimal baseline.
