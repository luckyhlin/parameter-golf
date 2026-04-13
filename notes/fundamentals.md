# ML Fundamentals for Parameter Golf

## What are we predicting?

Next-token prediction. Given a sequence [t1, t2, ..., tn], the model outputs a
probability distribution over the vocabulary for each position. The ground truth
is the actual next token in the text.

    input_ids  = [t1, t2, t3, ..., tn]
    target_ids = [t2, t3, t4, ..., tn+1]   (shifted by one)

Loss is cross-entropy between the predicted distribution and the one-hot target.
(train_gpt.py line 724: `F.cross_entropy(logits.float(), targets, reduction="mean")`)

## The dataset

FineWeb — a large corpus of cleaned web pages. Each "document" is one web page.

- Training: ~8B tokens from 80 shards (`fineweb_train_*.bin`)
- Validation: fixed first 50K documents (~62M tokens, `fineweb_val_*.bin`)

Documents are concatenated into a flat token stream. The model trains on
contiguous chunks of `train_seq_len` (default 1024) tokens, regardless of
document boundaries during training. During eval, document boundaries matter
for techniques like TTT (see notes/techniques/ttt.md).

## BPB (bits per byte)

The competition metric. Cross-entropy loss is in nats (natural log). To make it
tokenizer-agnostic (so different vocab sizes are comparable):

    bits_per_token = loss / ln(2)
    tokens_per_byte = total_tokens / total_bytes
    bpb = bits_per_token × tokens_per_byte

A tokenizer with bigger vocabulary has fewer tokens for the same text (lower
tokens_per_byte) but each token carries more information (higher bits_per_token).
BPB normalizes this out.

## Tokenizer

Converts raw text → integer token IDs. Our tokenizer is SentencePiece BPE with
1024 vocabulary (`fineweb_1024_bpe.model`).

With only 1024 tokens, each represents very short text (individual characters,
common short words, frequent byte sequences). Leaderboard entries use larger
vocabularies (4096, 8192) — each token represents longer pieces, so the model
sees more text per fixed sequence length.

The tokenizer is NOT learned during training. It's a fixed preprocessing step.

## Embedding

A learned lookup table: `nn.Embedding(vocab_size, model_dim)` = a matrix of
shape [1024, 512]. Token ID 42 → row 42 → a 512-dimensional vector.

This is the model's input: raw integer IDs have no geometric structure, but
learned embeddings place semantically similar tokens near each other in the
512-dimensional space.

## Tied embeddings (train_gpt.py lines 666-688, 717-718)

The same matrix is used for both:
- **Input**: `tok_emb(input_ids)` — look up rows (token → vector)
- **Output**: `F.linear(x, tok_emb.weight)` — project hidden state back to
  vocabulary logits (vector → score per token)

Why tie? With vocab=1024, dim=512, the embedding is 524K params. An untied
lm_head adds another 524K. Tying saves those parameters — significant when
artifact size is constrained to 16MB.

Trade-off: input and output representations share the same space, slightly
limiting expressiveness. In practice, for small vocab sizes the savings
outweigh the cost.

When `tie_embeddings=False`, a separate `CastedLinear(model_dim, vocab_size)`
is created as lm_head (line 688).

## PyTorch building blocks

### Tensors

In PyTorch, "tensor" means "n-dimensional array":
- 0-dim tensor = scalar (e.g., `torch.tensor(3.14)`)
- 1-dim tensor = vector (e.g., shape [512])
- 2-dim tensor = matrix (e.g., shape [512, 512])
- 3+ dim tensor = higher-order tensor (e.g., activations shape [B, T, 512])

Mathematically, tensors have specific transformation properties under coordinate changes,
but in PyTorch it's just the generic container for numerical data. Calling a matrix a
"tensor" is technically correct (a matrix is a rank-2 tensor) and is the convention
because the same APIs handle all dimensionalities.

### nn.Parameter

`nn.Parameter` is a `torch.Tensor` with a flag that says "I'm trainable — include me
in model.parameters()." That's the only difference from a raw tensor.

    self.x = torch.randn(8, 512)              # raw tensor — optimizer won't see it
    self.x = nn.Parameter(torch.randn(8, 512)) # Parameter — optimizer WILL see it

When you call `optimizer = Adam(model.parameters())`, PyTorch walks through all
nn.Module attributes and collects everything that's an nn.Parameter. This is how the
optimizer knows which tensors to update — it's a registration mechanism.

### nn.Linear

A thin wrapper around nn.Parameter:

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            self.weight = Parameter(torch.empty(out_features, in_features))
            self.bias = Parameter(torch.empty(out_features)) if bias else None

        def forward(self, x):
            return x @ self.weight.T + self.bias

So nn.Linear = one nn.Parameter (the weight matrix) + optional bias + a forward() that
does the matrix multiply. It IS a wrapper of nn.Parameter.

Consecutive nn.Linear layers without nonlinearities between them just multiply their
weight matrices together: Linear(512, 8) then Linear(8, 512) computes
x·W₁ᵀ·W₂ᵀ = x·(W₂W₁)ᵀ — a single rank-8 linear map.

### Gradient flow through frozen layers

"Freezing" a parameter (`param.requires_grad = False`) does NOT stop gradient flow
through that layer. A frozen layer is treated as a **fixed function** — its weight is
a constant, not a variable. The chain rule still uses the frozen weight's VALUE as a
multiplier; it just doesn't compute the gradient OF the loss w.r.t. that weight.

For a frozen linear layer y = Wh (W frozen, h is input from a trainable layer below):

    Two partial derivatives exist:
      ∂y/∂h = W        (Jacobian w.r.t. INPUT — uses W's value as a constant multiplier)
      ∂y/∂W = h        (Jacobian w.r.t. WEIGHT — would tell us how to update W)

    "Frozen" means: compute ∂y/∂h (so gradient can propagate to earlier trainable params),
    but SKIP computing ∂y/∂W (since we won't update W anyway).

Scalar walkthrough:

    a = 3 (trainable),  w = 5 (frozen),  x = 2 (input)
    h = a·x = 6,  y = w·h = 30,  L = y² = 900

    dL/dy = 2y = 60
    dy/dh = w = 5           ← frozen weight's VALUE used as multiplier
    dL/dh = 60 × 5 = 300   ← gradient passes THROUGH frozen layer
    dh/da = x = 2
    dL/da = 300 × 2 = 600  ← stored and used to update a

    dy/dw = h = 6           ← SKIPPED (requires_grad=False)
    dL/dw = 360             ← never computed or stored

In PyTorch, `param.requires_grad = False`:
1. PyTorch won't allocate a .grad tensor for this param (saves memory)
2. BUT the layer still participates in the computation graph
3. Gradients w.r.t. the layer's INPUT are computed (frozen weight's value is the multiplier)
4. Gradients w.r.t. the frozen WEIGHT are not computed

PyTorch stops backprop early if no parameter before a frozen layer requires gradients.
This is why LoRA in the middle of a frozen model is fast — backprop only runs from the
loss to the deepest trainable parameter.

### Gradient control: requires_grad vs no_grad vs inference_mode

Three mechanisms at different granularities:

**Per-parameter (fine-grained): `param.requires_grad = False`**

    param.requires_grad = False      # attribute assignment
    param.requires_grad_(False)      # in-place method (same effect)

A permanent flag on a specific tensor. Means: don't compute/store .grad for this tensor.
But operations involving this tensor still build a computation graph for OTHER tensors
that DO require gradients. Gradient still flows through (see above). This is what LoRA
uses to freeze base weights while keeping LoRA params trainable.

**Context manager (coarse-grained): `torch.no_grad()`**

    with torch.no_grad():
        y = model(x)    # no computation graph built AT ALL

Temporarily disables gradient computation for ALL operations inside the block, regardless
of individual requires_grad flags. No graph stored → saves memory + speeds up forward.
Does NOT modify any parameter's requires_grad flag — when you exit, everything is normal.
Outputs CAN be used in gradient computation later (they just lack graph history).

**Context manager (strictest): `torch.inference_mode()`**

    with torch.inference_mode():
        y = model(x)

Like no_grad but stricter: tensors created inside are marked "inference tensors" that
CANNOT be used in any future gradient computation (even outside the block). Allows PyTorch
to skip version counting for autograd, making it slightly faster than no_grad. Using an
inference-mode tensor in a gradient computation later raises an error.

This is why TTT can't use inference_mode (train_gpt.py line 250): TTT needs to score a
chunk (forward) then train on it (backward on same tensors). inference_mode blocks this.

| | requires_grad=False | torch.no_grad() | torch.inference_mode() |
|---|---|---|---|
| Scope | One tensor | All ops in block | All ops in block |
| Persistent? | Yes | No (context) | No (context) |
| Graph built for other tensors? | Yes | No | No |
| Outputs usable in grad later? | Yes | Yes | **No** (error) |
| Speed benefit | Minimal | Moderate | Highest |

### Tensor dimension conventions: dim=, keepdim, unsqueeze, broadcasting

**dim= in reduction operations (0-indexed)**

dim=k means "collapse (aggregate over) axis k." For a tensor of shape (d_out, d_in):
- dim=0: aggregate over rows → result shape (d_in,) or (1, d_in) with keepdim
- dim=1: aggregate over columns → result shape (d_out,) or (d_out, 1) with keepdim
- dim=-1: always the last axis (same as dim=1 for 2D, but dim=1 ≠ dim=-1 for 3D+)

Example: V.norm(dim=1, keepdim=True) for V shape (d_out, d_in):
- Computes L2 norm of each row: sqrt(Σ_j V[i,j]²) for each row i
- keepdim=False → shape (d_out,) — axis 1 removed
- keepdim=True → shape (d_out, 1) — axis 1 kept as size 1

**keepdim=True exists for broadcasting.** Without it:

    norms = V.norm(dim=1)           # (d_out,)
    V / norms                        # (d_out, d_in) / (d_out,)
                                     # aligns from right: d_in vs d_out → MISMATCH

With it:

    norms = V.norm(dim=1, keepdim=True)  # (d_out, 1)
    V / norms                             # (d_out, d_in) / (d_out, 1) → broadcasts ✓

Same convention in NumPy (axis= instead of dim=, keepdims= with 's').

**unsqueeze(i): insert a size-1 dimension at position i**

    m = torch.tensor([1, 2, 3])     # shape (3,)
    m.unsqueeze(0)                    # shape (1, 3) — new axis at front
    m.unsqueeze(1)                    # shape (3, 1) — new axis at end
    m.unsqueeze(-1)                   # shape (3, 1) — same (last position)

**Broadcasting: align shapes from the RIGHT, size-1 dims stretch to match**

    m.unsqueeze(1) * V    # (3, 1) * (3, 4)
                           # dim 0: 3 matches 3 ✓
                           # dim 1: 1 broadcasts to 4 ✓
                           # result: (3, 4) — each row i of V scaled by m[i]

This is mathematically equivalent to diag(m) @ V, but without forming the d_out × d_out
diagonal matrix.

### Weight storage convention: (out, in)

nn.Linear(in, out) stores weight as shape (out, in), following mathematical convention:
a linear map W: R^{in} → R^{out} is written y = Wx with W ∈ R^{out×in}.

The forward applies x @ W.T:
- x shape (..., in), W.T shape (in, out), result (..., out)
- The W.T transpose is free (view, not copy — just changes stride metadata)

**Matmul broadcasting:** the @ operator treats the last two dims as matrix dimensions and
broadcasts over all leading dims. So (B, T, in) @ (in, out) means: the (in, out) matrix
is applied independently to each element in D = (B, T). The weight has no batch dims —
broadcasting replicates it across D automatically. This is what makes right-multiply clean
for arbitrary batch shapes.

Why not W on the left (W @ x.T)? For x shape (B, T, in):
- .T reverses ALL dims: (B, T, in) → (in, T, B) — errors in PyTorch >=1.11 for dim > 2
- .mT transposes LAST TWO dims: (B, T, in) → (B, in, T)
- Either way, the result has batch dims in wrong positions, requiring extra reshaping.

In math/derivations, we write y = Wx with x ∈ R^{d_in} (no batch dims). Clean for single
vectors. In code, x has shape (B, T, d_in), so right-multiply x @ W.T handles D = (B, T).

### Jacobian

For a function f: R^n → R^m, the **Jacobian** is the m×n matrix of all partial derivatives:
J_{ij} = ∂f_i/∂x_j.

For a linear layer y = Wx (W ∈ R^{m×n}, x ∈ R^n): y_i = Σ_j W_{ij} x_j, so
∂y_i/∂x_j = W_{ij}. The Jacobian ∂y/∂x IS the weight matrix W itself.

This is why "passing gradient through a frozen layer" is cheap — the Jacobian is just W,
which is already in memory from the forward pass. No extra computation needed: we multiply
the incoming gradient by W (the same matrix), skip computing ∂L/∂W (which would be an
additional matmul we don't need since W won't be updated).

### Variance control across the model

Multiple mechanisms keep activation magnitudes stable at different points:

| Where | Scaling | Why | Derivation |
|-------|---------|-----|------------|
| Attention (train_gpt.py) | 1/√d_k | Keep softmax inputs near unit variance | Principled: Var[q·k] = d_k |
| LoRA | α/r | Keep learning dynamics stable across ranks | Empirical: effective LR scales ~linearly with r |
| RMSNorm (between layers) | x / RMS(x) | Normalize activations to unit RMS | Principled: force RMS = 1 |

RMSNorm operates on **activations between layers**. Attention scaling and LoRA scaling
operate on **specific operations** within a layer. All address the same principle.
