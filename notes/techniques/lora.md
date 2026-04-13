# LoRA (Low-Rank Adaptation)

Date: 2026-04-12

## Core idea

When you fine-tune a large pretrained model, the weight updates tend to live in a
low-dimensional subspace. You don't need to update all d×d parameters — you can
approximate the update with two much smaller matrices.

For a pretrained weight matrix W₀ ∈ R^{d_out × d_in}, instead of learning a full
update ΔW (which has d_out × d_in parameters), LoRA decomposes it:

    ΔW = B · A

where:
- A ∈ R^{r × d_in}    (the "down-projection")
- B ∈ R^{d_out × r}   (the "up-projection")
- r ≪ min(d_in, d_out) is the **rank**

The forward pass becomes:

    y = W₀x + ΔWx = W₀x + BAx

W₀ is **frozen** (no gradients). Only A and B are trained.

**Initialization matters:** A is initialized with random Gaussian, B is initialized to
**zero**. This means ΔW = BA = 0 at the start, so the model begins from exactly the
pretrained weights — training starts from the pretrained solution, not from a random
perturbation.

**Why it works:** neural network weight matrices are empirically low-rank during
fine-tuning. The directions that matter for task adaptation span a small subspace.
LoRA forces the update to be rank-r, which acts as both a compression and a regularizer
— it can't overfit to tiny fine-tuning datasets as easily as full-rank updates can.

## Parameter savings — concrete example

Say you have a linear layer with d_in=512, d_out=512 (like the query projection in
our model, `c_q`). That's **262,144** parameters.

With LoRA rank r=8:
- A: 8 × 512 = 4,096 params
- B: 512 × 8 = 4,096 params
- Total: **8,192 params** — a **32x reduction**

Apply this to c_q, c_v, and lm_head across all transformer blocks and you get ~100K
trainable parameters vs 17M total — about 0.6% of the model.

## "Rank" — two completely unrelated meanings

### Rank in LoRA (linear algebra)

The **rank** of a matrix is the dimension of the column space — the number of linearly
independent columns (or rows). A matrix M ∈ R^{m × n} has rank at most min(m, n).

When LoRA uses rank r=8, the update matrix ΔW = BA has at most 8 linearly independent
directions. Out of the full 512-dimensional space, you're only modifying 8 directions.
This is why it's called **low-rank** adaptation.

Typical LoRA ranks: 4, 8, 16, 32, 64. This project's TTT uses rank 8. Higher rank =
more expressive but more parameters and slower. Lower rank = cheaper but may underfit.

### Rank in distributed training (process rank)

In `torchrun` / DDP, **rank** is simply the **process ID**. Running
`torchrun --nproc_per_node=8 train_gpt.py` gives 8 processes with ranks 0-7.
Rank 0 is conventionally the "master" that handles logging, checkpointing, etc.
This is purely a naming convention from MPI (Message Passing Interface).

In the codebase: `if rank == 0: print(...)` — that's just "if I'm the first GPU process."
Nothing to do with linear algebra.

## Why LoRA resists catastrophic forgetting

### Why full fine-tuning is dangerous

A pretrained model's weights encode general knowledge across billions of tokens. These
weights sit at a point in a very high-dimensional parameter space that represents a
good solution for the pretraining distribution.

When you fully fine-tune on a small, narrow dataset (say, 500 tokens from one document
during TTT):

1. **All parameters are free to move.** With 17M trainable parameters and only 500 tokens
   of signal, the optimization is massively underdetermined. There are infinitely many
   directions the weights can move that reduce loss on this tiny dataset.

2. **Most of those directions destroy pretraining knowledge.** The optimizer picks the
   steepest descent direction for the fine-tuning loss. This direction is almost certainly
   not aligned with preserving performance on the pretraining distribution. With one
   gradient step, you're moving in a 17M-dimensional space based on 500 tokens of evidence.

3. **No constraint prevents overwriting.** The same neurons that encode general English
   grammar might get repurposed to memorize the specific document.

### Why LoRA is safer

LoRA constrains updates to a rank-r subspace (~8,192 params instead of 17M). This acts
as **implicit regularization**:

- The model physically *cannot* move in most directions in parameter space — it's
  restricted to a tiny subspace
- With fewer free parameters, there are fewer ways to overfit to the fine-tuning data
- The pretrained weights W₀ are literally frozen — they cannot be corrupted
- At worst, the LoRA adapter learns something unhelpful, and you can just discard it
  (set ΔW = 0) to get back the original model exactly

In our TTT setup, this is critical: we reset LoRA params between documents. If full
fine-tuning wrecked the general weights on document 1, restoring them requires keeping
a full checkpoint copy. With LoRA, the base weights never change — just zero out the
adapters.

## Comparison table: full fine-tune vs LoRA for TTT

| | Full fine-tune | LoRA |
|---|---|---|
| Params updated | All ~17M | rank × (in+out) per layer, ~100K |
| Backward speed | Full backward through all params | Only through LoRA params — much faster |
| Optimizer memory | Adam for 17M params (~130MB) | Adam for ~100K params (~1MB) |
| Catastrophic forgetting risk | High — few steps on 500 tokens can wreck general knowledge | Low — low-rank constraint is implicit regularization |
| Adaptation strength per step | Stronger | Weaker but safer |
| Our 3060 (6GB) | Feasible but tight on memory | Negligible memory overhead |

## Other PEFT (Parameter-Efficient Fine-Tuning) methods

| Method | How it works | Params added | SOTA status (2026) |
|--------|-------------|-------------|-------------|
| **Full fine-tuning** | Update all weights | 0 extra (but all trainable) | Gold standard for quality, impractical for large models |
| **LoRA** | Low-rank decomposition of weight updates | ~0.1-1% of original | Most widely used PEFT method |
| **QLoRA** | LoRA on 4-bit quantized base model | Same as LoRA, but base in 4-bit | Enables fine-tuning 65B+ models on single GPU |
| **DoRA** | Decompose into magnitude + direction, LoRA on direction only | Same as LoRA | Consistently beats LoRA across tasks |
| **Adapters** | Insert small bottleneck layers between existing layers | ~1-5% | Older, largely superseded by LoRA |
| **Prefix tuning** | Prepend learnable "virtual tokens" to K/V in attention | ~0.1% | Good for generation tasks, less popular now |
| **Prompt tuning** | Learn soft prompt embeddings prepended to input | Tiny (~100K) | Simple but weaker than LoRA |
| **IA3** | Learn element-wise rescaling vectors for K, V, and FFN | ~0.01% | Extremely lightweight, weaker than LoRA |
| **(b)GaLore** | Project gradients into low-rank subspace, full-rank updates | 0 extra params | Memory-efficient *pretraining*, not just fine-tuning |
| **ReFT** | Interventions on hidden representations, not weights | Small | Promising research direction (2024) |

Current landscape: LoRA is the de facto standard. QLoRA is the go-to when memory is
limited. DoRA is the strongest LoRA variant. Full fine-tuning with DeepSpeed/FSDP
remains the quality ceiling when you have the compute.

## DoRA (Weight-Decomposed Low-Rank Adaptation)

Liu et al., 2024. Observes that full fine-tuning changes both the **magnitude** and
**direction** of weight vectors, but standard LoRA conflates these two into a single
low-rank update, limiting expressiveness.

### The decomposition

Each weight column is decomposed into magnitude and direction:

    W = m · V/‖V‖

where:
- V is the **direction** matrix (unit-normalized columns)
- m is a learnable **magnitude** vector (one scalar per output dimension)

Then LoRA is applied only to the direction component:

    W' = (m + Δm) · (V + BA) / ‖V + BA‖

You train:
1. Δm — a vector of d_out scalars (cheap, unconstrained)
2. The usual LoRA matrices B, A (low-rank, for directional changes)

### Why it beats LoRA

In full fine-tuning, magnitude and direction updates have different learning dynamics.
LoRA's rank-r constraint forces both to share the same low-rank subspace. DoRA frees
magnitude to update independently (it's just a vector, very cheap) and reserves the
low-rank capacity entirely for directional changes.

Empirically, this closes ~40-60% of the gap between LoRA and full fine-tuning across
many benchmarks.

### Why not use DoRA in this project

1. **TTT speed budget.** We train LoRA for 1 gradient step per chunk across ~50K documents
   in 10 minutes. The extra column normalization in DoRA's forward pass (‖V + BA‖ per
   column) adds compute per step. With thousands of forward+backward passes in a tight
   time budget, this overhead matters.

2. **Marginal gain at rank 8 on a 17M model.** DoRA shines on larger models (7B+) where
   the gap between LoRA and full fine-tuning is wide. On our 17M model with rank 8, the
   difference is small — the model is already small enough that LoRA captures most of the
   useful update directions.

## Scaling factor: alpha / rank

### The problem

When you initialize LoRA, A has random values with some variance. As you change the rank,
the magnitude of BAx changes — higher rank means more terms are summed, which increases
the output variance.

If you double the rank from 8 to 16, the LoRA output roughly doubles in magnitude (more
random vectors summed). This means the learning rate that worked for rank 8 is too
aggressive for rank 16.

### The solution

    output = W₀x + (α/r) · BAx

- α (alpha) is a fixed hyperparameter (often set to r or 2r)
- Dividing by r normalizes out the rank dependency

If you set α = r, the scale is 1.0 and the rank cancellation means changing rank doesn't
change output magnitude. If α = 2r, the LoRA contribution is amplified by 2x.

### Relation to other scaling mechanisms

The principle is the same as other normalizations in the model: when you sum/dot-product
many terms, normalize to keep variance stable. But the specific formulas differ because
they address different operations:

| Where | Scaling | Why |
|-------|---------|-----|
| Attention | 1/√d_k | Dot product of two d_k-dim vectors has variance ~d_k; divide by √d_k → variance ~1 |
| LoRA | α/r | Sum of r rank-1 contributions; normalize so changing r doesn't change magnitude |
| RMSNorm | x / RMS(x) | Normalize activations to unit RMS after each layer, preventing magnitude drift across depth |

RMSNorm is applied to **activations** between layers. Attention scaling and LoRA scaling
are applied to specific **operations**. All solve the same fundamental problem (keeping
numbers in a well-behaved range) at different points in the computation.

The α/r form is partly principled (rank-normalization for hyperparameter stability) and
partly empirical (the specific value of alpha is tuned). It's not as clean as the 1/√d_k
derivation in attention, which follows directly from the variance of dot products. The
original LoRA paper chose this form so you can tune alpha once and freely change rank
without retuning learning rate.

## target_modules and gradient flow through frozen layers

### LoRA goes in the MIDDLE of the model

`target_modules` in peft (or manual wrapping) specifies which layers to apply LoRA to.
These are typically internal attention projections, NOT just the last layer:

    Embedding (frozen)
      → Block 0:
          → LayerNorm (frozen)
          → c_q: W₀x + BAx      ← LoRA here (A, B trainable)
          → c_k: W₀x             ← frozen, untouched
          → c_v: W₀x + BAx      ← LoRA here (A, B trainable)
          → attn output (frozen)
          → FFN (frozen)
      → Block 1: ... (same pattern)
      → LM head (frozen or LoRA)

The output shape of each LoRA-wrapped layer is identical to the original — it's a
drop-in replacement. The rest of the model sees no structural difference.

### How gradients flow through frozen layers

"Frozen" does NOT mean "gradient doesn't flow through." It means "we compute the
gradient of the loss w.r.t. the layer's input (needed for chain rule), but we don't
compute or store the gradient w.r.t. the frozen weight itself."

Concretely, for a frozen linear layer y = Wx:
- dL/dx = W^T · dL/dy — this IS computed, passed backward to earlier layers
- dL/dW = dL/dy · x^T — this is NOT computed (saves memory and compute)

In PyTorch, setting `param.requires_grad = False`:
1. PyTorch won't allocate a `.grad` tensor for this param (saves memory)
2. BUT the layer still participates in the computation graph
3. Gradients still flow THROUGH the layer to earlier layers via dL/dx
4. The frozen layer acts as a fixed function during backprop

So when you have frozen → LoRA → frozen:

    loss
      → frozen_layer_above: passes dL/dx backward, doesn't compute dL/dW
      → LoRA layer: computes and stores dL/dA and dL/dB — these get updated!
      → frozen_layer_below: would pass gradient through, but if nothing below
        it needs gradients, PyTorch stops early (saves compute)

The net effect is correct: only LoRA parameters get updated, but the gradients reaching
them account for the full forward computation through all the frozen layers above and
below.

**Efficiency:** PyTorch is smart about this. If no parameter before a frozen layer
requires gradients, the backward pass stops early — it doesn't waste compute propagating
gradients that nobody needs. This is why freezing most of the model and only training
LoRA is fast: the backward pass only needs to go from the loss back to the deepest
LoRA layer.

## Implementation: nn.Parameter vs nn.Linear

### What is nn.Parameter?

`nn.Parameter` is just a `torch.Tensor` with a flag that says "I'm trainable — include
me in model.parameters()."

    self.x = torch.randn(8, 512)              # raw tensor — optimizer won't see it
    self.x = nn.Parameter(torch.randn(8, 512)) # Parameter — optimizer WILL see it

When you call `optimizer = Adam(model.parameters())`, PyTorch walks through all
nn.Module attributes and collects everything that's an nn.Parameter. That's the only
difference — it's a registration mechanism.

### What is a tensor?

In PyTorch, "tensor" just means "n-dimensional array":
- 0-dim tensor = scalar
- 1-dim tensor = vector
- 2-dim tensor = matrix
- 3+ dim tensor = higher-order tensor

The term is used loosely in deep learning to mean "an array with any number of
dimensions." Mathematically, tensors have specific transformation properties under
coordinate changes, but in PyTorch it's just the generic container. Calling a matrix
a "tensor" is technically correct (a matrix is a rank-2 tensor) and is the convention
because the same code and APIs handle all dimensionalities uniformly.

### What is nn.Linear?

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            self.weight = Parameter(torch.empty(out_features, in_features))
            self.bias = Parameter(torch.empty(out_features)) if bias else None

        def forward(self, x):
            return x @ self.weight.T + self.bias

So nn.Linear is essentially a wrapper around nn.Parameter (for the weight) plus optional
bias plus a forward() that does the matmul. It IS a wrapper of nn.Parameter.

### Can you use nn.Linear for LoRA matrices?

Yes. This is functionally identical:

    self.lora_A = nn.Linear(d_in, rank, bias=False)   # weight: (rank, d_in)
    self.lora_B = nn.Linear(rank, d_out, bias=False)   # weight: (d_out, rank)

Many implementations do this. Hugging Face `peft` uses nn.Linear internally. Using
nn.Parameter directly is a style choice for clarity — makes explicit that LoRA is just
two matrices, not two "layers", and avoids the bias=True default footgun.

Consecutive nn.Linear layers (without bias or nonlinearity between them) is simply matrix
multiplication over consecutive weight matrices. nn.Linear(512, 8) then nn.Linear(8, 512)
computes x·W_A^T·W_B^T = x·(W_B·W_A)^T. The result is a rank-8 matrix — same as LoRA.

### Manual LoRA (minimal implementation)

    class LoRALinear(nn.Module):
        def __init__(self, base_linear, rank=8, alpha=1.0):
            super().__init__()
            self.base = base_linear
            self.base.weight.requires_grad_(False)  # freeze base

            d_out, d_in = base_linear.weight.shape
            self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
            self.scale = alpha / rank

        def forward(self, x):
            base_out = self.base(x)
            lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
            return base_out + lora_out

Usage — wraps an existing nn.Linear from the pretrained model:

    model.blocks[0].attn.c_q = LoRALinear(model.blocks[0].attn.c_q, rank=8)

You pass the existing layer, not raw shapes. The constructor reads dimensions from
base_linear.weight.shape.

Data flow for the query projection (d_in=512, d_out=512, rank=8):

    Input x: (B, T, 512)
        │
        ├──→ base_linear(x)  →  (B, T, 512)        # frozen W₀
        │
        └──→ x @ A.T @ B.T  →  (B, T, 512)         # LoRA path
                  │       │
              (B,T,512)@(512,8) = (B,T,8)            # compress to rank-8
                        then (B,T,8)@(8,512) = (B,T,512)  # expand back
        │
        sum → output: (B, T, 512)                    # same shape as original

### Libraries

| Library | Use case |
|---------|----------|
| **`peft` (Hugging Face)** | Most popular. Integrates with transformers, supports LoRA/QLoRA/DoRA/IA3/etc. |
| **`unsloth`** | Optimized LoRA/QLoRA — 2x faster, 60% less memory via custom CUDA kernels |
| **`axolotl`** | Fine-tuning framework built on peft with YAML configs |
| **`LLaMA-Factory`** | All-in-one fine-tuning with web UI, uses peft under the hood |
| **Manual PyTorch** | For custom setups like our TTT where you need tight control |

For standard fine-tuning (train once, save adapter, merge, deploy), use peft. For our TTT
(50K document resets, batched independent LoRA states, interleaved score-then-train),
manual PyTorch gives full control without fighting the library.

## In this project

Used for TTT (test-time training) during evaluation. Rank-8 LoRA adapters on lm_head,
c_q, c_v projections. Adam optimizer with lr=0.01, betas=(0.9, 0.95). One gradient step
per chunk. Reset between documents. See notes/techniques/ttt.md and
records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md.
