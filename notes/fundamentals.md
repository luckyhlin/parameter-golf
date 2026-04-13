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
through that layer. It means:

1. PyTorch won't allocate a .grad tensor for this param (saves memory)
2. BUT the layer still participates in the computation graph
3. Gradients w.r.t. the layer's **input** ARE computed and passed backward (needed for
   chain rule to reach earlier trainable params)
4. Gradients w.r.t. the frozen **weight** are NOT computed

For a frozen linear layer y = Wx:
- dL/dx = Wᵀ · dL/dy  ← IS computed, passed to earlier layers
- dL/dW = dL/dy · xᵀ   ← NOT computed (saves memory and compute)

PyTorch is smart: if no parameter before a frozen layer requires gradients, the backward
pass stops early. This is why training only LoRA adapters in the middle of a frozen model
is fast — backprop only runs from the loss to the deepest trainable parameter.

### Variance control across the model

Multiple mechanisms keep activation magnitudes stable. They all solve the same
fundamental problem (prevent numbers from blowing up or vanishing) at different points:

| Where | Scaling | Why |
|-------|---------|-----|
| Attention (train_gpt.py) | 1/√d_k | Dot product of two d_k-dim vectors has variance ~d_k; divide by √d_k → variance ~1 |
| LoRA | α/r | Sum of r rank-1 contributions; normalize so changing rank doesn't change output magnitude |
| RMSNorm (between layers) | x / RMS(x) | Normalize activations to unit RMS after each sublayer, preventing magnitude drift across depth |

RMSNorm operates on **activations between layers**. Attention scaling and LoRA scaling
operate on **specific operations** within a layer. All address the same principle.
