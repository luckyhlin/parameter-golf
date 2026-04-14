# Gated Linear Attention (GLA)

## Date: 2026-04-13

## Motivation

Standard softmax attention has O(T²) computation and O(T) memory (with FlashAttention).
For this project (T=1024, d=64), the quadratic cost is moderate, but exploring O(T) alternatives
is worthwhile to see if comparable val_bpb is achievable with less compute.

## How It Works

### Core recurrence

Replace softmax(QK^T)V with a gated recurrent state:

```
S_t = gate_t * S_{t-1} + k_t^T @ v_t    (state update: d×d matrix)
o_t = q_t @ S_t                            (query the state)
```

- `S_t` is a d×d matrix (head_dim × head_dim = 64×64 in our case)
- `gate_t ∈ (0, 1)` is a data-dependent scalar that controls how much history to retain
- No softmax, no quadratic QK^T materialization

### Complexity

- Per position: O(d²) for the state update and query → O(T·d²) total
- Compare standard attention: O(T²·d) total
- Linear attention wins when T >> d. At T=1024, d=64: both are ~67M ops, so **no speedup at this scale**

### Chunk-wise parallelism (what we implemented)

The pure recurrence is sequential and GPU-unfriendly. The chunk-wise algorithm splits T into NC chunks of size C:

1. **Intra-chunk** (parallel): O(C²) quadratic attention within each chunk. Since C is small (64), this is fast.
2. **Inter-chunk** (sequential loop over NC chunks): Propagate the d×d state between chunks. Each step is O(d²).

Total: O(T·C + T·d²/C). With C=d=64, this is O(T·d) — same as standard attention at this scale, but the constant factors differ.

### Data-dependent gating

The gate is produced by a learned projection: `gate_t = sigmoid(W_gate · x_t + b_gate)`.

- `F.logsigmoid` is used for numerical stability (avoids log(sigmoid(x)) overflow)
- Cumulative sum of log-gates gives the total decay between any two positions
- Gate bias initialized to 5.0 → initial gate ≈ 0.993 (slow decay, ~700 token effective context)

### No RoPE needed

Unlike softmax attention, GLA does not use RoPE. Positional information is encoded implicitly
by the gating mechanism — the decay naturally makes recent tokens more influential.

## Implementation: `train_gpt_linear.py`

### Changes from `train_gpt.py`

| Component | Original | Linear variant |
|-----------|----------|----------------|
| Attention class | `CausalSelfAttention` | `GatedLinearAttention` |
| Positional encoding | RoPE (`Rotary` + `apply_rotary_emb`) | Implicit via gating (removed RoPE) |
| SDP backends | Flash SDP enabled | Not used (no SDPA call) |
| New hyperparameter | — | `CHUNK_SIZE` (default 64) |
| New projection | — | `gate_proj`: dim → num_heads (with bias) |
| Extra params per layer | 0 | 512 × 8 + 8 = 4,104 (gate weight + bias) |

### GQA handling

K, V projections still use `num_kv_heads` for parameter savings. At runtime, K and V are
expanded to full `num_heads` via `.expand()`. Each Q head gets its own gate and therefore
its own diverging state, even within the same KV group.

### Running

Same command as `train_gpt.py`, just change the script name:

```bash
python train_gpt_linear.py
# or with custom chunk size:
CHUNK_SIZE=32 python train_gpt_linear.py
```

### Lessons from canonical `fla` library (2026-04-13)

First attempt diverged (loss → NaN) due to missing stability techniques from the
canonical `flash-linear-attention` (fla) implementation:

1. **Gate logit normalizer = 16 (critical).**
   Raw `F.logsigmoid` gives values like -0.05 per position. Over a 64-token chunk,
   cumulative decay = exp(-3.2) ≈ 0.04 — nearly total state erasure each chunk.
   Dividing by 16: exp(-0.2) ≈ 0.82 retention per chunk. Prevents both
   forward-pass magnitude explosion and backward gradient explosion through the
   recurrent state.

2. **Output gate: `RMSNorm(attn_out) * SiLU(g_proj(x))`.**
   The canonical implementation does NOT just normalize — it multiplies the
   normalized output by a learned sigmoidal gate from a separate projection.
   This is like the gating in Mamba/LSTM: it lets the model learn to suppress
   or amplify the attention contribution per-dimension.

3. **Low-rank gate projection: `dim → 16 → num_heads`.**
   Gate goes through a bottleneck MLP, not a single linear layer.
   Reduces parameters and regularizes the gate.

4. **No QK RMS norm in canonical** — but we keep ours since it was in the
   original softmax attention and helps stability.

### Debugging NaN — Root Cause Found (2026-04-13)

**Definitive root cause:** `torch.exp()` overflow in the decay matrix computation.

The gate projection can produce extreme logits for certain input tokens (e.g., gate_logit = -35.7
despite bias=5.0, because `W2 @ W1 @ x` has std ~2.0). After `logsigmoid(-35.7) / 16 = -2.23`
per position, the cumulative sum over 64 positions reaches -98.8. The decay diff
`cum_lg[i] - cum_lg[j]` reaches 97.4, and **`exp(97.4) = 4.6e42`** overflows float32 max
(3.4e38), producing Inf → NaN when multiplied with the causal mask zero.

**Debugging method:** `torch.autograd.detect_anomaly()` + per-line NaN checks inside the
GLA forward (monkey-patched in `debug_gla_nan.py`). This is the standard approach —
NOT weight/gradient dumps after the fact.

**The fix:** `log_gate = log_gate.clamp(min=-1.0)`. This caps per-position decay at
`exp(-1) ≈ 0.37` (63% decay per position), so the max cumsum over 64 positions is -64,
and the max diff is 64 → `exp(64) ≈ 1.7e27`, safely within float32.

**Other fixes retained:**
- fp32 computation inside the chunk loop (avoids bf16 precision loss)
- fp32 state accumulation
- V normalization (bounds state magnitude)
- g_proj/gate_proj routed to Adam (not Muon)
- gate_proj[1].bias initialized to 5.0
- NaN early-stopping safety net

### Concerns

1. **T=1024 is short** — linear attention's advantage over FlashAttention only shows at T >> d.
   At T=1024, d=64, the theoretical FLOP counts are similar.
2. **torch.compile compatibility** — the chunk loop has NC=16 iterations (fixed), which
   `torch.compile` should unroll. But untested at time of writing.
3. **Quality gap** — linear attention typically underperforms softmax attention at small scale.
   The gap narrows with model size.

## Related approaches

- **RetNet**: Fixed exponential decay (not data-dependent). Simpler but less expressive.
- **Mamba/Mamba-2**: SSM-based, equivalent to a form of linear attention with structured gating.
- **RWKV-6/7**: Linear RNN with time-mixing. Competitive at scale.
- **Based**: Hybrid of linear attention + sliding-window softmax.
