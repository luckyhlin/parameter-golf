# Debugging Log

## 2025-04-14: bf16 GLA attention — impossibly low loss (tensor reshape bug)

### Symptom

When the GLA chunk-wise attention loop runs in bf16 (under `torch.autocast`), training loss drops to 0.07 and val_bpb drops to 0.04 by step ~100. When the same loop runs in fp32 (inside `autocast(enabled=False)`), results are normal (val_bpb ~1.82). Both use `torch.compile(fullgraph=True)`.

### Relevant files

- `train_gpt_linear.py` — the GLA training script (current code has the fp32/working version)
- `train_gpt.py` — baseline softmax-attention script, used as ground truth for loss/eval code
- `debug_gla_nan.py` — pre-existing debug reproducer with per-line NaN checking

### Training logs

- `logs/gla-0413-1541.txt` — fp32 version, val_bpb 1.82 (correct)
- `logs/gla-bf16-0413-1706.txt` — bf16 version, val_bpb 0.04 (broken)
- `logs/gla-bf16-0413-1706_losses.csv` — per-step CSV: loss goes 6.9 → 2.88 (step 50) → 0.07 (step 100)

### Diagnostic test scripts

All scripts preserved in `debug_scripts/bf16_gla_reshape_bug/`. Each script tests one hypothesis:

| Script | What it tests | Key result |
|--------|--------------|------------|
| `test_gla_causality.py` | Standalone chunk loop: perturbation test + decay matrix inspection across {fp32,bf16} × {eager,compiled} × {autocast on/off} + extreme gate values | All pass. Causal mask works. No compile artifact. |
| `test_gla_causality2.py` | Full GLA module and full GPT model perturbation test at T=256 and T=1024, including per-position loss check | All pass. Full model is causal at init. |
| `test_gla_loss_compile.py` | Compiled vs eager loss comparison; bf16 vs fp32 loss comparison; logit-level diff; 10-step mini-training with compiled/eager cross-check | Compiled ≈ eager (diff 5.8e-5). bf16 ≈ fp32 at init. 10 steps normal. |
| `test_gla_crosseval.py` | **Definitive test**: load bf16-trained `final_model.pt`, evaluate with both bf16 and fp32 forwards | bf16: val_bpb=0.04. fp32: val_bpb=9.88. Smoking gun. |
| `test_gla_trained_weights.py` | Perturbation test with trained weights; intermediate diagnostics (decay matrix, state, attn); logit comparison | Perturbation LEAKS with trained weights. Decay mask is fine. Logit argmax acc: bf16=98.7%, fp32=0.6%. |
| `test_gla_trace_leak.py` | Step-by-step trace through GLA forward, checking every intermediate for cross-position diff | All intermediates zero diff. But used the CORRECT reshape — revealing the bug is in the reshape, not the loop. |

### Investigation timeline

#### Phase 1: Rule out causal mask / decay / compile hypotheses

**Test** (`test_gla_causality.py`): Extracted the chunk loop as a standalone function. Tested all 5 configurations:

```
Config                       before     at+after   result
fp32 eager                   0.00e+00   4.40e+04       OK
bf16 eager                   0.00e+00   4.42e+04       OK
bf16 autocast                0.00e+00   4.42e+04       OK
bf16 ac+compiled             0.00e+00   4.42e+04       OK
fp32 compiled                0.00e+00   4.40e+04       OK
```

Decay matrix upper triangle: exact zero in all cases, even with gates clamped at -1.0 where raw exp reaches 2.3×10²⁷. The fp32→bf16 cast after masking preserves zeros because 0.0 * any_value = 0.0 in fp32, and 0.0_fp32 → 0.0_bf16 exactly.

**Conclusion**: The chunk loop itself is correct. The causal mask works. torch.compile is not the issue.

#### Phase 2: Rule out full-model interaction

**Test** (`test_gla_causality2.py`): Tested the full GLA module (including projections, gates, output gate) and the full GPT model, both eager and compiled, at T=1024:

```
GPT bf16 autocast eager             before=0.00e+00  at+after=2.80e+00  OK
GPT bf16 autocast compiled          before=0.00e+00  at+after=2.82e+00  OK
```

**Conclusion**: Full model is causal at initialization. The leak doesn't come from the module/model structure at init.

#### Phase 3: Compare eval_val and loss code against baseline

Side-by-side diff of `train_gpt.py` vs `train_gpt_linear.py`: `GPT.forward()` is identical (same logit_softcap, same cross_entropy). `eval_val()` is byte-for-byte identical. `build_sentencepiece_luts()` is identical. The loss computation is correct and matches the working baseline.

#### Phase 4: Test compiled vs eager loss values

**Test** (`test_gla_loss_compile.py`): Monkey-patched the bf16 forward onto the model. Compared:

```
bf16 eager  loss: 6.941721
bf16 compiled loss: 6.941663  (diff: 5.8e-05)
fp32 eager loss: 6.941721
bf16 eager loss: 6.941721  (diff: 0.0)
```

Also ran 10 training steps: compiled and eager losses track within ~0.5 (expected from the parameter update between measurement points). No anomalous drop in 10 steps.

**Conclusion**: torch.compile doesn't change the loss. bf16 and fp32 produce identical loss at init. The anomaly must develop during training.

#### Phase 5: Cross-evaluate trained weights (BREAKTHROUGH)

**Test** (`test_gla_crosseval.py`): Loaded `final_model.pt` (the bf16-trained checkpoint, 328 steps). Evaluated on 256 val sequences with both forward paths:

| Forward | val_loss | val_bpb | Tokens |
|---------|----------|---------|--------|
| bf16 eager | 0.0645 | 0.0382 | 262144 |
| fp32 eager | **16.6771** | **9.8758** | 262144 |
| bf16 compiled | 0.0645 | 0.0382 | 262144 |
| fp32 compiled | 16.6708 | 9.8720 | 262144 |

Gate biases are still near initialization (~5.0, range 4.54–5.49).

**Conclusion**: The model learned weights that give impossibly low loss in bf16 but catastrophically bad loss in fp32. torch.compile doesn't change either result. The two forward paths produce fundamentally different outputs on the same weights.

#### Phase 6: Perturbation test with trained weights

**Test** (`test_gla_trained_weights.py`): Ran the causality perturbation test using the trained checkpoint:

```
Block 0 bf16: before=4.53e+00 at+after=1.79e+01 LEAK!
Block 4 bf16: before=6.14e+00 at+after=3.11e+01 LEAK!
```

But intermediate diagnostics showed all causal mechanisms working:
- `decay MASKED upper max: 0.0000e+00` — causal mask correct
- `attn*decay upper max: 0.0000e+00` — masked attention correct
- `state bf16 cast error: 0.0000e+00` — no precision loss in state

Gate-induced cumulative log values reach -51 (gates ~-0.8/position), giving raw decay values up to 2.1×10²² above the diagonal. But masking zeros them before the bf16 cast.

Logit comparison:
```
bf16: loss=0.0682  argmax_acc=0.9873
fp32: loss=16.3349  argmax_acc=0.0059
Logit diff: max=46.75  mean=5.90
bf16 top-5 at pos 0: [13.6, 3.7, 0.34, 0.34, 0.25]  (sharply peaked)
fp32 top-5 at pos 0: [11.6, 10.8, 10.7, 9.6, 9.4]    (flat)
```

**Conclusion**: The leak is real but NOT in the chunk loop. Something between the chunk loop and the final output creates cross-position dependencies with trained weights.

#### Phase 7: Step-by-step trace (PINPOINT)

**Test** (`test_gla_trace_leak.py`): Manually replicated every step of the bf16 forward, checking each intermediate for cross-position differences:

```
q after c_q:     before_pos=0.0000e+00  ok
k after c_k:     before_pos=0.0000e+00  ok
v after c_v:     before_pos=0.0000e+00  ok
q after rms_norm: before_pos=0.0000e+00  ok
log_gate:        before_pos=0.0000e+00  ok
chunk 0-6:       out_diff=0, state_diff=0, qkv_diff=0, lg_diff=0  (all zero)
chunk 7:         out_diff=4040 (expected — contains perturbed position)
After rms_norm:  diff = 0.0000e+00
g_proj diff:     0.0000e+00
Final output:    0.0000e+00   ← NO LEAK!
```

But this trace used `y.reshape(B, H, T, d).transpose(1,2).reshape(B, T, D)` — the CORRECT reshape. The bf16 forward uses `y.transpose(1,2).reshape(B, T, D)` on the 5D tensor. The difference IS the bug.

**Conclusion**: The leak is in the final reshape/transpose, not in any numerical operation.

### Root cause: 5D tensor transpose/reshape produces wrong position mapping

**The bug** is in how the bf16 forward converts the chunk-loop output from `(B, H, NC, C, d)` back to `(B, T, D)`.

**bf16 version (BUGGY), `train_gpt_linear.py` line 629–630 in the bf16 log:**
```python
y = F.rms_norm(output, (d,))                                    # output: (B, H, NC, C, d) — 5D
y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)    # transpose 5D → wrong!
```

**fp32 version (CORRECT), `train_gpt_linear.py` line 634–635:**
```python
y = F.rms_norm(output.reshape(bsz, H, seqlen, d), (d,)).bfloat16()  # merge NC+C → 4D first
y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)         # transpose 4D → correct
```

### Why this breaks causality

`transpose(1, 2)` on the 5D tensor `(B, H, NC, C, d)` swaps `H` and `NC`, producing `(B, NC, H, C, d)`. When this is then flattened to `(B, T, D)`, the `C` and `d` dimensions are interleaved with `H`. Concretely, with H=8, C=64, d=64 (D=512):

- Each output "position" spans `D = 512` elements in the flattened `(NC, H, C, d)` layout
- 512 elements = 1 head × 8 positions × 64 dims = data from **8 consecutive sequence positions** from a single head
- Output position 0 contains data from actual positions 0–7 (head 0)
- Output position 1 contains data from positions 8–15 (head 0)
- Output position 8 contains data from positions 0–7 (head 1)
- etc.

The correct 4D path `reshape(B, H, T, d).transpose(1, 2)` produces `(B, T, H, d)`, where each output position contains data from exactly **one** sequence position across all heads.

Verified with a concrete small example (H=8, NC=2, C=3, d=2):
```
bf16 (BUGGY):  position 0 gets data from positions [0, 1, 2]  — MIXED!
fp32 (CORRECT): position 0 gets data from position  [0]        — correct
```

### Why perturbation tests pass at initialization

`self.proj` (the output projection of GLA attention) is **zero-initialized** (`_zero_init = True`). At initialization, the entire attention contribution is zero regardless of input. The position-mixing has no observable effect because `proj(anything) = 0`. The perturbation test shows zero diff trivially.

Once training begins and `proj` learns non-zero weights, the mixed-position data propagates through `proj` and enters the residual stream. The model discovers and exploits the cross-position information leak.

### Why the model achieves near-perfect prediction

With the wrong reshape, each output "position" t has access to data from 8 actual positions within its chunk. Since the value vectors encode information about input tokens (through the v projection), the model can learn to:

1. Encode token identity in the attention values
2. Extract the target token's information from the wrong-reshape dimensions via `proj`
3. Predict the next token using data that should not be available

The model converges to loss ~0.07 (94% argmax accuracy on vocab 1024) within ~100 training steps. This is confirmed by logit inspection: bf16 produces sharply peaked logits (top logit ~13.6, second ~3.7), while fp32 on the same weights produces flat logits (top-5 all between 9–12).

### The fix

Replace the bf16 version's reshape sequence with the fp32 version's approach — merge `NC` and `C` before transposing:

```python
# Before (BUGGY):
y = F.rms_norm(output, (d,))
y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

# After (FIXED):
y = F.rms_norm(output.reshape(bsz, H, seqlen, d), (d,))
y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
```

The rms_norm normalizes over the last dim `(d,)` and is applied per-vector, so it doesn't matter whether the tensor is 5D or 4D — the normalization is identical.

### Lesson

Transposing a 5D tensor and then reshaping to 3D can silently mix semantic dimensions. The operation `(B, H, NC, C, d).transpose(1,2).reshape(B, T, D)` is NOT equivalent to `(B, H, NC, C, d).reshape(B, H, T, d).transpose(1,2).reshape(B, T, D)` because the former interleaves `C` with `H` while the latter correctly merges `NC` and `C` first. This bug was invisible at initialization due to zero-initialized output projections and only became observable after training.
