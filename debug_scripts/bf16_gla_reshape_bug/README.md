# Debug scripts: bf16 GLA reshape bug (2025-04-14)

Diagnostic scripts used to track down why the bf16 GLA attention produced
impossibly low loss (val_bpb 0.04 vs expected ~1.8).

**Root cause**: 5D tensor transpose/reshape at the end of the GLA forward
mixed sequence positions within each chunk. See `notes/debugging.md` for
the full writeup.

## Scripts (in investigation order)

All scripts should be run from the repo root:
```bash
cd /media/lin/disk2/parameter-golf
.venv/bin/python debug_scripts/bf16_gla_reshape_bug/<script>.py
```

### 1. `test_gla_causality.py`
**Tests**: Standalone chunk loop in {fp32,bf16} × {eager,compiled} × {autocast on/off}.
Perturbation test (modify one position, check earlier positions for leakage).
Decay matrix upper-triangle inspection. Extreme gate value stress test.

**Runtime**: ~10s

**Result**: All pass. Proved the chunk loop and causal mask are correct in isolation.

### 2. `test_gla_causality2.py`
**Tests**: Full GLA module (with projections, gates, output gate) and full GPT
model, both eager and compiled, at T=256 and T=1024. Includes per-position loss
check to rule out target-token leakage.

**Runtime**: ~20s

**Result**: All pass. The full model is causal at initialization. Note: this
uses the model's DEFAULT forward (fp32), not the bf16 forward — which is why
it passes. This was a key misdirection during investigation.

### 3. `test_gla_loss_compile.py`
**Tests**: Monkey-patches the bf16 GLA forward onto the model. Compares
compiled vs eager loss values, bf16 vs fp32 loss, logit-level diffs, and
runs a 10-step mini-training to check for anomalous loss drop.

**Runtime**: ~3.5 min (includes torch.compile warmup + 10 training steps)

**Result**: Compiled ≈ eager loss (diff 5.8e-5). bf16 = fp32 loss at init.
10 training steps show normal behavior. Proved torch.compile is not the issue
and the anomaly requires more training to manifest.

### 4. `test_gla_crosseval.py`
**Tests**: Loads the bf16-trained checkpoint (`final_model.pt`) and evaluates
on 256 validation sequences with 4 configurations: {bf16,fp32} × {eager,compiled}.

**Runtime**: ~2.5 min

**Result**: THE SMOKING GUN. bf16 gives val_bpb=0.04, fp32 gives val_bpb=9.88
on the exact same weights. Proved the two forward paths produce fundamentally
different outputs. Also prints gate bias statistics.

**Requires**: `final_model.pt` from the bf16 training run must exist in the repo root.

### 5. `test_gla_trained_weights.py`
**Tests**: Perturbation test with the trained checkpoint. Intermediate diagnostics
for the chunk loop (decay matrix, state, attention). Logit comparison (bf16 vs
fp32 argmax accuracy, top-5 logit values).

**Runtime**: ~2s

**Result**: LEAK detected with trained weights (before=4.53). But all intermediate
causal mechanisms are correct (masked decay upper triangle = 0, state cast error = 0).
Logit argmax acc: bf16=98.7%, fp32=0.6%. Narrowed the bug to outside the chunk loop.

**Requires**: `final_model.pt`

### 6. `test_gla_trace_leak.py`
**Tests**: Step-by-step trace through the GLA forward, checking EVERY intermediate
tensor (q, k, v, gates, per-chunk outputs, states, rms_norm, g_proj, final output)
for cross-position differences.

**Runtime**: ~2s

**Result**: All intermediates show zero diff at positions before the perturbation.
BUT — this script used the CORRECT reshape (`reshape(B,H,T,d).transpose(1,2)`),
not the bf16 version's buggy 5D reshape. The zero diffs at every step, combined
with the non-zero diffs in test 5, proved the bug is specifically in the
transpose/reshape sequence, not in any numerical computation.

**Requires**: `final_model.pt`

## The bug (one-liner summary)

```python
# BUGGY (bf16 version):
y.transpose(1, 2).reshape(B, T, D)  # on 5D tensor (B, H, NC, C, d)

# CORRECT (fp32 version):
y.reshape(B, H, T, d).transpose(1, 2).reshape(B, T, D)  # merge NC+C first
```

The 5D transpose swaps H and NC, leaving C interleaved with H in the flat layout.
Each output "position" then contains data from 8 different sequence positions.
