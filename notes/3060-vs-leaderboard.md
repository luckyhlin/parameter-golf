# RTX 3060 vs Leaderboard (8×H100) Compute Budget

Date: 2025-04-10

## Throughput comparison

| | Leaderboard (8×H100) | Our setup (1×3060) | Ratio |
|---|---|---|---|
| GPUs | 8×H100 SXM | 1×RTX 3060 Laptop Max-Q (6GB) | |
| TRAIN_BATCH_TOKENS | 524,288 | 65,536 | 8× fewer |
| grad_accum_steps | 1 (8//8) | 8 (8//1) | |
| Step time | ~86ms | ~849ms | ~10× slower |
| Steps in 10 min | ~7,000 | ~707 | ~10× fewer |
| Tokens/step | 524,288 | 65,536 | 8× fewer |
| **Tokens/sec** | **~6.1M** | **~77K** | **~79×** |

## Where the ~79× comes from

1. **8× from batch size / GPU count**: leaderboard processes 524K tokens per
   step across 8 GPUs; we process 65K on 1 GPU. Each optimizer update sees
   8× fewer tokens, meaning noisier gradient estimates on top of fewer steps.

2. **~10× from H100 vs 3060 per-GPU speed**: H100 SXM has ~990 TFLOPS bf16
   (dense tensor cores) vs 3060's much lower throughput. The gap is only ~10×
   (not 20-40×) because our tiny micro-batches (8 sequences × 1024 tokens)
   underutilize both GPUs — the H100's massive cores are starved more
   proportionally.

## Implications for depth experiments

Leaderboard entries go from 9L → 11L + MLP3× because at 8×H100 throughput,
the extra compute per step (~86ms → ~100ms) still leaves ~6,000 steps in 10
minutes — enough that the capacity gain outweighs the step loss.

On 1×3060, adding 2 layers would push step time from ~849ms to ~1,040ms
(estimated), reducing steps from ~707 to ~577. Whether extra capacity
compensates for 130 fewer steps is an empirical question — the trade-off is
harsher at our throughput level.

## What we lose beyond raw speed

- **Gradient quality**: 8× smaller effective batch → noisier gradient estimates
  per step. Leaderboard models get both more steps AND better gradients.
- **Total tokens seen**: leaderboard sees ~3.7B tokens (7000 × 524K); we see
  ~46M (707 × 65K). That's ~80× fewer tokens of training data.
- **Warmdown effectiveness**: warmdown schedule assumes enough steps for gradual
  decay. With only 707 steps and warmdown_iters=1200, warmdown barely activates
  (only kicks in when estimated remaining steps < 1200).
