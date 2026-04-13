# Experiment Runs

Hardware: RTX 3060 Laptop Max-Q (6GB VRAM), single GPU, 1×`torchrun --nproc_per_node=1`

Tokenizer: `fineweb_1024_bpe.model` (vocab_size=1024), dataset: `fineweb10B_sp1024`

## Run Log

| RUN_ID | Date | Change from baseline | Steps | step_avg | val_bpb | int8_zlib val_bpb | Peak Mem | Submission size |
|--------|------|---------------------|-------|----------|---------|-------------------|----------|-----------------|
| run-001 | 2025-04-10 | GRAD_ACCUM_STEPS=64 (patched) | 505 | 1189ms | 1.6306 | 1.6358 | 464 MiB | 9.97 MB |
| baseline_sp1024 | 2025-04-10 | Baseline (grad_accum=8) | 666 | 902ms | 1.5746 | 1.5772 | 1891 MiB | 11.03 MB |
| leaky_relu | 2025-04-10 | LeakyReLU(0.5)² | 707 | 849ms | 1.5530 | 1.5556 | 1891 MiB | 11.28 MB |
| leaky_relu_11L | 2025-04-11 | LeakyReLU(0.5)² + NUM_LAYERS=11 | 517 | 1162ms | 1.6170 | 1.6230 | 2216 MiB | 12.22 MB |
| leaky_relu_11L_mlp3x | 2025-04-11 | LeakyReLU(0.5)² + NUM_LAYERS=11 MLP_MULT=3 | 508 | 1182ms | 1.5980 | 1.6028 | 2501 MiB | 15.25 MB |

Common settings: TRAIN_BATCH_TOKENS=65536, VAL_BATCH_SIZE=65536, grad_accum_steps=8 (default 8//1), vocab_size=1024, seq_len=1024, iterations=20000, warmup_steps=20, max_wallclock=600s. Model defaults (unless noted): model_dim=512, num_layers=9, num_heads=8, num_kv_heads=4, mlp_mult=2

## Run Details

### run-001 — Baseline (reduced batch for 3060, GRAD_ACCUM_STEPS patched to 64)

**Command:**
```bash
TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=65536 GRAD_ACCUM_STEPS=64 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Key metrics:**
- model_params: 17,059,912
- step_avg: ~1,189ms
- val_loss: 2.7532 → val_bpb: 1.6306
- int8+zlib roundtrip: val_loss=2.7620, val_bpb=1.6358
- Submission size (int8+zlib): 9,918,643 bytes + 47,719 code = ~9.97 MB
- Peak memory: 464 MiB allocated, 476 MiB reserved

**Observations:**
- Only completed 505 of 20000 iterations before 600s wallclock cap
- Loss was still clearly decreasing at step 505 (train_loss 2.75 at cutoff)
- 464 MiB peak memory leaves some headroom on the 6 GB card
- The GRAD_ACCUM_STEPS=64 env var required a code patch (since reverted); default is `8 // world_size`
- Python 3.9 venv had `zip(strict=True)` compat issue — upgrading to 3.10

### baseline_sp1024 — Baseline with sp1024 tokenizer, default grad_accum

**Command:**
```bash
RUN_ID=baseline_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=65536 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Key metrics:**
- model_params: 17,059,912
- step_avg: ~902ms, 666 steps in 600s
- val_loss: 2.6587 → val_bpb: 1.5746
- int8+zlib roundtrip: val_loss=2.6631, val_bpb=1.5772
- Submission size (int8+zlib): 10,978,180 bytes + 47,686 code = ~11.03 MB
- Peak memory: 1891 MiB allocated, 1992 MiB reserved

**Observations:**
- Much faster than run-001 (902ms vs 1189ms/step) — grad_accum=8 vs 64 means fewer micro-batches
- Higher memory (1891 vs 464 MiB) because micro-batches are 8× larger
- Better val_bpb (1.5772 vs 1.6358) — more steps (666 vs 505) + better gradient quality

### leaky_relu — LeakyReLU(0.5)² activation (train_gpt.py line 616)

**Code change:**
```python
# Before (relu²):
x = torch.relu(self.fc(x))
# After (leaky_relu²):
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
```

**Key metrics:**
- model_params: 17,059,912
- step_avg: ~849ms, 707 steps in 600s
- val_loss: 2.6222 → val_bpb: 1.5530
- int8+zlib roundtrip: val_loss=2.6265, val_bpb=1.5556
- Submission size (int8+zlib): 11,230,473 bytes + 47,708 code = ~11.28 MB
- Peak memory: 1891 MiB allocated, 1992 MiB reserved

**Observations:**
- **-0.0217 bpb** improvement over baseline (1.5556 vs 1.5772 int8 roundtrip)
- Slightly faster per step (~849ms vs ~902ms), yielding 41 more steps (707 vs 666)
- Eliminates dead neurons: with ReLU², ~50% of hidden units produce zero output+gradient for z<0. LeakyReLU(0.5)² keeps all neurons active.
- Larger improvement than leaderboard ablations report (-0.003 bpb), likely because unoptimized baseline benefits more from eliminating dead neurons
- int8+zlib artifact is slightly larger (11.28 vs 11.03 MB) — leaky_relu weights may have different distribution

### leaky_relu_11L — Deeper model only (NUM_LAYERS=11)

**Changes from leaky_relu:** `NUM_LAYERS=11` (env var)

**Key metrics:**
- model_params: 20,734,552 (21% more than 9L baseline's 17.1M)
- step_avg: ~1,162ms, 517 steps in 600s
- val_loss: 2.7303 → val_bpb: 1.6170
- int8+zlib roundtrip: val_loss=2.7403, val_bpb=1.6230
- Submission size (int8+zlib): 12,174,681 bytes + 47,708 code = ~12.22 MB
- Peak memory: 2216 MiB allocated, 2330 MiB reserved

**Observations:**
- **Worse than 9L leaky_relu** by +0.067 bpb (1.6230 vs 1.5556)
- Also **worse than 11L+MLP3x** (1.6230 vs 1.6028) despite being smaller and getting
  9 more steps (517 vs 508). Going deeper without going wider is the worst trade-off:
  you pay the step time penalty of more layers without enough per-layer capacity to
  make each step count. The extra MLP width in MLP3x gives enough capacity per layer
  to more than offset the tiny step loss.
- 27% fewer steps than 9L (517 vs 707) — same data-starvation problem as 11L+MLP3x

### leaky_relu_11L_mlp3x — Deeper + wider model (NUM_LAYERS=11, MLP_MULT=3)

**Changes from leaky_relu:** `NUM_LAYERS=11 MLP_MULT=3` (env vars)

**Key metrics:**
- model_params: 26,658,144 (56% more than 9L baseline's 17.1M)
- step_avg: ~1,182ms, 508 steps in 600s
- val_loss: 2.6982 → val_bpb: 1.5980
- int8+zlib roundtrip: val_loss=2.7062, val_bpb=1.6028
- Submission size (int8+zlib): 15,198,553 bytes + 47,708 code = ~15.25 MB (close to 16MB cap)
- Peak memory: 2501 MiB allocated, 2606 MiB reserved

**Observations:**
- **Worse than 9L leaky_relu** by +0.047 bpb (1.6028 vs 1.5556) despite 56% more params
- 28% fewer steps (508 vs 707) due to heavier per-step compute (1182ms vs 849ms)
- On 8×H100, leaderboard entries find 11L+MLP3x clearly wins. The difference is compute
  budget: at ~80× more tokens seen, the data term in the scaling law saturates and extra
  capacity dominates. At our throughput, we're data-starved — losing 28% of steps hurts
  more than 56% more parameters helps.
- Submission size nearly hits the 16MB cap (15.25 MB). With int8+zlib this model barely
  fits; the leaderboard uses int6+GPTQ to compress more aggressively.

#### Scaling law analysis: why bigger model loses on 3060 but wins on 8×H100

Kaplan et al. (2020) scaling law: L(N,D) = (Nc/N)^αN + (Dc/D)^αD + L∞
where αN ≈ 0.076, αD ≈ 0.095.

For our two models (same batch size, so tokens ∝ steps):
- Small (9L): N0 params, D0 = 707 × 65536 ≈ 46M tokens
- Big (11L+3x): 1.56×N0 params, D1 = 508 × 65536 ≈ 33M tokens (D1/D0 = 0.72)

N term improvement: 1.56^0.076 ≈ 1.034 (3.4% better from more params)
D term penalty: 0.72^0.095 ≈ 0.969 (3.1% worse from fewer tokens)

These nearly cancel, and since αD > αN, the data penalty slightly wins → bigger
model is worse on our hardware.

At 8×H100 (~80× our throughput), both models see vastly more tokens. The D terms
become much smaller in absolute magnitude (power law diminishing returns), so the
3.1% relative penalty on D shrinks in absolute impact, while the N benefit stays
proportionally the same. Result: bigger model wins when data is abundant, loses
when data-starved.
