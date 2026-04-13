# Test-Time Training (TTT)

Date: 2025-04-11

TTT is an evaluation-time technique that adapts model weights to each document
during inference. It uses the separate 10-min eval budget, not training time.

## Protocol

```
load quantized model from disk → dequantize to float
checkpoint = copy of model weights

for each document in validation set:
    restore model weights from checkpoint
    for each chunk in document:
        1. SCORE: forward pass on chunk, accumulate loss/bpb (this is graded)
        2. TRAIN: backprop on the same chunk, optimizer step (legal — already scored)
    # discard adapted weights, move to next document
```

The key legality rule: you can only train on tokens you've already evaluated.
Score first, then train. Later chunks benefit from adaptation to earlier chunks
of the same document.

## Why reset between documents

The validation set is 50K web documents concatenated into a flat token stream.
Each document is unrelated to the next. TTT adapts to document-specific patterns
(vocabulary, style, topic). Carrying adapted weights across document boundaries
would hurt — physics-adapted weights are worse for a cooking blog.

## Implementation requirements (refactoring notes)

The current `eval_val` (train_gpt.py lines 219-278) can't support TTT because:

1. **`torch.inference_mode()` at line 250** disables gradient tracking entirely.
   TTT needs gradients for the training step after scoring.

2. **Monolithic eval loop** treats the entire validation set as one flat stream.
   TTT needs document-boundary-aware chunked iteration.

3. **No optimizer during eval**. TTT needs an optimizer (Adam for LoRA, or
   full optimizer for full fine-tuning) initialized per document.

Needed refactoring:
- `eval_chunk(model, chunk_tokens) → loss, bpb` — score one chunk (no_grad)
- `ttt_step(model, optimizer, chunk_tokens)` — train on scored chunk (grad enabled)
- Document boundary detection in the validation token stream
- Weight snapshot/restore per document (reuse warmup pattern from lines 938-957)

## Full fine-tuning vs LoRA for TTT

| | Full fine-tune | LoRA |
|---|---|---|
| Params updated | All ~17M | rank × (in+out) per layer, ~100K |
| Backward speed | Full backward through all params | Only through LoRA params — much faster |
| Optimizer memory | Adam for 17M params (~130MB) | Adam for ~100K params (~1MB) |
| Catastrophic forgetting risk | High — few steps on 500 tokens can wreck general knowledge | Low — low-rank constraint is implicit regularization |
| Adaptation strength per step | Stronger | Weaker but safer |
| Our 3060 (6GB) | Feasible but tight on memory | Negligible memory overhead |

LoRA is preferred on the leaderboard primarily for speed (many document cycles
in 10 min) and safety (won't catastrophically forget). For our small 17M model,
full fine-tuning is more plausible than on larger models — the gap narrows.

## Leaderboard impact

LoRA TTT ablation (from leaderboard submissions):

    Baseline (cross-doc, flat stream)     1.2278
    + Document-isolated eval              1.2168  (-0.0110)
    + Stride (chunk=256)                  1.1941  (-0.0337)
    + LoRA TTT                            1.1910  (-0.0368)

Most gain came from smarter eval strategy (doc isolation + striding), not TTT
itself. The current #1 entry (1.0810) uses score-first TTT. The former #1
(1.1147) dropped TTT — better GPTQ quantization more than compensated.

## Quantization is NOT part of TTT

TTT operates entirely in float memory. The int8/int6 quantized artifact is
loaded once at eval start, dequantized, and TTT runs on the float weights.
No re-quantization between chunks or documents.
