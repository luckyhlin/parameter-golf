# BigramHash

Date: 2025-04-11

A hash-based bigram embedding that gives the model explicit access to
consecutive token pairs, not just individual tokens.

## How it works

For each position, hash the (previous_token, current_token) pair into a
fixed-size bucket table, look up a learned embedding, project to model dim,
and add to the token embedding:

```python
prev_tokens = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
bigram_ids = (prev_tokens.long() * 1000003 + input_ids.long()) % bigram_buckets
bigram_out = bigram_proj(bigram_emb(bigram_ids))
x = x + bigram_out
```

(From records/track_non_record_16mb/2026-03-21_.../train_gpt.py lines 892-895)

## Why not just use all possible bigram pairs?

With vocab_size=1024, there are 1024² = 1,048,576 possible bigram pairs.
Storing a full embedding table for all of them would be huge. Instead, pairs
are hashed into a smaller table (e.g. 2048 buckets). Collisions happen but
the model learns to be robust — the embedding captures the average meaning
of all pairs that hash to the same bucket.

## What it gives the model

Standard token embeddings only know the current token. The transformer's
attention mechanism eventually lets tokens see each other, but that costs
compute. BigramHash gives immediate, free local context: "what two-token
pattern am I in?" This is especially useful for a small vocab (1024) where
individual tokens carry little information.

## Leaderboard impact

From the top entry breakdown: BigramHash 3072 contributed ~0.005-0.01 bpb.
Modest but essentially free — the embedding table and projection are tiny
relative to the transformer blocks.

## Not in our baseline

The baseline `train_gpt.py` does NOT include BigramHash. It's an architectural
addition found in leaderboard submissions. Would require adding the embedding
table, projection layer, and the forward-pass hash computation.
