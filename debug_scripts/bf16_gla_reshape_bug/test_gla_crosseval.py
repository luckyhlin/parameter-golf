"""
Definitive test: load the bf16-trained model weights and evaluate with
BOTH bf16 and fp32 GLA forward passes.

If fp32 gives normal loss (~1.8 bpb) but bf16 gives impossibly low (~0.04 bpb),
the bf16 forward produces wrong values with the learned weights.
"""
import io
import math
import os
import sys
import zlib

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_linear import (
    GPT, GatedLinearAttention, CastedLinear, Hyperparameters,
    restore_low_dim_params_to_fp32, load_validation_tokens,
    build_sentencepiece_luts, dequantize_state_dict_int8,
)
import sentencepiece as spm

DEVICE = torch.device("cuda")


def _bf16_forward(self, x):
    """bf16 GLA forward matching gla-bf16-0413-1706."""
    bsz, seqlen, dim = x.shape
    H, d = self.num_heads, self.head_dim
    C = self.chunk_size
    NC = seqlen // C
    q = self.c_q(x).reshape(bsz, seqlen, H, d).transpose(1, 2)
    k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, d).transpose(1, 2)
    v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, d).transpose(1, 2)
    if self.num_kv_heads != H:
        reps = H // self.num_kv_heads
        k = k[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
        v = v[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
    q = F.rms_norm(q, (d,))
    k = F.rms_norm(k, (d,))
    v = F.rms_norm(v, (d,))
    q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
    k = k * self.qk_scale
    log_gate = (F.logsigmoid(self.gate_proj(x)) / self.gate_logit_normalizer).float()
    log_gate = log_gate.clamp(min=-1.0)
    log_gate = log_gate.permute(0, 2, 1)
    qc = q.reshape(bsz, H, NC, C, d)
    kc = k.reshape(bsz, H, NC, C, d)
    vc = v.reshape(bsz, H, NC, C, d)
    lg = log_gate.reshape(bsz, H, NC, C)
    causal_mask = torch.tril(torch.ones(C, C, device=x.device, dtype=q.dtype))
    output = torch.empty_like(qc)
    state = torch.zeros(bsz, H, d, d, device=x.device, dtype=torch.float32)
    for c_idx in range(NC):
        q_c = qc[:, :, c_idx]
        k_c = kc[:, :, c_idx]
        v_c = vc[:, :, c_idx]
        cum_lg = torch.cumsum(lg[:, :, c_idx], dim=-1)
        decay = torch.exp(cum_lg.unsqueeze(-1) - cum_lg.unsqueeze(-2))
        decay = (decay * causal_mask).to(q.dtype)
        attn = torch.einsum("bhid,bhjd->bhij", q_c, k_c)
        o_intra = torch.einsum("bhij,bhjd->bhid", attn * decay, v_c)
        decay_from_state = torch.exp(cum_lg).to(q.dtype).unsqueeze(-1)
        o_inter = torch.einsum("bhid,bhde->bhie", q_c, state.to(q.dtype)) * decay_from_state
        output[:, :, c_idx] = o_intra + o_inter
        chunk_decay = torch.exp(cum_lg[:, :, -1])
        state = state * chunk_decay[:, :, None, None]
        d2e = torch.exp(cum_lg[:, :, -1:] - cum_lg).to(q.dtype)
        state = state + torch.einsum("bhc,bhci,bhcj->bhij", d2e, k_c, v_c).float()
    y = F.rms_norm(output, (d,))
    y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    y = y * F.silu(self.g_proj(x))
    return self.proj(y)


def eval_model(model, val_tokens, args, device, label, max_seqs=256):
    """Validation eval on a subset of val data for speed."""
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    total_seqs = min(total_seqs, max_seqs)
    batch_seqs = 8

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = (val_loss_sum / val_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte
    print(f"  [{label}] val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}  ({int(val_token_count.item())} tokens)")
    return val_loss, val_bpb


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()

    print("Loading bf16-trained model weights from final_model.pt ...")
    state_dict = torch.load("final_model.pt", map_location="cpu", weights_only=True)

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    print(f"Validation tokens: {val_tokens.numel() - 1}")

    # ── Eval 1: bf16 forward (matching the buggy training run) ───────────
    print("\n=== Eval with bf16 GLA forward (eager, no compile) ===")
    model_bf16 = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
    ).to(DEVICE).bfloat16()
    for module in model_bf16.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model_bf16)
    model_bf16.load_state_dict(state_dict, strict=True)
    for blk in model_bf16.blocks:
        blk.attn.forward = _bf16_forward.__get__(blk.attn, GatedLinearAttention)
    eval_model(model_bf16, val_tokens, args, DEVICE, "bf16 eager")

    # ── Eval 2: fp32 forward (the working version) ──────────────────────
    print("\n=== Eval with fp32 GLA forward (eager, no compile) ===")
    model_fp32 = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
    ).to(DEVICE).bfloat16()
    for module in model_fp32.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model_fp32)
    model_fp32.load_state_dict(state_dict, strict=True)
    # model_fp32 uses the default forward (which has autocast(enabled=False) for fp32 chunk loop)
    eval_model(model_fp32, val_tokens, args, DEVICE, "fp32 eager")

    # ── Eval 3: bf16 forward COMPILED ────────────────────────────────────
    print("\n=== Eval with bf16 GLA forward (compiled) ===")
    model_bf16_c = torch.compile(model_bf16, fullgraph=True, dynamic=False)
    eval_model(model_bf16_c, val_tokens, args, DEVICE, "bf16 compiled")

    # ── Eval 4: fp32 forward COMPILED ────────────────────────────────────
    print("\n=== Eval with fp32 GLA forward (compiled) ===")
    model_fp32_c = torch.compile(model_fp32, fullgraph=True, dynamic=False)
    eval_model(model_fp32_c, val_tokens, args, DEVICE, "fp32 compiled")

    # ── Inspect gate values ──────────────────────────────────────────────
    print("\n=== Gate value statistics (from trained model) ===")
    for i, blk in enumerate(model_bf16.blocks):
        gate_bias = blk.attn.gate_proj[1].bias.data
        print(f"  Block {i} gate_proj bias: min={gate_bias.min().item():.4f} max={gate_bias.max().item():.4f} mean={gate_bias.mean().item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
