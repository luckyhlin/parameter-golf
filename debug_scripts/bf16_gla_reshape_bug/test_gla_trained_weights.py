"""
Test causality and intermediate values with the TRAINED bf16 weights.
The trained model gives val_loss=0.065 in bf16 but 16.7 in fp32.
Something about the trained weights makes bf16 and fp32 diverge wildly.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_linear import (
    GPT, GatedLinearAttention, CastedLinear, Hyperparameters,
    restore_low_dim_params_to_fp32, load_validation_tokens,
)

DEVICE = torch.device("cuda")


def _bf16_forward_with_diagnostics(self, x):
    """bf16 forward that also prints intermediate diagnostics."""
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

    chunk_diagnostics = []
    for c_idx in range(NC):
        q_c = qc[:, :, c_idx]
        k_c = kc[:, :, c_idx]
        v_c = vc[:, :, c_idx]
        cum_lg = torch.cumsum(lg[:, :, c_idx], dim=-1)

        decay_raw = torch.exp(cum_lg.unsqueeze(-1) - cum_lg.unsqueeze(-2))
        decay = (decay_raw * causal_mask).to(q.dtype)

        upper_raw = torch.triu(decay_raw, diagonal=1)
        upper_masked = torch.triu(decay.float(), diagonal=1)

        attn = torch.einsum("bhid,bhjd->bhij", q_c, k_c)
        attn_masked = attn * decay
        upper_attn = torch.triu(attn_masked.float(), diagonal=1)

        o_intra = torch.einsum("bhij,bhjd->bhid", attn_masked, v_c)
        decay_from_state = torch.exp(cum_lg).to(q.dtype).unsqueeze(-1)
        o_inter = torch.einsum("bhid,bhde->bhie", q_c, state.to(q.dtype)) * decay_from_state
        output[:, :, c_idx] = o_intra + o_inter

        if c_idx < 3:
            chunk_diagnostics.append({
                "chunk": c_idx,
                "cum_lg_range": (cum_lg.min().item(), cum_lg.max().item()),
                "decay_raw_upper_max": upper_raw.abs().max().item(),
                "decay_masked_upper_max": upper_masked.abs().max().item(),
                "attn_masked_upper_max": upper_attn.abs().max().item(),
                "attn_range": (attn.min().item(), attn.max().item()),
                "o_intra_range": (o_intra.min().item(), o_intra.max().item()),
                "o_inter_range": (o_inter.min().item(), o_inter.max().item()),
                "state_range": (state.min().item(), state.max().item()),
                "state_bf16_range": (state.to(q.dtype).float().min().item(), state.to(q.dtype).float().max().item()),
                "state_cast_err": (state.float() - state.to(q.dtype).float()).abs().max().item(),
            })

        chunk_decay = torch.exp(cum_lg[:, :, -1])
        state = state * chunk_decay[:, :, None, None]
        d2e = torch.exp(cum_lg[:, :, -1:] - cum_lg).to(q.dtype)
        state = state + torch.einsum("bhc,bhci,bhcj->bhij", d2e, k_c, v_c).float()

    y = F.rms_norm(output, (d,))
    y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    y = y * F.silu(self.g_proj(x))
    return self.proj(y), chunk_diagnostics


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()
    state_dict = torch.load("final_model.pt", map_location="cpu", weights_only=True)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
    ).to(DEVICE).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ── Test 1: Perturbation test with trained weights ────────────────────
    print("=" * 70)
    print("TEST 1: Causality perturbation test with TRAINED weights")
    print("=" * 70)

    x_val = val_tokens[:1024].to(device=DEVICE, dtype=torch.int64).unsqueeze(0)
    x_emb = model.tok_emb(x_val)
    x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))

    gla = model.blocks[0].attn
    orig_forward = gla.forward

    gla.forward = lambda x: _bf16_forward_with_diagnostics(gla, x)[0]

    pos = 500
    x_in = x_emb.clone()
    x_pert = x_emb.clone()
    x_pert[:, pos] = torch.randn_like(x_pert[:, pos]) * 10

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            out1 = gla.forward(x_in)
            out2 = gla.forward(x_pert)

    diff = (out1.float() - out2.float()).abs()
    before = diff[:, :pos].max().item()
    at_after = diff[:, pos:].max().item()
    tag = "OK" if before == 0.0 else "LEAK!"
    print(f"  Block 0 bf16: before={before:.2e} at+after={at_after:.2e} {tag}")

    # Also test block 4 (middle)
    gla4 = model.blocks[4].attn
    gla4.forward = lambda x: _bf16_forward_with_diagnostics(gla4, x)[0]
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            out1 = gla4.forward(x_in)
            out2 = gla4.forward(x_pert)
    diff = (out1.float() - out2.float()).abs()
    before = diff[:, :pos].max().item()
    tag = "OK" if before == 0.0 else "LEAK!"
    print(f"  Block 4 bf16: before={before:.2e} at+after={diff[:, pos:].max().item():.2e} {tag}")

    # ── Test 2: Intermediate diagnostics ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Intermediate value diagnostics (block 0, first 3 chunks)")
    print("=" * 70)

    gla0 = model.blocks[0].attn
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            _, diags = _bf16_forward_with_diagnostics(gla0, x_in)

    for d in diags:
        print(f"\n  Chunk {d['chunk']}:")
        print(f"    cum_lg range: [{d['cum_lg_range'][0]:.4f}, {d['cum_lg_range'][1]:.4f}]")
        print(f"    decay raw upper max: {d['decay_raw_upper_max']:.4g}")
        print(f"    decay MASKED upper max: {d['decay_masked_upper_max']:.4e}")
        print(f"    attn*decay upper max: {d['attn_masked_upper_max']:.4e}")
        print(f"    attn range: [{d['attn_range'][0]:.4f}, {d['attn_range'][1]:.4f}]")
        print(f"    o_intra range: [{d['o_intra_range'][0]:.4f}, {d['o_intra_range'][1]:.4f}]")
        print(f"    o_inter range: [{d['o_inter_range'][0]:.4f}, {d['o_inter_range'][1]:.4f}]")
        print(f"    state range: [{d['state_range'][0]:.6f}, {d['state_range'][1]:.6f}]")
        print(f"    state bf16 cast error: {d['state_cast_err']:.4e}")

    # ── Test 3: Logit comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Logit comparison bf16 vs fp32 (trained weights)")
    print("=" * 70)

    def get_logits(model, ids, use_bf16_attn):
        x = model.tok_emb(ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(model.num_encoder_layers):
            x = model.blocks[i](x, x0)
            skips.append(x)
        for i in range(model.num_decoder_layers):
            if skips:
                x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = model.blocks[model.num_encoder_layers + i](x, x0)
        x = model.final_norm(x)
        if model.tie_embeddings:
            logits = F.linear(x, model.tok_emb.weight)
        else:
            logits = model.lm_head(x)
        return model.logit_softcap * torch.tanh(logits / model.logit_softcap)

    # Restore original forwards for fp32
    for blk in model.blocks:
        blk.attn.forward = type(blk.attn).forward.__get__(blk.attn, GatedLinearAttention)

    ids = val_tokens[:1024].to(device=DEVICE, dtype=torch.int64).unsqueeze(0)
    targets = val_tokens[1:1025].to(device=DEVICE, dtype=torch.int64).unsqueeze(0)

    # fp32 forward (default in current code)
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            logits_fp32 = get_logits(model, ids, False)
    loss_fp32 = F.cross_entropy(logits_fp32.float().reshape(-1, 1024), targets.reshape(-1)).item()
    preds_fp32 = logits_fp32.argmax(dim=-1)
    acc_fp32 = (preds_fp32 == targets).float().mean().item()

    # Patch for bf16 forward
    from test_gla_loss_compile import _bf16_forward
    for blk in model.blocks:
        blk.attn.forward = _bf16_forward.__get__(blk.attn, GatedLinearAttention)

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            logits_bf16 = get_logits(model, ids, True)
    loss_bf16 = F.cross_entropy(logits_bf16.float().reshape(-1, 1024), targets.reshape(-1)).item()
    preds_bf16 = logits_bf16.argmax(dim=-1)
    acc_bf16 = (preds_bf16 == targets).float().mean().item()

    diff = (logits_bf16.float() - logits_fp32.float()).abs()
    print(f"  Logit diff: max={diff.max().item():.4f}  mean={diff.mean().item():.4f}")
    print(f"  bf16: loss={loss_bf16:.4f}  argmax_acc={acc_bf16:.4f}")
    print(f"  fp32: loss={loss_fp32:.4f}  argmax_acc={acc_fp32:.4f}")

    print(f"\n  bf16 logit stats: min={logits_bf16.min().item():.4f} max={logits_bf16.max().item():.4f} mean={logits_bf16.float().mean().item():.4f}")
    print(f"  fp32 logit stats: min={logits_fp32.min().item():.4f} max={logits_fp32.max().item():.4f} mean={logits_fp32.float().mean().item():.4f}")

    top5_bf16 = logits_bf16.float().topk(5, dim=-1)
    print(f"  bf16 top-5 logit values (pos 0): {top5_bf16.values[0, 0].tolist()}")
    print(f"  fp32 top-5 logit values (pos 0): {logits_fp32.float().topk(5, dim=-1).values[0, 0].tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
