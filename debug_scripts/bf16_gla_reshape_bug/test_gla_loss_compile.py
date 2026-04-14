"""
Test: Does torch.compile change the LOSS value for the bf16 GLA model?

Hypothesis: torch.compile fuses the bf16 GLA forward + cross_entropy in a way
that produces incorrect (too-low) loss values. The model's logits may be correct,
but the scalar loss returned is wrong.

Test plan:
1. Create GPT model with bf16 GLA forward (monkey-patched)
2. Run compiled vs uncompiled on SAME input
3. Compare loss values and logits
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_linear import (
    GPT, GatedLinearAttention, CastedLinear, Hyperparameters,
    restore_low_dim_params_to_fp32,
)

DEVICE = torch.device("cuda")


def _bf16_forward(self, x):
    """bf16 GLA forward — matching the code from gla-bf16-0413-1706 log."""
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


def _fp32_forward(self, x):
    """fp32 GLA forward — the working version with autocast(enabled=False)."""
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

    with torch.autocast(device_type=x.device.type, enabled=False):
        q = q.float()
        k = k.float()
        v = v.float()

        qc = q.reshape(bsz, H, NC, C, d)
        kc = k.reshape(bsz, H, NC, C, d)
        vc = v.reshape(bsz, H, NC, C, d)
        lg = log_gate.reshape(bsz, H, NC, C)

        causal_mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.float32))
        output = torch.empty_like(qc)
        state = torch.zeros(bsz, H, d, d, device=x.device, dtype=torch.float32)

        for c_idx in range(NC):
            q_c = qc[:, :, c_idx]
            k_c = kc[:, :, c_idx]
            v_c = vc[:, :, c_idx]
            cum_lg = torch.cumsum(lg[:, :, c_idx], dim=-1)

            decay = torch.exp(cum_lg.unsqueeze(-1) - cum_lg.unsqueeze(-2))
            decay = decay * causal_mask

            attn = torch.einsum("bhid,bhjd->bhij", q_c, k_c)
            o_intra = torch.einsum("bhij,bhjd->bhid", attn * decay, v_c)

            decay_from_state = torch.exp(cum_lg).unsqueeze(-1)
            o_inter = torch.einsum("bhid,bhde->bhie", q_c, state) * decay_from_state

            output[:, :, c_idx] = o_intra + o_inter

            chunk_decay = torch.exp(cum_lg[:, :, -1])
            state = state * chunk_decay[:, :, None, None]
            d2e = torch.exp(cum_lg[:, :, -1:] - cum_lg)
            state = state + torch.einsum("bhc,bhci,bhcj->bhij", d2e, k_c, v_c)

    y = F.rms_norm(output.reshape(bsz, H, seqlen, d), (d,)).bfloat16()
    y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    y = y * F.silu(self.g_proj(x))
    return self.proj(y)


def make_model():
    args = Hyperparameters()
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
    return model


def extract_logits(model, input_ids):
    """Run model forward but return logits instead of loss."""
    x = model.tok_emb(input_ids)
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
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ids = torch.randint(0, 1024, (4, 1024), device=DEVICE, dtype=torch.int64)
    targets = torch.randint(0, 1024, (4, 1024), device=DEVICE, dtype=torch.int64)

    # ── TEST 1: bf16 forward, compiled vs uncompiled loss ────────────────
    section("TEST 1: bf16 GLA forward — compiled vs uncompiled LOSS")

    model = make_model()
    model.eval()

    for blk in model.blocks:
        blk.attn.forward = _bf16_forward.__get__(blk.attn, GatedLinearAttention)

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            loss_eager = model(ids, targets).item()
    print(f"  bf16 eager  loss: {loss_eager:.6f}")

    model_c = torch.compile(model, fullgraph=True, dynamic=False)
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            loss_compiled = model_c(ids, targets).item()
    print(f"  bf16 compiled loss: {loss_compiled:.6f}")
    print(f"  diff: {abs(loss_eager - loss_compiled):.6e}")
    if abs(loss_eager - loss_compiled) > 0.1:
        print("  !! SIGNIFICANT DIFFERENCE — torch.compile changes the loss!")
    else:
        print("  OK — losses match (within tolerance)")

    # ── TEST 2: bf16 vs fp32 forward, uncompiled ─────────────────────────
    section("TEST 2: bf16 vs fp32 GLA forward — uncompiled LOSS comparison")

    model2 = make_model()
    model2.eval()
    model2.load_state_dict(model.state_dict())

    for blk in model2.blocks:
        blk.attn.forward = _fp32_forward.__get__(blk.attn, GatedLinearAttention)

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            loss_fp32 = model2(ids, targets).item()
    print(f"  fp32 eager loss: {loss_fp32:.6f}")
    print(f"  bf16 eager loss: {loss_eager:.6f}")
    print(f"  diff: {abs(loss_fp32 - loss_eager):.6e}")

    # ── TEST 3: Check logits directly ─────────────────────────────────────
    section("TEST 3: Logit-level comparison (bf16 compiled vs bf16 eager)")

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            logits_eager = extract_logits(model, ids)

        with torch.autocast("cuda", torch.bfloat16):
            logits_compiled = extract_logits(model_c, ids)

    diff_logits = (logits_eager.float() - logits_compiled.float()).abs()
    print(f"  Logit diff: max={diff_logits.max().item():.4e}  mean={diff_logits.mean().item():.4e}")

    ce_eager = F.cross_entropy(logits_eager.float().reshape(-1, 1024), targets.reshape(-1), reduction="mean").item()
    ce_compiled = F.cross_entropy(logits_compiled.float().reshape(-1, 1024), targets.reshape(-1), reduction="mean").item()
    print(f"  CE from eager logits:    {ce_eager:.6f}")
    print(f"  CE from compiled logits: {ce_compiled:.6f}")
    print(f"  CE diff: {abs(ce_eager - ce_compiled):.6e}")

    # ── TEST 4: Quick training test — does loss drop anomalously? ─────────
    section("TEST 4: 10-step training with bf16 GLA — does loss drop anomalously?")

    model3 = make_model()
    for blk in model3.blocks:
        blk.attn.forward = _bf16_forward.__get__(blk.attn, GatedLinearAttention)
    model3.train()
    model3_c = torch.compile(model3, fullgraph=True, dynamic=False)

    opt = torch.optim.Adam(model3.parameters(), lr=0.001)

    from train_gpt_linear import TokenStream
    stream = TokenStream(Hyperparameters.train_files)

    print(f"  {'step':>4}  {'compiled':>12}  {'eager_eval':>12}")
    print(f"  {'----':>4}  {'--------':>12}  {'----------':>12}")

    for step in range(10):
        opt.zero_grad(set_to_none=True)
        tokens = stream.take(4 * 1024 + 1).to(dtype=torch.int64, device=DEVICE)
        x = tokens[:-1].reshape(4, 1024)
        y = tokens[1:].reshape(4, 1024)

        with torch.autocast("cuda", torch.bfloat16):
            loss = model3_c(x, y)
        compiled_loss = loss.item()
        loss.backward()
        opt.step()

        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                eager_loss = model3(x, y).item()

        print(f"  {step:>4}  {compiled_loss:>12.4f}  {eager_loss:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
