"""
Diagnostic part 2: test causality at the MODULE level (not just the chunk loop).

The standalone chunk loop passes all causality tests. This script checks whether
the leak emerges when torch.compile sees the full GLA module or full GPT model.

Tests:
  A. GLA module (includes projections, gate computation, output gate)
  B. Full GPT model (embedding → blocks → logits)
  C. Sequence length T=1024 (training config)
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt_linear import (
    GPT, GatedLinearAttention, CastedLinear, Hyperparameters,
    restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS,
)

DEVICE = torch.device("cuda")


def make_gla_module(dim=512, num_heads=8, num_kv_heads=4, chunk_size=64):
    gla = GatedLinearAttention(dim, num_heads, num_kv_heads, qk_gain_init=1.5, chunk_size=chunk_size)
    gla = gla.to(DEVICE).bfloat16()
    for m in gla.modules():
        if isinstance(m, CastedLinear):
            m.float()
    gla.eval()
    return gla


def make_gpt_model():
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
    model.eval()
    return model


@torch.no_grad()
def perturb_test_module(fn, x, pos, use_autocast=True):
    """Perturbation test on a module that takes x [B, T, D] and returns [B, T, D]."""
    x2 = x.clone()
    x2[:, pos] = torch.randn_like(x[:, pos]) * 10
    ctx = torch.autocast("cuda", torch.bfloat16) if use_autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        o1 = fn(x)
        o2 = fn(x2)
    diff = (o1.float() - o2.float()).abs()
    before = diff[:, :pos].max().item()
    at_after = diff[:, pos:].max().item()
    return before, at_after


@torch.no_grad()
def perturb_test_model_logits(model, input_ids, pos, use_autocast=True):
    """Perturbation test on the full GPT model at the logit level.

    Modifies one INPUT token and checks if logits at earlier positions change.
    """
    ids2 = input_ids.clone()
    vocab_size = model.tok_emb.weight.shape[0]
    ids2[:, pos] = (input_ids[:, pos] + torch.randint(1, vocab_size, (input_ids.shape[0],), device=DEVICE)) % vocab_size

    def get_logits(model, ids):
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

    ctx = torch.autocast("cuda", torch.bfloat16) if use_autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        logits1 = get_logits(model, input_ids)
        logits2 = get_logits(model, ids2)

    diff = (logits1.float() - logits2.float()).abs()
    before = diff[:, :pos].max().item()
    at_after = diff[:, pos:].max().item()
    return before, at_after


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def report(name, bef, aft):
    tag = "OK" if bef == 0.0 else "LEAK!"
    print(f"  {name:<35} before={bef:.2e}  at+after={aft:.2e}  {tag}")


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Test A: GLA module, T=256 ──────────────────────────────────────────
    section("TEST A: GLA module perturbation (T=256, C=64)")
    gla = make_gla_module()
    B, T, D = 2, 256, 512
    x = torch.randn(B, T, D, device=DEVICE, dtype=torch.bfloat16)
    pos = 100

    bef, aft = perturb_test_module(gla, x, pos, use_autocast=True)
    report("bf16 autocast eager", bef, aft)

    gla_c = torch.compile(gla, fullgraph=True)
    bef, aft = perturb_test_module(gla_c, x, pos, use_autocast=True)
    report("bf16 autocast compiled", bef, aft)

    # ── Test B: GLA module, T=1024 (training length) ──────────────────────
    section("TEST B: GLA module perturbation (T=1024, C=64)")
    gla2 = make_gla_module()
    x1024 = torch.randn(2, 1024, 512, device=DEVICE, dtype=torch.bfloat16)
    pos1024 = 500

    bef, aft = perturb_test_module(gla2, x1024, pos1024, use_autocast=True)
    report("bf16 autocast eager T=1024", bef, aft)

    gla2_c = torch.compile(gla2, fullgraph=True)
    bef, aft = perturb_test_module(gla2_c, x1024, pos1024, use_autocast=True)
    report("bf16 autocast compiled T=1024", bef, aft)

    # ── Test C: Full GPT model perturbation at logit level ────────────────
    section("TEST C: Full GPT model perturbation (T=1024)")
    model = make_gpt_model()
    ids = torch.randint(0, 1024, (2, 1024), device=DEVICE, dtype=torch.int64)
    pos_model = 500

    print("  (eager, no compile)")
    bef, aft = perturb_test_model_logits(model, ids, pos_model, use_autocast=True)
    report("GPT bf16 autocast eager", bef, aft)

    print("  (compiled, fullgraph=True)")
    model_c = torch.compile(model, fullgraph=True, dynamic=False)
    bef, aft = perturb_test_model_logits(model_c, ids, pos_model, use_autocast=True)
    report("GPT bf16 autocast compiled", bef, aft)

    # ── Test D: Check if model forward (loss) leaks ───────────────────────
    section("TEST D: Model loss comparison - does changing a future token change loss at earlier positions?")
    print("  Computing per-position losses with shifted targets...")

    def per_position_loss(model, input_ids, target_ids, use_autocast=True):
        """Return per-position cross-entropy losses."""
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
            logits = F.linear(x, model.tok_emb.weight)
        else:
            logits = model.lm_head(x)
        logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               target_ids.reshape(-1), reduction='none').reshape_as(target_ids)

    targets = torch.randint(0, 1024, (2, 1024), device=DEVICE, dtype=torch.int64)
    targets2 = targets.clone()
    targets2[:, pos_model:] = torch.randint(0, 1024, (2, 1024 - pos_model), device=DEVICE, dtype=torch.int64)

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            loss1_eager = per_position_loss(model, ids, targets)
            loss2_eager = per_position_loss(model, ids, targets2)
        with torch.autocast("cuda", torch.bfloat16):
            loss1_comp = per_position_loss(model_c, ids, targets)
            loss2_comp = per_position_loss(model_c, ids, targets2)

    diff_eager = (loss1_eager - loss2_eager).abs()
    diff_comp = (loss1_comp - loss2_comp).abs()
    print(f"  Eager: diff before pos {pos_model}: max={diff_eager[:, :pos_model].max().item():.2e}")
    print(f"  Compiled: diff before pos {pos_model}: max={diff_comp[:, :pos_model].max().item():.2e}")
    print(f"  (Non-zero means target tokens leak into logits - not a model bug, expected if targets affect loss)")

    print("\nDone.")


if __name__ == "__main__":
    main()
