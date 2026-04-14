"""
Diagnostic: bf16 GLA causality leak investigation.

Tests the chunk-wise GLA attention loop across configurations:
  {fp32, bf16} x {eager, compiled} x {autocast on/off}

Perturbation test: modify input at position P, verify output at positions < P
is unchanged (i.e., causal). Any non-zero diff before P means future tokens
are leaking into past positions.

Also inspects the decay matrix for non-zero upper-triangle values, and tests
whether bf16 exp overflow + mask interaction produces NaN (inf * 0 = NaN).
"""
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")


# ── chunk loop implementations ────────────────────────────────────────────────

def gla_chunk_fp32(q, k, v, log_gate, chunk_size):
    """fp32 ground truth (matches the working autocast(enabled=False) code)."""
    B, H, T, D = q.shape
    C = chunk_size
    NC = T // C
    q, k, v = q.float(), k.float(), v.float()
    qc, kc, vc = [x.reshape(B, H, NC, C, D) for x in (q, k, v)]
    lg = log_gate.float().reshape(B, H, NC, C)
    mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.float32))
    out = torch.empty_like(qc)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    for ci in range(NC):
        qc_i, kc_i, vc_i = qc[:, :, ci], kc[:, :, ci], vc[:, :, ci]
        clg = torch.cumsum(lg[:, :, ci], dim=-1)
        decay = torch.exp(clg.unsqueeze(-1) - clg.unsqueeze(-2)) * mask
        A = torch.einsum("bhid,bhjd->bhij", qc_i, kc_i)
        o_intra = torch.einsum("bhij,bhjd->bhid", A * decay, vc_i)
        dfs = torch.exp(clg).unsqueeze(-1)
        o_inter = torch.einsum("bhid,bhde->bhie", qc_i, S) * dfs
        out[:, :, ci] = o_intra + o_inter
        cd = torch.exp(clg[:, :, -1])
        S = S * cd[:, :, None, None]
        d2e = torch.exp(clg[:, :, -1:] - clg)
        S = S + torch.einsum("bhc,bhci,bhcj->bhij", d2e, kc_i, vc_i)
    return out.reshape(B, H, T, D)


def gla_chunk_bf16(q, k, v, log_gate, chunk_size):
    """bf16 version matching the buggy code from gla-bf16-0413-1706 log."""
    B, H, T, D = q.shape
    C = chunk_size
    NC = T // C
    qc, kc, vc = [x.reshape(B, H, NC, C, D) for x in (q, k, v)]
    lg = log_gate.float().reshape(B, H, NC, C)
    mask = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
    out = torch.empty_like(qc)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    for ci in range(NC):
        qc_i, kc_i, vc_i = qc[:, :, ci], kc[:, :, ci], vc[:, :, ci]
        clg = torch.cumsum(lg[:, :, ci], dim=-1)
        decay = (torch.exp(clg.unsqueeze(-1) - clg.unsqueeze(-2)) * mask).to(q.dtype)
        A = torch.einsum("bhid,bhjd->bhij", qc_i, kc_i)
        o_intra = torch.einsum("bhij,bhjd->bhid", A * decay, vc_i)
        dfs = torch.exp(clg).to(q.dtype).unsqueeze(-1)
        o_inter = torch.einsum("bhid,bhde->bhie", qc_i, S.to(q.dtype)) * dfs
        out[:, :, ci] = o_intra + o_inter
        cd = torch.exp(clg[:, :, -1])
        S = S * cd[:, :, None, None]
        d2e = torch.exp(clg[:, :, -1:] - clg).to(q.dtype)
        S = S + torch.einsum("bhc,bhci,bhcj->bhij", d2e, kc_i, vc_i).float()
    return out.reshape(B, H, T, D)


# ── perturbation test ─────────────────────────────────────────────────────────

@torch.no_grad()
def perturb_test(fn, q, k, v, lg, pos, cs, autocast):
    """Run forward on original + perturbed-at-`pos` inputs, return max diff."""
    B, H, D = q.shape[0], q.shape[1], q.shape[3]
    q2, k2, v2, lg2 = q.clone(), k.clone(), v.clone(), lg.clone()
    torch.manual_seed(999)
    q2[:, :, pos] = torch.randn(B, H, D, device=q.device, dtype=q.dtype) * 10
    k2[:, :, pos] = torch.randn(B, H, D, device=q.device, dtype=q.dtype) * 10
    v2[:, :, pos] = torch.randn(B, H, D, device=q.device, dtype=q.dtype) * 10
    lg2[:, :, pos] = -torch.rand(B, H, device=q.device) * 0.5
    ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if autocast
        else torch.autocast("cuda", enabled=False)
    )
    with ctx:
        o1 = fn(q, k, v, lg, cs)
        o2 = fn(q2, k2, v2, lg2, cs)
    diff = (o1.float() - o2.float()).abs()
    return diff[:, :, :pos].max().item(), diff[:, :, pos:].max().item()


# ── decay matrix inspection ──────────────────────────────────────────────────

def inspect_decay(lg_1d, chunk_size, label):
    """Build decay matrix from log-gates, check upper triangle for non-zeros."""
    C = chunk_size
    clg = torch.cumsum(lg_1d[:C].unsqueeze(0).unsqueeze(0), dim=-1)  # [1,1,C]
    diff = clg.unsqueeze(-1) - clg.unsqueeze(-2)  # [1,1,C,C]
    raw = torch.exp(diff)

    mask_fp32 = torch.tril(torch.ones(C, C, device=DEVICE))
    mask_bf16 = mask_fp32.bfloat16()

    masked_fp32 = raw * mask_fp32
    masked_bf16_eager = (raw * mask_bf16).bfloat16()

    upper_fp = torch.triu(masked_fp32, diagonal=1).abs().max().item()
    upper_bf = torch.triu(masked_bf16_eager.float(), diagonal=1).abs().max().item()

    raw_max_above = torch.triu(raw, diagonal=1).max().item()
    raw_bf16_above = torch.triu(torch.exp(diff.bfloat16()), diagonal=1)
    has_inf = raw_bf16_above.isinf().any().item()
    inf_times_zero = (raw_bf16_above * mask_bf16).isnan().any().item()

    print(f"  [{label}]")
    print(f"    Raw decay above diagonal: max={raw_max_above:.4g}")
    print(f"    fp32 masked upper-tri max: {upper_fp:.2e}")
    print(f"    bf16 masked upper-tri max: {upper_bf:.2e}")
    if has_inf:
        n_inf = raw_bf16_above.isinf().sum().item()
        print(f"    !! bf16 exp produces {n_inf} inf values above diagonal")
        print(f"    !! inf * bf16_zero = NaN: {inf_times_zero}")
    if upper_bf > 0:
        nz = (torch.triu(masked_bf16_eager.float(), diagonal=1).abs() > 0).sum().item()
        print(f"    !! bf16 has {nz} non-zero entries above diagonal!")
    return upper_fp, upper_bf, has_inf, inf_times_zero


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    B, H, T, D, C = 2, 8, 256, 64, 64
    pos = 100  # in chunk 1 (positions 64-127), local pos 36

    q_f = torch.randn(B, H, T, D, device=DEVICE)
    k_f = torch.randn(B, H, T, D, device=DEVICE)
    v_f = torch.randn(B, H, T, D, device=DEVICE)
    lg = -0.05 * torch.rand(B, H, T, device=DEVICE)

    q_b, k_b, v_b = q_f.bfloat16(), k_f.bfloat16(), v_f.bfloat16()

    # ── Test 1: Perturbation causality test ──────────────────────────────────
    print("=" * 70)
    print(f"TEST 1: Perturbation at position {pos}  (chunk_size={C})")
    print(f"Positions 0-{pos-1} must have zero diff for causal model.")
    print(f"{'Config':<28} {'before':>10} {'at+after':>10} {'result':>8}")
    print("-" * 70)

    configs = [
        ("fp32 eager",       gla_chunk_fp32, q_f, k_f, v_f, False, False),
        ("bf16 eager",       gla_chunk_bf16, q_b, k_b, v_b, False, False),
        ("bf16 autocast",    gla_chunk_bf16, q_b, k_b, v_b, True,  False),
        ("fp32 compiled",    gla_chunk_fp32, q_f, k_f, v_f, False, True),
        ("bf16 ac+compiled", gla_chunk_bf16, q_b, k_b, v_b, True,  True),
    ]

    for name, fn, q, k, v, ac, comp in configs:
        test_fn = torch.compile(fn, fullgraph=True) if comp else fn
        bef, aft = perturb_test(test_fn, q, k, v, lg, pos, C, ac)
        tag = "OK" if bef == 0.0 else "LEAK!"
        print(f"  {name:<26} {bef:>10.2e} {aft:>10.2e} {tag:>8}")

    # ── Test 2: Decay matrix upper-triangle ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Decay matrix upper-triangle inspection")
    print("=" * 70)

    inspect_decay(lg[0, 0], C, "small gates (~-0.05)")

    lg_extreme = -0.9 * torch.ones(T, device=DEVICE)
    inspect_decay(lg_extreme, C, "extreme gates (-0.9)")

    lg_clamp = -1.0 * torch.ones(T, device=DEVICE)
    inspect_decay(lg_clamp, C, "clamped gates (-1.0)")

    # ── Test 3: Extreme gates perturbation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Perturbation with extreme gates (near -1.0 clamp)")
    print("=" * 70)
    lg_ext = -0.9 * torch.ones(B, H, T, device=DEVICE)

    for name, fn, q, k, v, ac, comp in [
        ("bf16 eager extreme",    gla_chunk_bf16, q_b, k_b, v_b, False, False),
        ("bf16 ac+comp extreme",  gla_chunk_bf16, q_b, k_b, v_b, True,  True),
    ]:
        test_fn = torch.compile(fn, fullgraph=True) if comp else fn
        bef, aft = perturb_test(test_fn, q, k, v, lg_ext, pos, C, ac)
        tag = "OK" if bef == 0.0 else "LEAK!"
        print(f"  {name:<26} {bef:>10.2e} {aft:>10.2e} {tag:>8}")

    # ── Test 4: Output comparison (bf16 eager vs bf16 compiled) ──────────────
    print("\n" + "=" * 70)
    print("TEST 4: Output numerical comparison")
    print("=" * 70)
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            out_eager = gla_chunk_bf16(q_b, k_b, v_b, lg, C)
        fn_comp = torch.compile(gla_chunk_bf16, fullgraph=True)
        with torch.autocast("cuda", torch.bfloat16):
            out_comp = fn_comp(q_b, k_b, v_b, lg, C)
        out_fp32 = gla_chunk_fp32(q_f, k_f, v_f, lg, C)

    d_eager_vs_comp = (out_eager.float() - out_comp.float()).abs()
    d_eager_vs_fp32 = (out_eager.float() - out_fp32.float()).abs()
    d_comp_vs_fp32 = (out_comp.float() - out_fp32.float()).abs()

    print(f"  bf16 eager  vs bf16 compiled: max={d_eager_vs_comp.max().item():.2e}  mean={d_eager_vs_comp.mean().item():.2e}")
    print(f"  bf16 eager  vs fp32 eager:    max={d_eager_vs_fp32.max().item():.2e}  mean={d_eager_vs_fp32.mean().item():.2e}")
    print(f"  bf16 compiled vs fp32 eager:  max={d_comp_vs_fp32.max().item():.2e}  mean={d_comp_vs_fp32.mean().item():.2e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
