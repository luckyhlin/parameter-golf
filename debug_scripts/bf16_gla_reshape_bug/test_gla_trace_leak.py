"""
Trace exactly WHERE the causality leak occurs in the bf16 GLA forward
with trained weights.

Step-by-step: check each intermediate value for cross-position differences.
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

    x_val = val_tokens[:1024].to(device=DEVICE, dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            x_emb = model.tok_emb(x_val)
            x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))

    pos = 500
    x1 = x_emb.clone()
    x2 = x_emb.clone()
    torch.manual_seed(123)
    x2[:, pos] = torch.randn(1, x_emb.shape[-1], device=DEVICE, dtype=x_emb.dtype) * 10

    gla = model.blocks[0].attn
    H, d = gla.num_heads, gla.head_dim
    C = gla.chunk_size
    bsz, seqlen, dim = x1.shape
    NC = seqlen // C

    def check(name, t1, t2, dim_to_check=None):
        """Check if t1 and t2 differ at positions < pos."""
        if dim_to_check is not None:
            s1 = t1.select(dim_to_check, slice(None))
            s2 = t2.select(dim_to_check, slice(None))
        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH {t1.shape} vs {t2.shape}")
            return

        if len(t1.shape) == 4 and t1.shape[2] == seqlen:
            # shape [B, H, T, D]
            d_before = (t1[:, :, :pos] - t2[:, :, :pos]).abs().max().item()
            d_at = (t1[:, :, pos] - t2[:, :, pos]).abs().max().item()
        elif len(t1.shape) == 3 and t1.shape[1] == seqlen:
            # shape [B, T, D]
            d_before = (t1[:, :pos] - t2[:, :pos]).abs().max().item()
            d_at = (t1[:, pos] - t2[:, pos]).abs().max().item()
        elif len(t1.shape) == 3 and t1.shape[2] == seqlen:
            # shape [B, H, T]
            d_before = (t1[:, :, :pos] - t2[:, :, :pos]).abs().max().item()
            d_at = (t1[:, :, pos] - t2[:, :, pos]).abs().max().item()
        else:
            d_before = (t1.float() - t2.float()).abs().max().item()
            d_at = d_before
            print(f"  {name}: shape {list(t1.shape)} — total diff = {d_before:.4e}")
            return

        tag = "DIFF!" if d_before > 0 else "ok"
        print(f"  {name}: before_pos={d_before:.4e}  at_pos={d_at:.4e}  {tag}")

    print("=" * 70)
    print(f"Tracing leak in block 0 GLA bf16 forward (perturb pos {pos})")
    print("=" * 70)

    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            # Step 1: Projections
            q1 = gla.c_q(x1).reshape(bsz, seqlen, H, d).transpose(1, 2)
            q2 = gla.c_q(x2).reshape(bsz, seqlen, H, d).transpose(1, 2)
            check("q after c_q", q1, q2)

            k1 = gla.c_k(x1).reshape(bsz, seqlen, gla.num_kv_heads, d).transpose(1, 2)
            k2 = gla.c_k(x2).reshape(bsz, seqlen, gla.num_kv_heads, d).transpose(1, 2)
            check("k after c_k", k1, k2)

            v1 = gla.c_v(x1).reshape(bsz, seqlen, gla.num_kv_heads, d).transpose(1, 2)
            v2 = gla.c_v(x2).reshape(bsz, seqlen, gla.num_kv_heads, d).transpose(1, 2)
            check("v after c_v", v1, v2)

            # GQA expand
            if gla.num_kv_heads != H:
                reps = H // gla.num_kv_heads
                k1 = k1[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
                k2 = k2[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
                v1 = v1[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
                v2 = v2[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)

            # Step 2: RMS norm
            q1 = F.rms_norm(q1, (d,)); q2 = F.rms_norm(q2, (d,))
            k1 = F.rms_norm(k1, (d,)); k2 = F.rms_norm(k2, (d,))
            v1 = F.rms_norm(v1, (d,)); v2 = F.rms_norm(v2, (d,))
            check("q after rms_norm", q1, q2)
            check("k after rms_norm", k1, k2)
            check("v after rms_norm", v1, v2)

            # Step 3: Scale
            q1 = q1 * gla.q_gain.to(dtype=q1.dtype)[None, :, None, None]
            q2 = q2 * gla.q_gain.to(dtype=q2.dtype)[None, :, None, None]
            k1 = k1 * gla.qk_scale
            k2 = k2 * gla.qk_scale
            check("q after scale", q1, q2)
            check("k after scale", k1, k2)

            # Step 4: Gates
            lg1 = (F.logsigmoid(gla.gate_proj(x1)) / gla.gate_logit_normalizer).float()
            lg2 = (F.logsigmoid(gla.gate_proj(x2)) / gla.gate_logit_normalizer).float()
            lg1 = lg1.clamp(min=-1.0).permute(0, 2, 1)
            lg2 = lg2.clamp(min=-1.0).permute(0, 2, 1)
            check("log_gate", lg1, lg2)

            # Step 5: Chunk loop
            print("\n  --- Chunk loop ---")
            qc1 = q1.reshape(bsz, H, NC, C, d); qc2 = q2.reshape(bsz, H, NC, C, d)
            kc1 = k1.reshape(bsz, H, NC, C, d); kc2 = k2.reshape(bsz, H, NC, C, d)
            vc1 = v1.reshape(bsz, H, NC, C, d); vc2 = v2.reshape(bsz, H, NC, C, d)
            lgc1 = lg1.reshape(bsz, H, NC, C); lgc2 = lg2.reshape(bsz, H, NC, C)

            mask = torch.tril(torch.ones(C, C, device=DEVICE, dtype=q1.dtype))
            out1 = torch.empty_like(qc1); out2 = torch.empty_like(qc2)
            S1 = torch.zeros(bsz, H, d, d, device=DEVICE, dtype=torch.float32)
            S2 = torch.zeros(bsz, H, d, d, device=DEVICE, dtype=torch.float32)

            perturb_chunk = pos // C  # chunk 7

            for ci in range(NC):
                qi1, ki1, vi1 = qc1[:,:,ci], kc1[:,:,ci], vc1[:,:,ci]
                qi2, ki2, vi2 = qc2[:,:,ci], kc2[:,:,ci], vc2[:,:,ci]
                clg1 = torch.cumsum(lgc1[:,:,ci], dim=-1)
                clg2 = torch.cumsum(lgc2[:,:,ci], dim=-1)

                dec1 = (torch.exp(clg1.unsqueeze(-1) - clg1.unsqueeze(-2)) * mask).to(q1.dtype)
                dec2 = (torch.exp(clg2.unsqueeze(-1) - clg2.unsqueeze(-2)) * mask).to(q2.dtype)

                A1 = torch.einsum("bhid,bhjd->bhij", qi1, ki1)
                A2 = torch.einsum("bhid,bhjd->bhij", qi2, ki2)
                oi1 = torch.einsum("bhij,bhjd->bhid", A1 * dec1, vi1)
                oi2 = torch.einsum("bhij,bhjd->bhid", A2 * dec2, vi2)

                dfs1 = torch.exp(clg1).to(q1.dtype).unsqueeze(-1)
                dfs2 = torch.exp(clg2).to(q2.dtype).unsqueeze(-1)
                oe1 = torch.einsum("bhid,bhde->bhie", qi1, S1.to(q1.dtype)) * dfs1
                oe2 = torch.einsum("bhid,bhde->bhie", qi2, S2.to(q2.dtype)) * dfs2

                out1[:,:,ci] = oi1 + oe1
                out2[:,:,ci] = oi2 + oe2

                if ci <= perturb_chunk + 1:
                    chunk_diff = (out1[:,:,ci].float() - out2[:,:,ci].float()).abs().max().item()
                    state_diff = (S1 - S2).abs().max().item()
                    qkv_diff = max(
                        (qi1.float() - qi2.float()).abs().max().item(),
                        (ki1.float() - ki2.float()).abs().max().item(),
                        (vi1.float() - vi2.float()).abs().max().item(),
                    )
                    lg_diff = (clg1 - clg2).abs().max().item()
                    expect = "should be 0" if ci < perturb_chunk else "may differ"
                    print(f"    chunk {ci}: out_diff={chunk_diff:.4e}  state_diff={state_diff:.4e}  qkv_diff={qkv_diff:.4e}  lg_diff={lg_diff:.4e}  ({expect})")

                cd1 = torch.exp(clg1[:,:,-1]); cd2 = torch.exp(clg2[:,:,-1])
                S1 = S1 * cd1[:,:,None,None]; S2 = S2 * cd2[:,:,None,None]
                d2e1 = torch.exp(clg1[:,:,-1:] - clg1).to(q1.dtype)
                d2e2 = torch.exp(clg2[:,:,-1:] - clg2).to(q2.dtype)
                S1 = S1 + torch.einsum("bhc,bhci,bhcj->bhij", d2e1, ki1, vi1).float()
                S2 = S2 + torch.einsum("bhc,bhci,bhcj->bhij", d2e2, ki2, vi2).float()

            # Step 6: Post-chunk processing
            print("\n  --- Post-chunk ---")
            y1 = F.rms_norm(out1, (d,)); y2 = F.rms_norm(out2, (d,))
            y1 = y1.reshape(bsz, H, seqlen, d); y2 = y2.reshape(bsz, H, seqlen, d)
            d_rms = (y1[:,:,:pos].float() - y2[:,:,:pos].float()).abs().max().item()
            print(f"  After rms_norm (before pos): diff = {d_rms:.4e}")

            y1 = y1.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            y2 = y2.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

            g1 = F.silu(gla.g_proj(x1)); g2 = F.silu(gla.g_proj(x2))
            d_g = (g1[:, :pos].float() - g2[:, :pos].float()).abs().max().item()
            print(f"  g_proj diff (before pos): {d_g:.4e}")

            final1 = gla.proj(y1 * g1); final2 = gla.proj(y2 * g2)
            d_final = (final1[:, :pos].float() - final2[:, :pos].float()).abs().max().item()
            print(f"  Final output diff (before pos): {d_final:.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
