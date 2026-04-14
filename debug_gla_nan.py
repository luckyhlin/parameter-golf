"""
Minimal NaN reproducer for GLA attention.

Uses torch.autograd.detect_anomaly() and forward hooks to find the exact
operation that produces NaN. Runs WITHOUT torch.compile and with a small
batch to fit in 6GB VRAM.

Usage:
    python debug_gla_nan.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor

# Import model and data utilities from the training script
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt_linear import (
    GPT, CastedLinear, Muon, Hyperparameters, TokenStream,
    restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS,
    load_data_shard,
)


def register_nan_hooks(model: nn.Module):
    """Register forward hooks on every module to detect NaN outputs."""
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            tensors = []
            if isinstance(output, Tensor):
                tensors.append(("output", output))
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, Tensor):
                        tensors.append((f"output[{i}]", o))

            for tname, t in tensors:
                if t.isnan().any():
                    nan_count = t.isnan().sum().item()
                    total = t.numel()
                    print(f"\n!!! NaN DETECTED in FORWARD: {name}.{tname}")
                    print(f"    shape={list(t.shape)} dtype={t.dtype} nan_count={nan_count}/{total}")
                    print(f"    max={t[~t.isnan()].abs().max().item() if (~t.isnan()).any() else 'all_nan'}")

                    for iname, inp in enumerate(input if isinstance(input, tuple) else (input,)):
                        if isinstance(inp, Tensor):
                            inp_nan = inp.isnan().any().item()
                            print(f"    input[{iname}]: shape={list(inp.shape)} dtype={inp.dtype} "
                                  f"has_nan={inp_nan} max={inp.abs().max().item():.6g}")
                    raise RuntimeError(
                        f"NaN detected in forward output of {name} ({module.__class__.__name__})"
                    )

                if t.isinf().any():
                    inf_count = t.isinf().sum().item()
                    print(f"\n!!! Inf DETECTED in FORWARD: {name}.{tname}")
                    print(f"    shape={list(t.shape)} dtype={t.dtype} inf_count={inf_count}")

        return hook_fn

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    return hooks


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()

    # Build model (NO compile)
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # Build optimizers (same split as training script)
    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    optimizer_tok = torch.optim.Adam(
        [{"params": [model.tok_emb.weight], "lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    # Data: use small batches to fit without compile
    seq_len = args.train_seq_len
    batch_seqs = 4  # 4 sequences * 1024 tokens = 4096 tokens per micro-batch
    stream = TokenStream(args.train_files)

    # Register NaN detection hooks
    hooks = register_nan_hooks(model)
    print(f"Registered {len(hooks)} forward NaN hooks")

    # Monkey-patch GLA forward to add per-line NaN checks
    from train_gpt_linear import GatedLinearAttention
    _orig_forward = GatedLinearAttention.forward

    def _debug_forward(self, x: Tensor) -> Tensor:
        def check(name, t):
            if t.isnan().any():
                nc = t.isnan().sum().item()
                print(f"\n!!! NaN at GLA.{name}: shape={list(t.shape)} dtype={t.dtype} "
                      f"nan={nc}/{t.numel()} max_finite={t[~t.isnan()].abs().max().item() if (~t.isnan()).any() else 'all_nan'}")
                raise RuntimeError(f"NaN at GLA.{name}")
            if t.isinf().any():
                print(f"\n!!! Inf at GLA.{name}: shape={list(t.shape)} dtype={t.dtype} inf={t.isinf().sum().item()}")
                raise RuntimeError(f"Inf at GLA.{name}")

        bsz, seqlen, dim = x.shape
        H, d = self.num_heads, self.head_dim
        C = self.chunk_size
        NC = seqlen // C

        check("input_x", x)

        q = self.c_q(x).reshape(bsz, seqlen, H, d).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, d).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, d).transpose(1, 2)

        if self.num_kv_heads != H:
            reps = H // self.num_kv_heads
            k = k[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)
            v = v[:, :, None, :, :].expand(-1, -1, reps, -1, -1).reshape(bsz, H, seqlen, d)

        q = F.rms_norm(q, (d,)); check("q_after_rmsnorm", q)
        k = F.rms_norm(k, (d,)); check("k_after_rmsnorm", k)
        v = F.rms_norm(v, (d,)); check("v_after_rmsnorm", v)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        k = k * self.qk_scale
        check("q_scaled", q); check("k_scaled", k)

        log_gate = (F.logsigmoid(self.gate_proj(x)) / self.gate_logit_normalizer).float()
        log_gate = log_gate.clamp(min=-1.0)
        log_gate = log_gate.permute(0, 2, 1)
        check("log_gate", log_gate)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float(); k = k.float(); v = v.float()
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

                attn = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
                lg_slice = lg[:, :, c_idx]
                check(f"chunk_{c_idx}_lg_slice", lg_slice)
                print(f"  chunk_{c_idx}_lg_slice: min={lg_slice.min().item():.6g} max={lg_slice.max().item():.6g}")
                check(f"chunk_{c_idx}_cum_lg", cum_lg)
                print(f"  chunk_{c_idx}_cum_lg: min={cum_lg.min().item():.6g} max={cum_lg.max().item():.6g}")
                diff = cum_lg.unsqueeze(-1) - cum_lg.unsqueeze(-2)
                check(f"chunk_{c_idx}_cum_lg_diff", diff)
                print(f"  chunk_{c_idx}_diff: min={diff.min().item():.6g} max={diff.max().item():.6g}")
                check(f"chunk_{c_idx}_decay", decay)
                check(f"chunk_{c_idx}_attn_raw", attn)
                attn_decay = attn * decay
                check(f"chunk_{c_idx}_attn*decay", attn_decay)
                o_intra = torch.einsum('bhij,bhjd->bhid', attn_decay, v_c)

                decay_from_state = torch.exp(cum_lg).unsqueeze(-1)
                o_inter = torch.einsum('bhid,bhde->bhie', q_c, state) * decay_from_state

                output[:, :, c_idx] = o_intra + o_inter

                chunk_decay = torch.exp(cum_lg[:, :, -1])
                state = state * chunk_decay[:, :, None, None]
                d2e = torch.exp(cum_lg[:, :, -1:] - cum_lg)
                state = state + torch.einsum('bhc,bhci,bhcj->bhij', d2e, k_c, v_c)

                check(f"chunk_{c_idx}_o_intra", o_intra)
                check(f"chunk_{c_idx}_o_inter", o_inter)
                check(f"chunk_{c_idx}_state", state)

        check("output_fp32", output)
        y = F.rms_norm(output.reshape(bsz, H, seqlen, d), (d,))
        check("after_rmsnorm_fp32", y)
        y = y.bfloat16()
        check("after_rmsnorm_bf16", y)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

        g = F.silu(self.g_proj(x))
        check("g_proj_silu", g)
        y = y * g
        check("after_output_gate", y)
        return self.proj(y)

    GatedLinearAttention.forward = _debug_forward

    print(f"Running 20 training steps with detect_anomaly...")
    print("=" * 80)

    model.train()
    for step in range(20):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Get batch
        total_tokens = batch_seqs * seq_len + 1
        tokens = stream.take(total_tokens).to(dtype=torch.int64, device=device)
        x = tokens[:-1].reshape(batch_seqs, seq_len)
        y = tokens[1:].reshape(batch_seqs, seq_len)

        # Forward + backward with anomaly detection
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                with torch.autograd.detect_anomaly(check_nan=True):
                    loss = model(x, y)

            if loss.isnan().any():
                print(f"\nstep {step}: loss is NaN (detected AFTER forward, BEFORE backward)")
                print("Forward hooks did not catch it -- NaN may be in cross_entropy or logit_softcap")
                # Try backward to get detect_anomaly traceback
                try:
                    with torch.autograd.detect_anomaly(check_nan=True):
                        loss.backward()
                except RuntimeError as e:
                    print(f"\ndetect_anomaly caught in backward:\n{e}")
                break

            with torch.autograd.detect_anomaly(check_nan=True):
                loss.backward()

        except RuntimeError as e:
            print(f"\nstep {step}: RuntimeError caught!")
            print(f"{e}")
            break

        # Optimizer step
        for opt in optimizers:
            opt.step()

        print(f"step {step}: loss={loss.item():.4f}")

    # Cleanup
    for h in hooks:
        h.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
