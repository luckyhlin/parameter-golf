# Parameter Golf Baseline Code Walkthrough

Notes on `train_gpt_mlx.py` (Apple Silicon / MLX) and `train_gpt.py` (CUDA / PyTorch).

---

## What is MLX?

MLX is Apple's machine learning framework built specifically for Apple Silicon.
It is **not** API-compatible with PyTorch. Key differences:

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Hardware | CUDA GPUs (NVIDIA) | Apple Silicon (M1/M2/M3/M4) |
| Memory model | Separate CPU/GPU memory, explicit `.to(device)` | Unified memory, no explicit transfers |
| Execution model | Eager by default, opt-in `torch.compile` | Lazy by default, explicit `mx.eval()` to materialize |
| Forward method | `forward()` (called by `__call__` with hook dispatch) | `__call__()` directly |
| Autodiff | `loss.backward()` populates `.grad` on parameters | `nn.value_and_grad(model, fn)` returns (loss, grad_tree) |
| Compilation | `torch.compile(model)` wraps the module | `mx.compile(fn, inputs=..., outputs=...)` wraps a function |
| Multi-device | DDP / FSDP via `torch.distributed` + NCCL | Not applicable (single Apple Silicon chip) |
| Optimizer | `optimizer.step()` reads `.grad` from params | Custom `opt.step(model, grads_tree, ...)` takes explicit grad dict |

MLX API surface feels similar to PyTorch (nn.Module, nn.Linear, etc.) but the
underlying execution semantics differ. The biggest practical difference is the
lazy evaluation model: operations build a computation graph that only executes
when you call `mx.eval()` or `mx.synchronize()`.

---

## Q: What is `CastedLinear` for?

### MLX version (train_gpt_mlx.py, lines 280-287)

```python
class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T
```

### PyTorch version (train_gpt.py, lines 509-513)

```python
class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)
```

### What it does

CastedLinear implements **mixed-precision training** at the linear layer level:

- **Weight storage**: float32 (full precision)
- **Forward matmul**: bfloat16 (half precision, because `x.dtype` is bfloat16)
- **Optimizer update**: float32 (because the weight tensor itself is float32)

The `.astype(x.dtype)` / `.to(x.dtype)` cast IS an extra op, but it's a cheap
memory copy/reinterpret, not arithmetic. The expensive matmul still runs at
bfloat16 speed.

### Why not just store in bfloat16?

The problem is optimizer precision. When Adam or Muon computes:

    weight = weight - lr * update

If `weight` is bfloat16, small updates (smaller than the bfloat16 epsilon at
that magnitude) round to zero and are silently lost. Over thousands of steps,
this accumulates into "stale weights" that stop improving. Storing in float32
preserves these small deltas.

### Why not just use a regular nn.Linear?

A regular nn.Linear would also store in float32 by default, but it would
compute the matmul in float32 too (unless wrapped in autocast). CastedLinear
makes the cast explicit and unconditional.

### The bfloat16 precision problem, concretely

bfloat16 has only 7 mantissa bits → relative precision (epsilon) of 2^-7 ≈ 0.0078.
For a weight with value ~0.1, the smallest representable delta is ~0.00078.

A typical optimizer step with matrix_lr=0.04 and gradient magnitude ~0.001:

    update = lr × grad ≈ 0.04 × 0.001 = 0.00004

In bfloat16: 0.1 + 0.00004 rounds to 0.1. The update is completely lost.
In float32 (epsilon 2^-23 ≈ 1.2e-7): the same 0.00004 is easily representable.

### Data flow through one training step

Forward:
  1. x arrives in bf16 (COMPUTE_DTYPE)
  2. self.weight.astype(x.dtype) casts f32 weight to bf16 (cheap copy)
  3. x @ weight_bf16.T runs matmul at full bf16 hardware speed
  4. Output is bf16

Backward (autodiff):
  1. Output gradient arrives in bf16
  2. Weight gradient computed in bf16: grad_w = x.T @ grad_output
  3. Autodiff through .astype(): gradient of a cast is a cast in the reverse
     direction, so the bf16 grad is cast to f32 for the f32 weight parameter
  4. Gradient values were computed at bf16 precision but stored in f32

Optimizer update:
  1. weight_f32 = weight_f32 - lr * processed_grad (entirely in f32)
  2. Small deltas survive

### Why not just use autocast?

PyTorch's torch.autocast handles forward-pass casting automatically, and
train_gpt.py does use it. But autocast only affects forward/backward compute,
not the weight storage dtype. If the weight were stored in bf16, autocast would
compute in bf16 and the optimizer update (weight -= lr * grad) would also
happen in bf16, losing small updates. CastedLinear ensures storage is f32
regardless of autocast.

### Torch-specific detail

The PyTorch version inherits from nn.Linear (getting standard parameter
registration for free) and only overrides forward. The MLX version can't do
this because MLX's nn.Linear doesn't support the same inheritance pattern,
so it manually creates the weight from a temporary nn.Linear.

In the Torch version, the model is first cast to bfloat16 (.bfloat16()), then
CastedLinear modules are explicitly cast back to float32 (.float()):

```python
base_model = GPT(...).to(device).bfloat16()
for module in base_model.modules():
    if isinstance(module, CastedLinear):
        module.float()
```

This ensures the body of the model (embeddings, norms, etc.) lives in bfloat16
while CastedLinear weights stay in float32. The CastedLinear.forward() then
casts weights down to bfloat16 for the actual matmul.

The MLX version doesn't need this dance — weights start as float32, and the
input x is already bf16 because the embedding output was cast to COMPUTE_DTYPE
at the start of the forward pass.

---

## Q: Why does every block inherit nn.Module with __init__ and __call__?

### Short answer

This is the standard pattern for all modern ML frameworks (PyTorch, MLX, JAX/Flax).
`nn.Module` provides automatic parameter discovery, and the forward method
(`__call__` in MLX, `forward` in PyTorch) defines the computation graph.

### What nn.Module gives you

1. **Parameter registration**: any `mx.array` (MLX) or `nn.Parameter` / child
   `nn.Module` (PyTorch) assigned as an attribute in `__init__` is automatically
   tracked. This is how `model.parameters()`, `tree_flatten(model.state)`, etc.
   discover all trainable weights without manual bookkeeping.

2. **Autodiff integration**: `nn.value_and_grad(model, fn)` (MLX) or
   `loss.backward()` (PyTorch) needs to know which arrays are parameters to
   compute gradients for. The Module tree is the source of truth.

3. **Compilation**: `mx.compile(fn, inputs=model.state, outputs=model.state)`
   captures the full model state (including non-trainable buffers like RoPE
   frequencies) so the compiled graph doesn't hit "uncaptured input" errors.

4. **Serialization**: `model.state_dict()` / `tree_flatten(model.state)` walks
   the Module tree to produce a flat dict of all tensors for saving/loading.

### MLX vs PyTorch naming convention

- **MLX**: you define `__call__(self, x)` directly. MLX's Module base class
  doesn't add hooks or extra dispatch, so `__call__` IS the forward pass.

- **PyTorch**: you define `forward(self, x)`. The base `nn.Module.__call__`
  invokes hooks (pre-forward, post-forward) before/after calling `forward()`.
  You should never call `model.forward(x)` directly; always use `model(x)`.

### Even parameterless modules need it

`RMSNormNoWeight` (MLX) / `RMSNorm` (PyTorch) has no learnable parameters but
still inherits from nn.Module. This is so it composes correctly as a child of
`Block` -- the framework can traverse the full module tree uniformly. It also
means if you later add parameters, nothing else needs to change.

---

## Q: Why does validation run twice at the end?

### Answer: Two rounds, two different models

Both `train_gpt_mlx.py` and `train_gpt.py` run the full validation set TWICE
at the end, for different purposes:

**Round 1: Original model validation**

Triggered inside the training loop when `last_step` becomes True:

```python
# train_gpt_mlx.py, line 1004-1021
last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
    val_loss, val_bpb = eval_val(...)
    log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} ...")
```

This evaluates the trained model at full precision (bfloat16/float32 weights as
they were during training). The log line looks like:

    step:20000/20000 val_loss:X.XXXX val_bpb:X.XXXX ...

**Round 2: Quantized roundtrip validation**

After the training loop, the script:
1. Saves the raw model
2. Quantizes all weights to int8 + zlib compresses
3. Loads the compressed artifact back from disk
4. Dequantizes the int8 weights back to float
5. Runs the full validation set AGAIN

```python
# train_gpt_mlx.py, lines 1084-1100
quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
model.update(tree_unflatten(list(quant_flat.items())))
q_val_loss, q_val_bpb = eval_val(...)
log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} ...")
```

This is the number that matters for the challenge, because the submitted
artifact is the compressed int8 file. The gap between round 1 and round 2 tells
you how much quality you lost to quantization.

### What you see in your logs

```
val_progress:60550/60568    <-- tail of round 2 (quantized model)
val_progress:60568/60568
final_int8_zlib_roundtrip val_loss:3.9134 val_bpb:2.3178 eval_time:1137306ms
final_int8_zlib_roundtrip_exact val_loss:3.91343863 val_bpb:2.31775879
```

The round 1 results appear earlier in the log as `step:X/Y val_loss:... val_bpb:...`.

### Torch-specific detail

The PyTorch version does exactly the same two-round pattern. One difference: it
also logs `peak memory allocated` and `peak memory reserved` between the two
rounds, and serialization uses `torch.save` / `torch.load` + `io.BytesIO`
instead of `pickle` + `numpy`.

---

## Q: Forward pass, backward pass, optimizer update — how do they work?

### Forward pass

Input → computation → scalar loss. Each module's `__call__` (MLX) / `forward`
(PyTorch) is invoked in sequence. The framework records the computation graph
(every op, intermediate tensor, dtype cast) as it goes. This recorded graph
is what the backward pass traces through.

### Backward pass

Uses the chain rule traced backwards through the recorded graph. For each
operation, autograd calls that operation's backward function and propagates
gradients toward the inputs/parameters.

For a CastedLinear node, the recorded graph looks like:

    weight(f32) → astype(bf16) → weight_bf16 → matmul(x, W_bf16.T) → output(bf16)

Backward walks this in reverse:
    ∂loss/∂W_bf16 = x.T @ ∂loss/∂output       (computed in bf16)
    ∂loss/∂weight_f32 = (∂loss/∂W_bf16).astype(f32)   (cast reversal)

The autograd system casts the gradient back to the original parameter's dtype
because the `.astype()` was part of the recorded graph. This happens inside
the autodiff engine, not in __call__/forward.

### Optimizer update

Operates directly on parameter tensors, completely outside forward/backward.
The optimizer never calls __call__ or forward. It reads the gradient and
modifies the weight storage in-place:

    PyTorch: param.data -= lr * processed_grad   (f32 -= f32)
    MLX:     returns new param dict, model.update() replaces tensors

### Gradient precision

The gradient values have bf16 granularity (computed in bf16, cast to f32 for
storage). The f32 benefit is not better gradient precision — it's the ability to
accumulate small updates that would round to zero in bf16.

    bf16 weight: can represent additions > |w| × 2^-7
    f32 weight:  can represent additions > |w| × 2^-23   (65536× finer)

During warmdown as lr → 0, updates get tiny. F32 preserves them; bf16 loses them.

### PyTorch vs MLX autodiff

    # PyTorch: gradients live on parameters as .grad attributes
    loss.backward()           # populates param.grad for all params
    optimizer.step()          # reads param.grad, modifies param.data

    # MLX: gradients returned as separate dict
    loss, grads = value_and_grad(model, loss_fn)(x, y)
    opt.step(model, grads, ...)

---

## Q: PyTorch model setup — dtype assignment (train_gpt.py lines 838-842)

### The three-step dtype dance

```python
base_model = GPT(...).to(device).bfloat16()        # (1) everything → bf16
for module in base_model.modules():
    if isinstance(module, CastedLinear):
        module.float()                               # (2) CastedLinear → f32
restore_low_dim_params_to_fp32(base_model)           # (3) control params → f32
```

### Why .bfloat16() affects everything

`nn.Module.bfloat16()` is equivalent to `.to(torch.bfloat16)`. It recursively
walks the module tree and converts every `nn.Parameter` and registered buffer
to bfloat16. A single call touches every tensor in the model.

### Why .float() means f32

PyTorch follows the C convention for dtype shortcuts:

    .half()     → float16
    .bfloat16() → bfloat16
    .float()    → float32
    .double()   → float64

When called on an nn.Module, `.float()` recursively converts that module's own
parameters (and its children's parameters) to float32.

### Why .modules() finds everything

`nn.Module` maintains a tree of submodules. Assigning an `nn.Module` as an
attribute in `__init__` registers it as a child. `nn.ModuleList` does the same
for lists. `.modules()` does a recursive depth-first traversal:

    GPT
    ├── tok_emb (Embedding)
    ├── blocks (ModuleList)
    │   ├── Block 0
    │   │   ├── attn_norm (RMSNorm)
    │   │   ├── attn (CausalSelfAttention)
    │   │   │   ├── c_q (CastedLinear)
    │   │   │   ├── c_k (CastedLinear)
    │   │   │   ├── c_v (CastedLinear)
    │   │   │   ├── proj (CastedLinear)
    │   │   │   └── rotary (Rotary)
    │   │   ├── mlp_norm (RMSNorm)
    │   │   └── mlp (MLP)
    │   │       ├── fc (CastedLinear)
    │   │       └── proj (CastedLinear)
    │   ├── Block 1 ... Block 8
    ├── final_norm (RMSNorm)
    └── lm_head (CastedLinear or None)

### Complete parameter dtype map

Every parameter, traced through all three steps:

    Parameter                       Shape        After(1) After(2) After(3) Final
    ─────────────────────────────── ──────────── ──────── ──────── ──────── ─────
    tok_emb.weight                  [1024, 512]  bf16     skip     skip     bf16
    blocks.*.attn.c_q.weight        [512, 512]   bf16     f32      skip     f32
    blocks.*.attn.c_k.weight        [256, 512]   bf16     f32      skip     f32
    blocks.*.attn.c_v.weight        [256, 512]   bf16     f32      skip     f32
    blocks.*.attn.proj.weight       [512, 512]   bf16     f32      skip     f32
    blocks.*.attn.q_gain            [8]          bf16     skip     f32      f32
    blocks.*.mlp.fc.weight          [1024, 512]  bf16     f32      skip     f32
    blocks.*.mlp.proj.weight        [512, 1024]  bf16     f32      skip     f32
    blocks.*.attn_scale             [512]        bf16     skip     f32      f32
    blocks.*.mlp_scale              [512]        bf16     skip     f32      f32
    blocks.*.resid_mix              [2, 512]     bf16     skip     f32      f32
    skip_weights                    [4, 512]     bf16     skip     f32      f32

Step (2) catches: CastedLinear instances → all linear weights in attn/MLP.
Step (3) catches: ndim < 2 (vectors/scalars) OR matching CONTROL_TENSOR_NAME_PATTERNS.

resid_mix and skip_weights are 2D but match the name patterns — that's why
CONTROL_TENSOR_NAME_PATTERNS exists as a second condition in step (3).

### Why tok_emb.weight stays bf16

It's the only parameter that remains bf16. Deliberate reasons:
1. Uses Adam (not Muon) — Adam's adaptive per-element scaling (÷ √v) makes
   it more tolerant of bf16 precision than raw SGD-style updates
2. Relatively large learning rate (tied_embed_lr = 0.05)
3. No cast needed in forward: embedding lookup and tied output projection
   both naturally operate in bf16
4. MLX version does the same — initializes in f32 then casts to COMPUTE_DTYPE

### Why control params need f32

attn_scale, mlp_scale, resid_mix, q_gain, skip_weights modulate the residual
stream. A small change in attn_scale scales every attention output across every
position. bf16 quantization noise on these would show up as noisy training
dynamics. f32 gives clean, precise control.

---

## Full PyTorch training step walkthrough (train_gpt.py)

### Step 1: Zero gradients

    zero_grad_all()  →  param.grad = None for every param in every optimizer

### Step 2: Microbatch loop (grad_accum_steps times)

    x, y = train_loader.next_batch(...)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)         # forward pass, all matmuls in bf16
    (loss * grad_scale).backward()  # backward pass, grads accumulate on .grad

grad_scale = 1/grad_accum_steps, so each microbatch contributes 1/N of the
total gradient. Multiple .backward() calls SUM into .grad (because grad was
not zeroed between microbatches).

### Step 3: Adjust learning rates

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale    # warmdown scaling

### Step 4: Optional gradient clipping

    torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm)

### Step 5: Optimizer step

    for opt in optimizers:
        opt.step()

Adam (fused=True): reads param.grad, updates momentum/variance buffers,
applies param -= lr * m/(√v + ε) in the param's dtype (f32 for control/embed,
bf16 for tok_emb).

Muon: reads param.grad, applies momentum, Newton-Schulz orthogonalization
(internally in bf16), then param -= lr * g_ortho in param dtype (f32).

### Step 6: Clear gradients

    zero_grad_all()

---

## Torch-only differences worth noting

### Distributed training (DDP)

`train_gpt.py` supports multi-GPU via `torchrun` and `DistributedDataParallel`.
`grad_accum_steps = 8 // world_size` -- so 8 GPUs means no gradient accumulation,
1 GPU means 8 accumulation steps. The Muon optimizer uses `dist.all_reduce` to
sync updates across ranks. MLX has no equivalent (single-device only).

### torch.compile

The Torch version wraps the model with `torch.compile(base_model, dynamic=False,
fullgraph=True)` for kernel fusion. Also compiles the Newton-Schulz function:
`zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)`.

MLX uses `mx.compile(fn, inputs=model.state, outputs=model.state)` which is
more explicit about what state the compiled function captures.

### Warmup strategy

Both scripts run warmup steps to prime JIT compilation, but handle state
differently:

- **PyTorch**: saves model + optimizer state before warmup, runs real training
  steps (including `opt.step()`), then restores the saved snapshots afterward.
  This is necessary because `torch.compile` needs to see real backward passes.

- **MLX**: runs forward + backward but NEVER calls `opt.step()`. The parameters
  are never updated during warmup. Then resets only the data loader. This is
  cheaper because there's no snapshot/restore overhead.

### SDP backend selection

The Torch version explicitly selects Flash Attention as the only scaled dot
product attention backend:

```python
enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
```

MLX uses `mx.fast.scaled_dot_product_attention` which routes to Apple's Metal
Performance Shaders internally.

### Muon optimizer implementation

- **PyTorch**: inherits from `torch.optim.Optimizer`, handles DDP by sharding
  params across ranks (`if i % world_size == rank`), doing Newton-Schulz locally,
  then `all_reduce` to broadcast. Standard PyTorch optimizer API.

- **MLX**: plain Python class with a `step()` method that takes explicit params
  and grads dicts. No distributed support. Returns updated param dict instead of
  modifying in-place.

### Autocast

- **PyTorch**: uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
  context manager to automatically cast operations to bfloat16 where appropriate.

- **MLX**: no autocast. Instead, the model explicitly works in `COMPUTE_DTYPE =
  mx.bfloat16` by casting the embedding output at the start of the forward pass
  and keeping everything in that dtype throughout.

---

## What is bf16 (bfloat16)?

"Brain floating point 16", designed by Google Brain for deep learning.

### Bit layout comparison

    f32:    1 sign + 8 exponent + 23 mantissa = 32 bits
    f16:    1 sign + 5 exponent + 10 mantissa = 16 bits
    bf16:   1 sign + 8 exponent +  7 mantissa = 16 bits
                     ^^^^^^^^^^
                     same as f32!

### Properties

    Property            f16            bf16           f32
    ─────────────────── ────────────── ────────────── ──────────────
    Max value           ~65,504        ~3.4×10^38     ~3.4×10^38
    Min positive        ~6×10^-8       ~1.2×10^-38    ~1.2×10^-38
    Precision (eps)     2^-10 ≈ 0.001  2^-7 ≈ 0.0078  2^-23 ≈ 1.2e-7
    Bits                16             16             32

Key design: bf16 has WORSE precision than f16 (7 vs 10 mantissa bits) but the
SAME range as f32. This is crucial for training — gradients can span huge
dynamic ranges. f16 requires "loss scaling" to prevent underflow; bf16 never
needs this because it matches f32's range.

H100 tensor cores process bf16 matmuls at ~2× the throughput of f32.

---

## Why mixed-precision? What does it buy in this contest?

Mixed-precision (bf16 compute + f32 weight storage) is about TRAINING SPEED,
not the 16MB artifact constraint.

bf16 matmuls run ~2× faster on H100 tensor cores, roughly doubling tokens/sec.
With a 10-min wall-clock cap, that means ~2× more training steps. The f32
weight storage ensures these faster steps don't lose quality from accumulation
rounding errors. Net effect: same quality per step, 2× more steps.

This does NOT mean "higher precision = always better with unlimited time."
f64 offers negligible benefit over f32 for training because SGD gradients are
inherently noisy (stochastic minibatch sampling). The gradient noise is orders
of magnitude larger than f32 vs f64 numerical error. f32 is already more
precision than you need for weight accumulation. Nobody trains neural networks
in f64.

---

## The three independent contest constraints

    Constraint           What it limits                    Value
    ──────────────────── ───────────────────────────────── ─────────────
    16MB artifact        Compressed model file on disk     16,000,000 bytes
    10-min training      Wall-clock time to train          600 seconds
    10-min evaluation    Wall-clock time to run inference  600 seconds (separate!)

These are NOT the same as GPU memory during training. An H100 has 80GB VRAM.
The 16MB limit is on the FILE you save after training.

### Where quantization fits

Quantization is post-training compression, not a training technique:

    Training (10 min, mixed bf16/f32, using up to 80GB GPU memory)
        ↓
    Model in memory: ~5-10M params × 4 bytes = ~20-40MB
        ↓
    Post-training quantization: f32 → int8/int6/ternary + compression
        ↓
    Artifact on disk: <16MB
        ↓
    Evaluation (separate 10 min): load, dequantize, run inference

The quantization step itself takes seconds. Evaluation uses a separate time
budget. The key: better quantization (int6 vs int8, GPTQ vs naive) lets you
pack more parameters into 16MB.

### The real optimization variable is compressed bits, not parameter count

    Quantization     Bytes/param   Params in 16MB   Example
    ──────────────── ──────────── ──────────────── ──────────────────────
    f32              4             ~4M              (nobody does this)
    bf16             2             ~8M              (nobody does this)
    int8 + zlib      ~0.8          ~20M             Naive baseline
    int6 + zstd      ~0.5          ~32M             Several top entries
    ternary (1.58b)  ~0.2          ~80M             Ciprian (73.7M params)
    binary (1 bit)   ~0.13         ~128M            Ciprian unlimited (106M)

Better compression → more parameters → more capacity → lower loss. That's why
quantization-aware training (QAT), GPTQ, and ternary quantization dominate
the leaderboard.

### Is this L(N) or L(N,T)?

The README frames it as L(N), but the leaderboard track is really
L(compressed_bits, T) — minimize loss given:
  - a fixed budget of compressed bits (16MB artifact)
  - a fixed training time (10 min on 8×H100)
  - a fixed evaluation time (10 min on 8×H100)

The "unlimited compute" track removes the T constraint, making it closer to
true L(compressed_bits). The fact that compressed_bits ≠ N (parameter count)
is what makes aggressive quantization so valuable.

---

## Does kernel optimization help?

Yes, but it's a secondary lever. The top leaderboard entries use:
- Flash Attention 3 (Hopper warp-specialized kernels)
- Parallel Muon + Parameter Banking (batched Newton-Schulz via torch.bmm)

But the impact breakdown from the top entry (1.1147 bpb) shows:

    Technique                          Category         bpb impact
    ──────────────────────────────────────────────────── ──────────
    11L + MLP3x + various arch changes Architecture     ~0.08
    GPTQ int6 (vs baseline int8)       Compression      ~0.02
    BigramHash 3072                    Architecture     ~0.005-0.01
    LeakyReLU(0.5)²                    Architecture     ~0.003
    EMA + SWA weight averaging         Training         ~0.003
    XSA on all layers                  Architecture     ~0.002
    Flash Attn 3 + Parallel Muon       Kernel/systems   ~0.001 (indirect)

Kernel optimization helps indirectly: faster step → more steps in 10 min →
lower loss. At ~86ms/step, 10 min gives ~6,900 steps. Shaving 5ms/step gives
~7,400 steps — modest. Algorithmic gains are 10-100× larger.

torch.compile already fuses standard ops well. Triton helps most for
non-standard operations (custom XSA, QAT-aware forward, fused quantization).

---

## What is TTT (Test-Time Training)?

TTT is an EVALUATION-TIME technique, not a training technique. It uses the
separate 10-min eval budget.

### Protocol (from the Legal TTT submission)

    For each chunk of validation tokens:
      1. SCORE: Run model on chunk → accumulate loss/bytes for BPB
         (this counts toward the official score — already graded)
      2. TRAIN: Fine-tune model on the already-scored tokens
         (legal because tokens are already evaluated)
      3. Next chunk benefits from the adapted model

    Reset model between documents (no cross-document leakage)

The model adapts to each document's patterns during evaluation, so later tokens
get better predictions. Like a human reader "getting used to" an author's style.

### TTT is NOT depth recurrence

"11L" means 11 distinct transformer layers with independent weights.
It is NOT running the same layer 11 times (that would be depth recurrence /
Universal Transformer — still on the README's wishlist).

TTT is about adapting the WEIGHTS during evaluation. Depth recurrence is about
reusing the same weights for multiple forward passes.

### Impact (from leaderboard submissions)

LoRA TTT ablation (on naive baseline):

    Baseline (cross-doc, flat stream)     1.2278
    + Document-isolated eval              1.2168  (-0.0110)
    + Stride (chunk=256)                  1.1941  (-0.0337)
    + LoRA TTT                            1.1910  (-0.0368)

Most gain came from smarter eval strategy (doc isolation + striding), not TTT.

The current #1 entry (1.1147) DROPPED TTT entirely — better GPTQ quantization
more than compensated for the -0.0025 bpb TTT gain, and TTT conflicted with
their other techniques.

### TTT timing breakdown (from Legal TTT submission)

    Phase                                   Time
    ─────────────────────────────────────── ──────
    Training                                600s (10 min cap)
    Standard eval (int6 roundtrip + sliding) ~120s
    Legal TTT (score-first + adaptation)    ~410s
    Total eval                              ~530s (< 10 min eval cap)

---

## The full optimization landscape

    16MB artifact budget
        ├── Better compression (int8 → int6 → ternary → binary)
        │     → pack MORE parameters into 16MB
        ├── Better architecture (more layers, wider MLP, XSA, BigramHash)
        │     → get MORE out of each parameter
        └── QAT (quantization-aware training)
              → train knowing you'll be quantized, less roundtrip loss

    10-min training budget
        ├── Kernel optimization (Flash Attn 3, Parallel Muon)
        │     → marginal: ~5% more steps
        ├── Mixed-precision (bf16 compute + f32 storage)
        │     → already standard, ~2× throughput
        └── Training tricks (EMA, SWA, warmdown schedule, weight decay)
              → extract more quality from fixed step count

    10-min evaluation budget
        ├── Sliding window / strided eval
        │     → better use of context within each document
        ├── Document-isolated eval
        │     → don't confuse model across document boundaries
        └── TTT (test-time training)
              → adapt to each document's patterns during eval

---

## LeakyReLU² vs ReLU² vs plain ReLU

### The three activations (z = fc(x) is the pre-activation)

    Plain ReLU:        max(0, z)
    ReLU²:             max(0, z)²
    LeakyReLU(0.5)²:   (z if z>0, 0.5z if z<0)²

Concrete values:

    z       ReLU    ReLU²   LeakyReLU(0.5)²
    ─────   ─────   ─────   ────────────────
     2.0     2.0     4.0        4.0
     0.1     0.1     0.01       0.01
     0.0     0.0     0.0        0.0
    -0.1     0.0     0.0        0.0025  ← the difference
    -1.0     0.0     0.0        0.25
    -2.0     0.0     0.0        1.0

### Why ReLU² over plain ReLU

1. Smoother (C¹ continuous at 0 vs ReLU's hard kink) → easier optimization
2. Soft sparsity: small activations suppressed quadratically
3. Stronger non-linearity: piecewise-quadratic vs piecewise-linear

### Why LeakyReLU(0.5)² over ReLU²

The dead neuron problem. With ReLU², for any input ~50% of hidden units have
z < 0, producing exactly zero output and zero gradient. These neurons are
"dead" — they waste parameter budget.

    d/dz [relu(z)²]             = 2·relu(z)     = 0 when z < 0  (DEAD)
    d/dz [leaky_relu(z,0.5)²]  = 2·0.5·(0.5z)  = 0.5z when z<0 (ALIVE)

LeakyReLU(0.5)² keeps ALL neurons active while preserving the non-negative
output property (squaring). For a 16MB-constrained model, making every
parameter contribute gradient signal is essentially free capacity.

The slope of 0.5 balances: too close to 1.0 loses sparsity/gating effect,
too close to 0.0 brings back dead neurons.

Ablation showed -0.003 BPB from this one-line change.

---

## Distributed training analysis

### Current approach (baseline train_gpt.py)

1. DDP (DistributedDataParallel) with NCCL backend
2. grad_accum_steps = 8 // world_size (8 GPUs → no accumulation)
3. Muon optimizer: round-robin parameter sharding, one all_reduce per step
4. Validation: each rank evaluates disjoint val slice, all_reduce to combine

### Why DDP is already near-optimal for this model size

Model is tiny (~5-30M params). Communication math:

    Gradient all-reduce: ~20MB at bf16
    NVLink bandwidth (H100 SXM): 900 GB/s
    All-reduce time: ~0.02ms
    Step time: ~85ms
    Communication fraction: ~0.02%

Communication is essentially free. Step time is compute-bound (matmuls in
forward/backward + Muon Newton-Schulz iterations).

### Alternative strategies and why they don't help

    FSDP:     Shards params to save memory → model fits in 1 GPU, would add overhead
    Tensor P: Splits matmuls across GPUs → matrices too small, comm > savings
    Pipeline: Splits layers across GPUs → 11 layers / 8 GPUs = severe bubbles
    DDP:      Already near-optimal for this size

FSDP/TP/PP help when models are TOO LARGE for one GPU. This model is the
opposite — so small that any extra communication is proportionally expensive.

### What top entries already improved (Parallel Muon, PR #399)

The main distributed inefficiency was the Muon optimizer's Newton-Schulz.
Parallel Muon fixed this:
- 66 separate weights → 4 contiguous 3D parameter banks
- Per-param matmul → batched Newton-Schulz via torch.bmm
- DDP gradient sync → async reduce-scatter → local NS → async all-gather
- Result: ~85ms → ~83.3ms per step

### What could theoretically still help (diminishing returns)

    FP8 compute:       ~2× matmul throughput (H100 FP8 = 2× bf16). Biggest win.
    CUDA Graphs:       Eliminate CPU kernel launch overhead. Maybe 1-2ms/step.
    max-autotune:      Better torch.compile kernel selection. Maybe 1ms/step.
    Fused optimizer:   Combine optimizer with backward. Maybe 0.5ms/step.

At ~83ms/step the system is compute-bound. Distributed optimization is largely
exhausted. The next wins come from precision strategy (FP8) or custom kernels,
not from better parallelism. This is why the leaderboard is dominated by
algorithmic innovations rather than systems work.
