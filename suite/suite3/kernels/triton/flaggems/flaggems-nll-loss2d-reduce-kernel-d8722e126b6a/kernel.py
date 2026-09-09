# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _nll_loss2d_reduce_kernel(self_input_ptr, target_ptr, weight_ptr, partial_loss_ptr, partial_weight_ptr, N, C, H, W, total, HAS_WEIGHT: tl.constexpr, REDUCTION: tl.constexpr, IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    W_val = W
    HW = H * W_val
    n = offsets // HW
    hw_remain = offsets % HW
    h = hw_remain // W_val
    w = hw_remain % W_val
    target_index = n * HW + h * W_val + w
    t = tl.load(target_ptr + target_index, mask=mask, other=IGNORE_INDEX)
    valid = (t != IGNORE_INDEX) & mask
    t_clamped = tl.where(valid, t, 0)
    self_index = n * C * HW + t_clamped * HW + h * W_val + w
    s = tl.load(self_input_ptr + self_index, mask=mask, other=0.0).to(tl.float32)
    if HAS_WEIGHT:
        wgt = tl.load(weight_ptr + t_clamped, mask=mask, other=0.0).to(tl.float32)
    else:
        wgt = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    loss = -s * wgt
    loss = tl.where(valid, loss, 0.0)
    block_loss = tl.sum(loss)
    if REDUCTION == 2:
        tl.store(partial_loss_ptr + pid, block_loss)
    else:
        block_weight = tl.sum(tl.where(valid, wgt, 0.0))
        tl.store(partial_loss_ptr + pid, block_loss)
        tl.store(partial_weight_ptr + pid, block_weight)

def _resolve_kernel(**pipeguard_bindings):
    return _nll_loss2d_reduce_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
