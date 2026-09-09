# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _nll_loss2d_finalize_kernel(partial_loss_ptr, partial_weight_ptr, out_ptr, num_blocks, REDUCTION: tl.constexpr, OUT_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_blocks
    loss = tl.sum(tl.load(partial_loss_ptr + offsets, mask=mask, other=0.0))
    if REDUCTION == 2:
        result = loss
    else:
        tw = tl.sum(tl.load(partial_weight_ptr + offsets, mask=mask, other=0.0))
        result = loss / tw
    tl.store(out_ptr, result.to(OUT_DTYPE))

def _resolve_kernel(**pipeguard_bindings):
    return _nll_loss2d_finalize_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
