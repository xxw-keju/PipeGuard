# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _soft_margin_loss_backward_kernel(grad_output_ptr, self_ptr, target_ptr, out_ptr, n_elements, reduction: tl.constexpr, n_elements_reduced: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    self_vals = tl.load(self_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    grad_output_f = grad_output.to(tl.float32)
    self_f = self_vals.to(tl.float32)
    target_f = target.to(tl.float32)
    exp_term = tl.exp(target_f * self_f)
    grad = grad_output_f * -target_f / (1.0 + exp_term)
    if reduction == 1:
        grad = grad / n_elements_reduced
    tl.store(out_ptr + offsets, grad, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return _soft_margin_loss_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
