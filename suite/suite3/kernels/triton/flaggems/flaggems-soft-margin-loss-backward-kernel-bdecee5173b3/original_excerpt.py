@triton.jit
def _soft_margin_loss_backward_kernel(
    grad_output_ptr,
    self_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    reduction: tl.constexpr,
    n_elements_reduced: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    self_vals = tl.load(self_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Cast to float32 for computation
    grad_output_f = grad_output.to(tl.float32)
    self_f = self_vals.to(tl.float32)
    target_f = target.to(tl.float32)

    # Backward formula: grad_output * (-target) / (1 + exp(target * self))
    exp_term = tl.exp(target_f * self_f)
    grad = grad_output_f * (-target_f) / (1.0 + exp_term)

    # Apply mean reduction if needed
    if reduction == 1:
        # reduction == 1 means 'mean'
        grad = grad / n_elements_reduced

    tl.store(out_ptr + offsets, grad, mask=mask)
