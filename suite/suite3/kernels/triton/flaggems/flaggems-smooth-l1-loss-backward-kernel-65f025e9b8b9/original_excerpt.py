@triton.jit
def _smooth_l1_loss_backward_kernel(
    grad_output,
    inp,
    target,
    out,
    n_elements,
    reduction_elements,
    beta: tl.constexpr,
    reduction: tl.constexpr,
    GRAD_OUTPUT_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp_val = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = inp_val - target_val

    if beta == 0.0:
        grad = tl.where(diff == 0.0, float("nan"), tl.where(diff > 0.0, 1.0, -1.0))
    else:
        grad = tl.where(diff < -beta, -1.0, tl.where(diff > beta, 1.0, diff / beta))

    if GRAD_OUTPUT_SCALAR:
        grad_out = tl.load(grad_output).to(tl.float32)
        if reduction == 1:
            grad_out = grad_out * (1.0 / reduction_elements)
    else:
        grad_out = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        if reduction == 1:
            grad_out = grad_out * (1.0 / reduction_elements)
    tl.store(out + offsets, grad * grad_out, mask=mask)
