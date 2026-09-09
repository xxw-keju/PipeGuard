# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def rmsnorm_bwd_dx_kernel(dx_ptr, dz_ptr, x_ptr, weight_ptr, rsigma_ptr, M, N, zero_centered_gamma: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr, BLOCK_COL_SIZE: tl.constexpr):
    """
    Compute dx for RMSNorm backward using 2D tiling.
    Each program handles BLOCK_ROW_SIZE rows.
    """
    pid = _program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(x_ptr.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty
    dz_ptr = dz_ptr + pid * N
    x_ptr = x_ptr + pid * N
    dx_ptr = dx_ptr + pid * N
    rsigma = tl.load(rsigma_ptr + pid, mask=row_mask).to(cdtype)
    c1_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)
        if zero_centered_gamma:
            w = w + 1.0
        x_hat = x * rsigma
        c1_acc += tl.where(mask, x_hat * dz * w, 0.0)
    c1 = tl.sum(c1_acc, axis=1)[:, None] / N
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)
        if zero_centered_gamma:
            w = w + 1.0
        x_hat = x * rsigma
        dx = rsigma * (dz * w - x_hat * c1)
        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return rmsnorm_bwd_dx_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
