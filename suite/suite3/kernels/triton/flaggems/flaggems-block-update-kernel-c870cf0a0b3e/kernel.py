# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def _num_programs(axis: tl.constexpr):
    return tl.num_programs(axis).to(tl.int64)

@triton.jit(do_not_specialize=['r', 't', 'R', 'K', 'r_stride', 'out_r_stride'])
def block_update_kernel(inp, base, rscale_ptr, out, r, t, R, K, r_stride, k_stride, out_r_stride, out_k_stride, rscale_stride, HAS_OUT_LAYOUT: tl.constexpr, TILE: tl.constexpr):
    gridx = _program_id(0).to(tl.int64)
    gridy = _program_id(1).to(tl.int64)
    n_gridx = _num_programs(1)
    base += gridy * n_gridx + gridx
    rscale_ptr += gridy * rscale_stride
    for row in range(gridy, min(gridy + r, R)):
        d = tl.load(base)
        rscale = tl.load(rscale_ptr)
        base += gridx
        rscale_ptr += rscale_stride
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for _ in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            x += d
            x /= rscale
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, x, mask=cols < K)
            cols += TILE

def _resolve_kernel(**pipeguard_bindings):
    return block_update_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
