# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _unpack_padded_kernel(data_ptr, x_ptr, offsets_ptr, batch_sizes_ptr, input_size, batch_size, data_stride_0, x_stride_s, x_stride_b, x_stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(input_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < input_size
        vals = tl.load(data_ptr + row * data_stride_0 + offs_f, mask=active & f_mask, other=0.0)
        tl.store(x_ptr + t * x_stride_s + b * x_stride_b + offs_f * x_stride_f, vals, mask=f_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _unpack_padded_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
