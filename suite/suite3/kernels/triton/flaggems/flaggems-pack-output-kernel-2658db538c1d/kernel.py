# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _pack_output_kernel(out_padded_ptr, out_packed_ptr, offsets_ptr, batch_sizes_ptr, hidden_size, batch_size, out_stride_s, out_stride_b, out_stride_f, packed_stride_0, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // batch_size
    b = pid - t * batch_size
    bs_t = tl.load(batch_sizes_ptr + t).to(tl.int32)
    active = b < bs_t
    row = tl.load(offsets_ptr + t).to(tl.int32) + b
    for f_block in range(0, tl.cdiv(hidden_size, BLOCK_F)):
        offs_f = f_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < hidden_size
        vals = tl.load(out_padded_ptr + t * out_stride_s + b * out_stride_b + offs_f * out_stride_f, mask=f_mask, other=0.0)
        tl.store(out_packed_ptr + row * packed_stride_0 + offs_f, vals, mask=active & f_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _pack_output_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
