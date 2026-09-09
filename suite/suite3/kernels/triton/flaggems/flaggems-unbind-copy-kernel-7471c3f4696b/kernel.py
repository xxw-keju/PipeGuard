# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def unbind_copy_kernel(input_ptr, output_ptr, dim_size, dim_prod_post, num_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel for unbind_copy.

    Uses 2D grid: (num_blocks, dim_size)
    - _program_id(0): block index within one output tensor
    - _program_id(1): which output tensor (slice) to copy
    """
    pid_x = tl.program_id(0)
    slice_id = tl.program_id(1)
    block_start = pid_x * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_elements
    idx = block_start + offsets
    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post
    input_flat_idx = pre_idx * dim_size * dim_prod_post + slice_id * dim_prod_post + post_idx
    data = tl.load(input_ptr + input_flat_idx, mask=mask)
    slice_offset = slice_id * num_elements
    tl.store(output_ptr + slice_offset + idx, data, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return unbind_copy_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
