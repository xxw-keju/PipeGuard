# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_pipeliner.py#L114-L119
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def small_tma_load_kernel(input_desc, output_ptr, N: tl.constexpr):
    offsets = tl.arange(0, 4)
    for block_start in tl.range(0, N, 4, num_stages=3):
        value = input_desc.load([block_start])
        tl.store(output_ptr + block_start + offsets, value)


def _resolve_kernel(**pipeguard_bindings):
    return small_tma_load_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
