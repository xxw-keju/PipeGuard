# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L2412-L2422
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, CONSTANT_FIELD: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if CONSTANT_FIELD == "value":
            value = 1
            output = tl.full([BLOCK_SIZE], value=value, dtype=value.dtype)
            mask = offsets < n_elements
        elif CONSTANT_FIELD == "mask":
            output = offsets < n_elements
            mask = False
        tl.store(output_ptr + offsets, output, mask=mask)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
