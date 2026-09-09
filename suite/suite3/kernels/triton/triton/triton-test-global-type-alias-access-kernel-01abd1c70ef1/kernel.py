# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_compile_errors.py#L313-L315
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

TYPE_ALIAS = tl.pointer_type(tl.int32)

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel():
        a = TYPE_ALIAS  # noqa
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
