# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/gsan/test_gsan.py#L684-L686
# The exact upstream excerpt is stored in original_excerpt.py.

from __future__ import annotations

import triton

import triton.language as tl

@triton.jit
def _gsan_empty_kernel(out_ptr):
    tl.store(out_ptr, 0)


def _resolve_kernel(**pipeguard_bindings):
    return _gsan_empty_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
