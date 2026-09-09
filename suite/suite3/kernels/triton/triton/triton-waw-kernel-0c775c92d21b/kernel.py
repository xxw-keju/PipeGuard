# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/gsan/test_gsan_failures.py#L74-L82
# The exact upstream excerpt is stored in original_excerpt.py.

from __future__ import annotations

import triton

import triton.language as tl

# dependency slice: triton.experimental.gsan._testing_utils
@triton.jit
def nanosleep(duration):
    duration = tl.to_tensor(duration)
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.b32 $0, 0;", "=r, r", [duration], tl.int32, is_pure=False, pack=1)

@triton.jit
def atomic_poll(ptr, expect, sem: tl.constexpr = "relaxed", scope: tl.constexpr = "gpu"):
    while tl.atomic_add(ptr, 0, sem=sem, scope=scope) != expect:
        nanosleep(100)

@triton.jit
def _waw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 2)


def _resolve_kernel(**pipeguard_bindings):
    return _waw_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
