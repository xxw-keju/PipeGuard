# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_pipeliner.py#L75-L97
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def matmul_kernel_tma(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M) % M
    offs_bn = (pid_n * BLOCK_N) % N
    offs_am = tl.multiple_of(offs_am, BLOCK_M)
    offs_bn = tl.multiple_of(offs_bn, BLOCK_N)
    offs_k = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_ptr.load([offs_am, offs_k])
        b = b_ptr.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    accumulator = accumulator.to(tl.float16)
    output_ptr.store([offs_am, offs_bn], accumulator)


def _resolve_kernel(**pipeguard_bindings):
    return matmul_kernel_tma


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
