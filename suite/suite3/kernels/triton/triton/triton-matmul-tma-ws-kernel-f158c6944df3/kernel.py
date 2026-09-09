# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_warp_specialization.py#L222-L262
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def _maybe_tma_load(desc, ptr, off0, off1, USE_TMA: tl.constexpr):
    if USE_TMA:
        return desc.load((off0, off1))
    else:
        offs0 = off0 + tl.arange(0, desc.block_shape[0])
        offs1 = off1 + tl.arange(0, desc.block_shape[1])
        mask0 = offs0 < desc.shape[0]
        mask1 = offs1 < desc.shape[1]
        mask = mask0[:, None] & mask1[None, :]
        return tl.load(ptr + offs0[:, None] * desc.strides[0] + offs1[None, :] * desc.strides[1], mask=mask)

@triton.jit
def matmul_tma_ws_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        a_stride0, a_stride1,  #
        b_stride0, b_stride1,  #
        c_stride0, c_stride1,  #
        M, N, K,  #
        num_stages: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        USE_FP8: tl.constexpr,  #
        A_USE_TMA: tl.constexpr,  #
        B_USE_TMA: tl.constexpr,  #
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BLOCK_SIZE_K
        a = _maybe_tma_load(a_desc, a_ptr, off_am, off_k, A_USE_TMA)
        b = _maybe_tma_load(b_desc, b_ptr, off_bn, off_k, B_USE_TMA)
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
    c_desc.store((off_am, off_bn), c)


def _resolve_kernel(**pipeguard_bindings):
    return matmul_tma_ws_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
