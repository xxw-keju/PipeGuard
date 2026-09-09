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
