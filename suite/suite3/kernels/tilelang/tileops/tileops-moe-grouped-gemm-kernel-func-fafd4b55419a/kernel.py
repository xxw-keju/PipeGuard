# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import math
import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    K = pipeguard_bindings['K']
    N = pipeguard_bindings['N']
    accum_dtype = pipeguard_bindings['accum_dtype']
    dtype = pipeguard_bindings['dtype']
    max_tiles = pipeguard_bindings['max_tiles']
    num_experts = pipeguard_bindings['num_experts']
    numel = pipeguard_bindings['numel']

    @tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        B_shape = (num_experts, N, K)
        C_shape = (numel, N)
        A_shared_shape = (block_m, block_k)
        B_shared_shape = (block_n, block_k)
        _k_aligned = K % block_k == 0
        _b_copy_ok = _k_aligned
        A_shape = (numel + block_m, K) if _b_copy_ok else (numel, K)
        _num_pid_n = math.ceil(N / block_n)
        _total_ctas = max_tiles * _num_pid_n
        _num_pid_in_group = group_size_m * _num_pid_n

        @T.prim_func
        def _gemm_main(A: T.Tensor(A_shape, dtype), B: T.Tensor(B_shape, dtype), C: T.Tensor(C_shape, dtype), tile_expert_ids: T.Tensor([max_tiles], 'int32'), tile_row_offsets: T.Tensor([max_tiles], 'int32'), true_offsets: T.Tensor([num_experts], 'int32'), true_sizes: T.Tensor([num_experts], 'int32'), total_tiles: T.Tensor([1], 'int32')):
            with T.Kernel(_total_ctas, threads=threads) as (pid,):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                T.annotate_layout({A_shared: tilelang.layout.make_swizzled_layout(A_shared), B_shared: tilelang.layout.make_swizzled_layout(B_shared)})
                if group_size_m == 1:
                    bx = pid // T.int32(_num_pid_n)
                    by = pid % T.int32(_num_pid_n)
                else:
                    pid_in_group = pid % T.int32(_num_pid_in_group)
                    group_id = pid // T.int32(_num_pid_in_group)
                    first_pid_m = group_id * T.int32(group_size_m)
                    actual_gsm = T.min(T.int32(max_tiles) - first_pid_m, T.int32(group_size_m))
                    bx = first_pid_m + pid_in_group % actual_gsm
                    by = pid_in_group // actual_gsm
                if bx < total_tiles[0]:
                    expert_id = tile_expert_ids[bx]
                    row_in_expert = tile_row_offsets[bx]
                    m_start = true_offsets[expert_id] + row_in_expert
                    n_start = by * T.int32(block_n)
                    actual_rows = T.min(T.int32(block_m), true_sizes[expert_id] - row_in_expert)
                    actual_cols = T.min(T.int32(block_n), T.int32(N) - n_start)
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                        if _b_copy_ok:
                            T.copy(A[m_start:m_start + block_m, k * block_k:(k + 1) * block_k], A_shared)
                            T.copy(B[expert_id, n_start:n_start + block_n, k * block_k:(k + 1) * block_k], B_shared)
                        else:
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(i < actual_rows and j < K - k * block_k, A[m_start + i, k * block_k + j], 0)
                            for i, j in T.Parallel(block_n, block_k):
                                B_shared[i, j] = T.if_then_else(j < K - k * block_k and i < actual_cols, B[expert_id, n_start + i, k * block_k + j], 0)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_m, block_n):
                        if i < actual_rows and j < actual_cols:
                            C[m_start + i, n_start + j] = C_local[i, j]
        return _gemm_main
    return _func

def build_kernel(case):
    return _resolve_kernel(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
