    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _grouped_gemm_func(block_m, block_n, block_k, num_stages, threads):
        if not transpose_a:
            # NT / NN pattern: iterate over K, batch-offset lookup
            A_shape = (batch_sum, K)
            C_shape = (batch_sum, N)
            A_shared_shape = (block_m, block_k)
            if transpose_b:
                B_shape = (batch_count, N, K)
                B_shared_shape = (block_n, block_k)
            else:
                B_shape = (batch_count, K, N)
                B_shared_shape = (block_k, block_n)

            # Tiles enumerated per group; CTAs past the real count exit.
            _tiling = GroupTiling(batch_count, block_m)
            _num_pid_m = _tiling.tile_upper_bound(batch_sum)
            _num_pid_n = math.ceil(N / block_n)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
                batch_padded_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(_num_pid_m * _num_pid_n, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    s_cum = T.alloc_shared([batch_count + 1], "int32")
                    lo = T.alloc_local([1], "int32")
                    hi = T.alloc_local([1], "int32")
                    row = T.alloc_local([1], "int32")
                    cur_batch_idx = T.alloc_local([1], "int32")

                    _tiling.cumsum(batch_sizes, s_cum)

                    # M-major ordering: each M-tile processes all N-tiles
                    m_tile = pid // _num_pid_n
                    n_start = (pid % _num_pid_n) * block_n

                    if m_tile < s_cum[batch_count]:
                        _tiling.decode(m_tile, s_cum, lo, hi, cur_batch_idx, row)
                        m_start = batch_offsets[cur_batch_idx[0]] + row[0]
                        actual_rows = T.min(block_m, batch_sizes[cur_batch_idx[0]] - row[0])
                        actual_cols = T.min(block_n, N - n_start)
                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < K - k * block_k,
                                    A[m_start + i, k * block_k + j],
                                    0,
                                )
                            if transpose_b:
                                for i, j in T.Parallel(block_n, block_k):
                                    B_shared[i, j] = T.if_then_else(
                                        j < K - k * block_k and i < actual_cols,
                                        B[cur_batch_idx[0], n_start + i, k * block_k + j],
                                        0,
                                    )
                            else:
                                for i, j in T.Parallel(block_k, block_n):
                                    B_shared[i, j] = T.if_then_else(
                                        i < K - k * block_k and j < actual_cols,
                                        B[cur_batch_idx[0], k * block_k + i, n_start + j],
                                        0,
                                    )
                            T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)
                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]

        else:
            # TN / TT pattern: iterate per batch, outputs 3D tensor
            A_shape = (batch_sum, N)
            C_shape = (batch_count, N, K)
            A_shared_shape = (block_m, block_n)
            if transpose_b:
                B_shape = (K, batch_sum)
                B_shared_shape = (block_k, block_m)
            else:
                B_shape = (batch_sum, K)
                B_shared_shape = (block_m, block_k)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
                batch_padded_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(
                    batch_count, T.ceildiv(N, block_n) * T.ceildiv(K, block_k), threads=threads
                ) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_n, block_k], accum_dtype)

                    n_block_idx = by // T.ceildiv(K, block_k)
                    k_block_idx = by % T.ceildiv(K, block_k)
                    n_start = n_block_idx * block_n
                    k_start = k_block_idx * block_k
                    actual_N = T.min(block_n, N - n_start)
                    actual_K = T.min(block_k, K - k_start)
                    T.clear(C_local)

                    batch_start = batch_offsets[bx]
                    batch_size = batch_sizes[bx]

                    for m in T.Pipelined(T.ceildiv(batch_size, block_m), num_stages=num_stages):
                        m_start = batch_start + m * block_m
                        actual_rows = T.min(block_m, batch_size - m * block_m)
                        for i, j in T.Parallel(block_m, block_n):
                            A_shared[i, j] = T.if_then_else(
                                i < actual_rows and j < actual_N, A[m_start + i, n_start + j], 0
                            )
                        if transpose_b:
                            for i, j in T.Parallel(block_k, block_m):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_K and j < actual_rows, B[k_start + i, m_start + j], 0
                                )
                        else:
                            for i, j in T.Parallel(block_m, block_k):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < actual_K, B[m_start + i, k_start + j], 0
                                )
                        T.gemm(
                            A_shared, B_shared, C_local, transpose_A=True, transpose_B=transpose_b
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        if i < actual_N and j < actual_K:
                            C[bx, n_start + i, k_start + j] = C_local[i, j]

        return _grouped_gemm_main
