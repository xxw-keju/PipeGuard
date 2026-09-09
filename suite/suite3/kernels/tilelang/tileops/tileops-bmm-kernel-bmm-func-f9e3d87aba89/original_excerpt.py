    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bmm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 128,
    ) -> Callable:
        @T.prim_func
        def _bmm_main(
            a: T.Tensor((batch, m, k), dtype),
            b: T.Tensor((batch, k, n), dtype),
            c: T.Tensor((batch, m, n), dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared((block_k, block_n), dtype)
                c_smem = T.alloc_shared((block_m, block_n), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                    }
                )

                # L2 rasterization: reshape the (bx, by) traversal order into
                # panels of ``panel_size`` blocks along N so neighbouring waves
                # reuse the same A/B rows in L2.
                T.use_swizzle(10, enable=True)

                T.clear(c_local)
                m_start = by * block_m
                n_start = bx * block_n

                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = ki * block_k
                    T.copy(
                        a[bz, m_start : m_start + block_m, k_start : k_start + block_k],
                        a_smem,
                    )
                    T.copy(
                        b[bz, k_start : k_start + block_k, n_start : n_start + block_n],
                        b_smem,
                    )
                    T.gemm(a_smem, b_smem, c_local, policy=T.GemmWarpPolicy.FullRow)

                # Epilogue: stage fp32 accum through SMEM before GMEM store.
                # The M/N tail guard breaks coalescing off the wgmma fragment
                # layout; SMEM->GMEM stores stay coalesced under predication.
                T.copy(c_local, c_smem)
                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        c[bz, m_start + i, n_start + j] = c_smem[i, j]

        return _bmm_main
