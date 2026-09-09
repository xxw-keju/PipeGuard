    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2s_func(block_n: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_coop2s_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        k_iters = k // block_k

        @T.prim_func
        def _gemm_coop2s_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(n // block_n, m // block_m, threads=384) as (bx, by):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_k, block_n), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, block_n), dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        ks = ki * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        T.tma_copy(
                            a[m_start : m_start + half_m, ks : ks + block_k],
                            a_smem_top[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                            a_smem_bot[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            b[ks : ks + block_k, n_start : n_start + block_n],
                            b_smem[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_top[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_0,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.copy(c_local_0, c_cast_0)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_cast_0, c_smem_0)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_smem_0, c[m_start, n_start])
                else:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_bot[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.copy(c_local_1, c_cast_1)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_cast_1, c_smem_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_smem_1, c[m_start + half_m, n_start])

        return _gemm_coop2s_main
