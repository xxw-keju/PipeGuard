    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        panel_size: int = 10,
    ) -> Callable:
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        grid_size = -(-n // block_n) * -(-m // block_m)
        tma_epilogue = (n * 2) % 16 == 0 and grid_size > sm_count

        @T.prim_func
        def _gemm_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if tma_epilogue:
                    c_smem = T.alloc_shared((block_m, block_n), dtype)
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                            c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        }
                    )

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.

                m_start = by * block_m
                n_start = bx * block_n

                ps = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    # Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            k_start = ki * block_k
                            T.barrier_wait(ab_empty[slot], phase ^ 1)
                            with trace.range("tma", lane="tma"):
                                if trans_a:
                                    T.tma_copy(
                                        a[k_start : k_start + block_k, m_start : m_start + block_m],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                else:
                                    T.tma_copy(
                                        a[m_start : m_start + block_m, k_start : k_start + block_k],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                if trans_b:
                                    T.tma_copy(
                                        b[n_start : n_start + block_n, k_start : k_start + block_k],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                else:
                                    T.tma_copy(
                                        b[k_start : k_start + block_k, n_start : n_start + block_n],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                            with trace.range("arrive", lane="barrier"):
                                T.barrier_arrive(ab_full[slot])
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // 128
                    with trace.group("consumer", lead=128):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(ab_full[slot], phase)
                            with trace.range("mma", lane="wgmma"):
                                T.wgmma_gemm(
                                    a_smem[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local,
                                    transpose_A=trans_a,
                                    transpose_B=trans_b,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                            if ki > 0:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                                T.barrier_arrive(ab_empty[ps[0]])
                            ps[0] = slot

                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                        T.barrier_arrive(ab_empty[ps[0]])
                        if tma_epilogue:
                            with trace.range("epilogue"):
                                T.copy(c_local, c_smem)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(c_smem, c[m_start, n_start])
                        else:
                            with trace.range("epilogue"):
                                for i, j in T.Parallel(block_m, block_n):
                                    if m_start + i < m and n_start + j < n:
                                        c[m_start + i, n_start + j] = c_local[i, j]

                # Build-time flow declaration: producer "arrive" → consumer
                # "wait" (fixed per-iter pairing).
                trace.dag("arrive", "wait")

        # Materialize markers + append ``slots`` when traced; no-op them (identical
        # CUDA to an un-instrumented build) otherwise. Pairs with ``out_idx`` above.
        return trace.finalize(_gemm_main, traced=traced, max_events=1024)
