    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2_func(
        block_n: int = 256,
        block_k: int = 64,
        num_stages: int = 3,
        group_size_m: int = 16,
        stage_n: int = 0,
    ) -> Callable:
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        sn = block_n if stage_n <= 0 else stage_n
        n_chunks = block_n // sn
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)

        @T.macro
        def decode(flat_id, mt, nt):
            gin = T.int32(group_size_m * num_pid_n)
            gid = flat_id // gin
            first_m = gid * T.int32(group_size_m)
            gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
            mt[0] = first_m + (flat_id % gin) % gsize
            nt[0] = (flat_id % gin) // gsize

        @T.prim_func
        def _gemm_coop2_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, sn), dtype)
                c_smem_1 = T.alloc_shared((half_m, sn), dtype)

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

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
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
                                    b[n_start : n_start + block_n, ks : ks + block_k],
                                    b_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.min(T.int32(half_m), T.int32(m) - m_start)
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_0 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_top[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_0,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps0[0]])
                                ps0[0] = slot
                                gi_cons_0 = gi_cons_0 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                            T.copy(c_local_0, c_cast_0)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                                    T.copy(c_cast_0[:, c0 : c0 + sn], c_smem_0)
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                                    T.copy(c_smem_0, c[m_start, n_start + c0])
                            else:
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + i, n_start + j] = c_cast_0[i, j]

                else:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.max(
                                T.int32(0),
                                T.min(T.int32(half_m), T.int32(m) - m_start - T.int32(half_m)),
                            )
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_1 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_bot[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps1[0]])
                                ps1[0] = slot
                                gi_cons_1 = gi_cons_1 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                            T.copy(c_local_1, c_cast_1)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                                    T.copy(c_cast_1[:, c0 : c0 + sn], c_smem_1)
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                                    T.copy(c_smem_1, c[m_start + half_m, n_start + c0])
                            elif arows > T.int32(0):
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + half_m + i, n_start + j] = c_cast_1[i, j]

        return _gemm_coop2_main
