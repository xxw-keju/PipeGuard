    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=32):
        @T.prim_func
        def prefill_blocksolve_A_bthd_tl(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_t, threads=threads) as (
                bid,
                hid,
                cid,
            ):
                base = cid * block_t
                k0 = T.alloc_shared([block_c, block_k], dtype)
                k1 = T.alloc_shared([block_c, block_k], dtype)
                k2 = T.alloc_shared([block_c, block_k], dtype)
                k3 = T.alloc_shared([block_c, block_k], dtype)
                g_s = T.alloc_shared([block_t], dtype)
                beta_s = T.alloc_shared([block_t], dtype)
                gate0_s = T.alloc_shared([block_t], dtype)
                gate1_s = T.alloc_shared([block_t], dtype)
                gate2_s = T.alloc_shared([block_t], dtype)
                gate3_s = T.alloc_shared([block_t], dtype)
                beta_f_s = T.alloc_shared([block_t], dtype)
                a_s = T.alloc_shared([10, block_c, block_c], solve_dtype)
                i_s = T.alloc_shared([4, block_c, block_c], solve_dtype)
                work_s = T.alloc_shared([1, block_c, block_c], solve_dtype)

                G00 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G10 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G11 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G20 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G21 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G22 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G30 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G31 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G32 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G33 = T.alloc_fragment([block_c, block_c], accum_dtype)
                tmp = T.alloc_fragment([block_c, block_c], accum_dtype)

                T.annotate_layout(
                    {
                        k0: tilelang.layout.make_swizzled_layout(k0),
                        k1: tilelang.layout.make_swizzled_layout(k1),
                        k2: tilelang.layout.make_swizzled_layout(k2),
                        k3: tilelang.layout.make_swizzled_layout(k3),
                        a_s: tilelang.layout.make_swizzled_layout(a_s),
                        i_s: tilelang.layout.make_swizzled_layout(i_s),
                        work_s: tilelang.layout.make_swizzled_layout(work_s),
                    }
                )

                if use_gate:
                    T.copy(g[bid, base : base + block_t, hid], g_s, disable_tma=True)
                T.copy(
                    beta[bid, base : base + block_t, hid],
                    beta_s,
                    disable_tma=True,
                )

                T.clear(G00)
                T.clear(G10)
                T.clear(G11)
                T.clear(G20)
                T.clear(G21)
                T.clear(G22)
                T.clear(G30)
                T.clear(G31)
                T.clear(G32)
                T.clear(G33)

                for kt in T.Serial(dim_k // block_k):
                    koff = kt * block_k
                    T.async_copy(
                        k[bid, base : base + block_c, hid, koff : koff + block_k],
                        k0,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + block_c : base + 2 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k1,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + 2 * block_c : base + 3 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k2,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + 3 * block_c : base + 4 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k3,
                    )
                    T.ptx_wait_group(0)
                    T.sync_threads()

                    T.gemm(k0, k0, G00, transpose_B=True)
                    T.gemm(k1, k0, G10, transpose_B=True)
                    T.gemm(k1, k1, G11, transpose_B=True)
                    T.gemm(k2, k0, G20, transpose_B=True)
                    T.gemm(k2, k1, G21, transpose_B=True)
                    T.gemm(k2, k2, G22, transpose_B=True)
                    T.gemm(k3, k0, G30, transpose_B=True)
                    T.gemm(k3, k1, G31, transpose_B=True)
                    T.gemm(k3, k2, G32, transpose_B=True)
                    T.gemm(k3, k3, G33, transpose_B=True)

                for t in T.Parallel(block_t):
                    g_val = T.cast(g_s[t], accum_dtype) if use_gate else T.float32(0.0)
                    gate0_s[t] = T.exp2(
                        (g_val - (T.cast(g_s[0], accum_dtype) if use_gate else T.float32(0.0)))
                        * LOG2E
                    )
                    gate1_s[t] = T.exp2(
                        (
                            g_val
                            - (T.cast(g_s[block_c], accum_dtype) if use_gate else T.float32(0.0))
                        )
                        * LOG2E
                    )
                    gate2_s[t] = T.exp2(
                        (
                            g_val
                            - (
                                T.cast(g_s[2 * block_c], accum_dtype)
                                if use_gate
                                else T.float32(0.0)
                            )
                        )
                        * LOG2E
                    )
                    gate3_s[t] = T.exp2(
                        (
                            g_val
                            - (
                                T.cast(g_s[3 * block_c], accum_dtype)
                                if use_gate
                                else T.float32(0.0)
                            )
                        )
                        * LOG2E
                    )
                    beta_f_s[t] = beta_s[t]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    s00 = (
                        T.cast(beta_f_s[i], accum_dtype)
                        * T.cast(gate0_s[i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s11 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate1_s[block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s22 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    s33 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate3_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate3_s[3 * block_c + j], accum_dtype)
                    )
                    s10 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate0_s[block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s20 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s21 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s30 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s31 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s32 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    a_s[0, i, j] = T.if_then_else(
                        i > j,
                        -G00[i, j] * s00,
                        T.float32(0.0),
                    )
                    a_s[2, i, j] = T.if_then_else(
                        i > j,
                        -G11[i, j] * s11,
                        T.float32(0.0),
                    )
                    a_s[5, i, j] = T.if_then_else(
                        i > j,
                        -G22[i, j] * s22,
                        T.float32(0.0),
                    )
                    a_s[9, i, j] = T.if_then_else(
                        i > j,
                        -G33[i, j] * s33,
                        T.float32(0.0),
                    )
                    a_s[1, i, j] = G10[i, j] * s10
                    a_s[3, i, j] = G20[i, j] * s20
                    a_s[4, i, j] = G21[i, j] * s21
                    a_s[6, i, j] = G30[i, j] * s30
                    a_s[7, i, j] = G31[i, j] * s31
                    a_s[8, i, j] = G32[i, j] * s32
                    i_s[0, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[1, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[2, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[3, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                T.sync_threads()

                for _r in T.Serial(1):
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], i_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[0, i, j] = i_s[0, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], a_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[0, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], i_s[1, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[1, i, j] = i_s[1, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], a_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[2, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], i_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[2, i, j] = i_s[2, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], a_s[5, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[5, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], i_s[3, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[3, i, j] = i_s[3, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], a_s[9, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[9, i, j] = tmp[i, j]
                T.sync_threads()

                T.clear(tmp)
                T.gemm(i_s[1, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[1, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[2, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[4, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[3, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[4, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[2, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[3, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[6, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[7, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[3, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[6, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[7, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[7, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[3, :, :], a_s[8, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[2, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[8, i, j] = -tmp[i, j]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    A[bid, base + i, hid, j] = T.cast(i_s[0, i, j], dtype)
                    A[bid, base + i, hid, block_c + j] = T.cast(0, dtype)
                    A[bid, base + i, hid, 2 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + block_c + i, hid, j] = T.cast(a_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, block_c + j] = T.cast(i_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, 2 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + block_c + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + 2 * block_c + i, hid, j] = T.cast(a_s[3, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, block_c + j] = T.cast(a_s[4, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, 2 * block_c + j] = T.cast(
                        i_s[2, i, j], dtype
                    )
                    A[bid, base + 2 * block_c + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + 3 * block_c + i, hid, j] = T.cast(a_s[6, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, block_c + j] = T.cast(a_s[7, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 2 * block_c + j] = T.cast(
                        a_s[8, i, j], dtype
                    )
                    A[bid, base + 3 * block_c + i, hid, 3 * block_c + j] = T.cast(
                        i_s[3, i, j], dtype
                    )

        return prefill_blocksolve_A_bthd_tl
