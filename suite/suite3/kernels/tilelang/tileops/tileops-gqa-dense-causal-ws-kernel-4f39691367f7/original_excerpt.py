@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[3], pass_configs=_pc, compile_flags=_cf)
def _gqa_dense_causal_ws_kernel(
    B,
    H,
    Hkv,
    D,
    sm_scale,
    softcap,
    dtype,
    block_M=BLOCK_M,
    block_N=BLOCK_N,
    nsK=NSK,
    nsV=NSV,
    threads=THREADS,
):
    """Build the causal WS program; its online softmax carries the previous tile's alpha."""
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = H // Hkv
    accum = "float"
    half = block_M // 2
    Pol = T.GemmWarpPolicy.FullRow
    seq_len_q = T.dynamic("seq_len_q")
    seq_len_kv = T.dynamic("seq_len_kv")

    @T.macro
    def apply_softcap(acc_s, rows, cols):
        for i, j in T.Parallel(rows, cols):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.prim_func
    def main(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len_q, block_M), H, B, threads=threads) as (bx, by, bz):
            Qs = T.alloc_shared([2, half, D], dtype)
            Ks = T.alloc_shared([nsK, block_N, D], dtype)
            Vs = T.alloc_shared([nsV, block_N, D], dtype)
            Os = T.alloc_shared(
                [2, half, D], dtype
            )  # per-WG smem-staged output (FlashInfer epilogue)
            T.annotate_layout(
                {
                    Qs: make_swizzled_layout(Qs),
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                }
            )

            q_bar = T.alloc_barrier([32])  # 1-warp producer (FlashInfer NUM_PRODUCER_THREADS=32)
            kready = T.alloc_barrier([32] * nsK)
            kfree = T.alloc_barrier([NMMA] * nsK)
            vready = T.alloc_barrier([32] * nsV)
            vfree = T.alloc_barrier([NMMA] * nsV)

            cv = by // groups
            q0 = bx * block_M
            causal_offset = T.alloc_var("int32", init=seq_len_kv - seq_len_q)
            eff = T.alloc_var(
                "int32",
                init=T.min(
                    T.ceildiv(seq_len_kv, block_N),
                    T.ceildiv(q0 + block_M + causal_offset, block_N),
                ),
            )
            tx = T.get_thread_binding()

            if tx >= 256:  # ================= producer =================
                T.set_max_nreg(24, 0)  # producer is TMA-only: release regs to consumers
            if tx >= 256 and tx < 288:  # only 1 warp issues TMA + waits (rest of WG idle)
                T.tma_copy(Q[bz, q0 : q0 + half, by, :], Qs[0, :, :], barrier=q_bar)
                T.tma_copy(Q[bz, q0 + half : q0 + block_M, by, :], Qs[1, :, :], barrier=q_bar)
                T.mbarrier_arrive(q_bar)
                for k in T.serial(eff):
                    sk = k % nsK
                    T.mbarrier_wait_parity(kfree[sk], ((k // nsK) % 2) ^ 1)
                    T.tma_copy(
                        K[bz, k * block_N : (k + 1) * block_N, cv, :],
                        Ks[sk, :, :],
                        barrier=kready[sk],
                    )
                    T.mbarrier_arrive(kready[sk])
                    sv = k % nsV
                    T.mbarrier_wait_parity(vfree[sv], ((k // nsV) % 2) ^ 1)
                    T.tma_copy(
                        V[bz, k * block_N : (k + 1) * block_N, cv, :],
                        Vs[sv, :, :],
                        barrier=vready[sv],
                    )
                    T.mbarrier_arrive(vready[sv])

            with T.ws(0):
                T.set_max_nreg(240, 1)  # consumer grabs producer's released regs
                r0 = 0 * half
                my_bar = 1
                nxt_bar = 2
                acc_s = T.alloc_fragment([half, block_N], accum)
                pcast = T.alloc_fragment([half, block_N], dtype)  # register-P (rs-wgmma)
                acc_o = T.alloc_fragment([half, D], accum)
                sm = T.alloc_fragment([half], accum)
                smp = T.alloc_fragment([half], accum)
                alpha = T.alloc_fragment([half], accum)
                ss = T.alloc_fragment([half], accum)
                logsum = T.alloc_fragment([half], accum)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(alpha, 1.0)
                T.fill(sm, -T.infinity(accum))
                T.mbarrier_wait_parity(q_bar, 0)
                pass  # WG0 goes first

                # prologue: tile 0, QK + softmax (no PV)
                T.sync_threads(my_bar, NMMA)
                T.mbarrier_wait_parity(kready[0], 0)
                T.wgmma_gemm(
                    Qs[0, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, NMMA)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if q0 + r0 + causal_offset < block_N - 1:
                    mask_limit = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                if use_softcap:
                    apply_softcap(acc_s, half, block_N)
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                T.reduce_sum(acc_s, ss, dim=1)
                for i in T.Parallel(half):
                    logsum[i] = ss[i]
                T.copy(acc_s, pcast)

                nu = T.alloc_var(
                    "int32",
                    init=T.max(1, T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N))),
                )
                for k in T.serial(1, nu):
                    sk = k % nsK
                    svp = (k - 1) % nsV
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                    T.wgmma_gemm(
                        Qs[0, :, :],
                        Ks[sk, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(1)
                    T.mbarrier_arrive(kfree[sk])
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.copy(sm, smp)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i in T.Parallel(half):
                        alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp])
                    for i in T.Parallel(half):
                        logsum[i] = logsum[i] * alpha[i] + ss[i]
                    T.copy(acc_s, pcast)
                for k in T.serial(nu, eff):
                    sk = k % nsK
                    svp = (k - 1) % nsV
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                    T.wgmma_gemm(
                        Qs[0, :, :],
                        Ks[sk, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(1)
                    T.mbarrier_arrive(kfree[sk])
                    mask_limit_tail = q0 + r0 + causal_offset - k * block_N
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_tail + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.copy(sm, smp)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i in T.Parallel(half):
                        alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp])
                    for i in T.Parallel(half):
                        logsum[i] = logsum[i] * alpha[i] + ss[i]
                    T.copy(acc_s, pcast)

                svp = (eff - 1) % nsV
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.mbarrier_wait_parity(vready[svp], ((eff - 1) // nsV) % 2)
                T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp])
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[0, :, :])
                T.copy(Os[0, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

            with T.ws(1):
                T.set_max_nreg(240, 1)  # consumer grabs producer's released regs
                r0 = 1 * half
                my_bar = 2
                nxt_bar = 1
                acc_s = T.alloc_fragment([half, block_N], accum)
                pcast = T.alloc_fragment([half, block_N], dtype)  # register-P (rs-wgmma)
                acc_o = T.alloc_fragment([half, D], accum)
                sm = T.alloc_fragment([half], accum)
                smp = T.alloc_fragment([half], accum)
                alpha = T.alloc_fragment([half], accum)
                ss = T.alloc_fragment([half], accum)
                logsum = T.alloc_fragment([half], accum)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(alpha, 1.0)
                T.fill(sm, -T.infinity(accum))
                T.mbarrier_wait_parity(q_bar, 0)
                T.named_barrier_arrive(1, NMMA)  # prime WG0

                # prologue: tile 0, QK + softmax (no PV)
                T.sync_threads(my_bar, NMMA)
                T.mbarrier_wait_parity(kready[0], 0)
                T.wgmma_gemm(
                    Qs[1, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, NMMA)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if q0 + r0 + causal_offset < block_N - 1:
                    mask_limit_wg1 = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_wg1 + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                if use_softcap:
                    apply_softcap(acc_s, half, block_N)
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                T.reduce_sum(acc_s, ss, dim=1)
                for i in T.Parallel(half):
                    logsum[i] = ss[i]
                T.copy(acc_s, pcast)

                nu_wg1 = T.alloc_var(
                    "int32",
                    init=T.max(1, T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N))),
                )
                for k in T.serial(1, nu_wg1):
                    sk = k % nsK
                    svp_wg1 = (k - 1) % nsV
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                    T.wgmma_gemm(
                        Qs[1, :, :],
                        Ks[sk, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp_wg1], ((k - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp_wg1, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(1)
                    T.mbarrier_arrive(kfree[sk])
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.copy(sm, smp)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i in T.Parallel(half):
                        alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp_wg1])
                    for i in T.Parallel(half):
                        logsum[i] = logsum[i] * alpha[i] + ss[i]
                    T.copy(acc_s, pcast)
                for k in T.serial(nu_wg1, eff):
                    sk = k % nsK
                    svp_wg1_tail = (k - 1) % nsV
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                    T.wgmma_gemm(
                        Qs[1, :, :],
                        Ks[sk, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp_wg1_tail], ((k - 1) // nsV) % 2)
                    T.wgmma_gemm(
                        pcast, Vs[svp_wg1_tail, :, :], acc_o, policy=Pol, clear_accum=False
                    )
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(1)
                    T.mbarrier_arrive(kfree[sk])
                    mask_limit_wg1_tail = q0 + r0 + causal_offset - k * block_N
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_wg1_tail + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.copy(sm, smp)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i in T.Parallel(half):
                        alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp_wg1_tail])
                    for i in T.Parallel(half):
                        logsum[i] = logsum[i] * alpha[i] + ss[i]
                    T.copy(acc_s, pcast)

                svp_wg1_final = (eff - 1) % nsV
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.mbarrier_wait_parity(vready[svp_wg1_final], ((eff - 1) // nsV) % 2)
                T.wgmma_gemm(pcast, Vs[svp_wg1_final, :, :], acc_o, policy=Pol, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp_wg1_final])
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[1, :, :])
                T.copy(Os[1, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

    return main
