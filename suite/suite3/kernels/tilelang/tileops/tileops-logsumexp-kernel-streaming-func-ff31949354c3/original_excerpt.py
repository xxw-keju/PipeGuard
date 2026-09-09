    @tilelang.jit(out_idx=[1])
    def _func():
        @T.macro
        def merge_pair(dst_m, dst_s, src_m, src_s, m_new, m_safe):
            # Exponents subtract the ceiling-clamped max, never the true one,
            # so (+inf) - (+inf) = NaN cannot form; see _STREAM_POLICY.max_ceil.
            m_new[0] = T.max(dst_m[0], src_m)
            m_safe[0] = T.min(m_new[0], ceil)
            dst_s[0] = dst_s[0] * T.exp2(
                (T.min(dst_m[0], ceil) - m_safe[0]) * LOG2E
            ) + src_s * T.exp2((T.min(src_m, ceil) - m_safe[0]) * LOG2E)
            dst_m[0] = m_new[0]

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                held = T.alloc_local((cols_per_thread,), dtype)
                held_f = T.alloc_local((cols_per_thread,), "float32")
                slots = T.alloc_local((cols_per_thread,), "float32")
                m_run = T.alloc_local((1,), "float32")
                m_safe = T.alloc_local((1,), "float32")
                s_run = T.alloc_local((1,), "float32")
                m_new = T.alloc_local((1,), "float32")
                scale = T.alloc_local((1,), "float32")
                other_m = T.alloc_local((1,), "float32")
                other_s = T.alloc_local((1,), "float32")
                warp_m = T.alloc_shared((num_warps,), "float32")
                warp_s = T.alloc_shared((num_warps,), "float32")

                m_run[0] = floor
                m_safe[0] = floor
                for c in T.serial(cols_per_thread):
                    slots[c] = 0.0

                for t in T.serial(num_chunks):
                    for g in T.serial(vec_groups):
                        for c in T.vectorized(vec_elems):
                            held[g * vec_elems + c] = x[
                                row, t * chunk + tx * cols_per_thread + g * vec_elems + c
                            ]
                    for c in T.serial(cols_per_thread):
                        held_f[c] = T.cast(held[c], "float32")

                    m_new[0] = m_run[0]
                    for c in T.serial(cols_per_thread):
                        m_new[0] = T.max(m_new[0], held_f[c])
                    scale[0] = T.exp2((m_safe[0] - T.min(m_new[0], ceil)) * LOG2E)
                    m_run[0] = m_new[0]
                    m_safe[0] = T.min(m_new[0], ceil)
                    for c in T.serial(cols_per_thread):
                        slots[c] = slots[c] * scale[0] + T.exp2((held_f[c] - m_safe[0]) * LOG2E)

                s_run[0] = slots[0]
                for c in T.serial(1, cols_per_thread):
                    s_run[0] = s_run[0] + slots[c]

                for stage in T.serial(warp_stages):
                    # Bound locals: a bare expression is substituted per mention.
                    other_m[0] = T.shfl_xor(
                        m_run[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                    )
                    other_s[0] = T.shfl_xor(
                        s_run[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                    )
                    merge_pair(m_run, s_run, other_m[0], other_s[0], m_new, m_safe)
                if tx % WARP_LANES == 0:
                    warp_m[tx // WARP_LANES] = m_run[0]
                    warp_s[tx // WARP_LANES] = s_run[0]
                T.sync_threads()
                if tx == 0:
                    for w in T.serial(1, num_warps):
                        merge_pair(warp_m, warp_s, warp_m[w], warp_s[w], m_new, m_safe)
                    y[row] = T.cast(warp_m[0] + T.log(warp_s[0]), dtype)

        return main
