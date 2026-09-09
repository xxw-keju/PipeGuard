    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dense_group_start_scan_bthd(
            summary: T.Tensor([batch, head, num_groups, dim_k, dim_aug], dtype),
            group_start: T.Tensor([batch, head, num_groups, dim_k, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (
                vid,
                bid,
                hid,
            ):
                a_c = T.alloc_shared([dim_k, dim_k], dtype)
                b_c = T.alloc_shared([dim_k, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                v_offset = vid * BV

                for i, j in T.Parallel(dim_k, BV):
                    h_c[i, j] = 0.0

                for gid in T.Pipelined(num_groups, num_stages=num_stages):
                    for i, j in T.Parallel(dim_k, BV):
                        group_start[bid, hid, gid, i, v_offset + j] = h_c[i, j]
                        b_c[i, j] = summary[bid, hid, gid, i, v_offset + j]
                    for i, j in T.Parallel(dim_k, dim_k):
                        a_c[i, j] = summary[bid, hid, gid, i, dim_v + j]

                    T.clear(h_next_frag)
                    T.gemm(a_c, h_c, h_next_frag)
                    for i, j in T.Parallel(dim_k, BV):
                        h_c[i, j] = h_next_frag[i, j] + b_c[i, j]

        return dense_group_start_scan_bthd
