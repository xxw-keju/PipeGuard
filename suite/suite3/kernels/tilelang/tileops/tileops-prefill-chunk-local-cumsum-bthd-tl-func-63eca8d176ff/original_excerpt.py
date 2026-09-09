    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func():
        @T.prim_func
        def chunk_cumsum_bthd_kernel(
            g: T.Tensor([batch, seq_len, head], dtype),
            out: T.Tensor([batch, seq_len, head], dtype),
        ):
            with T.Kernel(batch, num_chunks, threads=128) as (bid, cid):
                base = cid * chunk_size
                acc_s = T.alloc_shared([head], "float32")

                for hid in T.Parallel(head):
                    acc_s[hid] = T.float32(0.0)
                for i in T.Serial(chunk_size):
                    for hid in T.Parallel(head):
                        acc_s[hid] = acc_s[hid] + T.cast(g[bid, base + i, hid], "float32")
                        out[bid, base + i, hid] = T.cast(acc_s[hid], dtype)

        return chunk_cumsum_bthd_kernel
