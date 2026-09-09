    @tilelang.jit(
        out_idx=[-2, -1],
        compile_flags=raw_compile_flags,
    )
    def _decode_func(threads=32):
        @T.prim_func
        def deltanet_decode_raw_cuda_flastyle(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                vid = block % (dim_v // v_tile)
                nh = block // (dim_v // v_tile)
                bid = nh // head
                hid = nh - bid * head
                v_lane = tx // raw_group_size
                k_rank = tx - v_lane * raw_group_size
                v_idx = vid * v_tile + v_lane
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim_k], "float32")
                q_shared = T.alloc_shared([dim_k], "float32")
                h = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                v_new = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim_k, 32)):
                    kk = tx + i * 32
                    if kk < dim_k:
                        k_shared[kk] = T.cast(k[bid, hid, kk], "float32")
                        q_shared[kk] = T.cast(q[bid, hid, kk], "float32")
                T.sync_threads()

                beta_val = T.cast(beta[bid, hid], "float32")

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_val = T.cast(state[bid, hid, kk, v_idx], "float32")
                    h[ii] = h_val
                    old_partial += h_val * k_shared[kk]

                old_val = old_partial + T.shfl_down(old_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    v_new = beta_val * (T.cast(v[bid, hid, v_idx], "float32") - old_val)
                v_new = T.shfl_sync(v_new, v_lane * raw_group_size, width=raw_group_size)

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_new = h[ii] + k_shared[kk] * v_new
                    new_state[bid, hid, kk, v_idx] = T.cast(h_new, dtype)
                    out_partial += h_new * q_shared[kk]

                out_val = out_partial + T.shfl_down(out_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    o[bid, hid, v_idx] = T.cast(out_val, dtype)

        return deltanet_decode_raw_cuda_flastyle
