    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={"tl.disable_data_race_check": True},
    )
    def kernel_func(block_h: int, threads: int):
        BLOCK_H = block_h

        @T.prim_func
        def da_cumsum_fwd_placed_main(
            dt: T.Tensor((B, S, H), accum_dtype),  # type: ignore
            A: T.Tensor((H,), accum_dtype),  # type: ignore
            dt_bias: T.Tensor((H,), accum_dtype),  # type: ignore
            dt_out: T.Tensor((B, H, C, Q), dtype),  # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(B * C, ROWS_PER_CTA),
                T.ceildiv(H, BLOCK_H),
                threads=threads,
            ) as (bc_tile, bh_tile):
                dA_shared = T.alloc_shared((ROWS_PER_CTA * BLOCK_H, Q), accum_dtype)

                # Each parallel tuple owns one distinct (row, head, pos).
                # TileLang's conservative verifier cannot prove this mapping.
                for row, head, pos in T.Parallel(ROWS_PER_CTA, BLOCK_H, Q):
                    bc = bc_tile * ROWS_PER_CTA + row
                    bh = bh_tile * BLOCK_H + head
                    valid = T.And(bc < B * C, bh < H)
                    safe_bc = T.min(bc, B * C - 1)
                    safe_b = safe_bc // C
                    safe_c = safe_bc % C
                    safe_h = T.min(bh, H - 1)
                    seq = safe_c * Q + pos
                    val = T.if_then_else(valid, dt[safe_b, seq, safe_h], T.float32(0.0))

                    if has_dt_bias:
                        val = val + T.if_then_else(bh < H, dt_bias[safe_h], T.float32(0.0))
                    if dt_softplus:
                        val = T.if_then_else(
                            val <= T.float32(20.0),
                            T.log(T.float32(1.0) + T.exp(val)),
                            val,
                        )
                    val = T.min(T.max(val, T.float32(dt_min)), T.float32(dt_max))
                    val = T.if_then_else(valid, val, T.float32(0.0))
                    with T.If(valid), T.Then():
                        dt_out[bc // C, bh, bc % C, pos] = T.cast(val, dtype)
                    slot = row * BLOCK_H + head
                    dA_shared[slot, pos] = val * T.if_then_else(bh < H, A[safe_h], T.float32(0.0))

                T.sync_threads()
                T.cumsum(dA_shared, dim=1)
                T.sync_threads()

                for row, head, pos in T.Parallel(ROWS_PER_CTA, BLOCK_H, Q):
                    bc = bc_tile * ROWS_PER_CTA + row
                    bh = bh_tile * BLOCK_H + head
                    with T.If(T.And(bc < B * C, bh < H)), T.Then():
                        bb = bc // C
                        cc = bc % C
                        slot = row * BLOCK_H + head
                        dA_cumsum[bb, bh, cc, pos] = dA_shared[slot, pos]

        return da_cumsum_fwd_placed_main
