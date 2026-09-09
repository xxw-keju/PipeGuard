@libentry()
@triton.heuristics({"BLOCK": _heur_flat_block})
@triton.jit(do_not_specialize=["TOTAL", "M", "N", "OUT_N"])
def _index_reduce_flat_kernel(
    out,
    index,
    src,
    count,
    touched,
    TOTAL,
    M,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    USE_COUNT: tl.constexpr,
    USE_TOUCHED: tl.constexpr,
    USE_CAS: tl.constexpr,
    INDEX_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    if INDEX_MAJOR:
        cols = offsets // M
        rows = offsets - cols * M
    else:
        rows = offsets // N
        cols = offsets - rows * N

    dst_cols = tl.load(index + cols, mask=mask, other=0).to(tl.int64)
    src_offsets = rows * N + cols
    out_offsets = rows * OUT_N + dst_cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)

    if REDUCE == 1:
        tl.atomic_add(out + out_offsets, values, mask=mask, sem="relaxed")
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(count + out_offsets, ones_i, mask=mask, sem="relaxed")
    elif REDUCE == 0:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur = tl.load(out + out_offsets, mask=mask, other=0.0)
            new = tl.where(stop, cur, cur * values)
            is_nan = new != new
            new = tl.where(is_nan, values, new)
            cas = tl.atomic_cas(out + out_offsets, cur, new, sem="relaxed")
            stop |= (cur == cas) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
    else:
        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur = tl.load(out + out_offsets, mask=mask, other=0.0)
                if REDUCE == 2:
                    new = tl.maximum(cur, values)
                else:
                    new = tl.minimum(cur, values)
                new = new.to(cur.dtype)
                cas = tl.atomic_cas(out + out_offsets, cur, new, sem="relaxed")
                stop |= cur == cas
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if REDUCE == 2:
                tl.atomic_max(out + out_offsets, values, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out + out_offsets, values, mask=mask, sem="relaxed")

    if USE_TOUCHED:
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(touched + out_offsets, ones_i, mask=mask, sem="relaxed")
