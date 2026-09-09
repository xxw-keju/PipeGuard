# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

@T.macro
def _prepared(x, row, col, in_bounds, op_kind: str):
    """One element, transformed into what *op_kind* reduces, or its identity."""
    value = T.if_then_else(in_bounds, T.cast(x[row, col], 'float32'), T.cast(0.0, 'float32'))
    if op_kind == 'l1':
        return T.abs(value)
    if op_kind == 'l2':
        return value * value
    return T.reinterpret(T.abs(value), 'int32')

@T.macro
def _finished(accumulated, op_kind: str, dtype: str):
    """The accumulator, in the output dtype."""
    if op_kind == 'inf':
        return T.cast(T.reinterpret(accumulated, 'float32'), dtype)
    return T.cast(accumulated, dtype)

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    dtype = pipeguard_bindings['dtype']
    num_tiles = pipeguard_bindings['num_tiles']
    op_kind = pipeguard_bindings['op_kind']
    out_dtype = pipeguard_bindings['out_dtype']
    partial = pipeguard_bindings['partial']
    tile_n = pipeguard_bindings['tile_n']
    work_dtype = pipeguard_bindings['work_dtype']

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                work = T.alloc_fragment((block_m, tile_n), work_dtype)
                acc = T.alloc_fragment((block_m,), work_dtype)
                tile_acc = T.alloc_fragment((block_m,), work_dtype)
                out_local = T.alloc_fragment((block_m,), out_dtype)
                T.fill(acc, 0)
                for t in T.Serial(num_tiles):
                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            work[i, j] = _prepared(x, pid_m * block_m + i, t * tile_n + j, T.And(pid_m * block_m + i < M, t * tile_n + j < N), op_kind)
                    if op_kind == 'inf':
                        T.reduce_max(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                if op_kind == 'l2' and (not partial):
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])
                for i in T.Parallel(block_m):
                    out_local[i] = _finished(acc[i], op_kind, out_dtype)
                T.copy(out_local, out[pid_m * block_m])
        return main
    return _func

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); n=a["N"]; tn=a["tile_n"]; partial=a.get("partial",False); a.update({"num_tiles":(((n+255)//256)*256+tn-1)//tn,"work_dtype":"int32","out_dtype":"float32" if partial else a["dtype"]})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
