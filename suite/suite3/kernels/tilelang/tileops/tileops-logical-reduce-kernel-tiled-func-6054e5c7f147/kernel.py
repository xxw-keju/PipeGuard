# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    _pad_val = pipeguard_bindings['_pad_val']
    _past_n = pipeguard_bindings['_past_n']
    dtype = pipeguard_bindings['dtype']
    num_tiles = pipeguard_bindings['num_tiles']
    op_kind = pipeguard_bindings['op_kind']
    out_dtype = pipeguard_bindings['out_dtype']
    tile_n = pipeguard_bindings['tile_n']

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        needs_mask = _past_n or M % block_m != 0

        @T.macro
        def compute(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                bool_vals = T.alloc_fragment((block_m, tile_n), 'float32')
                acc = T.alloc_fragment((block_m,), 'float32')
                tile_acc = T.alloc_fragment((block_m,), 'float32')
                out_local = T.alloc_fragment((block_m,), out_dtype)
                if op_kind == 'all':
                    T.fill(acc, 1.0)
                else:
                    T.fill(acc, 0.0)
                for t in T.Serial(num_tiles):
                    if needs_mask:
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        bool_vals[i, j] = T.if_then_else(T.cast(shared_buf[i, j], 'float32') != 0.0, 1.0, 0.0)
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        val = T.if_then_else(T.And(pid_m * block_m + i < M, t * tile_n + j < N), T.cast(x[pid_m * block_m + i, t * tile_n + j], 'float32'), T.cast(_pad_val, 'float32'))
                                        bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                bool_vals[i, j] = T.if_then_else(T.cast(shared_buf[i, j], 'float32') != 0.0, 1.0, 0.0)
                    if op_kind == 'any':
                        T.reduce_max(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    elif op_kind == 'all':
                        T.reduce_min(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.min(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                if op_kind == 'count_nonzero':
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], out_dtype)
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i] > 0.5, 'int8')
                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            compute(x, out)
        return main
    return _func

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); n=a["N"]; tn=a["tile_n"]; partial=a.pop("partial",False); nt=(((n+255)//256)*256+tn-1)//tn; a.update({"num_tiles":nt,"_past_n":nt*tn>n,"_pad_val":1.0 if a["op_kind"]=="all" else 0.0,"out_dtype":"float32" if partial and a["op_kind"]=="count_nonzero" else ("int64" if a["op_kind"]=="count_nonzero" else "int8")})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
