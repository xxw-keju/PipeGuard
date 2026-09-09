# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    N_padded = pipeguard_bindings['N_padded']
    _needs_pad = pipeguard_bindings['_needs_pad']
    _pad_val = pipeguard_bindings['_pad_val']
    dtype = pipeguard_bindings['dtype']
    op_kind = pipeguard_bindings['op_kind']
    out_dtype = pipeguard_bindings['out_dtype']

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.macro
        def compute(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                bool_vals = T.alloc_fragment((block_m, N_padded), 'float32')
                result = T.alloc_fragment((block_m,), 'float32')
                out_local = T.alloc_fragment((block_m,), out_dtype)
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            val = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), T.cast(x[pid_m * block_m + i, j], 'float32'), T.cast(_pad_val, 'float32'))
                            bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            bool_vals[i, j] = T.if_then_else(T.cast(shared_buf[i, j], 'float32') != 0.0, 1.0, 0.0)
                if op_kind == 'any':
                    T.reduce_max(bool_vals, result, dim=1)
                elif op_kind == 'all':
                    T.reduce_min(bool_vals, result, dim=1)
                else:
                    T.reduce_sum(bool_vals, result, dim=1)
                if op_kind == 'count_nonzero':
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i], out_dtype)
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i] > 0.5, 'int8')
                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            compute(x, out)
        return main
    return _func

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); n=a["N"]; partial=a.pop("partial",False); a.update({"N_padded":((n+255)//256)*256,"_needs_pad":n%256!=0,"_pad_val":1.0 if a["op_kind"]=="all" else 0.0,"out_dtype":"float32" if partial and a["op_kind"]=="count_nonzero" else ("int64" if a["op_kind"]=="count_nonzero" else "int8")})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
