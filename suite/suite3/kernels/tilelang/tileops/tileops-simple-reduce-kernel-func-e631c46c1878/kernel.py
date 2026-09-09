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

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), out_dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), 'float32')
                acc = T.alloc_fragment((block_m,), 'float32')
                out_local = T.alloc_fragment((block_m,), out_dtype)
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), T.cast(x[pid_m * block_m + i, j], 'float32'), T.cast(_pad_val, 'float32'))
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(shared_buf[i, j], 'float32')
                if op_kind == 'sum':
                    T.reduce_sum(x_f32, acc, dim=1)
                elif op_kind == 'mean':
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = acc[i] / float(N)
                elif op_kind == 'amax':
                    T.reduce_max(x_f32, acc, dim=1)
                elif op_kind == 'amin':
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = -x_f32[i, j]
                    T.reduce_max(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = -acc[i]
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], out_dtype)
                T.copy(out_local, out[pid_m * block_m])
        return main
    return _func

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); n=a["N"]; a.update({"N_padded":((n+255)//256)*256,"_needs_pad":n%256!=0,"_pad_val":float("-inf"),"out_dtype":a["dtype"]})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
