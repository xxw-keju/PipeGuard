# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    dtype = pipeguard_bindings['dtype']
    m = pipeguard_bindings['m']
    n = pipeguard_bindings['n']
    op_kind = pipeguard_bindings['op_kind']
    out_dtype = pipeguard_bindings['out_dtype']
    vec = pipeguard_bindings['vec']

    @tilelang.jit(out_idx=[1])
    def _func(threads: int):
        step = threads * vec
        steps = n // step
        exact = steps * step == n

        @T.prim_func
        def main(x: T.Tensor((m, n), dtype), out: T.Tensor((m,), out_dtype)):
            with T.Kernel(m, threads=threads) as bm:
                acc = T.alloc_fragment((threads, vec), 'float32')
                lane = T.alloc_fragment((threads,), 'float32')
                red = T.alloc_fragment((1,), 'float32')
                T.clear(acc)
                for k in T.serial(steps):
                    for i, j in T.Parallel(threads, vec):
                        acc[i, j] += T.Cast('float32', x[bm, (k * threads + i) * vec + j])
                if not exact:
                    for i, j in T.Parallel(threads, vec):
                        col = steps * step + i * vec + j
                        with T.If(col < n):
                            with T.Then():
                                acc[i, j] += T.Cast('float32', x[bm, col])
                T.reduce_sum(acc, lane, dim=1)
                T.reduce_sum(lane, red, dim=0)
                total = red[0] / float(n) if op_kind == 'mean' else red[0]
                out[bm] = T.Cast(out_dtype, total)
        return main
    return _func

def build_kernel(case):
    return _resolve_kernel(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
