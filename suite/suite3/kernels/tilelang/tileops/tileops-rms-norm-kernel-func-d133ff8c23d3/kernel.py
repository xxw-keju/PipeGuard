# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    N_padded = pipeguard_bindings['N_padded']
    dtype = pipeguard_bindings['dtype']
    eps = pipeguard_bindings['eps']
    partial_min_elements = pipeguard_bindings['partial_min_elements']
    sm_count = pipeguard_bindings['sm_count']

    @tilelang.jit(out_idx=[2])
    def _func(block_m, threads):
        row_guard = M % block_m != 0
        per_thread_partial = -(-M // block_m) > sm_count and N_padded % threads == 0 and (N_padded // threads >= partial_min_elements)

        @T.prim_func
        def main(x: T.Tensor[(M, N_padded), dtype], weight: T.Tensor[(N_padded,), dtype], y: T.Tensor[(M, N_padded), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                xsq_f32 = T.alloc_fragment((block_m, reduce_width), 'float32')
                sumsq = T.alloc_fragment((block_m,), 'float32')
                rrms = T.alloc_fragment((block_m,), 'float32')
                if per_thread_partial:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.clear(xsq_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            v = T.cast(shared_buf[i, k * threads + j], 'float32')
                            xsq_f32[i, j] += v * v
                    T.reduce_sum(xsq_f32, sumsq, dim=1)
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)
                    T.copy(shared_buf, x_local)
                else:
                    if row_guard:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(pid_m * block_m + i < M, x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                    else:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], 'float32') * T.cast(x_local[i, j], 'float32')
                    T.reduce_sum(xsq_f32, sumsq, dim=1)
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)
                for i, j in T.Parallel(block_m, N_padded):
                    if not row_guard or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast(T.cast(x_local[i, j], 'float32') * rrms[i] * T.cast(weight[j], 'float32'), dtype)
        return main
    return _func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
