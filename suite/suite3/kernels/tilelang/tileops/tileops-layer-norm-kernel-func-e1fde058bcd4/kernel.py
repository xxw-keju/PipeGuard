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
    needs_pad = pipeguard_bindings['needs_pad']
    pad_count = pipeguard_bindings['pad_count']
    partial_min_elements = pipeguard_bindings['partial_min_elements']
    sm_count = pipeguard_bindings['sm_count']

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):
        row_guard = M % block_m != 0
        per_thread_partial = -(-M // block_m) > sm_count and N_padded % threads == 0 and (N_padded // threads >= partial_min_elements)

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], weight: T.Tensor[(N,), dtype], bias: T.Tensor[(N,), dtype], y: T.Tensor[(M, N), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if per_thread_partial:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                x_f32 = T.alloc_fragment((block_m, reduce_width), 'float32')
                acc = T.alloc_fragment((block_m,), 'float32')
                mean_val = T.alloc_fragment((block_m,), 'float32')
                rstd = T.alloc_fragment((block_m,), 'float32')
                if per_thread_partial:
                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            shared_buf[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.clear(x_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            x_f32[i, j] += T.cast(shared_buf[i, k * threads + j], 'float32')
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)
                    T.clear(x_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            d = T.cast(shared_buf[i, k * threads + j], 'float32') - mean_val[i]
                            x_f32[i, j] += d * d
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt((acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps)
                    if not needs_pad:
                        T.copy(shared_buf, x_local)
                else:
                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                    elif row_guard:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(pid_m * block_m + i < M, x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                    else:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], 'float32')
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt((acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps)
                if needs_pad and per_thread_partial:
                    for i, j in T.Parallel(block_m, N_padded):
                        if T.And(pid_m * block_m + i < M, j < N):
                            y[pid_m * block_m + i, j] = (T.cast(shared_buf[i, j], 'float32') - mean_val[i]) * rstd[i] * T.cast(weight[j], 'float32') + T.cast(bias[j], 'float32')
                elif needs_pad:
                    for i, j in T.Parallel(block_m, N_padded):
                        if T.And(pid_m * block_m + i < M, j < N):
                            y[pid_m * block_m + i, j] = (T.cast(x_local[i, j], 'float32') - mean_val[i]) * rstd[i] * T.cast(weight[j], 'float32') + T.cast(bias[j], 'float32')
                else:
                    for i, j in T.Parallel(block_m, N_padded):
                        if not row_guard or pid_m * block_m + i < M:
                            y[pid_m * block_m + i, j] = T.cast((T.cast(x_local[i, j], 'float32') - mean_val[i]) * rstd[i] * T.cast(weight[j], 'float32') + T.cast(bias[j], 'float32'), dtype)
        return main
    return _func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
