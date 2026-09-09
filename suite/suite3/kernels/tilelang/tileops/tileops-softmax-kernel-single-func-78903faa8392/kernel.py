# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    N_padded = pipeguard_bindings['N_padded']
    _needs_pad = pipeguard_bindings['_needs_pad']
    _neg_inf = pipeguard_bindings['_neg_inf']
    _stages_row = pipeguard_bindings['_stages_row']
    dtype = pipeguard_bindings['dtype']
    op_kind = pipeguard_bindings['op_kind']

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], y: T.Tensor[(M, N_padded), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                staged = T.alloc_shared((block_m, N_padded if _stages_row else 1), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), 'float32')
                row_max = T.alloc_fragment((block_m,), 'float32')
                row_sum = T.alloc_fragment((block_m,), 'float32')
                row_scale = T.alloc_fragment((block_m,), 'float32')
                if _stages_row:
                    if _needs_pad:
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                staged[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), x[pid_m * block_m + i, j], T.cast(_neg_inf, dtype))
                    else:
                        T.copy(x[pid_m * block_m, 0], staged)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(staged[i, j], 'float32')
                elif _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), T.cast(x[pid_m * block_m + i, j], 'float32'), T.cast(_neg_inf, 'float32'))
                else:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x[pid_m * block_m + i, j], 'float32')
                T.fill(row_max, -T.infinity('float32'))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)
                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)
                if op_kind == 'softmax':
                    for i in T.Parallel(block_m):
                        row_scale[i] = 1.0 / row_sum[i]
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = x_f32[i, j] * row_scale[i]
                else:
                    for i in T.Parallel(block_m):
                        row_scale[i] = T.log(row_sum[i])
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(staged[i, j], 'float32') - row_max[i] - row_scale[i]
                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        y[pid_m * block_m + i, j] = T.cast(x_f32[i, j], dtype)
        return main
    return _func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    a = dict(case["build"]["factory_arguments"]); n = a["N"]
    a.update({"N_padded": ((n + 255) // 256) * 256, "_needs_pad": n % 256 != 0, "_neg_inf": float("-inf"), "_stages_row": a["op_kind"] == "log_softmax"})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
