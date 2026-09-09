# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    _needs_mask = pipeguard_bindings['_needs_mask']
    _neg_inf = pipeguard_bindings['_neg_inf']
    dtype = pipeguard_bindings['dtype']
    num_tiles = pipeguard_bindings['num_tiles']
    tile_n = pipeguard_bindings['tile_n']

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], y: T.Tensor[(M,), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                tile_local = T.alloc_fragment((block_m, tile_n), dtype)
                tile_f32 = T.alloc_fragment((block_m, tile_n), 'float32')
                row_max = T.alloc_fragment((block_m,), 'float32')
                row_sum = T.alloc_fragment((block_m,), 'float32')
                prev_max = T.alloc_fragment((block_m,), 'float32')
                tile_max = T.alloc_fragment((block_m,), 'float32')
                tile_sum = T.alloc_fragment((block_m,), 'float32')
                T.fill(row_max, -T.infinity('float32'))
                T.fill(row_sum, 0.0)
                for t in T.Serial(num_tiles):
                    if _needs_mask:
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                T.copy(shared_buf, tile_local)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(tile_local[i, j], 'float32')
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, t * tile_n + j < N), T.cast(x[pid_m * block_m + i, t * tile_n + j], 'float32'), T.cast(_neg_inf, 'float32'))
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        T.copy(shared_buf, tile_local)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.cast(tile_local[i, j], 'float32')
                    T.fill(tile_max, -T.infinity('float32'))
                    T.reduce_max(tile_f32, tile_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        prev_max[i] = row_max[i]
                        row_max[i] = T.max(row_max[i], tile_max[i])
                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                    T.reduce_sum(tile_f32, tile_sum, dim=1)
                    for i in T.Parallel(block_m):
                        row_sum[i] = row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]
                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])
                T.copy(out_local, y[pid_m * block_m])
        return main
    return _func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    a = dict(case["build"]["factory_arguments"]); n = a["N"]; tile_n = a["tile_n"]
    num_tiles = (((n + 255) // 256) * 256 + tile_n - 1) // tile_n
    a.update({"num_tiles": num_tiles, "_needs_mask": num_tiles * tile_n > n, "_neg_inf": float("-inf")})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
