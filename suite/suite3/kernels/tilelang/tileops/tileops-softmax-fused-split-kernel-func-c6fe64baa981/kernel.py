# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    dtype = pipeguard_bindings['dtype']
    fold = pipeguard_bindings['fold']
    num_segs = pipeguard_bindings['num_segs']
    op_kind = pipeguard_bindings['op_kind']
    seg_n = pipeguard_bindings['seg_n']
    threads = pipeguard_bindings['threads']

    @tilelang.jit(out_idx=[3])
    def _func():

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], seg_max: T.Tensor[(M * num_segs,), 'float32'], seg_sum: T.Tensor[(M * num_segs,), 'float32'], y: T.Tensor[(M, N), dtype]):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                held = T.alloc_fragment((1, seg_n), 'float32')
                shifted = T.alloc_fragment((1, seg_n), 'float32')
                seg_m = T.alloc_fragment((1,), 'float32')
                seg_s = T.alloc_fragment((1,), 'float32')
                part_max = T.alloc_fragment((1, threads), 'float32')
                part_sum = T.alloc_fragment((1, threads), 'float32')
                row_max = T.alloc_fragment((1,), 'float32')
                row_sum = T.alloc_fragment((1,), 'float32')
                row_scale = T.alloc_local((1,), 'float32')
                shift = T.alloc_local((1,), 'float32')
                for _, j in T.Parallel(1, seg_n):
                    held[0, j] = T.if_then_else(pid_s * seg_n + j < N, T.cast(x[pid_m, pid_s * seg_n + j], 'float32'), -T.infinity('float32'))
                T.fill(seg_m, -T.infinity('float32'))
                T.reduce_max(held, seg_m, dim=1, clear=False)
                shift[0] = T.if_then_else(seg_m[0] == -T.infinity('float32'), T.cast(0.0, 'float32'), seg_m[0])
                for _, j in T.Parallel(1, seg_n):
                    shifted[0, j] = T.exp(held[0, j] - shift[0])
                T.reduce_sum(shifted, seg_s, dim=1)
                seg_max[pid_m * num_segs + pid_s] = seg_m[0]
                seg_sum[pid_m * num_segs + pid_s] = seg_s[0]
                T.sync_grid()
                fold(seg_max, seg_sum, pid_m * num_segs, part_max, part_sum, row_max, row_sum)
                if op_kind == 'softmax':
                    row_scale[0] = T.exp(seg_m[0] - row_max[0]) / row_sum[0]
                else:
                    row_scale[0] = row_max[0] + T.log(row_sum[0])
                for _, j in T.Parallel(1, seg_n):
                    with T.If(pid_s * seg_n + j < N):
                        with T.Then():
                            if op_kind == 'softmax':
                                y[pid_m, pid_s * seg_n + j] = T.cast(shifted[0, j] * row_scale[0], dtype)
                            else:
                                y[pid_m, pid_s * seg_n + j] = T.cast(held[0, j] - row_scale[0], dtype)
        return main
    return _func

def _make_block_split_fold(num_segs, threads):
    rounds = (num_segs + threads - 1) // threads
    last = num_segs - 1
    def owned(r, t):
        return T.min(r * threads + t, last), r * threads + t <= last
    @T.macro
    def fold(seg_max, seg_sum, base, part_max, part_sum, row_max, row_sum):
        for _, t in T.Parallel(1, threads):
            part_max[0, t] = -T.infinity("float32")
            for r in T.serial(rounds):
                s, mine = owned(r, t)
                part_max[0, t] = T.max(part_max[0, t], T.if_then_else(mine, seg_max[base + s], -T.infinity("float32")))
        T.fill(row_max, -T.infinity("float32"))
        T.reduce_max(part_max, row_max, dim=1, clear=False)
        for _, t in T.Parallel(1, threads):
            part_sum[0, t] = 0.0
            for r in T.serial(rounds):
                s, mine = owned(r, t)
                part_sum[0, t] = part_sum[0, t] + T.if_then_else(T.And(mine, seg_max[base + s] != -T.infinity("float32")), seg_sum[base + s] * T.exp(seg_max[base + s] - row_max[0]), T.cast(0.0, "float32"))
        T.reduce_sum(part_sum, row_sum, dim=1)
    return fold

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    a = dict(case["build"]["factory_arguments"]); a["num_segs"] = (a["N"] + a["seg_n"] - 1) // a["seg_n"]; a["fold"] = _make_block_split_fold(a["num_segs"], a["threads"])
    return _resolve_kernel(**a)()
