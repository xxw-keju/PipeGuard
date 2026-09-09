# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
WARP_LANES: int = 32

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    dtype = pipeguard_bindings['dtype']
    fold = pipeguard_bindings['fold']
    num_segs = pipeguard_bindings['num_segs']

    @tilelang.jit(out_idx=[2])
    def _func():

        @T.prim_func
        def main(seg_max: T.Tensor[(M * num_segs,), 'float32'], seg_sum: T.Tensor[(M * num_segs,), 'float32'], y: T.Tensor[(M,), dtype]):
            with T.Kernel(M, threads=WARP_LANES) as pid_m:
                tx = T.get_thread_binding()
                row_max = T.alloc_local((1,), 'float32')
                row_sum = T.alloc_local((1,), 'float32')
                held = T.alloc_local((1,), 'float32')
                if tx == 0:
                    fold(seg_max, seg_sum, pid_m * num_segs, row_max, row_sum, held)
                    y[pid_m] = T.cast(row_max[0] + T.log(row_sum[0]), dtype)
        return main
    return _func

def _make_split_fold(num_segs):
    @T.macro
    def fold(seg_max, seg_sum, base, row_max, row_sum, held):
        row_max[0] = -T.infinity("float32")
        for s in T.serial(num_segs):
            row_max[0] = T.max(row_max[0], seg_max[base + s])
        row_sum[0] = 0.0
        for s in T.serial(num_segs):
            held[0] = seg_max[base + s]
            row_sum[0] = row_sum[0] + T.if_then_else(held[0] == -T.infinity("float32"), T.cast(0.0, "float32"), seg_sum[base + s] * T.exp(held[0] - row_max[0]))
    return fold

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    a = dict(case["build"]["factory_arguments"]); a["num_segs"] = (a.pop("N") + a.pop("seg_n") - 1) // case["build"]["factory_arguments"]["seg_n"]; a["fold"] = _make_split_fold(a["num_segs"])
    return _resolve_kernel(**a)()
