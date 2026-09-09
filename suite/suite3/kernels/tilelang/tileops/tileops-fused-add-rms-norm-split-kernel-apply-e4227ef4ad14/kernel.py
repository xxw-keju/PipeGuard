# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
WARP_LANES: int = 32

def _kernel_factory(N, eps, dtype):
    """Return the two factories that normalize one row across many blocks.

    A decode call is one row, so the row-per-block program launches a single
    block and the device idles behind it. These cut the row into
    :data:`_SPLIT_BLOCKS` pieces: the first sums the squares of its piece and
    writes the sum out, which is an output anyway; the second merges the
    partial sums, few enough that every block redoing it beats a launch that
    does it once, and scales its piece.

    The merge is a different order of fp32 additions from the single-block
    reduction, so results agree to tolerance rather than bit for bit.
    """
    accum = 'float32'

    @tilelang.jit
    def _stats(splits, threads, per):
        chunk = N // splits
        n_warps = max(threads // WARP_LANES, 1)

        @T.prim_func
        def main(x: T.Tensor([N], dtype), residual: T.Tensor([N], dtype), residual_out: T.Tensor([N], dtype), partial: T.Tensor([splits], accum)):
            with T.Kernel(splits, threads=threads) as bx:
                tx = T.get_thread_binding()
                acc = T.alloc_local([1], accum)
                a = T.alloc_local([per], dtype)
                b = T.alloc_local([per], dtype)
                total = T.alloc_local([per], dtype)
                acc[0] = T.cast(0, accum)
                for step in T.serial(chunk // (threads * per)):
                    base = bx * chunk + (step * threads + tx) * per
                    for i in T.vectorized(per):
                        a[i] = x[base + i]
                    for i in T.vectorized(per):
                        b[i] = residual[base + i]
                    for i in T.serial(per):
                        total[i] = T.cast(T.cast(a[i], accum) + T.cast(b[i], accum), dtype)
                        v = T.cast(total[i], accum)
                        acc[0] += v * v
                    for i in T.vectorized(per):
                        residual_out[base + i] = total[i]
                for step in T.serial(5):
                    acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
                warp_totals = T.alloc_shared([n_warps], accum)
                if tx % WARP_LANES == 0:
                    warp_totals[tx // WARP_LANES] = acc[0]
                T.sync_threads()
                if tx == 0:
                    acc[0] = T.cast(0, accum)
                    for w in T.serial(n_warps):
                        acc[0] += warp_totals[w]
                    partial[bx] = acc[0]
        return main

    @tilelang.jit(out_idx=[3])
    def _apply(splits, threads, per):
        chunk = N // splits

        @T.prim_func
        def main(summed: T.Tensor([N], dtype), partial: T.Tensor([splits], accum), weight: T.Tensor([N], dtype), y: T.Tensor([N], dtype)):
            with T.Kernel(splits, threads=threads) as bx:
                tx = T.get_thread_binding()
                total = T.alloc_local([1], accum)
                total[0] = T.cast(0, accum)
                for s in T.serial(splits):
                    total[0] += partial[s]
                rrms = T.rsqrt(total[0] / T.cast(N, accum) + T.cast(eps, accum))
                v = T.alloc_local([per], dtype)
                o = T.alloc_local([per], dtype)
                for step in T.serial(chunk // (threads * per)):
                    base = bx * chunk + (step * threads + tx) * per
                    for i in T.vectorized(per):
                        v[i] = summed[base + i]
                    for i in T.serial(per):
                        o[i] = T.cast(T.cast(v[i], accum) * rrms * T.cast(weight[base + i], accum), dtype)
                    for i in T.vectorized(per):
                        y[base + i] = o[i]
        return main
    return (_stats, _apply)

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
