# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str='float16') -> Callable:
    """Split-K variant of the warp-specialized GEMM mainloop.

    The K contraction is sliced across ``split_k`` CTAs (grid z). Each CTA
    runs the same producer/consumer pipeline as ``_gemm_kernel`` over its
    K slice and writes an fp32 partial tile to the workspace
    ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype. Splitting only pays off when the natural
    (M, N) grid underfills the GPU — see ``GemmKernel.forward`` for the
    dispatch. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed (``[K, M]``).
        trans_b: Whether ``B`` is stored transposed (``[N, K]``).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size, split_k)`` returns the compiled
        ``prim_func`` producing the fp32 workspace.
    """
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={'tl.disable_warp_specialized': True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_splitk_func(block_m: int=128, block_n: int=128, block_k: int=64, num_stages: int=4, panel_size: int=16, split_k: int=2) -> Callable:
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        if k_iters % split_k != 0:
            raise ValueError(f'split_k={split_k} must divide the K-tile count evenly (k={k}, block_k={block_k} -> {k_iters} tiles)')
        k_slice = k_iters // split_k
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_splitk_main(a: T.Tensor((k, m) if trans_a else (m, k), dtype), b: T.Tensor((n, k) if trans_b else (k, n), dtype), w: T.Tensor((split_k, m, n), accum_dtype)) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads) as (bx, by, bz):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.annotate_layout({a_smem: tilelang.layout.make_swizzled_layout(a_smem), b_smem: tilelang.layout.make_swizzled_layout(b_smem)})
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                ps = T.alloc_local((1,), 'int32')
                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        k_start = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        if trans_a:
                            T.tma_copy(a[k_start:k_start + block_k, m_start:m_start + block_m], a_smem[slot, :, :], barrier=ab_full[slot])
                        else:
                            T.tma_copy(a[m_start:m_start + block_m, k_start:k_start + block_k], a_smem[slot, :, :], barrier=ab_full[slot])
                        if trans_b:
                            T.tma_copy(b[n_start:n_start + block_n, k_start:k_start + block_k], b_smem[slot, :, :], barrier=ab_full[slot])
                        else:
                            T.tma_copy(b[k_start:k_start + block_k, n_start:n_start + block_n], b_smem[slot, :, :], barrier=ab_full[slot])
                        T.barrier_arrive(ab_full[slot])
                else:
                    T.inc_max_nreg(240)
                    num_accum_regs = block_m * block_n // 128
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(a_smem[slot, :, :], b_smem[slot, :, :], c_local, transpose_A=trans_a, transpose_B=trans_b, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                            T.barrier_arrive(ab_empty[ps[0]])
                        ps[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                    T.barrier_arrive(ab_empty[ps[0]])
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local[i, j]
        return _gemm_splitk_main
    return _gemm_splitk_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
