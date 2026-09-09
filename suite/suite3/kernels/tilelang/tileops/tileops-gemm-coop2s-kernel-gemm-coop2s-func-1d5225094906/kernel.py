# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang
_CONSUMER_BAR_WG0 = 8
_CONSUMER_BAR_WG1 = 9

def _kernel_factory(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str='float16') -> Callable:
    """Single-tile 2-consumer (cooperative) GEMM for small NN shapes (SM90).

    ``_gemm_coop2_kernel`` stripped of its persistent loop: the grid *is* the
    tile grid (``n / block_n`` by ``m / block_m``), so a CTA computes exactly
    one output tile and needs no cross-wave ring carry, no tile decode and no
    grouped swizzle. Small square shapes cannot amortize that machinery: the
    mainloop is only ``k / block_k`` iterations. One producer warpgroup issues
    TMA into a ``num_stages`` ring; two consumer warpgroups each own
    ``block_m // 2 = 64`` rows and share the ``B`` tile (split-A / shared-B).

    NN only (``A[m,k] @ B[k,n]``): ``B`` tiles load as ``(block_k, block_n)``
    and feed WGMMA with ``transpose_B=False``. Requires tiles that divide the
    problem exactly; the builder raises ``ValueError`` otherwise. Selected via
    config only (``coop2s: True``, pinned per-shape in
    ``GemmKernel._TUNED_CONFIGS``).

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``B`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NN layout).
        trans_b: Must be ``False`` (NN layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages)`` returns the compiled ``prim_func``.
    """
    if trans_a or trans_b:
        raise ValueError('_gemm_coop2s_kernel is NN-only (trans_a=False, trans_b=False)')
    accum_dtype = 'float'
    block_m = 128

    @tilelang.jit(out_idx=[-1], pass_configs={'tl.disable_warp_specialized': True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_coop2s_func(block_n: int=64, block_k: int=128, num_stages: int=4) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(f'_gemm_coop2s_kernel requires exact tiling: got m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}')
        half_m = block_m // 2
        nr = half_m * block_n // 128
        k_iters = k // block_k

        @T.prim_func
        def _gemm_coop2s_main(a: T.Tensor((m, k), dtype), b: T.Tensor((k, n), dtype), c: T.Tensor((m, n), dtype)) -> None:
            with T.Kernel(n // block_n, m // block_m, threads=384) as (bx, by):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_k, block_n), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, block_n), dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), dtype)
                T.annotate_layout({a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top), a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot), b_smem: tilelang.layout.make_swizzled_layout(b_smem), c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0), c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1)})
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps0 = T.alloc_local((1,), 'int32')
                ps1 = T.alloc_local((1,), 'int32')
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        ks = ki * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        T.tma_copy(a[m_start:m_start + half_m, ks:ks + block_k], a_smem_top[slot, :, :], barrier=ab_full[slot])
                        T.tma_copy(a[m_start + half_m:m_start + block_m, ks:ks + block_k], a_smem_bot[slot, :, :], barrier=ab_full[slot])
                        T.tma_copy(b[ks:ks + block_k, n_start:n_start + block_n], b_smem[slot, :, :], barrier=ab_full[slot])
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(a_smem_top[slot, :, :], b_smem[slot, :, :], c_local_0, transpose_B=False, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.copy(c_local_0, c_cast_0)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_cast_0, c_smem_0)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_smem_0, c[m_start, n_start])
                else:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(a_smem_bot[slot, :, :], b_smem[slot, :, :], c_local_1, transpose_B=False, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.copy(c_local_1, c_cast_1)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_cast_1, c_smem_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_smem_1, c[m_start + half_m, n_start])
        return _gemm_coop2s_main
    return _gemm_coop2s_func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    return _kernel_factory(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
