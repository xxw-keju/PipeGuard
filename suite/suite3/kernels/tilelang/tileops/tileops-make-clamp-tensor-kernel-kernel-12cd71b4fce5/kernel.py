# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(N, dtype, has_min, has_max, threads=256, npt=8):
    """Build the Tensor-bound clamp kernel: ``y = clamp(x, lo, hi)``.

    Inputs are flat and of length *N*; ``ClampTensorFwdKernel.forward``
    broadcasts them. ``has_min`` / ``has_max`` select the three forms the
    Tensor clamp, clamp_min and clamp_max ops take.

    The result is written back into ``x``'s fragment rather than a third
    data-typed one.
    """
    if not (has_min or has_max):
        raise ValueError('_make_clamp_tensor_kernel requires has_min or has_max to be True')
    if has_min and has_max:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), lo: T.Tensor((N,), dtype), hi: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    lo_reg = T.alloc_fragment((block_size,), dtype)
                    hi_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size:(bx + 1) * block_size], x_reg)
                    T.copy(lo[bx * block_size:(bx + 1) * block_size], lo_reg)
                    T.copy(hi[bx * block_size:(bx + 1) * block_size], hi_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        x32 = T.cast(x_reg[k], 'float32')
                        lo32 = T.cast(lo_reg[k], 'float32')
                        hi32 = T.cast(hi_reg[k], 'float32')
                        r = T.min(T.max(x32, lo32), hi32)
                        r = T.if_then_else(T.isnan(hi32), hi32, r)
                        r = T.if_then_else(T.isnan(lo32), lo32, r)
                        r = T.if_then_else(T.isnan(x32), x32, r)
                        x_reg[k] = T.Cast(dtype, r)
                    T.copy(x_reg, y[bx * block_size:(bx + 1) * block_size])
            return main
        return kernel
    take_max = bool(has_min)

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), bound: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                bound_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size:(bx + 1) * block_size], x_reg)
                T.copy(bound[bx * block_size:(bx + 1) * block_size], bound_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    x32 = T.cast(x_reg[k], 'float32')
                    b32 = T.cast(bound_reg[k], 'float32')
                    r = T.max(x32, b32) if take_max else T.min(x32, b32)
                    r = T.if_then_else(T.isnan(b32), b32, r)
                    r = T.if_then_else(T.isnan(x32), x32, r)
                    x_reg[k] = T.Cast(dtype, r)
                T.copy(x_reg, y[bx * block_size:(bx + 1) * block_size])
        return main
    return kernel

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
