# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(N, dtype, threads=256, npt=8):
    """Build Tensor-weight lerp kernel: out = a + weight * (b - a).

    ``LerpTensorFwdKernel.forward`` broadcasts ``input`` / ``end`` / ``weight``
    to the output shape and flattens them, so this PrimFunc sees three
    contiguous 1-D tensors of size ``N``, and computes in the input dtype.

    All three inputs move through register fragments so they share one
    vectorized access path; the result is written back into ``a``'s fragment.
    """

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(a: T.Tensor((N,), dtype), b: T.Tensor((N,), dtype), w: T.Tensor((N,), dtype), out: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                a_reg = T.alloc_fragment((block_size,), dtype)
                b_reg = T.alloc_fragment((block_size,), dtype)
                w_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(a[bx * block_size:(bx + 1) * block_size], a_reg)
                T.copy(b[bx * block_size:(bx + 1) * block_size], b_reg)
                T.copy(w[bx * block_size:(bx + 1) * block_size], w_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    a_reg[k] = a_reg[k] + w_reg[k] * (b_reg[k] - a_reg[k])
                T.copy(a_reg, out[bx * block_size:(bx + 1) * block_size])
        return main
    return kernel

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
