# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
CONV_SWIZZLE_PANEL = 2

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    c_in = pipeguard_bindings['c_in']
    c_out = pipeguard_bindings['c_out']
    dtype = pipeguard_bindings['dtype']
    h = pipeguard_bindings['h']
    has_bias = pipeguard_bindings['has_bias']
    hw = pipeguard_bindings['hw']
    n = pipeguard_bindings['n']
    w = pipeguard_bindings['w']

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'])
    def _conv2d_1x1_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv2d_1x1_body(x, weight, out, bias):
            x_flat = T.Tensor((n, c_in, hw), dtype, x.data)
            out_flat = T.Tensor((n, c_out, hw), dtype, out.data)
            with T.Kernel(T.ceildiv(hw, block_n), T.ceildiv(c_out, block_m), n, threads=threads) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)
                    T.copy(x_flat[bz, k_iter * block_k, bx * block_n], data_shared)
                    T.gemm(weight_shared, data_shared, out_local)
                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    hw_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (hw_idx < hw), T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype), T.cast(0.0, dtype))
                    else:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (hw_idx < hw), T.cast(out_local[i, j], dtype), T.cast(0.0, dtype))
                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])
        if has_bias:

            @T.prim_func
            def _conv2d_1x1_bias_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in), dtype), out: T.Tensor((n, c_out, h, w), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv2d_1x1_body(x, weight, out, bias)
            return _conv2d_1x1_bias_main

        @T.prim_func
        def _conv2d_1x1_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in), dtype), out: T.Tensor((n, c_out, h, w), dtype)):
            _conv2d_1x1_body(x, weight, out, None)
        return _conv2d_1x1_main
    return _conv2d_1x1_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
