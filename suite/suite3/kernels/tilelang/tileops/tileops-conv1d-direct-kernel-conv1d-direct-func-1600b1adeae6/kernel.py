# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
CONV_SWIZZLE_PANEL = 2

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    c_in = pipeguard_bindings['c_in']
    c_out = pipeguard_bindings['c_out']
    dilation_l = pipeguard_bindings['dilation_l']
    dtype = pipeguard_bindings['dtype']
    has_bias = pipeguard_bindings['has_bias']
    kernel_l = pipeguard_bindings['kernel_l']
    l_in = pipeguard_bindings['l_in']
    n = pipeguard_bindings['n']
    out_l = pipeguard_bindings['out_l']
    pad_left = pipeguard_bindings['pad_left']
    stride_l = pipeguard_bindings['stride_l']

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'])
    def _conv1d_direct_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv1d_direct_body(x, weight, out, bias):
            with T.Kernel(T.ceildiv(out_l, block_n), T.ceildiv(c_out, block_m), n, threads=threads) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                for kw in T.serial(kernel_l):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        ol = bx * block_n + j
                        il = ol * stride_l + kw * dilation_l - pad_left
                        valid = (oc < c_out) & (ol < out_l) & (il >= 0) & (il < l_in)
                        out_local[i, j] += T.if_then_else(valid, T.cast(x[bz, oc, il], accum_dtype) * T.cast(weight[oc, 0, kw], accum_dtype), T.cast(0.0, accum_dtype))
                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if oc < c_out and ol < out_l:
                        if has_bias:
                            out[bz, oc, ol] = T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype)
                        else:
                            out[bz, oc, ol] = T.cast(out_local[i, j], dtype)
        if has_bias:

            @T.prim_func
            def _conv1d_direct_bias_main(x: T.Tensor((n, c_in, l_in), dtype), weight: T.Tensor((c_out, 1, kernel_l), dtype), out: T.Tensor((n, c_out, out_l), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv1d_direct_body(x, weight, out, bias)
            return _conv1d_direct_bias_main

        @T.prim_func
        def _conv1d_direct_main(x: T.Tensor((n, c_in, l_in), dtype), weight: T.Tensor((c_out, 1, kernel_l), dtype), out: T.Tensor((n, c_out, out_l), dtype)):
            _conv1d_direct_body(x, weight, out, None)
        return _conv1d_direct_main
    return _conv1d_direct_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
