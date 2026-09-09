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
    k_total = pipeguard_bindings['k_total']
    kernel_l = pipeguard_bindings['kernel_l']
    l_in = pipeguard_bindings['l_in']
    n = pipeguard_bindings['n']
    out_l = pipeguard_bindings['out_l']
    pad_left = pipeguard_bindings['pad_left']
    stride_l = pipeguard_bindings['stride_l']

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'])
    def _conv1d_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv1d_body(x, weight_flat, out, bias):
            with T.Kernel(T.ceildiv(out_l, block_n), T.ceildiv(c_out, block_m), n, threads=threads) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                tile_ol_start = bx * block_n
                tile_ol_end = tile_ol_start + block_n - 1
                tile_input_start = tile_ol_start * stride_l - pad_left
                tile_input_end = tile_ol_end * stride_l + (kernel_l - 1) * dilation_l - pad_left
                tile_spatial_full = (tile_ol_end < out_l) & (tile_input_start >= 0) & (tile_input_end < l_in)
                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        ol = bx * block_n + j
                        kw = k_idx // c_in
                        ci = k_idx % c_in
                        il = ol * stride_l + kw * dilation_l - pad_left
                        if tile_spatial_full & ((k_iter + 1) * block_k <= k_total):
                            data_shared[i, j] = x[bz, ci, il]
                        else:
                            in_bound = (k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in)
                            data_shared[i, j] = T.if_then_else(in_bound, x[bz, ci, il], T.cast(0.0, dtype))
                    T.gemm(weight_shared, data_shared, out_local)
                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (ol < out_l), T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype), T.cast(0.0, dtype))
                    else:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (ol < out_l), T.cast(out_local[i, j], dtype), T.cast(0.0, dtype))
                T.copy(out_shared, out[bz, by * block_m, bx * block_n])
        if has_bias:

            @T.prim_func
            def _conv1d_bias_main(x: T.Tensor((n, c_in, l_in), dtype), weight_flat: T.Tensor((c_out, k_total), dtype), out: T.Tensor((n, c_out, out_l), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv1d_body(x, weight_flat, out, bias)
            return _conv1d_bias_main

        @T.prim_func
        def _conv1d_main(x: T.Tensor((n, c_in, l_in), dtype), weight_flat: T.Tensor((c_out, k_total), dtype), out: T.Tensor((n, c_out, out_l), dtype)):
            _conv1d_body(x, weight_flat, out, None)
        return _conv1d_main
    return _conv1d_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
