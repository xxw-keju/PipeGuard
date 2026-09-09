# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
CONV_SWIZZLE_PANEL = 2

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    c_in = pipeguard_bindings['c_in']
    c_in_g = pipeguard_bindings['c_in_g']
    c_out = pipeguard_bindings['c_out']
    c_out_g = pipeguard_bindings['c_out_g']
    dilation_h = pipeguard_bindings['dilation_h']
    dilation_w = pipeguard_bindings['dilation_w']
    dtype = pipeguard_bindings['dtype']
    groups = pipeguard_bindings['groups']
    h = pipeguard_bindings['h']
    has_bias = pipeguard_bindings['has_bias']
    k_total = pipeguard_bindings['k_total']
    kernel_h = pipeguard_bindings['kernel_h']
    kernel_w = pipeguard_bindings['kernel_w']
    n = pipeguard_bindings['n']
    out_h = pipeguard_bindings['out_h']
    out_hw = pipeguard_bindings['out_hw']
    out_w = pipeguard_bindings['out_w']
    pad_h = pipeguard_bindings['pad_h']
    pad_w = pipeguard_bindings['pad_w']
    stride_h = pipeguard_bindings['stride_h']
    stride_w = pipeguard_bindings['stride_w']
    w = pipeguard_bindings['w']

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'], pass_configs={'tl.enable_async_copy': False})
    def _conv2d_group_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv2d_group_body(x, weight, out, bias):
            with T.Kernel(T.ceildiv(out_hw, block_n), T.ceildiv(c_out_g, block_m), n * groups, threads=threads) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                batch_id = bz // groups
                group_id = bz % groups
                oc_base = group_id * c_out_g + by * block_m
                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[oc_base, k_iter * block_k], weight_shared)
                    for k, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + k
                        spatial_idx = bx * block_n + j
                        ci_g = k_idx // (kernel_h * kernel_w)
                        ci = group_id * c_in_g + ci_g
                        kernel_idx = k_idx % (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w
                        kw = kernel_idx % kernel_w
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        data_shared[k, j] = T.if_then_else((spatial_idx < out_hw) & (k_idx < k_total) & (ih >= 0) & (iw >= 0) & (ih < h) & (iw < w), x[batch_id, ci, ih, iw], T.cast(0.0, dtype))
                    T.gemm(weight_shared, data_shared, out_local)
                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else((oc_g < c_out_g) & (spatial_idx < out_hw), T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype), T.cast(0.0, dtype))
                    else:
                        out_shared[i, j] = T.if_then_else((oc_g < c_out_g) & (spatial_idx < out_hw), T.cast(out_local[i, j], dtype), T.cast(0.0, dtype))
                if c_out_g % block_m == 0:
                    T.copy(out_shared, out_flat[batch_id, oc_base, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        spatial_idx = bx * block_n + j
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        if oc_g < c_out_g and spatial_idx < out_hw:
                            out[batch_id, oc, oh, ow] = out_shared[i, j]
        if has_bias:

            @T.prim_func
            def _conv2d_group_bias_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype), out: T.Tensor((n, c_out, out_h, out_w), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv2d_group_body(x, weight, out, bias)
            return _conv2d_group_bias_main

        @T.prim_func
        def _conv2d_group_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype), out: T.Tensor((n, c_out, out_h, out_w), dtype)):
            _conv2d_group_body(x, weight, out, None)
        return _conv2d_group_main
    return _conv2d_group_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
