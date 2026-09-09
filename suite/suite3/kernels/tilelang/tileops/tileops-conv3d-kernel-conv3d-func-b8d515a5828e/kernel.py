# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
CONV_SWIZZLE_PANEL = 2

def _kernel_factory(n: int, c_in: int, d_in: int, h_in: int, w_in: int, c_out: int, kernel_d: int, kernel_h: int, kernel_w: int, stride_d: int, stride_h: int, stride_w: int, pad_d: int, pad_h: int, pad_w: int, dilation_d: int, dilation_h: int, dilation_w: int, has_bias: bool, dtype: str='float16'):
    accum_dtype = 'float'
    out_d = (d_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    k_total = kernel_d * kernel_h * kernel_w * c_in

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'], pass_configs={'tl.enable_async_copy': False})
    def _conv3d_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv3d_body(x, weight, out, bias):
            out_dhw = out_d * out_h * out_w
            with T.Kernel(T.ceildiv(out_dhw, block_n), T.ceildiv(c_out, block_m), n, threads=threads) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_dhw), dtype, out.data)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        spatial_idx = bx * block_n + j
                        ci = k_idx // (kernel_d * kernel_h * kernel_w)
                        kernel_idx = k_idx % (kernel_d * kernel_h * kernel_w)
                        kd = kernel_idx // (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w % kernel_h
                        kw = kernel_idx % kernel_w
                        od = spatial_idx // (out_h * out_w)
                        oh = spatial_idx // out_w % out_h
                        ow = spatial_idx % out_w
                        id_ = od * stride_d + kd * dilation_d - pad_d
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        in_bound = (spatial_idx < out_dhw) & (k_idx < k_total) & (id_ >= 0) & (ih >= 0) & (iw >= 0) & (id_ < d_in) & (ih < h_in) & (iw < w_in)
                        data_shared[i, j] = T.if_then_else(in_bound, x[bz, ci, id_, ih, iw], T.cast(0.0, dtype))
                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)
                    T.gemm(weight_shared, data_shared, out_local)
                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (spatial_idx < out_dhw), T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype), T.cast(0.0, dtype))
                    else:
                        out_shared[i, j] = T.if_then_else((oc < c_out) & (spatial_idx < out_dhw), T.cast(out_local[i, j], dtype), T.cast(0.0, dtype))
                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])
        if has_bias:

            @T.prim_func
            def _conv3d_bias_main(x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype), weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype), out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv3d_body(x, weight, out, bias)
            return _conv3d_bias_main

        @T.prim_func
        def _conv3d_main(x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype), weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype), out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype)):
            _conv3d_body(x, weight, out, None)
        return _conv3d_main
    return _conv3d_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
