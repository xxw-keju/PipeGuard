    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_body(x, weight, out, bias):
            out_hw = out_h * out_w
            with T.Kernel(
                T.ceildiv(out_hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        spatial_idx = bx * block_n + j
                        ci = k_idx // (kernel_h * kernel_w)
                        kernel_idx = k_idx % (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w
                        kw = kernel_idx % kernel_w
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        in_bound = (
                            (spatial_idx < out_hw)
                            & (k_idx < k_total)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (ih < h)
                            & (iw < w)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[bz, ci, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_hw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_hw),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv2d_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_body(x, weight, out, bias)

            return _conv2d_bias_main

        @T.prim_func
        def _conv2d_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ):
            _conv2d_body(x, weight, out, None)

        return _conv2d_main
