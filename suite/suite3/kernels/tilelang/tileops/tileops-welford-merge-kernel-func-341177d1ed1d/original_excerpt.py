        @tilelang.jit(out_idx=[2, 3])
        def _func():
            @T.prim_func
            def main(
                mean_in: T.Tensor[(A, B), "float32"],  # noqa: F821
                m2_in: T.Tensor[(A, B), "float32"],  # noqa: F821
                out_stat: T.Tensor[(B,), out_dtype],
                out_mean: T.Tensor[(B,), out_dtype],
            ):
                with T.Kernel(T.ceildiv(B, threads), threads=threads) as pid_b:
                    tx = T.get_thread_binding()
                    n_acc = T.alloc_local((1,), "float32")
                    mean_acc = T.alloc_local((1,), "float32")
                    m2_acc = T.alloc_local((1,), "float32")
                    delta = T.alloc_local((1,), "float32")

                    n_acc[0] = 0.0
                    mean_acc[0] = 0.0
                    m2_acc[0] = 0.0
                    with T.If(pid_b * threads + tx < B):  # noqa: SIM117
                        with T.Then():
                            for a in T.serial(A):
                                delta[0] = mean_in[a, pid_b * threads + tx] - mean_acc[0]
                                m2_acc[0] = (
                                    m2_acc[0]
                                    + m2_in[a, pid_b * threads + tx]
                                    + delta[0] * delta[0] * (n_acc[0] * nb / (n_acc[0] + nb))
                                )
                                mean_acc[0] = mean_acc[0] + delta[0] * (nb / (n_acc[0] + nb))
                                n_acc[0] = n_acc[0] + nb
                            if op_kind == "std":
                                out_stat[pid_b * threads + tx] = T.cast(
                                    T.sqrt(m2_acc[0] / denom), out_dtype
                                )
                            else:
                                out_stat[pid_b * threads + tx] = T.cast(
                                    m2_acc[0] / denom, out_dtype
                                )
                            out_mean[pid_b * threads + tx] = T.cast(mean_acc[0], out_dtype)

            return main
