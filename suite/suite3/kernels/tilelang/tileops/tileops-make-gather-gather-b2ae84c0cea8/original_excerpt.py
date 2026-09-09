    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gather():
        @T.prim_func
        def _gather_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
            permuted_idx: T.Tensor([physical_rows], "int32"),
            expert_input: T.Tensor([physical_rows, hidden_size], dtype),
        ):
            with T.Kernel(grid, threads=threads) as (bid,):
                tx = T.get_thread_binding()
                for local_row in T.serial(rows_per_block):
                    row = bid * rows_per_block + local_row
                    if row < physical_rows:
                        source = permuted_idx[row]
                        if not zero_fill:
                            T.copy(
                                hidden_states[source, 0:hidden_size],
                                expert_input[row, 0:hidden_size],
                            )
                        else:
                            if source < num_tokens:
                                T.copy(
                                    hidden_states[source, 0:hidden_size],
                                    expert_input[row, 0:hidden_size],
                                )
                            else:
                                for i in T.serial(T.ceildiv(hidden_size, threads)):
                                    column = i * threads + tx
                                    if column < hidden_size:
                                        expert_input[row, column] = T.Cast(dtype, 0)

        return _gather_main
