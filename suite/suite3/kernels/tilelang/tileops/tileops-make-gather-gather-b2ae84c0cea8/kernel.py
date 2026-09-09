# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    dtype = pipeguard_bindings['dtype']
    grid = pipeguard_bindings['grid']
    hidden_size = pipeguard_bindings['hidden_size']
    num_tokens = pipeguard_bindings['num_tokens']
    physical_rows = pipeguard_bindings['physical_rows']
    rows_per_block = pipeguard_bindings['rows_per_block']
    threads = pipeguard_bindings['threads']
    zero_fill = pipeguard_bindings['zero_fill']

    @tilelang.jit(out_idx=[], compile_flags=['-O3', '-DENABLE_BF16'])
    def _gather():

        @T.prim_func
        def _gather_main(hidden_states: T.Tensor([num_tokens, hidden_size], dtype), permuted_idx: T.Tensor([physical_rows], 'int32'), expert_input: T.Tensor([physical_rows, hidden_size], dtype)):
            with T.Kernel(grid, threads=threads) as (bid,):
                tx = T.get_thread_binding()
                for local_row in T.serial(rows_per_block):
                    row = bid * rows_per_block + local_row
                    if row < physical_rows:
                        source = permuted_idx[row]
                        if not zero_fill:
                            T.copy(hidden_states[source, 0:hidden_size], expert_input[row, 0:hidden_size])
                        elif source < num_tokens:
                            T.copy(hidden_states[source, 0:hidden_size], expert_input[row, 0:hidden_size])
                        else:
                            for i in T.serial(T.ceildiv(hidden_size, threads)):
                                column = i * threads + tx
                                if column < hidden_size:
                                    expert_input[row, column] = T.Cast(dtype, 0)
        return _gather_main
    return _gather

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); vector=8; threads=min(1024,a["hidden_size"]//vector)
    while threads>0 and a["hidden_size"]%threads: threads-=1
    a.update({"threads":max(threads,1),"rows_per_block":8,"grid":(a["physical_rows"]+7)//8})
    return _resolve_kernel(**a)()
