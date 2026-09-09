# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    A = pipeguard_bindings['A']
    B = pipeguard_bindings['B']
    out_dtype = pipeguard_bindings['out_dtype']
    threads = pipeguard_bindings['threads']

    @tilelang.jit(out_idx=[1])
    def _func():

        @T.prim_func
        def main(partials: T.Tensor[(A, B), 'float32'], out: T.Tensor[(B,), out_dtype]):
            with T.Kernel(T.ceildiv(B, threads), threads=threads) as pid_b:
                tx = T.get_thread_binding()
                acc = T.alloc_local((1,), 'int32')
                acc[0] = 0
                with T.If(pid_b * threads + tx < B):
                    with T.Then():
                        for a in T.serial(A):
                            acc[0] = T.max(acc[0], T.reinterpret(partials[a, pid_b * threads + tx], 'int32'))
                        out[pid_b * threads + tx] = T.cast(T.reinterpret(acc[0], 'float32'), out_dtype)
        return main
    return _func

def build_kernel(case):
    return _resolve_kernel(**case["build"]["factory_arguments"])()
