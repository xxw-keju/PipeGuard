# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    A = pipeguard_bindings['A']
    B = pipeguard_bindings['B']
    denom = pipeguard_bindings['denom']
    nb = pipeguard_bindings['nb']
    op_kind = pipeguard_bindings['op_kind']
    out_dtype = pipeguard_bindings['out_dtype']
    threads = pipeguard_bindings['threads']

    @tilelang.jit(out_idx=[2, 3])
    def _func():

        @T.prim_func
        def main(mean_in: T.Tensor[(A, B), 'float32'], m2_in: T.Tensor[(A, B), 'float32'], out_stat: T.Tensor[(B,), out_dtype], out_mean: T.Tensor[(B,), out_dtype]):
            with T.Kernel(T.ceildiv(B, threads), threads=threads) as pid_b:
                tx = T.get_thread_binding()
                n_acc = T.alloc_local((1,), 'float32')
                mean_acc = T.alloc_local((1,), 'float32')
                m2_acc = T.alloc_local((1,), 'float32')
                delta = T.alloc_local((1,), 'float32')
                n_acc[0] = 0.0
                mean_acc[0] = 0.0
                m2_acc[0] = 0.0
                with T.If(pid_b * threads + tx < B):
                    with T.Then():
                        for a in T.serial(A):
                            delta[0] = mean_in[a, pid_b * threads + tx] - mean_acc[0]
                            m2_acc[0] = m2_acc[0] + m2_in[a, pid_b * threads + tx] + delta[0] * delta[0] * (n_acc[0] * nb / (n_acc[0] + nb))
                            mean_acc[0] = mean_acc[0] + delta[0] * (nb / (n_acc[0] + nb))
                            n_acc[0] = n_acc[0] + nb
                        if op_kind == 'std':
                            out_stat[pid_b * threads + tx] = T.cast(T.sqrt(m2_acc[0] / denom), out_dtype)
                        else:
                            out_stat[pid_b * threads + tx] = T.cast(m2_acc[0] / denom, out_dtype)
                        out_mean[pid_b * threads + tx] = T.cast(mean_acc[0], out_dtype)
        return main
    return _func

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); a.update({"denom":float(a["A"]*a.pop("count_per")-a.pop("correction")),"nb":float(case["build"]["factory_arguments"]["count_per"]),"out_dtype":a.pop("dtype")})
    return _resolve_kernel(**a)()
