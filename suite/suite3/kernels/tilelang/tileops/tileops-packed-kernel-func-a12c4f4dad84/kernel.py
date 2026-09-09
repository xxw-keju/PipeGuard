# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    m = pipeguard_bindings['m']
    op_kind = pipeguard_bindings['op_kind']
    term = pipeguard_bindings['term']
    words = pipeguard_bindings['words']

    @tilelang.jit(out_idx=[1])
    def _func(threads: int, vec: int):
        step = threads * vec
        steps = words // step
        exact = steps * step == words

        @T.prim_func
        def main(x: T.Tensor((m, words), 'uint32'), out: T.Tensor((m,), 'int8')):
            with T.Kernel(m, threads=threads) as bm:
                acc = T.alloc_fragment((threads, vec), 'uint32')
                lane = T.alloc_fragment((threads,), 'uint32')
                red = T.alloc_fragment((1,), 'uint32')
                T.clear(acc)
                for k in T.serial(steps):
                    for i, j in T.Parallel(threads, vec):
                        acc[i, j] = acc[i, j] | term(op_kind, x[bm, (k * threads + i) * vec + j])
                if not exact:
                    for i, j in T.Parallel(threads, vec):
                        idx = steps * step + i * vec + j
                        with T.If(idx < words):
                            with T.Then():
                                acc[i, j] = acc[i, j] | term(op_kind, x[bm, idx])
                T.reduce_max(acc, lane, dim=1)
                T.reduce_max(lane, red, dim=0)
                found = red[0] != T.Cast('uint32', 0)
                out[bm] = T.Cast('int8', T.Not(found) if op_kind == 'all' else found)
        return main
    return _func

def _packed_word_term(op_kind, w):
    if op_kind == "any": return w
    ones=T.Cast("uint32",0x01010101); high=T.Cast("uint32",0x80808080)
    return (w-ones) & (~w) & high

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); a.update({"m":a.pop("M"),"words":a.pop("N")//4,"term":_packed_word_term})
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
