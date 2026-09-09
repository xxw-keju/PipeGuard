# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/layout/test_tilelang_layout_inference.py#L69-L79
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(out: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((128,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            total[0] = T.float32(0.0)
            for k in T.Parallel(128):
                acc[k] = T.float32(1.0)
            T.reduce_sum(acc, total, clear=False)
            if T.get_thread_binding() == 0:
                out[0] = total[0]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
