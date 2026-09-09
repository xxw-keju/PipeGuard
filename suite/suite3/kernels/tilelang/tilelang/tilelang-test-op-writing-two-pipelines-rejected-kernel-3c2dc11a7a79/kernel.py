# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_materialize_ws_schedule.py#L1294-L1350
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            X_shared = T.alloc_shared((64, 64), T.float16)
            Y_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("x", [X_shared], depth=2),
                        T.WSPipeline("y", [Y_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("x"),
                                    T.WSSync.producer_acquire("y"),
                                    "copy_both",
                                    T.WSSync.producer_commit("x"),
                                    T.WSSync.producer_commit("y"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("x"),
                                    T.WSSync.consumer_wait("y"),
                                    "copy_frag_x",
                                    "copy_frag_y",
                                    T.WSSync.consumer_release("x"),
                                    T.WSSync.consumer_release("y"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                # One statement (a T.Parallel op node) writing buffers of
                # two different pipelines.
                for u, v in T.Parallel(64, 64, annotations={T.WSID: "copy_both"}):
                    X_shared[u, v] = A[i * 64 + u, v]
                    Y_shared[u, v] = A[i * 64 + u, v]
                T.copy(X_shared, A_frag, annotations={T.WSID: "copy_frag_x"})
                T.copy(Y_shared, A_frag, annotations={T.WSID: "copy_frag_y"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
