from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


def cross_thread_cp_async(destination_stride: int):
    # Threads 0 and 1 each cp.async 16 bytes into S, wait their own group,
    # then a CTA barrier publishes S. Stride 0 makes the two copies write the
    # same 16 bytes (an unordered async WAW); stride 16 keeps them disjoint.
    @T.prim_func
    def main(A: T.Tensor((128,), T.uint8), Out: T.Tensor((32,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((128,), T.uint8)
            if tx < 2:
                T.ptx_cp_async(
                    T.access_ptr(S[tx * destination_stride], "w", 16),
                    T.access_ptr(A[tx * 16], "r", 16),
                    16,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            T.sync_threads()
            Out[tx] = S[tx]

    return main


target = tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
})
for name, stride in (("overlap", 0), ("disjoint", 16)):
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(
            cross_thread_cp_async(stride),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    Path(f"tilelang_cp_async_cross_thread_{name}_device.tvm.json").write_text(
        tvm.ir.save_json(artifact.device_mod)
    )
