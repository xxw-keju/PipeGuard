from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


@T.prim_func
def tma_store_kernel(destination: T.Tensor((16,), T.float32)):
    with T.Kernel(1, threads=128):
        source = T.alloc_shared((16,), T.float32)
        T.fill(source, 1.0)
        T.sync_threads()
        T.tma_copy(source[:], destination[:])
        T.tma_store_wait(0, read=False)


target = tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
})
with target:
    artifact = tilelang.lower(
        tma_store_kernel,
        target=target,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
text = tvm.ir.save_json(artifact.device_mod)
Path("tilelang_tma_store_device.tvm.json").write_text(text)
