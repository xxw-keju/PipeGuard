from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


@T.prim_func
def static_loop(
    source: T.Tensor((300,), T.int32),
    destination: T.Tensor((300,), T.int32),
):
    with T.Kernel(1, threads=1):
        for index in T.serial(300):
            if index % 2 == 0:
                destination[index] = source[index] + 1
            else:
                destination[index] = source[index] - 1


artifact = tilelang.lower(
    static_loop,
    target="cuda",
    enable_host_codegen=False,
    enable_device_compile=False,
)
Path("tilelang_static_loop_device.tvm.json").write_text(
    tvm.ir.save_json(artifact.device_mod)
)
