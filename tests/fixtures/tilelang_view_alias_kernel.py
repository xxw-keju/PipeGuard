from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


def view_alias(reversed_rows: bool):
    # Writes through the float16 tile and reads it back through an int8
    # T.view of the same shared storage (test_tilelang_language_view.py).
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), T.float16),
        B: T.Tensor((64, 128), T.int8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_stage = T.alloc_shared((2, 64, 64), T.float16, scope="shared.dyn")
            A_i8 = T.view(A_stage, (2, 64, 128), dtype=T.int8)
            T.annotate_layout({A_i8: T.Layout((2, 64, 128), lambda s, i, j: [s, i, j])})

            for i, j in T.Parallel(64, 64):
                A_stage[0, i, j] = A[i, j]

            for i, j in T.Parallel(64, 128):
                if reversed_rows:
                    B[i, j] = A_i8[0, 63 - i, j]
                else:
                    B[i, j] = A_i8[0, i, j]

    return main


target = tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
})
for name, reversed_rows in (("same_thread", False), ("cross_thread", True)):
    with target, tvm.transform.PassContext(
        opt_level=3, config={"tl.disable_thread_storage_sync": True}
    ):
        artifact = tilelang.lower(
            view_alias(reversed_rows),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    Path(f"tilelang_view_alias_{name}_device.tvm.json").write_text(
        tvm.ir.save_json(artifact.device_mod)
    )
