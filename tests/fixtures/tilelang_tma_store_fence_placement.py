from pathlib import Path

from tilelang import tvm
from tvm import tirx

# Moves the elected fence_proxy_async of tilelang_tma_store_device.tvm.json
# (produced by tilelang_tma_store_kernel.py) to every thread, either before
# its shared write ("before_write") or between the write and the CTA barrier
# ("after_write").
module = tvm.ir.load_json(Path("tilelang_tma_store_device.tvm.json").read_text())
(global_var, function), = module.functions.items()
fence = []


def is_fence(statement):
    return (
        isinstance(statement, tirx.Evaluate)
        and isinstance(statement.value, tirx.Call)
        and statement.value.op.name == "tl.fence_proxy_async"
    )


def drop_fence(statement):
    if not isinstance(statement, tirx.SeqStmt):
        return None
    fence.extend(item for item in statement.seq if is_fence(item))
    kept = [item for item in statement.seq if not is_fence(item)]
    return kept[0] if len(kept) == 1 else tirx.SeqStmt(kept)


body = tirx.stmt_functor.ir_transform(function.body, None, drop_fence, ["tirx.SeqStmt"])
assert len(fence) == 1


def place_fence(position):
    def visit(statement):
        if not isinstance(statement, tirx.SeqStmt):
            return None
        items = list(statement.seq)
        writes = [
            index for index, item in enumerate(items)
            if isinstance(item, tirx.IfThenElse) and isinstance(item.then_case, tirx.BufferStore)
        ]
        if not writes:
            return None
        index = writes[0] + (1 if position == "after_write" else 0)
        items.insert(index, fence[0])
        return tirx.SeqStmt(items)

    return tirx.stmt_functor.ir_transform(body, None, visit, ["tirx.SeqStmt"])


for position in ("before_write", "after_write"):
    mutated = tvm.IRModule({global_var: function.with_body(place_fence(position))})
    text = str(mutated)
    assert text.count("fence_proxy_async") == 1, text
    Path(f"tilelang_tma_store_fence_{position}_device.tvm.json").write_text(
        tvm.ir.save_json(mutated)
    )
