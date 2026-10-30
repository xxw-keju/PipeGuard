"""Exact descriptor recovery from TileLang host IR (no recipe heuristics)."""
from __future__ import annotations
from typing import Any

# CUtensorMapDataType -> element bytes (cuda.h); sub-byte and packed formats
# are absent and stay unresolved.
_TENSORMAP_ELEMENT_BYTES = {
    0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 4, 11: 4, 12: 4,
}


def host_descriptors(
    host_module: Any, static_bindings: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Recover tiled TMA descriptors from the host stub's own
    ``__tvm_tensormap_create_tiled`` arguments (desc, dtype, rank, address,
    dims, byte strides, box, element strides, interleave, swizzle, l2, oob;
    every per-dimension list innermost first). Descriptors that are not
    statically resolved are omitted, so the capture still fails closed on
    them."""
    from tilelang import tvm

    tir = tvm.tirx

    def static(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        substitution: dict[Any, Any] = {}

        def visit(node: Any) -> None:
            if isinstance(node, tir.Var) and node.name in static_bindings:
                substitution[node] = tir.IntImm(node.dtype, static_bindings[node.name])

        try:
            tir.stmt_functor.post_order_visit(value, visit)
            return int(tvm.arith.Analyzer().simplify(
                tir.stmt_functor.substitute(value, substitution)
            ))
        except Exception:
            return None

    descriptors: dict[str, dict[str, Any]] = {}
    for function in host_module.functions.values():
        attrs = dict(function.attrs) if function.attrs else {}
        raw = attrs.get("tma_descriptor_args")
        if raw is None or not hasattr(raw, "items"):
            continue
        for descriptor, arguments in raw.items():
            arguments = list(arguments)
            function_name = str(getattr(arguments[0], "value", arguments[0])) if arguments else ""
            if len(arguments) < 9 or function_name != "__tvm_tensormap_create_tiled":
                continue
            rank = static(arguments[3])
            element_bytes = _TENSORMAP_ELEMENT_BYTES.get(static(arguments[2]))
            address = arguments[4]
            if rank is None or not 1 <= rank <= 5 or len(arguments) != 9 + 4 * rank:
                continue
            values = [static(value) for value in arguments[5:]]
            if element_bytes is None or not isinstance(address, tir.Var) or None in values:
                continue
            dims, byte_strides, box, element_strides = (
                values[index * rank:(index + 1) * rank] for index in range(4)
            )
            interleave = values[4 * rank]
            if (
                interleave != 0
                or any(value != 1 for value in element_strides)
                or byte_strides[0] != element_bytes
                or any(value <= 0 or value % element_bytes for value in byte_strides)
            ):
                continue
            descriptors[descriptor.name] = {
                "allocation_id": address.name,
                "element_bytes": element_bytes,
                "shape": dims[::-1],
                "strides": [value // element_bytes for value in byte_strides[::-1]],
                "tile_shape": box[::-1],
                "coordinate_order": list(reversed(range(rank))),
            }
    return descriptors

