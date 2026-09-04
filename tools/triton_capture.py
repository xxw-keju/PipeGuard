#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "triton-snapshot-v2"
ADAPTER_VERSION = "triton-pgir-v2"


class CaptureError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def diagnostic(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def _git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


class TritonCapture:
    _DECLARATIONS = {
        "arith.constant",
        "tt.make_range",
        "tt.splat",
        "tt.addptr",
        "ttg.local_alloc",
        "ttg.memdesc_index",
    }
    _CONTAINERS = {"builtin.module", "tt.func", "tt.return", "scf.yield"}
    _PURE_OPERATIONS = {
        "arith.addi",
        "arith.andi",
        "arith.cmpi",
        "arith.index_cast",
        "arith.maxsi",
        "arith.minsi",
        "arith.muli",
        "arith.ori",
        "arith.remsi",
        "arith.select",
        "arith.subi",
        "arith.xori",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "tt.trans",
        "ttg.convert_layout",
        "ttg.memdesc_reshape",
        "ttg.memdesc_reinterpret",
        "ttg.memdesc_trans",
        "tt.dot",
        "tt.get_num_programs",
        "tt.get_program_id",
        "tt.reduce",
        "tt.scan",
    }
    _SUPPORTED = {
        "ttng.init_barrier",
        "ttng.arrive_barrier",
        "ttng.wait_barrier",
        "ttg.async_copy_global_to_local",
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.local_load",
        "tt.load",
        "tt.store",
    }

    def __init__(
        self,
        triton: Any,
        module: Any,
        source: str,
        command: str,
        source_ir_sha256: str,
        grid_ctas: int,
    ):
        self.triton = triton
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        self.grid_ctas = grid_ctas
        if grid_ctas != 1:
            raise CaptureError(
                "TRITON_MULTI_CTA_UNSUPPORTED",
                "PGIR-v1 capture requires explicit single-CTA launch evidence",
                {"grid_ctas": grid_ctas},
            )
        self.operations: list[Any] = []
        module.walk(self.operations.append)
        functions = [operation for operation in self.operations if operation.get_name() == "tt.func"]
        if len(functions) != 1:
            raise CaptureError(
                "TRITON_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one TTGIR tt.func",
                {"functions": len(functions)},
            )
        self.function = functions[0]
        self.block_operations: dict[int, list[int]] = {}
        self.blocks: dict[int, Any] = {}
        self.region_blocks: dict[int, set[int]] = {}
        for index, operation in enumerate(self.operations):
            block = operation.get_block()
            if block is None:
                continue
            block_id = block.id()
            region_id = block.get_parent().id()
            self.blocks[block_id] = block
            self.block_operations.setdefault(block_id, []).append(index)
            self.region_blocks.setdefault(region_id, set()).add(block_id)
        function_region = self.function.get_region(0)
        entry_blocks = sorted(self.region_blocks.get(function_region.id(), set()))
        if len(entry_blocks) != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "TTGIR function must expose one statically selected entry block",
                {"blocks": len(entry_blocks)},
            )
        self.entry_block = entry_blocks[0]
        self.num_ctas = self._integer_module_attr("ttg.num-ctas")
        self.num_warps = self._integer_module_attr("ttg.num-warps")
        self.threads_per_warp = self._integer_module_attr("ttg.threads-per-warp")
        target = module.get_operation().get_str_attr("ttg.target")
        if target != "cuda:90":
            raise CaptureError(
                "TRITON_TARGET_UNSUPPORTED",
                "Triton capture supports only cuda:90 TTGIR",
                {"target": target},
            )
        if self.num_ctas != 1:
            raise CaptureError(
                "TRITON_MULTI_CTA_UNSUPPORTED",
                "PGIR-v1 Triton capture requires ttg.num-ctas = 1",
                {"num_ctas": self.num_ctas},
            )
        if self.threads_per_warp != 32 or self.num_warps <= 0:
            raise CaptureError(
                "TRITON_PARTICIPANTS_UNRESOLVED",
                "Triton warp configuration is outside the static SM90 subset",
                {"num_warps": self.num_warps, "threads_per_warp": self.threads_per_warp},
            )
        self.actor = "cta"
        self.cta_threads = self.num_warps * self.threads_per_warp
        self.result_definitions: dict[int, tuple[int, int, Any]] = {}
        self.constants: dict[int, Any] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, dict[str, Any]] = {}
        self.output_operations: list[dict[str, Any]] = []
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}
        self.sequence = 0
        self.ranges: dict[int, tuple[int, int]] = {}
        self.pointer_bases: dict[int, tuple[str, int]] = {}
        self.pointer_values: dict[int, tuple[str, int, tuple[int, ...]]] = {}
        self.pointer_regions: dict[int, str] = {}
        self.pointer_region_sets: dict[int, tuple[str, ...]] = {}
        self.memdesc_regions: dict[int, str] = {}
        self.external_allocations: dict[int, str] = {}
        self.staged_tokens: set[int] = set()
        self.group_tokens: dict[int, int] = {}
        self.wait_tokens: set[int] = set()
        self.cp_group_seq = 0
        self.executable_nodes_seen = 0
        self.classified_nodes = 0
        self.emitting_nodes = 0
        self.elided_nodes = 0
        self.static_expansions = 0
        self.static_values: dict[int, Any] = {}
        self.current_activation_path: tuple[Any, ...] = ("root",)

    def _integer_module_attr(self, name: str) -> int:
        value = self.module.get_int_attr(name)
        if value is None:
            raise CaptureError(
                "TRITON_MODULE_ATTRIBUTE_MISSING",
                "TTGIR module lacks a required static integer attribute",
                {"attribute": name},
            )
        return int(value)

    def _value_extent(self, value: Any) -> tuple[int, int]:
        type_text = str(value.get_type())
        try:
            raw_shape = value.get_shape()
        except RuntimeError:
            raw_shape = None
        shape = [int(item) for item in raw_shape] if raw_shape else []
        if not shape:
            shape_match = re.search(r"<((?:\d+x)+)", type_text)
            if shape_match is not None:
                shape = [int(item) for item in shape_match.group(1).split("x") if item]
        match = re.search(r"(?:^|[<!x])(?:i|ui|f|bf)(\d+)", type_text)
        if not shape or match is None:
            raise CaptureError(
                "TRITON_REGION_UNRESOLVED",
                "SSA value element type or shape is unresolved",
                {"type": str(value.get_type())},
            )
        bits = int(match.group(1))
        if bits <= 0 or bits % 8:
            raise CaptureError(
                "TRITON_REGION_UNRESOLVED",
                "SSA value element type is not byte-aligned",
                {"type": str(value.get_type())},
            )
        elements = 1
        for dimension in shape:
            if dimension <= 0:
                raise CaptureError(
                    "TRITON_REGION_UNRESOLVED",
                    "SSA value has a nonpositive static dimension",
                    {"shape": shape},
                )
            elements *= dimension
        return elements, bits // 8

    def _region(self, allocation: str, address_space: str, offset: int, extent: int) -> str:
        identifier = f"region:{address_space.lower()}:{allocation}:{offset}:{extent}"
        self.regions.setdefault(identifier, {
            "id": identifier,
            "allocation_id": allocation,
            "address_space": address_space,
            "offset": offset,
            "extent": extent,
        })
        return identifier

    def _set_pointer_value(
        self,
        value_id: int,
        allocation: str,
        width: int,
        offsets: tuple[int, ...],
    ) -> None:
        if not offsets or width <= 0:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "pointer value has no finite static byte footprint",
            )
        intervals: list[tuple[int, int]] = []
        for offset in sorted(set(offsets)):
            byte_offset = offset * width
            if byte_offset < 0:
                raise CaptureError(
                    "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                    "pointer offset is negative relative to its external base",
                    {"offset": offset, "width": width},
                )
            intervals.append((byte_offset, byte_offset + width))
        merged: list[list[int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        regions = tuple(
            self._region(allocation, "GLOBAL", start, end - start)
            for start, end in merged
        )
        self.pointer_values[value_id] = (allocation, width, offsets)
        self.pointer_region_sets[value_id] = regions
        if len(regions) == 1:
            self.pointer_regions[value_id] = regions[0]
        else:
            self.pointer_regions.pop(value_id, None)

    def _masked_pointer_regions(self, value_id: int, mask: Any) -> tuple[str, ...]:
        if value_id not in self.pointer_values:
            return ()
        allocation, width, offsets = self.pointer_values[value_id]
        if isinstance(mask, (bool, int)):
            return self.pointer_region_sets.get(value_id, ()) if bool(mask) else ()
        if not isinstance(mask, (tuple, list)) or len(mask) != len(offsets):
            raise CaptureError(
                "TRITON_PREDICATE_UNRESOLVED",
                "memory mask shape does not match the pointer tensor",
                {"pointer_elements": len(offsets)},
            )
        selected = sorted(set(
            offset for offset, predicate in zip(offsets, mask) if bool(predicate)
        ))
        merged: list[list[int]] = []
        for offset in selected:
            start = offset * width
            end = start + width
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return tuple(
            self._region(allocation, "GLOBAL", start, end - start)
            for start, end in merged
        )

    def _resolve_addptr(self, operation: Any, index: int) -> bool:
        if operation.get_num_operands() != 2 or operation.get_num_results() != 1:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "tt.addptr provenance is malformed",
                {"operation_index": index},
            )
        base = self.pointer_values.get(operation.get_operand(0).id())
        offset_value = self.static_values.get(operation.get_operand(1).id())
        if base is None or offset_value is None:
            return False
        allocation, width, base_offsets = base
        if isinstance(offset_value, int) and not isinstance(offset_value, bool):
            offsets = (offset_value,) * len(base_offsets)
        elif isinstance(offset_value, (tuple, list)) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in offset_value
        ):
            offsets = tuple(offset_value)
        else:
            return False
        if len(offsets) != len(base_offsets):
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "tt.addptr offset shape does not match pointer shape",
                {
                    "operation_index": index,
                    "pointer_elements": len(base_offsets),
                    "offset_elements": len(offsets),
                },
            )
        self._set_pointer_value(
            operation.get_result(0).id(),
            allocation,
            width,
            tuple(base_offset + offset for base_offset, offset in zip(base_offsets, offsets)),
        )
        return True

    def _async_resource(self) -> str:
        identifier = f"cp_async:{self.actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [self.actor],
            "capacity": 8,
            "protocol": "CP_ASYNC",
            "scope": "CTA",
        })
        return identifier

    def _record_definitions(self) -> None:
        for index, operation in enumerate(self.operations):
            for result_index in range(operation.get_num_results()):
                result = operation.get_result(result_index)
                self.result_definitions[result.id()] = (index, result_index, result)
            if operation.get_name() == "arith.constant" and operation.get_num_results() == 1:
                try:
                    value = operation.get_constant_value()
                    self.constants[operation.get_result(0).id()] = value
                    self.static_values[operation.get_result(0).id()] = value
                except RuntimeError:
                    pass

        for index, operation in enumerate(self.operations):
            name = operation.get_name()
            if name == "tt.make_range":
                start = operation.get_int_attr("start")
                end = operation.get_int_attr("end")
                if operation.get_num_results() != 1 or start is None or end is None or int(end) <= int(start):
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.make_range must have static increasing bounds",
                        {"operation_index": index},
                    )
                elements, _width = self._value_extent(operation.get_result(0))
                if int(end) - int(start) != elements:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.make_range bounds do not match its static result shape",
                        {"operation_index": index, "elements": elements},
                    )
                result_id = operation.get_result(0).id()
                self.ranges[result_id] = (int(start), int(end))
                self.static_values[result_id] = tuple(range(int(start), int(end)))
            elif name == "tt.splat":
                if operation.get_num_operands() != 1 or operation.get_num_results() != 1:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.splat pointer provenance is malformed",
                        {"operation_index": index},
                    )
                operand = operation.get_operand(0)
                if "!tt.ptr<" not in str(operand.get_type()):
                    continue
                allocation = self.external_allocations.setdefault(
                    operand.id(), f"global-arg:{len(self.external_allocations)}"
                )
                elements, width = self._value_extent(operation.get_result(0))
                result_id = operation.get_result(0).id()
                self.pointer_bases[result_id] = (allocation, width)
                self._set_pointer_value(result_id, allocation, width, (0,) * elements)
            elif name == "tt.addptr":
                self._resolve_addptr(operation, index)
            elif name == "ttg.local_alloc":
                if operation.get_num_results() != 1:
                    raise CaptureError(
                        "TRITON_REGION_UNRESOLVED",
                        "ttg.local_alloc must have one memdesc result",
                        {"operation_index": index},
                    )
                result = operation.get_result(0)
                elements, width = self._value_extent(result)
                allocation = f"shared:ssa:{index}:0"
                self.memdesc_regions[result.id()] = self._region(
                    allocation, "SHARED", 0, elements * width
                )

    def _constant(self, value: Any, role: str) -> Any:
        if value.id() not in self.static_values:
            raise CaptureError(
                "TRITON_SSA_VALUE_UNRESOLVED",
                "Triton operand is not statically resolved in this activation",
                {"role": role, "type": str(value.get_type())},
            )
        return self.static_values[value.id()]

    def _resource(self, value: Any) -> str:
        definition = self.result_definitions.get(value.id())
        if definition is None:
            raise CaptureError(
                "TRITON_BARRIER_IDENTITY_UNRESOLVED",
                "mbarrier SSA value has no local definition",
            )
        operation_index, result_index, result = definition
        if self.operations[operation_index].get_name() != "ttg.local_alloc":
            raise CaptureError(
                "TRITON_BARRIER_ALIAS_UNSUPPORTED",
                "mbarrier must refer directly to a ttg.local_alloc result",
                {"defining_operation": self.operations[operation_index].get_name()},
            )
        result_type = str(result.get_type())
        if "!ttg.memdesc<" not in result_type or "xi64" not in result_type or "shared_memory" not in result_type:
            raise CaptureError(
                "TRITON_BARRIER_TYPE_UNSUPPORTED",
                "mbarrier allocation type is not a shared i64 memdesc",
                {"type": result_type},
            )
        identifier = f"mbarrier:ssa:{operation_index}:{result_index}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "mbarrier",
            "participants": [self.actor],
        })
        return identifier

    def _emit(self, op: str, operation_index: int, **fields: Any) -> None:
        ir_operation = self.operations[operation_index].get_name()
        activation_path = (*self.current_activation_path, operation_index)
        self.output_operations.append({
            "id": f"triton:{self.actor}:e{self.sequence:05d}:op{operation_index:05d}",
            "op": op,
            "actor": self.actor,
            "seq": self.sequence,
            "evidence": {
                "source_location": "/".join(map(str, activation_path)),
                "activation_path": list(activation_path),
                "recognizer_rule": f"{ir_operation}->{op}:v1",
                "ir_operation": ir_operation,
            },
            **fields,
        })
        self.sequence += 1

    def _capture_memory(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        operand_count = operation.get_num_operands()
        valid_counts = {1, 2, 3} if name == "tt.load" else {2, 3}
        if operand_count not in valid_counts:
            raise CaptureError(
                "TRITON_MEMORY_ACCESS_UNRESOLVED",
                "ordinary memory access has an unsupported operand form",
                {"operation": name, "operation_index": index, "operands": operand_count},
            )
        pointer_id = operation.get_operand(0).id()
        regions = self.pointer_region_sets.get(pointer_id)
        if not regions:
            raise CaptureError(
                "TRITON_MEMORY_ACCESS_UNRESOLVED",
                "ordinary memory pointer provenance is unresolved",
                {"operation": name, "operation_index": index},
            )
        mask_position = 1 if name == "tt.load" and operand_count >= 2 else 2 if name == "tt.store" and operand_count == 3 else None
        if mask_position is not None:
            mask_id = operation.get_operand(mask_position).id()
            if mask_id not in self.static_values:
                raise CaptureError(
                    "TRITON_PREDICATE_UNRESOLVED",
                    "ordinary memory mask is not statically resolved",
                    {"operation": name, "operation_index": index},
                )
            regions = self._masked_pointer_regions(pointer_id, self.static_values[mask_id])
        for region in regions:
            self._emit(
                "triton.mem.read" if name == "tt.load" else "triton.mem.write",
                index,
                region=region,
            )

    def _capture_async(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        resource = self._async_resource() if name != "ttg.local_load" else None
        if name == "ttg.async_copy_global_to_local":
            operand_count = operation.get_num_operands()
            if operand_count not in {2, 3, 4} or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_CP_ASYNC_UNRESOLVED",
                    "async copy requires source, destination, optional mask/other, and one token result",
                    {"operation_index": index, "operands": operand_count},
                )
            pointer_id = operation.get_operand(0).id()
            pointer = self.pointer_values.get(pointer_id)
            destination = self.memdesc_regions.get(operation.get_operand(1).id())
            if pointer is None or destination is None:
                raise CaptureError(
                    "TRITON_CP_ASYNC_UNRESOLVED",
                    "async copy source provenance or destination memdesc is unresolved",
                    {"operation_index": index},
                )
            allocation, width, offsets = pointer
            destination_region = self.regions[destination]
            expected_extent = len(offsets) * width
            if destination_region["extent"] != expected_extent:
                raise CaptureError(
                    "TRITON_CP_ASYNC_UNRESOLVED",
                    "async copy source shape and destination byte interval differ",
                    {
                        "source_extent": expected_extent,
                        "destination_extent": destination_region["extent"],
                    },
                )
            mask: Any = True
            if operand_count >= 3:
                mask_id = operation.get_operand(2).id()
                if mask_id not in self.static_values:
                    raise CaptureError(
                        "TRITON_PREDICATE_UNRESOLVED",
                        "async copy mask is not statically resolved",
                        {"operation_index": index},
                    )
                mask = self.static_values[mask_id]
            if isinstance(mask, (bool, int)):
                predicates = (bool(mask),) * len(offsets)
            elif isinstance(mask, (tuple, list)) and len(mask) == len(offsets):
                predicates = tuple(bool(value) for value in mask)
            else:
                raise CaptureError(
                    "TRITON_PREDICATE_UNRESOLVED",
                    "async copy mask shape does not match the source pointer tensor",
                    {"operation_index": index},
                )
            source = self.pointer_regions.get(pointer_id)
            if source is not None and all(predicates):
                self._emit(
                    "triton.cp_async.issue",
                    index,
                    resource=resource,
                    source_region=source,
                    destination_region=destination,
                    bytes=expected_extent,
                    group_owner=self.actor,
                    predicate=True,
                )
            else:
                for element, (offset, predicate) in enumerate(zip(offsets, predicates)):
                    if not predicate:
                        continue
                    source_element = self._region(
                        allocation,
                        "GLOBAL",
                        offset * width,
                        width,
                    )
                    destination_element = self._region(
                        destination_region["allocation_id"],
                        destination_region["address_space"],
                        destination_region["offset"] + element * width,
                        width,
                    )
                    self._emit(
                        "triton.cp_async.issue",
                        index,
                        resource=resource,
                        source_region=source_element,
                        destination_region=destination_element,
                        bytes=width,
                        group_owner=self.actor,
                        predicate=True,
                    )
            self.staged_tokens.add(operation.get_result(0).id())
            return
        if name == "ttg.async_commit_group":
            tokens = {
                operation.get_operand(operand).id()
                for operand in range(operation.get_num_operands())
            }
            if not tokens.issubset(self.staged_tokens) or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_CP_ASYNC_COMMIT_UNRESOLVED",
                    "async commit token operands do not match staged async copies",
                    {"operation_index": index},
                )
            self._emit(
                "triton.cp_async.commit_group",
                index,
                resource=resource,
                group_owner=self.actor,
                group_seq=self.cp_group_seq,
            )
            self.group_tokens[operation.get_result(0).id()] = self.cp_group_seq
            self.cp_group_seq += 1
            self.staged_tokens.clear()
            return
        if name == "ttg.async_wait":
            tokens = [
                operation.get_operand(operand).id()
                for operand in range(operation.get_num_operands())
            ]
            wait_count = operation.get_int_attr("num")
            if (
                any(token not in self.group_tokens for token in tokens)
                or wait_count is None
                or not 0 <= int(wait_count) <= 7
                or operation.get_num_results() != 1
            ):
                raise CaptureError(
                    "TRITON_CP_ASYNC_WAIT_UNRESOLVED",
                    "async wait requires committed SSA tokens and static num in [0, 7]",
                    {"operation_index": index, "wait_group_count": wait_count},
                )
            self._emit(
                "triton.cp_async.wait_group",
                index,
                resource=resource,
                group_owner=self.actor,
                wait_group_count=int(wait_count),
            )
            self.wait_tokens.add(operation.get_result(0).id())
            return
        if operation.get_num_operands() not in {1, 2}:
            raise CaptureError(
                "TRITON_LOCAL_LOAD_UNRESOLVED",
                "ttg.local_load requires a direct memdesc and optional observed wait token",
                {"operation_index": index},
            )
        region = self.memdesc_regions.get(operation.get_operand(0).id())
        if region is None:
            raise CaptureError(
                "TRITON_LOCAL_LOAD_UNRESOLVED",
                "local load memdesc provenance is unresolved",
                {"operation_index": index},
            )
        if (
            operation.get_num_operands() == 2
            and operation.get_operand(1).id() not in self.wait_tokens
        ):
            raise CaptureError(
                "TRITON_LOCAL_LOAD_UNRESOLVED",
                "local load completion token is unresolved",
                {"operation_index": index},
            )
        self._emit("triton.mem.read", index, region=region)

    def _region_block(self, region: Any, operation_index: int) -> int:
        blocks = sorted(self.region_blocks.get(region.id(), set()))
        if len(blocks) != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "static control region must contain exactly one block",
                {"operation_index": operation_index, "blocks": len(blocks)},
            )
        return blocks[0]

    def _execute_declaration(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        if name == "ttg.memdesc_index":
            if operation.get_num_operands() != 2 or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.memdesc_index requires base, static index, and one result",
                    {"operation_index": index},
                )
            base_region_id = self.memdesc_regions.get(operation.get_operand(0).id())
            subview_index = self.static_values.get(operation.get_operand(1).id())
            if base_region_id is None or not isinstance(subview_index, int) or isinstance(subview_index, bool):
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.memdesc_index base or index is not statically resolved",
                    {"operation_index": index},
                )
            elements, width = self._value_extent(operation.get_result(0))
            subview_extent = elements * width
            base_region = self.regions[base_region_id]
            offset = base_region["offset"] + subview_index * subview_extent
            if subview_index < 0 or offset + subview_extent > base_region["offset"] + base_region["extent"]:
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.memdesc_index is outside the static base allocation",
                    {"operation_index": index, "index": subview_index},
                )
            self.memdesc_regions[operation.get_result(0).id()] = self._region(
                base_region["allocation_id"],
                base_region["address_space"],
                offset,
                subview_extent,
            )
            return
        if name == "tt.addptr":
            self._resolve_addptr(operation, index)
            return
        if name == "tt.splat" and operation.get_num_operands() == 1 and operation.get_num_results() == 1:
            operand = operation.get_operand(0)
            if "!tt.ptr<" in str(operand.get_type()):
                return
            value = self.static_values.get(operand.id())
            if value is None:
                return
            elements, _width = self._value_extent(operation.get_result(0))
            self.static_values[operation.get_result(0).id()] = tuple(value for _ in range(elements))

    def _evaluate_pure(self, operation: Any, index: int) -> None:
        if operation.get_num_results() != 1:
            return
        name = operation.get_name()
        result = operation.get_result(0)
        if name == "tt.get_program_id":
            axis = operation.get_int_attr("axis")
            if axis not in {0, 1, 2}:
                raise CaptureError(
                    "TRITON_STATIC_VALUE_UNRESOLVED",
                    "tt.get_program_id axis is not static",
                    {"operation_index": index, "axis": axis},
                )
            self.static_values[result.id()] = 0
            return
        if name == "tt.get_num_programs":
            axis = operation.get_int_attr("axis")
            if axis not in {0, 1, 2}:
                raise CaptureError(
                    "TRITON_STATIC_VALUE_UNRESOLVED",
                    "tt.get_num_programs axis is not static",
                    {"operation_index": index, "axis": axis},
                )
            self.static_values[result.id()] = 1
            return
        if operation.get_num_operands() == 1:
            operand_id = operation.get_operand(0).id()
            if operand_id in self.pointer_values:
                allocation, width, offsets = self.pointer_values[operand_id]
                self._set_pointer_value(result.id(), allocation, width, offsets)
            if operand_id in self.memdesc_regions:
                region_id = self.memdesc_regions[operand_id]
                if name in {"ttg.memdesc_reshape", "ttg.memdesc_reinterpret", "ttg.memdesc_trans"}:
                    elements, width = self._value_extent(result)
                    if elements * width != self.regions[region_id]["extent"]:
                        raise CaptureError(
                            "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                            "memdesc view does not preserve the exact byte footprint",
                            {"operation": name, "operation_index": index},
                        )
                self.memdesc_regions[result.id()] = region_id
        operands = [
            self.static_values.get(operation.get_operand(position).id())
            for position in range(operation.get_num_operands())
        ]
        if name in {
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
            "tt.trans",
            "ttg.convert_layout",
            "arith.index_cast",
        } and len(operands) == 1 and operands[0] is not None:
            self.static_values[result.id()] = operands[0]
            return
        if name == "arith.select" and len(operands) == 3 and operands[0] is not None:
            condition, true_value, false_value = operands
            if true_value is None or false_value is None:
                return
            if isinstance(condition, (tuple, list)):
                if not isinstance(true_value, (tuple, list)) or not isinstance(false_value, (tuple, list)):
                    return
                if len(condition) != len(true_value) or len(condition) != len(false_value):
                    return
                self.static_values[result.id()] = tuple(
                    true_item if bool(predicate) else false_item
                    for predicate, true_item, false_item in zip(condition, true_value, false_value)
                )
            else:
                self.static_values[result.id()] = true_value if bool(condition) else false_value
            return
        if len(operands) != 2 or any(value is None for value in operands):
            return
        left, right = operands
        scalar_result = isinstance(left, int) and not isinstance(left, bool) and isinstance(right, int) and not isinstance(right, bool)
        if scalar_result:
            left_values = (left,)
            right_values = (right,)
        elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) and len(left) == len(right):
            left_values = tuple(left)
            right_values = tuple(right)
        else:
            return
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (*left_values, *right_values)):
            return
        if name == "arith.cmpi":
            predicate = operation.get_int_attr("predicate")
            comparisons = {
                0: lambda a, b: a == b,
                1: lambda a, b: a != b,
                2: lambda a, b: a < b,
                3: lambda a, b: a <= b,
                4: lambda a, b: a > b,
                5: lambda a, b: a >= b,
            }
            comparison = comparisons.get(predicate)
            if comparison is not None:
                values = tuple(int(comparison(a, b)) for a, b in zip(left_values, right_values))
                self.static_values[result.id()] = values[0] if scalar_result else values
            return
        functions = {
            "arith.addi": lambda a, b: a + b,
            "arith.andi": lambda a, b: a & b,
            "arith.maxsi": max,
            "arith.minsi": min,
            "arith.muli": lambda a, b: a * b,
            "arith.ori": lambda a, b: a | b,
            "arith.remsi": lambda a, b: a % b,
            "arith.subi": lambda a, b: a - b,
            "arith.xori": lambda a, b: a ^ b,
        }
        function = functions.get(name)
        if function is None or (name == "arith.remsi" and any(value == 0 for value in right_values)):
            return
        values = tuple(function(a, b) for a, b in zip(left_values, right_values))
        self.static_values[result.id()] = values[0] if scalar_result else values

    def _execute_block(self, block_id: int, path: tuple[Any, ...]) -> list[Any]:
        for index in self.block_operations.get(block_id, []):
            operation = self.operations[index]
            name = operation.get_name()
            if name in {"scf.yield", "scf.condition"}:
                return [
                    self._constant(operation.get_operand(position), f"{name} operand")
                    for position in range(operation.get_num_operands())
                ]
            if name == "tt.return":
                return []
            if name == "scf.for":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                self._execute_scf_for(operation, index, path)
                continue
            if name == "scf.if":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                self._execute_scf_if(operation, index, path)
                continue
            if name == "scf.while":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                self._execute_scf_while(operation, index, path)
                continue
            if name == "scf.execute_region":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                block = self._region_block(operation.get_region(0), index)
                yielded = self._execute_block(block, (*path, "execute_region", index))
                if len(yielded) != operation.get_num_results():
                    raise CaptureError(
                        "TRITON_CONTROL_FLOW_UNRESOLVED",
                        "scf.execute_region yield/result counts disagree",
                        {"operation_index": index},
                    )
                for position, value in enumerate(yielded):
                    self.static_values[operation.get_result(position).id()] = value
                continue
            self.current_activation_path = path
            self._execute_operation(operation, index)
        return []

    def _execute_scf_for(
        self,
        operation: Any,
        index: int,
        path: tuple[Any, ...],
    ) -> None:
        if operation.get_num_operands() < 3 or operation.get_num_regions() != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for has malformed bounds or body",
                {"operation_index": index},
            )
        lower = self._constant(operation.get_operand(0), "scf.for lower bound")
        upper = self._constant(operation.get_operand(1), "scf.for upper bound")
        step = self._constant(operation.get_operand(2), "scf.for step")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (lower, upper, step)) or step == 0:
            raise CaptureError(
                "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                "scf.for bounds and step must be static integers",
                {"operation_index": index, "lower": lower, "upper": upper, "step": step},
            )
        block_id = self._region_block(operation.get_region(0), index)
        block = self.blocks[block_id]
        carried = [
            self._constant(operation.get_operand(position), "scf.for initial value")
            for position in range(3, operation.get_num_operands())
        ]
        arguments = [block.get_argument(position) for position in range(block.get_num_arguments())]
        if len(arguments) != 1 + len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for block arguments do not match loop-carried values",
                {"operation_index": index},
            )
        for iteration, induction in enumerate(range(lower, upper, step)):
            self.static_expansions += 1
            self.static_values[arguments[0].id()] = induction
            for argument, value in zip(arguments[1:], carried):
                self.static_values[argument.id()] = value
            yielded = self._execute_block(block_id, (*path, "for", index, iteration, induction))
            if len(yielded) != len(carried):
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "scf.for yield count does not match loop-carried values",
                    {"operation_index": index},
                )
            carried = yielded
        if operation.get_num_results() != len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for result count does not match loop-carried values",
                {"operation_index": index},
            )
        for position, value in enumerate(carried):
            self.static_values[operation.get_result(position).id()] = value

    def _execute_scf_if(
        self,
        operation: Any,
        index: int,
        path: tuple[Any, ...],
    ) -> None:
        if operation.get_num_operands() != 1 or operation.get_num_regions() not in {1, 2}:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.if has malformed condition or regions",
                {"operation_index": index},
            )
        condition = self._constant(operation.get_operand(0), "scf.if condition")
        if condition not in {0, 1, -1, False, True}:
            raise CaptureError(
                "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                "scf.if condition must be a static Boolean",
                {"operation_index": index, "condition": condition},
            )
        branch = 0 if bool(condition) else 1
        yielded: list[Any] = []
        if branch < operation.get_num_regions():
            block = self._region_block(operation.get_region(branch), index)
            yielded = self._execute_block(block, (*path, "if", index, int(bool(condition))))
        if len(yielded) != operation.get_num_results():
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.if yield/result counts disagree",
                {"operation_index": index},
            )
        for position, value in enumerate(yielded):
            self.static_values[operation.get_result(position).id()] = value

    def _execute_scf_while(
        self,
        operation: Any,
        index: int,
        path: tuple[Any, ...],
    ) -> None:
        if operation.get_num_regions() != 2:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.while must contain before and after regions",
                {"operation_index": index},
            )
        carried = [
            self._constant(operation.get_operand(position), "scf.while initial value")
            for position in range(operation.get_num_operands())
        ]
        before_block_id = self._region_block(operation.get_region(0), index)
        after_block_id = self._region_block(operation.get_region(1), index)
        before_block = self.blocks[before_block_id]
        after_block = self.blocks[after_block_id]
        if (
            before_block.get_num_arguments() != len(carried)
            or after_block.get_num_arguments() != len(carried)
        ):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.while block arguments do not match carried values",
                {"operation_index": index},
            )
        seen: set[tuple[Any, ...]] = set()
        iteration = 0
        while True:
            fingerprint = tuple(carried)
            if fingerprint in seen:
                raise CaptureError(
                    "TRITON_NONFINITE_CONTROL_FLOW",
                    "scf.while repeats a statically resolved carried state",
                    {"operation_index": index, "iteration": iteration},
                )
            seen.add(fingerprint)
            for position, value in enumerate(carried):
                self.static_values[before_block.get_argument(position).id()] = value
            condition_values = self._execute_block(
                before_block_id,
                (*path, "while", index, iteration, "before"),
            )
            if len(condition_values) != 1 + len(carried):
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "scf.condition does not return condition and carried values",
                    {"operation_index": index},
                )
            condition, *forwarded = condition_values
            if condition not in {0, 1, -1, False, True}:
                raise CaptureError(
                    "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "scf.while condition is not statically Boolean",
                    {"operation_index": index, "condition": condition},
                )
            if not bool(condition):
                carried = forwarded
                break
            self.static_expansions += 1
            for position, value in enumerate(forwarded):
                self.static_values[after_block.get_argument(position).id()] = value
            carried = self._execute_block(
                after_block_id,
                (*path, "while", index, iteration, "after"),
            )
            if len(carried) != after_block.get_num_arguments():
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "scf.while after-region yield count is inconsistent",
                    {"operation_index": index},
                )
            iteration += 1
        if operation.get_num_results() != len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.while result count does not match carried values",
                {"operation_index": index},
            )
        for position, value in enumerate(carried):
            self.static_values[operation.get_result(position).id()] = value

    def _execute_operation(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        if name in self._CONTAINERS:
            return
        self.executable_nodes_seen += 1
        self.classified_nodes += 1
        before = len(self.output_operations)
        if name == "ttg.local_alloc":
            if operation.get_num_operands() not in {0, 1}:
                raise CaptureError(
                    "TRITON_OPERATION_UNSUPPORTED",
                    "ttg.local_alloc has an unsupported initialized form",
                    {"operation": name, "operation_index": index},
                )
            if operation.get_num_operands() == 1:
                region = self.memdesc_regions.get(operation.get_result(0).id())
                if region is None:
                    raise CaptureError(
                        "TRITON_REGION_UNRESOLVED",
                        "initialized ttg.local_alloc result region is unresolved",
                        {"operation_index": index},
                    )
                self._emit("triton.mem.write", index, region=region)
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name in self._DECLARATIONS:
            self._execute_declaration(operation, index)
            self.elided_nodes += 1
            return
        if name.startswith(("arith.", "math.")) or name in self._PURE_OPERATIONS:
            self._evaluate_pure(operation, index)
            self.elided_nodes += 1
            return
        if name not in self._SUPPORTED:
            raise CaptureError(
                "TRITON_OPERATION_UNSUPPORTED",
                "TTGIR operation has no sound capture rule",
                {"operation": name, "operation_index": index},
            )
        if name in {"tt.load", "tt.store"}:
            self._capture_memory(operation, index)
        elif name in {
            "ttg.async_copy_global_to_local",
            "ttg.async_commit_group",
            "ttg.async_wait",
            "ttg.local_load",
        }:
            self._capture_async(operation, index)
        else:
            if operation.get_num_operands() < 1:
                raise CaptureError(
                    "TRITON_OPERATION_MALFORMED",
                    "mbarrier operation lacks its SSA resource operand",
                    {"operation": name},
                )
            resource = self._resource(operation.get_operand(0))
            if name == "ttng.init_barrier":
                count = operation.get_int_attr("count")
                if count is None or int(count) <= 0:
                    raise CaptureError(
                        "TRITON_MBARRIER_COUNT_UNRESOLVED",
                        "ttng.init_barrier count must be a static positive integer",
                    )
                self._emit(
                    "triton.mbarrier.init",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    count=int(count),
                )
            elif name == "ttng.arrive_barrier":
                count = operation.get_int_attr("count")
                if count is None or int(count) <= 0 or operation.get_num_operands() != 2:
                    raise CaptureError(
                        "TRITON_MBARRIER_ARRIVE_UNRESOLVED",
                        "ttng.arrive_barrier requires static count and predicate",
                    )
                predicate_operand = operation.get_operand(1)
                predicate_value = self._constant(predicate_operand, "arrive predicate")
                if str(predicate_operand.get_type()) != "i1" or predicate_value not in {0, 1, -1}:
                    raise CaptureError(
                        "TRITON_PREDICATE_UNRESOLVED",
                        "ttng.arrive_barrier predicate must be a scalar i1 constant",
                        {"value": str(predicate_value), "type": str(predicate_operand.get_type())},
                    )
                generation = self.arrival_rank.get(resource, 0)
                self.arrival_rank[resource] = generation + 1
                self._emit(
                    "triton.mbarrier.arrive",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    generation=generation,
                    count=int(count),
                    predicate=bool(predicate_value),
                )
            else:
                if operation.get_num_operands() != 2:
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier requires one scalar phase SSA operand",
                    )
                parity = self._constant(operation.get_operand(1), "wait phase")
                if not isinstance(parity, int) or isinstance(parity, bool):
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier phase must be a scalar integer constant",
                        {"value": str(parity)},
                    )
                generation = self.wait_rank.get(resource, 0)
                if parity != generation % 2:
                    raise CaptureError(
                        "TRITON_PARITY_UNRESOLVED",
                        "wait phase SSA disagrees with the recovered gap-free stream rank",
                        {"resource": resource, "rank": generation, "parity": parity},
                    )
                self.wait_rank[resource] = generation + 1
                self._emit(
                    "triton.mbarrier.wait_parity",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    generation=generation,
                    raw_parity=parity,
                    logical_stream_id=f"{self.actor}:{resource}",
                    stream_rank=generation,
                    polling_coalesced=True,
                )
        if len(self.output_operations) > before:
            self.emitting_nodes += 1
        else:
            self.elided_nodes += 1

    def capture(self) -> dict[str, Any]:
        self._record_definitions()
        if len(self.external_allocations) > 1:
            raise CaptureError(
                "TRITON_ALIAS_UNRESOLVED",
                "multiple external pointer bases require explicit alias evidence",
                {"external_bases": len(self.external_allocations)},
            )
        self._execute_block(self.entry_block, ("root",))
        if not self.output_operations:
            raise CaptureError(
                "TRITON_OPERATIONS_MISSING",
                "TTGIR module contains no supported protocol operations",
            )
        triton_root = Path(self.triton.__file__).resolve().parents[1]
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "profile": {
                "id": "sm90-single-cta-static-v1",
                "source_ir_sha256": self.source_ir_sha256,
                "compiler_ir_stage": "post-ttgir-pipeline",
                "target_rule": "triton-cuda-90-explicit-grid-v1",
            },
            "producer": {
                "dsl_version": str(self.triton.__version__),
                "dsl_commit": _git_revision(triton_root),
                "backend_commit": "triton-nvidia-ttgir",
                "adapter_version": ADAPTER_VERSION,
                "command": self.command,
            },
            "target": {
                "architecture": "sm90",
                "warp_size": 32,
                "cta_threads": self.cta_threads,
                "shared_memory_bytes": 0,
                "grid_scope": "single_cta",
            },
            "extraction": {
                "complete": True,
                "stage": "post-ttgir-pipeline",
                "source": self.source,
                "finite_control_flow": True,
                "statically_known_participants": True,
                "fair_async_completion": True,
                "no_cluster_effects": True,
                "alias_precision": "EXACT",
                "parity_normalization_status": "RESOLVED",
                "static_expansions": self.static_expansions,
                "executable_nodes_seen": self.executable_nodes_seen,
                "classified_nodes": self.classified_nodes,
                "emitting_nodes": self.emitting_nodes,
                "elided_nodes": self.elided_nodes,
                "events_emitted": len(self.output_operations),
                "allocation_identity_evidence": "ttgir-ssa-definition-identity-v1",
                "alias_evidence": "single-external-base-v1",
                "participant_evidence": "ttgir-static-warp-geometry-plus-grid-cli-v1",
            },
            "actors": [{
                "id": self.actor,
                "kind": "PARTITION",
                "participants": list(range(self.cta_threads)),
            }],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [self.regions[key] for key in sorted(self.regions)],
            "operations": self.output_operations,
            "unsupported_constructs": [],
        }


def capture_file(path: Path, source: str, command: str, grid_ctas: int) -> dict[str, Any]:
    try:
        import triton
        from triton._C.libtriton import ir, nvidia
    except ImportError as error:
        raise CaptureError(
            "TRITON_RUNTIME_UNAVAILABLE",
            "Triton capture must run in the isolated Triton environment",
            {"error": str(error)},
        ) from error
    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    try:
        source_ir = path.read_bytes()
        module = ir.parse_mlir_module(str(path), context)
        module.verify()
    except (OSError, RuntimeError) as error:
        raise CaptureError(
            "TRITON_IR_INPUT_INVALID",
            "TTGIR MLIR module could not be parsed",
            {"path": str(path), "error": str(error)},
        ) from error
    try:
        return TritonCapture(
            triton,
            module,
            source,
            command,
            hashlib.sha256(source_ir).hexdigest(),
            grid_ctas,
        ).capture()
    except CaptureError:
        raise
    except Exception as error:
        raise CaptureError(
            "TRITON_CAPTURE_FAILURE",
            "Triton binding or capture analysis failed closed",
            {"error_type": type(error).__name__, "error": str(error)},
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a strict Triton TTGIR snapshot for PipeGuard")
    parser.add_argument("input", type=Path, help="one post-pipeline TTGIR MLIR module")
    parser.add_argument("--source", required=True, help="stable source/provenance label")
    parser.add_argument(
        "--grid-ctas",
        required=True,
        type=int,
        help="static total CTA count in the launch grid; PGIR-v1 requires 1",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(args.input, args.source, " ".join(sys.argv), args.grid_ctas)
    except CaptureError as error:
        print(json.dumps({
            "captured": False,
            "verdict": "UNSUPPORTED",
            "diagnostics": [error.diagnostic()],
        }, sort_keys=True, separators=(",", ":")))
        return 2
    options = {"sort_keys": True, "ensure_ascii": False}
    if args.pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    print(json.dumps(result, **options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
