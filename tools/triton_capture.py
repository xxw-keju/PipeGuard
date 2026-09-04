#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from itertools import product
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
        "ttng.inval_barrier",
        "ttng.arrive_barrier",
        "ttng.wait_barrier",
        "ttg.async_copy_global_to_local",
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.local_load",
        "tt.load",
        "tt.store",
        "ttng.async_tma_copy_global_to_local",
        "ttng.async_tma_copy_local_to_global",
        "ttng.async_tma_store_wait",
        "ttng.barrier_expect",
        "ttng.fence_async_shared",
        "ttng.warp_group_dot",
        "ttng.warp_group_dot_wait",
    }

    def __init__(
        self,
        triton: Any,
        module: Any,
        source: str,
        command: str,
        source_ir_sha256: str,
        grid_ctas: int,
        binding_evidence: dict[str, Any] | None = None,
    ):
        self.triton = triton
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        self.grid_ctas = grid_ctas
        self.binding_evidence = binding_evidence or {}
        self.unit_attributes = {
            int(index): set(attributes)
            for index, attributes in self.binding_evidence.get("unit_attributes", {}).items()
        }
        self.array_attributes = {
            int(index): attributes
            for index, attributes in self.binding_evidence.get("array_attributes", {}).items()
        }
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
        entry = self.blocks[self.entry_block]
        self.function_argument_indices = {
            entry.get_argument(index).id(): index
            for index in range(entry.get_num_arguments())
        }
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
        total_num_warps = module.get_int_attr("ttg.total-num-warps")
        self.total_num_warps = int(total_num_warps) if total_num_warps is not None else self.num_warps
        if self.total_num_warps < self.num_warps:
            raise CaptureError(
                "TRITON_PARTICIPANTS_UNRESOLVED",
                "total warp count is smaller than the default warp count",
                {"num_warps": self.num_warps, "total_num_warps": self.total_num_warps},
            )
        self.cta_threads = self.total_num_warps * self.threads_per_warp
        self.actor_specs: dict[str, dict[str, Any]] = {
            self.actor: {
                "id": self.actor,
                "kind": "WARPGROUP" if self.num_warps == 4 else "PARTITION",
                "participants": list(range(self.num_warps * self.threads_per_warp)),
            },
        }
        self.current_actor = self.actor
        self.actor_sequences: dict[str, int] = {}
        self.last_actor_event: dict[str, str] = {}
        self.pending_actor_dependencies: dict[str, list[str]] = {}
        self.result_definitions: dict[int, tuple[int, int, Any]] = {}
        self.constants: dict[int, Any] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, dict[str, Any]] = {}
        self.output_operations: list[dict[str, Any]] = []
        self.arrival_rank: dict[tuple[str, int], int] = {}
        self.wait_rank: dict[tuple[str, int], int] = {}
        self.barrier_lifecycle: dict[str, int] = {}
        self.sequence = 0
        self.ranges: dict[int, tuple[int, int]] = {}
        self.pointer_bases: dict[int, tuple[str, int]] = {}
        self.pointer_values: dict[int, tuple[str, int, tuple[int, ...]]] = {}
        self.pointer_regions: dict[int, str] = {}
        self.pointer_region_sets: dict[int, tuple[str, ...]] = {}
        self.memdesc_regions: dict[int, str] = {}
        self.descriptor_regions: dict[int, str] = {}
        self.barrier_arguments: dict[int, str] = {}
        self.external_allocations: dict[int, str] = {}
        self.staged_tokens: set[int] = set()
        self.group_tokens: dict[int, int] = {}
        self.wait_tokens: set[int] = set()
        self.cp_group_seq = 0
        self.wgmma_group_seq = 0
        self.tma_store_group_seq = 0
        self.tma_expect_rank: dict[tuple[str, int], int] = {}
        self.tma_issue_generation: dict[tuple[str, int], int] = {}
        self.wgmma_results: dict[int, int] = {}
        self.register_regions: dict[int, str] = {}
        self.pending_proxy_fence: int | None = None
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

    def _record_function_arguments(self) -> None:
        block = self.blocks[self.entry_block]
        for index in range(block.get_num_arguments()):
            argument = block.get_argument(index)
            type_text = str(argument.get_type())
            if "!ttg.memdesc<" in type_text:
                elements, width = self._value_extent(argument)
                allocation = f"shared-arg:{index}"
                region = self._region(allocation, "SHARED", 0, elements * width)
                self.memdesc_regions[argument.id()] = region
                if "xi64" in type_text:
                    self.barrier_arguments[argument.id()] = f"mbarrier:arg:{index}"
            elif "!tt.tensordesc<" in type_text:
                elements, width = self._value_extent(argument)
                self.descriptor_regions[argument.id()] = self._region(
                    f"tensordesc-arg:{index}",
                    "GLOBAL",
                    0,
                    elements * width,
                )

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
        identifier = f"cp_async:{self.current_actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [self.current_actor],
            "capacity": 8,
            "protocol": "CP_ASYNC",
            "scope": "CTA",
        })
        return identifier

    def _wgmma_resource(self) -> str:
        participants = self.actor_specs[self.current_actor]["participants"]
        if (
            len(participants) != 128
            or participants != list(range(participants[0], participants[0] + 128))
            or participants[0] % 128
        ):
            raise CaptureError(
                "TRITON_WGMMA_PARTICIPANTS_UNRESOLVED",
                "warp_group_dot owner must be one complete aligned 128-lane warpgroup",
                {"actor": self.current_actor, "participants": participants},
            )
        identifier = f"wgmma:{self.current_actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [self.current_actor],
            "capacity": 8,
            "protocol": "WGMMA",
            "scope": "CTA",
        })
        return identifier

    def _register_region(self, value: Any) -> str:
        value_id = value.id()
        if value_id in self.register_regions:
            return self.register_regions[value_id]
        elements, width = self._value_extent(value)
        definition = self.result_definitions.get(value_id)
        if definition is None:
            raise CaptureError(
                "TRITON_REGISTER_IDENTITY_UNRESOLVED",
                "register accumulator must have a stable local SSA definition",
                {"type": str(value.get_type())},
            )
        operation_index, result_index, _result = definition
        allocation = f"register:ssa:{operation_index}:{result_index}"
        region = self._region(allocation, "REGISTER", 0, elements * width)
        self.register_regions[value_id] = region
        return region

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
                argument_index = self.function_argument_indices.get(operand.id())
                label = (
                    f"global-arg:{argument_index}"
                    if argument_index is not None
                    else f"global-base:{len(self.external_allocations)}"
                )
                allocation = self.external_allocations.setdefault(operand.id(), label)
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
                allocation_offset = operation.get_int_attr("allocation.offset")
                allocation = "shared-memory" if allocation_offset is not None else f"shared:ssa:{index}:0"
                offset = int(allocation_offset) if allocation_offset is not None else 0
                self.memdesc_regions[result.id()] = self._region(
                    allocation, "SHARED", offset, elements * width
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
        if value.id() in self.barrier_arguments:
            identifier = self.barrier_arguments[value.id()]
            self.resources.setdefault(identifier, {
                "id": identifier,
                "kind": "mbarrier",
                "participants": [self.current_actor],
            })
            if self.current_actor not in self.resources[identifier]["participants"]:
                self.resources[identifier]["participants"].append(self.current_actor)
            return identifier
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
            "participants": [self.current_actor],
        })
        if self.current_actor not in self.resources[identifier]["participants"]:
            self.resources[identifier]["participants"].append(self.current_actor)
        return identifier

    def _emit(self, op: str, operation_index: int, **fields: Any) -> None:
        ir_operation = self.operations[operation_index].get_name()
        activation_path = (*self.current_activation_path, operation_index)
        actor = self.current_actor
        sequence = self.actor_sequences.get(actor, 0)
        dependencies = self.pending_actor_dependencies.pop(actor, [])
        if dependencies:
            dependencies.extend(fields.pop("depends_on", []))
            fields["depends_on"] = sorted(set(dependencies))
        identifier = f"triton:{actor}:e{sequence:05d}:op{operation_index:05d}"
        self.output_operations.append({
            "id": identifier,
            "op": op,
            "actor": actor,
            "seq": sequence,
            "evidence": {
                "source_location": "/".join(map(str, activation_path)),
                "activation_path": list(activation_path),
                "recognizer_rule": f"{ir_operation}->{op}:v1",
                "ir_operation": ir_operation,
            },
            **fields,
        })
        self.actor_sequences[actor] = sequence + 1
        self.last_actor_event[actor] = identifier

    def _tma_store_resource(self) -> str:
        identifier = f"tma_store:{self.current_actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [self.current_actor],
            "capacity": 8,
            "protocol": "TMA_STORE",
            "scope": "CTA",
        })
        return identifier

    def _tma_descriptor_segments(
        self,
        index: int,
        descriptor_region: str | None,
        coordinates: list[int],
        counterpart: str,
        *,
        is_load: bool,
    ) -> list[tuple[str, str, int]]:
        counterpart_region = self.regions[counterpart]
        if descriptor_region is not None and all(value == 0 for value in coordinates):
            descriptor = self.regions[descriptor_region]
            if descriptor["extent"] == counterpart_region["extent"]:
                return [
                    (descriptor_region, counterpart, descriptor["extent"])
                    if is_load
                    else (counterpart, descriptor_region, descriptor["extent"])
                ]
        operation_evidence = self.array_attributes.get(index, {})
        evidence = operation_evidence.get("descriptor") if isinstance(operation_evidence, dict) else None
        required = {
            "allocation_id",
            "element_bytes",
            "shape",
            "strides",
            "tile_shape",
            "coordinate_order",
        }
        if not isinstance(evidence, dict) or set(evidence) != required:
            raise CaptureError(
                "TRITON_TMA_DESCRIPTOR_UNRESOLVED",
                "nonzero or noncontiguous TMA coordinates require SHA-bound descriptor evidence",
                {"operation_index": index, "coordinates": coordinates},
            )
        allocation = evidence["allocation_id"]
        element_bytes = evidence["element_bytes"]
        shape = evidence["shape"]
        strides = evidence["strides"]
        tile_shape = evidence["tile_shape"]
        order = evidence["coordinate_order"]
        rank = len(shape) if isinstance(shape, list) else -1
        if (
            not isinstance(allocation, str)
            or not allocation
            or not isinstance(element_bytes, int)
            or isinstance(element_bytes, bool)
            or element_bytes <= 0
            or rank <= 0
            or not isinstance(strides, list)
            or not isinstance(tile_shape, list)
            or not isinstance(order, list)
            or len(strides) != rank
            or len(tile_shape) != rank
            or sorted(order) != list(range(rank))
            or len(coordinates) < rank
            or any(not isinstance(value, int) or value <= 0 for value in (*shape, *strides, *tile_shape))
            or any(value != 0 for value in coordinates[rank:])
        ):
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "TMA descriptor array evidence is malformed",
                {"operation_index": index},
            )
        ordered_coordinates = [coordinates[position] for position in order]
        for coordinate, tile, dimension in zip(ordered_coordinates, tile_shape, shape):
            if coordinate < 0 or coordinate + tile > dimension:
                raise CaptureError(
                    "TRITON_TMA_DESCRIPTOR_UNRESOLVED",
                    "TMA tile lies outside the evidenced descriptor shape",
                    {"operation_index": index, "coordinates": coordinates},
                )
        elements = 1
        for dimension in tile_shape:
            elements *= dimension
        if elements * element_bytes != counterpart_region["extent"]:
            raise CaptureError(
                "TRITON_TMA_DESCRIPTOR_UNRESOLVED",
                "TMA descriptor tile and shared counterpart extents differ",
                {"operation_index": index},
            )
        runs: list[list[int]] = []
        counterpart_offset = counterpart_region["offset"]
        for logical_index in product(*(range(dimension) for dimension in tile_shape)):
            descriptor_element = sum(
                (coordinate + item) * stride
                for coordinate, item, stride in zip(
                    ordered_coordinates,
                    logical_index,
                    strides,
                )
            )
            descriptor_offset = descriptor_element * element_bytes
            if (
                runs
                and descriptor_offset == runs[-1][0] + runs[-1][2]
                and counterpart_offset == runs[-1][1] + runs[-1][2]
            ):
                runs[-1][2] += element_bytes
            else:
                runs.append([descriptor_offset, counterpart_offset, element_bytes])
            counterpart_offset += element_bytes
        result: list[tuple[str, str, int]] = []
        for descriptor_offset, shared_offset, extent in runs:
            global_region = self._region(allocation, "GLOBAL", descriptor_offset, extent)
            shared_region = self._region(
                counterpart_region["allocation_id"],
                "SHARED",
                shared_offset,
                extent,
            )
            result.append(
                (global_region, shared_region, extent)
                if is_load
                else (shared_region, global_region, extent)
            )
        return result

    def _capture_tma_store(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        resource = self._tma_store_resource()
        if name == "ttng.async_tma_store_wait":
            wait_count = operation.get_int_attr("pendings")
            if (
                operation.get_num_operands() != 0
                or operation.get_num_results() != 0
                or wait_count is None
                or not 0 <= int(wait_count) <= 7
            ):
                raise CaptureError(
                    "TRITON_TMA_STORE_WAIT_UNRESOLVED",
                    "TMA store wait requires static pendings in [0, 7]",
                    {"operation_index": index, "pendings": wait_count},
                )
            if index not in self.unit_attributes:
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_REQUIRED",
                    "Python bindings cannot distinguish TMA store wait unit attributes",
                    {"operation_index": index},
                )
            attributes = self.unit_attributes[index]
            if attributes - {"read_only"}:
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_INVALID",
                    "TMA store wait binding evidence contains unknown unit attributes",
                    {"operation_index": index, "attributes": sorted(attributes)},
                )
            wait_mode = "READ_ONLY" if "read_only" in attributes else "GLOBAL"
            self._emit(
                "triton.tma_store.wait_group",
                index,
                resource=resource,
                group_owner=self.current_actor,
                wait_group_count=int(wait_count),
                wait_mode=wait_mode,
            )
            return
        if operation.get_num_operands() < 2 or operation.get_num_results() != 0:
            raise CaptureError(
                "TRITON_TMA_STORE_UNRESOLVED",
                "TMA store has malformed descriptor, coordinates, or source",
                {"operation_index": index},
            )
        descriptor = self.descriptor_regions.get(operation.get_operand(0).id())
        coordinates = [
            self._constant(operation.get_operand(position), "TMA store coordinate")
            for position in range(1, operation.get_num_operands() - 1)
        ]
        source = self.memdesc_regions.get(
            operation.get_operand(operation.get_num_operands() - 1).id()
        )
        if source is None:
            raise CaptureError(
                "TRITON_TMA_STORE_UNRESOLVED",
                "TMA store shared source provenance is unresolved",
                {"operation_index": index},
            )
        segments = self._tma_descriptor_segments(
            index,
            descriptor,
            coordinates,
            source,
            is_load=False,
        )
        if self.pending_proxy_fence is not None:
            self._emit(
                "triton.proxy_fence",
                self.pending_proxy_fence,
                from_domain="GENERIC",
                to_domain="TMA",
                scope="CTA",
                regions=sorted({source_region for source_region, _destination, _bytes in segments}),
                visibility_mask={
                    "shared": True,
                    "global_read": False,
                    "global_write": False,
                    "tensor_read": False,
                    "tensor_write": False,
                    "control_only": False,
                },
            )
            self.pending_proxy_fence = None
        for source_region, destination_region, byte_count in segments:
            self._emit(
                "triton.tma_store.issue",
                index,
                resource=resource,
                group_owner=self.current_actor,
                source_region=source_region,
                destination_region=destination_region,
                bytes=byte_count,
            )
        self._emit(
            "triton.tma_store.commit_group",
            index,
            resource=resource,
            group_owner=self.current_actor,
            group_seq=self.tma_store_group_seq,
        )
        self.tma_store_group_seq += 1

    def _capture_tma_load(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        if name == "ttng.barrier_expect":
            if operation.get_num_operands() != 2:
                raise CaptureError(
                    "TRITON_TMA_EXPECT_UNRESOLVED",
                    "barrier_expect requires barrier and predicate operands",
                    {"operation_index": index},
                )
            resource = self._resource(operation.get_operand(0))
            byte_count = operation.get_int_attr("size")
            predicate_value = self._constant(operation.get_operand(1), "barrier_expect predicate")
            if (
                byte_count is None
                or int(byte_count) <= 0
                or predicate_value not in {0, 1, -1, False, True}
            ):
                raise CaptureError(
                    "TRITON_TMA_EXPECT_UNRESOLVED",
                    "barrier_expect size and predicate must be static",
                    {"operation_index": index, "size": byte_count},
                )
            lifecycle = self.barrier_lifecycle.get(resource, 0)
            key = (resource, lifecycle)
            generation = self.tma_expect_rank.get(key, 0)
            self._emit(
                "triton.mbarrier.arrive_expect_tx",
                index,
                resource=resource,
                lifecycle_epoch=lifecycle,
                generation=generation,
                count=1,
                bytes=int(byte_count),
                predicate=bool(predicate_value),
            )
            self.tma_issue_generation[key] = generation
            if predicate_value:
                self.tma_expect_rank[key] = generation + 1
            return
        if operation.get_num_operands() < 5 or operation.get_num_results() != 0:
            raise CaptureError(
                "TRITON_TMA_LOAD_UNRESOLVED",
                "TMA load has malformed descriptor, coordinates, barrier, destination, or predicate",
                {"operation_index": index},
            )
        descriptor = self.descriptor_regions.get(operation.get_operand(0).id())
        coordinate_values = [
            self._constant(operation.get_operand(position), "TMA coordinate")
            for position in range(1, operation.get_num_operands() - 3)
        ]
        barrier_operand = operation.get_operand(operation.get_num_operands() - 3)
        destination = self.memdesc_regions.get(
            operation.get_operand(operation.get_num_operands() - 2).id()
        )
        predicate_value = self._constant(
            operation.get_operand(operation.get_num_operands() - 1),
            "TMA load predicate",
        )
        if destination is None or predicate_value not in {0, 1, -1, False, True}:
            raise CaptureError(
                "TRITON_TMA_LOAD_UNRESOLVED",
                "TMA destination or predicate provenance is unresolved",
                {
                    "operation_index": index,
                    "coordinates": coordinate_values,
                    "predicate": predicate_value,
                },
            )
        resource = self._resource(barrier_operand)
        lifecycle = self.barrier_lifecycle.get(resource, 0)
        key = (resource, lifecycle)
        if key not in self.tma_issue_generation:
            raise CaptureError(
                "TRITON_TMA_LOAD_UNRESOLVED",
                "TMA load lacks a preceding barrier_expect generation",
                {"operation_index": index, "resource": resource},
            )
        if predicate_value:
            segments = self._tma_descriptor_segments(
                index,
                descriptor,
                coordinate_values,
                destination,
                is_load=True,
            )
            for source_region, destination_region, byte_count in segments:
                self._emit(
                    "triton.tma_load.issue",
                    index,
                    resource=resource,
                    source_region=source_region,
                    destination_region=destination_region,
                    bytes=byte_count,
                    lifecycle_epoch=lifecycle,
                    generation=self.tma_issue_generation[key],
                )

    def _capture_wgmma(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        resource = self._wgmma_resource()
        if name == "ttng.warp_group_dot":
            if operation.get_num_operands() < 3 or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_WGMMA_UNRESOLVED",
                    "warp_group_dot requires two shared operands, accumulator, and one result",
                    {"operation_index": index},
                )
            regions = [
                self.memdesc_regions.get(operation.get_operand(position).id())
                for position in (0, 1)
            ]
            if any(region is None for region in regions):
                raise CaptureError(
                    "TRITON_WGMMA_UNRESOLVED",
                    "warp_group_dot shared operand provenance is unresolved",
                    {"operation_index": index},
                )
            if self.pending_proxy_fence is not None:
                self._emit(
                    "triton.proxy_fence",
                    self.pending_proxy_fence,
                    from_domain="GENERIC",
                    to_domain="TENSOR_CORE",
                    scope="CTA",
                    regions=regions,
                    visibility_mask={
                        "shared": True,
                        "global_read": False,
                        "global_write": False,
                        "tensor_read": False,
                        "tensor_write": False,
                        "control_only": False,
                    },
                )
                self.pending_proxy_fence = None
            destination = self._register_region(operation.get_operand(2))
            owner = self.current_actor
            self._emit(
                "triton.wgmma.fence_operand",
                index,
                resource=resource,
                group_owner=owner,
                regions=regions,
            )
            self._emit(
                "triton.wgmma.issue",
                index,
                resource=resource,
                group_owner=owner,
                regions=regions,
                destination_region=destination,
            )
            self._emit(
                "triton.wgmma.commit_group",
                index,
                resource=resource,
                group_owner=owner,
                group_seq=self.wgmma_group_seq,
            )
            result = operation.get_result(0)
            self.register_regions[result.id()] = destination
            self.wgmma_results[result.id()] = self.wgmma_group_seq
            self.wgmma_group_seq += 1
            return
        wait_count = operation.get_int_attr("pendings")
        if (
            wait_count is None
            or not 0 <= int(wait_count) <= 7
            or operation.get_num_operands() < 1
            or operation.get_num_results() != operation.get_num_operands()
            or not any(
                operation.get_operand(position).id() in self.wgmma_results
                for position in range(operation.get_num_operands())
            )
        ):
            raise CaptureError(
                "TRITON_WGMMA_WAIT_UNRESOLVED",
                "warp_group_dot_wait requires known dot results and static pendings",
                {"operation_index": index, "pendings": wait_count},
            )
        for completion_kind in ("OPERANDS_RELEASED", "RESULT_READY"):
            self._emit(
                "triton.wgmma.wait_group",
                index,
                resource=resource,
                group_owner=self.current_actor,
                wait_group_count=int(wait_count),
                completion_kind=completion_kind,
            )
        for position in range(operation.get_num_operands()):
            operand = operation.get_operand(position)
            result = operation.get_result(position)
            if operand.id() in self.register_regions:
                self.register_regions[result.id()] = self.register_regions[operand.id()]
            if operand.id() in self.memdesc_regions:
                self.memdesc_regions[result.id()] = self.memdesc_regions[operand.id()]
            if operand.id() in self.wgmma_results:
                self.wgmma_results[result.id()] = self.wgmma_results[operand.id()]

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
                    group_owner=self.current_actor,
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
                        group_owner=self.current_actor,
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
                group_owner=self.current_actor,
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
                group_owner=self.current_actor,
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
            if name in {"scf.yield", "scf.condition", "ttg.warp_yield", "ttg.warp_return"}:
                return [
                    self._constant(operation.get_operand(position), f"{name} operand")
                    for position in range(operation.get_num_operands())
                ]
            if name == "tt.return":
                return []
            if name == "ttg.warp_specialize":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                self._execute_warp_specialize(operation, index, path)
                continue
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

    def _forward_value(self, source: Any, target: Any) -> None:
        source_id = source.id()
        target_id = target.id()
        for mapping in (
            self.static_values,
            self.pointer_values,
            self.pointer_regions,
            self.pointer_region_sets,
            self.memdesc_regions,
            self.descriptor_regions,
            self.barrier_arguments,
            self.register_regions,
            self.wgmma_results,
        ):
            if source_id in mapping:
                mapping[target_id] = mapping[source_id]

    def _execute_warp_specialize(
        self,
        operation: Any,
        index: int,
        path: tuple[Any, ...],
    ) -> None:
        evidence = self.array_attributes.get(index)
        if not isinstance(evidence, dict) or set(evidence) != {
            "partition_num_warps",
            "warp_group_start_ids",
        }:
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_REQUIRED",
                "warp_specialize requires SHA-bound partition array attributes",
                {"operation_index": index},
            )
        partition_num_warps = evidence["partition_num_warps"]
        start_ids = evidence["warp_group_start_ids"]
        if (
            not isinstance(partition_num_warps, list)
            or not isinstance(start_ids, list)
            or len(partition_num_warps) != len(start_ids)
            or any(not isinstance(value, int) or value <= 0 for value in partition_num_warps)
            or any(not isinstance(value, int) or value < 0 for value in start_ids)
            or operation.get_num_regions() != 2
        ):
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "warp_specialize partition evidence is malformed",
                {"operation_index": index, "evidence": evidence},
            )
        parent_actor = self.current_actor
        parent_frontier = self.last_actor_event.get(parent_actor)
        default_block = self._region_block(operation.get_region(0), index)
        self._execute_block(default_block, (*path, "warp_specialize", index, "default"))

        wrapper_block = self._region_block(operation.get_region(1), index)
        wrappers = [
            operation_index
            for operation_index in self.block_operations.get(wrapper_block, [])
            if self.operations[operation_index].get_name() == "ttg.warp_specialize.partitions"
        ]
        if len(wrappers) != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "warp_specialize must contain one partition wrapper",
                {"operation_index": index},
            )
        wrapper_index = wrappers[0]
        wrapper = self.operations[wrapper_index]
        if wrapper.get_num_regions() != len(partition_num_warps):
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "partition evidence count differs from partition regions",
                {"operation_index": index},
            )
        self.executable_nodes_seen += 1
        self.classified_nodes += 1
        self.elided_nodes += 1
        partition_tails: list[str] = []
        occupied = set(self.actor_specs[parent_actor]["participants"])
        for partition, (num_warps, start_warp) in enumerate(zip(partition_num_warps, start_ids)):
            participants = list(range(
                start_warp * self.threads_per_warp,
                (start_warp + num_warps) * self.threads_per_warp,
            ))
            if not participants or participants[-1] >= self.cta_threads or occupied.intersection(participants):
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_INVALID",
                    "warp partition participants overlap or exceed CTA geometry",
                    {"operation_index": index, "partition": partition},
                )
            occupied.update(participants)
            actor = f"partition:{partition}"
            self.actor_specs[actor] = {
                "id": actor,
                "kind": "WARPGROUP" if len(participants) == 128 and participants[0] % 128 == 0 else "PARTITION",
                "participants": participants,
            }
            if parent_frontier is not None:
                self.pending_actor_dependencies[actor] = [parent_frontier]
            block_id = self._region_block(wrapper.get_region(partition), wrapper_index)
            block = self.blocks[block_id]
            if block.get_num_arguments() != wrapper.get_num_operands():
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "warp partition arguments do not match captured values",
                    {"operation_index": index, "partition": partition},
                )
            for position in range(block.get_num_arguments()):
                self._forward_value(wrapper.get_operand(position), block.get_argument(position))
            self.current_actor = actor
            self._execute_block(
                block_id,
                (*path, "warp_specialize", index, "partition", partition),
            )
            tail = self.last_actor_event.get(actor)
            if tail is not None:
                partition_tails.append(tail)
        self.current_actor = parent_actor
        if partition_tails:
            self.pending_actor_dependencies[parent_actor] = partition_tails

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
        if name in {"ttng.async_tma_copy_local_to_global", "ttng.async_tma_store_wait"}:
            self._capture_tma_store(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name in {"ttng.barrier_expect", "ttng.async_tma_copy_global_to_local"}:
            self._capture_tma_load(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name == "ttng.fence_async_shared":
            if (
                operation.get_num_operands() != 0
                or operation.get_num_results() != 0
                or operation.get_bool_attr("bCluster") is not False
                or self.pending_proxy_fence is not None
            ):
                raise CaptureError(
                    "TRITON_PROXY_FENCE_UNRESOLVED",
                    "fence_async_shared must be one unresolved CTA-scope fence",
                    {"operation_index": index},
                )
            self.pending_proxy_fence = index
            self.emitting_nodes += 1
            return
        if name in {"ttng.warp_group_dot", "ttng.warp_group_dot_wait"}:
            self._capture_wgmma(operation, index)
        elif name in {"tt.load", "tt.store"}:
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
            lifecycle = self.barrier_lifecycle.get(resource, 0)
            barrier_key = (resource, lifecycle)
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
                    lifecycle_epoch=lifecycle,
                    count=int(count),
                )
            elif name == "ttng.inval_barrier":
                if operation.get_num_operands() != 1:
                    raise CaptureError(
                        "TRITON_MBARRIER_INVALIDATE_UNRESOLVED",
                        "ttng.inval_barrier requires one barrier operand",
                        {"operation_index": index},
                    )
                self._emit(
                    "triton.mbarrier.invalidate",
                    index,
                    resource=resource,
                    lifecycle_epoch=lifecycle,
                )
                self.barrier_lifecycle[resource] = lifecycle + 1
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
                generation = self.arrival_rank.get(barrier_key, 0)
                self.arrival_rank[barrier_key] = generation + 1
                self._emit(
                    "triton.mbarrier.arrive",
                    index,
                    resource=resource,
                    lifecycle_epoch=lifecycle,
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
                generation = self.wait_rank.get(barrier_key, 0)
                if parity != generation % 2:
                    raise CaptureError(
                        "TRITON_PARITY_UNRESOLVED",
                        "wait phase SSA disagrees with the recovered gap-free stream rank",
                        {"resource": resource, "rank": generation, "parity": parity},
                    )
                self.wait_rank[barrier_key] = generation + 1
                self._emit(
                    "triton.mbarrier.wait_parity",
                    index,
                    resource=resource,
                    lifecycle_epoch=lifecycle,
                    generation=generation,
                    raw_parity=parity,
                    logical_stream_id=f"{self.current_actor}:{resource}",
                    stream_rank=generation,
                    polling_coalesced=True,
                )
        if len(self.output_operations) > before:
            self.emitting_nodes += 1
        else:
            self.elided_nodes += 1

    def capture(self) -> dict[str, Any]:
        self._record_function_arguments()
        self._record_definitions()
        if len(self.external_allocations) > 1:
            raise CaptureError(
                "TRITON_ALIAS_UNRESOLVED",
                "multiple external pointer bases require explicit alias evidence",
                {"external_bases": len(self.external_allocations)},
            )
        self._execute_block(self.entry_block, ("root",))
        if self.pending_proxy_fence is not None:
            raise CaptureError(
                "TRITON_PROXY_FENCE_UNRESOLVED",
                "fence_async_shared has no statically resolved async consumer",
                {"operation_index": self.pending_proxy_fence},
            )
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
                "participant_evidence": (
                    "ttgir-sha-bound-warp-partitions-plus-grid-cli-v1"
                    if len(self.actor_specs) > 1
                    else "ttgir-static-warp-geometry-plus-grid-cli-v1"
                ),
            },
            "actors": [self.actor_specs[key] for key in sorted(self.actor_specs)],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [self.regions[key] for key in sorted(self.regions)],
            "operations": self.output_operations,
            "unsupported_constructs": [],
        }


def capture_file(
    path: Path,
    source: str,
    command: str,
    grid_ctas: int,
    binding_evidence_path: Path | None = None,
) -> dict[str, Any]:
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
    source_digest = hashlib.sha256(source_ir).hexdigest()
    binding_evidence: dict[str, Any] = {}
    if binding_evidence_path is not None:
        try:
            evidence = json.loads(binding_evidence_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "Triton binding evidence could not be read",
                {"path": str(binding_evidence_path), "error": str(error)},
            ) from error
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {
                "evidence_version",
                "source_ir_sha256",
                "unit_attributes",
                "array_attributes",
            }
            or evidence.get("evidence_version") != "triton-binding-evidence-v1"
            or evidence.get("source_ir_sha256") != source_digest
            or not isinstance(evidence.get("unit_attributes"), dict)
            or not isinstance(evidence.get("array_attributes"), dict)
            or any(
                not str(index).isdigit()
                or not isinstance(attributes, list)
                or any(not isinstance(attribute, str) for attribute in attributes)
                for index, attributes in evidence.get("unit_attributes", {}).items()
            )
        ):
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "Triton binding evidence is malformed or bound to different IR",
                {"path": str(binding_evidence_path)},
            )
        binding_evidence = evidence
    try:
        return TritonCapture(
            triton,
            module,
            source,
            command,
            source_digest,
            grid_ctas,
            binding_evidence,
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
    parser.add_argument(
        "--binding-evidence",
        type=Path,
        help="SHA-bound unit/array attributes hidden by Triton Python bindings",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(
            args.input,
            args.source,
            " ".join(sys.argv),
            args.grid_ctas,
            args.binding_evidence,
        )
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
