#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
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


def _final_ptx_proxy_fence_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CaptureError(
            "TRITON_FINAL_PTX_EVIDENCE_INVALID",
            "final PTX evidence could not be read as UTF-8 text",
            {"path": str(path), "error": str(error)},
        ) from error
    fence_positions = [
        match.start()
        for match in re.finditer(r"\bfence\.proxy\.async\.shared::cta\s*;", text)
    ]
    wgmma_positions = [
        match.start()
        for match in re.finditer(r"\bwgmma\.mma_async(?:\.sync)?\.aligned\b", text)
    ]
    if len(fence_positions) > 1:
        raise CaptureError(
            "TRITON_FINAL_PTX_PROXY_FENCE_AMBIGUOUS",
            "multiple final-PTX proxy fences cannot be mapped to TTGIR consumers exactly",
            {"path": str(path), "proxy_fences": len(fence_positions)},
        )
    if fence_positions and (
        not wgmma_positions or fence_positions[0] >= wgmma_positions[0]
    ):
        raise CaptureError(
            "TRITON_FINAL_PTX_PROXY_FENCE_UNRESOLVED",
            "final-PTX proxy fence is not before the first WGMMA issue",
            {
                "path": str(path),
                "proxy_fences": len(fence_positions),
                "wgmma_issues": len(wgmma_positions),
            },
        )
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "proxy_fence_before_first_wgmma": bool(fence_positions),
        "proxy_fence_count": len(fence_positions),
        "wgmma_instruction_count": len(wgmma_positions),
        "method": "final-ptx-exact-opcode-order-v1",
    }


class TritonCapture:
    _DECLARATIONS = {
        "arith.constant",
        "tt.make_range",
        "tt.splat",
        "tt.addptr",
        "ttg.global_scratch_alloc",
        "ttg.local_alloc",
        "ttg.memdesc_index",
        "ttg.memdesc_subslice",
    }
    _CONTAINERS = {"builtin.module", "tt.func", "tt.return", "scf.yield"}
    _VALUE_ONLY_PURE_OPERATIONS = {
        "llvm.intr.assume",
        "tt.clampf",
        "tt.extern_elementwise",
        "tt.fp_to_fp",
        "tt.join",
        "tt.mulhiui",
        "ttg.fp4_to_fp",
        "tt.split",
        "tt.unsplat",
    }
    _PURE_OPERATIONS = {
        "arith.addi",
        "arith.andi",
        "arith.cmpi",
        "arith.divsi",
        "arith.divui",
        "arith.extsi",
        "arith.extui",
        "arith.index_cast",
        "arith.maxsi",
        "arith.minsi",
        "arith.muli",
        "arith.ori",
        "arith.remsi",
        "arith.select",
        "arith.subi",
        "arith.trunci",
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
        "ttg.local_store",
        "ttg.local_dealloc",
        "ttg.barrier",
        "tt.load",
        "tt.store",
        "ttng.async_tma_copy_global_to_local",
        "ttng.async_tma_copy_local_to_global",
        "ttng.async_tma_store_wait",
        "ttng.barrier_expect",
        "ttng.fence_async_shared",
        "ttng.warp_group_dot",
        "ttng.warp_group_dot_wait",
        "ttng.tensormap_create",
        "ttng.tensormap_fenceproxy_acquire",
        "ttng.reinterpret_tensor_descriptor",
    }

    @staticmethod
    def _block_assembly(block: Any) -> str:
        saved_stderr = os.dup(2)
        try:
            with tempfile.TemporaryFile() as output:
                os.dup2(output.fileno(), 2)
                block.dump()
                os.dup2(saved_stderr, 2)
                output.seek(0)
                return output.read().decode("utf-8", errors="replace")
        finally:
            try:
                os.dup2(saved_stderr, 2)
            finally:
                os.close(saved_stderr)

    def __init__(
        self,
        triton: Any,
        module: Any,
        source: str,
        command: str,
        source_ir_sha256: str,
        launch_grid: tuple[int, int, int],
        selected_cta: tuple[int, int, int],
        binding_evidence: dict[str, Any] | None = None,
        final_ptx_evidence: dict[str, Any] | None = None,
    ):
        self.triton = triton
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        self.launch_grid = launch_grid
        self.selected_cta = selected_cta
        self.binding_evidence = binding_evidence or {}
        self.final_ptx_evidence = final_ptx_evidence
        self.final_ptx_proxy_fence_pending = bool(
            final_ptx_evidence
            and final_ptx_evidence["proxy_fence_before_first_wgmma"]
        )
        self.unit_attributes = {
            int(index): set(attributes)
            for index, attributes in self.binding_evidence.get("unit_attributes", {}).items()
        }
        self.array_attributes = {
            int(index): attributes
            for index, attributes in self.binding_evidence.get("array_attributes", {}).items()
        }
        self.pointer_argument_allocations = {
            int(index): allocation
            for index, allocation in self.binding_evidence.get("pointer_allocations", {}).items()
        }
        self.scalar_argument_values = {
            int(index): value
            for index, value in self.binding_evidence.get("scalar_values", {}).items()
        }
        self.allocation_constant_values = {
            allocation: evidence["value"]
            for allocation, evidence in self.binding_evidence.get(
                "allocation_values", {}
            ).items()
            if evidence.get("kind") == "CONSTANT"
        }
        self.evidenced_external_allocations = set(self.pointer_argument_allocations.values())
        self.operations: list[Any] = []
        module.walk(self.operations.append)
        self.wgmma_accumulator_inputs = {
            operation.get_operand(2).id()
            for operation in self.operations
            if operation.get_name() == "ttng.warp_group_dot"
            and operation.get_num_operands() >= 3
        }
        invalid_operation_indices = sorted(
            index
            for index in {*self.unit_attributes, *self.array_attributes}
            if index < 0 or index >= len(self.operations)
        )
        if invalid_operation_indices:
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "binding evidence references an unknown TTGIR operation",
                {"indices": invalid_operation_indices, "operations": len(self.operations)},
            )
        functions = [operation for operation in self.operations if operation.get_name() == "tt.func"]
        public_functions = [
            function
            for function in functions
            if function.get_str_attr("sym_visibility") == "public"
        ]
        if len(public_functions) != 1:
            raise CaptureError(
                "TRITON_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one public TTGIR tt.func entry",
                {"functions": len(functions), "public_functions": len(public_functions)},
            )
        self.function = public_functions[0]
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
        function_blocks = sorted(self.region_blocks.get(function_region.id(), set()))
        self.cfg_successors: dict[int, tuple[int, ...]] = {}
        if len(function_blocks) == 1:
            self.entry_block = function_blocks[0]
        else:
            labels: dict[str, int] = {}
            successor_labels: dict[int, tuple[str, ...]] = {}
            for block_id in function_blocks:
                assembly = self._block_assembly(self.blocks[block_id])
                label = re.search(r"(?m)^\^(bb\d+)", assembly)
                if label is None or label.group(1) in labels:
                    raise CaptureError(
                        "TRITON_CONTROL_FLOW_UNRESOLVED",
                        "finite CFG block labels are unavailable or ambiguous",
                        {"blocks": len(function_blocks)},
                    )
                labels[label.group(1)] = block_id
                branch = next(
                    (
                        line.strip()
                        for line in assembly.splitlines()
                        if line.strip().startswith(("cf.br ", "cf.cond_br "))
                    ),
                    None,
                )
                successor_labels[block_id] = (
                    tuple(re.findall(r"\^(bb\d+)", branch)) if branch is not None else ()
                )
            if any(label not in labels for targets in successor_labels.values() for label in targets):
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "finite CFG references an unavailable successor block",
                )
            self.cfg_successors = {
                block_id: tuple(labels[label] for label in targets)
                for block_id, targets in successor_labels.items()
            }
            targets = {target for successors in self.cfg_successors.values() for target in successors}
            entries = [block_id for block_id in function_blocks if block_id not in targets]
            if len(entries) != 1:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "finite CFG must expose one entry block",
                    {"blocks": len(function_blocks), "entries": len(entries)},
                )
            self.entry_block = entries[0]
        self.function_entry_blocks = {
            self.function.get_str_attr("sym_name"): self.entry_block,
        }
        for function in functions:
            if function is self.function:
                continue
            symbol = function.get_str_attr("sym_name")
            blocks = sorted(self.region_blocks.get(function.get_region(0).id(), set()))
            if symbol is None or len(blocks) != 1:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "private TTGIR callees must have one statically expandable block",
                    {"symbol": symbol, "blocks": len(blocks)},
                )
            self.function_entry_blocks[symbol] = blocks[0]
        self.active_cfg_blocks: set[int] = set()
        self.active_function_calls: set[str] = set()
        entry = self.blocks[self.entry_block]
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
        evidenced_total_num_warps = self.num_warps
        for index, attributes in self.array_attributes.items():
            if (
                self.operations[index].get_name() != "ttg.warp_specialize"
                or not isinstance(attributes, dict)
                or set(attributes) != {"partition_num_warps", "warp_group_start_ids"}
            ):
                continue
            counts = attributes.get("partition_num_warps")
            starts = attributes.get("warp_group_start_ids")
            if (
                isinstance(counts, list)
                and isinstance(starts, list)
                and len(counts) == len(starts)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in (*counts, *starts))
            ):
                evidenced_total_num_warps = max(
                    evidenced_total_num_warps,
                    *(start + count for start, count in zip(starts, counts)),
                )
        self.total_num_warps = (
            int(total_num_warps)
            if total_num_warps is not None
            else evidenced_total_num_warps
        )
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
        self.wait_rank: dict[tuple[str, int, str], int] = {}
        self.barrier_lifecycle: dict[str, int] = {}
        self.collective_barrier_invocations: dict[str, int] = {}
        self.sequence = 0
        self.ranges: dict[int, tuple[int, int]] = {}
        self.scalar_pointer_arguments: dict[int, tuple[str, int]] = {}
        self.pointer_values: dict[int, tuple[str, int, tuple[int, ...]]] = {}
        self.pointer_regions: dict[int, str] = {}
        self.pointer_region_sets: dict[int, tuple[str, ...]] = {}
        self.memdesc_regions: dict[int, str] = {}
        self.descriptor_regions: dict[int, str] = {}
        self.global_scratch_allocations: dict[int, tuple[str, int]] = {}
        self.tensormap_sources: dict[int, tuple[str, int, tuple[int, ...]]] = {}
        self.fenced_tensormaps: set[int] = set()
        self.barrier_arguments: dict[int, str] = {}
        self.scalar_pointer_allocations: set[str] = set()
        self.external_allocations: set[str] = set()
        self.staged_tokens: set[int] = set()
        self.group_tokens: dict[int, int] = {}
        self.wait_tokens: set[int] = set()
        self.cp_group_seq = 0
        self.wgmma_group_sequences: dict[str, int] = {}
        self.tma_store_group_sequences: dict[str, int] = {}
        self.tma_expect_rank: dict[tuple[str, int], int] = {}
        self.tma_issue_generation: dict[tuple[str, int], int] = {}
        self.wgmma_results: dict[int, int] = {}
        self.pending_wgmma_owners: set[str] = set()
        self.register_regions: dict[int, str] = {}
        self.pending_proxy_fence: int | None = None
        self.pending_shared_write_allocations: set[str] = set()
        self.pending_shared_read_allocations: set[tuple[str, str]] = set()
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

    @staticmethod
    def _value_shape(value: Any) -> list[int]:
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
        return shape

    @staticmethod
    def _blocked_layout(type_text: str) -> tuple[list[int], list[int], list[int], list[int]] | None:
        match = re.search(r"#ttg\.blocked<\{([^}]*)\}>", type_text)
        if match is None:
            return None
        body = match.group(1)

        def array(name: str) -> list[int]:
            field = re.search(rf"\b{name}\s*=\s*\[([^]]*)\]", body)
            if field is None:
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "blocked layout omits a required distribution field",
                    {"field": name, "type": type_text},
                )
            try:
                values = [int(item.strip()) for item in field.group(1).split(",")]
            except ValueError as error:
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "blocked layout distribution is not integral",
                    {"field": name, "type": type_text},
                ) from error
            if not values or any(value <= 0 for value in values):
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "blocked layout distribution must contain positive dimensions",
                    {"field": name, "type": type_text},
                )
            return values

        size_per_thread = array("sizePerThread")
        threads_per_warp = array("threadsPerWarp")
        warps_per_cta = array("warpsPerCTA")
        order_match = re.search(r"\border\s*=\s*\[([^]]*)\]", body)
        if order_match is None:
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "blocked layout omits its dimension order",
                {"type": type_text},
            )
        try:
            order = [int(item.strip()) for item in order_match.group(1).split(",")]
        except ValueError as error:
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "blocked layout order is not integral",
                {"type": type_text},
            ) from error
        rank = len(size_per_thread)
        if (
            len(threads_per_warp) != rank
            or len(warps_per_cta) != rank
            or sorted(order) != list(range(rank))
        ):
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "blocked layout dimensions or order are inconsistent",
                {"type": type_text},
            )

        slice_match = re.search(r"#ttg\.slice<\{\s*dim\s*=\s*(\d+)", type_text)
        if slice_match is not None:
            dimension = int(slice_match.group(1))
            if dimension >= rank:
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "slice layout removes an out-of-range blocked-layout dimension",
                    {"dimension": dimension, "rank": rank, "type": type_text},
                )
            removed_distribution = (
                size_per_thread[dimension],
                threads_per_warp[dimension],
                warps_per_cta[dimension],
            )
            if removed_distribution != (1, 1, 1):
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "slice layout removes a distributed dimension requiring replication evidence",
                    {
                        "dimension": dimension,
                        "distribution": list(removed_distribution),
                        "type": type_text,
                    },
                )
            size_per_thread.pop(dimension)
            threads_per_warp.pop(dimension)
            warps_per_cta.pop(dimension)
            order = [
                item - 1 if item > dimension else item
                for item in order
                if item != dimension
            ]
        return size_per_thread, threads_per_warp, warps_per_cta, order

    @staticmethod
    def _shared_order(type_text: str, rank: int) -> list[int]:
        """Return a shared encoding's dimension order, fastest-varying first."""
        row_major = list(range(rank - 1, -1, -1))
        nvmma = re.search(r"#ttg\.nvmma_shared<\{([^}]*)\}>", type_text)
        if nvmma is not None:
            if rank >= 2 and re.search(r"\btransposed\s*=\s*true\b", nvmma.group(1)):
                return [rank - 2, rank - 1, *range(rank - 3, -1, -1)]
            return row_major
        swizzled = re.search(r"#ttg\.swizzled_shared<\{([^}]*)\}>", type_text)
        if swizzled is not None:
            field = re.search(r"\border\s*=\s*\[([^]]*)\]", swizzled.group(1))
            try:
                order = [int(item.strip()) for item in field.group(1).split(",")] if field else None
            except ValueError:
                order = None
            if order is None or sorted(order) != list(range(rank)):
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "swizzled shared order is not a permutation of the memdesc dimensions",
                    {"type": type_text},
                )
            return order
        return row_major

    @classmethod
    def _memdesc_positions(cls, memdesc: Any) -> list[int] | None:
        """Map row-major memdesc elements to their element slot in the region.

        Shared memory is modelled as the encoding's logical linearization
        (fastest dimension innermost, swizzling ignored); None means the
        order is row-major, so element indices are the slots.
        """
        shape = cls._value_shape(memdesc)
        order = cls._shared_order(str(memdesc.get_type()), len(shape))
        if order == list(range(len(shape) - 1, -1, -1)):
            return None
        strides = [0] * len(shape)
        stride = 1
        for dimension in order:
            strides[dimension] = stride
            stride *= shape[dimension]
        positions = [0]
        for extent, dimension_stride in zip(shape, strides):
            positions = [
                position + coordinate * dimension_stride
                for position in positions
                for coordinate in range(extent)
            ]
        return positions

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and value & (value - 1) == 0

    @classmethod
    def _blocked_lane_elements(
        cls,
        type_text: str,
        shape: list[int],
        participants: list[int],
        *,
        representative_only: bool,
    ) -> dict[int, tuple[int, ...]]:
        layout = cls._blocked_layout(type_text)
        if layout is None:
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "tensor memory access requires an explicit blocked layout",
                {"type": type_text},
            )
        size_per_thread, threads_per_warp, warps_per_cta, order = layout
        if (
            len(shape) != len(size_per_thread)
            or any(not cls._is_power_of_two(value) for value in shape)
            or any(not cls._is_power_of_two(value) for value in (*size_per_thread, *threads_per_warp, *warps_per_cta))
        ):
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "the supported exact blocked-layout inversion requires power-of-two static dimensions",
                {"type": type_text, "shape": shape},
            )

        def product_of(values: list[int]) -> int:
            result = 1
            for value in values:
                result *= value
            return result

        if product_of(threads_per_warp) != 32:
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "blocked layout threadsPerWarp does not match SM90 warp size",
                {"threads_per_warp": threads_per_warp},
            )
        expected_threads = 32 * product_of(warps_per_cta)
        if len(participants) != expected_threads:
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "blocked layout warp count does not match the active execution owner",
                {
                    "participants": len(participants),
                    "layout_threads": expected_threads,
                },
            )

        scales = [1] * len(shape)

        def bases(extents: list[int]) -> list[tuple[int, ...]]:
            result: list[tuple[int, ...]] = []
            for dimension in order:
                bits = extents[dimension].bit_length() - 1
                for _ in range(bits):
                    basis = [0] * len(shape)
                    basis[dimension] = scales[dimension]
                    scales[dimension] *= 2
                    result.append(tuple(basis))
            return result

        register_bases = bases(size_per_thread)
        lane_bases = bases(threads_per_warp)
        warp_bases = bases(warps_per_cta)
        for dimension, desired in enumerate(shape):
            actual = scales[dimension]
            if actual < desired:
                if desired % actual or not cls._is_power_of_two(desired // actual):
                    raise CaptureError(
                        "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                        "blocked layout cannot be extended exactly to the tensor shape",
                        {"dimension": dimension, "actual": actual, "desired": desired},
                    )
                while actual < desired:
                    basis = [0] * len(shape)
                    basis[dimension] = actual
                    register_bases.append(tuple(basis))
                    actual *= 2
                    scales[dimension] = actual

        def truncate(items: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
            return [
                tuple(value if value < shape[dimension] else 0 for dimension, value in enumerate(basis))
                for basis in items
            ]

        register_bases = truncate(register_bases)
        lane_bases = truncate(lane_bases)
        warp_bases = truncate(warp_bases)
        unique_register_bases = [basis for basis in register_bases if any(basis)]
        lane_free_mask = sum(
            1 << bit for bit, basis in enumerate(lane_bases) if not any(basis)
        )
        warp_free_mask = sum(
            1 << bit for bit, basis in enumerate(warp_bases) if not any(basis)
        )

        def apply(input_value: int, input_bases: list[tuple[int, ...]]) -> list[int]:
            coordinates = [0] * len(shape)
            for bit, basis in enumerate(input_bases):
                if input_value & (1 << bit):
                    coordinates = [
                        coordinate ^ contribution
                        for coordinate, contribution in zip(coordinates, basis)
                    ]
            return coordinates

        result: dict[int, tuple[int, ...]] = {}
        register_count = 1 << len(unique_register_bases)
        for local_thread, physical_lane in enumerate(participants):
            lane = local_thread % 32
            warp = local_thread // 32
            if representative_only and (lane & lane_free_mask or warp & warp_free_mask):
                continue
            lane_coordinates = apply(lane, lane_bases)
            warp_coordinates = apply(warp, warp_bases)
            elements: list[int] = []
            for register in range(register_count):
                register_coordinates = apply(register, unique_register_bases)
                coordinates = [
                    register_coordinate ^ lane_coordinate ^ warp_coordinate
                    for register_coordinate, lane_coordinate, warp_coordinate in zip(
                        register_coordinates,
                        lane_coordinates,
                        warp_coordinates,
                    )
                ]
                if any(coordinate >= extent for coordinate, extent in zip(coordinates, shape)):
                    raise CaptureError(
                        "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                        "blocked layout inversion produced an out-of-range tensor coordinate",
                        {"type": type_text, "coordinates": coordinates, "shape": shape},
                    )
                flat = 0
                for coordinate, extent in zip(coordinates, shape):
                    flat = flat * extent + coordinate
                if flat not in elements:
                    elements.append(flat)
            result[physical_lane] = tuple(elements)
        return result

    @staticmethod
    def _broadcast_values(
        values: tuple[int, ...],
        source_shape: list[int],
        result_shape: list[int],
    ) -> tuple[int, ...]:
        if not source_shape or not result_shape or len(source_shape) > len(result_shape):
            raise ValueError("broadcast requires ranked tensor operands")
        padded_source = [1] * (len(result_shape) - len(source_shape)) + source_shape
        if any(source not in {1, result} for source, result in zip(padded_source, result_shape)):
            raise ValueError("broadcast shapes are incompatible")
        source_elements = 1
        for dimension in source_shape:
            source_elements *= dimension
        if len(values) != source_elements:
            raise ValueError("broadcast source value count does not match its shape")
        strides: list[int] = []
        stride = 1
        for dimension in reversed(padded_source):
            strides.append(stride)
            stride *= dimension
        strides.reverse()
        broadcasted = []
        for result_index in product(*(range(dimension) for dimension in result_shape)):
            source_offset = sum(
                (0 if dimension == 1 else item) * source_stride
                for item, dimension, source_stride in zip(result_index, padded_source, strides)
            )
            broadcasted.append(values[source_offset])
        return tuple(broadcasted)

    def _value_extent(self, value: Any) -> tuple[int, int]:
        type_text = str(value.get_type())
        shape = self._value_shape(value)
        match = re.search(r"(?:^|[<!x])(?:i|ui|f|bf)(\d+)", type_text)
        if not shape or match is None:
            raise CaptureError(
                "TRITON_REGION_UNRESOLVED",
                "SSA value element type or shape is unresolved",
                {"type": str(value.get_type())},
            )
        bits = int(match.group(1))
        if bits == 1 and re.search(r"!tt\.ptr<i1\b", type_text):
            # Booleans occupy one byte in memory.
            bits = 8
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

    @staticmethod
    def _scalar_pointer_width(type_text: str) -> int | None:
        pointer = re.fullmatch(r"!tt\.ptr<([^,>]+)(?:,\s*\d+)?>", type_text)
        if pointer is None:
            return None
        element = re.match(r"^(?:i|ui|f|bf)(\d+)", pointer.group(1))
        if element is None:
            return None
        bits = int(element.group(1))
        return bits // 8 if bits > 0 and bits % 8 == 0 else None

    @staticmethod
    def _integer_width(type_text: str) -> int | None:
        match = re.search(r"(?:^|x)(?:i|ui)(\d+)(?:$|[,>])", type_text)
        return int(match.group(1)) if match is not None else None

    @staticmethod
    def _unsigned_integer(value: int, bits: int) -> int:
        return value & ((1 << bits) - 1)

    @classmethod
    def _signed_integer(cls, value: int, bits: int) -> int:
        value = cls._unsigned_integer(value, bits)
        sign = 1 << (bits - 1)
        return value - (1 << bits) if value & sign else value

    def _ensure_pointer_value(
        self,
        value_id: int,
    ) -> tuple[str, int, tuple[int, ...]] | None:
        pointer = self.pointer_values.get(value_id)
        if pointer is not None:
            return pointer
        argument = self.scalar_pointer_arguments.get(value_id)
        if argument is None:
            return None
        allocation, width = argument
        self._set_pointer_value(value_id, allocation, width, (0,))
        return self.pointer_values[value_id]

    def _mark_external_pointer(
        self,
        value_id: int,
    ) -> tuple[str, int, tuple[int, ...]] | None:
        pointer = self._ensure_pointer_value(value_id)
        if pointer is not None and pointer[0] in self.scalar_pointer_allocations:
            self.external_allocations.add(pointer[0])
        return pointer

    def _require_resolved_external_aliases(self) -> None:
        if (
            len(self.external_allocations) > 1
            and not self.external_allocations <= self.evidenced_external_allocations
        ):
            raise CaptureError(
                "TRITON_ALIAS_UNRESOLVED",
                "multiple external pointer bases require explicit alias evidence",
                {"external_bases": len(self.external_allocations)},
            )

    def _record_function_arguments(self) -> None:
        block = self.blocks[self.entry_block]
        argument_count = block.get_num_arguments()
        invalid_scalar_indices = sorted(
            index for index in self.scalar_argument_values
            if index < 0 or index >= argument_count
        )
        invalid_pointer_indices = sorted(
            index for index in self.pointer_argument_allocations
            if (
                index < 0
                or index >= argument_count
                or not str(block.get_argument(index).get_type()).startswith("!tt.ptr<")
            )
        )
        if invalid_scalar_indices or invalid_pointer_indices:
            raise CaptureError(
                "TRITON_BINDING_EVIDENCE_INVALID",
                "argument evidence references an unknown or incompatible TTGIR argument",
                {
                    "scalar_indices": invalid_scalar_indices,
                    "pointer_indices": invalid_pointer_indices,
                    "arguments": argument_count,
                },
            )
        for index in range(argument_count):
            argument = block.get_argument(index)
            type_text = str(argument.get_type())
            if index in self.scalar_argument_values:
                value = self.scalar_argument_values[index]
                if (
                    not re.fullmatch(r"(?:i|ui)\d+|index", type_text)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                ):
                    raise CaptureError(
                        "TRITON_BINDING_EVIDENCE_INVALID",
                        "scalar argument evidence requires an integer TTGIR scalar",
                        {"index": index, "type": type_text, "value": value},
                    )
                bits = self._integer_width(type_text)
                self.static_values[argument.id()] = (
                    self._signed_integer(value, bits) if bits is not None else value
                )
            if type_text.startswith("!tt.ptr<"):
                width = self._scalar_pointer_width(type_text)
                if width is None and re.fullmatch(r"!tt\.ptr<i1(?:,\s*\d+)?>", type_text):
                    # Booleans occupy one byte in memory; accesses must go
                    # through an i8 bitcast (see _capture_memory).
                    width = 1
                if width is not None:
                    allocation = self.pointer_argument_allocations.get(
                        index,
                        f"global-arg:{index}",
                    )
                    self.scalar_pointer_arguments[argument.id()] = (
                        allocation,
                        width,
                    )
                    self.scalar_pointer_allocations.add(allocation)
            elif "!ttg.memdesc<" in type_text:
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

    def _regions_for_element_offsets(
        self,
        allocation: str,
        width: int,
        offsets: list[int],
    ) -> tuple[str, ...]:
        intervals = sorted({(offset * width, offset * width + width) for offset in offsets})
        merged: list[list[int]] = []
        for start, end in intervals:
            if start < 0:
                raise CaptureError(
                    "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                    "pointer offset is negative relative to its external base",
                    {"offset": start // width, "width": width},
                )
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return tuple(
            self._region(allocation, "GLOBAL", start, end - start)
            for start, end in merged
        )

    def _lane_memory_regions(
        self,
        operation: Any,
        pointer_id: int,
        mask: Any,
        *,
        is_store: bool,
    ) -> dict[int, tuple[str, ...]]:
        pointer = self.pointer_values.get(pointer_id)
        if pointer is None:
            return {}
        allocation, width, offsets = pointer
        participants = self.actor_specs[self.current_actor]["participants"]
        if participants != list(range(len(participants))):
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "nonzero warp-partition lane remapping requires explicit lowered thread-id evidence",
                {"actor": self.current_actor, "participants": participants},
            )
        type_text = str(operation.get_operand(0).get_type())
        shape = self._value_shape(operation.get_operand(0))
        if shape:
            elements = 1
            for extent in shape:
                elements *= extent
            if elements != len(offsets):
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "pointer tensor values do not match its static shape",
                    {
                        "operation": operation.get_name(),
                        "pointer_elements": len(offsets),
                        "shape": shape,
                    },
                )
            lane_elements = self._blocked_lane_elements(
                type_text,
                shape,
                participants,
                representative_only=is_store,
            )
        else:
            active = participants[:1] if is_store else participants
            lane_elements = {lane: (0,) for lane in active}

        if isinstance(mask, (bool, int)):
            predicates = (bool(mask),) * len(offsets)
        elif isinstance(mask, (tuple, list)) and len(mask) == len(offsets):
            predicates = tuple(bool(value) for value in mask)
        else:
            raise CaptureError(
                "TRITON_PREDICATE_UNRESOLVED",
                "memory mask shape does not match the pointer tensor",
                {"operation": operation.get_name(), "pointer_elements": len(offsets)},
            )

        result: dict[int, tuple[str, ...]] = {}
        for lane, element_indices in sorted(lane_elements.items()):
            lane_offsets = [
                offsets[element]
                for element in element_indices
                if predicates[element]
            ]
            if lane_offsets:
                result[lane] = self._regions_for_element_offsets(
                    allocation,
                    width,
                    lane_offsets,
                )
        return result

    def _lane_memdesc_regions(
        self,
        value: Any,
        memdesc: Any,
        region_id: str,
        *,
        representative_only: bool,
    ) -> dict[int, tuple[str, ...]]:
        region = self.regions[region_id]
        shape = self._value_shape(value)
        elements, width = self._value_extent(value)
        positions = self._memdesc_positions(memdesc)
        if elements * width != region["extent"] or (
            positions is not None and shape != self._value_shape(memdesc)
        ):
            raise CaptureError(
                "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                "distributed tensor and memdesc shapes or byte extents differ",
                {"type": str(value.get_type()), "region": region_id},
            )
        participants = self.actor_specs[self.current_actor]["participants"]
        lane_elements = self._blocked_lane_elements(
            str(value.get_type()),
            shape,
            participants,
            representative_only=representative_only,
        )
        result: dict[int, tuple[str, ...]] = {}
        for lane, indices in lane_elements.items():
            slots = indices if positions is None else [positions[element] for element in indices]
            intervals: list[list[int]] = []
            for slot in sorted(slots):
                offset = region["offset"] + slot * width
                if intervals and offset == intervals[-1][0] + intervals[-1][1]:
                    intervals[-1][1] += width
                else:
                    intervals.append([offset, width])
            if intervals:
                result[lane] = tuple(
                    self._region(
                        region["allocation_id"],
                        region["address_space"],
                        offset,
                        extent,
                    )
                    for offset, extent in intervals
                )
        return result

    def _cp_async_lane_effects(
        self,
        operation: Any,
        pointer_id: int,
        destination: str,
        predicates: tuple[bool, ...],
    ) -> dict[str, Any]:
        pointer = self.pointer_values.get(pointer_id)
        if pointer is None:
            raise CaptureError(
                "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED",
                "cp.async source pointer values are unresolved",
            )
        allocation, width, offsets = pointer
        if width <= 0:
            raise CaptureError(
                "TRITON_CP_ASYNC_TRANSFER_WIDTH_UNSUPPORTED",
                "cp.async element width is not a positive byte count",
                {"width": width},
            )
        shape = self._value_shape(operation.get_operand(0))
        elements = 1
        for extent in shape:
            elements *= extent
        if not shape or elements != len(offsets) or len(predicates) != len(offsets):
            raise CaptureError(
                "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED",
                "cp.async pointer shape, offsets, and mask must agree exactly",
                {
                    "shape": shape,
                    "pointer_elements": len(offsets),
                    "mask_elements": len(predicates),
                },
            )
        participants = self.actor_specs[self.current_actor]["participants"]
        lane_elements = self._blocked_lane_elements(
            str(operation.get_operand(0).get_type()),
            shape,
            participants,
            representative_only=True,
        )
        destination_region = self.regions[destination]
        positions = self._memdesc_positions(operation.get_operand(1))
        if positions is not None and shape != self._value_shape(operation.get_operand(1)):
            raise CaptureError(
                "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED",
                "cp.async source and destination memdesc shapes differ",
                {"shape": shape, "destination": str(operation.get_operand(1).get_type())},
            )
        slots = range(len(offsets)) if positions is None else positions
        lanes: list[int] = []
        transfer_offsets = [0]
        source_regions: list[str] = []
        destination_regions: list[str] = []
        byte_counts: list[int] = []
        for lane, element_indices in sorted(lane_elements.items()):
            selected = [
                element for element in element_indices if predicates[element]
            ]
            if not selected:
                continue
            lanes.append(lane)
            # Triton vectorizes a thread's elements that are contiguous in
            # both global and shared memory into one cp.async; each
            # instruction moves an aligned 4, 8, or 16 bytes.
            runs: list[list[int]] = []
            for element in sorted(selected, key=lambda item: slots[item]):
                if (
                    runs
                    and slots[element] == slots[runs[-1][-1]] + 1
                    and offsets[element] == offsets[runs[-1][-1]] + 1
                ):
                    runs[-1].append(element)
                else:
                    runs.append([element])
            for run in runs:
                source_offset = offsets[run[0]] * width
                if source_offset < 0:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "cp.async source offset is negative relative to its allocation",
                        {"element": run[0], "offset": source_offset},
                    )
                destination_offset = destination_region["offset"] + slots[run[0]] * width
                remaining = len(run) * width
                while remaining:
                    size = next(
                        (
                            candidate
                            for candidate in (16, 8, 4)
                            if candidate <= remaining
                            and source_offset % candidate == 0
                            and destination_offset % candidate == 0
                        ),
                        None,
                    )
                    if size is None:
                        raise CaptureError(
                            "TRITON_CP_ASYNC_TRANSFER_WIDTH_UNSUPPORTED",
                            "SM90 cp.async requires aligned per-thread transfers of 4, 8, or 16 bytes",
                            {
                                "width": width,
                                "run_bytes": len(run) * width,
                                "remaining_bytes": remaining,
                                "source_offset": source_offset,
                                "destination_offset": destination_offset,
                            },
                        )
                    source_regions.append(
                        self._region(allocation, "GLOBAL", source_offset, size)
                    )
                    destination_regions.append(self._region(
                        destination_region["allocation_id"],
                        destination_region["address_space"],
                        destination_offset,
                        size,
                    ))
                    byte_counts.append(size)
                    source_offset += size
                    destination_offset += size
                    remaining -= size
            transfer_offsets.append(len(source_regions))
        selected_slots = [
            slots[element] for element, predicate in enumerate(predicates) if predicate
        ]
        return {
            "lanes": lanes,
            "transfer_offsets": transfer_offsets,
            "source_regions": source_regions,
            "destination_regions": destination_regions,
            "bytes": byte_counts,
            "destination_slot_span": (
                (min(selected_slots), max(selected_slots)) if selected_slots else None
            ),
        }

    def _resolve_addptr(self, operation: Any, index: int) -> bool:
        if operation.get_num_operands() != 2 or operation.get_num_results() != 1:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "tt.addptr provenance is malformed",
                {"operation_index": index},
            )
        base = self._mark_external_pointer(operation.get_operand(0).id())
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

    def _emit_shared_membar(self, index: int) -> None:
        participants = self.actor_specs[self.current_actor]["participants"]
        complete_cta = participants == list(range(self.cta_threads))
        kind = "cta_barrier" if complete_cta else "named_barrier"
        identifier = f"{kind}:{self.current_actor}:membar"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": kind,
            "participants": [self.current_actor],
            "capacity": len(participants),
            "scope": "CTA",
            "visibility_mask": {
                "shared": True,
                "global_read": False,
                "global_write": False,
                "tensor_read": False,
                "tensor_write": False,
                "control_only": False,
            },
        })
        invocation = self.collective_barrier_invocations.get(identifier, 0)
        self.collective_barrier_invocations[identifier] = invocation + 1
        self._emit(
            "triton.barrier.arrive",
            index,
            resource=identifier,
            invocation_id=invocation,
            count=len(participants),
        )
        self._emit(
            "triton.barrier.observe",
            index,
            resource=identifier,
            invocation_id=invocation,
        )
        self.pending_shared_write_allocations.clear()
        self.pending_shared_read_allocations.clear()

    def _membar_before_shared_write(self, allocation: str, index: int) -> None:
        # Triton's Membar pass also orders a shared write after an
        # overlapping earlier read by the same threads (WAR).
        if (self.current_actor, allocation) in self.pending_shared_read_allocations:
            self._emit_shared_membar(index)

    def _capture_predicated_bar_sync_inline_asm(
        self,
        operation: Any,
        index: int,
    ) -> None:
        assembly = operation.get_str_attr("asm_string")
        constraints = operation.get_str_attr("constraints")
        packed = operation.get_int_attr("packed_element")
        pattern = re.fullmatch(
            r"\s*\.reg\s+\.pred\s+p\s*;\s*"
            r"setp\.lt\.u32\s+p\s*,\s*\$1\s*,\s*(\d+)\s*;\s*"
            r"@p\s+bar\.sync\s+0\s*;\s*"
            r"mov\.u32\s+\$0\s*,\s*\$1\s*;\s*",
            assembly or "",
        )
        if (
            operation.get_bool_attr("pure") is not False
            or constraints != "=r,r"
            or packed != 1
            or operation.get_num_operands() != 1
            or operation.get_num_results() != 1
            or pattern is None
        ):
            raise CaptureError(
                "TRITON_EFFECTFUL_INLINE_ASM_UNSUPPORTED",
                "effectful inline assembly requires an exact semantic capture rule",
                {"operation_index": index},
            )
        threshold = int(pattern.group(1))
        operand = operation.get_operand(0)
        lanes = self.static_values.get(operand.id())
        if (
            not isinstance(lanes, (tuple, list))
            or tuple(lanes) != tuple(range(self.cta_threads))
            or not 0 < threshold < self.cta_threads
            or str(operation.get_result(0).get_type()) != str(operand.get_type())
        ):
            raise CaptureError(
                "TRITON_INLINE_BARRIER_PARTICIPANTS_UNRESOLVED",
                "predicated bar.sync requires the exact static CTA lane range",
                {
                    "operation_index": index,
                    "threshold": threshold,
                    "cta_threads": self.cta_threads,
                },
            )
        active = list(range(threshold))
        identifier = "cta_barrier:inline_asm:0"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "cta_barrier",
            "participants": [self.current_actor],
            "capacity": self.cta_threads,
            "scope": "CTA",
            "visibility_mask": {
                "shared": True,
                "global_read": True,
                "global_write": True,
                "tensor_read": False,
                "tensor_write": False,
                "control_only": False,
            },
        })
        invocation = self.collective_barrier_invocations.get(identifier, 0)
        self.collective_barrier_invocations[identifier] = invocation + 1
        execution_evidence = {
            "program_instances": active,
            "executors": active,
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "method": "ttgir-effectful-inline-asm-predicated-bar-sync-v1",
        }
        self._emit(
            "triton.barrier.arrive",
            index,
            resource=identifier,
            invocation_id=invocation,
            count=1,
            execution_evidence=execution_evidence,
        )
        self._emit(
            "triton.barrier.observe",
            index,
            resource=identifier,
            invocation_id=invocation,
            execution_evidence=execution_evidence,
        )
        self.static_values[operation.get_result(0).id()] = tuple(lanes)
        self.pending_shared_write_allocations.clear()
        self.pending_shared_read_allocations.clear()


    def _capture_collective_barrier(self, operation: Any, index: int) -> None:
        if (
            operation.get_num_operands() != 0
            or operation.get_num_results() != 0
            or operation.get_str_attr("scope") not in {None, "local"}
        ):
            raise CaptureError(
                "TRITON_COLLECTIVE_BARRIER_UNRESOLVED",
                "ttg.barrier must be an operand-free local barrier",
                {"operation_index": index},
            )
        participants = self.actor_specs[self.current_actor]["participants"]
        complete_cta = participants == list(range(self.cta_threads))
        kind = "cta_barrier" if complete_cta else "named_barrier"
        identifier = f"{kind}:{self.current_actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": kind,
            "participants": [self.current_actor],
            "capacity": len(participants),
            "scope": "CTA",
            "visibility_mask": {
                "shared": True,
                "global_read": True,
                "global_write": True,
                "tensor_read": False,
                "tensor_write": False,
                "control_only": False,
            },
        })
        invocation = self.collective_barrier_invocations.get(identifier, 0)
        self.collective_barrier_invocations[identifier] = invocation + 1
        self._emit(
            "triton.barrier.arrive",
            index,
            resource=identifier,
            invocation_id=invocation,
            count=len(participants),
        )
        self._emit(
            "triton.barrier.observe",
            index,
            resource=identifier,
            invocation_id=invocation,
        )
        self.pending_shared_write_allocations.clear()
        self.pending_shared_read_allocations.clear()

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

    def _async_engine_actor(self, protocol: str) -> str:
        identifier = f"engine:{protocol.lower()}:{self.current_actor}"
        participants = self.actor_specs[self.current_actor]["participants"]
        self.actor_specs.setdefault(identifier, {
            "id": identifier,
            "kind": "ASYNC_ENGINE",
            "participants": [participants[0]],
        })
        return identifier

    def _wgmma_resource(self, operation: Any | None = None, index: int | None = None) -> str:
        identifier = f"wgmma:{self.current_actor}"
        if identifier in self.resources:
            return identifier
        if operation is None:
            raise CaptureError(
                "TRITON_WGMMA_PARTICIPANTS_UNRESOLVED",
                "WGMMA resource use precedes its statically evidenced issue",
                {"actor": self.current_actor, "operation_index": index},
            )
        actor = self.actor_specs[self.current_actor]
        participants = actor["participants"]
        aligned = (
            participants
            and participants == list(range(participants[0], participants[0] + len(participants)))
            and participants[0] % 128 == 0
        )
        complete_warpgroups = len(participants) % 128 == 0
        supported_owner = (
            actor["kind"] == "WARPGROUP" and len(participants) == 128
        ) or (
            actor["kind"] == "PARTITION" and len(participants) >= 128
        )
        encoded_warps = None
        values = [
            operation.get_operand(position)
            for position in range(operation.get_num_operands())
        ] + [
            operation.get_result(position)
            for position in range(operation.get_num_results())
        ]
        for value in values:
            match = re.search(r"warpsPerCTA\s*=\s*\[([^]]+)\]", str(value.get_type()))
            if match is None:
                continue
            dimensions = [int(item.strip()) for item in match.group(1).split(",")]
            encoded_warps = 1
            for dimension in dimensions:
                encoded_warps *= dimension
            break
        if (
            not aligned
            or not complete_warpgroups
            or not supported_owner
            or (len(participants) > 128 and encoded_warps != len(participants) // 32)
        ):
            raise CaptureError(
                "TRITON_WGMMA_PARTICIPANTS_UNRESOLVED",
                "warp_group_dot owner must match complete aligned warpgroups in its MMA layout",
                {
                    "actor": self.current_actor,
                    "participants": participants,
                    "encoded_warps": encoded_warps,
                    "operation_index": index,
                },
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
                    result = operation.get_result(0)
                    value = operation.get_constant_value()
                    shape = self._value_shape(result)
                    if shape and value is not None and not isinstance(value, (tuple, list)):
                        elements = 1
                        for dimension in shape:
                            elements *= dimension
                        value = tuple(value for _ in range(elements))
                    self.constants[result.id()] = value
                    self.static_values[result.id()] = value
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
                base = self._mark_external_pointer(operand.id())
                if base is None:
                    continue
                allocation, width, offsets = base
                if len(offsets) != 1:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.splat pointer operand is not scalar",
                        {"operation_index": index},
                    )
                elements, result_width = self._value_extent(operation.get_result(0))
                if result_width != width:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.splat changes pointer element width",
                        {
                            "operation_index": index,
                            "source_width": width,
                            "result_width": result_width,
                        },
                    )
                self._set_pointer_value(
                    operation.get_result(0).id(),
                    allocation,
                    width,
                    offsets * elements,
                )
            elif name == "tt.addptr":
                self._resolve_addptr(operation, index)
            elif name == "ttg.global_scratch_alloc":
                if operation.get_num_operands() != 0 or operation.get_num_results() != 1:
                    raise CaptureError(
                        "TRITON_GLOBAL_SCRATCH_UNRESOLVED",
                        "global scratch allocation must have no operands and one result",
                        {"operation_index": index},
                    )
                result = operation.get_result(0)
                width = self._scalar_pointer_width(str(result.get_type()))
                nbytes = operation.get_int_attr("nbytes")
                alignment = operation.get_int_attr("alignment")
                if (
                    width is None
                    or nbytes is None
                    or int(nbytes) <= 0
                    or alignment is None
                    or int(alignment) <= 0
                ):
                    raise CaptureError(
                        "TRITON_GLOBAL_SCRATCH_UNRESOLVED",
                        "global scratch allocation lacks a finite size or alignment",
                        {"operation_index": index},
                    )
                allocation = f"global-scratch:ssa:{index}:0"
                self.global_scratch_allocations[result.id()] = (allocation, int(nbytes))
                self._set_pointer_value(result.id(), allocation, width, (0,))
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
        else:
            result_type = str(value.get_type())
            region_id = self.memdesc_regions.get(value.id())
            if region_id is None:
                definition = self.result_definitions.get(value.id())
                raise CaptureError(
                    "TRITON_BARRIER_IDENTITY_UNRESOLVED",
                    "mbarrier SSA value has no exact shared-memory region",
                    {
                        "defining_operation": (
                            self.operations[definition[0]].get_name()
                            if definition is not None
                            else None
                        ),
                    },
                )
            region = self.regions[region_id]
            if (
                "!ttg.memdesc<" not in result_type
                or "xi64" not in result_type
                or region["address_space"] != "SHARED"
                or region["extent"] != 8
            ):
                raise CaptureError(
                    "TRITON_BARRIER_TYPE_UNSUPPORTED",
                    "mbarrier value is not one exact shared i64 element",
                    {"type": result_type, "region": region_id},
                )
            identifier = f"mbarrier:{region_id}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "mbarrier",
            "participants": [self.current_actor],
        })
        if self.current_actor not in self.resources[identifier]["participants"]:
            self.resources[identifier]["participants"].append(self.current_actor)
        return identifier

    def _protocol_execution_evidence(self, op: str) -> dict[str, Any]:
        participants = list(self.actor_specs[self.current_actor]["participants"])
        if op.startswith("triton.wgmma.") or op.startswith("triton.barrier."):
            executors = participants
            transition = "ONCE"
            rendezvous = "ALL_PROGRAM_INSTANCES"
        elif op == "triton.mbarrier.wait_parity":
            executors = participants[:1]
            transition = "ONCE"
            rendezvous = "ALL_PROGRAM_INSTANCES"
        elif op.startswith("triton.cp_async."):
            executors = participants
            transition = "PER_INSTANCE"
            rendezvous = "NONE"
        elif op == "triton.proxy_fence":
            executors = participants
            transition = "ONCE"
            rendezvous = "ALL_PROGRAM_INSTANCES"
        else:
            executors = participants[:1]
            transition = "ONCE"
            rendezvous = "NONE"
        return {
            "program_instances": participants,
            "executors": executors,
            "transition": transition,
            "rendezvous": rendezvous,
            "method": (
                "ttgir-final-nvvm-cp-async-per-thread-v1"
                if op.startswith("triton.cp_async.")
                else f"ttgir-{op}-lowering-contract-v1"
            ),
        }

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
        if (
            "execution_evidence" not in fields
            and op not in {"triton.mem.read", "triton.mem.write"}
        ):
            fields["execution_evidence"] = self._protocol_execution_evidence(op)
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
                "TMA memory footprints require SHA-bound descriptor shape and stride evidence",
                {"operation_index": index, "coordinates": coordinates},
            )
        allocation = evidence["allocation_id"]
        if descriptor_region is not None:
            known_allocation = self.regions[descriptor_region]["allocation_id"]
            if (
                not known_allocation.startswith("tensordesc-arg:")
                and allocation != known_allocation
            ):
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_INVALID",
                    "TMA descriptor evidence contradicts its source pointer allocation",
                    {
                        "operation_index": index,
                        "evidenced_allocation": allocation,
                        "source_allocation": known_allocation,
                    },
                )
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
        elements = 1
        for dimension in tile_shape:
            elements *= dimension
        if elements * element_bytes != counterpart_region["extent"]:
            raise CaptureError(
                "TRITON_TMA_DESCRIPTOR_UNRESOLVED",
                "TMA descriptor tile and shared counterpart extents differ",
                {"operation_index": index},
            )
        runs: list[list[Any]] = []
        counterpart_offset = counterpart_region["offset"]
        zero_fill_allocation = (
            f"tma-zero-fill:op{index}" if is_load else f"tma-oob-sink:op{index}"
        )
        for logical_index in product(*(range(dimension) for dimension in tile_shape)):
            descriptor_index = [
                coordinate + item
                for coordinate, item in zip(ordered_coordinates, logical_index)
            ]
            in_bounds = all(
                0 <= item < dimension
                for item, dimension in zip(descriptor_index, shape)
            )
            run_allocation = allocation if in_bounds else zero_fill_allocation
            descriptor_element = (
                sum(item * stride for item, stride in zip(descriptor_index, strides))
                if in_bounds
                else counterpart_offset - counterpart_region["offset"]
            )
            descriptor_offset = descriptor_element * element_bytes if in_bounds else descriptor_element
            if (
                runs
                and run_allocation == runs[-1][0]
                and descriptor_offset == runs[-1][1] + runs[-1][3]
                and counterpart_offset == runs[-1][2] + runs[-1][3]
            ):
                runs[-1][3] += element_bytes
            else:
                runs.append([run_allocation, descriptor_offset, counterpart_offset, element_bytes])
            counterpart_offset += element_bytes
        result: list[tuple[str, str, int]] = []
        for run_allocation, descriptor_offset, shared_offset, extent in runs:
            global_region = self._region(run_allocation, "GLOBAL", descriptor_offset, extent)
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

    def _capture_tensormap(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        if name == "ttng.tensormap_create":
            if operation.get_num_operands() < 3 or operation.get_num_results() != 0:
                raise CaptureError(
                    "TRITON_TENSORMAP_UNRESOLVED",
                    "tensormap creation requires scratch storage, a base pointer, and static metadata",
                    {"operation_index": index},
                )
            scratch = operation.get_operand(0).id()
            if scratch not in self.global_scratch_allocations:
                raise CaptureError(
                    "TRITON_TENSORMAP_UNRESOLVED",
                    "tensormap destination is not a finite global scratch allocation",
                    {"operation_index": index},
                )
            source = self._mark_external_pointer(operation.get_operand(1).id())
            metadata = [
                self.static_values.get(operation.get_operand(position).id())
                for position in range(2, operation.get_num_operands())
            ]
            if source is None or any(value is None for value in metadata):
                raise CaptureError(
                    "TRITON_TENSORMAP_UNRESOLVED",
                    "tensormap base pointer or metadata is unresolved",
                    {"operation_index": index},
                )
            self.tensormap_sources[scratch] = source
            return
        if name == "ttng.tensormap_fenceproxy_acquire":
            if operation.get_num_operands() != 1 or operation.get_num_results() != 0:
                raise CaptureError(
                    "TRITON_TENSORMAP_UNRESOLVED",
                    "tensormap fence requires one scratch descriptor",
                    {"operation_index": index},
                )
            scratch = operation.get_operand(0).id()
            if scratch not in self.tensormap_sources:
                raise CaptureError(
                    "TRITON_TENSORMAP_UNRESOLVED",
                    "tensormap fence has no preceding descriptor creation",
                    {"operation_index": index},
                )
            self.fenced_tensormaps.add(scratch)
            return
        if operation.get_num_operands() != 1 or operation.get_num_results() != 1:
            raise CaptureError(
                "TRITON_TENSORMAP_UNRESOLVED",
                "descriptor reinterpretation requires one scratch pointer and one result",
                {"operation_index": index},
            )
        scratch = operation.get_operand(0).id()
        if scratch not in self.fenced_tensormaps:
            raise CaptureError(
                "TRITON_TENSORMAP_UNRESOLVED",
                "descriptor reinterpretation lacks a preceding proxy-acquire fence",
                {"operation_index": index},
            )
        allocation, source_width, offsets = self.tensormap_sources[scratch]
        if len(offsets) != 1:
            raise CaptureError(
                "TRITON_TENSORMAP_UNRESOLVED",
                "tensormap base pointer is not scalar",
                {"operation_index": index},
            )
        result = operation.get_result(0)
        elements, result_width = self._value_extent(result)
        if source_width != result_width:
            raise CaptureError(
                "TRITON_TENSORMAP_UNRESOLVED",
                "tensormap element width disagrees with its source pointer",
                {"operation_index": index},
            )
        self.descriptor_regions[result.id()] = self._region(
            allocation,
            "GLOBAL",
            offsets[0] * source_width,
            elements * result_width,
        )

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
        owner = self.current_actor
        sequence = self.tma_store_group_sequences.get(owner, 0)
        self._emit(
            "triton.tma_store.commit_group",
            index,
            resource=resource,
            group_owner=owner,
            group_seq=sequence,
        )
        self.tma_store_group_sequences[owner] = sequence + 1

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
            if self.pending_proxy_fence is not None:
                self._emit(
                    "triton.proxy_fence",
                    self.pending_proxy_fence,
                    from_domain="GENERIC",
                    to_domain="TMA",
                    scope="CTA",
                    regions=sorted({
                        source_region
                        for source_region, _destination, _bytes in segments
                    }),
                    visibility_mask={
                        "shared": False,
                        "global_read": False,
                        "global_write": True,
                        "tensor_read": False,
                        "tensor_write": False,
                        "control_only": False,
                    },
                )
                self.pending_proxy_fence = None
            self._emit(
                "triton.tma_load.issue",
                index,
                resource=resource,
                transfers=[
                    {
                        "source_region": source_region,
                        "destination_region": destination_region,
                        "bytes": byte_count,
                    }
                    for source_region, destination_region, byte_count in segments
                ],
                bytes=sum(byte_count for _source, _destination, byte_count in segments),
                lifecycle_epoch=lifecycle,
                generation=self.tma_issue_generation[key],
            )

    def _capture_wgmma(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        resource = self._wgmma_resource(operation, index)
        if name == "ttng.warp_group_dot":
            if operation.get_num_operands() < 3 or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_WGMMA_UNRESOLVED",
                    "warp_group_dot requires two shared operands, accumulator, and one result",
                    {"operation_index": index},
                )
            operand_regions: list[str] = []
            shared_regions: list[str] = []
            for position in (0, 1):
                operand = operation.get_operand(position)
                region = self.memdesc_regions.get(operand.id())
                if region is not None:
                    shared_regions.append(region)
                    operand_regions.append(region)
                    continue
                if "tensor<" in str(operand.get_type()):
                    operand_regions.append(self._register_region(operand))
                    continue
                raise CaptureError(
                    "TRITON_WGMMA_UNRESOLVED",
                    "warp_group_dot operand provenance is unresolved",
                    {"operation_index": index, "operand": position},
                )
            if not shared_regions:
                raise CaptureError(
                    "TRITON_WGMMA_UNRESOLVED",
                    "SM90 warp_group_dot requires at least one shared-memory operand",
                    {"operation_index": index},
                )
            if self.final_ptx_proxy_fence_pending:
                if self.pending_proxy_fence is None:
                    self.pending_proxy_fence = index
                self.final_ptx_proxy_fence_pending = False
            if self.pending_proxy_fence is not None:
                self._emit(
                    "triton.proxy_fence",
                    self.pending_proxy_fence,
                    from_domain="GENERIC",
                    to_domain="TENSOR_CORE",
                    scope="CTA",
                    regions=shared_regions,
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
            sequence = self.wgmma_group_sequences.get(owner, 0)
            engine = self._async_engine_actor("WGMMA")
            self._emit(
                "triton.wgmma.fence_operand",
                index,
                resource=resource,
                group_owner=owner,
                regions=shared_regions,
            )
            self._emit(
                "triton.wgmma.issue",
                index,
                resource=resource,
                group_owner=owner,
                regions=operand_regions,
                destination_region=destination,
                engine_id=engine,
                engine_seq=sequence,
            )
            self._emit(
                "triton.wgmma.commit_group",
                index,
                resource=resource,
                group_owner=owner,
                group_seq=sequence,
            )
            result = operation.get_result(0)
            self.register_regions[result.id()] = destination
            self.wgmma_results[result.id()] = sequence
            self.wgmma_group_sequences[owner] = sequence + 1
            self.pending_wgmma_owners.add(owner)
            return
        wait_count = operation.get_int_attr("pendings")
        has_known_result = any(
            operation.get_operand(position).id() in self.wgmma_results
            for position in range(operation.get_num_operands())
        )
        if (
            wait_count is None
            or not 0 <= int(wait_count) <= 7
            or operation.get_num_operands() < 1
            or operation.get_num_results() != operation.get_num_operands()
            or (not has_known_result and self.current_actor in self.pending_wgmma_owners)
        ):
            raise CaptureError(
                "TRITON_WGMMA_WAIT_UNRESOLVED",
                "warp_group_dot_wait requires known dot results or an empty group stream and static pendings",
                {"operation_index": index, "pendings": wait_count},
            )
        if not has_known_result:
            for position in range(operation.get_num_operands()):
                self._forward_value(operation.get_operand(position), operation.get_result(position))
            return
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
        if int(wait_count) == 0:
            self.pending_wgmma_owners.discard(self.current_actor)

    def _capture_wgmma_result_reads(self, operation: Any, index: int) -> None:
        if operation.get_name() == "arith.select" and operation.get_num_operands() == 3:
            condition = self.static_values.get(operation.get_operand(0).id())
            if condition in {0, 1, -1, False, True} and operation.get_num_results() == 1:
                selected = operation.get_operand(1 if bool(condition) else 2)
                self._forward_value(selected, operation.get_result(0))
                return
        regions = {
            self.register_regions[operation.get_operand(position).id()]
            for position in range(operation.get_num_operands())
            if (
                operation.get_operand(position).id() in self.wgmma_results
                and operation.get_operand(position).id() in self.register_regions
            )
        }
        for region in sorted(regions):
            self._emit("triton.mem.read", index, region=region)

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
        if re.search(r"!tt\.ptr<i1\b", str(operation.get_operand(0).get_type())):
            raise CaptureError(
                "TRITON_MEMORY_ACCESS_UNRESOLVED",
                "ordinary memory access through a sub-byte pointer is unsupported",
                {"operation": name, "operation_index": index},
            )
        self._mark_external_pointer(pointer_id)
        regions = self.pointer_region_sets.get(pointer_id)
        if not regions:
            raise CaptureError(
                "TRITON_MEMORY_ACCESS_UNRESOLVED",
                "ordinary memory pointer provenance is unresolved",
                {"operation": name, "operation_index": index},
            )
        mask_position = 1 if name == "tt.load" and operand_count >= 2 else 2 if name == "tt.store" and operand_count == 3 else None
        mask: Any = True
        if mask_position is not None:
            mask_id = operation.get_operand(mask_position).id()
            if mask_id not in self.static_values:
                raise CaptureError(
                    "TRITON_PREDICATE_UNRESOLVED",
                    "ordinary memory mask is not statically resolved",
                    {"operation": name, "operation_index": index},
                )
            mask = self.static_values[mask_id]
            regions = self._masked_pointer_regions(pointer_id, mask)
        lane_regions = self._lane_memory_regions(
            operation,
            pointer_id,
            mask,
            is_store=name == "tt.store",
        )
        for region in regions:
            coarse = self.regions[region]
            lanes: list[int] = []
            region_offsets = [0]
            exact_region_ids: list[str] = []
            for lane, exact_regions in lane_regions.items():
                selected = []
                for exact_region in exact_regions:
                    exact = self.regions[exact_region]
                    if (
                        exact["allocation_id"] == coarse["allocation_id"]
                        and exact["address_space"] == coarse["address_space"]
                        and exact["offset"] < coarse["offset"] + coarse["extent"]
                        and coarse["offset"] < exact["offset"] + exact["extent"]
                    ):
                        selected.append(exact_region)
                if selected:
                    lanes.append(lane)
                    exact_region_ids.extend(selected)
                    region_offsets.append(len(exact_region_ids))
            if not lanes:
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "memory region has no statically identified executing lane",
                    {"operation": name, "operation_index": index, "region": region},
                )
            self._emit(
                "triton.mem.read" if name == "tt.load" else "triton.mem.write",
                index,
                region=region,
                lane_effects={
                    "lanes": lanes,
                    "region_offsets": region_offsets,
                    "regions": exact_region_ids,
                },
                execution_evidence={
                    "scope": "INSTANCE",
                    "method": "ttg-blocked-linear-layout-v1",
                    "source": str(operation.get_operand(0).get_type()),
                    "redundant_policy": (
                        "STORE_REPRESENTATIVE" if name == "tt.store" else "LOAD_ALL"
                    ),
                },
            )
        if name == "tt.load" and operation.get_num_results() == 1:
            allocations = {
                self.regions[region]["allocation_id"] for region in regions
            }
            if len(allocations) == 1:
                value = self.allocation_constant_values.get(next(iter(allocations)))
                mask_enabled = (
                    mask is True
                    or isinstance(mask, (tuple, list)) and all(bool(item) for item in mask)
                )
                if value is not None and mask_enabled:
                    elements, _width = self._value_extent(operation.get_result(0))
                    self.static_values[operation.get_result(0).id()] = (
                        value if elements == 1 else tuple(value for _ in range(elements))
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
            pointer = self._mark_external_pointer(pointer_id)
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
            lane_effects = self._cp_async_lane_effects(
                operation,
                pointer_id,
                destination,
                predicates,
            )
            first_slot, last_slot = lane_effects.pop("destination_slot_span") or (0, 0)
            if lane_effects["lanes"]:
                selected = [
                    element for element, predicate in enumerate(predicates) if predicate
                ]
                selected_offsets = [offsets[element] for element in selected]
                # The exact effects are the per-thread transfers in lane_effects;
                # the operation-level regions only summarize their byte spans
                # (strided tiles and partial masks span gaps).
                source_start = min(selected_offsets) * width
                source_region = self._region(
                    allocation,
                    "GLOBAL",
                    source_start,
                    (max(selected_offsets) + 1) * width - source_start,
                )
                destination_region_id = self._region(
                    destination_region["allocation_id"],
                    destination_region["address_space"],
                    destination_region["offset"] + first_slot * width,
                    (last_slot - first_slot + 1) * width,
                )
                self._membar_before_shared_write(destination_region["allocation_id"], index)
                self._emit(
                    "triton.cp_async.issue",
                    index,
                    resource=resource,
                    source_region=source_region,
                    destination_region=destination_region_id,
                    bytes=sum(lane_effects["bytes"]),
                    group_owner=self.current_actor,
                    predicate=True,
                    lane_effects=lane_effects,
                    execution_evidence={
                        "program_instances": lane_effects["lanes"],
                        "executors": lane_effects["lanes"],
                        "transition": "PER_INSTANCE",
                        "rendezvous": "NONE",
                        "method": "ttgir-final-nvvm-cp-async-per-thread-v1",
                    },
                )
                self.pending_shared_write_allocations.add(
                    destination_region["allocation_id"]
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
            unresolved_tokens = [
                position
                for position, token in enumerate(tokens)
                if token not in self.group_tokens
            ]
            wait_count = operation.get_int_attr("num")
            if (
                unresolved_tokens
                or wait_count is None
                or not 0 <= int(wait_count) <= 15
                or operation.get_num_results() != 1
            ):
                raise CaptureError(
                    "TRITON_CP_ASYNC_WAIT_UNRESOLVED",
                    "async wait requires committed SSA tokens and static SM90 num in [0, 15]",
                    {
                        "operation_index": index,
                        "unresolved_token_operands": unresolved_tokens,
                        "wait_group_count": wait_count,
                    },
                )
            self._emit(
                "triton.cp_async.wait_group",
                index,
                resource=resource,
                group_owner=self.current_actor,
                wait_group_count=int(wait_count),
            )
            # Triton's Membar pass places a CTA barrier after every async
            # wait, making the completed copies visible to all threads.
            self._emit_shared_membar(index)
            result_id = operation.get_result(0).id()
            self.wait_tokens.add(result_id)
            self.group_tokens[result_id] = max(
                (self.group_tokens[token] for token in tokens),
                default=self.cp_group_seq - 1,
            )
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
        allocation = self.regions[region]["allocation_id"]
        if allocation in self.pending_shared_write_allocations:
            self._emit_shared_membar(index)
        if (
            operation.get_num_operands() == 2
            and operation.get_operand(1).id() not in self.wait_tokens
        ):
            raise CaptureError(
                "TRITON_LOCAL_LOAD_UNRESOLVED",
                "local load completion token is unresolved",
                {"operation_index": index},
            )
        result_type = str(operation.get_result(0).get_type())
        if "#ttg.dot_op<" in result_type:
            # MMA operand layouts are not inverted per lane; every thread is
            # modelled as reading the whole tile, which over-approximates
            # its ldmatrix footprint.
            elements, width = self._value_extent(operation.get_result(0))
            if elements * width != self.regions[region]["extent"]:
                raise CaptureError(
                    "TRITON_LAYOUT_OWNERSHIP_UNRESOLVED",
                    "dot operand and memdesc byte extents differ",
                    {"type": result_type, "region": region},
                )
            participants = self.actor_specs[self.current_actor]["participants"]
            lane_regions = {lane: (region,) for lane in participants}
            method = "ttg-dot-operand-whole-tile-v1"
        else:
            lane_regions = self._lane_memdesc_regions(
                operation.get_result(0),
                operation.get_operand(0),
                region,
                representative_only=False,
            )
            method = "ttg-blocked-linear-layout-v1"
        lanes = sorted(lane_regions)
        flattened = [item for lane in lanes for item in lane_regions[lane]]
        offsets = [0]
        for lane in lanes:
            offsets.append(offsets[-1] + len(lane_regions[lane]))
        self._emit(
            "triton.mem.read",
            index,
            region=region,
            lane_effects={
                "lanes": lanes,
                "region_offsets": offsets,
                "regions": flattened,
            },
            execution_evidence={
                "scope": "INSTANCE",
                "method": method,
                "source": result_type,
                "redundant_policy": "LOAD_ALL",
            },
        )
        self.pending_shared_read_allocations.add((self.current_actor, allocation))

    def _capture_local_store(self, operation: Any, index: int) -> None:
        if operation.get_num_operands() != 2 or operation.get_num_results() != 0:
            raise CaptureError(
                "TRITON_LOCAL_STORE_UNRESOLVED",
                "ttg.local_store requires one value and one direct memdesc",
                {"operation_index": index},
            )
        value = operation.get_operand(0)
        region = self.memdesc_regions.get(operation.get_operand(1).id())
        if region is None:
            raise CaptureError(
                "TRITON_LOCAL_STORE_UNRESOLVED",
                "local store memdesc provenance is unresolved",
                {"operation_index": index},
            )
        elements, width = self._value_extent(value)
        if elements * width != self.regions[region]["extent"]:
            raise CaptureError(
                "TRITON_LOCAL_STORE_UNRESOLVED",
                "local store value and destination byte intervals differ",
                {
                    "operation_index": index,
                    "source_extent": elements * width,
                    "destination_extent": self.regions[region]["extent"],
                },
            )
        lane_regions = self._lane_memdesc_regions(
            value,
            operation.get_operand(1),
            region,
            representative_only=True,
        )
        lanes = sorted(lane_regions)
        flattened = [item for lane in lanes for item in lane_regions[lane]]
        offsets = [0]
        for lane in lanes:
            offsets.append(offsets[-1] + len(lane_regions[lane]))
        self._membar_before_shared_write(self.regions[region]["allocation_id"], index)
        self._emit(
            "triton.mem.write",
            index,
            region=region,
            lane_effects={
                "lanes": lanes,
                "region_offsets": offsets,
                "regions": flattened,
            },
            execution_evidence={
                "scope": "INSTANCE",
                "method": "ttg-blocked-linear-layout-v1",
                "source": str(value.get_type()),
                "redundant_policy": "STORE_REPRESENTATIVE",
            },
        )
        self.pending_shared_write_allocations.add(
            self.regions[region]["allocation_id"]
        )

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
        if name == "ttg.memdesc_subslice":
            if operation.get_num_operands() != 1 or operation.get_num_results() != 1:
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.memdesc_subslice requires one base and one result",
                    {"operation_index": index},
                )
            base_region_id = self.memdesc_regions.get(operation.get_operand(0).id())
            attributes = self.array_attributes.get(index)
            if base_region_id is None:
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.memdesc_subslice base is not statically resolved",
                    {"operation_index": index},
                )
            if not isinstance(attributes, dict) or set(attributes) != {"memdesc_byte_offset"}:
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_REQUIRED",
                    "ttg.memdesc_subslice requires a SHA-bound byte offset",
                    {"operation_index": index},
                )
            byte_offset = attributes["memdesc_byte_offset"]
            if not isinstance(byte_offset, int) or isinstance(byte_offset, bool) or byte_offset < 0:
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_INVALID",
                    "ttg.memdesc_subslice byte offset must be a nonnegative integer",
                    {"operation_index": index, "byte_offset": byte_offset},
                )
            elements, width = self._value_extent(operation.get_result(0))
            subview_extent = elements * width
            base_region = self.regions[base_region_id]
            offset = base_region["offset"] + byte_offset
            if offset + subview_extent > base_region["offset"] + base_region["extent"]:
                raise CaptureError(
                    "TRITON_BINDING_EVIDENCE_INVALID",
                    "ttg.memdesc_subslice is outside the static base allocation",
                    {
                        "operation_index": index,
                        "byte_offset": byte_offset,
                        "extent": subview_extent,
                    },
                )
            self.memdesc_regions[operation.get_result(0).id()] = self._region(
                base_region["allocation_id"],
                base_region["address_space"],
                offset,
                subview_extent,
            )
            return
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
                base = self._mark_external_pointer(operand.id())
                if base is None:
                    return
                allocation, width, offsets = base
                if len(offsets) != 1:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.splat pointer operand is not scalar",
                        {"operation_index": index},
                    )
                elements, result_width = self._value_extent(operation.get_result(0))
                if result_width != width:
                    raise CaptureError(
                        "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                        "tt.splat changes pointer element width",
                        {
                            "operation_index": index,
                            "source_width": width,
                            "result_width": result_width,
                        },
                    )
                self._set_pointer_value(
                    operation.get_result(0).id(),
                    allocation,
                    width,
                    offsets * elements,
                )
                return
            value = self.static_values.get(operand.id())
            if value is None:
                return
            shape = self._value_shape(operation.get_result(0))
            if not shape or any(dimension <= 0 for dimension in shape):
                raise CaptureError(
                    "TRITON_STATIC_VALUE_UNRESOLVED",
                    "tt.splat result requires a positive static shape",
                    {"operation_index": index, "shape": shape},
                )
            elements = 1
            for dimension in shape:
                elements *= dimension
            self.static_values[operation.get_result(0).id()] = tuple(value for _ in range(elements))

    def _execute_bitcast(self, operation: Any, index: int) -> None:
        if operation.get_num_operands() != 1 or operation.get_num_results() != 1:
            raise CaptureError(
                "TRITON_OPERATION_UNSUPPORTED",
                "tt.bitcast requires one operand and one result",
                {"operation_index": index},
            )
        operand = operation.get_operand(0)
        result = operation.get_result(0)
        operand_type = str(operand.get_type())
        result_type = str(result.get_type())
        address_markers = ("!tt.ptr<", "!ttg.memdesc<", "!tt.tensordesc<")
        if not any(marker in operand_type or marker in result_type for marker in address_markers):
            return
        if "!tt.ptr<" not in operand_type or "!tt.ptr<" not in result_type:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "tt.bitcast cannot change between pointer and non-pointer address values",
                {"operation_index": index, "operand_type": operand_type, "result_type": result_type},
            )
        pointer = self._ensure_pointer_value(operand.id())
        if pointer is None:
            return
        allocation, source_width, offsets = pointer
        result_width = (
            self._scalar_pointer_width(result_type)
            if result_type.startswith("!tt.ptr<")
            else self._value_extent(result)[1]
        )
        if result_width is None or source_width != result_width:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "tt.bitcast changes pointer element width",
                {
                    "operation_index": index,
                    "source_width": source_width,
                    "result_width": result_width,
                },
            )
        self._set_pointer_value(result.id(), allocation, result_width, offsets)

    def _execute_poison(self, operation: Any, index: int) -> None:
        if operation.get_num_operands() != 0 or operation.get_num_results() != 1:
            raise CaptureError(
                "TRITON_OPERATION_UNSUPPORTED",
                "ub.poison requires no operands and one result",
                {"operation_index": index},
            )

    def _execute_value_only_pure(self, operation: Any, index: int) -> None:
        values = [
            operation.get_operand(position)
            for position in range(operation.get_num_operands())
        ] + [
            operation.get_result(position)
            for position in range(operation.get_num_results())
        ]
        address_types = [
            str(value.get_type())
            for value in values
            if any(
                marker in str(value.get_type())
                for marker in ("!tt.ptr<", "!ttg.memdesc<", "!tt.tensordesc<")
            )
        ]
        if address_types:
            raise CaptureError(
                "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                "value-only operation cannot carry pointer or descriptor provenance",
                {
                    "operation": operation.get_name(),
                    "operation_index": index,
                    "types": address_types,
                },
            )

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
            self.static_values[result.id()] = self.selected_cta[axis]
            return
        if name == "tt.get_num_programs":
            axis = operation.get_int_attr("axis")
            if axis not in {0, 1, 2}:
                raise CaptureError(
                    "TRITON_STATIC_VALUE_UNRESOLVED",
                    "tt.get_num_programs axis is not static",
                    {"operation_index": index, "axis": axis},
                )
            self.static_values[result.id()] = self.launch_grid[axis]
            return
        if operation.get_num_operands() == 1:
            operand_id = operation.get_operand(0).id()
            if operand_id in self.pointer_values:
                allocation, width, offsets = self.pointer_values[operand_id]
                if name == "tt.broadcast":
                    try:
                        offsets = self._broadcast_values(
                            offsets,
                            self._value_shape(operation.get_operand(0)),
                            self._value_shape(result),
                        )
                    except ValueError as error:
                        raise CaptureError(
                            "TRITON_POINTER_PROVENANCE_UNRESOLVED",
                            "tt.broadcast pointer shape is unresolved",
                            {"operation_index": index, "reason": str(error)},
                        ) from error
                self._set_pointer_value(result.id(), allocation, width, offsets)
            elif operand_id in self.scalar_pointer_arguments:
                self.scalar_pointer_arguments[result.id()] = self.scalar_pointer_arguments[operand_id]
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
        } and len(operands) == 1 and operands[0] is not None:
            value = operands[0]
            if name == "tt.broadcast" and isinstance(value, (tuple, list)):
                try:
                    value = self._broadcast_values(
                        tuple(value),
                        self._value_shape(operation.get_operand(0)),
                        self._value_shape(result),
                    )
                except ValueError as error:
                    raise CaptureError(
                        "TRITON_STATIC_VALUE_UNRESOLVED",
                        "tt.broadcast static shape is unresolved",
                        {"operation_index": index, "reason": str(error)},
                    ) from error
            self.static_values[result.id()] = value
            return
        if name == "arith.index_cast" and len(operands) == 1 and operands[0] is not None:
            bits = self._integer_width(str(result.get_type()))
            value = operands[0]
            convert = (lambda item: self._signed_integer(item, bits)) if bits is not None else (lambda item: item)
            if isinstance(value, int) and not isinstance(value, bool):
                self.static_values[result.id()] = convert(value)
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value
            ):
                self.static_values[result.id()] = tuple(convert(item) for item in value)
            return
        if name in {"arith.extsi", "arith.extui"} and len(operands) == 1 and operands[0] is not None:
            source_bits = self._integer_width(str(operation.get_operand(0).get_type()))
            result_bits = self._integer_width(str(result.get_type()))
            if source_bits is None or result_bits is None:
                return

            def extend(value: int) -> int:
                source = (
                    self._signed_integer(value, source_bits)
                    if name == "arith.extsi"
                    else self._unsigned_integer(value, source_bits)
                )
                return self._signed_integer(source, result_bits)

            value = operands[0]
            if isinstance(value, int) and not isinstance(value, bool):
                self.static_values[result.id()] = extend(value)
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value
            ):
                self.static_values[result.id()] = tuple(extend(item) for item in value)
            return
        if name == "arith.trunci" and len(operands) == 1 and operands[0] is not None:
            bits = self._integer_width(str(result.get_type()))
            if bits is None:
                return
            value = operands[0]
            if isinstance(value, int) and not isinstance(value, bool):
                self.static_values[result.id()] = self._signed_integer(value, bits)
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value
            ):
                self.static_values[result.id()] = tuple(
                    self._signed_integer(item, bits) for item in value
                )
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
        operand_bits = self._integer_width(str(operation.get_operand(0).get_type()))
        result_bits = self._integer_width(str(result.get_type()))
        if name == "arith.cmpi":
            predicate = operation.get_int_attr("predicate")
            operand_type = str(operation.get_operand(0).get_type())
            if predicate is None or (operand_bits is None and operand_type != "index"):
                return
            unsigned = predicate >= 6
            if operand_bits is None:
                normalize = lambda value, _bits: value
                comparison_bits = 0
            else:
                normalize = self._unsigned_integer if unsigned else self._signed_integer
                comparison_bits = operand_bits
            comparisons = {
                0: lambda a, b: a == b,
                1: lambda a, b: a != b,
                2: lambda a, b: a < b,
                3: lambda a, b: a <= b,
                4: lambda a, b: a > b,
                5: lambda a, b: a >= b,
                6: lambda a, b: a < b,
                7: lambda a, b: a <= b,
                8: lambda a, b: a > b,
                9: lambda a, b: a >= b,
            }
            comparison = comparisons.get(predicate)
            if comparison is not None:
                values = tuple(
                    int(comparison(normalize(a, comparison_bits), normalize(b, comparison_bits)))
                    for a, b in zip(left_values, right_values)
                )
                self.static_values[result.id()] = values[0] if scalar_result else values
            return
        result_type = str(result.get_type())
        if result_bits is None and result_type != "index":
            return

        def signed_divide(left: int, right: int) -> int:
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient

        def signed_remainder(left: int, right: int) -> int:
            return left - signed_divide(left, right) * right

        if result_bits is None:
            signed_left = unsigned_left = tuple(left_values)
            signed_right = unsigned_right = tuple(right_values)
        else:
            signed_left = tuple(self._signed_integer(value, result_bits) for value in left_values)
            signed_right = tuple(self._signed_integer(value, result_bits) for value in right_values)
            unsigned_left = tuple(self._unsigned_integer(value, result_bits) for value in left_values)
            unsigned_right = tuple(self._unsigned_integer(value, result_bits) for value in right_values)
        if name in {"arith.ceildivsi", "arith.divsi", "arith.remsi", "arith.maxsi", "arith.minsi"}:
            evaluated_left, evaluated_right = signed_left, signed_right
        elif name == "arith.divui":
            evaluated_left, evaluated_right = unsigned_left, unsigned_right
        else:
            evaluated_left, evaluated_right = left_values, right_values
        if name in {"arith.ceildivsi", "arith.divsi", "arith.divui", "arith.remsi"} and any(
            value == 0 for value in evaluated_right
        ):
            return
        if result_bits is not None and name == "arith.divsi" and any(
            left == -(1 << (result_bits - 1)) and right == -1
            for left, right in zip(evaluated_left, evaluated_right)
        ):
            return
        functions = {
            "arith.addi": lambda a, b: a + b,
            "arith.andi": lambda a, b: a & b,
            "arith.ceildivsi": lambda a, b: -((-a) // b),
            "arith.divsi": signed_divide,
            "arith.divui": lambda a, b: a // b,
            "arith.maxsi": max,
            "arith.minsi": min,
            "arith.muli": lambda a, b: a * b,
            "arith.ori": lambda a, b: a | b,
            "arith.remsi": signed_remainder,
            "arith.subi": lambda a, b: a - b,
            "arith.xori": lambda a, b: a ^ b,
        }
        function = functions.get(name)
        if function is None:
            return
        values = tuple(
            function(a, b) if result_bits is None else self._signed_integer(function(a, b), result_bits)
            for a, b in zip(evaluated_left, evaluated_right)
        )
        self.static_values[result.id()] = values[0] if scalar_result else values

    def _execute_call(self, operation: Any, index: int, path: tuple[Any, ...]) -> None:
        symbol = operation.get_flat_symbol_ref_attr("callee")
        block_id = self.function_entry_blocks.get(symbol)
        if symbol is None or block_id is None:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "tt.call target is unavailable",
                {"operation_index": index, "callee": symbol},
            )
        block = self.blocks[block_id]
        if (
            operation.get_num_operands() != block.get_num_arguments()
            or operation.get_num_results() != 0
        ):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "static tt.call requires matching operands and no return values",
                {
                    "operation_index": index,
                    "callee": symbol,
                    "operands": operation.get_num_operands(),
                    "arguments": block.get_num_arguments(),
                    "results": operation.get_num_results(),
                },
            )
        if symbol in self.active_function_calls:
            raise CaptureError(
                "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                "recursive tt.call is outside the finite static profile",
                {"operation_index": index, "callee": symbol},
            )
        for position in range(operation.get_num_operands()):
            self._forward_value(operation.get_operand(position), block.get_argument(position))
        self.active_function_calls.add(symbol)
        self.active_cfg_blocks.add(block_id)
        try:
            self._execute_block(block_id, (*path, "call", index, symbol))
        finally:
            self.active_cfg_blocks.remove(block_id)
            self.active_function_calls.remove(symbol)

    def _execute_block(self, block_id: int, path: tuple[Any, ...]) -> list[Any]:
        for index in self.block_operations.get(block_id, []):
            operation = self.operations[index]
            name = operation.get_name()
            if name in {"scf.yield", "scf.condition", "ttg.warp_yield", "ttg.warp_return"}:
                return [
                    self.static_values.get(operation.get_operand(position).id())
                    for position in range(operation.get_num_operands())
                ]
            if name == "tt.call":
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                self.static_expansions += 1
                self._execute_call(operation, index, path)
                continue
            if name in {"cf.br", "cf.cond_br"}:
                self.executable_nodes_seen += 1
                self.classified_nodes += 1
                self.elided_nodes += 1
                successors = self.cfg_successors.get(block_id, ())
                if name == "cf.br":
                    if operation.get_num_operands() != 0 or len(successors) != 1:
                        raise CaptureError(
                            "TRITON_CONTROL_FLOW_UNRESOLVED",
                            "static cf.br with block arguments is unsupported",
                            {"operation_index": index, "successors": len(successors)},
                        )
                    target = successors[0]
                else:
                    if operation.get_num_operands() != 1 or len(successors) != 2:
                        raise CaptureError(
                            "TRITON_CONTROL_FLOW_UNRESOLVED",
                            "static cf.cond_br with block arguments is unsupported",
                            {"operation_index": index, "successors": len(successors)},
                        )
                    condition = self._constant(operation.get_operand(0), "cf.cond_br condition")
                    if condition not in {0, 1, -1, False, True}:
                        raise CaptureError(
                            "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                            "cf.cond_br condition must be a static Boolean",
                            {"operation_index": index, "condition": condition},
                        )
                    target = successors[0 if bool(condition) else 1]
                if self.blocks[target].get_num_arguments() != 0:
                    raise CaptureError(
                        "TRITON_CONTROL_FLOW_UNRESOLVED",
                        "static CFG successor block arguments are unsupported",
                        {"operation_index": index},
                    )
                if target in self.active_cfg_blocks:
                    raise CaptureError(
                        "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "cyclic cf control flow requires structured static lowering",
                        {"operation_index": index},
                    )
                self.active_cfg_blocks.add(target)
                try:
                    return self._execute_block(target, (*path, "cf", index, target))
                finally:
                    self.active_cfg_blocks.remove(target)
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

    def _value_fact_maps(self) -> tuple[dict[int, Any], ...]:
        return (
            self.static_values,
            self.scalar_pointer_arguments,
            self.pointer_values,
            self.pointer_regions,
            self.pointer_region_sets,
            self.memdesc_regions,
            self.descriptor_regions,
            self.group_tokens,
            self.barrier_arguments,
            self.register_regions,
            self.wgmma_results,
        )

    def _clear_value_facts(self, value_id: int) -> None:
        for mapping in self._value_fact_maps():
            mapping.pop(value_id, None)
        self.staged_tokens.discard(value_id)
        self.wait_tokens.discard(value_id)

    def _forward_value(self, source: Any, target: Any) -> None:
        source_id = source.id()
        target_id = target.id()
        if source_id == target_id:
            return
        self._clear_value_facts(target_id)
        for mapping in self._value_fact_maps():
            if source_id in mapping:
                mapping[target_id] = mapping[source_id]
        if source_id in self.staged_tokens:
            self.staged_tokens.add(target_id)
        if source_id in self.wait_tokens:
            self.wait_tokens.add(target_id)

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
            self.static_values.get(operation.get_operand(position).id())
            for position in range(3, operation.get_num_operands())
        ]
        arguments = [block.get_argument(position) for position in range(block.get_num_arguments())]
        if len(arguments) != 1 + len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for block arguments do not match loop-carried values",
                {"operation_index": index},
            )
        carried_sources = [
            operation.get_operand(position)
            for position in range(3, operation.get_num_operands())
        ]
        yield_operations = [
            self.operations[operation_index]
            for operation_index in self.block_operations.get(block_id, [])
            if self.operations[operation_index].get_name() == "scf.yield"
        ]
        if len(yield_operations) != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for body must contain one yield terminator",
                {"operation_index": index, "yields": len(yield_operations)},
            )
        yield_operation = yield_operations[0]
        for iteration, induction in enumerate(range(lower, upper, step)):
            self.static_expansions += 1
            self.static_values[arguments[0].id()] = induction
            for argument, value, source in zip(arguments[1:], carried, carried_sources):
                self.static_values[argument.id()] = value
                self._forward_value(source, argument)
                type_text = str(argument.get_type())
                if (
                    "tensor<" in type_text
                    and "!tt.ptr<" not in type_text
                    and (
                        argument.id() in self.wgmma_accumulator_inputs
                        or re.search(r"(?:^|x)f32(?:$|[,>])", type_text)
                    )
                    and argument.id() not in self.register_regions
                ):
                    self.register_regions[argument.id()] = self._register_region(source)
            yielded = self._execute_block(block_id, (*path, "for", index, iteration, induction))
            if len(yielded) != len(carried):
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "scf.for yield count does not match loop-carried values",
                    {"operation_index": index},
                )
            carried_sources = [
                yield_operation.get_operand(position)
                for position in range(yield_operation.get_num_operands())
            ]
            for source, argument in zip(carried_sources, arguments[1:]):
                self._forward_value(source, argument)
            carried = yielded
        if operation.get_num_results() != len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.for result count does not match loop-carried values",
                {"operation_index": index},
            )
        for position, value in enumerate(carried):
            result = operation.get_result(position)
            self.static_values[result.id()] = value
            self._forward_value(carried_sources[position], result)

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
        yield_operation = None
        if branch < operation.get_num_regions():
            region = operation.get_region(branch)
            blocks = sorted(self.region_blocks.get(region.id(), set()))
            if not blocks and operation.get_num_results() == 0:
                return
            if len(blocks) != 1:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "selected scf.if region must contain one block unless it is an empty no-result branch",
                    {"operation_index": index, "blocks": len(blocks)},
                )
            block_id = blocks[0]
            yield_operations = [
                self.operations[operation_index]
                for operation_index in self.block_operations.get(block_id, [])
                if self.operations[operation_index].get_name() == "scf.yield"
            ]
            if len(yield_operations) != 1:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "selected scf.if block must contain one yield terminator",
                    {"operation_index": index, "yields": len(yield_operations)},
                )
            yield_operation = yield_operations[0]
            yielded = self._execute_block(block_id, (*path, "if", index, int(bool(condition))))
        if len(yielded) != operation.get_num_results():
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.if yield/result counts disagree",
                {"operation_index": index},
            )
        for position, value in enumerate(yielded):
            result = operation.get_result(position)
            if yield_operation is not None:
                self._forward_value(yield_operation.get_operand(position), result)
            self.static_values[result.id()] = value

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
        carried_sources = [
            operation.get_operand(position)
            for position in range(operation.get_num_operands())
        ]
        carried = [self.static_values.get(source.id()) for source in carried_sources]
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
        condition_operations = [
            self.operations[operation_index]
            for operation_index in self.block_operations.get(before_block_id, [])
            if self.operations[operation_index].get_name() == "scf.condition"
        ]
        yield_operations = [
            self.operations[operation_index]
            for operation_index in self.block_operations.get(after_block_id, [])
            if self.operations[operation_index].get_name() == "scf.yield"
        ]
        if len(condition_operations) != 1 or len(yield_operations) != 1:
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.while regions must contain one condition and one yield terminator",
                {
                    "operation_index": index,
                    "conditions": len(condition_operations),
                    "yields": len(yield_operations),
                },
            )
        condition_operation = condition_operations[0]
        yield_operation = yield_operations[0]
        seen: set[tuple[Any, ...]] = set()
        iteration = 0
        while True:
            fingerprint = tuple(repr(value) for value in carried)
            if fingerprint in seen:
                raise CaptureError(
                    "TRITON_NONFINITE_CONTROL_FLOW",
                    "scf.while repeats a statically resolved carried state",
                    {"operation_index": index, "iteration": iteration},
                )
            seen.add(fingerprint)
            for position, (value, source) in enumerate(zip(carried, carried_sources)):
                argument = before_block.get_argument(position)
                self._forward_value(source, argument)
                if value is not None:
                    self.static_values[argument.id()] = value
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
            forwarded_sources = [
                condition_operation.get_operand(position)
                for position in range(1, condition_operation.get_num_operands())
            ]
            if condition not in {0, 1, -1, False, True}:
                raise CaptureError(
                    "TRITON_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "scf.while condition is not statically Boolean",
                    {"operation_index": index, "condition": condition},
                )
            if not bool(condition):
                carried = forwarded
                carried_sources = forwarded_sources
                break
            self.static_expansions += 1
            for position, (value, source) in enumerate(zip(forwarded, forwarded_sources)):
                argument = after_block.get_argument(position)
                self._forward_value(source, argument)
                if value is not None:
                    self.static_values[argument.id()] = value
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
            carried_sources = [
                yield_operation.get_operand(position)
                for position in range(yield_operation.get_num_operands())
            ]
            iteration += 1
        if operation.get_num_results() != len(carried):
            raise CaptureError(
                "TRITON_CONTROL_FLOW_UNRESOLVED",
                "scf.while result count does not match carried values",
                {"operation_index": index},
            )
        for position, (value, source) in enumerate(zip(carried, carried_sources)):
            result = operation.get_result(position)
            self._forward_value(source, result)
            if value is not None:
                self.static_values[result.id()] = value

    def _execute_operation(self, operation: Any, index: int) -> None:
        name = operation.get_name()
        if name in self._CONTAINERS:
            return
        if name not in {
            "arith.constant",
            "tt.make_range",
            "ttg.global_scratch_alloc",
            "ttg.local_alloc",
        }:
            for position in range(operation.get_num_results()):
                self._clear_value_facts(operation.get_result(position).id())
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
                self._membar_before_shared_write(self.regions[region]["allocation_id"], index)
                self._emit("triton.mem.write", index, region=region)
                self.pending_shared_write_allocations.add(
                    self.regions[region]["allocation_id"]
                )
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name in self._DECLARATIONS:
            self._execute_declaration(operation, index)
            self.elided_nodes += 1
            return
        if name.startswith(("arith.", "math.")) or name in self._PURE_OPERATIONS:
            self._capture_wgmma_result_reads(operation, index)
            self._evaluate_pure(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name == "tt.bitcast":
            self._capture_wgmma_result_reads(operation, index)
            self._execute_bitcast(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name == "ub.poison":
            self._execute_poison(operation, index)
            self.elided_nodes += 1
            return
        if name == "tt.elementwise_inline_asm":
            if operation.get_bool_attr("pure") is False:
                self._capture_predicated_bar_sync_inline_asm(operation, index)
                self.emitting_nodes += 1
                return
            self._capture_wgmma_result_reads(operation, index)
            self._execute_value_only_pure(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name in self._VALUE_ONLY_PURE_OPERATIONS:
            self._capture_wgmma_result_reads(operation, index)
            self._execute_value_only_pure(operation, index)
            if len(self.output_operations) > before:
                self.emitting_nodes += 1
            else:
                self.elided_nodes += 1
            return
        if name not in self._SUPPORTED:
            raise CaptureError(
                "TRITON_OPERATION_UNSUPPORTED",
                "TTGIR operation has no sound capture rule",
                {"operation": name, "operation_index": index},
            )
        if name in {
            "ttng.tensormap_create",
            "ttng.tensormap_fenceproxy_acquire",
            "ttng.reinterpret_tensor_descriptor",
        }:
            self._capture_tensormap(operation, index)
            self.elided_nodes += 1
            return
        if name == "ttg.local_dealloc":
            if (
                operation.get_num_operands() != 1
                or operation.get_num_results() != 0
                or operation.get_operand(0).id() not in self.memdesc_regions
            ):
                raise CaptureError(
                    "TRITON_MEMDESC_PROVENANCE_UNRESOLVED",
                    "ttg.local_dealloc requires one exact memdesc allocation",
                    {"operation_index": index},
                )
            self.elided_nodes += 1
            return
        if name == "ttg.barrier":
            self._capture_collective_barrier(operation, index)
            self.emitting_nodes += 1
            return
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
        elif name == "ttg.local_store":
            self._capture_local_store(operation, index)
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
                attributes = self.unit_attributes.get(index, set())
                implicit_wait = "implicit_wgmma_wait_group_0" in attributes
                if attributes - {"implicit_wgmma_wait_group_0"}:
                    raise CaptureError(
                        "TRITON_BINDING_EVIDENCE_INVALID",
                        "barrier arrival binding evidence contains unknown unit attributes",
                        {"operation_index": index, "attributes": sorted(attributes)},
                    )
                if self.current_actor in self.pending_wgmma_owners:
                    if not implicit_wait:
                        raise CaptureError(
                            "TRITON_BINDING_EVIDENCE_REQUIRED",
                            "barrier arrival after WGMMA requires final-lowering wait evidence",
                            {"operation_index": index},
                        )
                    wgmma_resource = self._wgmma_resource()
                    for completion_kind in ("OPERANDS_RELEASED", "RESULT_READY"):
                        self._emit(
                            "triton.wgmma.wait_group",
                            index,
                            resource=wgmma_resource,
                            group_owner=self.current_actor,
                            wait_group_count=0,
                            completion_kind=completion_kind,
                        )
                    self.pending_wgmma_owners.discard(self.current_actor)
                elif implicit_wait:
                    raise CaptureError(
                        "TRITON_BINDING_EVIDENCE_INVALID",
                        "implicit WGMMA wait evidence has no pending WGMMA group",
                        {"operation_index": index},
                    )
                count = operation.get_int_attr("count")
                operand_count = operation.get_num_operands()
                if count is None or int(count) <= 0 or operand_count not in {1, 2}:
                    raise CaptureError(
                        "TRITON_MBARRIER_ARRIVE_UNRESOLVED",
                        "ttng.arrive_barrier requires static count and an optional predicate",
                    )
                predicate_value: Any = True
                if operand_count == 2:
                    predicate_operand = operation.get_operand(1)
                    predicate_value = self._constant(predicate_operand, "arrive predicate")
                    if str(predicate_operand.get_type()) != "i1" or predicate_value not in {0, 1, -1}:
                        raise CaptureError(
                            "TRITON_PREDICATE_UNRESOLVED",
                            "ttng.arrive_barrier predicate must be a scalar i1 constant",
                            {"value": str(predicate_value), "type": str(predicate_operand.get_type())},
                        )
                generation = self.arrival_rank.get(barrier_key, 0)
                if predicate_value:
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
                operand_count = operation.get_num_operands()
                if operand_count < 2:
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier requires a barrier and scalar phase SSA operand",
                    )
                parity = self._constant(operation.get_operand(1), "wait phase")
                if not isinstance(parity, int) or isinstance(parity, bool):
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier phase must be a scalar integer constant",
                        {"value": str(parity)},
                    )
                predicate_value: Any = True
                if operand_count >= 3:
                    predicate_operand = operation.get_operand(2)
                    predicate_value = self._constant(predicate_operand, "wait predicate")
                    if (
                        str(predicate_operand.get_type()) != "i1"
                        or predicate_value not in {0, 1, -1, False, True}
                    ):
                        raise CaptureError(
                            "TRITON_PREDICATE_UNRESOLVED",
                            "ttng.wait_barrier predicate must be a scalar i1 constant",
                            {
                                "value": str(predicate_value),
                                "type": str(predicate_operand.get_type()),
                            },
                        )
                if not predicate_value:
                    self.elided_nodes += 1
                    return
                parity_bit = parity & 1
                wait_key = (resource, lifecycle, self.current_actor)
                generation = self.wait_rank.get(wait_key, 0)
                priming_wait = (
                    generation == 0
                    and parity_bit == 1
                    and self.arrival_rank.get(barrier_key, 0) == 0
                    and self.tma_expect_rank.get(barrier_key, 0) == 0
                )
                if not priming_wait:
                    self.wait_rank[wait_key] = generation + 1
                    self._emit(
                        "triton.mbarrier.wait_parity",
                        index,
                        resource=resource,
                        lifecycle_epoch=lifecycle,
                        generation=generation,
                        raw_parity=parity_bit,
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
        self._require_resolved_external_aliases()
        self.active_cfg_blocks.add(self.entry_block)
        try:
            self._execute_block(self.entry_block, ("root",))
        finally:
            self.active_cfg_blocks.remove(self.entry_block)
        self._require_resolved_external_aliases()
        if self.pending_proxy_fence is not None:
            raise CaptureError(
                "TRITON_PROXY_FENCE_UNRESOLVED",
                "fence_async_shared has no statically resolved async consumer",
                {"operation_index": self.pending_proxy_fence},
            )
        if self.final_ptx_proxy_fence_pending:
            raise CaptureError(
                "TRITON_FINAL_PTX_PROXY_FENCE_UNRESOLVED",
                "final-PTX proxy fence has no statically resolved WGMMA consumer",
                {"evidence": self.final_ptx_evidence},
            )
        triton_root = Path(self.triton.__file__).resolve().parents[1]
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "profile": {
                "id": "sm90-single-cta-static-parametric-v1",
                "source_ir_sha256": self.source_ir_sha256,
                "compiler_ir_stage": "post-ttgir-pipeline",
                "target_rule": "triton-cuda-90-selected-cta-grid-v1",
                "final_ptx_evidence": self.final_ptx_evidence,
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
                "launch_grid": list(self.launch_grid),
                "selected_cta": list(self.selected_cta),
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
                "alias_evidence": (
                    "source-bound-pointer-allocations-v1"
                    if self.pointer_argument_allocations
                    else "single-external-base-v1"
                ),
                "participant_evidence": (
                    "ttgir-sha-bound-warp-partitions-plus-selected-grid-v1"
                    if len(self.actor_specs) > 1
                    else "ttgir-static-warp-geometry-plus-selected-grid-v1"
                ),
            },
            "actors": [self.actor_specs[key] for key in sorted(self.actor_specs)],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [self.regions[key] for key in sorted(self.regions)],
            "operations": self.output_operations,
            "unsupported_constructs": [],
        }


def _normalize_grid_selection(
    launch_grid: list[int] | tuple[int, ...] | None,
    grid_ctas: int | None,
    selected_cta: list[int] | tuple[int, ...],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if launch_grid is None:
        if isinstance(grid_ctas, bool) or not isinstance(grid_ctas, int) or grid_ctas < 1:
            raise CaptureError(
                "TRITON_GRID_EVIDENCE_INVALID",
                "capture requires a finite positive launch grid",
                {"grid_ctas": grid_ctas},
            )
        launch_grid = [grid_ctas]
    if (
        not isinstance(launch_grid, (list, tuple))
        or not 1 <= len(launch_grid) <= 3
        or any(isinstance(extent, bool) or not isinstance(extent, int) or extent < 1 for extent in launch_grid)
    ):
        raise CaptureError(
            "TRITON_GRID_EVIDENCE_INVALID",
            "launch grid must contain one to three positive integer extents",
            {"launch_grid": launch_grid},
        )
    normalized_grid = tuple((*launch_grid, *(1 for _ in range(3 - len(launch_grid)))))
    total_ctas = normalized_grid[0] * normalized_grid[1] * normalized_grid[2]
    if grid_ctas is not None and grid_ctas != total_ctas:
        raise CaptureError(
            "TRITON_GRID_EVIDENCE_INVALID",
            "launch grid dimensions disagree with the recorded total CTA count",
            {"launch_grid": list(normalized_grid), "grid_ctas": grid_ctas},
        )
    if (
        not isinstance(selected_cta, (list, tuple))
        or len(selected_cta) != 3
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected_cta)
    ):
        raise CaptureError(
            "TRITON_CTA_SELECTION_INVALID",
            "selected CTA must contain three nonnegative integer coordinates",
            {"selected_cta": selected_cta},
        )
    normalized_cta = tuple(selected_cta)
    if any(index >= extent for index, extent in zip(normalized_cta, normalized_grid)):
        raise CaptureError(
            "TRITON_CTA_SELECTION_INVALID",
            "selected CTA is outside the finite launch grid",
            {"launch_grid": list(normalized_grid), "selected_cta": list(normalized_cta)},
        )
    return normalized_grid, normalized_cta


def capture_file(
    path: Path,
    source: str,
    command: str,
    grid_ctas: int | None = None,
    binding_evidence_path: Path | None = None,
    launch_grid: list[int] | tuple[int, ...] | None = None,
    selected_cta: list[int] | tuple[int, ...] = (0, 0, 0),
    final_ptx_path: Path | None = None,
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
    final_ptx_evidence = _final_ptx_proxy_fence_evidence(final_ptx_path)
    normalized_grid, normalized_cta = _normalize_grid_selection(
        launch_grid,
        grid_ctas,
        selected_cta,
    )
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
        required_evidence = {
            "evidence_version",
            "source_ir_sha256",
            "unit_attributes",
            "array_attributes",
        }
        allowed_evidence = required_evidence | {
            "pointer_allocations", "scalar_values", "allocation_values"
        }
        pointer_allocations = evidence.get("pointer_allocations", {}) if isinstance(evidence, dict) else {}
        scalar_values = evidence.get("scalar_values", {}) if isinstance(evidence, dict) else {}
        allocation_values = evidence.get("allocation_values", {}) if isinstance(evidence, dict) else {}
        if (
            not isinstance(evidence, dict)
            or not required_evidence <= set(evidence) <= allowed_evidence
            or evidence.get("evidence_version") != "triton-binding-evidence-v1"
            or evidence.get("source_ir_sha256") != source_digest
            or not isinstance(evidence.get("unit_attributes"), dict)
            or not isinstance(evidence.get("array_attributes"), dict)
            or not isinstance(pointer_allocations, dict)
            or not isinstance(scalar_values, dict)
            or not isinstance(allocation_values, dict)
            or any(
                not str(index).isdigit()
                or not isinstance(value, int)
                or isinstance(value, bool)
                for index, value in scalar_values.items()
            )
            or any(
                not isinstance(allocation, str)
                or not allocation
                or not isinstance(value, dict)
                or set(value) != {"kind", "value"}
                or value.get("kind") != "CONSTANT"
                or not isinstance(value.get("value"), int)
                or isinstance(value.get("value"), bool)
                for allocation, value in allocation_values.items()
            )
            or any(
                not str(index).isdigit()
                or not isinstance(allocation, str)
                or not allocation
                for index, allocation in pointer_allocations.items()
            )
            or not set(allocation_values).issubset(
                set(pointer_allocations.values())
            )
            or any(
                not str(index).isdigit()
                or not isinstance(attributes, list)
                or any(not isinstance(attribute, str) for attribute in attributes)
                for index, attributes in evidence.get("unit_attributes", {}).items()
            )
            or any(
                not str(index).isdigit() or not isinstance(attributes, dict)
                for index, attributes in evidence.get("array_attributes", {}).items()
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
            normalized_grid,
            normalized_cta,
            binding_evidence,
            final_ptx_evidence,
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
        type=int,
        help="legacy static total CTA count; interpreted as a one-dimensional launch grid",
    )
    parser.add_argument(
        "--launch-grid",
        type=int,
        nargs="+",
        help="one to three exact launch-grid extents",
    )
    parser.add_argument(
        "--selected-cta",
        type=int,
        nargs=3,
        default=(0, 0, 0),
        metavar=("X", "Y", "Z"),
        help="CTA coordinate to analyze; defaults to 0 0 0",
    )
    parser.add_argument(
        "--binding-evidence",
        type=Path,
        help="SHA-bound unit/array attributes hidden by Triton Python bindings",
    )
    parser.add_argument(
        "--final-ptx",
        type=Path,
        help="same-compilation final PTX used to recover backend-inserted proxy fences",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(
            args.input,
            args.source,
            " ".join(sys.argv),
            grid_ctas=args.grid_ctas,
            binding_evidence_path=args.binding_evidence,
            launch_grid=args.launch_grid,
            selected_cta=args.selected_cta,
            final_ptx_path=args.final_ptx,
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
