#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import product
import subprocess
import sys
from pathlib import Path
from typing import Any

from tilelang_static import (
    cp_async_transfer_bytes,
    dtype_bit_width,
    dtype_byte_width,
    is_pure_payload_call,
    peel_static_byte_offsets,
    valid_set_max_nreg,
    vector_values,
)

SNAPSHOT_VERSION = "tilelang-snapshot-v2"
ADAPTER_VERSION = "tilelang-pgir-v2"


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


def _stable_path(path: tuple[Any, ...]) -> str:
    encoded = "/".join(map(str, path)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


class TileLangCapture:
    def __init__(
        self,
        tvm: Any,
        tirx: Any,
        module: Any,
        source: str,
        command: str,
        source_ir_sha256: str,
        descriptor_evidence: dict[str, Any] | None = None,
    ):
        self.tvm = tvm
        self.tirx = tirx
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        self.descriptor_evidence = descriptor_evidence or {}
        functions = [func for func in module.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(functions) != 1:
            raise CaptureError(
                "TILELANG_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one lowered TileLang PrimFunc",
                {"primfuncs": len(functions)},
            )
        self.func = functions[0]
        self.thread_extents = self._thread_extents()
        self.cta_threads = self.thread_extents.get("threadIdx.x")
        if self.cta_threads is None or not 1 <= self.cta_threads <= 1024:
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "threadIdx.x extent must be a static value in [1, 1024]",
                {"thread_extents": self.thread_extents},
            )
        for dimension in ("threadIdx.y", "threadIdx.z"):
            if self.thread_extents.get(dimension, 1) != 1:
                raise CaptureError(
                    "TILELANG_MULTIDIMENSIONAL_THREADS_UNSUPPORTED",
                    "initial TileLang capture requires unit threadIdx.y/z extents",
                    {"thread_extents": self.thread_extents},
                )
        for dimension in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
            if self.thread_extents.get(dimension, 1) != 1:
                raise CaptureError(
                    "TILELANG_MULTI_CTA_UNSUPPORTED",
                    "PGIR-v1 TileLang capture requires one statically selected CTA",
                    {"thread_extents": self.thread_extents},
                )
        self.actor_ids = tuple(f"lane:{lane:04d}" for lane in range(self.cta_threads))
        call_names: set[str] = set()
        warpgroup_register_data: set[Any] = set()

        def record_call(node: Any) -> None:
            if isinstance(node, tirx.Call):
                name = getattr(node.op, "name", "")
                call_names.add(name)
                if name == "tl.warpgroup_fence_operand" and len(node.args) == 4:
                    warpgroup_register_data.add(node.args[1])

        tirx.stmt_functor.post_order_visit(self.func.body, record_call)
        self.uses_warpgroup = any(
            name.startswith("tl.warpgroup_") or name == "tl.ptx_wgmma_ss"
            for name in call_names
        )
        self.warpgroup_register_data = warpgroup_register_data
        if self.uses_warpgroup and self.cta_threads % 128:
            raise CaptureError(
                "TILELANG_WGMMA_PARTICIPANTS_UNRESOLVED",
                "warpgroup operations require a CTA size divisible by 128",
                {"cta_threads": self.cta_threads},
            )
        self.warpgroup_ids = tuple(
            f"warpgroup:{group:02d}"
            for group in range(self.cta_threads // 128)
        ) if self.uses_warpgroup else ()
        self.all_actor_ids = (*self.actor_ids, *self.warpgroup_ids)
        self.actor_sequences: dict[str, int] = {}
        self.warpgroup_sync_events: dict[tuple[str, str], str] = {}
        self.wgmma_descriptors: dict[Any, str] = {}
        self.wgmma_group_seq: dict[str, int] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, dict[str, Any]] = {}
        self.operations: list[dict[str, Any]] = []
        self.static_expansions = 0
        self.executable_nodes_seen = 0
        self.classified_nodes = 0
        self.emitting_nodes = 0
        self.elided_nodes = 0
        self.allocation_ids: dict[Any, str] = {}
        self.used_allocation_ids: set[str] = set()
        self.global_allocation_keys: set[Any] = set()
        attrs = dict(self.func.attrs) if self.func.attrs else {}
        self.noalias = bool(attrs.get("tirx.noalias", False))
        target = attrs.get("target")
        self.target_text = str(target)
        target_kind = getattr(getattr(target, "kind", None), "name", None)
        target_attrs = dict(getattr(target, "attrs", {})) if target is not None else {}
        target_arch = str(target_attrs.get("arch", ""))
        target_warp_size = int(target_attrs.get("thread_warp_size", 0))
        if target_kind != "cuda" or target_arch not in {"sm_90", "sm_90a"} or target_warp_size != 32:
            raise CaptureError(
                "TILELANG_TARGET_UNSUPPORTED",
                "TileLang capture supports only CUDA SM90 targets with 32-lane warps",
                {
                    "target": self.target_text,
                    "kind": target_kind,
                    "arch": target_arch,
                    "warp_size": target_warp_size,
                },
            )
        self.target_rule = f"tilelang-cuda-{target_arch}-v1"

    def classify_leaf(self, operation_count_before: int) -> None:
        self.executable_nodes_seen += 1
        self.classified_nodes += 1
        if len(self.operations) > operation_count_before:
            self.emitting_nodes += 1
        else:
            self.elided_nodes += 1

    def allocation_id(self, data: Any, scope: str, actor: str, label: str) -> str:
        key: Any = (data, actor) if scope.startswith("local") else data
        if key in self.allocation_ids:
            return self.allocation_ids[key]
        if scope == "global":
            if self.global_allocation_keys and not self.noalias:
                raise CaptureError(
                    "TILELANG_ALIAS_UNRESOLVED",
                    "multiple global pointer bases require tirx.noalias evidence",
                    {"allocation": label},
                )
            self.global_allocation_keys.add(key)
        preferred = f"{actor}:{label}" if scope.startswith("local") else label
        identifier = preferred
        suffix = 1
        while identifier in self.used_allocation_ids:
            identifier = f"{preferred}:{suffix}"
            suffix += 1
        self.allocation_ids[key] = identifier
        self.used_allocation_ids.add(identifier)
        return identifier

    def allocation_for_buffer(self, buffer: Any, actor: str) -> str:
        scope = str(buffer.scope())
        label = getattr(buffer.data, "name", None) or buffer.name
        return self.allocation_id(buffer.data, scope, actor, label)

    def _thread_extents(self) -> dict[str, int]:
        attrs = dict(self.func.attrs) if self.func.attrs else {}
        raw = attrs.get("thread_extent")
        if raw is None or not hasattr(raw, "items"):
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "lowered PrimFunc lacks static thread_extent metadata",
            )
        result: dict[str, int] = {}
        for key, value in raw.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError) as error:
                raise CaptureError(
                    "TILELANG_PARTICIPANTS_UNRESOLVED",
                    "thread extent is not an integer",
                    {"dimension": str(key), "value": str(value)},
                ) from error
        return result

    def capture(self) -> dict[str, Any]:
        for lane in range(self.cta_threads):
            walker = LaneWalker(self, lane)
            walker.walk(self.func.body, ("root",))
            walker.finalize()
        has_mbarrier = any(
            resource["kind"] == "mbarrier" for resource in self.resources.values()
        )
        if has_mbarrier and not any(
            operation["op"] == "tilelang.mbarrier.init"
            for operation in self.operations
        ):
            raise CaptureError(
                "TILELANG_MBARRIER_INIT_MISSING",
                "captured mbarrier operations require an explicit static initialization",
            )
        tilelang_root = Path(sys.modules["tilelang"].__file__).resolve().parents[1]
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "profile": {
                "id": "sm90-single-cta-static-v1",
                "source_ir_sha256": self.source_ir_sha256,
                "compiler_ir_stage": "post-inject-fence-proxy",
                "target_rule": self.target_rule,
            },
            "producer": {
                "dsl_version": str(getattr(sys.modules["tilelang"], "__version__", "unavailable")),
                "dsl_commit": _git_revision(tilelang_root),
                "backend_commit": str(getattr(self.tvm, "__version__", "tilelang-tirx")),
                "adapter_version": ADAPTER_VERSION,
                "command": self.command,
            },
            "target": {
                "architecture": "sm90",
                "warp_size": 32,
                "cta_threads": self.cta_threads,
                "shared_memory_bytes": 65536,
                "grid_scope": "single_cta",
            },
            "extraction": {
                "complete": True,
                "stage": "post-inject-fence-proxy",
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
                "events_emitted": len(self.operations),
                "allocation_identity_evidence": "tir-buffer-data-object-and-actor-identity-v1",
                "alias_evidence": (
                    "tirx.noalias-plus-sha-bound-tma-descriptors-v1"
                    if self.descriptor_evidence
                    else "tirx.noalias-v1" if self.noalias else "single-global-base-v1"
                ),
                "participant_evidence": "primfunc-thread-extent-lane-enumeration-v1",
            },
            "actors": [
                *[
                    {"id": actor, "kind": "LANE", "participants": [lane]}
                    for lane, actor in enumerate(self.actor_ids)
                ],
                *[
                    {
                        "id": actor,
                        "kind": "WARPGROUP",
                        "participants": list(range(group * 128, (group + 1) * 128)),
                    }
                    for group, actor in enumerate(self.warpgroup_ids)
                ],
            ],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [self.regions[key] for key in sorted(self.regions)],
            "operations": sorted(
                self.operations,
                key=lambda operation: (operation["actor"], operation["seq"], operation["id"]),
            ),
            "unsupported_constructs": [],
        }

    def mbarrier(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> str:
        allocation, base_offset = evaluator.buffer_allocation(buffer)
        offset, extent = evaluator.byte_interval(buffer, indices)
        offset += base_offset
        if "barrier" not in evaluator.buffer_scope(buffer):
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier operand is not in shared.barrier storage",
                {"buffer": buffer.name, "scope": evaluator.buffer_scope(buffer)},
            )
        if extent != 8 or offset % 8:
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier storage must be one aligned eight-byte object",
                {"buffer": buffer.name, "offset": offset, "extent": extent},
            )
        identifier = f"mbarrier:{allocation}:{offset}:{extent}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "mbarrier",
            "participants": list(self.all_actor_ids),
        })
        return identifier

    def cta_barrier(self, path: tuple[Any, ...]) -> str:
        identifier = f"cta:{_stable_path(path)}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "cta_barrier",
            "participants": list(self.actor_ids),
            "capacity": self.cta_threads,
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
        return identifier

    def async_group(self, actor: str) -> str:
        identifier = f"cp_async:{actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [actor],
            "capacity": 8,
            "protocol": "CP_ASYNC",
            "scope": "CTA",
        })
        return identifier

    def tma_store_group(self, owner: str) -> str:
        identifier = f"tma_store:{owner}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [owner],
            "capacity": 8,
            "protocol": "TMA_STORE",
            "scope": "CTA",
        })
        return identifier

    def warpgroup_for_lane(self, lane: int) -> str:
        group = lane // 128
        if group >= len(self.warpgroup_ids):
            raise CaptureError(
                "TILELANG_WGMMA_PARTICIPANTS_UNRESOLVED",
                "lane is outside the statically recovered warpgroup topology",
                {"lane": lane},
            )
        return self.warpgroup_ids[group]

    def wgmma_resource(self, owner: str) -> str:
        identifier = f"wgmma:{owner}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [owner],
            "capacity": 8,
            "protocol": "WGMMA",
            "scope": "CTA",
        })
        return identifier

    def tma_segments(
        self,
        descriptor: Any,
        coordinates: list[int],
        destination: str,
    ) -> list[tuple[str, str, int]]:
        name = getattr(descriptor, "name", None)
        evidence = self.descriptor_evidence.get(name)
        if not isinstance(evidence, dict) or set(evidence) != {
            "allocation_id",
            "element_bytes",
            "shape",
            "strides",
            "tile_shape",
            "coordinate_order",
        }:
            raise CaptureError(
                "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                "TMA descriptor requires exact shape/stride evidence bound to this IR",
                {"descriptor": name},
            )
        allocation = evidence["allocation_id"]
        element_bytes = evidence["element_bytes"]
        shape = evidence["shape"]
        strides = evidence["strides"]
        tile_shape = evidence["tile_shape"]
        coordinate_order = evidence["coordinate_order"]
        rank = len(shape) if isinstance(shape, list) else -1
        if (
            not isinstance(allocation, str)
            or not allocation
            or isinstance(element_bytes, bool)
            or not isinstance(element_bytes, int)
            or element_bytes <= 0
            or rank <= 0
            or not isinstance(strides, list)
            or not isinstance(tile_shape, list)
            or len(strides) != rank
            or len(tile_shape) != rank
            or not isinstance(coordinate_order, list)
            or sorted(coordinate_order) != list(range(rank))
            or len(coordinates) < rank
            or any(not isinstance(value, int) or value <= 0 for value in shape)
            or any(not isinstance(value, int) or value <= 0 for value in strides)
            or any(not isinstance(value, int) or value <= 0 for value in tile_shape)
            or any(value != 0 for value in coordinates[rank:])
        ):
            raise CaptureError(
                "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                "TMA descriptor evidence or coordinate rank is malformed",
                {"descriptor": name, "coordinates": coordinates},
            )
        coordinates = [coordinates[index] for index in coordinate_order]
        for coordinate, tile, dimension in zip(coordinates, tile_shape, shape):
            if coordinate < 0 or coordinate + tile > dimension:
                raise CaptureError(
                    "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                    "TMA tile lies outside the evidenced descriptor shape",
                    {"descriptor": name, "coordinates": coordinates},
                )
        destination_region = self.regions[destination]
        elements = 1
        for dimension in tile_shape:
            elements *= dimension
        if elements * element_bytes != destination_region["extent"]:
            raise CaptureError(
                "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                "TMA tile footprint differs from destination access interval",
                {
                    "descriptor": name,
                    "tile_bytes": elements * element_bytes,
                    "destination_bytes": destination_region["extent"],
                },
            )
        runs: list[list[int]] = []
        destination_offset = destination_region["offset"]
        for logical_index in product(*(range(dimension) for dimension in tile_shape)):
            source_element = sum(
                (coordinate + index) * stride
                for coordinate, index, stride in zip(coordinates, logical_index, strides)
            )
            source_offset = source_element * element_bytes
            if (
                runs
                and source_offset == runs[-1][0] + runs[-1][2]
                and destination_offset == runs[-1][1] + runs[-1][2]
            ):
                runs[-1][2] += element_bytes
            else:
                runs.append([source_offset, destination_offset, element_bytes])
            destination_offset += element_bytes
        return [
            (
                self.region(allocation, "global", source_offset, extent),
                self.region(
                    destination_region["allocation_id"],
                    "shared",
                    destination_offset,
                    extent,
                ),
                extent,
            )
            for source_offset, destination_offset, extent in runs
        ]

    def region(self, allocation: str, scope: str, offset: int, extent: int) -> str:
        address_space = {
            "shared": "SHARED",
            "shared.dyn": "SHARED",
            "global": "GLOBAL",
            "local": "REGISTER",
        }.get(scope)
        if address_space is None and scope.startswith("local"):
            address_space = "REGISTER"
        if address_space is None:
            raise CaptureError(
                "TILELANG_ADDRESS_SPACE_UNSUPPORTED",
                "buffer scope has no sound PGIR mapping",
                {"allocation": allocation, "scope": scope},
            )
        if offset < 0 or extent <= 0:
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "memory region requires a nonnegative offset and positive extent",
                {"allocation": allocation, "offset": offset, "extent": extent},
            )
        identifier = f"region:{address_space.lower()}:{allocation}:{offset}:{extent}"
        self.regions.setdefault(identifier, {
            "id": identifier,
            "allocation_id": allocation,
            "address_space": address_space,
            "offset": offset,
            "extent": extent,
        })
        return identifier

    def memory_regions(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> list[str]:
        allocation, base_offset = evaluator.buffer_allocation(buffer)
        return [
            self.region(
                allocation,
                evaluator.buffer_scope(buffer),
                base_offset + offset,
                extent,
            )
            for offset, extent in evaluator.byte_intervals(buffer, indices)
        ]

    def memory_region(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> str:
        regions = self.memory_regions(buffer, indices, evaluator)
        if len(regions) != 1:
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "protocol operand requires one exact contiguous region",
                {"buffer": buffer.name, "regions": regions},
            )
        return regions[0]


class LaneWalker:
    def __init__(self, capture: TileLangCapture, lane: int):
        self.capture = capture
        self.tvm = capture.tvm
        self.tirx = capture.tirx
        self.lane = lane
        self.actor = capture.actor_ids[lane]
        self.bindings: dict[Any, int] = {}
        self.pointer_aliases: dict[Any, tuple[Any, int]] = {}
        self.sequence = 0
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}
        self.cp_group_seq = 0
        self.cp_staged = 0
        self.tma_store_group_seq = 0
        self.pending_proxy: dict[str, Any] | None = None
        self.current_ir_operation = "unknown"
        self.wgmma_accumulator_region: str | None = None
        self.wgmma_fence_seen = False
        self.wgmma_arrived = False
        self.active_warpgroup_dependency: str | None = None
        self.last_lane_event_id: str | None = None

    def warpgroup_owner(self) -> str:
        return self.capture.warpgroup_for_lane(self.lane)

    def is_warpgroup_leader(self) -> bool:
        return self.lane % 128 == 0

    def warpgroup_sync_key(self, path: tuple[Any, ...]) -> tuple[str, str]:
        return self.warpgroup_owner(), _stable_path(path)

    def publish_warpgroup_dependency(self, event_id: str, path: tuple[Any, ...]) -> None:
        self.capture.warpgroup_sync_events[self.warpgroup_sync_key(path)] = event_id
        self.active_warpgroup_dependency = event_id

    def observe_warpgroup_dependency(self, path: tuple[Any, ...]) -> None:
        key = self.warpgroup_sync_key(path)
        if key not in self.capture.warpgroup_sync_events:
            raise CaptureError(
                "TILELANG_WARPGROUP_ORDER_UNRESOLVED",
                "nonleader lane reached warpgroup synchronization before its leader evidence",
                {"lane": self.lane, "path": list(path)},
            )
        self.active_warpgroup_dependency = self.capture.warpgroup_sync_events[key]

    def finalize(self) -> None:
        if self.pending_proxy is not None:
            raise CaptureError(
                "TILELANG_PROXY_TARGET_UNRESOLVED",
                "fence_proxy_async has no immediately following supported async issue",
                {"event": self.pending_proxy["id"]},
            )

    def emit(self, op: str, path: tuple[Any, ...], **fields: Any) -> None:
        ir_operation = fields.pop("_ir_operation", self.current_ir_operation)
        recognizer_rule = fields.pop(
            "_recognizer_rule",
            f"{ir_operation}->{op}:v1",
        )
        actor = fields.pop("_actor", self.actor)
        if self.pending_proxy is not None and op not in {
            "tilelang.cp_async.issue",
            "tilelang.tma_load.issue",
            "tilelang.tma_store.issue",
            "tilelang.wgmma.issue",
        }:
            raise CaptureError(
                "TILELANG_PROXY_TARGET_UNRESOLVED",
                "fence_proxy_async must be followed immediately by a supported async issue",
                {"event": self.pending_proxy["id"], "next_operation": op},
            )
        if actor == self.actor and self.active_warpgroup_dependency is not None:
            fields.setdefault("depends_on", [self.active_warpgroup_dependency])
        sequence = self.capture.actor_sequences.get(actor, 0)
        if actor != self.actor and sequence == 0 and self.last_lane_event_id is not None:
            fields.setdefault("depends_on", [self.last_lane_event_id])
        identifier = f"tl:{actor}:e{sequence:05d}:{_stable_path((*path, op))}"
        self.capture.operations.append({
            "id": identifier,
            "op": op,
            "actor": actor,
            "seq": sequence,
            "evidence": {
                "source_location": "/".join(map(str, path)),
                "activation_path": [str(item) for item in path],
                "recognizer_rule": recognizer_rule,
                "ir_operation": ir_operation,
            },
            **fields,
        })
        self.capture.actor_sequences[actor] = sequence + 1
        if actor == self.actor:
            self.last_lane_event_id = identifier

    def eval_int(self, expression: Any) -> int | None:
        if isinstance(expression, self.tirx.IntImm):
            return int(expression.value)
        if isinstance(expression, self.tirx.Var):
            return self.bindings.get(expression)
        try:
            replacements = {
                variable: self.tirx.IntImm(variable.dtype, value)
                for variable, value in self.bindings.items()
            }
            substituted = self.tirx.stmt_functor.substitute(expression, replacements)
            simplified = self.capture.tvm.arith.Analyzer().simplify(substituted)
            if isinstance(simplified, self.tirx.IntImm):
                return int(simplified.value)
        except Exception:
            pass
        name = type(expression).__name__
        if name == "Cast":
            return self.eval_int(expression.value)
        if name == "Not":
            value = self.eval_int(expression.a)
            return None if value is None else int(not value)
        if name == "Select":
            condition = self.eval_int(expression.condition)
            if condition is None:
                return None
            return self.eval_int(expression.true_value if condition else expression.false_value)
        if name == "Let":
            value = self.eval_int(expression.value)
            if value is None:
                return None
            old = self.bindings.get(expression.var)
            self.bindings[expression.var] = value
            result = self.eval_int(expression.body)
            if old is None:
                self.bindings.pop(expression.var, None)
            else:
                self.bindings[expression.var] = old
            return result

        def trunc_div(left: int, right: int) -> int:
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient

        binary = {
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b,
            "Div": trunc_div,
            "FloorDiv": lambda a, b: a // b,
            "Mod": lambda a, b: a - trunc_div(a, b) * b,
            "FloorMod": lambda a, b: a % b,
            "Min": min,
            "Max": max,
            "EQ": lambda a, b: int(a == b),
            "NE": lambda a, b: int(a != b),
            "LT": lambda a, b: int(a < b),
            "LE": lambda a, b: int(a <= b),
            "GT": lambda a, b: int(a > b),
            "GE": lambda a, b: int(a >= b),
            "And": lambda a, b: int(bool(a) and bool(b)),
            "Or": lambda a, b: int(bool(a) or bool(b)),
        }
        function = binary.get(name)
        if function is not None:
            left = self.eval_int(expression.a)
            right = self.eval_int(expression.b)
            if left is None or right is None or (name in {"Div", "FloorDiv", "Mod", "FloorMod"} and right == 0):
                return None
            return function(left, right)
        if isinstance(expression, self.tirx.Call):
            call_name = getattr(expression.op, "name", "")
            if call_name == "tl.tl_shuffle_elect" and len(expression.args) == 1:
                thread_extent = self.eval_int(expression.args[0])
                if thread_extent is None or thread_extent < 0:
                    return None
                if thread_extent == 0:
                    return int(self.lane == 0)
                group_extent = ((thread_extent + 31) // 32) * 32
                return int(self.lane % group_extent == 0)
            functions = {
                "tirx.bitwise_xor": lambda a, b: a ^ b,
                "tirx.bitwise_and": lambda a, b: a & b,
                "tirx.bitwise_or": lambda a, b: a | b,
                "tirx.shift_left": lambda a, b: a << b,
                "tirx.shift_right": lambda a, b: a >> b,
            }
            function = functions.get(call_name)
            if function is not None and len(expression.args) == 2:
                left = self.eval_int(expression.args[0])
                right = self.eval_int(expression.args[1])
                if left is not None and right is not None:
                    if call_name in {"tirx.shift_left", "tirx.shift_right"} and right < 0:
                        return None
                    return function(left, right)
        return None

    def eval_pointer(self, expression: Any) -> tuple[Any, int] | None:
        if isinstance(expression, self.tirx.Var):
            return self.pointer_aliases.get(expression, (expression, 0))
        if (
            isinstance(expression, self.tirx.Call)
            and getattr(expression.op, "name", "") == "tirx.handle_add_byte_offset"
            and len(expression.args) == 2
        ):
            base = self.eval_pointer(expression.args[0])
            byte_offset = self.eval_int(expression.args[1])
            if base is None or byte_offset is None:
                return None
            return base[0], base[1] + byte_offset
        return None

    def buffer_allocation(self, buffer: Any) -> tuple[str, int]:
        resolved = self.eval_pointer(buffer.data)
        if resolved is None:
            raise CaptureError(
                "TILELANG_ALIAS_UNRESOLVED",
                "buffer data pointer cannot be reduced to a static allocation and byte offset",
                {"buffer": buffer.name, "data": str(buffer.data)},
            )
        data, byte_offset = resolved
        label = getattr(data, "name", None) or buffer.name
        scope = self.buffer_scope(buffer)
        allocation_actor = self.actor
        if data in self.capture.warpgroup_register_data and scope.startswith("local"):
            owner = self.warpgroup_owner()
            elements = 1
            for dimension in buffer.shape:
                value = self.eval_int(dimension)
                if value is None or value <= 0:
                    raise CaptureError(
                        "TILELANG_WGMMA_ACCUMULATOR_UNRESOLVED",
                        "warpgroup accumulator shape must be static",
                        {"buffer": buffer.name},
                    )
                elements *= value
            per_lane_bytes = elements * self._dtype_bytes(buffer)
            byte_offset += self.lane % 128 * per_lane_bytes
            allocation_actor = owner
        allocation = self.capture.allocation_id(
            data,
            scope,
            allocation_actor,
            label,
        )
        return allocation, byte_offset

    def buffer_scope(self, buffer: Any) -> str:
        raw = buffer.scope()
        return str(raw)

    def _dtype_bytes(self, buffer: Any) -> int:
        width = dtype_byte_width(buffer.dtype)
        if width is None:
            raise CaptureError(
                "TILELANG_DTYPE_UNSUPPORTED",
                "buffer dtype has no static byte width",
                {"buffer": buffer.name, "dtype": str(buffer.dtype)},
            )
        return width

    def byte_intervals(self, buffer: Any, indices: Any) -> list[tuple[int, int]]:
        shape = [self.eval_int(item) for item in buffer.shape]
        elem_offset = self.eval_int(buffer.elem_offset)
        raw_strides = list(buffer.strides)
        strides = [self.eval_int(item) for item in raw_strides]
        index_vectors = [vector_values(index, self.eval_int) for index in indices]
        unresolved = shape + strides + [elem_offset]
        if (
            len(shape) != len(index_vectors)
            or (raw_strides and len(strides) != len(shape))
            or any(value is None for value in unresolved)
            or any(vector is None for vector in index_vectors)
        ):
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "buffer region offset, strides, shape, or indices are symbolic",
                {
                    "buffer": buffer.name,
                    "indices": [str(item) for item in indices],
                    "strides": [str(item) for item in raw_strides],
                    "elem_offset": str(buffer.elem_offset),
                },
            )
        vectors = [vector for vector in index_vectors if vector is not None]
        lane_count = max((len(vector) for vector in vectors), default=1)
        if any(len(vector) not in {1, lane_count} for vector in vectors):
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "vectorized buffer indices have incompatible lane counts",
                {"buffer": buffer.name},
            )
        assert elem_offset is not None
        linears: list[int] = []
        for lane in range(lane_count):
            index_values = [vector[0] if len(vector) == 1 else vector[lane] for vector in vectors]
            for dimension, index in zip(shape, index_values):
                assert dimension is not None
                if dimension <= 0 or not 0 <= index < dimension:
                    raise CaptureError(
                        "TILELANG_REGION_OUT_OF_BOUNDS",
                        "buffer access is outside its static shape",
                        {"buffer": buffer.name, "shape": shape, "indices": index_values},
                    )
            if raw_strides:
                linear = elem_offset + sum(
                    index * int(stride)
                    for index, stride in zip(index_values, strides)
                )
            else:
                linear = 0
                for dimension, index in zip(shape, index_values):
                    assert dimension is not None
                    linear = linear * dimension + index
                linear += elem_offset
            linears.append(linear)
        width = self._dtype_bytes(buffer)
        merged: list[list[int]] = []
        for linear in sorted(set(linears)):
            start = linear * width
            end = start + width
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end - start) for start, end in merged]

    def byte_interval(self, buffer: Any, indices: Any) -> tuple[int, int]:
        intervals = self.byte_intervals(buffer, indices)
        if len(intervals) != 1:
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "protocol operand requires one exact contiguous byte interval",
                {"buffer": buffer.name, "intervals": intervals},
            )
        return intervals[0]

    def _access_ptr_parts(self, expression: Any) -> tuple[Any, int]:
        peeled = peel_static_byte_offsets(
            expression,
            lambda node: (
                getattr(node.op, "name", "")
                if isinstance(node, self.tirx.Call)
                else None
            ),
            self.eval_int,
        )
        if peeled is None:
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async memory operand has a dynamic byte-offset wrapper",
                {"expression": str(expression)},
            )
        core, wrapper_byte_offset = peeled
        if (
            not isinstance(core, self.tirx.Call)
            or getattr(core.op, "name", "") not in {"tirx.tvm_access_ptr", "tir.tvm_access_ptr"}
            or len(core.args) != 5
        ):
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async memory operand must reduce to a five-argument tvm_access_ptr",
                {"expression": str(expression)},
            )
        return core, wrapper_byte_offset

    def access_ptr_dtype(self, expression: Any) -> Any:
        core, _wrapper_byte_offset = self._access_ptr_parts(expression)
        return core.args[0].dtype

    def access_ptr_region(
        self,
        expression: Any,
        required_flag: int,
        logical_elements: int | None = None,
    ) -> tuple[str, int]:
        core, wrapper_byte_offset = self._access_ptr_parts(expression)
        offset = self.eval_int(core.args[2])
        extent = self.eval_int(core.args[3])
        flags = self.eval_int(core.args[4])
        data = core.args[1]
        allocation_label = getattr(data, "name", None)
        if offset is None or extent is None or flags is None or allocation_label is None:
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async access pointer has symbolic offset, extent, flags, or allocation",
                {"expression": str(expression)},
            )
        if flags & required_flag != required_flag:
            raise CaptureError(
                "TILELANG_ACCESS_MODE_MISMATCH",
                "async access pointer flags do not include the required mode",
                {"flags": flags, "required_flag": required_flag},
            )
        bits = dtype_bit_width(core.args[0].dtype)
        scope_values = re.findall(r'"([^"]+)"', str(getattr(data, "type_annotation", "")))
        element_count = extent if logical_elements is None else logical_elements
        if (
            bits is None
            or not scope_values
            or offset < 0
            or element_count <= 0
            or offset * bits % 8
            or element_count * bits % 8
        ):
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async access pointer dtype, address space, or byte footprint is unresolved",
                {"expression": str(expression)},
            )
        byte_offset = offset * bits // 8
        byte_extent = element_count * bits // 8
        scope = scope_values[-1]
        resolved = self.eval_pointer(data)
        if resolved is None:
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async access pointer base alias is unresolved",
                {"expression": str(expression)},
            )
        base, alias_offset = resolved
        base_label = getattr(base, "name", None) or allocation_label
        allocation = self.capture.allocation_id(base, scope, self.actor, base_label)
        region = self.capture.region(
            allocation,
            scope,
            alias_offset + wrapper_byte_offset + byte_offset,
            byte_extent,
        )
        return region, byte_extent

    def _access_enabled(self, access: Any) -> bool:
        predicate = getattr(access, "predicate", None)
        if predicate is None:
            return True
        value = self.eval_int(predicate)
        if value is None:
            raise CaptureError(
                "TILELANG_PREDICATE_UNRESOLVED",
                "memory predicate is not statically resolvable",
                {"predicate": str(predicate)},
            )
        return bool(value)

    def _buffer_loads(self, expression: Any) -> list[Any]:
        loads: list[Any] = []
        unsupported_effects: list[dict[str, Any]] = []
        deferred_errors: list[CaptureError] = []

        def visit(node: Any) -> None:
            if isinstance(node, self.tirx.BufferLoad):
                try:
                    if self._access_enabled(node):
                        loads.append(node)
                except CaptureError as error:
                    deferred_errors.append(error)
                return
            if not isinstance(node, self.tirx.Call):
                return
            call_name = getattr(node.op, "name", "")
            if is_pure_payload_call(call_name, len(node.args)):
                return
            effect = node.op.get_attr("TCallEffectKind")
            effect_value = getattr(effect, "value", None)
            pure_values = {
                self.tirx.CallEffectKind.ExprAnnotation.value,
                self.tirx.CallEffectKind.Pure.value,
            }
            if effect_value not in pure_values:
                unsupported_effects.append({
                    "operation": getattr(node.op, "name", ""),
                    "effect_kind": effect_value,
                })

        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression), visit)
        if deferred_errors:
            raise deferred_errors[0]
        if unsupported_effects:
            raise CaptureError(
                "TILELANG_EXPRESSION_EFFECT_UNSUPPORTED",
                "nested TIRX call is not proven side-effect free",
                {"calls": unsupported_effects},
            )
        return loads

    def _emit_expression_loads(self, expression: Any, path: tuple[Any, ...]) -> None:
        for index, load in enumerate(self._buffer_loads(expression)):
            for region_index, region in enumerate(
                self.capture.memory_regions(load.buffer, load.indices, self)
            ):
                self.emit(
                    "tilelang.mem.read",
                    (*path, "read", index, region_index),
                    _ir_operation="BufferLoad",
                    region=region,
                )

    def walk(self, statement: Any, path: tuple[Any, ...]) -> None:
        if statement is None:
            return
        name = type(statement).__name__
        if name == "SeqStmt":
            for index, child in enumerate(statement.seq):
                self.walk(child, (*path, "seq", index))
            return
        if name == "AttrStmt":
            if statement.attr_key == "thread_extent":
                tag = str(getattr(statement.node, "thread_tag", ""))
                extent = self.eval_int(statement.value)
                variable = getattr(statement.node, "var", None)
                if extent is None or variable is None:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "thread binding is not statically resolvable",
                        {"tag": tag, "path": list(path)},
                    )
                had_old = variable in self.bindings
                old = self.bindings.get(variable)
                if tag == "threadIdx.x":
                    self.bindings[variable] = self.lane
                elif tag in {"threadIdx.y", "threadIdx.z", "blockIdx.x", "blockIdx.y", "blockIdx.z"}:
                    if extent != 1:
                        raise CaptureError(
                            "TILELANG_SCOPE_UNSUPPORTED",
                            "non-x thread and block bindings must have unit extent",
                            {"tag": tag, "extent": extent},
                        )
                    self.bindings[variable] = 0
                else:
                    raise CaptureError(
                        "TILELANG_THREAD_BINDING_UNSUPPORTED",
                        "thread binding tag is unsupported",
                        {"tag": tag},
                    )
                self.walk(statement.body, (*path, "attr", tag))
                if had_old:
                    assert old is not None
                    self.bindings[variable] = old
                else:
                    self.bindings.pop(variable, None)
                return
            if statement.attr_key == "warp_specialize":
                enabled = self.eval_int(statement.value)
                if enabled != 1:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "warp_specialize attribute must be statically enabled",
                        {"value": str(statement.value)},
                    )
                self.walk(statement.body, (*path, "attr", "warp_specialize"))
                return
            transparent_attributes = {
                "buffer_dim_align",
                "double_buffer_scope",
                "lexical_alloc_scope",
                "pipeline_exec_scope",
                "pipeline_scope",
                "pragma_auto_unroll_max_step",
                "pragma_unroll_explicit",
                "realize_scope",
                "software_pipeline_order",
                "software_pipeline_stage",
                "storage_scope",
            }
            if statement.attr_key in transparent_attributes:
                self.walk(statement.body, (*path, "attr", statement.attr_key))
                return
            raise CaptureError(
                "TILELANG_ATTRIBUTE_UNSUPPORTED",
                "lowered TileLang statement attribute is unsupported",
                {"attribute": statement.attr_key, "path": list(path)},
            )
        if name in {"AllocBuffer", "DeclBuffer"}:
            self.buffer_allocation(statement.buffer)
            body = getattr(statement, "body", None)
            if body is not None:
                self.walk(body, (*path, "buffer", statement.buffer.name))
            return
        if name == "Bind":
            before = len(self.capture.operations)
            value = self.eval_int(statement.value)
            if value is not None:
                self.bindings[statement.var] = value
            else:
                pointer = self.eval_pointer(statement.value)
                if pointer is None:
                    raise CaptureError(
                        "TILELANG_STATIC_VALUE_UNRESOLVED",
                        "TIRX Bind value is neither a static integer nor a static pointer alias",
                        {"variable": str(statement.var), "value": str(statement.value)},
                    )
                self.pointer_aliases[statement.var] = pointer
            self.capture.classify_leaf(before)
            return
        if name == "LetStmt":
            value = self.eval_int(statement.value)
            if value is None:
                raise CaptureError(
                    "TILELANG_STATIC_VALUE_UNRESOLVED",
                    "TIRX LetStmt value is not statically resolvable",
                    {"variable": str(statement.var), "value": str(statement.value)},
                )
            had_old = statement.var in self.bindings
            old = self.bindings.get(statement.var)
            self.bindings[statement.var] = value
            self.walk(statement.body, (*path, "let", statement.var.name))
            if had_old:
                assert old is not None
                self.bindings[statement.var] = old
            else:
                self.bindings.pop(statement.var, None)
            return
        if name == "AssertStmt":
            condition = self.eval_int(statement.condition)
            if condition is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TIRX assertion condition is not statically resolvable",
                    {"condition": str(statement.condition)},
                )
            if not condition:
                raise CaptureError(
                    "TILELANG_STATIC_ASSERTION_FAILED",
                    "TIRX static assertion is false",
                    {"condition": str(statement.condition)},
                )
            self.walk(statement.body, (*path, "assert"))
            return
        if name == "IfThenElse":
            condition = self.eval_int(statement.condition)
            if condition is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TileLang branch condition cannot be evaluated per lane",
                    {"condition": str(statement.condition), "lane": self.lane},
                )
            branch = statement.then_case if condition else statement.else_case
            self.walk(branch, (*path, "if", int(bool(condition))))
            return
        if name == "While":
            seen_states: set[frozenset[tuple[Any, int]]] = set()
            iteration = 0
            while True:
                condition = self.eval_int(statement.condition)
                if condition is None:
                    raise CaptureError(
                        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "TileLang while condition cannot be evaluated statically",
                        {"condition": str(statement.condition), "iteration": iteration},
                    )
                if not condition:
                    break
                state = frozenset(self.bindings.items())
                if state in seen_states:
                    raise CaptureError(
                        "TILELANG_NONFINITE_CONTROL_FLOW",
                        "TileLang while loop repeats a static state",
                        {"condition": str(statement.condition), "iteration": iteration},
                    )
                seen_states.add(state)
                self.capture.static_expansions += 1
                self.walk(statement.body, (*path, "while", iteration))
                iteration += 1
            return
        if name == "For":
            minimum = self.eval_int(statement.min)
            extent = self.eval_int(statement.extent)
            if minimum is None or extent is None or extent < 0:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TileLang loop must have a finite static extent",
                    {"loop": str(statement.loop_var), "extent": str(statement.extent)},
                )
            variable = statement.loop_var
            had_old = variable in self.bindings
            old = self.bindings.get(variable)
            for iteration in range(minimum, minimum + extent):
                self.capture.static_expansions += 1
                self.bindings[variable] = iteration
                self.walk(statement.body, (*path, "for", variable.name, iteration))
            if had_old:
                assert old is not None
                self.bindings[variable] = old
            else:
                self.bindings.pop(variable, None)
            return
        if name == "BufferStore":
            before = len(self.capture.operations)
            self.current_ir_operation = name
            for index, load in enumerate(self._buffer_loads(statement.value)):
                for region_index, region in enumerate(
                    self.capture.memory_regions(load.buffer, load.indices, self)
                ):
                    self.emit(
                        "tilelang.mem.read",
                        (*path, "rhs", index, region_index),
                        _ir_operation="BufferLoad",
                        region=region,
                    )
            if self._access_enabled(statement):
                for region_index, region in enumerate(
                    self.capture.memory_regions(statement.buffer, statement.indices, self)
                ):
                    self.emit("tilelang.mem.write", (*path, region_index), region=region)
            self.capture.classify_leaf(before)
            return
        if name == "Evaluate":
            before = len(self.capture.operations)
            self.evaluate(statement.value, path)
            self.capture.classify_leaf(before)
            return
        if name in {"BlockRealize", "SBlockRealize"}:
            block = statement.block
            iter_vars = list(block.iter_vars)
            iter_values = list(statement.iter_values)
            if len(iter_vars) != len(iter_values):
                raise CaptureError(
                    "TILELANG_BLOCK_BINDING_UNRESOLVED",
                    "BlockRealize iter values do not match block iter variables",
                    {"block": str(block.name_hint)},
                )
            saved: list[tuple[Any, bool, int | None]] = []
            for iter_var, value_expression in zip(iter_vars, iter_values):
                value = self.eval_int(value_expression)
                if value is None:
                    raise CaptureError(
                        "TILELANG_BLOCK_BINDING_UNRESOLVED",
                        "BlockRealize iter value is not statically resolvable",
                        {"block": str(block.name_hint), "value": str(value_expression)},
                    )
                saved.append((iter_var.var, iter_var.var in self.bindings, self.bindings.get(iter_var.var)))
                self.bindings[iter_var.var] = value
            predicate = self.eval_int(statement.predicate)
            if predicate is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "BlockRealize predicate is not statically resolvable",
                    {"block": str(block.name_hint), "predicate": str(statement.predicate)},
                )
            if predicate:
                reduction_vars = [
                    iter_var for iter_var in iter_vars
                    if int(iter_var.iter_type) == self.tirx.IterVar.CommReduce
                ]
                run_init = block.init is not None
                for iter_var in reduction_vars:
                    minimum = self.eval_int(iter_var.dom.min) if iter_var.dom is not None else None
                    if minimum is None:
                        raise CaptureError(
                            "TILELANG_BLOCK_INIT_UNRESOLVED",
                            "reduction block minimum is not statically resolvable",
                            {"block": str(block.name_hint)},
                        )
                    run_init = run_init and self.bindings[iter_var.var] == minimum
                if run_init:
                    self.walk(block.init, (*path, "block", block.name_hint, "init"))
                self.walk(block.body, (*path, "block", block.name_hint, "body"))
            for variable, had_old, old in reversed(saved):
                if had_old:
                    assert old is not None
                    self.bindings[variable] = old
                else:
                    self.bindings.pop(variable, None)
            return
        if name in {"Block", "SBlock"}:
            if statement.init is not None:
                self.walk(statement.init, (*path, "block", statement.name_hint, "init"))
            self.walk(statement.body, (*path, "block", statement.name_hint, "body"))
            return
        raise CaptureError(
            "TILELANG_STATEMENT_UNSUPPORTED",
            "lowered TileLang statement has no sound capture rule",
            {"statement": name, "path": list(path)},
        )

    def evaluate(self, expression: Any, path: tuple[Any, ...]) -> None:
        self.current_ir_operation = type(expression).__name__
        if isinstance(expression, self.tirx.IntImm):
            return
        if isinstance(expression, self.tirx.BufferLoad):
            if self._access_enabled(expression):
                for region_index, region in enumerate(
                    self.capture.memory_regions(expression.buffer, expression.indices, self)
                ):
                    self.emit("tilelang.mem.read", (*path, region_index), region=region)
            return
        if not isinstance(expression, self.tirx.Call):
            self._emit_expression_loads(expression, path)
            return
        name = getattr(expression.op, "name", "")
        self.current_ir_operation = name
        if name == "tl.set_max_nreg":
            if not valid_set_max_nreg(expression.args, self.eval_int):
                raise CaptureError(
                    "TILELANG_SET_MAX_NREG_UNRESOLVED",
                    "set_max_nreg requires a static PTX register count and action",
                    {"args": [str(argument) for argument in expression.args]},
                )
            return
        if name == "tl.prefetch_tma_descriptor":
            if len(expression.args) != 1:
                raise CaptureError(
                    "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                    "prefetch_tma_descriptor requires one descriptor operand",
                )
            return
        if name == "tl.ptx_fence_barrier_init":
            if expression.args:
                raise CaptureError(
                    "TILELANG_OPERATION_MALFORMED",
                    "ptx_fence_barrier_init must not have operands",
                )
            return
        if name == "tl.fence_proxy_async":
            if expression.args or self.pending_proxy is not None:
                raise CaptureError(
                    "TILELANG_PROXY_TARGET_UNRESOLVED",
                    "fence_proxy_async must be operand-free and have one unresolved target",
                )
            self.emit(
                "tilelang.proxy_fence",
                path,
                from_domain="GENERIC",
                to_domain="ASYNC_COPY",
                scope="CTA",
                regions=[],
                visibility_mask={
                    "shared": False,
                    "global_read": False,
                    "global_write": False,
                    "tensor_read": False,
                    "tensor_write": False,
                    "control_only": False,
                },
            )
            self.pending_proxy = self.capture.operations[-1]
            return
        if name in {"tirx.tvm_storage_sync", "tir.tvm_storage_sync"}:
            if len(expression.args) != 1 or str(expression.args[0]).strip('"') not in {"shared", "shared.dyn"}:
                raise CaptureError(
                    "TILELANG_BARRIER_SCOPE_UNSUPPORTED",
                    "only full-CTA shared storage synchronization is supported",
                    {"operation": name, "args": [str(arg) for arg in expression.args]},
                )
            resource = self.capture.cta_barrier(path)
            self.emit(
                "tilelang.barrier.arrive",
                (*path, "arrive"),
                resource=resource,
                invocation_id=0,
                count=1,
            )
            self.emit(
                "tilelang.barrier.observe",
                (*path, "observe"),
                resource=resource,
                invocation_id=0,
            )
            return
        if name in {"tirx.ptx_init_barrier_thread_count", "tir.ptx_init_barrier_thread_count"}:
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_MBARRIER_COUNT_UNRESOLVED",
                    "mbarrier initialization requires barrier and count operands",
                    {"operation": name, "operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            count = self.eval_int(expression.args[1])
            if count is None or count <= 0:
                raise CaptureError(
                    "TILELANG_MBARRIER_COUNT_UNRESOLVED",
                    "mbarrier initialization count must be a static positive integer",
                    {"operation": name},
                )
            self.emit(
                "tilelang.mbarrier.init",
                path,
                resource=resource,
                lifecycle_epoch=0,
                count=count,
            )
            return
        if name in {"tirx.ptx_arrive_barrier", "tir.ptx_arrive_barrier"}:
            if len(expression.args) != 1:
                raise CaptureError(
                    "TILELANG_MBARRIER_ARRIVE_UNRESOLVED",
                    "mbarrier arrival requires exactly one barrier operand",
                    {"operation": name, "operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            generation = self.arrival_rank.get(resource, 0)
            self.arrival_rank[resource] = generation + 1
            self.emit(
                "tilelang.mbarrier.arrive",
                path,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                count=1,
                predicate=True,
            )
            return
        if name == "tl.mbarrier_wait_parity":
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_PARITY_UNRESOLVED",
                    "mbarrier wait requires barrier and parity operands",
                    {"operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            parity = self.eval_int(expression.args[1])
            generation = self.wait_rank.get(resource, 0)
            if (
                parity == 1
                and generation == 0
                and self.arrival_rank.get(resource, 0) == 0
            ):
                return
            if parity is None or parity != generation % 2:
                raise CaptureError(
                    "TILELANG_PARITY_UNRESOLVED",
                    "wait parity does not match the recovered gap-free stream rank",
                    {"resource": resource, "rank": generation, "parity": parity},
                )
            if self.capture.uses_warpgroup and not self.is_warpgroup_leader():
                self.observe_warpgroup_dependency(path)
                return
            self.wait_rank[resource] = generation + 1
            actor = self.warpgroup_owner() if self.capture.uses_warpgroup else self.actor
            self.emit(
                "tilelang.mbarrier.wait_parity",
                path,
                _actor=actor,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                raw_parity=parity,
                logical_stream_id=f"{actor}:{resource}",
                stream_rank=generation,
                polling_coalesced=True,
            )
            if self.capture.uses_warpgroup:
                self.publish_warpgroup_dependency(self.capture.operations[-1]["id"], path)
            return
        if name == "tl.mbarrier_expect_tx":
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_TMA_EXPECT_UNRESOLVED",
                    "mbarrier_expect_tx requires barrier and byte count",
                )
            resource = self._barrier_operand(expression, 0)
            byte_count = self.eval_int(expression.args[1])
            if byte_count is None or byte_count <= 0:
                raise CaptureError(
                    "TILELANG_TMA_EXPECT_UNRESOLVED",
                    "mbarrier expected transaction bytes must be static and positive",
                )
            generation = self.arrival_rank.get(resource, 0)
            self.emit(
                "tilelang.mbarrier.expect_tx",
                path,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                bytes=byte_count,
            )
            return
        if name == "tl.tma_load":
            if len(expression.args) < 4:
                raise CaptureError(
                    "TILELANG_TMA_LOAD_UNRESOLVED",
                    "tma_load requires descriptor, barrier, destination, and coordinates",
                )
            descriptor = expression.args[0]
            resource = self._barrier_operand(expression, 1)
            destination, _destination_bytes = self.access_ptr_region(expression.args[2], 2)
            coordinates = [self.eval_int(argument) for argument in expression.args[3:]]
            if any(value is None for value in coordinates):
                raise CaptureError(
                    "TILELANG_TMA_LOAD_UNRESOLVED",
                    "TMA coordinates must be statically resolved",
                    {"coordinates": [str(argument) for argument in expression.args[3:]]},
                )
            generation = self.arrival_rank.get(resource, 0)
            segments = self.capture.tma_segments(
                descriptor,
                [int(value) for value in coordinates],
                destination,
            )
            self.emit(
                "tilelang.tma_load.issue",
                path,
                resource=resource,
                transfers=[
                    {
                        "source_region": source,
                        "destination_region": destination_segment,
                        "bytes": byte_count,
                    }
                    for source, destination_segment, byte_count in segments
                ],
                bytes=sum(byte_count for _source, _destination, byte_count in segments),
                lifecycle_epoch=0,
                generation=generation,
            )
            return
        if name == "tl.tma_store":
            if len(expression.args) != 5:
                raise CaptureError(
                    "TILELANG_TMA_STORE_UNRESOLVED",
                    "tma_store requires destination, source, bytes, reduction, and eviction operands",
                )
            destination, destination_bytes = self.access_ptr_region(expression.args[0], 2)
            source, source_bytes = self.access_ptr_region(expression.args[1], 1)
            byte_count = self.eval_int(expression.args[2])
            if (
                byte_count is None
                or byte_count <= 0
                or byte_count > source_bytes
                or byte_count > destination_bytes
                or self.eval_int(expression.args[3]) is None
                or self.eval_int(expression.args[4]) is None
            ):
                raise CaptureError(
                    "TILELANG_TMA_STORE_UNRESOLVED",
                    "TMA store byte count or mode operands are unresolved",
                )
            if self.pending_proxy is not None:
                self.pending_proxy["to_domain"] = "TMA"
                self.pending_proxy["regions"] = [source]
                self.pending_proxy["visibility_mask"]["shared"] = True
                self.pending_proxy = None
            owner = self.actor
            resource = self.capture.tma_store_group(owner)
            self.emit(
                "tilelang.tma_store.issue",
                path,
                resource=resource,
                group_owner=owner,
                source_region=source,
                destination_region=destination,
                bytes=byte_count,
            )
            return
        if name == "tl.tma_store_arrive":
            if expression.args:
                raise CaptureError(
                    "TILELANG_TMA_STORE_COMMIT_UNRESOLVED",
                    "tma_store_arrive must not have operands",
                )
            if self.lane % 128:
                return
            owner = self.actor
            resource = self.capture.tma_store_group(owner)
            self.emit(
                "tilelang.tma_store.commit_group",
                path,
                resource=resource,
                group_owner=owner,
                group_seq=self.tma_store_group_seq,
            )
            self.tma_store_group_seq += 1
            return
        if name == "tl.tma_store_wait":
            wait_count = self.eval_int(expression.args[0]) if len(expression.args) == 2 else None
            read_only = self.eval_int(expression.args[1]) if len(expression.args) == 2 else None
            if wait_count is None or not 0 <= wait_count <= 7 or read_only not in {0, 1}:
                raise CaptureError(
                    "TILELANG_TMA_STORE_WAIT_UNRESOLVED",
                    "tma_store_wait requires static count and read modifier",
                )
            if self.lane % 128:
                return
            owner = self.actor
            resource = self.capture.tma_store_group(owner)
            self.emit(
                "tilelang.tma_store.wait_group",
                path,
                resource=resource,
                group_owner=owner,
                wait_group_count=wait_count,
                wait_mode="READ_ONLY" if read_only else "GLOBAL",
            )
            return
        if name == "tl.initialize_wgmma_descriptor":
            if (
                len(expression.args) != 5
                or not isinstance(expression.args[0], self.tirx.BufferLoad)
            ):
                raise CaptureError(
                    "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                    "initialize_wgmma_descriptor has malformed descriptor or source operands",
                )
            region, _extent = self.access_ptr_region(expression.args[1], 1)
            for argument in expression.args[2:]:
                if self.eval_int(argument) is None:
                    raise CaptureError(
                        "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                        "WGMMA descriptor layout operands must be static",
                    )
            descriptor = expression.args[0]
            self.capture.wgmma_descriptors[descriptor.buffer.data] = region
            return
        if name == "tl.warpgroup_fence_operand":
            if len(expression.args) != 4:
                raise CaptureError(
                    "TILELANG_WGMMA_FENCE_UNRESOLVED",
                    "warpgroup_fence_operand requires dtype, accumulator, offset, and extent",
                )
            if not self.is_warpgroup_leader():
                return
            import re

            dtype_match = re.search(r"(\d+)", str(expression.args[0]))
            data = expression.args[1]
            offset = self.eval_int(expression.args[2])
            extent = self.eval_int(expression.args[3])
            if dtype_match is None or offset is None or extent is None or extent <= 0:
                raise CaptureError(
                    "TILELANG_WGMMA_FENCE_UNRESOLVED",
                    "warpgroup accumulator region is not statically resolved",
                )
            width = int(dtype_match.group(1)) // 8
            owner = self.warpgroup_owner()
            label = getattr(data, "name", "accumulator")
            allocation = self.capture.allocation_id(data, "local", owner, label)
            self.wgmma_accumulator_region = self.capture.region(
                allocation,
                "local",
                offset * width,
                extent * width * 128,
            )
            self.wgmma_fence_seen = True
            return
        if name == "tl.warpgroup_arrive":
            if expression.args:
                raise CaptureError(
                    "TILELANG_WGMMA_FENCE_UNRESOLVED",
                    "warpgroup_arrive must not have operands",
                )
            if self.is_warpgroup_leader():
                self.wgmma_arrived = True
            return
        if name == "tl.ptx_wgmma_ss":
            if len(expression.args) != 15:
                raise CaptureError(
                    "TILELANG_WGMMA_UNRESOLVED",
                    "ptx_wgmma_ss operand signature is unsupported",
                    {"operands": len(expression.args)},
                )
            if not self.is_warpgroup_leader():
                return
            source_regions = [
                self.capture.wgmma_descriptors.get(expression.args[position])
                for position in (6, 8)
            ]
            if (
                any(region is None for region in source_regions)
                or self.wgmma_accumulator_region is None
                or not self.wgmma_fence_seen
                or not self.wgmma_arrived
                or any(self.eval_int(expression.args[position]) is None for position in (7, 9, 11, 12, 13, 14))
            ):
                raise CaptureError(
                    "TILELANG_WGMMA_UNRESOLVED",
                    "WGMMA descriptors, accumulator, fence, or static offsets are unresolved",
                )
            owner = self.warpgroup_owner()
            resource = self.capture.wgmma_resource(owner)
            self.emit(
                "tilelang.wgmma.fence_operand",
                (*path, "fence"),
                _actor=owner,
                resource=resource,
                group_owner=owner,
                regions=source_regions,
            )
            self.emit(
                "tilelang.wgmma.issue",
                path,
                _actor=owner,
                resource=resource,
                group_owner=owner,
                regions=source_regions,
                destination_region=self.wgmma_accumulator_region,
            )
            return
        if name == "tl.warpgroup_commit_batch":
            if expression.args:
                raise CaptureError(
                    "TILELANG_WGMMA_COMMIT_UNRESOLVED",
                    "warpgroup_commit_batch must not have operands",
                )
            if not self.is_warpgroup_leader():
                return
            owner = self.warpgroup_owner()
            resource = self.capture.wgmma_resource(owner)
            group_seq = self.capture.wgmma_group_seq.get(owner, 0)
            self.emit(
                "tilelang.wgmma.commit_group",
                path,
                _actor=owner,
                resource=resource,
                group_owner=owner,
                group_seq=group_seq,
            )
            self.capture.wgmma_group_seq[owner] = group_seq + 1
            return
        if name == "tl.warpgroup_wait":
            wait_count = self.eval_int(expression.args[0]) if len(expression.args) == 1 else None
            if wait_count is None or not 0 <= wait_count <= 7:
                raise CaptureError(
                    "TILELANG_WGMMA_WAIT_UNRESOLVED",
                    "warpgroup_wait requires a static count in [0, 7]",
                )
            if not self.is_warpgroup_leader():
                self.observe_warpgroup_dependency(path)
                return
            owner = self.warpgroup_owner()
            resource = self.capture.wgmma_resource(owner)
            for completion_kind in ("OPERANDS_RELEASED", "RESULT_READY"):
                self.emit(
                    "tilelang.wgmma.wait_group",
                    (*path, completion_kind.lower()),
                    _actor=owner,
                    resource=resource,
                    group_owner=owner,
                    wait_group_count=wait_count,
                    completion_kind=completion_kind,
                )
            self.publish_warpgroup_dependency(self.capture.operations[-1]["id"], path)
            return
        if name == "tl.ptx_cp_async":
            if len(expression.args) != 3:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async requires destination, source, and logical element count operands",
                )
            logical_count = self.eval_int(expression.args[2])
            byte_count = cp_async_transfer_bytes(
                logical_count,
                self.access_ptr_dtype(expression.args[0]),
                self.access_ptr_dtype(expression.args[1]),
            ) if logical_count is not None else None
            if byte_count is None:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async transfer width must resolve to 4, 8, or 16 bytes",
                    {"logical_elements": logical_count},
                )
            destination, destination_bytes = self.access_ptr_region(
                expression.args[0],
                2,
                logical_count,
            )
            source, source_bytes = self.access_ptr_region(
                expression.args[1],
                1,
                logical_count,
            )
            if byte_count != destination_bytes or byte_count != source_bytes:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async source and destination transfer widths differ",
                    {
                        "bytes": byte_count,
                        "source_extent": source_bytes,
                        "destination_extent": destination_bytes,
                    },
                )
            if self.pending_proxy is not None:
                source_space = self.capture.regions[source]["address_space"]
                mask_field = {
                    "SHARED": "shared",
                    "GLOBAL": "global_write",
                    "TENSOR": "tensor_write",
                }.get(source_space)
                if mask_field is None:
                    raise CaptureError(
                        "TILELANG_PROXY_REGION_UNSUPPORTED",
                        "fence_proxy_async source region has no supported visibility mask",
                        {"region": source, "address_space": source_space},
                    )
                self.pending_proxy["regions"] = [source]
                self.pending_proxy["visibility_mask"][mask_field] = True
                self.pending_proxy = None
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.issue",
                path,
                resource=resource,
                source_region=source,
                destination_region=destination,
                bytes=byte_count,
                group_owner=self.actor,
                predicate=True,
            )
            self.cp_staged += 1
            return
        if name in {"tirx.ptx_commit_group", "tir.ptx_commit_group"}:
            if expression.args:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_COMMIT_UNRESOLVED",
                    "ptx_commit_group must not have operands",
                )
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.commit_group",
                path,
                resource=resource,
                group_owner=self.actor,
                group_seq=self.cp_group_seq,
            )
            self.cp_group_seq += 1
            self.cp_staged = 0
            return
        if name in {"tirx.ptx_wait_group", "tir.ptx_wait_group"}:
            wait_count = self.eval_int(expression.args[0]) if len(expression.args) == 1 else None
            if wait_count is None or not 0 <= wait_count <= 7:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_WAIT_UNRESOLVED",
                    "ptx_wait_group count must be a static integer in [0, 7]",
                    {"wait_group_count": wait_count},
                )
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.wait_group",
                path,
                resource=resource,
                group_owner=self.actor,
                wait_group_count=wait_count,
            )
            return
        self._emit_expression_loads(expression, path)

    def _barrier_operand(self, call: Any, index: int) -> str:
        if len(call.args) <= index or not isinstance(call.args[index], self.tirx.BufferLoad):
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier operand must be a static BufferLoad",
                {"operation": getattr(call.op, "name", "")},
            )
        operand = call.args[index]
        return self.capture.mbarrier(operand.buffer, operand.indices, self)


def capture_file(
    path: Path,
    source: str,
    command: str,
    descriptor_evidence_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import tilelang
        from tilelang import tvm
        from tvm import tirx
    except ImportError as error:
        raise CaptureError(
            "TILELANG_RUNTIME_UNAVAILABLE",
            "TileLang capture must run in an isolated TileLang environment",
            {"error": str(error)},
        ) from error
    try:
        source_ir = path.read_text()
        module = tvm.ir.load_json(source_ir)
    except (OSError, ValueError, TypeError) as error:
        raise CaptureError(
            "TILELANG_IR_INPUT_INVALID",
            "TileLang IR JSON could not be loaded",
            {"path": str(path), "error": str(error)},
        ) from error
    source_digest = hashlib.sha256(source_ir.encode("utf-8")).hexdigest()
    descriptor_evidence: dict[str, Any] = {}
    if descriptor_evidence_path is not None:
        try:
            evidence = json.loads(descriptor_evidence_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CaptureError(
                "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                "TileLang descriptor evidence could not be read",
                {"path": str(descriptor_evidence_path), "error": str(error)},
            ) from error
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"evidence_version", "source_ir_sha256", "descriptors"}
            or evidence.get("evidence_version") != "tilelang-tma-descriptor-evidence-v1"
            or evidence.get("source_ir_sha256") != source_digest
            or not isinstance(evidence.get("descriptors"), dict)
        ):
            raise CaptureError(
                "TILELANG_TMA_DESCRIPTOR_UNRESOLVED",
                "TileLang descriptor evidence is malformed or bound to different IR",
                {"path": str(descriptor_evidence_path)},
            )
        descriptor_evidence = evidence["descriptors"]
    try:
        return TileLangCapture(
            tvm,
            tirx,
            module,
            source,
            command,
            source_digest,
            descriptor_evidence,
        ).capture()
    except CaptureError:
        raise
    except Exception as error:
        raise CaptureError(
            "TILELANG_CAPTURE_FAILURE",
            "TileLang binding or capture analysis failed closed",
            {"error_type": type(error).__name__, "error": str(error)},
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a strict TileLang TIRX snapshot for PipeGuard")
    parser.add_argument("input", type=Path, help="TVM JSON containing one lowered TileLang device PrimFunc")
    parser.add_argument("--source", required=True, help="stable source/provenance label")
    parser.add_argument(
        "--descriptor-evidence",
        type=Path,
        help="SHA-bound exact TMA descriptor shape/stride evidence",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(
            args.input,
            args.source,
            " ".join(sys.argv),
            args.descriptor_evidence,
        )
    except CaptureError as error:
        result = {
            "captured": False,
            "verdict": "UNSUPPORTED",
            "diagnostics": [error.diagnostic()],
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
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
