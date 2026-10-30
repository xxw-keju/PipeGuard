#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import heapq
import json
import math
import re
from itertools import product
import subprocess
import sys
from pathlib import Path
from typing import Any

from .tilelang_static import (
    cp_async_transfer_bytes,
    dtype_bit_width,
    dtype_byte_width,
    is_pure_payload_call,
    peel_static_byte_offsets,
    valid_set_max_nreg,
    vector_values,
)

def is_empty_signed_cuda_loop(statement: Any, minimum: int | None, extent: int | None) -> bool:
    # Pinned CodeGenC emits `for (signed i=0; i<extent; ++i)` for
    # serial/unrolled loops. A proven negative signed upper bound executes
    # zero iterations. Unsigned or nonzero-min cases remain conservative.
    return (minimum == 0 and extent is not None and extent < 0
            and str(statement.loop_var.dtype) in {"int32", "int64"}
            and str(statement.extent.dtype) in {"int32", "int64"}
            and int(statement.kind) in {0, 3})


SNAPSHOT_VERSION = "tilelang-snapshot-v3"
ADAPTER_VERSION = "tilelang-pgir-v3"
# Whole-allocation coverage for proxy fences; matches target shared_memory_bytes.
SHARED_WINDOW_BYTES = 65536
WGMMA_PROOF_VERSION = "pipeguard-wgmma-stage-union-proof-v1"
_WGMMA_SHAPE = re.compile(r"^m(\d+)n(\d+)k(\d+)$")
_WGMMA_DTYPE_BYTES = {"fp16": 2}
_WGMMA_LAYOUT_DTYPE_BYTES = {
    "fp16": 2, "bf16": 2, "tf32": 4,
    "e4m3": 1, "e5m2": 1, "s8": 1, "u8": 1,
}
# Descriptor layout_type -> swizzle row width in bytes (0 is the interleaved
# 8x16B core-matrix layout).
_WGMMA_LAYOUT_ROW_BYTES = {0: 16, 1: 128, 2: 64, 3: 32}


# Scalar device-scope read-modify-writes: (destination pointer, value[, order]).
_SCALAR_ATOMICS = frozenset({
    f"tl.atomic_{kind}{suffix}_elem_op"
    for kind in ("add", "max", "min", "or")
    for suffix in ("", "_ret")
})
# Vector atomic adds: (destination pointer, value pointer) over 2 or 4 elements.
_VECTOR_ATOMICS = {
    f"tl.atomic_addx{width}{suffix}_elem_op": width
    for width in (2, 4)
    for suffix in ("", "_ret")
}
_ATOMICS = _SCALAR_ATOMICS | _VECTOR_ATOMICS.keys()

StaticLoopState = tuple[
    frozenset[tuple[Any, int]],
    frozenset[tuple[tuple[Any, int], int]],
]


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


def _wgmma_string(value: Any) -> str:
    return str(getattr(value, "value", value)).strip('"')


def _wgmma_issue_operand(
    *,
    slot: str,
    shape: str,
    dtype: str,
    k_major: bool,
    descriptor_offset_units: int,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    match = _WGMMA_SHAPE.fullmatch(shape)
    if match is None or tuple(map(int, match.groups())) != (64, 64, 16):
        raise CaptureError(
            "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED",
            "exact group-union proof supports only the selected m64n64k16 profiles",
            {"shape": shape, "slot": slot},
        )
    element_bytes = _WGMMA_DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise CaptureError(
            "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED",
            "exact group-union proof requires an admitted operand dtype",
            {"dtype": dtype, "slot": slot},
        )
    layout = (
        descriptor.get("layout_type"),
        descriptor.get("leading_byte_offset"),
        descriptor.get("stride_byte_offset"),
    )
    if layout not in {(1, 0, 64), (1, 1, 64), (2, 1, 32)}:
        raise CaptureError(
            "TILELANG_WGMMA_DESCRIPTOR_LAYOUT_UNSUPPORTED",
            "descriptor layout is outside the selected Suite 5 proof profiles",
            {"layout": list(layout), "slot": slot},
        )
    if descriptor_offset_units < 0:
        raise CaptureError(
            "TILELANG_WGMMA_DESCRIPTOR_OFFSET_UNRESOLVED",
            "WGMMA descriptor offset must be nonnegative",
            {"descriptor_offset_units_16b": descriptor_offset_units, "slot": slot},
        )
    issue_bytes = 64 * 16 * element_bytes
    stride_units = 2 if k_major else 128
    if descriptor_offset_units % stride_units:
        raise CaptureError(
            "TILELANG_WGMMA_DESCRIPTOR_OFFSET_UNRESOLVED",
            "WGMMA descriptor offset does not select an exact K16 slice",
            {
                "descriptor_offset_units_16b": descriptor_offset_units,
                "expected_stride_units_16b": stride_units,
                "slot": slot,
            },
        )
    return {
        "slot": slot,
        "shape": shape,
        "dtype": dtype,
        "k_major": k_major,
        "descriptor_id": descriptor["descriptor_id"],
        "descriptor_layout": {
            "layout_type": layout[0],
            "leading_byte_offset": layout[1],
            "stride_byte_offset": layout[2],
        },
        "descriptor_offset_units_16b": descriptor_offset_units,
        "slice_index": descriptor_offset_units // stride_units,
        "issue_bytes": issue_bytes,
        "selected_stage": descriptor["selected_stage"],
        "coordinate_space": "logical-stage-byte-order-before-bijective-swizzle",
    }


def _wgmma_layout_hull(
    *,
    mn: int,
    k: int,
    element_bytes: int,
    k_major: bool,
    layout_type: Any,
    leading: Any,
    stride: Any,
    start: int,
    hedge_swapped: bool = True,
) -> tuple[int, int] | None:
    """Byte hull of one WGMMA operand read under the canonical GMMA layouts.

    Offsets follow CuTe make_gmma_desc in 16-byte units: K-major
    ((8,m),(2,k)):((W,SBO),(1|LBO,-)), swizzled MN-major ((W,m),(8,k)):
    ((1,LBO),(W,SBO)), interleaved MN-major ((1,m),(8,k)):((-,SBO),(1,LBO)).
    The LBO/SBO roles are also taken swapped and the larger hull kept, and a
    swizzled chunk is widened to its whole swizzle row, so the hull covers
    every byte the hardware can read. Without ``hedge_swapped`` only the
    PTX/CuTe roles are used.
    """
    row_bytes = _WGMMA_LAYOUT_ROW_BYTES.get(layout_type)
    if (
        row_bytes is None
        or not isinstance(leading, int)
        or not isinstance(stride, int)
        or leading < 0
        or stride < 0
        or start < 0
        or mn <= 0
        or k <= 0
    ):
        return None
    width = row_bytes // 16
    swizzled = layout_type != 0
    k_units = -(-k * element_bytes // 16)
    mn_units = -(-mn * element_bytes // 16)
    high = 0
    orders = ((leading, stride), (stride, leading)) if hedge_swapped else ((leading, stride),)
    for lbo, sbo in orders:
        if k_major:
            last = (
                (min(mn, 8) - 1) * width
                + (mn - 1) // 8 * sbo
                + (k_units - 1) * (1 if swizzled else lbo)
            )
        elif swizzled:
            last = (
                min(mn_units, width) - 1
                + (mn_units - 1) // width * lbo
                + (min(k, 8) - 1) * width
                + (k - 1) // 8 * sbo
            )
        else:
            last = (mn_units - 1) * sbo + min(k, 8) - 1 + (k - 1) // 8 * lbo
        high = max(high, start + 16 * last)
    if not swizzled:
        return start, high + 16
    return start - start % row_bytes, high - high % row_bytes + row_bytes


def _prove_wgmma_group(
    owner: str,
    resource: str,
    group_seq: int,
    commit_event_id: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if not issues:
        raise CaptureError(
            "TILELANG_WGMMA_COMMITTED_GROUP_EMPTY",
            "an exact WGMMA region proof requires at least one staged issue",
            {"owner": owner, "group_seq": group_seq},
        )
    operands_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        for operand in issue["operands"]:
            operands_by_slot[operand["slot"]].append({
                "issue_template_id": issue["issue_template_id"],
                "engine_seq": issue["engine_seq"],
                **operand,
            })
    proved_operands: list[dict[str, Any]] = []
    for slot in sorted(operands_by_slot):
        operands = sorted(operands_by_slot[slot], key=lambda item: item["engine_seq"])
        identities = {
            (item["descriptor_id"], item["selected_stage"]["region_id"])
            for item in operands
        }
        if len(identities) != 1:
            raise CaptureError(
                "TILELANG_WGMMA_COMMITTED_GROUP_STAGE_MIXED",
                "one committed operand group refers to more than one descriptor stage",
                {"owner": owner, "group_seq": group_seq, "slot": slot},
            )
        issue_sizes = {item["issue_bytes"] for item in operands}
        if len(issue_sizes) != 1:
            raise CaptureError(
                "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED",
                "one committed operand group has inconsistent issue footprints",
                {"owner": owner, "group_seq": group_seq, "slot": slot},
            )
        issue_bytes = next(iter(issue_sizes))
        stage = operands[0]["selected_stage"]
        stage_bytes = stage["extent"]
        if stage_bytes <= 0 or stage_bytes % issue_bytes:
            raise CaptureError(
                "TILELANG_WGMMA_COMMITTED_GROUP_REGION_MISMATCH",
                "selected stage is not an integral union of WGMMA issue footprints",
                {
                    "owner": owner,
                    "group_seq": group_seq,
                    "slot": slot,
                    "stage_bytes": stage_bytes,
                    "issue_bytes": issue_bytes,
                },
            )
        expected_slices = stage_bytes // issue_bytes
        slice_indexes = sorted(item["slice_index"] for item in operands)
        if slice_indexes != list(range(expected_slices)):
            raise CaptureError(
                "TILELANG_WGMMA_COMMITTED_GROUP_REGION_MISMATCH",
                "committed WGMMA issues do not exactly partition the selected stage",
                {
                    "owner": owner,
                    "group_seq": group_seq,
                    "slot": slot,
                    "actual_slice_indexes": slice_indexes,
                    "expected_slice_indexes": list(range(expected_slices)),
                },
            )
        actual_intervals = [
            {"offset": index * issue_bytes, "extent": issue_bytes}
            for index in slice_indexes
        ]
        actual_union: list[dict[str, int]] = []
        for interval in actual_intervals:
            if (
                actual_union
                and actual_union[-1]["offset"] + actual_union[-1]["extent"]
                == interval["offset"]
            ):
                actual_union[-1]["extent"] += interval["extent"]
            else:
                actual_union.append(dict(interval))
        expected_union = [{"offset": 0, "extent": stage_bytes}]
        if actual_union != expected_union:
            raise CaptureError(
                "TILELANG_WGMMA_COMMITTED_GROUP_REGION_MISMATCH",
                "normalized WGMMA issue union differs from the selected stage",
                {
                    "owner": owner,
                    "group_seq": group_seq,
                    "slot": slot,
                    "actual_union": actual_union,
                    "expected_union": expected_union,
                },
            )
        proved_operands.append({
            "slot": slot,
            "shape": operands[0]["shape"],
            "dtype": operands[0]["dtype"],
            "k_major": operands[0]["k_major"],
            "descriptor_id": operands[0]["descriptor_id"],
            "descriptor_layout": operands[0]["descriptor_layout"],
            "selected_stage": stage,
            "issues": [
                {
                    "issue_template_id": item["issue_template_id"],
                    "engine_seq": item["engine_seq"],
                    "descriptor_offset_units_16b": item["descriptor_offset_units_16b"],
                    "slice_index": item["slice_index"],
                    "actual_logical_intervals": [actual_intervals[index]],
                }
                for index, item in enumerate(operands)
            ],
            "actual_logical_union": actual_union,
            "expected_logical_union": expected_union,
            "coordinate_space": operands[0]["coordinate_space"],
            "region_precision": "EXACT_GROUP_UNION",
            "equal": True,
        })
    return {
        "owner": owner,
        "resource": resource,
        "group_seq": group_seq,
        "commit_template_id": commit_event_id,
        "issue_template_ids": [issue["issue_template_id"] for issue in issues],
        "engine_seqs": [issue["engine_seq"] for issue in issues],
        "operands": proved_operands,
        "status": "PROVED",
    }


class CompactTraceBuilder:
    _COLLECTIVE_OPS = {
        "tilelang.wgmma.fence_operand",
        "tilelang.wgmma.issue",
        "tilelang.wgmma.commit_group",
        "tilelang.wgmma.wait_group",
    }
    _ELECTED_OPS = {
        "tilelang.tma_load.issue",
        "tilelang.tma_store.issue",
        "tilelang.tma_store.commit_group",
        "tilelang.tma_store.wait_group",
    }

    def __init__(self, capture: "TileLangCapture"):
        self.capture = capture
        self.actor_map = {
            actor: ("LANE", (lane,))
            for lane, actor in enumerate(capture.actor_ids)
        }
        self.actor_map.update({
            actor: ("WARPGROUP", tuple(range(group * 128, (group + 1) * 128)))
            for group, actor in enumerate(capture.warpgroup_ids)
        })
        self.records: list[dict[str, Any]] = []
        self.record_by_id: dict[str, dict[str, Any]] = {}
        self.traces: dict[int, list[str]] = defaultdict(list)
        self.occurrences: dict[tuple[str, tuple[Any, ...]], int] = defaultdict(int)
        self.first_seen = 0

    @staticmethod
    def _base(operation: dict[str, Any]) -> tuple[Any, ...]:
        evidence = operation["evidence"]
        fields = tuple(sorted(
            key for key in operation
            if key not in {"id", "op", "actor", "seq", "depends_on", "evidence"}
        ))
        return (
            operation["op"],
            evidence["source_location"],
            tuple(evidence["activation_path"]),
            evidence["recognizer_rule"],
            evidence["ir_operation"],
            fields,
        )

    def add_trace(self, lane: int, operations: list[dict[str, Any]]) -> None:
        for operation in operations:
            actor = operation["actor"]
            kind, participants = self.actor_map[actor]
            base = self._base(operation)
            occurrence_key = (actor, base)
            occurrence = self.occurrences[occurrence_key]
            self.occurrences[occurrence_key] += 1
            scope = (
                "COLLECTIVE"
                if operation["op"] in self._COLLECTIVE_OPS or kind == "WARPGROUP"
                else "ELECTED"
                if operation["op"] in self._ELECTED_OPS
                else "INSTANCE"
            )
            instances = participants if scope == "COLLECTIVE" else (lane,)
            grouping_actor = actor if scope in {"COLLECTIVE", "ELECTED"} else None
            initial_key = (base, occurrence, scope, grouping_actor)
            record = {
                "operation": operation,
                "initial_key": initial_key,
                "instances": instances,
                "origin_instance": lane,
                "scope": scope,
                "owner": actor if scope == "COLLECTIVE" else None,
                "first_seen": self.first_seen,
            }
            self.first_seen += 1
            self.records.append(record)
            self.record_by_id[operation["id"]] = record
            self.traces[lane].append(operation["id"])

    def discard(self, operation_ids: set[str]) -> None:
        if not operation_ids:
            return
        self.records = [
            record for record in self.records
            if record["operation"]["id"] not in operation_ids
        ]
        for identifier in operation_ids:
            self.record_by_id.pop(identifier, None)
        for lane, trace in self.traces.items():
            self.traces[lane] = [
                identifier for identifier in trace if identifier not in operation_ids
            ]

    def add_marker(self, lane: int, operation_id: str) -> None:
        if operation_id not in self.record_by_id:
            raise CaptureError(
                "TILELANG_PARAMETRIC_ORDER_UNRESOLVED",
                "collective marker references an unknown compact operation",
                {"operation": operation_id, "lane": lane},
            )
        if operation_id not in self.traces[lane]:
            self.traces[lane].append(operation_id)

    def _relation(self, source: dict[str, Any], target: dict[str, Any]) -> str:
        if target["scope"] == "COLLECTIVE":
            return "COLLECTIVE"
        if source["scope"] == "COLLECTIVE":
            return "ALL_INSTANCES"
        if source["origin_instance"] in target["instances"]:
            return "SAME_INSTANCE"
        return "ALL_INSTANCES"

    def _partition(self) -> tuple[dict[str, int], dict[int, list[dict[str, Any]]]]:
        key_order: dict[tuple[Any, ...], int] = {}
        labels: dict[str, int] = {}
        for record in self.records:
            key = record["initial_key"]
            labels[record["operation"]["id"]] = key_order.setdefault(key, len(key_order))

        while True:
            signatures: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
            for record in self.records:
                dependencies = tuple(sorted(
                    (
                        labels[dependency],
                        self._relation(record, self.record_by_id[dependency]),
                    )
                    for dependency in record["operation"].get("depends_on", [])
                ))
                signatures.append((record, (record["initial_key"], dependencies)))
            signature_order: dict[tuple[Any, ...], int] = {}
            updated = {
                record["operation"]["id"]: signature_order.setdefault(
                    signature, len(signature_order)
                )
                for record, signature in signatures
            }
            if all(updated[identifier] == labels[identifier] for identifier in labels):
                labels = updated
                break
            if len(set(updated.values())) == len(set(labels.values())):
                labels = updated
                break
            labels = updated

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            groups[labels[record["operation"]["id"]]].append(record)
        return labels, groups

    @staticmethod
    def _operation_fields(operation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in operation.items()
            if key not in {"id", "op", "actor", "seq", "depends_on", "evidence"}
        }

    def finish(self) -> list[dict[str, Any]]:
        labels, groups = self._partition()
        graph: dict[int, set[int]] = {label: set() for label in groups}
        indegree = {label: 0 for label in groups}

        def add_edge(before: int, after: int) -> None:
            if before == after or after in graph[before]:
                return
            graph[before].add(after)
            indegree[after] += 1

        for trace in self.traces.values():
            compact_trace = [labels[identifier] for identifier in trace]
            for before, after in zip(compact_trace, compact_trace[1:]):
                add_edge(before, after)
        for record in self.records:
            current = labels[record["operation"]["id"]]
            for dependency in record["operation"].get("depends_on", []):
                add_edge(labels[dependency], current)

        priority = {
            label: min(record["first_seen"] for record in records)
            for label, records in groups.items()
        }
        ready = [(priority[label], label) for label, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[int] = []
        while ready:
            _position, label = heapq.heappop(ready)
            ordered.append(label)
            for following in sorted(graph[label], key=priority.get):
                indegree[following] -= 1
                if indegree[following] == 0:
                    heapq.heappush(ready, (priority[following], following))
        if len(ordered) != len(groups):
            raise CaptureError(
                "TILELANG_PARAMETRIC_ORDER_UNRESOLVED",
                "lane trace grouping introduced a cyclic template order",
            )

        template_ids = {
            label: f"tl:template:{order:08d}:{_stable_path((groups[label][0]['initial_key'],))}"
            for order, label in enumerate(ordered)
        }
        templates: list[dict[str, Any]] = []
        for order, label in enumerate(ordered):
            records = sorted(groups[label], key=lambda item: item["instances"])
            representative = records[0]
            operation = representative["operation"]
            scope = representative["scope"]
            if scope == "COLLECTIVE":
                instances = list(representative["instances"])
                field_rows = [self._operation_fields(operation)]
            else:
                by_instance: dict[int, dict[str, Any]] = {}
                for record in records:
                    for instance in record["instances"]:
                        if instance in by_instance:
                            raise CaptureError(
                                "TILELANG_PARAMETRIC_INSTANCE_DUPLICATE",
                                "one compact operation contains duplicate lane evidence",
                                {"operation": operation["op"], "instance": instance},
                            )
                        by_instance[instance] = self._operation_fields(record["operation"])
                instances = sorted(by_instance)
                field_rows = [by_instance[instance] for instance in instances]

            field_names = set(field_rows[0])
            if any(set(row) != field_names for row in field_rows):
                raise CaptureError(
                    "TILELANG_PARAMETRIC_FIELDS_UNRESOLVED",
                    "grouped lane operations disagree on their field set",
                    {"operation": operation["op"]},
                )
            fields: dict[str, Any] = {}
            instance_fields: dict[str, list[Any]] = {}
            for name in sorted(field_names):
                values = [row[name] for row in field_rows]
                if len(values) == 1 or all(value == values[0] for value in values[1:]):
                    fields[name] = values[0]
                else:
                    instance_fields[name] = values

            dependency_pairs: set[tuple[int, str]] = set()
            for record in records:
                for dependency in record["operation"].get("depends_on", []):
                    target = self.record_by_id[dependency]
                    dependency_pairs.add((labels[dependency], self._relation(record, target)))
            template = {
                "id": template_ids[label],
                "op": operation["op"],
                "order": order,
                "execution_scope": scope,
                "instances": instances,
                "fields": fields,
                "evidence": deepcopy(operation["evidence"]),
            }
            if representative["owner"] is not None:
                template["owner"] = representative["owner"]
            if instance_fields:
                template["instance_fields"] = instance_fields
            if dependency_pairs:
                template["dependencies"] = [
                    {"template": template_ids[target], "relation": relation}
                    for target, relation in sorted(
                        dependency_pairs,
                        key=lambda item: (template_ids[item[0]], item[1]),
                    )
                ]
            templates.append(template)
        return templates


class GroupedLaneWalker:
    """Walk one static TIR control-flow tree for a set of concrete lane states."""

    _TRANSPARENT_ATTRIBUTES = {
        "buffer_dim_align",
        "double_buffer_scope",
        "lexical_alloc_scope",
        "tl.min_blocks_per_sm",
        "pipeline_exec_scope",
        "pipeline_scope",
        "pragma_auto_unroll_max_step",
        "pragma_unroll_explicit",
        "realize_scope",
        "reduce_scope",
        "software_pipeline_order",
        "software_pipeline_stage",
        "storage_scope",
        "tl.assume",
        "kWarpSpecializationScope",
    }

    def __init__(
        self,
        capture: "TileLangCapture",
        walkers: list["LaneWalker"],
        builder: CompactTraceBuilder,
    ):
        self.capture = capture
        self.walkers = walkers
        self.builder = builder

    def _leaf(
        self,
        statement: Any,
        path: tuple[Any, ...],
        walkers: list["LaneWalker"],
    ) -> None:
        collective_ids: list[tuple[str, tuple[int, ...], int]] = []
        leaf_lanes = tuple(walker.lane for walker in walkers)
        for walker in walkers:
            self.capture.operations = []
            walker.leaf_lanes = leaf_lanes
            try:
                walker.walk(statement, path)
            finally:
                walker.leaf_lanes = None
            emitted = self.capture.operations
            self.builder.add_trace(walker.lane, emitted)
            for operation in emitted:
                actor = operation["actor"]
                actor_info = self.builder.actor_map[actor]
                if actor_info[0] == "WARPGROUP":
                    collective_ids.append((operation["id"], actor_info[1], walker.lane))
        self.capture.operations = []
        for operation_id, participants, emitting_lane in collective_ids:
            for lane in participants:
                if lane != emitting_lane:
                    self.builder.add_marker(lane, operation_id)

    @staticmethod
    def _save_binding(walker: "LaneWalker", variable: Any) -> tuple[bool, int | None]:
        return variable in walker.bindings, walker.bindings.get(variable)

    @staticmethod
    def _restore_binding(
        walker: "LaneWalker",
        variable: Any,
        saved: tuple[bool, int | None],
    ) -> None:
        had_old, old = saved
        if had_old:
            assert old is not None
            walker.bindings[variable] = old
        else:
            walker.bindings.pop(variable, None)

    def walk(
        self,
        statement: Any,
        path: tuple[Any, ...],
        walkers: list["LaneWalker"] | None = None,
    ) -> None:
        if statement is None:
            return
        active = self.walkers if walkers is None else walkers
        if not active:
            return
        name = type(statement).__name__
        if name == "SeqStmt":
            for index, child in enumerate(statement.seq):
                self.walk(child, (*path, "seq", index), active)
            return
        if name == "AttrStmt":
            if statement.attr_key == "thread_extent":
                tag = str(getattr(statement.node, "thread_tag", ""))
                variable = getattr(statement.node, "var", None)
                block_dimensions = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
                axis = block_dimensions.get(tag)
                saved = [(walker, self._save_binding(walker, variable)) for walker in active]
                for walker in active:
                    extent = walker.eval_int(statement.value)
                    if extent is None and axis is not None and self.capture.launch_grid_explicit:
                        extent = self.capture.launch_grid[axis]
                    if extent is None or variable is None:
                        raise CaptureError(
                            "TILELANG_PARTICIPANTS_UNRESOLVED",
                            "thread binding is not statically resolvable",
                            {"tag": tag, "path": list(path)},
                        )
                    if tag in {"threadIdx.x", "threadIdx.y", "threadIdx.z"}:
                        self.capture.thread_variables[variable] = tag
                        walker.bindings[variable] = walker.thread_coordinate(tag)
                    elif axis is not None:
                        if extent != self.capture.launch_grid[axis]:
                            raise CaptureError(
                                "TILELANG_GRID_EVIDENCE_INVALID",
                                "block binding extent disagrees with launch-grid evidence",
                                {"tag": tag, "extent": extent, "launch_grid": list(self.capture.launch_grid)},
                            )
                        walker.bindings[variable] = self.capture.selected_cta[axis]
                        self.capture.note_block_variable(axis, variable)
                    else:
                        raise CaptureError(
                            "TILELANG_THREAD_BINDING_UNSUPPORTED",
                            "thread binding tag is unsupported",
                            {"tag": tag},
                        )
                self.walk(statement.body, (*path, "attr", tag), active)
                for walker, old in saved:
                    self._restore_binding(walker, variable, old)
                return
            if statement.attr_key == "warp_specialize":
                if any(walker.eval_int(statement.value) != 1 for walker in active):
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "warp_specialize attribute must be statically enabled",
                        {"value": str(statement.value)},
                    )
                self.walk(statement.body, (*path, "attr", "warp_specialize"), active)
                return
            if statement.attr_key == "threadblock_swizzle_pattern":
                remapped = self.capture.swizzled_block_bindings(statement.value)
                saved = [
                    (walker, variable, self._save_binding(walker, variable))
                    for walker in active
                    for variable in remapped
                ]
                for walker in active:
                    walker.bindings.update(remapped)
                self.walk(statement.body, (*path, "attr", statement.attr_key), active)
                for walker, variable, old in saved:
                    self._restore_binding(walker, variable, old)
                return
            if statement.attr_key in self._TRANSPARENT_ATTRIBUTES:
                self.walk(statement.body, (*path, "attr", statement.attr_key), active)
                return
            self._leaf(statement, path, active)
            return
        if name in {"AllocBuffer", "DeclBuffer"}:
            for walker in active:
                walker.buffer_allocation(statement.buffer)
            body = getattr(statement, "body", None)
            if body is not None:
                self.walk(body, (*path, "buffer", statement.buffer.name), active)
            return
        if name == "LetStmt":
            saved = [(walker, self._save_binding(walker, statement.var)) for walker in active]
            opaque = []
            for walker in active:
                previous_lanes = walker.leaf_lanes
                walker.leaf_lanes = tuple(w.lane for w in active)
                try:
                    value = walker.eval_int(statement.value)
                finally:
                    walker.leaf_lanes = previous_lanes
                if value is None:
                    if str(statement.var.dtype) == "handle":
                        raise CaptureError(
                            "TILELANG_STATIC_VALUE_UNRESOLVED",
                            "TIRX LetStmt value is not statically resolvable",
                            {"variable": str(statement.var), "value": str(statement.value)},
                        )
                    # A data value stays opaque; see LaneWalker's Bind rule.
                    opaque.append(walker)
                    walker.bindings.pop(statement.var, None)
                else:
                    walker.bindings[statement.var] = value
            if opaque:
                self._leaf(
                    self.capture.tirx.Evaluate(statement.value),
                    (*path, "let-value"),
                    opaque,
                )
            self.walk(statement.body, (*path, "let", statement.var.name), active)
            for walker, old in saved:
                self._restore_binding(walker, statement.var, old)
            return
        if name == "AssertStmt":
            for walker in active:
                condition = walker.eval_int(statement.condition)
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
            self.walk(getattr(statement, "body", None), (*path, "assert"), active)
            return
        if name == "IfThenElse":
            branches: dict[bool, list[LaneWalker]] = {False: [], True: []}
            for walker in active:
                condition = walker.eval_int(statement.condition)
                if condition is None and walker._register_only_branch(statement):
                    # Register-only arms hold no synchronization: each lane
                    # walks the branch as one leaf.
                    self._leaf(statement, path, active)
                    return
                if condition is None:
                    raise CaptureError(
                        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "TileLang branch condition cannot be evaluated per lane",
                        {"condition": str(statement.condition), "lane": walker.lane},
                    )
                branches[bool(condition)].append(walker)
            if branches[True]:
                self.walk(statement.then_case, (*path, "if", 1), branches[True])
            if branches[False]:
                self.walk(statement.else_case, (*path, "if", 0), branches[False])
            return
        if name == "While":
            remaining = list(active)
            seen_states: dict[int, set[StaticLoopState]] = defaultdict(set)
            iteration = 0
            while remaining:
                continuing = []
                for walker in remaining:
                    condition = walker.eval_int(statement.condition)
                    if condition is None:
                        raise CaptureError(
                            "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                            "TileLang while condition cannot be evaluated statically",
                            {"condition": str(statement.condition), "iteration": iteration},
                        )
                    if not condition:
                        continue
                    state = walker.static_loop_state()
                    if state in seen_states[walker.lane]:
                        raise CaptureError(
                            "TILELANG_NONFINITE_CONTROL_FLOW",
                            "TileLang while loop repeats a static state",
                            {"condition": str(statement.condition), "iteration": iteration},
                        )
                    seen_states[walker.lane].add(state)
                    continuing.append(walker)
                if not continuing:
                    break
                self.capture.static_expansions += len(continuing)
                self.walk(statement.body, (*path, "while", iteration), continuing)
                remaining = continuing
                iteration += 1
            return
        if name == "For":
            groups: dict[tuple[int, int], list[LaneWalker]] = defaultdict(list)
            for walker in active:
                minimum = walker.eval_int(statement.min)
                extent = walker.eval_int(statement.extent)
                if minimum is None or extent is None or (extent < 0 and not is_empty_signed_cuda_loop(statement, minimum, extent)):
                    raise CaptureError(
                        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "TileLang loop must have a finite static extent",
                        {"loop": str(statement.loop_var), "extent": str(statement.extent)},
                    )
                groups[(minimum, extent)].append(walker)
            variable = statement.loop_var
            for (minimum, extent), group in groups.items():
                saved = [(walker, self._save_binding(walker, variable)) for walker in group]
                for iteration in range(minimum, minimum + extent):
                    self.capture.static_expansions += len(group)
                    for walker in group:
                        walker.bindings[variable] = iteration
                    self.walk(
                        statement.body,
                        (*path, "for", variable.name, iteration),
                        group,
                    )
                for walker, old in saved:
                    self._restore_binding(walker, variable, old)
            return
        if name in {"Block", "SBlock"}:
            if statement.init is not None:
                self.walk(statement.init, (*path, "block", statement.name_hint, "init"), active)
            self.walk(statement.body, (*path, "block", statement.name_hint, "body"), active)
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
            saved_by_lane: dict[int, list[tuple[Any, tuple[bool, int | None]]]] = {}
            selected: list[LaneWalker] = []
            initialize: list[LaneWalker] = []
            for walker in active:
                saved_by_lane[walker.lane] = []
                for iter_var, value_expression in zip(iter_vars, iter_values):
                    value = walker.eval_int(value_expression)
                    if value is None:
                        raise CaptureError(
                            "TILELANG_BLOCK_BINDING_UNRESOLVED",
                            "BlockRealize iter value is not statically resolvable",
                            {"block": str(block.name_hint), "value": str(value_expression)},
                        )
                    saved_by_lane[walker.lane].append(
                        (iter_var.var, self._save_binding(walker, iter_var.var))
                    )
                    walker.bindings[iter_var.var] = value
                predicate = walker.eval_int(statement.predicate)
                if predicate is None:
                    raise CaptureError(
                        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "BlockRealize predicate is not statically resolvable",
                        {"block": str(block.name_hint), "predicate": str(statement.predicate)},
                    )
                if predicate:
                    selected.append(walker)
                    run_init = block.init is not None
                    for iter_var in iter_vars:
                        if int(iter_var.iter_type) != self.capture.tirx.IterVar.CommReduce:
                            continue
                        minimum = walker.eval_int(iter_var.dom.min) if iter_var.dom is not None else None
                        if minimum is None:
                            raise CaptureError(
                                "TILELANG_BLOCK_INIT_UNRESOLVED",
                                "reduction block minimum is not statically resolvable",
                                {"block": str(block.name_hint)},
                            )
                        run_init = run_init and walker.bindings[iter_var.var] == minimum
                    if run_init:
                        initialize.append(walker)
            if initialize:
                self.walk(block.init, (*path, "block", block.name_hint, "init"), initialize)
            if selected:
                self.walk(block.body, (*path, "block", block.name_hint, "body"), selected)
            for walker in active:
                for variable, old in reversed(saved_by_lane[walker.lane]):
                    self._restore_binding(walker, variable, old)
            return
        if name in {"Bind", "BufferStore", "Evaluate"}:
            self._leaf(statement, path, active)
            return
        self._leaf(statement, path, active)


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
        binding_evidence: dict[str, Any] | None = None,
        launch_grid: tuple[int, int, int] = (1, 1, 1),
        selected_cta: tuple[int, int, int] = (0, 0, 0),
        launch_grid_explicit: bool = False,
    ):
        self.tvm = tvm
        self.tirx = tirx
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        self.descriptor_evidence = descriptor_evidence or {}
        self.binding_evidence = binding_evidence or {}
        self.launch_grid = launch_grid
        self.selected_cta = selected_cta
        self.launch_grid_explicit = launch_grid_explicit
        if launch_grid[0] * launch_grid[1] * launch_grid[2] > 1:
            print(
                "WARNING: CROSS_CTA_OUT_OF_SCOPE: launch grid "
                f"{list(launch_grid)} has multiple CTAs but capture analyzes the "
                f"single selected CTA {list(selected_cta)}; verdicts are per-CTA "
                "and cross-CTA conflicts (e.g. global write/write races between "
                "blocks) are NOT checked.",
                file=sys.stderr,
            )
        self.scalar_bindings: dict[Any, int] = {}
        # blockIdx.{x,y,z} variables, in axis order once all are bound.
        self.block_variables: list[Any] = []
        self.thread_variables: dict[Any, str] = {}
        self.pointer_table_allocations: dict[tuple[str, int], str] = {}
        functions = [func for func in module.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(functions) != 1:
            raise CaptureError(
                "TILELANG_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one lowered TileLang PrimFunc",
                {"primfuncs": len(functions)},
            )
        self.func = functions[0]
        params_by_name = {parameter.name: parameter for parameter in self.func.params}
        self.scalar_bindings = {
            params_by_name[name]: value
            for name, value in self.binding_evidence.get("scalar_values", {}).items()
            if name in params_by_name
        }
        self.pointer_table_allocations = {
            (table, index): allocation
            for table, allocations in self.binding_evidence.get(
                "pointer_table_allocations", {}
            ).items()
            for index, allocation in enumerate(allocations)
        }
        self.buffer_values = {
            name: tuple(values)
            for name, values in self.binding_evidence.get("buffer_values", {}).items()
        }
        self._validate_constant_inputs(params_by_name)
        self.thread_extents = self._thread_extents()
        self.thread_shape = tuple(self.thread_extents.get(f"threadIdx.{axis}", 1)
                                  for axis in "xyz")
        self.cta_threads = math.prod(self.thread_shape)
        if (not any(f"threadIdx.{axis}" in self.thread_extents for axis in "xyz")
                or any(extent <= 0 for extent in self.thread_shape)
                or not 1 <= self.cta_threads <= 1024):
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "static CUDA thread dimensions must have a product in [1, 1024]",
                {"thread_extents": self.thread_extents},
            )
        for axis, dimension in enumerate(("blockIdx.x", "blockIdx.y", "blockIdx.z")):
            extent = self.thread_extents.get(dimension, 1)
            if extent != self.launch_grid[axis]:
                code = (
                    "TILELANG_GRID_EVIDENCE_INVALID"
                    if self.launch_grid_explicit
                    else "TILELANG_MULTI_CTA_UNSUPPORTED"
                )
                raise CaptureError(
                    code,
                    "lowered block extent disagrees with the selected launch grid",
                    {
                        "dimension": dimension,
                        "thread_extent": extent,
                        "launch_grid": list(self.launch_grid),
                    },
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
            name.startswith("tl.warpgroup_") or name in {"tl.ptx_wgmma_ss", "tl.ptx_wgmma_rs"}
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
        self.wgmma_descriptors: dict[tuple[str, Any], str] = {}
        self.wgmma_descriptor_bases: dict[tuple[str, Any], str] = {}
        self.wgmma_descriptor_stage_bytes: dict[tuple[str, Any], int] = {}
        self.wgmma_descriptor_windows: dict[tuple[str, Any], dict[str, Any]] = {}
        self.tma_destination_groups: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self.wgmma_group_seq: dict[str, int] = {}
        self.wgmma_engine_seq: dict[str, int] = {}
        self.wgmma_descriptor_metadata: dict[tuple[str, Any], dict[str, Any]] = {}
        self.wgmma_pending_proof_issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.wgmma_proof_groups: list[dict[str, Any]] = []
        self.wgmma_proof_diagnostics: list[dict[str, Any]] = []
        self.wgmma_region_proof: dict[str, Any] | None = None
        self.resources: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, dict[str, Any]] = {}
        self.buffer_layouts: dict[Any, tuple[list[int], list[int], int, bool, int]] = {}
        self.operations: list[dict[str, Any]] = []
        # fence_proxy_async placeholders (all lanes), one per async-proxy
        # domain, each awaiting that domain's next async consumer.
        self.pending_proxies: list[tuple[str, dict[str, Any]]] = []
        self.bound_proxies: list[dict[str, Any]] = []
        self.discarded_proxies: set[str] = set()
        self.static_expansions = 0
        self.executable_nodes_seen = 0
        self.classified_nodes = 0
        self.emitting_nodes = 0
        self.elided_nodes = 0
        self.allocation_ids: dict[Any, str] = {}
        self.used_allocation_ids: set[str] = set()
        self.global_allocation_keys: set[Any] = set()
        self.tma_global_allocations: set[str] = set()
        self.tma_zero_fill_count = 0
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

    def _validate_constant_inputs(self, params_by_name: dict[str, Any]) -> None:
        # Recorded tensor contents specialize read-only global inputs. Never
        # substitute them for mutable local/shared state or an escaping pointer.
        for name, values in self.buffer_values.items():
            parameter = params_by_name.get(name)
            if parameter is None or str(parameter.dtype) != "handle":
                raise CaptureError("TILELANG_BINDING_EVIDENCE_INVALID", "constant tensor must name a pointer parameter", {"buffer": name})
            loads, invalid = [], []
            def contains_pointer(expression: Any) -> bool:
                found = []
                self.tirx.stmt_functor.post_order_visit(expression, lambda n: found.append(n) if isinstance(n, self.tirx.Var) and n.same_as(parameter) else None)
                return bool(found)
            def visit(node: Any) -> None:
                if isinstance(node, (self.tirx.BufferLoad, self.tirx.BufferStore)) and node.buffer.data.same_as(parameter):
                    if isinstance(node, self.tirx.BufferStore):
                        invalid.append("written input")
                    else:
                        loads.append(node.buffer)
                        if node.buffer.scope() != "global" or node.buffer.name != name or len(node.buffer.shape) != 1:
                            invalid.append("noncanonical buffer view")
                        extent = node.buffer.shape[0]
                        if isinstance(extent, self.tirx.IntImm) and int(extent) != len(values):
                            invalid.append("input value count differs from buffer extent")
                elif isinstance(node, self.tirx.Call) and contains_pointer(node):
                    invalid.append("pointer escapes through a call")
                elif type(node).__name__ in {"Bind", "LetStmt"} and contains_pointer(node.value):
                    invalid.append("pointer alias is not proven read-only")
            self.tirx.stmt_functor.post_order_visit(self.func.body, visit)
            if invalid or not loads:
                raise CaptureError("TILELANG_BINDING_EVIDENCE_INVALID", "constant tensor requires exact read-only scalar loads", {"buffer": name, "reasons": sorted(set(invalid)), "loads": len(loads)})

    def _thread_extents(self) -> dict[str, int]:
        attrs = dict(self.func.attrs) if self.func.attrs else {}
        raw = attrs.get("thread_extent")
        if raw is None or not hasattr(raw, "items"):
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "lowered PrimFunc lacks static thread_extent metadata",
            )
        # An explicitly empty launch map on a literal no-op device function
        # has no participant-dependent effects. One canonical actor represents
        # its empty trace; never invent thread bindings for a real body.
        if not raw and isinstance(self.func.body, self.tirx.Evaluate) and isinstance(self.func.body.value, self.tirx.IntImm):
            return {"threadIdx.x": 1}
        result: dict[str, int] = {}
        block_dimensions = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
        for key, value in raw.items():
            dimension = str(key)
            try:
                extent = int(value)
            except (TypeError, ValueError) as error:
                axis = block_dimensions.get(dimension)
                if axis is None or not self.launch_grid_explicit:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "thread extent is not an integer",
                        {"dimension": dimension, "value": str(value)},
                    ) from error
                extent = self.launch_grid[axis]
                if isinstance(value, self.tirx.Var):
                    self.scalar_bindings[value] = extent
            result[dimension] = extent
        return result

    def capture(self) -> dict[str, Any]:
        builder = CompactTraceBuilder(self)
        walkers = [LaneWalker(self, lane) for lane in range(self.cta_threads)]
        GroupedLaneWalker(self, walkers, builder).walk(self.func.body, ("root",))
        for walker in walkers:
            walker.finalize()
        # A fence with no later async consumer orders nothing.
        builder.discard(
            self.discarded_proxies
            | {operation["id"] for _domain, operation in self.pending_proxies}
        )
        self.pending_proxies = []
        # A proxy fence covers all CTA shared memory, including allocations
        # and offsets first referenced by a later async consumer. Finalize only
        # after the complete trace has exposed those regions.
        shared_ends: dict[str, int] = {}
        for region in self.regions.values():
            if region["address_space"] == "SHARED":
                allocation = region["allocation_id"]
                shared_ends[allocation] = max(
                    shared_ends.get(allocation, 0), region["offset"] + region["extent"]
                )
        shared_regions = [self.region(allocation, "shared", 0, end)
                          for allocation, end in sorted(shared_ends.items())]
        for operation in self.bound_proxies:
            operation["regions"] = shared_regions
        self.shared_memory_bytes = max(SHARED_WINDOW_BYTES, sum(shared_ends.values()))
        self.finalize_wgmma_region_proof()
        compact_operations = builder.finish()
        logical_events = sum(
            len(operation["instances"])
            for operation in compact_operations
        )
        self.operations = []
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
                "shared_memory_bytes": self.shared_memory_bytes,
                "grid_scope": "single_cta",
                **({
                    "launch_grid": list(self.launch_grid),
                    "selected_cta": list(self.selected_cta),
                } if self.launch_grid_explicit else {}),
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
                "operation_records": len(compact_operations),
                "logical_events_emitted": logical_events,
                "allocation_identity_evidence": "tir-buffer-data-object-and-actor-identity-v1",
                "alias_evidence": (
                    "tirx.noalias-plus-sha-bound-runtime-evidence-v1"
                    if self.descriptor_evidence or self.pointer_table_allocations
                    else "tirx.noalias-v1" if self.noalias else "single-global-base-v1"
                ),
                "participant_evidence": "primfunc-thread-extent-lane-enumeration-v1",
                "lane_ownership_evidence": "tir-thread-binding-active-lane-evaluation-v1",
                "predicate_evidence": "tir-static-predicate-active-lane-splitting-v1",
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
            "operations": compact_operations,
            "unsupported_constructs": [],
        }

    def record_wgmma_issue_proof(
        self,
        owner: str,
        issue_template_id: str,
        engine_seq: int,
        operands: list[dict[str, Any]],
    ) -> None:
        self.wgmma_pending_proof_issues[owner].append({
            "issue_template_id": issue_template_id,
            "engine_seq": engine_seq,
            "operands": operands,
        })

    def record_wgmma_commit_proof(
        self,
        owner: str,
        resource: str,
        group_seq: int,
        commit_template_id: str,
    ) -> None:
        issues = self.wgmma_pending_proof_issues.pop(owner, [])
        try:
            self.wgmma_proof_groups.append(_prove_wgmma_group(
                owner, resource, group_seq, commit_template_id, issues
            ))
        except CaptureError as error:
            self.wgmma_proof_diagnostics.append(error.diagnostic())

    def finalize_wgmma_region_proof(self) -> None:
        for owner, issues in sorted(self.wgmma_pending_proof_issues.items()):
            if issues:
                self.wgmma_proof_diagnostics.append(CaptureError(
                    "TILELANG_WGMMA_UNCOMMITTED_ISSUES",
                    "WGMMA issues remain outside a committed group",
                    {
                        "owner": owner,
                        "issue_template_ids": [item["issue_template_id"] for item in issues],
                    },
                ).diagnostic())
        issue_count = sum(len(group["issue_template_ids"]) for group in self.wgmma_proof_groups)
        if self.wgmma_proof_diagnostics:
            status = "UNSUPPORTED"
        elif self.wgmma_proof_groups:
            status = "PROVED"
        else:
            status = "NOT_APPLICABLE"
        self.wgmma_region_proof = {
            "schema_version": WGMMA_PROOF_VERSION,
            "claim": "committed-group-actual-issue-region-union-equals-selected-stage-regions",
            "status": status,
            "scope": {
                "dsl": "tilelang",
                "compiler_ir_stage": "post-inject-fence-proxy",
                "source": self.source,
                "selected_cta": list(self.selected_cta),
                "target": self.target_rule,
            },
            "bindings": {
                "source_ir_sha256": self.source_ir_sha256,
                "adapter_version": ADAPTER_VERSION,
            },
            "groups": sorted(
                self.wgmma_proof_groups,
                key=lambda group: (group["owner"], group["group_seq"]),
            ),
            "diagnostics": self.wgmma_proof_diagnostics,
            "summary": {
                "groups": len(self.wgmma_proof_groups),
                "issues": issue_count,
                "proved_groups": sum(group["status"] == "PROVED" for group in self.wgmma_proof_groups),
                "failed_groups": len(self.wgmma_proof_diagnostics),
            },
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

    def named_barrier(
        self,
        path: tuple[Any, ...],
        participant_lanes: tuple[int, ...],
    ) -> str:
        identifier = f"named:{_stable_path(path)}"
        participants = [self.actor_ids[lane] for lane in participant_lanes]
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "named_barrier",
            "participants": participants,
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
        return identifier

    def async_group(self, actor: str) -> str:
        identifier = f"cp_async:{actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [actor],
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
            "protocol": "WGMMA",
            "scope": "CTA",
        })
        return identifier

    def record_tma_destination_group(
        self,
        segments: list[tuple[str, str, int]],
    ) -> None:
        destinations = [self.regions[destination] for _source, destination, _bytes in segments]
        if not destinations:
            return
        allocation = destinations[0]["allocation_id"]
        if any(
            region["allocation_id"] != allocation
            or region["address_space"] != "SHARED"
            for region in destinations
        ):
            return
        intervals = sorted(
            (region["offset"], region["offset"] + region["extent"])
            for region in destinations
        )
        start, end = intervals[0]
        for interval_start, interval_end in intervals[1:]:
            if interval_start > end:
                return
            end = max(end, interval_end)
        self.tma_destination_groups[allocation].add((start, end))

    def select_wgmma_descriptor_stage(
        self,
        key: tuple[str, Any],
        base_id: str,
        byte_offset: int,
    ) -> str | None:
        base = self.regions[base_id]
        target_offset = base["offset"] + byte_offset
        matches = {
            (start, end)
            for start, end in self.tma_destination_groups.get(
                base["allocation_id"], set()
            )
            if start == target_offset
            and end <= base["offset"] + base["extent"]
        }
        extents = {end - start for start, end in matches}
        known_extent = self.wgmma_descriptor_stage_bytes.get(key)
        if known_extent is not None:
            extents = {extent for extent in extents if extent == known_extent}
        if len(extents) != 1:
            # No exact TMA stage (e.g. cp.async or plain-store filled tiles):
            # each issue's footprint is bounded from the descriptor layout.
            if not 0 <= byte_offset < base["extent"]:
                raise CaptureError(
                    "TILELANG_WGMMA_OPERAND_REGION_UNRESOLVED",
                    "WGMMA descriptor offset leaves its base region",
                    {"region": base_id, "byte_offset": byte_offset},
                )
            self.wgmma_descriptor_windows[key] = {
                "allocation_id": base["allocation_id"],
                "base_offset": base["offset"],
                "base_end": base["offset"] + base["extent"],
                "start": target_offset,
            }
            return None
        self.wgmma_descriptor_windows.pop(key, None)
        stage_bytes = next(iter(extents))
        self.wgmma_descriptor_stage_bytes[key] = stage_bytes
        return self.region(
            base["allocation_id"],
            "shared",
            target_offset,
            stage_bytes,
        )

    def wgmma_layout_region(
        self,
        key: tuple[str, Any],
        *,
        shape: str,
        dtype: str,
        operand: str,
        k_major: bool,
        descriptor_offset_units: int,
    ) -> str:
        window = self.wgmma_descriptor_windows[key]
        metadata = self.wgmma_descriptor_metadata.get(key, {})
        details = {
            "shape": shape,
            "dtype": dtype,
            "operand": operand,
            "k_major": k_major,
            "descriptor_offset_units_16b": descriptor_offset_units,
            "layout_type": metadata.get("layout_type"),
            "leading_byte_offset": metadata.get("leading_byte_offset"),
            "stride_byte_offset": metadata.get("stride_byte_offset"),
            "window": window,
        }
        match = _WGMMA_SHAPE.fullmatch(shape)
        element_bytes = _WGMMA_LAYOUT_DTYPE_BYTES.get(dtype)
        if match is None or element_bytes is None:
            raise CaptureError(
                "TILELANG_WGMMA_OPERAND_REGION_UNRESOLVED",
                "WGMMA layout footprint requires a known shape and operand dtype",
                details,
            )
        m, n, k = map(int, match.groups())
        hull = _wgmma_layout_hull(
            mn=m if operand == "A" else n,
            k=k,
            element_bytes=element_bytes,
            k_major=k_major,
            layout_type=metadata.get("layout_type"),
            leading=metadata.get("leading_byte_offset"),
            stride=metadata.get("stride_byte_offset"),
            start=window["start"] + 16 * descriptor_offset_units,
        )
        if hull is not None and hull[1] > window["base_end"]:
            # The swapped-role hedge reaches past the descriptor's base
            # buffer; the PTX/CuTe roles (MN-major: LBO strides MN atoms, SBO
            # strides 8-row K groups) bound the read.
            hull = _wgmma_layout_hull(
                mn=m if operand == "A" else n,
                k=k,
                element_bytes=element_bytes,
                k_major=k_major,
                layout_type=metadata.get("layout_type"),
                leading=metadata.get("leading_byte_offset"),
                stride=metadata.get("stride_byte_offset"),
                start=window["start"] + 16 * descriptor_offset_units,
                hedge_swapped=False,
            )
        if (
            hull is None
            or hull[0] < window["base_offset"]
            or hull[1] > window["base_end"]
        ):
            raise CaptureError(
                "TILELANG_WGMMA_OPERAND_REGION_UNRESOLVED",
                "WGMMA layout footprint is not bounded by the descriptor base region",
                {**details, "hull": list(hull) if hull else None},
            )
        return self.region(
            window["allocation_id"], "shared", hull[0], hull[1] - hull[0]
        )

    def tma_segments(
        self,
        descriptor: Any,
        coordinates: list[int],
        destination: str,
        *,
        store: bool = False,
    ) -> list[tuple[str, str, int]]:
        # Returns (global, shared, bytes) runs; `destination` names the
        # shared tile, which is the source when `store` is set.
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
        if allocation not in self.tma_global_allocations:
            # An evidenced descriptor base is another global pointer base.
            if self.global_allocation_keys and not self.noalias:
                raise CaptureError(
                    "TILELANG_ALIAS_UNRESOLVED",
                    "multiple global pointer bases require tirx.noalias evidence",
                    {"allocation": allocation},
                )
            self.global_allocation_keys.add(("tma-descriptor", allocation))
            self.tma_global_allocations.add(allocation)
        # Out-of-bounds box elements are zero-filled without a global read;
        # they come from a private per-transfer source instead.
        zero_fill = f"tma-zero-fill:{name}:{self.tma_zero_fill_count}"
        if not store:
            self.tma_zero_fill_count += 1
        runs: list[list[Any]] = []
        destination_offset = destination_region["offset"]
        for logical_index in product(*(range(dimension) for dimension in tile_shape)):
            descriptor_index = [
                coordinate + index for coordinate, index in zip(coordinates, logical_index)
            ]
            in_bounds = all(
                0 <= index < dimension for index, dimension in zip(descriptor_index, shape)
            )
            if store and not in_bounds:
                # A TMA store drops out-of-bounds box elements.
                destination_offset += element_bytes
                continue
            source_allocation = allocation if in_bounds else zero_fill
            source_offset = (
                sum(index * stride for index, stride in zip(descriptor_index, strides))
                * element_bytes
                if in_bounds
                else destination_offset - destination_region["offset"]
            )
            if (
                runs
                and source_allocation == runs[-1][0]
                and source_offset == runs[-1][1] + runs[-1][3]
                and destination_offset == runs[-1][2] + runs[-1][3]
            ):
                runs[-1][3] += element_bytes
            else:
                runs.append([source_allocation, source_offset, destination_offset, element_bytes])
            destination_offset += element_bytes
        return [
            (
                self.region(source_allocation, "global", source_offset, extent),
                self.region(
                    destination_region["allocation_id"],
                    "shared",
                    destination_offset,
                    extent,
                ),
                extent,
            )
            for source_allocation, source_offset, destination_offset, extent in runs
        ]

    def note_block_variable(self, axis: int, variable: Any) -> None:
        if len(self.block_variables) == axis:
            self.block_variables.append(variable)
        elif axis >= len(self.block_variables) or not self.block_variables[axis].same_as(variable):
            raise CaptureError(
                "TILELANG_THREAD_BINDING_UNSUPPORTED",
                "blockIdx bindings must be unique and bound in axis order",
                {"axis": axis},
            )

    def swizzled_block_bindings(self, value: Any) -> dict[Any, int]:
        # threadblock_swizzle_pattern rebinds blockIdx for its body to
        # tl::rasterization2D{Row,Column}<panel> of the selected CTA
        # (tl_templates/cuda/threadblock_swizzle.h).
        arguments = list(getattr(value, "args", ()))
        pattern = getattr(arguments[0], "value", None) if arguments else None
        panel = int(arguments[1]) if len(arguments) == 2 and isinstance(
            arguments[1], self.tirx.IntImm
        ) else 0
        attrs = getattr(self.func, "attrs", None) or {}
        if (
            getattr(value, "op", None) is None
            or getattr(value.op, "name", "") not in {"tir.tvm_tuple", "tirx.tvm_tuple"}
            or pattern not in {"rasterization2DRow", "rasterization2DColumn"}
            or panel <= 0
            or "cluster_dims" in attrs
            or not self.block_variables
        ):
            raise CaptureError(
                "TILELANG_ATTRIBUTE_UNSUPPORTED",
                "threadblock swizzle pattern is unsupported",
                {"attribute": "threadblock_swizzle_pattern", "value": str(value)},
            )
        grid_x, grid_y, _grid_z = self.launch_grid
        block_x, block_y, block_z = self.selected_cta
        block = block_x + block_y * grid_x
        grid = grid_x * grid_y
        major = grid_x if pattern == "rasterization2DRow" else grid_y
        panel_size = panel * major
        panel_offset = block % panel_size
        panel_index = block // panel_size
        total_panels = (grid + panel_size - 1) // panel_size
        stride = panel if panel_index + 1 < total_panels else (
            grid - panel_index * panel_size
        ) // major
        across = panel_offset // stride
        if panel_index & 1:
            across = major - 1 - across
        along = panel_offset % stride + panel_index * panel
        column, row = (across, along) if pattern == "rasterization2DRow" else (along, across)
        remapped = (column, row, block_z)
        if any(remapped[axis] for axis in range(len(self.block_variables), 3)):
            raise CaptureError(
                "TILELANG_ATTRIBUTE_UNSUPPORTED",
                "threadblock swizzle remaps an unbound blockIdx axis",
                {"attribute": "threadblock_swizzle_pattern", "remapped": list(remapped)},
            )
        return dict(zip(self.block_variables, remapped))

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
        self.bindings: dict[Any, int] = dict(capture.scalar_bindings)
        self.pointer_aliases: dict[Any, tuple[Any, int]] = {}
        self.scalar_buffer_values: dict[tuple[Any, int], int] = {}
        self.sequence = 0
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}
        self.pending_arrive_expect: dict[str, int] = {}
        self.arrival_kinds_since_wait: dict[str, set[str]] = {}
        self.cp_group_seq = 0
        self.cp_staged = 0
        self.tma_store_group_seq = 0
        self.current_ir_operation = "unknown"
        self.wgmma_accumulator_region: str | None = None
        self.wgmma_fence_seen = False
        self.wgmma_arrived = False
        self.active_warpgroup_dependency: str | None = None
        self.last_lane_event_id: str | None = None
        # Lanes that reach the Evaluate leaf being walked, when the grouped
        # walker resolved all enclosing control flow.
        self.leaf_lanes: tuple[int, ...] | None = None

    def static_loop_state(self) -> StaticLoopState:
        # Mutable local counters lower to BufferStore/BufferLoad, so their
        # progress lives outside the ordinary variable bindings. Snapshot both
        # maps: a repeated pair still detects stationary or cyclic loops.
        return (frozenset(self.bindings.items()),
                frozenset(self.scalar_buffer_values.items()))

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
        if self.pending_arrive_expect:
            raise CaptureError(
                "TILELANG_TMA_EXPECT_UNRESOLVED",
                "combined arrive-expect is not followed by a matching TMA issue",
                {"resources": sorted(self.pending_arrive_expect)},
            )

    def bind_pending_proxy(self, to_domain: str, regions: list[str]) -> None:
        pending = [
            operation for domain, operation in self.capture.pending_proxies
            if domain == to_domain
        ]
        if not pending:
            return
        self.capture.pending_proxies = [
            item for item in self.capture.pending_proxies if item[0] != to_domain
        ]
        # tl::fence_proxy_async lowers to fence.proxy.async.shared::cta: it is
        # not region-scoped within the CTA shared window, but it orders only
        # shared memory. Binding the fence only to the first following async
        # operation's exact source bytes under-approximates that guarantee and
        # makes a second async operation after the same fence (e.g. two
        # tma_stores from different offsets of buf_dyn_shmem) spuriously
        # unordered, so expand each SHARED region to its entire allocation.
        # Global and tensor-memory regions get no visibility from this fence.
        expanded = sorted({
            self.capture.region(
                self.capture.regions[region]["allocation_id"],
                "shared",
                0,
                SHARED_WINDOW_BYTES,
            )
            for region in regions
            if self.capture.regions[region]["address_space"] == "SHARED"
        })
        if not expanded:
            self.capture.discarded_proxies.update(operation["id"] for operation in pending)
            return
        for operation in pending:
            self.capture.bound_proxies.append(operation)
            # The placeholder was emitted at the fence's own program position,
            # so the fence orders only generic accesses that precede it. Every
            # lane's pending fence binds to the CTA's next async consumer: the
            # writer's own fence, then a barrier, then an elected issue is
            # the canonical publication pattern.
            operation.update(
                from_domain="GENERIC",
                to_domain=to_domain,
                scope="CTA",
                regions=expanded,
                visibility_mask={
                    "shared": True,
                    "global_read": False,
                    "global_write": False,
                    "tensor_read": False,
                    "tensor_write": False,
                    "control_only": False,
                },
            )

    def emit(self, op: str, path: tuple[Any, ...], **fields: Any) -> None:
        ir_operation = fields.pop("_ir_operation", self.current_ir_operation)
        recognizer_rule = fields.pop(
            "_recognizer_rule",
            f"{ir_operation}->{op}:v1",
        )
        actor = fields.pop("_actor", self.actor)
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
        if isinstance(expression, self.tirx.Call):
            op = getattr(expression.op, "name", "")
            if op in {"tl.activemask", "tirx.tvm_warp_activemask", "tir.tvm_warp_activemask"} and not expression.args:
                if self.leaf_lanes is None:
                    return None
                return sum(1 << (lane % 32) for lane in self.leaf_lanes if lane // 32 == self.lane // 32)
            if op == "tl.ballot_sync" and len(expression.args) == 2:
                mask = self.eval_int(expression.args[0])
                if mask is None or self.leaf_lanes is None:
                    return None
                first = self.lane // 32 * 32
                participants = [lane for lane in range(first, min(first + 32, self.capture.cta_threads)) if mask & (1 << (lane % 32))]
                if self.lane not in participants or not set(participants) <= set(self.leaf_lanes):
                    return None
                nodes = []
                self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression.args[1]), nodes.append)
                allowed = set(self.capture.thread_variables) | set(self.capture.scalar_bindings) | set(self.capture.block_variables)
                if any(isinstance(n, self.tirx.BufferLoad) or isinstance(n, self.tirx.Call)
                       or isinstance(n, self.tirx.Var) and n not in allowed for n in nodes):
                    return None
                saved = dict(self.bindings)
                x, y, _ = self.capture.thread_shape
                bits = 0
                try:
                    for lane in participants:
                        for variable, tag in self.capture.thread_variables.items():
                            self.bindings[variable] = {"threadIdx.x": lane % x, "threadIdx.y": lane // x % y, "threadIdx.z": lane // (x * y)}[tag]
                        predicate = self.eval_int(expression.args[1])
                        if predicate is None:
                            return None
                        if predicate:
                            bits |= 1 << (lane % 32)
                finally:
                    self.bindings = saved
                return bits
        if isinstance(expression, self.tirx.BufferLoad) and len(expression.indices) == 1:
            index = self.eval_int(expression.indices[0])
            if index is not None:
                local = self.scalar_buffer_values.get((str(expression.buffer.data), index))
                if local is not None:
                    return local
                values = self.capture.buffer_values.get(expression.buffer.name)
                if values is not None and 0 <= index < len(values):
                    return values[index]
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
            if call_name in {"tir.if_then_else", "tirx.if_then_else"} and len(expression.args) == 3:
                condition = self.eval_int(expression.args[0])
                if condition is not None:
                    return self.eval_int(expression.args[1] if condition else expression.args[2])
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
        return None

    def eval_pointer(self, expression: Any) -> tuple[Any, int] | None:
        if isinstance(expression, self.tirx.Var):
            return self.pointer_aliases.get(expression, (expression, 0))
        if (
            isinstance(expression, self.tirx.Call)
            and getattr(expression.op, "name", "") in {"tirx.reinterpret", "tir.reinterpret"}
            and len(expression.args) == 1
            and isinstance(expression.args[0], self.tirx.BufferLoad)
        ):
            load = expression.args[0]
            if len(load.indices) != 1:
                return None
            indices = vector_values(load.indices[0], self.eval_int)
            if indices is None or len(indices) != 1:
                return None
            allocation = self.capture.pointer_table_allocations.get(
                (load.buffer.name, indices[0])
            )
            if allocation is None:
                return None
            return allocation, 0
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
        label = data if isinstance(data, str) else getattr(data, "name", None) or buffer.name
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
        # Views (e.g. a dtype-changing T.view) share the data var but not
        # the element width, shape, or strides.
        layout_key = (
            buffer.data,
            str(buffer.dtype),
            tuple(str(item) for item in buffer.shape),
            tuple(str(item) for item in buffer.strides),
            str(buffer.elem_offset),
        )
        layout = self.capture.buffer_layouts.get(layout_key)
        if layout is None:
            raw_shape = list(buffer.shape)
            raw_strides = list(buffer.strides)
            shape_values = [self.eval_int(item) for item in raw_shape]
            stride_values = [self.eval_int(item) for item in raw_strides]
            elem_offset_value = self.eval_int(buffer.elem_offset)
            if any(value is None for value in shape_values + stride_values + [elem_offset_value]):
                shape: list[int | None] = shape_values
                strides: list[int | None] = stride_values
                elem_offset = elem_offset_value
                width = self._dtype_bytes(buffer)
            else:
                shape = [int(value) for value in shape_values]
                strides = [int(value) for value in stride_values]
                elem_offset = int(elem_offset_value)
                width = self._dtype_bytes(buffer)
                self.capture.buffer_layouts[layout_key] = (
                    shape,
                    strides,
                    elem_offset,
                    bool(raw_strides),
                    width,
                )
        else:
            shape, strides, elem_offset, has_strides, width = layout
            raw_strides = strides if has_strides else []
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

    def _ldmatrix_operand(
        self,
        expression: Any,
        required_flag: int,
        hardware_bytes: int,
    ) -> tuple[str, int]:
        """Region of the bytes an ldmatrix operand touches, and the declared
        access-pointer footprint."""
        core, _wrapper_byte_offset = self._access_ptr_parts(expression)
        extent = self.eval_int(core.args[3])
        bits = dtype_bit_width(core.args[0].dtype)
        if extent is None or bits is None or hardware_bytes * 8 % bits:
            return self.access_ptr_region(expression, required_flag)
        declared_bytes = extent * bits // 8
        if declared_bytes < hardware_bytes:
            return "", declared_bytes
        region, _byte_count = self.access_ptr_region(
            expression, required_flag, hardware_bytes * 8 // bits
        )
        return region, declared_bytes

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
        base_label = base if isinstance(base, str) else getattr(base, "name", None) or allocation_label
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

    def thread_coordinate(self, tag: str) -> int:
        # CUDA linear thread ID is x + blockDim.x * (y + blockDim.y * z).
        x, y, _z = self.capture.thread_shape
        return {"threadIdx.x": self.lane % x,
                "threadIdx.y": (self.lane // x) % y,
                "threadIdx.z": self.lane // (x * y)}[tag]

    def _debug_extern(self, call: Any) -> bool:
        if getattr(call.op, "name", "") not in {"tir.call_extern", "tirx.call_extern"}:
            return False
        args = list(call.args)
        if not args or not isinstance(args[0], self.tirx.StringImm):
            return False
        # Pinned cuda/debug.h passes values to printf; it never dereferences
        # user pointers. The string arguments must be literals. BufferLoad
        # arguments are still visited and emitted by _buffer_loads.
        name = args[0].value
        strings = {"debug_print_var": (3, 1),
                   "debug_print_buffer_value": (5, 2),
                   "debug_print_msg": (2, 1)}
        spec = strings.get(name)
        return bool(spec and len(args) == spec[0] and all(
            isinstance(arg, self.tirx.StringImm) for arg in args[1:1 + spec[1]]))

    def _register_rng(self, call: Any) -> bool:
        # cuda/codegen_cuda.cc allocates one curand state at function scope
        # per thread. These operations affect only that private state; values
        # remain opaque if later used as memory addresses or branch guards.
        name = getattr(call.op, "name", "")
        if name == "tl.rng_init":
            return (len(call.args) == 4 and isinstance(call.args[3], self.tirx.StringImm)
                    and all(str(arg.dtype) != "handle" for arg in call.args[:3]))
        if name == "tl.rng_rand":
            return not call.args
        return (name == "tl.rng_rand_float" and len(call.args) == 1
                and isinstance(call.args[0], self.tirx.StringImm)
                and call.args[0].value in {"uniform", "normal"})

    def _cta_vote(self, call: Any) -> bool:
        if getattr(call.op, "name", "") not in {"tl.syncthreads_count", "tl.syncthreads_and", "tl.syncthreads_or"} or len(call.args) != 1:
            return False
        # Keep data-loading predicates unsupported here until their precise
        # evaluation order can be represented alongside the collective.
        memory = []
        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(call.args[0]), lambda n: memory.append(n) if isinstance(n, self.tirx.BufferLoad) else None)
        return not memory

    @staticmethod
    def _allreduce_participants(call: Any) -> tuple[int, int] | None:
        if getattr(call.op, "name", "") not in {"tir.call_extern", "tirx.call_extern"}:
            return None
        arguments = list(call.args)
        if not arguments:
            return None
        callee = str(arguments[0]).strip('"')
        match = re.fullmatch(
            r"tl::AllReduce<[^,>]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*tl::NamedBarrier<(\d+)>>::run",
            callee,
        )
        if match is None:
            return None
        threads, scale, offset, barrier_threads = map(int, match.groups())
        if threads <= 0 or scale <= 0 or threads % scale or (threads // scale) & (threads // scale - 1):
            return None
        # reduce.h instantiates Barrier::sync only for butterfly offsets >=32.
        # A warp-only reduction never invokes its NamedBarrier policy, even
        # when that policy carries the enclosing CTA's larger thread count.
        if threads <= 32 or threads == scale:
            return offset, 0
        return offset, barrier_threads

    def _emit_expression_collectives(
        self,
        expression: Any,
        path: tuple[Any, ...],
        votes_only: bool = False,
    ) -> None:
        calls: list[tuple[Any, int, int]] = []

        def visit(node: Any) -> None:
            if not isinstance(node, self.tirx.Call):
                return
            participant_range = None if votes_only else self._allreduce_participants(node)
            if self._cta_vote(node):
                participant_range = (0, self.capture.cta_threads)
            if participant_range is not None:
                calls.append((node, *participant_range))

        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression), visit)
        for index, (_call, first_lane, participant_count) in enumerate(calls):
            if participant_count == 0:
                continue
            participants = tuple(range(first_lane, first_lane + participant_count))
            if (
                first_lane < 0
                or participant_count <= 0
                or participants[-1] >= len(self.capture.actor_ids)
                or self.lane not in participants
            ):
                raise CaptureError(
                    "TILELANG_PARTICIPANTS_UNRESOLVED",
                    "TileLang all-reduce named-barrier participants are invalid",
                    {
                        "first_lane": first_lane,
                        "participants": participant_count,
                        "cta_threads": len(self.capture.actor_ids),
                        "executing_lane": self.lane,
                    },
                )
            resource = self.capture.named_barrier(
                (*path, "allreduce", index), participants
            )
            self.emit(
                "tilelang.barrier.arrive",
                (*path, "allreduce", index, "arrive"),
                _ir_operation=getattr(_call.op, "name", ""),
                resource=resource,
                invocation_id=0,
                count=1,
            )
            self.emit(
                "tilelang.barrier.observe",
                (*path, "allreduce", index, "observe"),
                _ir_operation=getattr(_call.op, "name", ""),
                resource=resource,
                invocation_id=0,
            )

    def _is_warp_shuffle(self, call: Any) -> bool:
        # A warp shuffle exchanges registers only and orders no memory, so its
        # mask and width never matter here; the shuffled value is the issuing
        # lane's own operand, whose loads are still collected.
        name = getattr(call.op, "name", "")
        if name in {"tl.shfl_xor_sync", "tl.shfl_down_sync", "tl.shfl_up_sync", "tl.shfl_sync"}:
            return len(call.args) == 4
        # Warp votes and matches exchange register values the same way.
        if name in {
            "tl.ballot_sync", "tl.any_sync", "tl.all_sync", "tl.match_any_sync", "tl.match_all_sync",
            "tl.ballot", "tl.activemask",
        }:
            return all(str(getattr(arg, "dtype", "")) != "handle" for arg in call.args)
        if name in {
            "tir.tvm_warp_shuffle", "tir.tvm_warp_shuffle_down", "tir.tvm_warp_shuffle_up",
            "tirx.tvm_warp_shuffle", "tirx.tvm_warp_shuffle_down", "tirx.tvm_warp_shuffle_up",
        }:
            return len(call.args) == 5
        return name in {"tir.tvm_warp_activemask", "tirx.tvm_warp_activemask"} and not call.args

    def _prune_static_selects(self, expression: Any) -> Any:
        # if_then_else evaluates only its taken branch: a guarded load such as
        # T.if_then_else(in_bound, x[i - pad], 0) never reads x when the
        # lane's guard is false. Guards that read memory are kept whole.
        def condition_reads(condition: Any) -> bool:
            found: list[Any] = []

            def visit(node: Any) -> None:
                if isinstance(node, self.tirx.BufferLoad):
                    found.append(node)

            self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(condition), visit)
            return bool(found)

        def preorder(node: Any) -> Any:
            if (
                not isinstance(node, self.tirx.Call)
                or getattr(node.op, "name", "") not in {"tir.if_then_else", "tirx.if_then_else"}
                or len(node.args) != 3
                or condition_reads(node.args[0])
            ):
                return None
            condition = self.eval_int(node.args[0])
            if condition is None:
                return None
            return self._prune_static_selects(node.args[1] if condition else node.args[2])

        return self.tirx.stmt_functor.ir_transform(
            self.tirx.Evaluate(expression), preorder, None
        ).value

    def _buffer_loads(
        self,
        expression: Any,
        atomics: list[Any] | None = None,
    ) -> list[Any]:
        expression = self._prune_static_selects(expression)
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
            if atomics is not None and (call_name in _ATOMICS or re.fullmatch(r"tl\.ld[gs](32|64|128)", call_name)):
                atomics.append(node)
                return
            if (
                is_pure_payload_call(call_name, len(node.args))
                or self._is_warp_shuffle(node)
                or call_name in {"tir.tvm_access_ptr", "tirx.tvm_access_ptr"}
                or self._allreduce_participants(node) is not None
                or self._debug_extern(node)
                or self._register_rng(node)
                or self._cta_vote(node)
            ):
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

    def _lane_local_expression(self, expression: Any) -> bool:
        # No cross-lane collective or atomic: evaluating it affects only the
        # lane's own registers and reads.
        found: list[Any] = []

        def visit(node: Any) -> None:
            if isinstance(node, self.tirx.Call) and (
                getattr(node.op, "name", "") in _ATOMICS
                or self._is_warp_shuffle(node)
                or self._allreduce_participants(node) is not None
                or self._cta_vote(node)
            ):
                found.append(node)

        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression), visit)
        return not found

    def _register_only_branch(self, statement: Any) -> bool:
        if not self._lane_local_expression(statement.condition):
            return False

        def register_only(node: Any) -> bool:
            if node is None:
                return True
            kind = type(node).__name__
            if kind == "SeqStmt":
                return all(register_only(child) for child in node.seq)
            if kind == "IfThenElse":
                return (
                    self._lane_local_expression(node.condition)
                    and register_only(node.then_case)
                    and register_only(node.else_case)
                )
            return (
                kind == "BufferStore"
                and getattr(node, "predicate", None) is None
                and self.buffer_scope(node.buffer).startswith("local")
                and self._lane_local_expression(node.value)
                and all(self._lane_local_expression(index) for index in node.indices)
            )

        return register_only(statement.then_case) and register_only(statement.else_case)

    def _walk_register_branch(self, statement: Any, path: tuple[Any, ...]) -> None:
        # A data-dependent branch whose arms only update the lane's own
        # registers: both arms' reads may happen, their register writes are
        # may-writes, and the written registers' values become opaque.
        self._emit_expression_loads(statement.condition, (*path, "if", "condition"))
        saved = dict(self.scalar_buffer_values)
        written: set[str] = set()

        def stored(node: Any) -> None:
            if isinstance(node, self.tirx.BufferStore):
                written.add(str(node.buffer.data))

        for arm, branch in ((1, statement.then_case), (0, statement.else_case)):
            if branch is None:
                continue
            self.scalar_buffer_values = dict(saved)
            self.tirx.stmt_functor.post_order_visit(branch, stored)
            self.walk(branch, (*path, "if", "may", arm))
        self.scalar_buffer_values = {
            key: value for key, value in saved.items() if key[0] not in written
        }

    def _emit_atomic(self, call: Any, path: tuple[Any, ...]) -> None:
        # Device-scope read-modify-write of one destination element (or of
        # 2/4 for vector adds); loads in a value expression are emitted with
        # the enclosing expression, a value pointer is read here.
        name = getattr(call.op, "name", "")
        if self._explicit_memory(call, path):
            return
        width = _VECTOR_ATOMICS.get(name)
        if width is not None:
            if len(call.args) not in {2, 3} or (
                len(call.args) == 3 and self.eval_int(call.args[2]) is None
            ):
                raise CaptureError(
                    "TILELANG_ATOMIC_UNRESOLVED",
                    "vector atomic requires a destination, a value, and a static memory order",
                    {"operation": name, "args": [str(arg) for arg in call.args]},
                )
            value_name = getattr(getattr(call.args[1], "op", None), "name", "")
            if value_name in {"tir.tvm_access_ptr", "tirx.tvm_access_ptr"}:
                value_region, _byte_count = self.access_ptr_region(call.args[1], 1, width)
                self.emit("tilelang.mem.read", (*path, "value"), _ir_operation=name, region=value_region)
            region, _byte_count = self.access_ptr_region(call.args[0], 2, width)
            self.emit("tilelang.mem.atomic", path, _ir_operation=name, region=region)
            return
        if len(call.args) not in {2, 3} or (
            len(call.args) == 3 and self.eval_int(call.args[2]) is None
        ):
            raise CaptureError(
                "TILELANG_ATOMIC_UNRESOLVED",
                "atomic requires a destination, a value, and a static memory order",
                {"operation": getattr(call.op, "name", ""), "args": [str(arg) for arg in call.args]},
            )
        region, _byte_count = self.access_ptr_region(call.args[0], 2, 1)
        self.emit("tilelang.mem.atomic", path, _ir_operation=getattr(call.op, "name", ""), region=region)

    def _emit_expression_loads(self, expression: Any, path: tuple[Any, ...]) -> None:
        atomics: list[Any] = []
        for index, load in enumerate(self._buffer_loads(expression, atomics)):
            for region_index, region in enumerate(
                self.capture.memory_regions(load.buffer, load.indices, self)
            ):
                self.emit(
                    "tilelang.mem.read",
                    (*path, "read", index, region_index),
                    _ir_operation="BufferLoad",
                    region=region,
                )
        for index, atomic in enumerate(atomics):
            self._emit_atomic(atomic, (*path, "atomic", index))
        self._emit_expression_collectives(expression, (*path, "vote"), votes_only=True)

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
                block_dimensions = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
                axis = block_dimensions.get(tag)
                if extent is None and axis is not None and self.capture.launch_grid_explicit:
                    extent = self.capture.launch_grid[axis]
                if extent is None or variable is None:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "thread binding is not statically resolvable",
                        {"tag": tag, "path": list(path)},
                    )
                had_old = variable in self.bindings
                old = self.bindings.get(variable)
                if tag in {"threadIdx.x", "threadIdx.y", "threadIdx.z"}:
                    self.capture.thread_variables[variable] = tag
                    self.bindings[variable] = self.thread_coordinate(tag)
                elif axis is not None:
                    if extent != self.capture.launch_grid[axis]:
                        raise CaptureError(
                            "TILELANG_GRID_EVIDENCE_INVALID",
                            "block binding extent disagrees with launch-grid evidence",
                            {"tag": tag, "extent": extent, "launch_grid": list(self.capture.launch_grid)},
                        )
                    self.bindings[variable] = self.capture.selected_cta[axis]
                    self.capture.note_block_variable(axis, variable)
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
            if statement.attr_key == "threadblock_swizzle_pattern":
                remapped = self.capture.swizzled_block_bindings(statement.value)
                saved = {variable: (variable in self.bindings, self.bindings.get(variable)) for variable in remapped}
                self.bindings.update(remapped)
                self.walk(statement.body, (*path, "attr", statement.attr_key))
                for variable, (had_old, old) in saved.items():
                    if had_old:
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
                "tl.min_blocks_per_sm",
                "pipeline_exec_scope",
                "pipeline_scope",
                "pragma_auto_unroll_max_step",
                "pragma_unroll_explicit",
                "pragma_unroll_factor",
                "realize_scope",
                "reduce_scope",
                "software_pipeline_order",
                "software_pipeline_stage",
                "storage_scope",
                "tl.assume",
                "kWarpSpecializationScope",
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
            elif str(statement.var.dtype) != "handle":
                # A data value (a loaded or computed scalar) stays opaque: its
                # loads are recorded, and any later use of it as an address,
                # extent, or branch condition fails resolution there.
                self._emit_expression_loads(statement.value, (*path, "value"))
                self.bindings.pop(statement.var, None)
            else:
                pointer = self.eval_pointer(statement.value)
                if pointer is None:
                    raise CaptureError(
                        "TILELANG_STATIC_VALUE_UNRESOLVED",
                        "TIRX Bind value is neither a static integer nor a static pointer alias",
                        {"variable": str(statement.var), "value": str(statement.value)},
                    )
                self._emit_expression_loads(statement.value, (*path, "value"))
                self.pointer_aliases[statement.var] = pointer
            self.capture.classify_leaf(before)
            return
        if name == "LetStmt":
            value = self.eval_int(statement.value)
            if value is None:
                if str(statement.var.dtype) == "handle":
                    raise CaptureError(
                        "TILELANG_STATIC_VALUE_UNRESOLVED",
                        "TIRX LetStmt value is not statically resolvable",
                        {"variable": str(statement.var), "value": str(statement.value)},
                    )
                self._emit_expression_loads(statement.value, (*path, "let-value"))
            had_old = statement.var in self.bindings
            old = self.bindings.get(statement.var)
            if value is None:
                self.bindings.pop(statement.var, None)
            else:
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
            self.walk(getattr(statement, "body", None), (*path, "assert"))
            return
        if name == "IfThenElse":
            condition = self.eval_int(statement.condition)
            if condition is None and self._register_only_branch(statement):
                self._walk_register_branch(statement, path)
                return
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
            seen_states: set[StaticLoopState] = set()
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
                state = self.static_loop_state()
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
            if minimum is None or extent is None or (extent < 0 and not is_empty_signed_cuda_loop(statement, minimum, extent)):
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
            self._emit_expression_collectives(statement.value, (*path, "rhs"))
            atomics: list[Any] = []
            for index, load in enumerate(self._buffer_loads(statement.value, atomics)):
                for region_index, region in enumerate(
                    self.capture.memory_regions(load.buffer, load.indices, self)
                ):
                    self.emit(
                        "tilelang.mem.read",
                        (*path, "rhs", index, region_index),
                        _ir_operation="BufferLoad",
                        region=region,
                    )
            for index, atomic in enumerate(atomics):
                self._emit_atomic(atomic, (*path, "rhs", "atomic", index))
            if self._access_enabled(statement):
                for region_index, region in enumerate(
                    self.capture.memory_regions(statement.buffer, statement.indices, self)
                ):
                    self.emit("tilelang.mem.write", (*path, region_index), region=region)
                if len(statement.indices) == 1 and self.buffer_scope(statement.buffer).startswith("local"):
                    scalar_index = self.eval_int(statement.indices[0])
                    scalar_value = self.eval_int(statement.value)
                    if scalar_index is not None:
                        key = (str(statement.buffer.data), scalar_index)
                        if scalar_value is None:
                            self.scalar_buffer_values.pop(key, None)
                        else:
                            self.scalar_buffer_values[key] = scalar_value
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

    def _explicit_memory(self, call: Any, path: tuple[Any, ...]) -> bool:
        name = getattr(call.op, "name", "")
        match = re.fullmatch(r"tl\.(ld|st)(g|s)(32|64|128)", name)
        if not match:
            return False
        direction, space, bits = match.groups()
        store = direction == "st"
        arity = 2 if store else 1
        if len(call.args) not in ({arity, arity + 1} if space == "g" else {arity}):
            raise CaptureError("TILELANG_MEMORY_ACCESS_UNRESOLVED", "explicit memory intrinsic has invalid arity", {"operation": name})
        if store:
            # C++ evaluates the value argument even for a predicated-off store.
            self._emit_expression_loads(call.args[1], (*path, "value"))
        if len(call.args) > arity:
            predicate = self.eval_int(call.args[-1])
            if predicate is None:
                raise CaptureError("TILELANG_PREDICATE_UNRESOLVED", "explicit memory predicate is data dependent", {"operation": name})
            if not predicate:
                return True
        width = dtype_bit_width(self.access_ptr_dtype(call.args[0]))
        if width is None or int(bits) % width:
            raise CaptureError("TILELANG_MEMORY_ACCESS_UNRESOLVED", "explicit memory width is incompatible with pointer type", {"operation": name})
        region, _ = self.access_ptr_region(call.args[0], 2 if store else 1, int(bits) // width)
        expected_space = "GLOBAL" if space == "g" else "SHARED"
        if self.capture.regions[region]["address_space"] != expected_space:
            raise CaptureError("TILELANG_MEMORY_ACCESS_UNRESOLVED", "explicit memory intrinsic address space mismatch", {"operation": name})
        self.emit("tilelang.mem.write" if store else "tilelang.mem.read", path, _ir_operation=name, region=region)
        return True

    def _extern_region(self, expression: Any, index: int, count: int, mode: int) -> str:
        """Exact element slice of a typed pointer used by a known C++ helper."""
        if isinstance(expression, self.tirx.Var):
            annotation = expression.type_annotation
            dtype = getattr(getattr(annotation, "element_type", None), "dtype", None)
            scope = str(getattr(annotation, "storage_scope", ""))
            resolved = self.eval_pointer(expression)
            width = dtype_byte_width(dtype) if dtype is not None else None
            if resolved is None or width is None or not scope:
                raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "external pointer needs an exact typed allocation", {"pointer": str(expression)})
            data, offset = resolved
            allocation = self.capture.allocation_id(data, scope, self.actor, getattr(data, "name", str(data)))
        elif (isinstance(expression, self.tirx.Call)
              and getattr(expression.op, "name", "") in {"tir.address_of", "tirx.address_of"}
              and len(expression.args) == 1 and isinstance(expression.args[0], self.tirx.BufferLoad)):
            load = expression.args[0]
            allocation, base_offset = self.buffer_allocation(load.buffer)
            element_offset, _ = self.byte_interval(load.buffer, load.indices)
            scope, offset = self.buffer_scope(load.buffer), base_offset + element_offset
            width = self._dtype_bytes(load.buffer)
        else:
            region, _ = self.access_ptr_region(expression, mode, 1)
            info = self.capture.regions[region]
            allocation, scope, offset = info["allocation_id"], {"SHARED": "shared", "GLOBAL": "global", "REGISTER": "local"}[info["address_space"]], info["offset"]
            width = dtype_byte_width(self.access_ptr_dtype(expression))
        if width is None or index < 0 or count <= 0:
            raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "external pointer slice must have static positive byte width")
        return self.capture.region(allocation, scope, offset + index * width, count * width)

    def _extern_barrier(self, path: tuple[Any, ...], first: int, count: int) -> None:
        lanes = tuple(range(first, first + count))
        if not lanes or first < 0 or lanes[-1] >= self.capture.cta_threads or self.lane not in lanes:
            raise CaptureError("TILELANG_PARTICIPANTS_UNRESOLVED", "external collective has invalid participants", {"first": first, "count": count, "lane": self.lane})
        resource = self.capture.named_barrier(path, lanes)
        self.emit("tilelang.barrier.arrive", (*path, "arrive"), resource=resource, invocation_id=0, count=1)
        self.emit("tilelang.barrier.observe", (*path, "observe"), resource=resource, invocation_id=0)

    def _evaluate_extern(self, call: Any, path: tuple[Any, ...]) -> bool:
        if getattr(call.op, "name", "") not in {"tir.call_extern", "tirx.call_extern"} or not call.args:
            return False
        if not isinstance(call.args[0], self.tirx.StringImm):
            return False
        callee = call.args[0].value
        args = list(call.args[1:])
        if callee == "__nanosleep" and len(args) == 1 and str(args[0].dtype) != "handle":
            self._emit_expression_loads(args[0], (*path, "delay"))
            return True
        match = re.fullmatch(r"tl::AllReduce<(tl::(?:Sum|Max|Min)Op(?:_bf16x2|_fp16x2)?),\s*(\d+),\s*(\d+),\s*(\d+),\s*tl::NamedBarrier<(\d+)>,\s*(\d+),\s*(\d+)>::run_batch", callee)
        if match:
            _reducer, threads, scale, first, barrier_threads, batch, stride = match.groups()
            threads, scale, first, barrier_threads, batch, stride = map(int, (threads, scale, first, barrier_threads, batch, stride))
            if (scale <= 0 or threads < scale or threads % scale or (threads // scale) & (threads // scale - 1)
                    or batch <= 0 or len(args) not in {1, 2}):
                raise CaptureError("TILELANG_EXTERN_SIGNATURE_INVALID", "invalid batched AllReduce specialization", {"callee": callee})
            if threads == scale:
                return True
            # reduce.h uses threadIdx.x, not linear thread ID, for workspace.
            lane = self.thread_coordinate("threadIdx.x") - first
            local = self._extern_region(args[0], 0, batch, 3)
            self.emit("tilelang.mem.read", (*path, "values", "read"), region=local)
            level = threads
            while level > scale:
                distance = level // 2
                if distance >= 32:
                    if len(args) != 2 or lane < 0 or (batch > 1 and stride <= 0):
                        raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "cross-warp AllReduce requires its shared workspace")
                    self._extern_barrier((*path, "reduce", level, "before"), first, barrier_threads)
                    for channel in range(batch):
                        region = self._extern_region(args[1], lane + channel * stride, 1, 2)
                        self.emit("tilelang.mem.write", (*path, "reduce", level, channel, "write"), region=region)
                    self._extern_barrier((*path, "reduce", level, "after"), first, barrier_threads)
                    for channel in range(batch):
                        region = self._extern_region(args[1], (lane ^ distance) + channel * stride, 1, 1)
                        self.emit("tilelang.mem.read", (*path, "reduce", level, channel, "read"), region=region)
                level //= 2
            self.emit("tilelang.mem.write", (*path, "values", "write"), region=local)
            self.scalar_buffer_values.clear()
            return True
        match = re.fullmatch(r"tl::Cum(Sum|Max)1D<(32|64|128|256|512|1024),\s*(true|false)>::run", callee)
        if match:
            if len(args) != 3 or self.eval_int(args[2]) is None:
                raise CaptureError("TILELANG_EXTERN_SIGNATURE_INVALID", "one-dimensional scan requires two pointers and a static extent")
            extent = self.eval_int(args[2])
            tx = self.thread_coordinate("threadIdx.x")
            if tx >= 32 or extent <= 0:
                return True
            segments = range((extent + 31) // 32)
            if match.group(3) == "true":
                segments = reversed(segments)
            for segment in segments:
                index = segment * 32 + tx
                if index < extent:
                    self.emit("tilelang.mem.read", (*path, segment, "read"), region=self._extern_region(args[0], index, 1, 1))
                    self.emit("tilelang.mem.write", (*path, segment, "write"), region=self._extern_region(args[1], index, 1, 2))
            self.scalar_buffer_values.clear()
            return True
        match = re.fullmatch(r"tl::Cum(Sum|Max)2D<(32|64|128|256|512|1024),\s*([01]),\s*(true|false)>::run", callee)
        if match:
            _op, threads, axis, reverse = match.groups()
            threads, axis = int(threads), int(axis)
            if len(args) != 6:
                raise CaptureError("TILELANG_EXTERN_SIGNATURE_INVALID", "scan requires two pointers and four dimensions")
            dims = [self.eval_int(arg) for arg in args[2:]]
            if any(value is None for value in dims):
                raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "scan dimensions and strides must be static")
            height, width, src_stride, dst_stride = dims
            if height <= 0 or width <= 0:
                return True
            if min(src_stride, dst_stride) <= 0:
                raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "scan strides must be positive")
            tx = self.thread_coordinate("threadIdx.x")
            item, lane, tile = tx // 32, tx % 32, threads // 32
            outer, extent = (height, width) if axis else (width, height)
            for block in range((outer + tile - 1) // tile):
                line = block * tile + item
                if line >= outer:
                    break
                segments = range((extent + 31) // 32)
                if reverse == "true":
                    segments = reversed(segments)
                for segment in segments:
                    index = segment * 32 + lane
                    if index >= extent:
                        continue
                    src_index = line * src_stride + index if axis else line + index * src_stride
                    dst_index = line * dst_stride + index if axis else line + index * dst_stride
                    self.emit("tilelang.mem.read", (*path, block, segment, "read"), region=self._extern_region(args[0], src_index, 1, 1))
                    self.emit("tilelang.mem.write", (*path, block, segment, "write"), region=self._extern_region(args[1], dst_index, 1, 2))
            self.scalar_buffer_values.clear()
            return True
        if callee == "DP4A" and len(args) == 3:
            for index, arg in enumerate(args):
                width = dtype_byte_width(self.access_ptr_dtype(arg))
                if width is None or 4 % width:
                    raise CaptureError("TILELANG_EXTERN_POINTER_UNRESOLVED", "DP4A requires three 32-bit footprints")
                region = self._extern_region(arg, 0, 4 // width, 3 if index == 2 else 1)
                self.emit("tilelang.mem.read", (*path, index, "read"), region=region)
                if index == 2:
                    self.emit("tilelang.mem.write", (*path, index, "write"), region=region)
            self.scalar_buffer_values.clear()
            return True
        return False

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
        if self._evaluate_extern(expression, path) or self._explicit_memory(expression, path):
            return
        if name in _ATOMICS:
            self._emit_expression_loads(expression, path)
            return
        if name in {"tl.device_assert", "tl.device_assert_with_msg"}:
            expected_args = 1 if name == "tl.device_assert" else 2
            if (len(expression.args) != expected_args
                    or (expected_args == 2 and not isinstance(expression.args[1], self.tirx.StringImm))
                    or self.eval_int(expression.args[0]) != 1):
                raise CaptureError("TILELANG_ASSERTION_UNRESOLVED", "device assertion must be statically true; thread termination is not modeled", {"operation": name, "args": [str(arg) for arg in expression.args]})
            self._emit_expression_loads(expression.args[0], path)
            return
        if name == "tl.set_max_nreg":
            if not valid_set_max_nreg(expression.args, self.eval_int):
                raise CaptureError(
                    "TILELANG_SET_MAX_NREG_UNRESOLVED",
                    "set_max_nreg requires a static PTX register count and action",
                    {"args": [str(argument) for argument in expression.args]},
                )
            return
        if name == "tl.no_set_max_nreg" and not expression.args:
            # A compiler hint; pinned CUDA codegen emits no instruction.
            return
        if name in {"tl.pdl_sync", "tl.pdl_trigger"} and not expression.args:
            # Grid dependency control orders whole grids against each other;
            # it adds no ordering between threads of the selected CTA.
            return
        if name in {"tirx.ptx_mma", "tir.ptx_mma"}:
            # mma.sync reads and writes only the issuing warp's registers.
            operands = expression.args[6:11:2] if len(expression.args) >= 13 else []
            scopes = [
                re.findall(r'"([^"]+)"', str(getattr(operand, "type_annotation", "")))
                for operand in operands
            ]
            if len(operands) != 3 or not all(
                isinstance(operand, self.tirx.Var) and scope and scope[-1].startswith("local")
                for operand, scope in zip(operands, scopes)
            ):
                raise CaptureError(
                    "TILELANG_MEMORY_ACCESS_UNRESOLVED",
                    "ptx_mma operands must be thread-local register fragments",
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
            if expression.args:
                raise CaptureError(
                    "TILELANG_PROXY_TARGET_UNRESOLVED",
                    "fence_proxy_async must be operand-free",
                )
            # Reserve the fence's program position now. fence.proxy.async is
            # not specific to one async-proxy unit, so one placeholder per
            # domain is bound (regions included) by that domain's next async
            # consumer; placeholders no consumer binds are discarded.
            last_lane_event_id = self.last_lane_event_id
            for domain in ("ASYNC_COPY", "TMA", "TENSOR_CORE"):
                self.emit(
                    "tilelang.proxy_fence",
                    (*path, domain.lower()),
                    _ir_operation="tl.fence_proxy_async",
                )
                self.capture.pending_proxies.append(
                    (domain, self.capture.operations[-1])
                )
            self.last_lane_event_id = last_lane_event_id
            return
        if name in {"tirx.tvm_storage_sync", "tir.tvm_storage_sync"}:
            scope = str(expression.args[0]).strip('"') if expression.args else None
            if scope not in {"shared", "shared.dyn"} or len(expression.args) not in {1, 3}:
                raise CaptureError(
                    "TILELANG_BARRIER_SCOPE_UNSUPPORTED",
                    "shared synchronization requires a full-CTA or static named-barrier form",
                    {"operation": name, "args": [str(arg) for arg in expression.args]},
                )
            if len(expression.args) == 1:
                resource = self.capture.cta_barrier(path)
            else:
                barrier_id = self.eval_int(expression.args[1])
                participant_count = self.eval_int(expression.args[2])
                if (
                    barrier_id is None
                    or barrier_id < 0
                    or participant_count is None
                    or participant_count <= 0
                ):
                    raise CaptureError(
                        "TILELANG_BARRIER_SCOPE_UNSUPPORTED",
                        "named barrier id and participant count must be static",
                        {"operation": name, "args": [str(arg) for arg in expression.args]},
                    )
                if self.leaf_lanes is not None and len(self.leaf_lanes) == participant_count:
                    # Exactly the lanes reaching this barrier arrive on it.
                    participants = self.leaf_lanes
                else:
                    first_lane = self.lane // participant_count * participant_count
                    participants = tuple(range(first_lane, first_lane + participant_count))
                first_lane = participants[0]
                if participants[-1] >= len(self.capture.actor_ids):
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "named barrier participants exceed the selected CTA",
                        {"first_lane": first_lane, "count": participant_count},
                    )
                resource = self.capture.named_barrier(
                    (*path, "named", barrier_id, first_lane), participants
                )
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
        if name == "tl.sync_warp":
            # Statically proven masks order exactly their participating lanes.
            mask = self.eval_int(expression.args[0]) if expression.args else 0xFFFFFFFF
            warp_first = self.lane // 32 * 32
            participants = tuple(lane for lane in range(warp_first, min(warp_first + 32, len(self.capture.actor_ids)))
                                 if mask is not None and mask & (1 << (lane % 32)))
            if (
                len(expression.args) > 1
                or mask is None
                or self.lane not in participants
                or (
                    self.leaf_lanes is not None
                    and tuple(lane for lane in self.leaf_lanes if lane // 32 * 32 == warp_first)
                    != participants
                )
            ):
                raise CaptureError(
                    "TILELANG_BARRIER_SCOPE_UNSUPPORTED",
                    "sync_warp requires a static mask reached by exactly its participating warp lanes",
                    {"operation": name, "args": [str(arg) for arg in expression.args]},
                )
            resource = self.capture.named_barrier((*path, "warp", warp_first, mask), participants)
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
        noinc = name == "tl.ptx_cp_async_barrier_noinc"
        if noinc or name in {"tirx.ptx_arrive_barrier", "tir.ptx_arrive_barrier"}:
            if len(expression.args) != 1:
                raise CaptureError(
                    "TILELANG_MBARRIER_ARRIVE_UNRESOLVED",
                    "mbarrier arrival requires exactly one barrier operand",
                    {"operation": name, "operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            kinds = self.arrival_kinds_since_wait.setdefault(resource, set())
            kinds.add("cp_async_noinc" if noinc else "plain")
            if len(kinds) > 1:
                raise CaptureError(
                    "TILELANG_MBARRIER_PHASE_UNRESOLVED",
                    "mixed deferred and plain arrivals may contribute twice to one phase; per-lane arrival rank is insufficient",
                    {"resource": resource, "lane": self.lane, "arrival_kinds": sorted(kinds)},
                )
            if noinc:
                generation = self.arrival_rank.get(resource, 0)
                self.emit("tilelang.cp_async.mbarrier_arrive", path,
                          resource=resource, async_resource=self.capture.async_group(self.actor),
                          group_owner=self.actor, lifecycle_epoch=0,
                          generation=generation, count=1, predicate=True)
                self.arrival_rank[resource] = generation + 1
                return
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
            if parity is None:
                raise CaptureError(
                    "TILELANG_PARITY_UNRESOLVED",
                    "mbarrier wait parity is not statically resolved",
                    {"resource": resource, "rank": generation},
                )
            if self.capture.uses_warpgroup and not self.is_warpgroup_leader():
                self.observe_warpgroup_dependency(path)
                return
            self.wait_rank[resource] = generation + 1
            self.arrival_kinds_since_wait.pop(resource, None)
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
        if name in {"tirx.ptx_arrive_barrier_expect_tx", "tir.ptx_arrive_barrier_expect_tx"}:
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_TMA_EXPECT_UNRESOLVED",
                    "ptx_arrive_barrier_expect_tx requires a barrier and byte count",
                )
            resource = self._barrier_operand(expression, 0)
            byte_count = self.eval_int(expression.args[1])
            if byte_count is None or byte_count <= 0:
                raise CaptureError(
                    "TILELANG_TMA_EXPECT_UNRESOLVED",
                    "mbarrier expected transaction bytes must be static and positive",
                )
            generation = self.arrival_rank.get(resource, 0)
            if resource in self.pending_arrive_expect:
                raise CaptureError(
                    "TILELANG_TMA_EXPECT_UNRESOLVED",
                    "combined arrive-expect has no intervening TMA issue",
                    {"resource": resource, "generation": generation},
                )
            self.pending_arrive_expect[resource] = byte_count
            self.emit(
                "tilelang.mbarrier.arrive_expect_tx",
                path,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                count=1,
                bytes=byte_count,
                predicate=True,
            )
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
                    "tma_load requires a source, destination, barrier, and static transfer description",
                )
            if len(expression.args) == 5 and isinstance(expression.args[2], self.tirx.BufferLoad):
                destination, destination_bytes = self.access_ptr_region(expression.args[0], 2)
                source, source_bytes = self.access_ptr_region(expression.args[1], 1)
                resource = self._barrier_operand(expression, 2)
                byte_count = self.eval_int(expression.args[3])
                mode = self.eval_int(expression.args[4])
                if (
                    byte_count is None
                    or byte_count <= 0
                    or byte_count > source_bytes
                    or byte_count > destination_bytes
                    or mode is None
                ):
                    raise CaptureError(
                        "TILELANG_TMA_LOAD_UNRESOLVED",
                        "direct TMA load byte count, regions, or mode is unresolved",
                        {
                            "bytes": byte_count,
                            "source_extent": source_bytes,
                            "destination_extent": destination_bytes,
                        },
                    )
                segments = [(source, destination, byte_count)]
            elif isinstance(expression.args[1], self.tirx.BufferLoad):
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
                segments = self.capture.tma_segments(
                    descriptor,
                    [int(value) for value in coordinates],
                    destination,
                )
            else:
                raise CaptureError(
                    "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                    "TMA load barrier operand is not a static BufferLoad",
                    {"operation": name},
                )
            self.bind_pending_proxy(
                "TMA",
                [destination for _source, destination, _bytes in segments],
            )
            generation = self.arrival_rank.get(resource, 0)
            self.capture.record_tma_destination_group(segments)
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
            if resource in self.pending_arrive_expect:
                remaining = self.pending_arrive_expect[resource] - sum(n for _s, _d, n in segments)
                if remaining < 0:
                    raise CaptureError("TILELANG_TMA_EXPECT_UNRESOLVED",
                                       "TMA issues exceed the pending arrive-expect byte budget",
                                       {"resource": resource, "remaining_bytes": remaining})
                # One arrive-expect can cover several TMA instructions (e.g.
                # four B panels). All belong to the same barrier generation.
                if remaining == 0:
                    del self.pending_arrive_expect[resource]
                    self.arrival_rank[resource] = generation + 1
                else:
                    self.pending_arrive_expect[resource] = remaining
            return
        if name == "tl.tma_store" and isinstance(expression.args[0], self.tirx.Var):
            # Descriptor form: desc, shared source, coordinates..., need_reduce,
            # eviction policy. Preserve the atomic destination reduction.
            if len(expression.args) < 5:
                raise CaptureError(
                    "TILELANG_TMA_STORE_UNRESOLVED",
                    "descriptor tma_store requires a source, coordinates, reduction, and eviction operands",
                )
            source, _source_bytes = self.access_ptr_region(expression.args[1], 1)
            operands = [self.eval_int(argument) for argument in expression.args[2:]]
            if any(value is None for value in operands) or operands[-2] not in {0, 1}:
                raise CaptureError(
                    "TILELANG_TMA_STORE_UNRESOLVED",
                    "TMA store coordinates and mode operands must be statically resolved",
                    {"operands": [str(argument) for argument in expression.args[2:]]},
                )
            segments = self.capture.tma_segments(
                expression.args[0], [int(value) for value in operands[:-2]], source, store=True
            )
            # A swizzled tile permutes elements within its shared rows, so
            # once any element is dropped the per-run shared bytes are only
            # known up to the whole source tile.
            partial = sum(count for _destination, _source, count in segments) != (
                self.capture.regions[source]["extent"]
            )
            self.bind_pending_proxy("TMA", [source])
            owner = self.actor
            resource = self.capture.tma_store_group(owner)
            for index, (destination, source_segment, byte_count) in enumerate(segments):
                self.emit(
                    "tilelang.tma_store.issue",
                    path if len(segments) == 1 else (*path, f"segment{index}"),
                    resource=resource,
                    group_owner=owner,
                    source_region=source if partial else source_segment,
                    destination_region=destination,
                    reduction=bool(operands[-2]),
                    bytes=byte_count,
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
                or self.eval_int(expression.args[3]) not in {0, 1}
                or self.eval_int(expression.args[4]) is None
            ):
                raise CaptureError(
                    "TILELANG_TMA_STORE_UNRESOLVED",
                    "TMA store byte count or mode operands are unresolved",
                )
            self.bind_pending_proxy("TMA", [source])
            owner = self.actor
            resource = self.capture.tma_store_group(owner)
            self.emit(
                "tilelang.tma_store.issue",
                path,
                resource=resource,
                group_owner=owner,
                source_region=source,
                destination_region=destination,
                reduction=bool(self.eval_int(expression.args[3])),
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
            if not self.is_warpgroup_leader():
                return
            region, _extent = self.access_ptr_region(expression.args[1], 1)
            for argument in expression.args[2:]:
                if self.eval_int(argument) is None:
                    raise CaptureError(
                        "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                        "WGMMA descriptor layout operands must be static",
                    )
            descriptor = expression.args[0]
            owner = self.warpgroup_owner()
            key = (owner, descriptor.buffer.data)
            layout_type, leading_byte_offset, stride_byte_offset = (
                self.eval_int(argument) for argument in expression.args[2:]
            )
            stage_region = self.capture.select_wgmma_descriptor_stage(key, region, 0)
            descriptor_name = getattr(
                descriptor.buffer, "name", getattr(descriptor.buffer.data, "name", "descriptor")
            )
            self.capture.wgmma_descriptor_bases[key] = region
            self.capture.wgmma_descriptors[key] = stage_region
            self.capture.wgmma_descriptor_metadata[key] = {
                "descriptor_id": f"{owner}:{descriptor_name}",
                "base_region": region,
                "selected_stage": None if stage_region is None else {
                    "region_id": stage_region,
                    **self.capture.regions[stage_region],
                },
                "stage_byte_offset": 0,
                "layout_type": layout_type,
                "leading_byte_offset": leading_byte_offset,
                "stride_byte_offset": stride_byte_offset,
            }
            return
        if name == "tl.increase_descriptor_offset":
            if (
                len(expression.args) != 2
                or not isinstance(expression.args[0], self.tirx.BufferLoad)
            ):
                raise CaptureError(
                    "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                    "increase_descriptor_offset requires a descriptor and static byte offset",
                )
            if not self.is_warpgroup_leader():
                return
            owner = self.warpgroup_owner()
            key = (owner, expression.args[0].buffer.data)
            base_id = self.capture.wgmma_descriptor_bases.get(key)
            byte_offset = self.eval_int(expression.args[1])
            if base_id is None or byte_offset is None or byte_offset < 0:
                raise CaptureError(
                    "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                    "WGMMA descriptor base or byte offset is unresolved",
                )
            stage_region = self.capture.select_wgmma_descriptor_stage(
                key, base_id, byte_offset
            )
            self.capture.wgmma_descriptors[key] = stage_region
            metadata = self.capture.wgmma_descriptor_metadata.get(key)
            if metadata is None:
                raise CaptureError(
                    "TILELANG_WGMMA_DESCRIPTOR_UNRESOLVED",
                    "WGMMA descriptor metadata is missing for an offset update",
                )
            metadata["selected_stage"] = None if stage_region is None else {
                "region_id": stage_region,
                **self.capture.regions[stage_region],
            }
            metadata["stage_byte_offset"] = byte_offset
            return
        if name == "tl.warpgroup_fence_operand":
            if len(expression.args) != 4:
                raise CaptureError(
                    "TILELANG_WGMMA_FENCE_UNRESOLVED",
                    "warpgroup_fence_operand requires dtype, accumulator, offset, and extent",
                )
            if not self.is_warpgroup_leader():
                return
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
        if name in {"tl.ptx_wgmma_ss", "tl.ptx_wgmma_rs"}:
            expected_operands = 15 if name == "tl.ptx_wgmma_ss" else 14
            if len(expression.args) != expected_operands:
                raise CaptureError(
                    "TILELANG_WGMMA_UNRESOLVED",
                    f"{name} operand signature is unsupported",
                    {"operands": len(expression.args)},
                )
            if not self.is_warpgroup_leader():
                return
            owner = self.warpgroup_owner()
            descriptor_positions = (6, 8) if name == "tl.ptx_wgmma_ss" else (7,)
            offset_positions = (
                (7, 9, 11, 12, 13, 14)
                if name == "tl.ptx_wgmma_ss"
                else (6, 8, 10, 11, 12, 13)
            )
            descriptor_keys = [
                (owner, expression.args[position])
                for position in descriptor_positions
            ]
            if name == "tl.ptx_wgmma_ss":
                layout_specs = {
                    descriptor_keys[0]: ("A", 1, 3, 7),
                    descriptor_keys[1]: ("B", 2, 4, 9),
                }
            else:
                layout_specs = {descriptor_keys[0]: ("B", 1, 3, 8)}
            source_regions = []
            for key in descriptor_keys:
                if key in self.capture.wgmma_descriptor_windows:
                    operand, major_position, dtype_position, offset_position = layout_specs[key]
                    k_major = self.eval_int(expression.args[major_position])
                    descriptor_offset = self.eval_int(expression.args[offset_position])
                    if k_major not in {0, 1} or descriptor_offset is None:
                        raise CaptureError(
                            "TILELANG_WGMMA_OPERAND_REGION_UNRESOLVED",
                            "WGMMA layout footprint requires static major order and offset",
                            {"operand": operand},
                        )
                    source_regions.append(self.capture.wgmma_layout_region(
                        key,
                        shape=_wgmma_string(expression.args[0]),
                        dtype=_wgmma_string(expression.args[dtype_position]),
                        operand=operand,
                        k_major=bool(k_major),
                        descriptor_offset_units=descriptor_offset,
                    ))
                else:
                    source_regions.append(self.capture.wgmma_descriptors.get(key))
            source_regions = list(dict.fromkeys(source_regions))
            if (
                any(region is None for region in source_regions)
                or self.wgmma_accumulator_region is None
                or not self.wgmma_fence_seen
                or any(
                    self.eval_int(expression.args[position]) is None
                    for position in offset_positions
                )
            ):
                raise CaptureError(
                    "TILELANG_WGMMA_UNRESOLVED",
                    "WGMMA descriptors, accumulator, fence, or static offsets are unresolved",
                    {
                        "descriptors": [region is not None for region in source_regions],
                        "accumulator": self.wgmma_accumulator_region is not None,
                        "fence": self.wgmma_fence_seen,
                        "offsets": [str(expression.args[position]) for position in offset_positions],
                    },
                )
            resource = self.capture.wgmma_resource(owner)
            engine_seq = self.capture.wgmma_engine_seq.get(owner, 0)
            if name == "tl.ptx_wgmma_ss":
                proof_specs = (
                    ("A_SHARED", descriptor_keys[0], 7, 1, 3),
                    ("B_SHARED", descriptor_keys[1], 9, 2, 4),
                )
            else:
                proof_specs = (("B_SHARED", descriptor_keys[0], 8, 1, 3),)
            proof_operands: list[dict[str, Any]] = []
            try:
                for slot, key, offset_position, major_position, dtype_position in proof_specs:
                    metadata = self.capture.wgmma_descriptor_metadata.get(key)
                    descriptor_offset = self.eval_int(expression.args[offset_position])
                    k_major = self.eval_int(expression.args[major_position])
                    if metadata is not None and metadata["selected_stage"] is None:
                        raise CaptureError(
                            "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED",
                            "exact group-union proof requires a TMA-staged descriptor",
                            {"slot": slot, "engine_seq": engine_seq},
                        )
                    if metadata is None or descriptor_offset is None or k_major not in {0, 1}:
                        raise CaptureError(
                            "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED",
                            "WGMMA issue lacks exact descriptor state, offset, or major order",
                            {"slot": slot, "engine_seq": engine_seq},
                        )
                    proof_operands.append(_wgmma_issue_operand(
                        slot=slot,
                        shape=_wgmma_string(expression.args[0]),
                        dtype=_wgmma_string(expression.args[dtype_position]),
                        k_major=bool(k_major),
                        descriptor_offset_units=descriptor_offset,
                        descriptor=metadata,
                    ))
            except CaptureError as error:
                proof_operands = []
                self.capture.wgmma_proof_diagnostics.append(error.diagnostic())
            self.bind_pending_proxy("TENSOR_CORE", source_regions)
            if self.wgmma_arrived:
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
                engine_id=owner,
                engine_seq=engine_seq,
                regions=source_regions,
                destination_region=self.wgmma_accumulator_region,
            )
            if proof_operands:
                self.capture.record_wgmma_issue_proof(
                    owner,
                    self.capture.operations[-1]["id"],
                    engine_seq,
                    proof_operands,
                )
            self.capture.wgmma_engine_seq[owner] = engine_seq + 1
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
            self.capture.record_wgmma_commit_proof(
                owner,
                resource,
                group_seq,
                self.capture.operations[-1]["id"],
            )
            self.capture.wgmma_group_seq[owner] = group_seq + 1
            return
        # tl.wait_wgmma lowers to the same tl::wait_wgmma<n> as warpgroup_wait.
        if name in {"tl.warpgroup_wait", "tl.wait_wgmma"}:
            wait_count = self.eval_int(expression.args[0]) if len(expression.args) == 1 else None
            if wait_count is None or not 0 <= wait_count <= 7:
                raise CaptureError(
                    "TILELANG_WGMMA_WAIT_UNRESOLVED",
                    "warpgroup_wait requires a static count in [0, 7]",
                )
            first = self.lane // 128 * 128
            full = tuple(range(first, first + 128))
            if self.leaf_lanes is not None and not set(full) <= set(self.leaf_lanes):
                # Represent the aligned collective's participation obligation
                # with a control-only rendezvous. Do not promote a partial wait
                # into a full-warpgroup completion event.
                resource = self.capture.named_barrier((*path, "wgmma-participation"), full)
                self.capture.resources[resource]["visibility_mask"] = {
                    "shared": False, "global_read": False, "global_write": False,
                    "tensor_read": False, "tensor_write": False, "control_only": True,
                }
                self.emit("tilelang.barrier.arrive", (*path, "participation-arrive"),
                          resource=resource, invocation_id=0, count=1)
                self.emit("tilelang.barrier.observe", (*path, "participation-observe"),
                          resource=resource, invocation_id=0)
                return
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
        if name == "tl.ptx_ldmatrix":
            matrices = self.eval_int(expression.args[1]) if len(expression.args) == 4 else None
            if matrices not in {1, 2, 4}:
                raise CaptureError(
                    "TILELANG_MEMORY_ACCESS_UNRESOLVED",
                    "ptx_ldmatrix has an unsupported operand signature",
                    {"operands": len(expression.args)},
                )
            # TileLang may declare wider access pointers than the instruction
            # touches; the hardware footprint is one 32-bit register per
            # matrix and one 16-byte row per participating lane.
            destination, destination_bytes = self._ldmatrix_operand(
                expression.args[3], 2, 4 * matrices
            )
            if destination_bytes < 4 * matrices:
                raise CaptureError(
                    "TILELANG_MEMORY_ACCESS_UNRESOLVED",
                    "ptx_ldmatrix must fill one 32-bit register per matrix per lane",
                    {"bytes": destination_bytes, "matrices": matrices},
                )
            # Lane i supplies the row address of matrix i // 8; only the first
            # 8 * matrices lanes' addresses are read.
            if self.lane % 32 < 8 * matrices:
                source, byte_count = self._ldmatrix_operand(expression.args[2], 1, 16)
                if byte_count < 16:
                    raise CaptureError(
                        "TILELANG_MEMORY_ACCESS_UNRESOLVED",
                        "ptx_ldmatrix m8n8 must read exactly sixteen bytes per row",
                        {"bytes": byte_count},
                    )
                self.emit("tilelang.mem.read", (*path, "row"), region=source)
            self.emit("tilelang.mem.write", path, region=destination)
            return
        if name == "tl.ptx_stmatrix":
            args = list(expression.args)
            shape = str(args[-1].value) if args and isinstance(args[-1], self.tirx.StringImm) else "m8n8"
            if args and isinstance(args[-1], self.tirx.StringImm):
                args.pop()
            matrices = self.eval_int(args[1]) if len(args) >= 2 else None
            transpose = self.eval_int(args[0]) if args else None
            if shape != "m8n8" or matrices not in {1, 2, 4} or transpose not in {0, 1} or len(args) != 3 + matrices:
                raise CaptureError(
                    "TILELANG_MEMORY_ACCESS_UNRESOLVED",
                    "stmatrix requires m8n8, x1/x2/x4 and one register per matrix",
                    {"operands": len(expression.args), "shape": shape, "matrices": matrices},
                )
            for index, value in enumerate(args[3:]):
                self._emit_expression_loads(value, (*path, "register", index))
            # As for ldmatrix, only lanes [0, 8*num) supply row addresses;
            # the other lanes supply register data, not additional rows.
            if self.lane % 32 < 8 * matrices:
                # The access_ptr extent describes this lane's register
                # fragment (4*num bytes), not the hardware's 16-byte row.
                bits = dtype_bit_width(self.access_ptr_dtype(args[2]))
                if bits is None or 128 % bits:
                    raise CaptureError("TILELANG_MEMORY_ACCESS_UNRESOLVED",
                                       "stmatrix destination element width is unresolved")
                destination, _ = self.access_ptr_region(args[2], 2, 128 // bits)
                self.emit("tilelang.mem.write", path, region=destination)
            return
        if name == "tl.ptx_cp_async":
            if len(expression.args) not in {3, 4}:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async requires destination, source, and logical element count operands",
                )
            predicate = self.eval_int(expression.args[3]) if len(expression.args) == 4 else 1
            if predicate is None:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async predicate must be statically resolved per lane",
                    {"predicate": str(expression.args[3])},
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
            if predicate:
                source, source_bytes = self.access_ptr_region(
                    expression.args[1],
                    1,
                    logical_count,
                )
            else:
                # A false predicate copies zero source bytes: the destination
                # is zero-filled and the (possibly out-of-bounds) source is
                # never read. Nothing writes the zero-fill source.
                source = self.capture.region("cp-async-zero-fill", "global", 0, byte_count)
                source_bytes = byte_count
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
            self.bind_pending_proxy("ASYNC_COPY", [destination])
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
            # cp.async.wait_group takes any nonnegative constant; a count at
            # or above the pending groups waits for nothing.
            if wait_count is None or wait_count < 0:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_WAIT_UNRESOLVED",
                    "ptx_wait_group count must be a static nonnegative integer",
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


def _normalize_grid_selection(
    launch_grid: list[int] | tuple[int, ...] | None,
    grid_ctas: int | None,
    selected_cta: list[int] | tuple[int, ...],
) -> tuple[tuple[int, int, int], tuple[int, int, int], bool]:
    explicit = launch_grid is not None or grid_ctas is not None
    if launch_grid is None:
        if grid_ctas is None:
            launch_grid = [1]
        elif isinstance(grid_ctas, bool) or not isinstance(grid_ctas, int) or grid_ctas < 1:
            raise CaptureError(
                "TILELANG_GRID_EVIDENCE_INVALID",
                "capture requires a finite positive launch grid",
                {"grid_ctas": grid_ctas},
            )
        else:
            launch_grid = [grid_ctas]
    if (
        not isinstance(launch_grid, (list, tuple))
        or not 1 <= len(launch_grid) <= 3
        or any(
            isinstance(extent, bool) or not isinstance(extent, int) or extent < 1
            for extent in launch_grid
        )
    ):
        raise CaptureError(
            "TILELANG_GRID_EVIDENCE_INVALID",
            "launch grid must contain one to three positive integer extents",
            {"launch_grid": launch_grid},
        )
    normalized_grid = tuple((*launch_grid, *(1 for _ in range(3 - len(launch_grid)))))
    total_ctas = normalized_grid[0] * normalized_grid[1] * normalized_grid[2]
    if grid_ctas is not None and grid_ctas != total_ctas:
        raise CaptureError(
            "TILELANG_GRID_EVIDENCE_INVALID",
            "launch grid dimensions disagree with the recorded total CTA count",
            {"launch_grid": list(normalized_grid), "grid_ctas": grid_ctas},
        )
    if (
        not isinstance(selected_cta, (list, tuple))
        or len(selected_cta) != 3
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in selected_cta
        )
    ):
        raise CaptureError(
            "TILELANG_CTA_SELECTION_INVALID",
            "selected CTA must contain three nonnegative integer coordinates",
            {"selected_cta": selected_cta},
        )
    normalized_cta = tuple(selected_cta)
    if any(index >= extent for index, extent in zip(normalized_cta, normalized_grid)):
        raise CaptureError(
            "TILELANG_CTA_SELECTION_INVALID",
            "selected CTA is outside the finite launch grid",
            {"launch_grid": list(normalized_grid), "selected_cta": list(normalized_cta)},
        )
    return normalized_grid, normalized_cta, explicit


def capture_file(
    path: Path,
    source: str,
    command: str,
    descriptor_evidence_path: Path | None = None,
    binding_evidence_path: Path | None = None,
    launch_grid: list[int] | tuple[int, ...] | None = None,
    grid_ctas: int | None = None,
    selected_cta: list[int] | tuple[int, ...] = (0, 0, 0),
    wgmma_proof_output_path: Path | None = None,
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
    binding_evidence: dict[str, Any] = {}
    if binding_evidence_path is not None:
        try:
            evidence = json.loads(binding_evidence_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CaptureError(
                "TILELANG_BINDING_EVIDENCE_INVALID",
                "TileLang runtime binding evidence could not be read",
                {"path": str(binding_evidence_path), "error": str(error)},
            ) from error
        required = {
            "evidence_version",
            "source_ir_sha256",
            "scalar_values",
            "pointer_table_allocations",
            "provenance",
        }
        optional = {"buffer_values"}
        scalar_values = evidence.get("scalar_values", {}) if isinstance(evidence, dict) else {}
        pointer_tables = evidence.get("pointer_table_allocations", {}) if isinstance(evidence, dict) else {}
        buffer_values = evidence.get("buffer_values", {}) if isinstance(evidence, dict) else {}
        if (
            not isinstance(evidence, dict)
            or not required.issubset(evidence)
            or set(evidence) - required - optional
            or evidence.get("evidence_version") != "tilelang-binding-evidence-v1"
            or evidence.get("source_ir_sha256") != source_digest
            or not isinstance(scalar_values, dict)
            or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
                for name, value in scalar_values.items()
            )
            or not isinstance(pointer_tables, dict)
            or any(
                not isinstance(table, str)
                or not table
                or not isinstance(allocations, list)
                or any(not isinstance(item, str) or not item for item in allocations)
                for table, allocations in pointer_tables.items()
            )
            or not isinstance(buffer_values, dict)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(values, list)
                or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
                for name, values in buffer_values.items()
            )
            or not isinstance(evidence.get("provenance"), dict)
        ):
            raise CaptureError(
                "TILELANG_BINDING_EVIDENCE_INVALID",
                "TileLang runtime binding evidence is malformed or bound to different IR",
                {"path": str(binding_evidence_path)},
            )
        binding_evidence = evidence
    normalized_grid, normalized_cta, grid_explicit = _normalize_grid_selection(
        launch_grid,
        grid_ctas,
        selected_cta,
    )
    try:
        capture = TileLangCapture(
            tvm,
            tirx,
            module,
            source,
            command,
            source_digest,
            descriptor_evidence,
            binding_evidence,
            normalized_grid,
            normalized_cta,
            grid_explicit,
        )
        snapshot = capture.capture()
        if wgmma_proof_output_path is not None:
            proof = capture.wgmma_region_proof
            if proof is None:
                raise CaptureError(
                    "TILELANG_WGMMA_PROOF_BINDING_INVALID",
                    "WGMMA proof was not finalized with the capture",
                )
            encoded = json.dumps(
                proof, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ) + "\n"
            temporary = wgmma_proof_output_path.with_suffix(
                wgmma_proof_output_path.suffix + ".tmp"
            )
            temporary.write_text(encoded)
            temporary.replace(wgmma_proof_output_path)
        return snapshot
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
    parser.add_argument(
        "--binding-evidence",
        type=Path,
        help="SHA-bound scalar and indirect-pointer allocation evidence",
    )
    parser.add_argument("--launch-grid", nargs="+", type=int)
    parser.add_argument("--grid-ctas", type=int)
    parser.add_argument("--selected-cta", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument(
        "--wgmma-proof-output",
        type=Path,
        help="write a SHA-bound committed-group WGMMA region proof sidecar",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(
            args.input,
            args.source,
            " ".join(sys.argv),
            args.descriptor_evidence,
            args.binding_evidence,
            launch_grid=args.launch_grid,
            grid_ctas=args.grid_ctas,
            selected_cta=args.selected_cta,
            wgmma_proof_output_path=args.wgmma_proof_output,
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
