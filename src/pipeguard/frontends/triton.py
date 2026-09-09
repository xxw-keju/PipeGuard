from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ..model import Module, module_from_dict
from .common import (
    FrontendError,
    operation_to_event,
    prefixed_event_types,
    prefixed_operation_fields,
    prefixed_optional_fields,
    validate_snapshot_envelope,
)

SNAPSHOT_VERSION = "triton-snapshot-v2"
LEGACY_SNAPSHOT_VERSION = "triton-snapshot-v1"
ADAPTER_VERSION = "triton-pgir-v2"
_STREAM_ID = "program:triton:lanes"

_OPERATION_FIELDS = prefixed_operation_fields("triton")
_OPTIONAL_BY_OPERATION = prefixed_optional_fields("triton")
for _operation in _OPERATION_FIELDS:
    _OPTIONAL_BY_OPERATION[_operation] = (
        _OPTIONAL_BY_OPERATION.get(_operation, set()) | {"execution_evidence"}
    )
for _memory_operation in ("triton.mem.read", "triton.mem.write"):
    _OPTIONAL_BY_OPERATION[_memory_operation] = (
        _OPTIONAL_BY_OPERATION.get(_memory_operation, set()) | {"lane_effects"}
    )
_OPTIONAL_BY_OPERATION["triton.cp_async.issue"] = (
    _OPTIONAL_BY_OPERATION.get("triton.cp_async.issue", set()) | {"lane_effects"}
)
_EVENT_TYPES = prefixed_event_types("triton")

_RENDEZVOUS_TYPES = {
    "BarrierArrive",
    "BarrierObserve",
    "MBarrierWaitParity",
    "WgmmaFenceOperand",
    "WgmmaIssue",
    "WgmmaCommitGroup",
    "WgmmaWaitGroup",
    "ProxyFence",
}
_PER_INSTANCE_TYPES = {
    "CpAsyncIssue",
    "CpAsyncCommitGroup",
    "CpAsyncWaitGroup",
}


def _validate_snapshot(snapshot: dict[str, Any]) -> bool:
    return validate_snapshot_envelope(
        snapshot,
        dsl="Triton",
        snapshot_prefix="triton",
        stage="post-ttgir-pipeline",
        v2_adapter_version=ADAPTER_VERSION,
        profile_id=frozenset({
            "sm90-single-cta-static-v1",
            "sm90-single-cta-static-parametric-v1",
        }),
    )


def _event(
    operation: dict[str, Any],
    index: int,
    require_evidence: bool = False,
) -> dict[str, Any]:
    operation = dict(operation)
    operation.pop("lane_effects", None)
    operation.pop("execution_evidence", None)
    return operation_to_event(
        operation,
        index,
        prefix="triton",
        parity_derivation_kind="PHASE_SSA",
        operation_fields=_OPERATION_FIELDS,
        optional_by_operation=_OPTIONAL_BY_OPERATION,
        event_types=_EVENT_TYPES,
        require_evidence=require_evidence,
    )


def _resource_participants(
    resources: list[dict[str, Any]],
    actors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in resources:
        resource = deepcopy(raw)
        participants: list[Any] = []
        covered_lanes: set[int] = set()
        for participant in resource["participants"]:
            actor = actors.get(participant)
            if actor is None:
                raise FrontendError(
                    "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                    "resource references an unknown execution owner",
                    {"resource": resource.get("id"), "participant": participant},
                )
            if actor["kind"] == "ASYNC_ENGINE":
                participants.append(participant)
            elif (
                resource.get("protocol") == "WGMMA"
                or resource.get("kind") in {"cta_barrier", "named_barrier"}
            ):
                participants.append(participant)
            else:
                covered_lanes.update(actor["participants"])
        if covered_lanes:
            participants.append({
                "stream": _STREAM_ID,
                "instances": sorted(covered_lanes),
            })
        resource["participants"] = participants
        result.append(resource)
    return result


def _contract(
    event_type: str,
    participants: list[int],
) -> tuple[list[int], list[int], str, str]:
    if not participants:
        raise FrontendError(
            "TRITON_EXECUTION_SCOPE_UNRESOLVED",
            "operation has no statically known program participants",
        )
    if event_type in _RENDEZVOUS_TYPES:
        executors = (
            participants
            if event_type.startswith(("Barrier", "Wgmma", "CpAsync", "ProxyFence"))
            else [participants[0]]
        )
        return participants, executors, "ONCE", "ALL_PROGRAM_INSTANCES"
    if event_type in _PER_INSTANCE_TYPES:
        return participants, participants, "PER_INSTANCE", "NONE"
    return participants, [participants[0]], "ONCE", "NONE"


def _memory_templates(
    operation: dict[str, Any],
    event: dict[str, Any],
    actor: dict[str, Any],
    order: int,
) -> list[dict[str, Any]]:
    lane_effects = operation.get("lane_effects")
    execution_evidence = operation.get("execution_evidence")
    if lane_effects is None and execution_evidence is None:
        participants = list(actor["participants"])
        if not participants:
            raise FrontendError(
                "TRITON_LANE_OWNERSHIP_UNRESOLVED",
                "aggregate memory operation has no representative participant",
                {"operation": operation["id"], "actor": operation["actor"]},
            )
        return [{
            "id": f"{operation['id']}:aggregate",
            "type": event["type"],
            "stream": _STREAM_ID,
            "order": order,
            "program_instances": [participants[0]],
            "executors": [participants[0]],
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "owner": operation["actor"],
            "fields": {"region": event["region"]},
        }]
    if not isinstance(lane_effects, dict) or set(lane_effects) != {
        "lanes", "region_offsets", "regions",
    }:
        raise FrontendError(
            "TRITON_LANE_OWNERSHIP_UNRESOLVED",
            "ordinary memory operation lacks columnar lane-indexed effect evidence",
            {"operation": operation["id"]},
        )
    if not isinstance(execution_evidence, dict) or set(execution_evidence) != {
        "scope", "method", "source", "redundant_policy",
    }:
        raise FrontendError(
            "TRITON_LANE_OWNERSHIP_UNRESOLVED",
            "ordinary memory operation lacks structured execution evidence",
            {"operation": operation["id"]},
        )
    method = execution_evidence["method"]
    expected_policy = {
        ("ttg-blocked-linear-layout-v1", "triton.mem.read"): "LOAD_ALL",
        ("ttg-blocked-linear-layout-v1", "triton.mem.write"): "STORE_REPRESENTATIVE",
    }.get((method, operation["op"]))
    if (
        execution_evidence["scope"] != "INSTANCE"
        or expected_policy is None
        or not isinstance(execution_evidence["source"], str)
        or not execution_evidence["source"]
        or execution_evidence["redundant_policy"] != expected_policy
    ):
        raise FrontendError(
            "TRITON_LANE_OWNERSHIP_UNRESOLVED",
            "ordinary memory execution evidence is incomplete or inconsistent",
            {"operation": operation["id"], "evidence": execution_evidence},
        )
    participant_set = set(actor["participants"])
    lanes = lane_effects["lanes"]
    offsets = lane_effects["region_offsets"]
    region_values = lane_effects["regions"]
    if (
        not isinstance(lanes, list)
        or not lanes
        or any(isinstance(lane, bool) or not isinstance(lane, int) for lane in lanes)
        or lanes != sorted(set(lanes))
        or not set(lanes).issubset(participant_set)
        or not isinstance(offsets, list)
        or len(offsets) != len(lanes) + 1
        or offsets[0] != 0
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        or any(first >= second for first, second in zip(offsets, offsets[1:]))
        or not isinstance(region_values, list)
        or offsets[-1] != len(region_values)
        or any(not isinstance(region, str) or not region for region in region_values)
    ):
        raise FrontendError(
            "TRITON_LANE_OWNERSHIP_UNRESOLVED",
            "columnar lane effects must provide sorted participants and valid CSR regions",
            {"operation": operation["id"]},
        )
    lane_regions = {
        lane: tuple(region_values[offsets[index]:offsets[index + 1]])
        for index, lane in enumerate(lanes)
    }
    templates: list[dict[str, Any]] = []
    for access in range(max(len(regions) for regions in lane_regions.values())):
        active_lanes = [
            lane for lane, regions in sorted(lane_regions.items())
            if access < len(regions)
        ]
        templates.append({
            "id": f"{operation['id']}:access:{access}",
            "type": event["type"],
            "stream": _STREAM_ID,
            "order": order + access,
            "program_instances": active_lanes,
            "executors": active_lanes,
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {},
            "instance_fields": {
                "region": [lane_regions[lane][access] for lane in active_lanes],
            },
        })
    return templates


def _cp_async_templates(
    operation: dict[str, Any],
    event: dict[str, Any],
    actor: dict[str, Any],
    order: int,
) -> list[dict[str, Any]]:
    execution = operation.get("execution_evidence")
    lane_effects = operation.get("lane_effects")
    if (
        not isinstance(execution, dict)
        or set(execution) != {
            "program_instances", "executors", "transition", "rendezvous", "method"
        }
        or execution["transition"] != "PER_INSTANCE"
        or execution["rendezvous"] != "NONE"
        or execution["program_instances"] != execution["executors"]
        or execution["method"] != "ttgir-final-nvvm-cp-async-per-thread-v1"
    ):
        raise FrontendError(
            "TRITON_EXECUTION_SCOPE_UNRESOLVED",
            "cp.async issue lacks exact per-thread lowering evidence",
            {"operation": operation["id"], "evidence": execution},
        )
    if not isinstance(lane_effects, dict) or set(lane_effects) != {
        "lanes", "transfer_offsets", "source_regions", "destination_regions", "bytes"
    }:
        raise FrontendError(
            "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED",
            "cp.async issue lacks columnar per-thread transfer evidence",
            {"operation": operation["id"]},
        )
    lanes = lane_effects["lanes"]
    offsets = lane_effects["transfer_offsets"]
    sources = lane_effects["source_regions"]
    destinations = lane_effects["destination_regions"]
    byte_counts = lane_effects["bytes"]
    participant_set = set(actor["participants"])
    if (
        not isinstance(lanes, list)
        or not lanes
        or lanes != sorted(set(lanes))
        or any(isinstance(lane, bool) or not isinstance(lane, int) for lane in lanes)
        or not set(lanes).issubset(participant_set)
        or execution["program_instances"] != lanes
        or not isinstance(offsets, list)
        or len(offsets) != len(lanes) + 1
        or offsets[0] != 0
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        or any(first >= second for first, second in zip(offsets, offsets[1:]))
        or offsets[-1] != len(sources)
        or len(sources) != len(destinations)
        or len(sources) != len(byte_counts)
        or any(not isinstance(region, str) or not region for region in (*sources, *destinations))
        or any(value not in {4, 8, 16} for value in byte_counts)
    ):
        raise FrontendError(
            "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED",
            "cp.async per-thread transfer evidence is malformed",
            {"operation": operation["id"], "lane_effects": lane_effects},
        )
    templates: list[dict[str, Any]] = []
    for access in range(max(
        offsets[index + 1] - offsets[index] for index in range(len(lanes))
    )):
        active = [
            (lane, offsets[index] + access)
            for index, lane in enumerate(lanes)
            if offsets[index] + access < offsets[index + 1]
        ]
        active_lanes = [lane for lane, _index in active]
        indices = [index for _lane, index in active]
        templates.append({
            "id": f"{operation['id']}:access:{access}",
            "type": event["type"],
            "stream": _STREAM_ID,
            "order": order + access,
            "program_instances": active_lanes,
            "executors": active_lanes,
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {
                "resource": event["resource"],
                "group_owner": "$self",
                "predicate": True,
            },
            "instance_fields": {
                "source_region": [sources[index] for index in indices],
                "destination_region": [destinations[index] for index in indices],
                "bytes": [byte_counts[index] for index in indices],
            },
        })
    return templates


def _parametric_module(
    snapshot: dict[str, Any],
    events: list[dict[str, Any]],
) -> Module:
    operations = snapshot["operations"]
    actors = {actor["id"]: actor for actor in snapshot["actors"]}
    requires_execution_evidence = (
        snapshot["profile"]["id"] == "sm90-single-cta-static-parametric-v1"
    )
    templates: list[dict[str, Any]] = []
    operation_templates: dict[str, list[dict[str, Any]]] = {}

    for operation, event in zip(operations, events):
        actor = actors.get(operation["actor"])
        if actor is None or actor["kind"] == "ASYNC_ENGINE":
            raise FrontendError(
                "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                "source operation references an invalid execution owner",
                {"operation": operation["id"], "actor": operation["actor"]},
            )
        order = len(templates)
        if operation["op"] in {"triton.mem.read", "triton.mem.write"}:
            created = _memory_templates(operation, event, actor, order)
        elif event["type"] == "CpAsyncIssue":
            created = _cp_async_templates(operation, event, actor, order)
        else:
            execution = operation.get("execution_evidence")
            if execution is None:
                if requires_execution_evidence:
                    raise FrontendError(
                        "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                        "parameterized-profile protocol operation lacks lowered execution evidence",
                        {"operation": operation["id"]},
                    )
                participants, executors, transition, rendezvous = _contract(
                    event["type"], list(actor["participants"])
                )
            elif (
                not isinstance(execution, dict)
                or set(execution) != {
                    "program_instances", "executors", "transition", "rendezvous", "method"
                }
                or not isinstance(execution["method"], str)
                or not execution["method"]
            ):
                raise FrontendError(
                    "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                    "protocol operation lacks complete lowered execution evidence",
                    {"operation": operation["id"]},
                )
            else:
                participants = list(execution["program_instances"])
                executors = list(execution["executors"])
                transition = execution["transition"]
                rendezvous = execution["rendezvous"]
            if event["type"] in {"CpAsyncCommitGroup", "CpAsyncWaitGroup"} and (
                participants != list(actor["participants"])
                or executors != list(actor["participants"])
                or transition != "PER_INSTANCE"
                or rendezvous != "NONE"
                or execution.get("method")
                != "ttgir-final-nvvm-cp-async-per-thread-v1"
            ):
                raise FrontendError(
                    "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                    "cp.async commit/wait lacks exact per-thread lowering evidence",
                    {"operation": operation["id"], "evidence": execution},
                )
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"id", "type", "actor", "seq", "depends_on"}
            }
            if (
                transition == "PER_INSTANCE"
                and event["type"].startswith(("CpAsync", "TmaStore"))
                and "group_owner" in fields
            ):
                fields["group_owner"] = "$self"
            template = {
                "id": operation["id"],
                "type": event["type"],
                "stream": _STREAM_ID,
                "order": order,
                "program_instances": participants,
                "executors": executors,
                "transition": transition,
                "rendezvous": rendezvous,
                "fields": fields,
            }
            if event["type"].startswith(("Barrier", "Wgmma")) or (
                event["type"].startswith("CpAsync") and transition == "ONCE"
            ):
                template["owner"] = operation["actor"]
            created = [template]
        templates.extend(created)
        operation_templates[operation["id"]] = created

    for operation in operations:
        dependencies = operation.get("depends_on", [])
        if not dependencies:
            continue
        for current in operation_templates[operation["id"]]:
            encoded: list[dict[str, str]] = []
            for dependency_id in dependencies:
                target_templates = operation_templates.get(dependency_id)
                if target_templates is None:
                    raise FrontendError(
                        "TRITON_EXECUTION_SCOPE_UNRESOLVED",
                        "operation dependency references an unknown captured operation",
                        {"operation": operation["id"], "depends_on": dependency_id},
                    )
                for target in target_templates:
                    if target["transition"] == "ONCE":
                        relation = "COLLECTIVE"
                    elif set(current["program_instances"]).issubset(
                        target["program_instances"]
                    ):
                        relation = "SAME_INSTANCE"
                    else:
                        relation = "ALL_INSTANCES"
                    encoded.append({"template": target["id"], "relation": relation})
            current["dependencies"] = encoded

    extraction = snapshot["extraction"]
    digest = snapshot["profile"]["source_ir_sha256"]
    return module_from_dict({
        "schema_version": "pgir-parametric-v1",
        "producer": {"dsl": "TRITON", **snapshot["producer"]},
        "target": snapshot["target"],
        "assumptions": {
            "finite_control_flow": extraction["finite_control_flow"],
            "statically_known_participants": extraction["statically_known_participants"],
            "fair_async_completion": extraction["fair_async_completion"],
            "no_cluster_effects": extraction["no_cluster_effects"],
            "alias_precision": extraction["alias_precision"],
            "parity_normalization_status": extraction["parity_normalization_status"],
        },
        "control_structure": {"finite_event_graph": extraction["finite_control_flow"]},
        "owners": [dict(actor) for actor in snapshot["actors"]],
        "program_streams": [{
            "id": _STREAM_ID,
            "kind": "LANE",
            "instances": list(range(snapshot["target"]["cta_threads"])),
        }],
        "resources": _resource_participants(snapshot["resources"], actors),
        "regions": snapshot["regions"],
        "event_templates": templates,
        "frontend_provenance": {
            "extraction_complete": extraction["complete"],
            "source": extraction["source"],
        },
        "frontend_evidence": {
            "participant_mapping": {
                "status": "RESOLVED",
                "method": extraction["participant_evidence"],
                "source": digest,
            },
            "lane_ownership": {
                "status": "RESOLVED",
                "method": "ttgir-post-coalesce-layout-inversion-v1",
                "source": digest,
            },
            "predicate_mapping": {
                "status": "RESOLVED",
                "method": "ttgir-static-mask-plus-redundant-thread-predicate-v1",
                "source": digest,
            },
            "alias": {
                "status": "EXACT",
                "method": extraction["alias_evidence"],
                "source": digest,
            },
        },
        "unsupported_constructs": snapshot["unsupported_constructs"],
    })


def _legacy_explicit_module(
    snapshot: dict[str, Any],
    events: list[dict[str, Any]],
) -> Module:
    producer = snapshot["producer"]
    extraction = snapshot["extraction"]
    return module_from_dict({
        "schema_version": "pgir-v1",
        "producer": {
            "dsl": "TRITON",
            "dsl_version": producer["dsl_version"],
            "dsl_commit": producer["dsl_commit"],
            "backend_commit": producer["backend_commit"],
            "adapter_version": producer["adapter_version"],
            "command": producer["command"],
        },
        "target": snapshot["target"],
        "assumptions": {
            "finite_control_flow": extraction["finite_control_flow"],
            "statically_known_participants": extraction["statically_known_participants"],
            "fair_async_completion": extraction["fair_async_completion"],
            "no_cluster_effects": extraction["no_cluster_effects"],
            "alias_precision": extraction["alias_precision"],
            "parity_normalization_status": extraction["parity_normalization_status"],
        },
        "control_structure": {"finite_event_graph": extraction["finite_control_flow"]},
        "actors": snapshot["actors"],
        "resources": snapshot["resources"],
        "regions": snapshot["regions"],
        "events": events,
        "frontend_provenance": {
            "extraction_complete": extraction["complete"],
            "source": extraction["source"],
        },
        "unsupported_constructs": snapshot["unsupported_constructs"],
    })


def triton_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError("FRONTEND_FIELD_TYPE", "Triton snapshot root must be an object")
    require_evidence = _validate_snapshot(snapshot)
    events = [
        _event(operation, index, require_evidence)
        for index, operation in enumerate(snapshot["operations"])
    ]
    unified_evidence = (
        snapshot.get("profile", {}).get("id")
        == "sm90-single-cta-static-parametric-v1"
        or any(
            "lane_effects" in operation or "execution_evidence" in operation
            for operation in snapshot["operations"]
        )
    )
    if snapshot["snapshot_version"] == LEGACY_SNAPSHOT_VERSION or not unified_evidence:
        return _legacy_explicit_module(snapshot, events)
    return _parametric_module(snapshot, events)


def load_triton_snapshot(path: str | Path) -> Module:
    try:
        snapshot = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FrontendError(
            "FRONTEND_INPUT_INVALID",
            "Triton snapshot could not be read as JSON",
            {"path": str(path), "error": str(error)},
        ) from error
    return triton_snapshot_to_module(snapshot)
