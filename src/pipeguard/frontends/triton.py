from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import Module, module_from_dict
from .common import FrontendError, require_exact_keys

SNAPSHOT_VERSION = "triton-snapshot-v1"
ADAPTER_VERSION = "triton-pgir-v1"

_COMMON_OPERATION_FIELDS = {"id", "op", "actor", "seq"}
_OPTIONAL_OPERATION_FIELDS = {"depends_on"}
_OPERATION_FIELDS = {
    "triton.mem.read": {"region"},
    "triton.mem.write": {"region"},
    "triton.barrier.arrive": {"resource", "invocation_id", "count"},
    "triton.barrier.observe": {"resource", "invocation_id"},
    "triton.mbarrier.init": {"resource", "lifecycle_epoch", "count"},
    "triton.mbarrier.invalidate": {"resource", "lifecycle_epoch"},
    "triton.mbarrier.arrive": {"resource", "lifecycle_epoch", "generation", "count"},
    "triton.mbarrier.expect_tx": {"resource", "lifecycle_epoch", "generation", "bytes"},
    "triton.mbarrier.arrive_expect_tx": {"resource", "lifecycle_epoch", "generation", "count", "bytes"},
    "triton.mbarrier.wait_parity": {
        "resource", "lifecycle_epoch", "generation", "raw_parity",
        "logical_stream_id", "stream_rank", "polling_coalesced",
    },
    "triton.tma_load.issue": {"resource", "source_region", "destination_region", "bytes", "lifecycle_epoch", "generation"},
    "triton.cp_async.issue": {"resource", "source_region", "destination_region", "bytes", "group_owner"},
    "triton.cp_async.commit_group": {"resource", "group_owner", "group_seq"},
    "triton.cp_async.wait_group": {"resource", "group_owner", "wait_group_count"},
    "triton.wgmma.fence_operand": {"resource", "group_owner", "regions"},
    "triton.wgmma.issue": {"resource", "group_owner", "regions", "destination_region"},
    "triton.wgmma.commit_group": {"resource", "group_owner", "group_seq"},
    "triton.wgmma.wait_group": {"resource", "group_owner", "wait_group_count", "completion_kind"},
    "triton.tma_store.issue": {"resource", "group_owner", "source_region", "destination_region", "bytes"},
    "triton.tma_store.commit_group": {"resource", "group_owner", "group_seq"},
    "triton.tma_store.wait_group": {"resource", "group_owner", "wait_group_count", "wait_mode"},
    "triton.channel.create": {"resource"},
    "triton.channel.producer_acquire": {"resource", "slot", "logical_phase"},
    "triton.channel.producer_commit": {"resource", "slot", "logical_phase"},
    "triton.channel.consumer_wait": {"resource", "slot", "logical_phase"},
    "triton.channel.consumer_release": {"resource", "slot", "logical_phase"},
    "triton.proxy_fence": {"from_domain", "to_domain", "scope", "regions", "visibility_mask"},
}
_OPTIONAL_BY_OPERATION = {
    "triton.barrier.arrive": {"predicate"},
    "triton.mbarrier.arrive": {"predicate"},
    "triton.mbarrier.arrive_expect_tx": {"predicate"},
    "triton.tma_load.issue": {"engine_id", "engine_seq"},
    "triton.cp_async.issue": {"engine_id", "engine_seq", "predicate"},
    "triton.wgmma.issue": {"engine_id", "engine_seq"},
    "triton.tma_store.issue": {"engine_id", "engine_seq"},
}
_EVENT_TYPES = {
    "triton.mem.read": "MemRead",
    "triton.mem.write": "MemWrite",
    "triton.barrier.arrive": "BarrierArrive",
    "triton.barrier.observe": "BarrierObserve",
    "triton.mbarrier.init": "MBarrierInit",
    "triton.mbarrier.invalidate": "MBarrierInvalidate",
    "triton.mbarrier.arrive": "MBarrierArrive",
    "triton.mbarrier.expect_tx": "MBarrierExpectTx",
    "triton.mbarrier.arrive_expect_tx": "MBarrierArriveExpectTx",
    "triton.mbarrier.wait_parity": "MBarrierWaitParity",
    "triton.tma_load.issue": "TmaLoadIssue",
    "triton.cp_async.issue": "CpAsyncIssue",
    "triton.cp_async.commit_group": "CpAsyncCommitGroup",
    "triton.cp_async.wait_group": "CpAsyncWaitGroup",
    "triton.wgmma.fence_operand": "WgmmaFenceOperand",
    "triton.wgmma.issue": "WgmmaIssue",
    "triton.wgmma.commit_group": "WgmmaCommitGroup",
    "triton.wgmma.wait_group": "WgmmaWaitGroup",
    "triton.tma_store.issue": "TmaStoreIssue",
    "triton.tma_store.commit_group": "TmaStoreCommitGroup",
    "triton.tma_store.wait_group": "TmaStoreWaitGroup",
    "triton.channel.create": "ChannelCreate",
    "triton.channel.producer_acquire": "ProducerAcquire",
    "triton.channel.producer_commit": "ProducerCommit",
    "triton.channel.consumer_wait": "ConsumerWait",
    "triton.channel.consumer_release": "ConsumerRelease",
    "triton.proxy_fence": "ProxyFence",
}


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    require_exact_keys(
        snapshot,
        {
            "snapshot_version",
            "producer",
            "target",
            "extraction",
            "actors",
            "resources",
            "regions",
            "operations",
            "unsupported_constructs",
        },
        set(),
        "/",
    )
    if snapshot["snapshot_version"] != SNAPSHOT_VERSION:
        raise FrontendError(
            "FRONTEND_SCHEMA_VERSION_UNSUPPORTED",
            "Triton snapshot version is not supported",
            {"snapshot_version": snapshot["snapshot_version"]},
        )
    require_exact_keys(
        snapshot["producer"],
        {"dsl_version", "dsl_commit", "backend_commit", "adapter_version", "command"},
        set(),
        "/producer",
    )
    require_exact_keys(
        snapshot["target"],
        {"architecture", "warp_size", "cta_threads", "shared_memory_bytes", "grid_scope"},
        set(),
        "/target",
    )
    require_exact_keys(
        snapshot["extraction"],
        {
            "complete",
            "stage",
            "source",
            "finite_control_flow",
            "statically_known_participants",
            "fair_async_completion",
            "no_cluster_effects",
            "alias_precision",
            "parity_normalization_status",
        },
        set(),
        "/extraction",
    )
    extraction = snapshot["extraction"]
    if not isinstance(extraction["complete"], bool):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "Triton extraction complete field must be Boolean",
            {"location": "/extraction/complete"},
        )
    if not extraction["complete"]:
        raise FrontendError(
            "FRONTEND_EXTRACTION_INCOMPLETE",
            "Triton extraction is incomplete and cannot produce PGIR",
        )
    if extraction["stage"] != "post-ttgir-pipeline":
        raise FrontendError(
            "FRONTEND_STAGE_UNSUPPORTED",
            "Triton snapshot must be captured after the TTGIR pipeline",
            {"stage": extraction["stage"]},
        )
    if snapshot["unsupported_constructs"]:
        raise FrontendError(
            "FRONTEND_UNSUPPORTED_CONSTRUCTS",
            "Triton snapshot contains unsupported constructs",
            {"constructs": snapshot["unsupported_constructs"]},
        )
    if not isinstance(snapshot["operations"], list) or not snapshot["operations"]:
        raise FrontendError(
            "FRONTEND_OPERATIONS_MISSING",
            "Triton snapshot must contain a nonempty operation list",
        )


def _event(operation: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "Triton operation must be an object",
            {"location": f"/operations/{index}"},
        )
    op = operation.get("op")
    if op not in _OPERATION_FIELDS:
        raise FrontendError(
            "FRONTEND_UNKNOWN_OPERATION",
            "Triton operation has no sound PGIR mapping",
            {"location": f"/operations/{index}", "operation": op},
        )
    required = _COMMON_OPERATION_FIELDS | _OPERATION_FIELDS[op]
    optional = _OPTIONAL_OPERATION_FIELDS | _OPTIONAL_BY_OPERATION.get(op, set())
    require_exact_keys(operation, required, optional, f"/operations/{index}")
    event = {
        "id": operation["id"],
        "type": _EVENT_TYPES[op],
        "actor": operation["actor"],
        "seq": operation["seq"],
    }
    if "depends_on" in operation:
        event["depends_on"] = operation["depends_on"]
    excluded = _COMMON_OPERATION_FIELDS | {
        "op", "logical_stream_id", "stream_rank", "polling_coalesced",
    }
    for field, value in operation.items():
        if field not in excluded and field != "depends_on":
            event[field] = value
    if op == "triton.mbarrier.wait_parity":
        event["parity_evidence"] = {
            "derivation_kind": "PHASE_SSA",
            "logical_stream_id": operation["logical_stream_id"],
            "lifecycle_epoch": operation["lifecycle_epoch"],
            "stream_rank": operation["stream_rank"],
            "polling_coalesced": operation["polling_coalesced"],
        }
    return event


def triton_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError("FRONTEND_FIELD_TYPE", "Triton snapshot root must be an object")
    _validate_snapshot(snapshot)
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
        "events": [_event(operation, index) for index, operation in enumerate(snapshot["operations"])],
        "frontend_provenance": {
            "extraction_complete": extraction["complete"],
            "source": extraction["source"],
        },
        "unsupported_constructs": snapshot["unsupported_constructs"],
    })


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
