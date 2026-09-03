from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import Module, module_from_dict
from .common import FrontendError, require_exact_keys

SNAPSHOT_VERSION = "tilelang-snapshot-v1"
ADAPTER_VERSION = "tilelang-pgir-v1"

_COMMON_OPERATION_FIELDS = {"id", "op", "actor", "seq"}
_OPTIONAL_OPERATION_FIELDS = {"depends_on"}
_OPERATION_FIELDS = {
    "tilelang.mem.read": {"region"},
    "tilelang.mem.write": {"region"},
    "tilelang.barrier.arrive": {"resource", "invocation_id", "count"},
    "tilelang.barrier.observe": {"resource", "invocation_id"},
    "tilelang.mbarrier.init": {"resource", "lifecycle_epoch", "count"},
    "tilelang.mbarrier.invalidate": {"resource", "lifecycle_epoch"},
    "tilelang.mbarrier.arrive": {"resource", "lifecycle_epoch", "generation", "count"},
    "tilelang.mbarrier.expect_tx": {"resource", "lifecycle_epoch", "generation", "bytes"},
    "tilelang.mbarrier.arrive_expect_tx": {"resource", "lifecycle_epoch", "generation", "count", "bytes"},
    "tilelang.mbarrier.wait_parity": {
        "resource", "lifecycle_epoch", "generation", "raw_parity",
        "logical_stream_id", "stream_rank", "polling_coalesced",
    },
    "tilelang.tma_load.issue": {"resource", "source_region", "destination_region", "bytes", "lifecycle_epoch", "generation"},
    "tilelang.cp_async.issue": {"resource", "source_region", "destination_region", "bytes", "group_owner"},
    "tilelang.cp_async.commit_group": {"resource", "group_owner", "group_seq"},
    "tilelang.cp_async.wait_group": {"resource", "group_owner", "wait_group_count"},
    "tilelang.wgmma.fence_operand": {"resource", "group_owner", "regions"},
    "tilelang.wgmma.issue": {"resource", "group_owner", "regions", "destination_region"},
    "tilelang.wgmma.commit_group": {"resource", "group_owner", "group_seq"},
    "tilelang.wgmma.wait_group": {"resource", "group_owner", "wait_group_count", "completion_kind"},
    "tilelang.tma_store.issue": {"resource", "group_owner", "source_region", "destination_region", "bytes"},
    "tilelang.tma_store.commit_group": {"resource", "group_owner", "group_seq"},
    "tilelang.tma_store.wait_group": {"resource", "group_owner", "wait_group_count", "wait_mode"},
    "tilelang.channel.create": {"resource"},
    "tilelang.channel.producer_acquire": {"resource", "slot", "logical_phase"},
    "tilelang.channel.producer_commit": {"resource", "slot", "logical_phase"},
    "tilelang.channel.consumer_wait": {"resource", "slot", "logical_phase"},
    "tilelang.channel.consumer_release": {"resource", "slot", "logical_phase"},
    "tilelang.proxy_fence": {"from_domain", "to_domain", "scope", "regions", "visibility_mask"},
}
_OPTIONAL_BY_OPERATION = {
    "tilelang.barrier.arrive": {"predicate"},
    "tilelang.mbarrier.arrive": {"predicate"},
    "tilelang.mbarrier.arrive_expect_tx": {"predicate"},
    "tilelang.tma_load.issue": {"engine_id", "engine_seq"},
    "tilelang.cp_async.issue": {"engine_id", "engine_seq", "predicate"},
    "tilelang.wgmma.issue": {"engine_id", "engine_seq"},
    "tilelang.tma_store.issue": {"engine_id", "engine_seq"},
}
_EVENT_TYPES = {
    "tilelang.mem.read": "MemRead",
    "tilelang.mem.write": "MemWrite",
    "tilelang.barrier.arrive": "BarrierArrive",
    "tilelang.barrier.observe": "BarrierObserve",
    "tilelang.mbarrier.init": "MBarrierInit",
    "tilelang.mbarrier.invalidate": "MBarrierInvalidate",
    "tilelang.mbarrier.arrive": "MBarrierArrive",
    "tilelang.mbarrier.expect_tx": "MBarrierExpectTx",
    "tilelang.mbarrier.arrive_expect_tx": "MBarrierArriveExpectTx",
    "tilelang.mbarrier.wait_parity": "MBarrierWaitParity",
    "tilelang.tma_load.issue": "TmaLoadIssue",
    "tilelang.cp_async.issue": "CpAsyncIssue",
    "tilelang.cp_async.commit_group": "CpAsyncCommitGroup",
    "tilelang.cp_async.wait_group": "CpAsyncWaitGroup",
    "tilelang.wgmma.fence_operand": "WgmmaFenceOperand",
    "tilelang.wgmma.issue": "WgmmaIssue",
    "tilelang.wgmma.commit_group": "WgmmaCommitGroup",
    "tilelang.wgmma.wait_group": "WgmmaWaitGroup",
    "tilelang.tma_store.issue": "TmaStoreIssue",
    "tilelang.tma_store.commit_group": "TmaStoreCommitGroup",
    "tilelang.tma_store.wait_group": "TmaStoreWaitGroup",
    "tilelang.channel.create": "ChannelCreate",
    "tilelang.channel.producer_acquire": "ProducerAcquire",
    "tilelang.channel.producer_commit": "ProducerCommit",
    "tilelang.channel.consumer_wait": "ConsumerWait",
    "tilelang.channel.consumer_release": "ConsumerRelease",
    "tilelang.proxy_fence": "ProxyFence",
}


def _require_boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend snapshot field must be Boolean",
            {"location": location},
        )
    return value


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
            "TileLang snapshot version is not supported",
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
    if not _require_boolean(extraction["complete"], "/extraction/complete"):
        raise FrontendError(
            "FRONTEND_EXTRACTION_INCOMPLETE",
            "TileLang extraction is incomplete and cannot produce PGIR",
        )
    if extraction["stage"] != "post-inject-fence-proxy":
        raise FrontendError(
            "FRONTEND_STAGE_UNSUPPORTED",
            "TileLang snapshot must be captured after proxy-fence insertion",
            {"stage": extraction["stage"]},
        )
    if snapshot["unsupported_constructs"]:
        raise FrontendError(
            "FRONTEND_UNSUPPORTED_CONSTRUCTS",
            "TileLang snapshot contains unsupported constructs",
            {"constructs": snapshot["unsupported_constructs"]},
        )
    if not isinstance(snapshot["operations"], list) or not snapshot["operations"]:
        raise FrontendError(
            "FRONTEND_OPERATIONS_MISSING",
            "TileLang snapshot must contain a nonempty operation list",
        )


def _event(operation: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang operation must be an object",
            {"location": f"/operations/{index}"},
        )
    op = operation.get("op")
    if op not in _OPERATION_FIELDS:
        raise FrontendError(
            "FRONTEND_UNKNOWN_OPERATION",
            "TileLang operation has no sound PGIR mapping",
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
    if op == "tilelang.mbarrier.wait_parity":
        event["parity_evidence"] = {
            "derivation_kind": "PARITY_EXPRESSION",
            "logical_stream_id": operation["logical_stream_id"],
            "lifecycle_epoch": operation["lifecycle_epoch"],
            "stream_rank": operation["stream_rank"],
            "polling_coalesced": operation["polling_coalesced"],
        }
    return event


def tilelang_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang snapshot root must be an object",
        )
    _validate_snapshot(snapshot)
    producer = snapshot["producer"]
    extraction = snapshot["extraction"]
    raw = {
        "schema_version": "pgir-v1",
        "producer": {
            "dsl": "TILELANG",
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
        "events": [
            _event(operation, index)
            for index, operation in enumerate(snapshot["operations"])
        ],
        "frontend_provenance": {
            "extraction_complete": extraction["complete"],
            "source": extraction["source"],
        },
        "unsupported_constructs": snapshot["unsupported_constructs"],
    }
    return module_from_dict(raw)


def load_tilelang_snapshot(path: str | Path) -> Module:
    try:
        snapshot = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FrontendError(
            "FRONTEND_INPUT_INVALID",
            "TileLang snapshot could not be read as JSON",
            {"path": str(path), "error": str(error)},
        ) from error
    return tilelang_snapshot_to_module(snapshot)
