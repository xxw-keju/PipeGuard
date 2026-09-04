from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


class FrontendError(Exception):
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


def require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    if not isinstance(value, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend snapshot value must be an object",
            {"location": location},
        )
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise FrontendError(
            "FRONTEND_FIELDS_MISSING",
            "frontend snapshot object is missing required fields",
            {"location": location, "fields": missing},
        )
    if unknown:
        raise FrontendError(
            "FRONTEND_FIELDS_UNKNOWN",
            "frontend snapshot object contains unknown fields",
            {"location": location, "fields": unknown},
        )


COMMON_OPERATION_FIELDS = {"id", "op", "actor", "seq"}
OPTIONAL_OPERATION_FIELDS = {"depends_on", "evidence"}
CANONICAL_OPERATION_FIELDS = {
    "mem.read": {"region"},
    "mem.write": {"region"},
    "barrier.arrive": {"resource", "invocation_id", "count"},
    "barrier.observe": {"resource", "invocation_id"},
    "mbarrier.init": {"resource", "lifecycle_epoch", "count"},
    "mbarrier.invalidate": {"resource", "lifecycle_epoch"},
    "mbarrier.arrive": {"resource", "lifecycle_epoch", "generation", "count"},
    "mbarrier.expect_tx": {"resource", "lifecycle_epoch", "generation", "bytes"},
    "mbarrier.arrive_expect_tx": {
        "resource", "lifecycle_epoch", "generation", "count", "bytes",
    },
    "mbarrier.wait_parity": {
        "resource", "lifecycle_epoch", "generation", "raw_parity",
        "logical_stream_id", "stream_rank", "polling_coalesced",
    },
    "tma_load.issue": {
        "resource", "source_region", "destination_region", "bytes",
        "lifecycle_epoch", "generation",
    },
    "cp_async.issue": {
        "resource", "source_region", "destination_region", "bytes", "group_owner",
    },
    "cp_async.commit_group": {"resource", "group_owner", "group_seq"},
    "cp_async.wait_group": {"resource", "group_owner", "wait_group_count"},
    "wgmma.fence_operand": {"resource", "group_owner", "regions"},
    "wgmma.issue": {"resource", "group_owner", "regions", "destination_region"},
    "wgmma.commit_group": {"resource", "group_owner", "group_seq"},
    "wgmma.wait_group": {
        "resource", "group_owner", "wait_group_count", "completion_kind",
    },
    "tma_store.issue": {
        "resource", "group_owner", "source_region", "destination_region", "bytes",
    },
    "tma_store.commit_group": {"resource", "group_owner", "group_seq"},
    "tma_store.wait_group": {
        "resource", "group_owner", "wait_group_count", "wait_mode",
    },
    "channel.create": {"resource"},
    "channel.producer_acquire": {"resource", "slot", "logical_phase"},
    "channel.producer_commit": {"resource", "slot", "logical_phase"},
    "channel.consumer_wait": {"resource", "slot", "logical_phase"},
    "channel.consumer_release": {"resource", "slot", "logical_phase"},
    "proxy_fence": {"from_domain", "to_domain", "scope", "regions", "visibility_mask"},
}
CANONICAL_OPTIONAL_FIELDS = {
    "barrier.arrive": {"predicate"},
    "mbarrier.arrive": {"predicate"},
    "mbarrier.arrive_expect_tx": {"predicate"},
    "tma_load.issue": {"engine_id", "engine_seq"},
    "cp_async.issue": {"engine_id", "engine_seq", "predicate"},
    "wgmma.issue": {"engine_id", "engine_seq"},
    "tma_store.issue": {"engine_id", "engine_seq"},
}
CANONICAL_EVENT_TYPES = {
    "mem.read": "MemRead",
    "mem.write": "MemWrite",
    "barrier.arrive": "BarrierArrive",
    "barrier.observe": "BarrierObserve",
    "mbarrier.init": "MBarrierInit",
    "mbarrier.invalidate": "MBarrierInvalidate",
    "mbarrier.arrive": "MBarrierArrive",
    "mbarrier.expect_tx": "MBarrierExpectTx",
    "mbarrier.arrive_expect_tx": "MBarrierArriveExpectTx",
    "mbarrier.wait_parity": "MBarrierWaitParity",
    "tma_load.issue": "TmaLoadIssue",
    "cp_async.issue": "CpAsyncIssue",
    "cp_async.commit_group": "CpAsyncCommitGroup",
    "cp_async.wait_group": "CpAsyncWaitGroup",
    "wgmma.fence_operand": "WgmmaFenceOperand",
    "wgmma.issue": "WgmmaIssue",
    "wgmma.commit_group": "WgmmaCommitGroup",
    "wgmma.wait_group": "WgmmaWaitGroup",
    "tma_store.issue": "TmaStoreIssue",
    "tma_store.commit_group": "TmaStoreCommitGroup",
    "tma_store.wait_group": "TmaStoreWaitGroup",
    "channel.create": "ChannelCreate",
    "channel.producer_acquire": "ProducerAcquire",
    "channel.producer_commit": "ProducerCommit",
    "channel.consumer_wait": "ConsumerWait",
    "channel.consumer_release": "ConsumerRelease",
    "proxy_fence": "ProxyFence",
}

_BASE_EXTRACTION_FIELDS = {
    "complete",
    "stage",
    "source",
    "finite_control_flow",
    "statically_known_participants",
    "fair_async_completion",
    "no_cluster_effects",
    "alias_precision",
    "parity_normalization_status",
}
_V2_EXTRACTION_FIELDS = {
    "static_expansions",
    "executable_nodes_seen",
    "classified_nodes",
    "emitting_nodes",
    "elided_nodes",
    "events_emitted",
    "allocation_identity_evidence",
    "alias_evidence",
    "participant_evidence",
}
_EVIDENCE_FIELDS = {
    "source_location",
    "activation_path",
    "recognizer_rule",
    "ir_operation",
}


def prefixed_operation_fields(prefix: str) -> dict[str, set[str]]:
    return {
        f"{prefix}.{suffix}": fields
        for suffix, fields in CANONICAL_OPERATION_FIELDS.items()
    }


def prefixed_optional_fields(prefix: str) -> dict[str, set[str]]:
    return {
        f"{prefix}.{suffix}": fields
        for suffix, fields in CANONICAL_OPTIONAL_FIELDS.items()
    }


def prefixed_event_types(prefix: str) -> dict[str, str]:
    return {
        f"{prefix}.{suffix}": event_type
        for suffix, event_type in CANONICAL_EVENT_TYPES.items()
    }


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend snapshot field must be a nonempty string",
            {"location": location},
        )
    return value


def _require_nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend snapshot accounting field must be a nonnegative integer",
            {"location": location},
        )
    return value


def validate_snapshot_envelope(
    snapshot: dict[str, Any],
    *,
    dsl: str,
    snapshot_prefix: str,
    stage: str,
    v2_adapter_version: str,
) -> bool:
    version = snapshot.get("snapshot_version") if isinstance(snapshot, dict) else None
    v1 = f"{snapshot_prefix}-snapshot-v1"
    v2 = f"{snapshot_prefix}-snapshot-v2"
    if version not in {v1, v2}:
        raise FrontendError(
            "FRONTEND_SCHEMA_VERSION_UNSUPPORTED",
            f"{dsl} snapshot version is not supported",
            {"snapshot_version": version},
        )
    is_v2 = version == v2
    root_fields = {
        "snapshot_version",
        "producer",
        "target",
        "extraction",
        "actors",
        "resources",
        "regions",
        "operations",
        "unsupported_constructs",
    }
    if is_v2:
        root_fields.add("profile")
    require_exact_keys(snapshot, root_fields, set(), "/")
    require_exact_keys(
        snapshot["producer"],
        {"dsl_version", "dsl_commit", "backend_commit", "adapter_version", "command"},
        set(),
        "/producer",
    )
    producer = snapshot["producer"]
    if is_v2 and producer["adapter_version"] != v2_adapter_version:
        raise FrontendError(
            "FRONTEND_ADAPTER_VERSION_UNSUPPORTED",
            f"{dsl} snapshot was not produced by the required v2 adapter",
            {
                "adapter_version": producer["adapter_version"],
                "required": v2_adapter_version,
            },
        )
    require_exact_keys(
        snapshot["target"],
        {"architecture", "warp_size", "cta_threads", "shared_memory_bytes", "grid_scope"},
        set(),
        "/target",
    )
    target = snapshot["target"]
    if (
        target["architecture"] != "sm90"
        or target["warp_size"] != 32
        or target["grid_scope"] != "single_cta"
    ):
        raise FrontendError(
            "FRONTEND_TARGET_UNSUPPORTED",
            f"{dsl} snapshot is outside the static SM90 single-CTA profile",
            {"target": target},
        )
    cta_threads = target["cta_threads"]
    if isinstance(cta_threads, bool) or not isinstance(cta_threads, int) or not 1 <= cta_threads <= 1024:
        raise FrontendError(
            "FRONTEND_TARGET_UNSUPPORTED",
            f"{dsl} snapshot has an invalid static CTA size",
            {"cta_threads": cta_threads},
        )
    extraction_fields = _BASE_EXTRACTION_FIELDS | (_V2_EXTRACTION_FIELDS if is_v2 else set())
    require_exact_keys(snapshot["extraction"], extraction_fields, set(), "/extraction")
    extraction = snapshot["extraction"]
    if not isinstance(extraction["complete"], bool):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            f"{dsl} extraction complete field must be Boolean",
            {"location": "/extraction/complete"},
        )
    if not extraction["complete"]:
        raise FrontendError(
            "FRONTEND_EXTRACTION_INCOMPLETE",
            f"{dsl} extraction is incomplete and cannot produce PGIR",
        )
    if extraction["stage"] != stage:
        raise FrontendError(
            "FRONTEND_STAGE_UNSUPPORTED",
            f"{dsl} snapshot was captured at an unsupported IR stage",
            {"stage": extraction["stage"], "required": stage},
        )
    if snapshot["unsupported_constructs"]:
        raise FrontendError(
            "FRONTEND_UNSUPPORTED_CONSTRUCTS",
            f"{dsl} snapshot contains unsupported constructs",
            {"constructs": snapshot["unsupported_constructs"]},
        )
    operations = snapshot["operations"]
    if not isinstance(operations, list) or not operations:
        raise FrontendError(
            "FRONTEND_OPERATIONS_MISSING",
            f"{dsl} snapshot must contain a nonempty operation list",
        )
    if not is_v2:
        return False

    require_exact_keys(
        snapshot["profile"],
        {"id", "source_ir_sha256", "compiler_ir_stage", "target_rule"},
        set(),
        "/profile",
    )
    profile = snapshot["profile"]
    if profile["id"] != "sm90-single-cta-static-v1":
        raise FrontendError(
            "FRONTEND_PROFILE_UNSUPPORTED",
            f"{dsl} snapshot profile is not supported",
            {"profile": profile["id"]},
        )
    digest = profile["source_ir_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise FrontendError(
            "FRONTEND_PROVENANCE_INVALID",
            f"{dsl} source IR digest is not a lowercase SHA-256 value",
            {"source_ir_sha256": digest},
        )
    if profile["compiler_ir_stage"] != stage:
        raise FrontendError(
            "FRONTEND_STAGE_UNSUPPORTED",
            f"{dsl} profile and extraction stages disagree",
            {
                "profile_stage": profile["compiler_ir_stage"],
                "extraction_stage": extraction["stage"],
            },
        )
    _require_nonempty_string(profile["target_rule"], "/profile/target_rule")
    for field in (
        "allocation_identity_evidence",
        "alias_evidence",
        "participant_evidence",
    ):
        _require_nonempty_string(extraction[field], f"/extraction/{field}")
    accounting = {
        field: _require_nonnegative_integer(extraction[field], f"/extraction/{field}")
        for field in (
            "static_expansions",
            "executable_nodes_seen",
            "classified_nodes",
            "emitting_nodes",
            "elided_nodes",
            "events_emitted",
        )
    }
    if (
        accounting["classified_nodes"] != accounting["executable_nodes_seen"]
        or accounting["emitting_nodes"] + accounting["elided_nodes"]
        != accounting["classified_nodes"]
        or accounting["emitting_nodes"] > accounting["events_emitted"]
        or accounting["events_emitted"] != len(operations)
    ):
        raise FrontendError(
            "FRONTEND_EXTRACTION_ACCOUNTING_INVALID",
            f"{dsl} snapshot does not account for every executable IR node",
            {"accounting": accounting, "operation_count": len(operations)},
        )
    return True


def operation_to_event(
    operation: dict[str, Any],
    index: int,
    *,
    prefix: str,
    parity_derivation_kind: str,
    operation_fields: Mapping[str, set[str]],
    optional_by_operation: Mapping[str, set[str]],
    event_types: Mapping[str, str],
    require_evidence: bool = False,
) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend operation must be an object",
            {"location": f"/operations/{index}"},
        )
    op = operation.get("op")
    if op not in operation_fields:
        raise FrontendError(
            "FRONTEND_UNKNOWN_OPERATION",
            f"{prefix} operation has no sound PGIR mapping",
            {"location": f"/operations/{index}", "operation": op},
        )
    required = COMMON_OPERATION_FIELDS | operation_fields[op]
    if require_evidence:
        required = required | {"evidence"}
    optional = OPTIONAL_OPERATION_FIELDS | optional_by_operation.get(op, set())
    require_exact_keys(operation, required, optional, f"/operations/{index}")
    evidence = operation.get("evidence")
    if evidence is not None:
        require_exact_keys(evidence, _EVIDENCE_FIELDS, set(), f"/operations/{index}/evidence")
        for field in ("source_location", "recognizer_rule", "ir_operation"):
            _require_nonempty_string(evidence[field], f"/operations/{index}/evidence/{field}")
        activation_path = evidence["activation_path"]
        if not isinstance(activation_path, list) or not activation_path:
            raise FrontendError(
                "FRONTEND_PROVENANCE_INVALID",
                "frontend event activation path must be a nonempty list",
                {"location": f"/operations/{index}/evidence/activation_path"},
            )
        if any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in activation_path):
            raise FrontendError(
                "FRONTEND_PROVENANCE_INVALID",
                "frontend event activation path contains an unsupported component",
                {"location": f"/operations/{index}/evidence/activation_path"},
            )
    event = {
        "id": operation["id"],
        "type": event_types[op],
        "actor": operation["actor"],
        "seq": operation["seq"],
    }
    if "depends_on" in operation:
        event["depends_on"] = operation["depends_on"]
    excluded = COMMON_OPERATION_FIELDS | {
        "op",
        "logical_stream_id",
        "stream_rank",
        "polling_coalesced",
        "depends_on",
        "evidence",
    }
    for field, value in operation.items():
        if field not in excluded:
            event[field] = value
    if op == f"{prefix}.mbarrier.wait_parity":
        event["parity_evidence"] = {
            "derivation_kind": parity_derivation_kind,
            "logical_stream_id": operation["logical_stream_id"],
            "lifecycle_epoch": operation["lifecycle_epoch"],
            "stream_rank": operation["stream_rank"],
            "polling_coalesced": operation["polling_coalesced"],
        }
    return event
