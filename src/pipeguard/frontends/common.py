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
        "resource", "bytes", "lifecycle_epoch", "generation",
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
    "proxy_fence": {"from_domain", "to_domain", "scope", "regions", "visibility_mask"},
}
CANONICAL_OPTIONAL_FIELDS = {
    "barrier.arrive": {"predicate"},
    "mbarrier.arrive": {"predicate"},
    "mbarrier.arrive_expect_tx": {"predicate"},
    "tma_load.issue": {"source_region", "destination_region", "transfers", "engine_id", "engine_seq"},
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
_V3_EXTRACTION_FIELDS = {
    "static_expansions",
    "executable_nodes_seen",
    "classified_nodes",
    "emitting_nodes",
    "elided_nodes",
    "operation_records",
    "logical_events_emitted",
    "allocation_identity_evidence",
    "alias_evidence",
    "participant_evidence",
    "lane_ownership_evidence",
    "predicate_evidence",
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
    v2_adapter_version: str | frozenset[str],
    v3_adapter_version: str | frozenset[str] | None = None,
    profile_id: str | frozenset[str] = "sm90-single-cta-static-v1",
) -> bool:
    version = snapshot.get("snapshot_version") if isinstance(snapshot, dict) else None
    v1 = f"{snapshot_prefix}-snapshot-v1"
    v2 = f"{snapshot_prefix}-snapshot-v2"
    v3 = f"{snapshot_prefix}-snapshot-v3"
    accepted_versions = {v1, v2} | ({v3} if v3_adapter_version is not None else set())
    if version not in accepted_versions:
        raise FrontendError(
            "FRONTEND_SCHEMA_VERSION_UNSUPPORTED",
            f"{dsl} snapshot version is not supported",
            {"snapshot_version": version},
        )
    is_v2 = version == v2
    is_v3 = version == v3
    is_evidenced = is_v2 or is_v3
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
    if is_evidenced:
        root_fields.add("profile")
    require_exact_keys(snapshot, root_fields, set(), "/")
    require_exact_keys(
        snapshot["producer"],
        {"dsl_version", "dsl_commit", "backend_commit", "adapter_version", "command"},
        set(),
        "/producer",
    )
    producer = snapshot["producer"]
    adapter_spec = v3_adapter_version if is_v3 else v2_adapter_version
    accepted_adapters = (
        frozenset({adapter_spec})
        if isinstance(adapter_spec, str)
        else adapter_spec
    )
    if is_evidenced and producer["adapter_version"] not in accepted_adapters:
        raise FrontendError(
            "FRONTEND_ADAPTER_VERSION_UNSUPPORTED",
            f"{dsl} snapshot was not produced by a supported adapter",
            {
                "adapter_version": producer["adapter_version"],
                "required": sorted(accepted_adapters),
            },
        )
    require_exact_keys(
        snapshot["target"],
        {"architecture", "warp_size", "cta_threads", "shared_memory_bytes", "grid_scope"},
        {"launch_grid", "selected_cta"} if is_evidenced else set(),
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
    has_launch_grid = "launch_grid" in target
    has_selected_cta = "selected_cta" in target
    if has_launch_grid != has_selected_cta:
        raise FrontendError(
            "FRONTEND_TARGET_UNSUPPORTED",
            f"{dsl} selected-CTA evidence must include both launch grid and CTA coordinate",
            {"target": target},
        )
    if has_launch_grid:
        launch_grid = target["launch_grid"]
        selected_cta = target["selected_cta"]
        if (
            not isinstance(launch_grid, list)
            or len(launch_grid) != 3
            or any(isinstance(extent, bool) or not isinstance(extent, int) or extent < 1 for extent in launch_grid)
            or not isinstance(selected_cta, list)
            or len(selected_cta) != 3
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected_cta)
            or any(index >= extent for index, extent in zip(selected_cta, launch_grid))
        ):
            raise FrontendError(
                "FRONTEND_TARGET_UNSUPPORTED",
                f"{dsl} snapshot has invalid selected-CTA launch evidence",
                {"launch_grid": launch_grid, "selected_cta": selected_cta},
            )
    versioned_extraction_fields = (
        _V3_EXTRACTION_FIELDS if is_v3 else _V2_EXTRACTION_FIELDS if is_v2 else set()
    )
    extraction_fields = _BASE_EXTRACTION_FIELDS | versioned_extraction_fields
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
    if not isinstance(operations, list):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            f"{dsl} snapshot operations must be a list",
            {"location": "/operations"},
        )
    if not is_evidenced:
        return False

    require_exact_keys(
        snapshot["profile"],
        {"id", "source_ir_sha256", "compiler_ir_stage", "target_rule"},
        {"final_ptx_evidence"},
        "/profile",
    )
    profile = snapshot["profile"]
    accepted_profiles = (
        frozenset({profile_id})
        if isinstance(profile_id, str)
        else profile_id
    )
    if profile["id"] not in accepted_profiles:
        raise FrontendError(
            "FRONTEND_PROFILE_UNSUPPORTED",
            f"{dsl} snapshot profile is not supported",
            {"profile": profile["id"], "required": sorted(accepted_profiles)},
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
    final_ptx = profile.get("final_ptx_evidence")
    if final_ptx is not None:
        require_exact_keys(
            final_ptx,
            {
                "sha256", "bytes", "proxy_fence_before_first_wgmma",
                "proxy_fence_count", "wgmma_instruction_count", "method",
            },
            set(),
            "/profile/final_ptx_evidence",
        )
        final_digest = final_ptx["sha256"]
        if (
            not isinstance(final_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", final_digest) is None
            or _require_nonnegative_integer(
                final_ptx["bytes"], "/profile/final_ptx_evidence/bytes"
            ) == 0
            or not isinstance(final_ptx["proxy_fence_before_first_wgmma"], bool)
            or _require_nonnegative_integer(
                final_ptx["proxy_fence_count"],
                "/profile/final_ptx_evidence/proxy_fence_count",
            ) not in {0, 1}
            or _require_nonnegative_integer(
                final_ptx["wgmma_instruction_count"],
                "/profile/final_ptx_evidence/wgmma_instruction_count",
            ) == 0
            or final_ptx["proxy_fence_before_first_wgmma"]
            != bool(final_ptx["proxy_fence_count"])
            or final_ptx["method"] != "final-ptx-exact-opcode-order-v1"
        ):
            raise FrontendError(
                "FRONTEND_PROVENANCE_INVALID",
                f"{dsl} final PTX evidence is malformed",
                {"final_ptx_evidence": final_ptx},
            )
    evidence_fields = [
        "allocation_identity_evidence",
        "alias_evidence",
        "participant_evidence",
    ]
    if is_v3:
        evidence_fields.extend(["lane_ownership_evidence", "predicate_evidence"])
    for field in evidence_fields:
        _require_nonempty_string(extraction[field], f"/extraction/{field}")
    count_fields = [
        "static_expansions",
        "executable_nodes_seen",
        "classified_nodes",
        "emitting_nodes",
        "elided_nodes",
        *(
            ["operation_records", "logical_events_emitted"]
            if is_v3
            else ["events_emitted"]
        ),
    ]
    accounting = {
        field: _require_nonnegative_integer(extraction[field], f"/extraction/{field}")
        for field in count_fields
    }
    structurally_invalid = (
        accounting["classified_nodes"] != accounting["executable_nodes_seen"]
        or accounting["emitting_nodes"] + accounting["elided_nodes"]
        != accounting["classified_nodes"]
    )
    if is_v3:
        count_invalid = (
            accounting["operation_records"] != len(operations)
            or accounting["emitting_nodes"] > accounting["logical_events_emitted"]
            or accounting["operation_records"] > accounting["logical_events_emitted"]
        )
    else:
        count_invalid = (
            accounting["emitting_nodes"] > accounting["events_emitted"]
            or accounting["events_emitted"] != len(operations)
        )
    if structurally_invalid or count_invalid:
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
    if op == f"{prefix}.tma_load.issue":
        has_legacy_transfer = "source_region" in operation or "destination_region" in operation
        has_grouped_transfers = "transfers" in operation
        if has_legacy_transfer == has_grouped_transfers:
            raise FrontendError(
                "FRONTEND_TMA_TRANSFER_REPRESENTATION_INVALID",
                "TMA load must use exactly one legacy region pair or one nonempty transfers list",
                {"location": f"/operations/{index}"},
            )
        if has_legacy_transfer and (
            "source_region" not in operation or "destination_region" not in operation
        ):
            raise FrontendError(
                "FRONTEND_TMA_TRANSFER_REPRESENTATION_INVALID",
                "legacy TMA load representation requires both source and destination regions",
                {"location": f"/operations/{index}"},
            )
        if has_grouped_transfers:
            transfers = operation["transfers"]
            if not isinstance(transfers, list) or not transfers:
                raise FrontendError(
                    "FRONTEND_TMA_TRANSFER_REPRESENTATION_INVALID",
                    "TMA load transfers must be a nonempty list",
                    {"location": f"/operations/{index}/transfers"},
                )
            transfer_bytes = 0
            for transfer_index, transfer in enumerate(transfers):
                location = f"/operations/{index}/transfers/{transfer_index}"
                require_exact_keys(
                    transfer,
                    {"source_region", "destination_region", "bytes"},
                    set(),
                    location,
                )
                _require_nonempty_string(transfer["source_region"], f"{location}/source_region")
                _require_nonempty_string(transfer["destination_region"], f"{location}/destination_region")
                byte_count = _require_nonnegative_integer(transfer["bytes"], f"{location}/bytes")
                if byte_count == 0:
                    raise FrontendError(
                        "FRONTEND_FIELD_TYPE",
                        "TMA transfer byte count must be positive",
                        {"location": f"{location}/bytes"},
                    )
                transfer_bytes += byte_count
            if transfer_bytes != operation["bytes"]:
                raise FrontendError(
                    "FRONTEND_TMA_TRANSFER_BYTES_MISMATCH",
                    "TMA load bytes must equal the sum of its transfer records",
                    {
                        "location": f"/operations/{index}",
                        "bytes": operation["bytes"],
                        "transfer_bytes": transfer_bytes,
                    },
                )
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
