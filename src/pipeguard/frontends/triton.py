from __future__ import annotations

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

_OPERATION_FIELDS = prefixed_operation_fields("triton")
_OPTIONAL_BY_OPERATION = prefixed_optional_fields("triton")
_EVENT_TYPES = prefixed_event_types("triton")


def _validate_snapshot(snapshot: dict[str, Any]) -> bool:
    return validate_snapshot_envelope(
        snapshot,
        dsl="Triton",
        snapshot_prefix="triton",
        stage="post-ttgir-pipeline",
        v2_adapter_version=ADAPTER_VERSION,
    )


def _event(
    operation: dict[str, Any],
    index: int,
    require_evidence: bool = False,
) -> dict[str, Any]:
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


def triton_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError("FRONTEND_FIELD_TYPE", "Triton snapshot root must be an object")
    require_evidence = _validate_snapshot(snapshot)
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
        "events": [
            _event(operation, index, require_evidence)
            for index, operation in enumerate(snapshot["operations"])
        ],
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
