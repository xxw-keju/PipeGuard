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

SNAPSHOT_VERSION = "tilelang-snapshot-v2"
LEGACY_SNAPSHOT_VERSION = "tilelang-snapshot-v1"
ADAPTER_VERSION = "tilelang-pgir-v2"

_OPERATION_FIELDS = prefixed_operation_fields("tilelang")
_OPTIONAL_BY_OPERATION = prefixed_optional_fields("tilelang")
_EVENT_TYPES = prefixed_event_types("tilelang")


def _validate_snapshot(snapshot: dict[str, Any]) -> bool:
    return validate_snapshot_envelope(
        snapshot,
        dsl="TileLang",
        snapshot_prefix="tilelang",
        stage="post-inject-fence-proxy",
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
        prefix="tilelang",
        parity_derivation_kind="PARITY_EXPRESSION",
        require_evidence=require_evidence,
    )


def tilelang_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang snapshot root must be an object",
        )
    require_evidence = _validate_snapshot(snapshot)
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
            _event(operation, index, require_evidence)
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
