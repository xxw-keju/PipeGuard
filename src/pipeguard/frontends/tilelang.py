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
    require_exact_keys,
    validate_snapshot_envelope,
)

SNAPSHOT_VERSION = "tilelang-snapshot-v3"
LEGACY_SNAPSHOT_VERSION = "tilelang-snapshot-v1"
V2_SNAPSHOT_VERSION = "tilelang-snapshot-v2"
ADAPTER_VERSION = "tilelang-pgir-v3"
V2_ADAPTER_VERSION = "tilelang-pgir-v2"

_OPERATION_FIELDS = prefixed_operation_fields("tilelang")
_OPTIONAL_BY_OPERATION = prefixed_optional_fields("tilelang")
_EVENT_TYPES = prefixed_event_types("tilelang")
_TEMPLATE_REQUIRED = {
    "id",
    "op",
    "order",
    "execution_scope",
    "instances",
    "fields",
    "evidence",
}
_TEMPLATE_OPTIONAL = {"owner", "instance_fields", "dependencies"}
_STREAM_ID = "program:tilelang:lanes"


def _validate_snapshot(snapshot: dict[str, Any]) -> bool:
    return validate_snapshot_envelope(
        snapshot,
        dsl="TileLang",
        snapshot_prefix="tilelang",
        stage="post-inject-fence-proxy",
        v2_adapter_version=V2_ADAPTER_VERSION,
        v3_adapter_version=ADAPTER_VERSION,
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
        operation_fields=_OPERATION_FIELDS,
        optional_by_operation=_OPTIONAL_BY_OPERATION,
        event_types=_EVENT_TYPES,
        require_evidence=require_evidence,
    )


def _representative_operation(template: dict[str, Any], index: int) -> dict[str, Any]:
    require_exact_keys(template, _TEMPLATE_REQUIRED, _TEMPLATE_OPTIONAL, f"/operations/{index}")
    fields = template["fields"]
    instance_fields = template.get("instance_fields", {})
    instances = template["instances"]
    if not isinstance(fields, dict) or not isinstance(instance_fields, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang compact operation fields must be objects",
            {"location": f"/operations/{index}"},
        )
    if not isinstance(instances, list) or not instances:
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang compact operation requires a nonempty instance set",
            {"location": f"/operations/{index}/instances"},
        )
    if any(
        not isinstance(values, list) or len(values) != len(instances)
        for values in instance_fields.values()
    ):
        raise FrontendError(
            "FRONTEND_INSTANCE_FIELDS_INVALID",
            "TileLang instance fields must align one-to-one with active lanes",
            {"location": f"/operations/{index}/instance_fields"},
        )
    overlap = sorted(set(fields).intersection(instance_fields))
    if overlap:
        raise FrontendError(
            "FRONTEND_INSTANCE_FIELDS_INVALID",
            "TileLang fields cannot be both common and instance-indexed",
            {"location": f"/operations/{index}", "fields": overlap},
        )
    return {
        "id": template["id"],
        "op": template["op"],
        "actor": template.get("owner", "tilelang:parametric"),
        "seq": template["order"],
        "evidence": template["evidence"],
        **deepcopy(fields),
        **{
            name: deepcopy(values[0])
            for name, values in instance_fields.items()
        },
    }


def _canonical_template(template: dict[str, Any], index: int) -> dict[str, Any]:
    representative = _representative_operation(template, index)
    event = _event(representative, index, require_evidence=True)
    fields = deepcopy(template["fields"])
    instance_fields = deepcopy(template.get("instance_fields", {}))

    if template["op"] == "tilelang.mbarrier.wait_parity":
        parity_names = (
            "logical_stream_id",
            "stream_rank",
            "polling_coalesced",
        )
        parity_is_indexed = (
            "lifecycle_epoch" in instance_fields
            or any(name in instance_fields for name in parity_names)
        )
        if parity_is_indexed:
            values = []
            for position, _instance in enumerate(template["instances"]):
                values.append({
                    "derivation_kind": "PARITY_EXPRESSION",
                    "logical_stream_id": (
                        instance_fields["logical_stream_id"][position]
                        if "logical_stream_id" in instance_fields
                        else fields["logical_stream_id"]
                    ),
                    "lifecycle_epoch": (
                        instance_fields["lifecycle_epoch"][position]
                        if "lifecycle_epoch" in instance_fields
                        else fields["lifecycle_epoch"]
                    ),
                    "stream_rank": (
                        instance_fields["stream_rank"][position]
                        if "stream_rank" in instance_fields
                        else fields["stream_rank"]
                    ),
                    "polling_coalesced": (
                        instance_fields["polling_coalesced"][position]
                        if "polling_coalesced" in instance_fields
                        else fields["polling_coalesced"]
                    ),
                })
            instance_fields["parity_evidence"] = values
        else:
            fields["parity_evidence"] = deepcopy(event["parity_evidence"])
        for name in parity_names:
            fields.pop(name, None)
            instance_fields.pop(name, None)

    scope = template["execution_scope"]
    instances = list(template["instances"])
    result = {
        "id": template["id"],
        "type": event["type"],
        "stream": _STREAM_ID,
        "order": template["order"],
        "program_instances": instances,
        "executors": instances,
        "transition": "PER_INSTANCE" if scope == "INSTANCE" else "ONCE",
        "rendezvous": (
            "ALL_PROGRAM_INSTANCES" if scope == "COLLECTIVE" else "NONE"
        ),
        "fields": fields,
    }
    if "owner" in template:
        result["owner"] = template["owner"]
    if instance_fields:
        result["instance_fields"] = instance_fields
    if template.get("dependencies"):
        result["dependencies"] = deepcopy(template["dependencies"])
    return result


def _parameterized_resources(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    actors = {actor["id"]: actor for actor in snapshot["actors"]}
    resources = []
    for raw in snapshot["resources"]:
        resource = deepcopy(raw)
        stream_instances: set[int] = set()
        owners: list[str] = []
        for participant in resource["participants"]:
            if not isinstance(participant, str) or participant not in actors:
                raise FrontendError(
                    "FRONTEND_PARTICIPANT_MAPPING_UNRESOLVED",
                    "TileLang resource references an unknown execution owner",
                    {"resource": resource.get("id"), "participant": participant},
                )
            actor = actors[participant]
            if actor["kind"] == "LANE":
                stream_instances.update(actor["participants"])
            else:
                owners.append(participant)
                if (
                    actor["kind"] != "ASYNC_ENGINE"
                    and resource.get("protocol") != "WGMMA"
                ):
                    stream_instances.update(actor["participants"])
        participants: list[Any] = list(dict.fromkeys(owners))
        if stream_instances:
            participants.append({
                "stream": _STREAM_ID,
                "instances": sorted(stream_instances),
            })
        resource["participants"] = participants
        resources.append(resource)
    return resources


def _parameterize_actor_fields(
    templates: list[dict[str, Any]],
    actors: list[dict[str, Any]],
) -> None:
    lane_actors = {
        actor["id"]: actor["participants"][0]
        for actor in actors
        if actor["kind"] == "LANE" and len(actor["participants"]) == 1
    }
    for template in templates:
        for bindings_name in ("fields", "instance_fields"):
            bindings = template.get(bindings_name, {})
            if "group_owner" not in bindings:
                continue
            value = bindings["group_owner"]
            if bindings_name == "fields":
                if value in lane_actors:
                    bindings["group_owner"] = "$self"
            else:
                bindings["group_owner"] = [
                    "$self" if item in lane_actors else item
                    for item in value
                ]


def _parametric_module(snapshot: dict[str, Any]) -> Module:
    producer = snapshot["producer"]
    extraction = snapshot["extraction"]
    digest = snapshot["profile"]["source_ir_sha256"]
    templates = [
        _canonical_template(template, index)
        for index, template in enumerate(snapshot["operations"])
    ]
    _parameterize_actor_fields(templates, snapshot["actors"])
    logical_events = sum(len(template["program_instances"]) for template in templates)
    if logical_events != extraction["logical_events_emitted"]:
        raise FrontendError(
            "FRONTEND_EXTRACTION_ACCOUNTING_INVALID",
            "TileLang compact operations disagree with the logical event count",
            {
                "logical_events_emitted": extraction["logical_events_emitted"],
                "template_instances": logical_events,
            },
        )
    return module_from_dict({
        "schema_version": "pgir-parametric-v1",
        "producer": {"dsl": "TILELANG", **producer},
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
        "owners": snapshot["actors"],
        "program_streams": [{
            "id": _STREAM_ID,
            "kind": "LANE",
            "instances": list(range(snapshot["target"]["cta_threads"])),
        }],
        "resources": _parameterized_resources(snapshot),
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
                "method": extraction["lane_ownership_evidence"],
                "source": digest,
            },
            "predicate_mapping": {
                "status": "RESOLVED",
                "method": extraction["predicate_evidence"],
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


def _explicit_module(snapshot: dict[str, Any], require_evidence: bool) -> Module:
    producer = snapshot["producer"]
    extraction = snapshot["extraction"]
    return module_from_dict({
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
    })


def tilelang_snapshot_to_module(snapshot: dict[str, Any]) -> Module:
    if not isinstance(snapshot, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "TileLang snapshot root must be an object",
        )
    require_evidence = _validate_snapshot(snapshot)
    if snapshot["snapshot_version"] == SNAPSHOT_VERSION:
        return _parametric_module(snapshot)
    return _explicit_module(snapshot, require_evidence)


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
