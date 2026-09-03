from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

from .protocols import RESOURCE_KINDS, VisibilityMask

IMPLEMENTED_RESOURCE_KINDS = frozenset({"mbarrier", "cta_barrier", "named_barrier", "async_group"})
IMPLEMENTED_ASYNC_PROTOCOLS = frozenset({"CP_ASYNC"})

IMPLEMENTED_EVENTS = frozenset({
    "MemRead", "MemWrite", "BarrierArrive", "BarrierObserve",
    "MBarrierInit", "MBarrierInvalidate", "MBarrierArrive", "MBarrierExpectTx",
    "MBarrierArriveExpectTx", "MBarrierWaitParity", "TmaLoadIssue",
    "CpAsyncIssue", "CpAsyncCommitGroup", "CpAsyncWaitGroup",
})
KNOWN_UNSUPPORTED_EVENTS = frozenset({
    "AtomicReadModifyWrite", "Fork", "Join", "BarrierRelease",
    "MBarrierTxComplete", "MBarrierPhaseComplete",
    "CpAsyncGroupComplete",
    "TmaLoadComplete", "TmaStoreIssue", "TmaStoreSourceReleased",
    "TmaStoreGlobalComplete", "TmaStoreCommitGroup", "TmaStoreWaitGroup",
    "WgmmaFenceOperand", "WgmmaIssue", "WgmmaCommitGroup", "WgmmaResultReady",
    "WgmmaOperandsReleased", "WgmmaWaitGroup", "ProxyFence", "ChannelCreate",
    "ProducerAcquire", "ProducerCommit", "ConsumerWait", "ConsumerRelease",
})


class PGIRError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def diagnostic(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass(frozen=True, slots=True)
class Actor:
    id: str
    kind: str
    participants: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Resource:
    id: str
    kind: str
    participants: tuple[str, ...]
    capacity: int | None = None
    protocol: str | None = None
    scope: str | None = None
    visibility_mask: VisibilityMask | None = None


@dataclass(frozen=True, slots=True)
class Region:
    id: str
    allocation_id: str
    address_space: str
    offset: int
    extent: int


@dataclass(frozen=True, slots=True)
class ParityEvidence:
    derivation_kind: str
    logical_stream_id: str
    lifecycle_epoch: int
    stream_rank: int
    polling_coalesced: bool


@dataclass(frozen=True, slots=True)
class Event:
    id: str
    type: str
    actor: str
    seq: int
    depends_on: tuple[str, ...] = ()
    resource: str | None = None
    region: str | None = None
    source_region: str | None = None
    destination_region: str | None = None
    count: int | None = None
    bytes: int | None = None
    lifecycle_epoch: int | None = None
    generation: int | None = None
    raw_parity: int | None = None
    parity_evidence: ParityEvidence | None = None
    predicate: bool = True
    invocation_id: int | None = None
    group_owner: str | None = None
    group_seq: int | None = None
    wait_group_count: int | None = None
    completion_kind: str | None = None
    wait_mode: str | None = None
    engine_id: str | None = None
    engine_seq: int | None = None
    slot: int | None = None
    logical_phase: int | None = None
    from_domain: str | None = None
    to_domain: str | None = None
    scope: str | None = None
    regions: tuple[str, ...] = ()
    visibility_mask: VisibilityMask | None = None


@dataclass(frozen=True, slots=True)
class Module:
    raw: dict[str, Any]
    actors: tuple[Actor, ...]
    resources: tuple[Resource, ...]
    regions: tuple[Region, ...]
    events: tuple[Event, ...]
    events_by_actor: dict[str, tuple[Event, ...]] = field(compare=False, hash=False)

    @property
    def event_map(self) -> dict[str, Event]:
        return {event.id: event for event in self.events}

    @property
    def resource_map(self) -> dict[str, Resource]:
        return {resource.id: resource for resource in self.resources}

    @property
    def region_map(self) -> dict[str, Region]:
        return {region.id: region for region in self.regions}

    @property
    def actor_map(self) -> dict[str, Actor]:
        return {actor.id: actor for actor in self.actors}

    @property
    def required_logical_waits(self) -> dict[tuple[str, int, int], frozenset[str]]:
        required: dict[tuple[str, int, int], set[str]] = {}
        for event in self.events:
            if event.type != "MBarrierWaitParity":
                continue
            assert event.resource is not None and event.lifecycle_epoch is not None and event.generation is not None
            required.setdefault((event.resource, event.lifecycle_epoch, event.generation), set()).add(event.id)
        return {key: frozenset(event_ids) for key, event_ids in required.items()}


def _schema() -> dict[str, Any]:
    return json.loads(files("pipeguard.specs").joinpath("pgir-v1.schema.json").read_text())


def stable_completion_id(issue_id: str, generation: int) -> str:
    return f"completion:tma-load:{issue_id}:g{generation:08d}"


def stable_phase_id(resource_id: str, lifecycle_epoch: int, generation: int) -> str:
    return f"completion:mbarrier-phase:{resource_id}:l{lifecycle_epoch:08d}:g{generation:08d}"


def _stable_component(value: str) -> str:
    return f"{len(value.encode('utf-8')):08d}:{value}"


def stable_collective_release_id(resource_id: str, invocation_id: int) -> str:
    return f"completion:barrier-release:{_stable_component(resource_id)}:i{invocation_id:08d}"


def stable_group_id(protocol: str, owner: str, group_seq: int) -> str:
    return f"group:{protocol.lower()}:{_stable_component(owner)}:g{group_seq:08d}"


def stable_group_completion_id(protocol: str, owner: str, group_seq: int, completion_kind: str) -> str:
    return (
        f"completion:group:{protocol.lower()}:{_stable_component(owner)}:"
        f"g{group_seq:08d}:{completion_kind.lower()}"
    )


def stable_channel_handoff_id(resource_id: str, slot: int, logical_phase: int) -> str:
    return (
        f"completion:channel-handoff:{_stable_component(resource_id)}:"
        f"s{slot:08d}:p{logical_phase:08d}"
    )


def stable_digest(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _strict_integer(value: Any, path: str) -> None:
    if type(value) is not int:
        raise PGIRError("STRICT_INTEGER_REQUIRED", "JSON integer must decode as a Python int", {"path": path, "value": value})


def _strict_integers(data: Any) -> None:
    if not isinstance(data, dict):
        return
    target = data.get("target")
    if isinstance(target, dict):
        for name in ("warp_size", "cta_threads", "shared_memory_bytes"):
            if name in target:
                _strict_integer(target[name], f"/target/{name}")
    for index, actor in enumerate(data.get("actors", [])):
        for lane_index, lane in enumerate(actor.get("participants", [])):
            _strict_integer(lane, f"/actors/{index}/participants/{lane_index}")
    for index, resource in enumerate(data.get("resources", [])):
        if "capacity" in resource:
            _strict_integer(resource["capacity"], f"/resources/{index}/capacity")
    for index, region in enumerate(data.get("regions", [])):
        for name in ("offset", "extent"):
            if name in region:
                _strict_integer(region[name], f"/regions/{index}/{name}")
    for index, event in enumerate(data.get("events", [])):
        for name in (
            "seq", "count", "bytes", "lifecycle_epoch", "generation", "raw_parity",
            "invocation_id", "group_seq", "wait_group_count", "engine_seq", "slot",
            "logical_phase",
        ):
            if name in event:
                _strict_integer(event[name], f"/events/{index}/{name}")
        evidence = event.get("parity_evidence")
        if isinstance(evidence, dict):
            for name in ("lifecycle_epoch", "stream_rank"):
                if name in evidence:
                    _strict_integer(evidence[name], f"/events/{index}/parity_evidence/{name}")


def _schema_validate(data: Any) -> None:
    _strict_integers(data)
    validator = jsonschema.Draft202012Validator(_schema())
    errors = sorted(validator.iter_errors(data), key=lambda error: (list(error.absolute_path), error.message))
    if errors:
        error = errors[0]
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise PGIRError("SCHEMA_INVALID", error.message, {"path": path or "/"})


def _unique(kind: str, values: list[str]) -> None:
    seen: set[str] = set()
    duplicate = next((value for value in values if value in seen or seen.add(value)), None)
    if duplicate is not None:
        raise PGIRError("DUPLICATE_ID", f"duplicate {kind} id: {duplicate}", {"id": duplicate, "kind": kind})


def _require(event: Event, *fields: str) -> None:
    missing = [name for name in fields if getattr(event, name) is None]
    if missing:
        raise PGIRError("EVENT_FIELDS_MISSING", f"{event.type} requires {', '.join(missing)}", {"event": event.id, "fields": missing})


def _semantic_validate(module: Module) -> None:
    actors = module.actor_map
    resources = module.resource_map
    regions = module.region_map
    events = module.event_map
    _unique("actor", [actor.id for actor in module.actors])
    _unique("resource", [resource.id for resource in module.resources])
    _unique("region", [region.id for region in module.regions])
    _unique("event", [event.id for event in module.events])

    assumptions = module.raw["assumptions"]
    unresolved = sorted(
        key for key, expected in {
            "finite_control_flow": True, "statically_known_participants": True,
            "parity_normalization_status": "RESOLVED", "alias_precision": "EXACT",
            "fair_async_completion": True, "no_cluster_effects": True,
        }.items() if assumptions.get(key) != expected
    )
    if unresolved:
        code = "INCONCLUSIVE_PARITY" if "parity_normalization_status" in unresolved else "UNRESOLVED_ASSUMPTIONS"
        raise PGIRError(code, "SAFE preconditions are unresolved", {"assumptions": unresolved})
    if module.raw["unsupported_constructs"]:
        raise PGIRError("UNSUPPORTED_CONSTRUCT", "frontend reported unsupported constructs", {"constructs": sorted(module.raw["unsupported_constructs"])})

    cta_threads = module.raw["target"]["cta_threads"]
    for actor in module.actors:
        invalid_lanes = sorted(lane for lane in actor.participants if lane >= cta_threads)
        if invalid_lanes:
            raise PGIRError("PARTICIPANT_OUT_OF_RANGE", "actor participant exceeds CTA bounds", {"actor": actor.id, "lanes": invalid_lanes, "cta_threads": cta_threads})
    for resource in module.resources:
        dangling = sorted(set(resource.participants) - actors.keys())
        if dangling:
            raise PGIRError("UNRESOLVED_PARTICIPANTS", "resource has unknown participants", {"resource": resource.id, "actors": dangling})
        if resource.kind not in RESOURCE_KINDS:
            raise PGIRError("UNKNOWN_RESOURCE_KIND", "resource kind is unknown", {"resource": resource.id, "kind": resource.kind})
        if resource.kind not in IMPLEMENTED_RESOURCE_KINDS:
            raise PGIRError("KNOWN_RESOURCE_UNSUPPORTED", "resource kind is not implemented yet", {"resource": resource.id, "kind": resource.kind})
        if resource.kind == "mbarrier":
            unexpected_resource_fields = [
                name for name in ("capacity", "protocol", "scope", "visibility_mask")
                if getattr(resource, name) is not None
            ]
            if unexpected_resource_fields:
                raise PGIRError(
                    "RESOURCE_FIELDS_UNEXPECTED",
                    "mbarrier resource carries fields for another protocol",
                    {"resource": resource.id, "fields": unexpected_resource_fields},
                )
        elif resource.kind in {"cta_barrier", "named_barrier"}:
            missing = [
                name for name in ("capacity", "scope", "visibility_mask")
                if getattr(resource, name) is None
            ]
            if missing:
                raise PGIRError(
                    "RESOURCE_FIELDS_MISSING",
                    "collective barrier resource lacks required metadata",
                    {"resource": resource.id, "fields": missing},
                )
            if resource.protocol is not None:
                raise PGIRError(
                    "RESOURCE_FIELDS_UNEXPECTED",
                    "collective barrier cannot carry an async-group protocol",
                    {"resource": resource.id, "fields": ["protocol"]},
                )
            participant_lanes: list[int] = []
            for actor_id in resource.participants:
                participant_lanes.extend(actors[actor_id].participants)
            if len(participant_lanes) != len(set(participant_lanes)):
                raise PGIRError(
                    "OVERLAPPING_COLLECTIVE_PARTICIPANTS",
                    "collective barrier actors must provide disjoint lane sets",
                    {"resource": resource.id},
                )
            if resource.capacity != len(participant_lanes):
                raise PGIRError(
                    "COLLECTIVE_CAPACITY_MISMATCH",
                    "collective barrier capacity must equal its complete participant lane set",
                    {"resource": resource.id, "capacity": resource.capacity, "participant_lanes": sorted(participant_lanes)},
                )
            if resource.kind == "cta_barrier" and set(participant_lanes) != set(range(cta_threads)):
                raise PGIRError(
                    "CTA_PARTICIPANT_SET_INCOMPLETE",
                    "CTA barrier must cover every lane in the CTA exactly once",
                    {"resource": resource.id, "participant_lanes": sorted(participant_lanes), "cta_threads": cta_threads},
                )
            assert resource.visibility_mask is not None
            mask = resource.visibility_mask
            data_visibility = any((
                mask.shared,
                mask.global_read,
                mask.global_write,
                mask.tensor_read,
                mask.tensor_write,
            ))
            if mask.control_only == data_visibility:
                raise PGIRError(
                    "INVALID_VISIBILITY_MASK",
                    "collective barrier must select either control-only or at least one data visibility domain",
                    {"resource": resource.id, "visibility_mask": mask.artifact()},
                )
        else:
            missing = [
                name for name in ("capacity", "protocol", "scope")
                if getattr(resource, name) is None
            ]
            if missing:
                raise PGIRError(
                    "RESOURCE_FIELDS_MISSING",
                    "async-group resource lacks required metadata",
                    {"resource": resource.id, "fields": missing},
                )
            if resource.visibility_mask is not None:
                raise PGIRError(
                    "RESOURCE_FIELDS_UNEXPECTED",
                    "async-group resource cannot carry collective visibility metadata",
                    {"resource": resource.id, "fields": ["visibility_mask"]},
                )
            if resource.protocol not in IMPLEMENTED_ASYNC_PROTOCOLS:
                raise PGIRError(
                    "KNOWN_RESOURCE_UNSUPPORTED",
                    "async-group protocol is not implemented yet",
                    {"resource": resource.id, "protocol": resource.protocol},
                )
            if resource.protocol == "CP_ASYNC" and resource.capacity > 8:
                raise PGIRError(
                    "CP_ASYNC_CAPACITY_UNSUPPORTED",
                    "cp.async supports at most eight uncompleted committed groups",
                    {"resource": resource.id, "capacity": resource.capacity},
                )

    cp_async_owners: dict[str, list[str]] = {}
    for resource in module.resources:
        if resource.kind == "async_group" and resource.protocol == "CP_ASYNC":
            for owner in resource.participants:
                cp_async_owners.setdefault(owner, []).append(resource.id)
    ambiguous_owners = {
        owner: sorted(resource_ids)
        for owner, resource_ids in cp_async_owners.items()
        if len(resource_ids) > 1
    }
    if ambiguous_owners:
        raise PGIRError(
            "AMBIGUOUS_GROUP_OWNER_RESOURCE",
            "one cp.async owner cannot span multiple async-group resources in PGIR v1",
            {"owners": ambiguous_owners},
        )

    required_fields = {
        "MemRead": ("region",), "MemWrite": ("region",),
        "BarrierArrive": ("resource", "count", "invocation_id"),
        "BarrierObserve": ("resource", "invocation_id"),
        "CpAsyncIssue": ("resource", "source_region", "destination_region", "bytes", "group_owner"),
        "CpAsyncCommitGroup": ("resource", "group_owner", "group_seq"),
        "CpAsyncWaitGroup": ("resource", "group_owner", "wait_group_count"),
        "MBarrierInit": ("resource", "count", "lifecycle_epoch"), "MBarrierInvalidate": ("resource", "lifecycle_epoch"),
        "MBarrierArrive": ("resource", "count", "lifecycle_epoch", "generation"),
        "MBarrierExpectTx": ("resource", "bytes", "lifecycle_epoch", "generation"),
        "MBarrierArriveExpectTx": ("resource", "count", "bytes", "lifecycle_epoch", "generation"),
        "MBarrierWaitParity": ("resource", "lifecycle_epoch", "generation", "raw_parity", "parity_evidence"),
        "TmaLoadIssue": ("resource", "source_region", "destination_region", "bytes", "lifecycle_epoch", "generation"),
    }
    common = {"id", "type", "actor", "seq", "depends_on"}
    specific = {
        "MemRead": {"region"}, "MemWrite": {"region"},
        "BarrierArrive": {"resource", "count", "invocation_id", "predicate"},
        "BarrierObserve": {"resource", "invocation_id"},
        "CpAsyncIssue": {
            "resource", "source_region", "destination_region", "bytes", "group_owner",
            "engine_id", "engine_seq", "predicate",
        },
        "CpAsyncCommitGroup": {"resource", "group_owner", "group_seq"},
        "CpAsyncWaitGroup": {"resource", "group_owner", "wait_group_count"},
        "MBarrierInit": {"resource", "count", "lifecycle_epoch"}, "MBarrierInvalidate": {"resource", "lifecycle_epoch"},
        "MBarrierArrive": {"lifecycle_epoch", "resource", "count", "generation", "predicate"},
        "MBarrierExpectTx": {"lifecycle_epoch", "resource", "bytes", "generation"},
        "MBarrierArriveExpectTx": {"lifecycle_epoch", "resource", "count", "bytes", "generation", "predicate"},
        "MBarrierWaitParity": {"lifecycle_epoch", "resource", "generation", "raw_parity", "parity_evidence"},
        "TmaLoadIssue": {
            "lifecycle_epoch", "resource", "source_region", "destination_region", "bytes",
            "generation", "engine_id", "engine_seq",
        },
    }
    raw_events = {item["id"]: item for item in module.raw["events"]}
    for event in module.events:
        if event.id.startswith(("completion:", "group:")):
            raise PGIRError("RESERVED_ID_NAMESPACE", "real event ID uses a synthetic identity namespace", {"event": event.id})
        if event.actor not in actors:
            raise PGIRError("DANGLING_REFERENCE", "event references unknown actor", {"event": event.id, "actor": event.actor})
        if event.type not in IMPLEMENTED_EVENTS:
            code = "KNOWN_EVENT_UNSUPPORTED" if event.type in KNOWN_UNSUPPORTED_EVENTS else "UNKNOWN_EVENT"
            raise PGIRError(code, f"event type is unsupported: {event.type}", {"event": event.id, "type": event.type})
        if event.type == "MBarrierWaitParity" and event.parity_evidence is None:
            raise PGIRError("INCONCLUSIVE_PARITY", "wait lacks per-event logical-generation resolution evidence", {"event": event.id})
        _require(event, *required_fields[event.type])
        unexpected = sorted(set(raw_events[event.id]) - common - specific[event.type])
        if unexpected:
            raise PGIRError("EVENT_FIELDS_UNEXPECTED", f"{event.type} has fields that are not meaningful", {"event": event.id, "fields": unexpected})
        if event.resource is not None and event.resource not in resources:
            raise PGIRError("DANGLING_REFERENCE", "event references unknown resource", {"event": event.id, "resource": event.resource})
        referenced_regions = [
            value for value in (event.region, event.source_region, event.destination_region)
            if value is not None
        ] + list(event.regions)
        dangling_regions = sorted(value for value in referenced_regions if value not in regions)
        if dangling_regions:
            raise PGIRError("DANGLING_REFERENCE", "event references unknown region", {"event": event.id, "regions": dangling_regions})
        dangling_dependencies = sorted(set(event.depends_on) - events.keys())
        if dangling_dependencies:
            raise PGIRError("DANGLING_REFERENCE", "event has dangling dependencies", {"event": event.id, "depends_on": dangling_dependencies})
        if event.id in event.depends_on:
            raise PGIRError("SELF_DEPENDENCY", "event cannot depend on itself", {"event": event.id})
        if event.resource is not None and event.actor not in resources[event.resource].participants:
            raise PGIRError("UNRESOLVED_PARTICIPANTS", "event actor is outside resource participant set", {"event": event.id, "resource": event.resource})
        if event.type in {"BarrierArrive", "BarrierObserve"}:
            resource = resources[event.resource]
            if resource.kind not in {"cta_barrier", "named_barrier"}:
                raise PGIRError(
                    "RESOURCE_KIND_MISMATCH",
                    "collective barrier event must reference a CTA or named barrier resource",
                    {"event": event.id, "resource": event.resource, "kind": resource.kind},
                )
            if event.type == "BarrierArrive" and event.count != len(actors[event.actor].participants):
                raise PGIRError(
                    "COLLECTIVE_ARRIVAL_COUNT_MISMATCH",
                    "collective arrival count must cover the actor's complete lane set",
                    {"event": event.id, "count": event.count, "actor_lanes": list(actors[event.actor].participants)},
                )
        elif event.type in {"CpAsyncIssue", "CpAsyncCommitGroup", "CpAsyncWaitGroup"}:
            resource = resources[event.resource]
            if resource.kind != "async_group" or resource.protocol != "CP_ASYNC":
                raise PGIRError(
                    "RESOURCE_KIND_MISMATCH",
                    "cp.async event must reference a CP_ASYNC group resource",
                    {"event": event.id, "resource": event.resource, "kind": resource.kind, "protocol": resource.protocol},
                )
            if event.group_owner != event.actor:
                raise PGIRError(
                    "GROUP_OWNER_MISMATCH",
                    "cp.async groups are owned by the issuing program actor",
                    {"event": event.id, "actor": event.actor, "group_owner": event.group_owner},
                )
            if actors[event.actor].kind == "ASYNC_ENGINE":
                raise PGIRError(
                    "INVALID_GROUP_OWNER",
                    "an ASYNC_ENGINE actor cannot own a program-visible cp.async group",
                    {"event": event.id, "actor": event.actor},
                )
            if (
                event.type == "CpAsyncWaitGroup"
                and event.wait_group_count >= resource.capacity
            ):
                raise PGIRError(
                    "WAIT_GROUP_COUNT_OUT_OF_RANGE",
                    "cp.async wait_group count must be smaller than the group capacity",
                    {
                        "event": event.id,
                        "wait_group_count": event.wait_group_count,
                        "capacity": resource.capacity,
                    },
                )
            if event.type == "CpAsyncIssue" and (
                regions[event.source_region].address_space != "GLOBAL"
                or regions[event.destination_region].address_space != "SHARED"
            ):
                raise PGIRError(
                    "UNSUPPORTED_CP_ASYNC_REGION",
                    "cp.async requires an explicit global source and shared destination",
                    {"event": event.id},
                )
        elif event.resource is not None and resources[event.resource].kind != "mbarrier":
            raise PGIRError(
                "RESOURCE_KIND_MISMATCH",
                "mbarrier/TMA-load event must reference an mbarrier resource",
                {"event": event.id, "resource": event.resource, "kind": resources[event.resource].kind},
            )
        if (event.engine_id is None) != (event.engine_seq is None):
            raise PGIRError(
                "ASYNC_ENGINE_EVIDENCE_INCOMPLETE",
                "engine_id and engine_seq must be provided together",
                {"event": event.id},
            )
        if event.engine_id is not None:
            engine = actors.get(event.engine_id)
            if engine is None or engine.kind != "ASYNC_ENGINE":
                raise PGIRError(
                    "INVALID_ASYNC_ENGINE",
                    "explicit engine ordering must reference an ASYNC_ENGINE actor",
                    {"event": event.id, "engine": event.engine_id},
                )
        if event.type == "TmaLoadIssue" and (
            regions[event.source_region].address_space != "GLOBAL"
            or regions[event.destination_region].address_space != "SHARED"
        ):
            raise PGIRError("UNSUPPORTED_TMA_REGION", "TMA load requires an explicit global source and shared destination", {"event": event.id})

    engine_streams: dict[str, list[Event]] = {}
    for event in module.events:
        if event.engine_id is not None:
            engine_streams.setdefault(event.engine_id, []).append(event)
    for engine_id, ordered_events in sorted(engine_streams.items()):
        ordered_events.sort(key=lambda event: (event.engine_seq, event.id))
        sequence = [event.engine_seq for event in ordered_events]
        if sequence != list(range(len(sequence))):
            raise PGIRError(
                "ASYNC_ENGINE_SEQUENCE_GAP",
                "explicit async-engine sequence must be unique, gap-free, and start at zero",
                {"engine": engine_id, "sequence": sequence},
            )

    commit_streams: dict[tuple[str, str], list[Event]] = {}
    for event in module.events:
        if event.type == "CpAsyncCommitGroup":
            commit_streams.setdefault((event.resource, event.group_owner), []).append(event)
    for stream, commits in sorted(commit_streams.items()):
        commits.sort(key=lambda event: (event.group_seq, event.seq, event.id))
        sequence = [event.group_seq for event in commits]
        if sequence != list(range(len(sequence))):
            raise PGIRError(
                "COMMIT_GROUP_SEQUENCE_GAP",
                "commit-group sequence must be unique, gap-free, and start at zero",
                {"stream": stream, "sequence": sequence},
            )

    wait_streams: dict[tuple[str, str, int, str], list[Event]] = {}
    canonical_stream_ids: dict[tuple[str, str, int], set[str]] = {}
    for event in module.events:
        if event.type != "MBarrierWaitParity":
            continue
        assert event.resource is not None and event.generation is not None and event.raw_parity is not None and event.parity_evidence is not None
        evidence = event.parity_evidence
        if event.lifecycle_epoch != evidence.lifecycle_epoch:
            raise PGIRError("LOGICAL_WAIT_LIFECYCLE_MISMATCH", "wait event and evidence name different lifecycle epochs", {"event": event.id, "event_lifecycle": event.lifecycle_epoch, "evidence_lifecycle": evidence.lifecycle_epoch})
        if not evidence.polling_coalesced:
            raise PGIRError("POLLING_NOT_COALESCED", "logical wait/query identity requires a coalesced polling occurrence", {"event": event.id})
        if event.raw_parity != event.generation % 2:
            raise PGIRError("MALFORMED_STREAM_PARITY", "raw wait parity does not match its logical generation", {"event": event.id, "generation": event.generation, "raw_parity": event.raw_parity})
        canonical_stream = (event.actor, event.resource, evidence.lifecycle_epoch)
        canonical_stream_ids.setdefault(canonical_stream, set()).add(evidence.logical_stream_id)
        wait_streams.setdefault((*canonical_stream, evidence.logical_stream_id), []).append(event)
    for stream, identifiers in sorted(canonical_stream_ids.items()):
        if len(identifiers) != 1:
            raise PGIRError(
                "LOGICAL_WAIT_STREAM_ID_MISMATCH",
                "one canonical actor/barrier/lifecycle stream cannot be split across logical stream IDs",
                {"stream": stream, "logical_stream_ids": sorted(identifiers)},
            )
    for stream, waits in sorted(wait_streams.items()):
        generations = [event.generation for event in waits]
        seen_generations: set[int] = set()
        duplicate_generation = None
        for generation in generations:
            if generation in seen_generations:
                duplicate_generation = generation
                break
            seen_generations.add(generation)
        if duplicate_generation is not None:
            raise PGIRError("DUPLICATE_LOGICAL_WAIT", "logical stream contains duplicate waits for one generation", {"stream": stream, "generation": duplicate_generation, "events": sorted(event.id for event in waits if event.generation == duplicate_generation)})
        ranked = sorted(waits, key=lambda event: (event.parity_evidence.stream_rank, event.seq, event.id))
        ranks = [event.parity_evidence.stream_rank for event in ranked]
        if ranks != list(range(len(ranked))):
            raise PGIRError("LOGICAL_WAIT_RANK_GAP", "logical wait ranks must be gap-free and start at zero", {"stream": stream, "ranks": ranks})
        mismatch = next((event for event in ranked if event.generation != event.parity_evidence.stream_rank), None)
        if mismatch is not None:
            raise PGIRError("LOGICAL_WAIT_GENERATION_MISMATCH", "logical generation must equal stream rank in a fresh lifecycle", {"event": mismatch.id, "generation": mismatch.generation, "stream_rank": mismatch.parity_evidence.stream_rank})
        program_order = sorted(waits, key=lambda event: (event.seq, event.id))
        if [event.id for event in ranked] != [event.id for event in program_order]:
            raise PGIRError("LOGICAL_WAIT_ORDER_MISMATCH", "logical stream rank disagrees with actor program order", {"stream": stream})

    # The current static lift is complete only for fresh, gap-free lifecycles.
    # A represented generation k can retire target k-1 only when every earlier
    # generation has a statically complete arrival count. Such a covered
    # boundary needs a nonempty LogicalWait group; absence is a support-domain
    # failure, not a hardware synchronization hazard.
    required = module.required_logical_waits
    init_counts: dict[tuple[str, int], int] = {}
    arrival_counts: dict[tuple[str, int, int], int] = {}
    for event in module.events:
        if event.type == "MBarrierInit":
            assert event.resource is not None and event.lifecycle_epoch is not None and event.count is not None
            init_counts.setdefault((event.resource, event.lifecycle_epoch), event.count)
        elif event.type in {"MBarrierArrive", "MBarrierArriveExpectTx"} and event.predicate:
            assert event.resource is not None and event.lifecycle_epoch is not None and event.generation is not None and event.count is not None
            key = (event.resource, event.lifecycle_epoch, event.generation)
            arrival_counts[key] = arrival_counts.get(key, 0) + event.count
    for (resource, lifecycle_epoch), capacity in sorted(init_counts.items()):
        generation = 0
        while arrival_counts.get((resource, lifecycle_epoch, generation), 0) == capacity:
            if generation > 0 and not required.get((resource, lifecycle_epoch, generation - 1)):
                raise PGIRError(
                    "UNSUPPORTED_MISSING_OBSERVE_GROUP",
                    "covered phase completion lacks a required LogicalWait observe group",
                    {"resource": resource, "lifecycle_epoch": lifecycle_epoch, "target_generation": generation - 1},
                )
            generation += 1

    synthetic_ids = {
        stable_completion_id(event.id, event.generation)
        for event in module.events if event.type == "TmaLoadIssue"
    } | {
        stable_phase_id(event.resource, event.lifecycle_epoch, event.generation)
        for event in module.events if event.resource is not None and event.generation is not None
    } | {
        stable_collective_release_id(event.resource, event.invocation_id)
        for event in module.events if event.type in {"BarrierArrive", "BarrierObserve"}
    } | {
        stable_group_completion_id("CP_ASYNC", event.group_owner, event.group_seq, "GROUP_COMPLETE")
        for event in module.events if event.type == "CpAsyncCommitGroup"
    } | {
        stable_group_id("CP_ASYNC", event.group_owner, event.group_seq)
        for event in module.events if event.type == "CpAsyncCommitGroup"
    }
    collisions = sorted(synthetic_ids & events.keys())
    if collisions:
        raise PGIRError("STABLE_ID_COLLISION", "real and synthetic node IDs collide", {"ids": collisions})

    for actor_id, trace in module.events_by_actor.items():
        seqs = [event.seq for event in trace]
        if seqs != list(range(len(seqs))):
            raise PGIRError("NONCONTIGUOUS_SEQUENCE", "actor sequence numbers must start at zero and be contiguous", {"actor": actor_id, "sequence": seqs})


def module_from_dict(data: dict[str, Any]) -> Module:
    _schema_validate(data)
    actors = tuple(Actor(item["id"], item["kind"], tuple(sorted(item["participants"]))) for item in data["actors"])
    resources = tuple(Resource(
        item["id"],
        item["kind"],
        tuple(sorted(item["participants"])),
        capacity=item.get("capacity"),
        protocol=item.get("protocol"),
        scope=item.get("scope"),
        visibility_mask=VisibilityMask.from_mapping(item["visibility_mask"]) if "visibility_mask" in item else None,
    ) for item in data["resources"])
    regions = tuple(Region(item["id"], item["allocation_id"], item["address_space"], item["offset"], item["extent"]) for item in data["regions"])
    events = tuple(Event(
        id=item["id"], type=item["type"], actor=item["actor"], seq=item["seq"],
        depends_on=tuple(sorted(item.get("depends_on", []))), resource=item.get("resource"),
        region=item.get("region"), source_region=item.get("source_region"), destination_region=item.get("destination_region"),
        count=item.get("count"), bytes=item.get("bytes"), lifecycle_epoch=item.get("lifecycle_epoch"), generation=item.get("generation"), raw_parity=item.get("raw_parity"),
        parity_evidence=ParityEvidence(**item["parity_evidence"]) if "parity_evidence" in item else None,
        predicate=item.get("predicate", True), invocation_id=item.get("invocation_id"),
        group_owner=item.get("group_owner"), group_seq=item.get("group_seq"),
        wait_group_count=item.get("wait_group_count"), completion_kind=item.get("completion_kind"),
        wait_mode=item.get("wait_mode"), engine_id=item.get("engine_id"), engine_seq=item.get("engine_seq"),
        slot=item.get("slot"), logical_phase=item.get("logical_phase"),
        from_domain=item.get("from_domain"), to_domain=item.get("to_domain"), scope=item.get("scope"),
        regions=tuple(sorted(item.get("regions", []))),
        visibility_mask=VisibilityMask.from_mapping(item["visibility_mask"]) if "visibility_mask" in item else None,
    ) for item in data["events"])
    grouped = {actor.id: tuple(sorted((event for event in events if event.actor == actor.id), key=lambda event: (event.seq, event.id))) for actor in actors}
    module = Module(data, actors, resources, regions, events, grouped)
    _semantic_validate(module)
    return module


def load_module(path: str | Path) -> Module:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PGIRError("JSON_INVALID", str(error)) from error
    if not isinstance(data, dict):
        raise PGIRError("SCHEMA_INVALID", "PGIR document must be a JSON object")
    return module_from_dict(data)


def normalize_module(module: Module) -> dict[str, Any]:
    normalized = json.loads(json.dumps(module.raw))
    normalized["actors"] = sorted(normalized["actors"], key=lambda item: item["id"])
    normalized["resources"] = sorted(normalized["resources"], key=lambda item: item["id"])
    normalized["regions"] = sorted(normalized["regions"], key=lambda item: item["id"])
    normalized["events"] = sorted(normalized["events"], key=lambda item: (item["actor"], item["seq"], item["id"]))
    for actor in normalized["actors"]:
        actor["participants"] = sorted(actor["participants"])
    for resource in normalized["resources"]:
        resource["participants"] = sorted(resource["participants"])
    for event in normalized["events"]:
        if "depends_on" in event:
            event["depends_on"] = sorted(event["depends_on"])
        if "regions" in event:
            event["regions"] = sorted(event["regions"])
    normalized["unsupported_constructs"] = sorted(normalized["unsupported_constructs"])
    return normalized
