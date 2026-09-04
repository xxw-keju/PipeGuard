from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import fastjsonschema
import jsonschema

from .protocols import RESOURCE_KINDS, VisibilityMask

IMPLEMENTED_RESOURCE_KINDS = frozenset({"mbarrier", "cta_barrier", "named_barrier", "cluster_barrier", "async_group", "channel"})
IMPLEMENTED_ASYNC_PROTOCOLS = frozenset({"CP_ASYNC", "WGMMA", "TMA_STORE"})

IMPLEMENTED_EVENTS = frozenset({
    "MemRead", "MemWrite", "BarrierArrive", "BarrierObserve",
    "MBarrierInit", "MBarrierInvalidate", "MBarrierArrive", "MBarrierExpectTx",
    "MBarrierArriveExpectTx", "MBarrierWaitParity", "TmaLoadIssue",
    "CpAsyncIssue", "CpAsyncCommitGroup", "CpAsyncWaitGroup",
    "WgmmaFenceOperand", "WgmmaIssue", "WgmmaCommitGroup", "WgmmaWaitGroup",
    "TmaStoreIssue", "TmaStoreCommitGroup", "TmaStoreWaitGroup",
    "ChannelCreate", "ProducerAcquire", "ProducerCommit", "ConsumerWait", "ConsumerRelease",
    "ProxyFence",
})
KNOWN_UNSUPPORTED_EVENTS = frozenset({
    "AtomicReadModifyWrite", "Fork", "Join", "BarrierRelease",
    "MBarrierTxComplete", "MBarrierPhaseComplete",
    "CpAsyncGroupComplete",
    "TmaLoadComplete", "TmaStoreSourceReleased", "TmaStoreGlobalComplete",
    "WgmmaResultReady", "WgmmaOperandsReleased",
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
    cta_rank: int | None = None


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
    owner_cta_rank: int | None = None


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
    _event_map: dict[str, Event] = field(compare=False, hash=False, repr=False)
    _resource_map: dict[str, Resource] = field(compare=False, hash=False, repr=False)
    _region_map: dict[str, Region] = field(compare=False, hash=False, repr=False)
    _actor_map: dict[str, Actor] = field(compare=False, hash=False, repr=False)

    @property
    def event_map(self) -> dict[str, Event]:
        return self._event_map

    @property
    def resource_map(self) -> dict[str, Resource]:
        return self._resource_map

    @property
    def region_map(self) -> dict[str, Region]:
        return self._region_map

    @property
    def actor_map(self) -> dict[str, Actor]:
        return self._actor_map

    @property
    def required_logical_waits(self) -> dict[tuple[str, int, int], frozenset[str]]:
        required: dict[tuple[str, int, int], set[str]] = {}
        for event in self.events:
            if event.type != "MBarrierWaitParity":
                continue
            assert event.resource is not None and event.lifecycle_epoch is not None and event.generation is not None
            required.setdefault((event.resource, event.lifecycle_epoch, event.generation), set()).add(event.id)
        return {key: frozenset(event_ids) for key, event_ids in required.items()}


def _bucket_events_by_actor(
    actors: tuple[Actor, ...],
    events: tuple[Event, ...],
) -> dict[str, tuple[Event, ...]]:
    buckets: dict[str, list[Event]] = {actor.id: [] for actor in actors}
    for event in events:
        bucket = buckets.get(event.actor)
        if bucket is not None:
            bucket.append(event)
    return {
        actor_id: tuple(sorted(bucket, key=lambda event: (event.seq, event.id)))
        for actor_id, bucket in buckets.items()
    }


def _dangling_dependencies(
    dependencies: tuple[str, ...],
    events: Mapping[str, Event],
) -> list[str]:
    return sorted({dependency for dependency in dependencies if dependency not in events})


@lru_cache(maxsize=None)
def _schema(version: str) -> dict[str, Any]:
    if not isinstance(version, str) or version not in {"pgir-v1", "pgir-v2"}:
        raise PGIRError(
            "UNKNOWN_SCHEMA_VERSION",
            "PGIR schema version is not supported",
            {"schema_version": version},
        )
    return json.loads(files("pipeguard.specs").joinpath(f"{version}.schema.json").read_text())


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


def engine_completion_streams(module: Module) -> dict[str, tuple[str, ...]]:
    """Lift explicit issue order into finite typed completion streams."""

    terminal_by_issue: dict[str, str] = {}
    for event in module.events:
        if event.type == "TmaLoadIssue":
            assert event.generation is not None
            terminal_by_issue[event.id] = stable_completion_id(event.id, event.generation)

    terminal_kinds = {
        "CpAsyncCommitGroup": ("CP_ASYNC", "GROUP_COMPLETE"),
        "WgmmaCommitGroup": ("WGMMA", "RESULT_READY"),
        "TmaStoreCommitGroup": ("TMA_STORE", "GLOBAL_COMPLETE"),
    }
    issue_types = {"CpAsyncIssue", "WgmmaIssue", "TmaStoreIssue"}
    for actor in sorted(module.events_by_actor):
        staged: dict[str, list[str]] = {}
        for event in module.events_by_actor[actor]:
            if event.type in issue_types:
                assert event.resource is not None
                if event.type != "CpAsyncIssue" or event.predicate:
                    staged.setdefault(event.resource, []).append(event.id)
                continue
            terminal = terminal_kinds.get(event.type)
            if terminal is None:
                continue
            assert event.resource is not None and event.group_owner is not None
            assert event.group_seq is not None
            protocol, completion_kind = terminal
            completion = stable_group_completion_id(
                protocol,
                event.group_owner,
                event.group_seq,
                completion_kind,
            )
            for issue in staged.pop(event.resource, []):
                terminal_by_issue[issue] = completion

    engine_events: dict[str, list[Event]] = {}
    for event in module.events:
        if event.engine_id is not None:
            engine_events.setdefault(event.engine_id, []).append(event)

    streams: dict[str, tuple[str, ...]] = {}
    endpoint_engines: dict[str, str] = {}
    for engine, events in sorted(engine_events.items()):
        endpoints: list[str] = []
        seen: set[str] = set()
        for event in sorted(events, key=lambda item: (item.engine_seq, item.id)):
            endpoint = terminal_by_issue.get(event.id)
            if endpoint is None:
                raise PGIRError(
                    "ASYNC_ENGINE_COMPLETION_UNRESOLVED",
                    "explicit async-engine issue has no statically resolved terminal completion",
                    {"engine": engine, "event": event.id},
                )
            previous_engine = endpoint_engines.setdefault(endpoint, engine)
            if previous_engine != engine:
                raise PGIRError(
                    "AMBIGUOUS_ASYNC_ENGINE_COMPLETION",
                    "one typed completion endpoint cannot belong to multiple async engines",
                    {"completion": endpoint, "engines": sorted({previous_engine, engine})},
                )
            if endpoints and endpoints[-1] == endpoint:
                continue
            if endpoint in seen:
                raise PGIRError(
                    "AMBIGUOUS_ASYNC_ENGINE_COMPLETION",
                    "one typed completion endpoint cannot reappear non-contiguously in an engine stream",
                    {"engine": engine, "completion": endpoint},
                )
            endpoints.append(endpoint)
            seen.add(endpoint)
        streams[engine] = tuple(endpoints)
    return streams


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
        shape = target.get("cluster_shape")
        if isinstance(shape, dict):
            for name in ("x", "y", "z"):
                if name in shape:
                    _strict_integer(shape[name], f"/target/cluster_shape/{name}")
    for index, actor in enumerate(data.get("actors", [])):
        if "cta_rank" in actor:
            _strict_integer(actor["cta_rank"], f"/actors/{index}/cta_rank")
        for lane_index, lane in enumerate(actor.get("participants", [])):
            _strict_integer(lane, f"/actors/{index}/participants/{lane_index}")
    for index, resource in enumerate(data.get("resources", [])):
        if "capacity" in resource:
            _strict_integer(resource["capacity"], f"/resources/{index}/capacity")
    for index, region in enumerate(data.get("regions", [])):
        for name in ("offset", "extent", "owner_cta_rank"):
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


@lru_cache(maxsize=None)
def _fast_schema_validator(version: str) -> Any:
    return fastjsonschema.compile(_schema(version))


@lru_cache(maxsize=None)
def _diagnostic_schema_validator(version: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(_schema(version))


def _schema_validate(data: Any) -> None:
    _strict_integers(data)
    version = data.get("schema_version") if isinstance(data, dict) else None
    try:
        _fast_schema_validator(version)(data)
        return
    except fastjsonschema.JsonSchemaException:
        pass
    errors = sorted(
        _diagnostic_schema_validator(version).iter_errors(data),
        key=lambda error: (list(error.absolute_path), error.message),
    )
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
    missing = [
        name for name in fields
        if getattr(event, name) is None
        or (name == "regions" and not getattr(event, name))
    ]
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
    schema_version = module.raw["schema_version"]
    expected_assumptions: dict[str, Any] = {
        "finite_control_flow": True,
        "statically_known_participants": True,
        "parity_normalization_status": "RESOLVED",
        "alias_precision": "EXACT",
        "fair_async_completion": True,
        "no_cluster_effects": schema_version == "pgir-v1",
    }
    if schema_version == "pgir-v2":
        expected_assumptions.update({
            "cluster_topology_resolved": True,
            "remote_alias_precision": "EXACT",
        })
    unresolved = sorted(
        key for key, expected in expected_assumptions.items()
        if assumptions.get(key) != expected
    )
    if unresolved:
        code = "INCONCLUSIVE_PARITY" if "parity_normalization_status" in unresolved else "UNRESOLVED_ASSUMPTIONS"
        raise PGIRError(code, "SAFE preconditions are unresolved", {"assumptions": unresolved})
    if module.raw["unsupported_constructs"]:
        raise PGIRError("UNSUPPORTED_CONSTRUCT", "frontend reported unsupported constructs", {"constructs": sorted(module.raw["unsupported_constructs"])})

    cta_threads = module.raw["target"]["cta_threads"]
    cluster_ctas = 1
    if schema_version == "pgir-v2":
        shape = module.raw["target"]["cluster_shape"]
        cluster_ctas = shape["x"] * shape["y"] * shape["z"]
        if cluster_ctas > 8:
            raise PGIRError(
                "NONPORTABLE_CLUSTER_SIZE",
                "the supported portable SM90 cluster subset contains at most eight CTAs",
                {"cluster_shape": shape, "cluster_ctas": cluster_ctas},
            )
    for actor in module.actors:
        if schema_version == "pgir-v1" and actor.cta_rank is not None:
            raise PGIRError(
                "V1_CLUSTER_FIELD_FORBIDDEN",
                "PGIR v1 actors cannot carry a CTA rank",
                {"actor": actor.id},
            )
        if schema_version == "pgir-v2" and (
            actor.cta_rank is None or actor.cta_rank >= cluster_ctas
        ):
            raise PGIRError(
                "CTA_RANK_OUT_OF_RANGE",
                "actor CTA rank is outside the static cluster shape",
                {"actor": actor.id, "cta_rank": actor.cta_rank, "cluster_ctas": cluster_ctas},
            )
        invalid_lanes = sorted(lane for lane in actor.participants if lane >= cta_threads)
        if invalid_lanes:
            raise PGIRError("PARTICIPANT_OUT_OF_RANGE", "actor participant exceeds CTA bounds", {"actor": actor.id, "lanes": invalid_lanes, "cta_threads": cta_threads})
    if schema_version == "pgir-v2":
        represented_ranks = {actor.cta_rank for actor in module.actors if actor.kind != "ASYNC_ENGINE"}
        if represented_ranks != set(range(cluster_ctas)):
            raise PGIRError(
                "CLUSTER_TOPOLOGY_INCOMPLETE",
                "every CTA in the static cluster must have at least one program actor",
                {"represented_ranks": sorted(represented_ranks), "cluster_ctas": cluster_ctas},
            )
        for region in module.regions:
            if region.address_space == "DSMEM" and (
                region.owner_cta_rank is None or region.owner_cta_rank >= cluster_ctas
            ):
                raise PGIRError(
                    "DSMEM_OWNER_OUT_OF_RANGE",
                    "DSMEM region owner is outside the static cluster shape",
                    {"region": region.id, "owner_cta_rank": region.owner_cta_rank, "cluster_ctas": cluster_ctas},
                )
    for resource in module.resources:
        dangling = sorted(set(resource.participants) - actors.keys())
        if dangling:
            raise PGIRError("UNRESOLVED_PARTICIPANTS", "resource has unknown participants", {"resource": resource.id, "actors": dangling})
        if resource.kind not in RESOURCE_KINDS:
            raise PGIRError("UNKNOWN_RESOURCE_KIND", "resource kind is unknown", {"resource": resource.id, "kind": resource.kind})
        if resource.kind not in IMPLEMENTED_RESOURCE_KINDS:
            raise PGIRError("KNOWN_RESOURCE_UNSUPPORTED", "resource kind is not implemented yet", {"resource": resource.id, "kind": resource.kind})
        if schema_version == "pgir-v2" and resource.kind != "cluster_barrier":
            participant_ranks = {actors[actor].cta_rank for actor in resource.participants}
            if len(participant_ranks) != 1:
                raise PGIRError(
                    "CROSS_CTA_LOCAL_RESOURCE",
                    "non-cluster resources cannot span multiple CTAs",
                    {"resource": resource.id, "cta_ranks": sorted(participant_ranks)},
                )
            if resource.scope == "CLUSTER":
                raise PGIRError(
                    "RESOURCE_SCOPE_MISMATCH",
                    "only cluster barriers may use CLUSTER scope in the supported PGIR v2 subset",
                    {"resource": resource.id, "kind": resource.kind},
                )
            if resource.visibility_mask is not None and resource.visibility_mask.dsmem:
                raise PGIRError(
                    "UNSUPPORTED_DSMEM_VISIBILITY",
                    "only cluster barriers may publish DSMEM visibility",
                    {"resource": resource.id, "kind": resource.kind},
                )
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
        elif resource.kind in {"cta_barrier", "named_barrier", "cluster_barrier"}:
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
            participant_slots = [
                (actors[actor_id].cta_rank or 0, lane)
                for actor_id in resource.participants
                for lane in actors[actor_id].participants
            ]
            if len(participant_slots) != len(set(participant_slots)):
                raise PGIRError(
                    "OVERLAPPING_COLLECTIVE_PARTICIPANTS",
                    "collective barrier actors must provide disjoint CTA-qualified lane sets",
                    {"resource": resource.id},
                )
            if resource.capacity != len(participant_slots):
                raise PGIRError(
                    "COLLECTIVE_CAPACITY_MISMATCH",
                    "collective barrier capacity must equal its complete participant lane set",
                    {"resource": resource.id, "capacity": resource.capacity, "participant_slots": participant_slots},
                )
            if resource.kind == "cta_barrier":
                rank = participant_slots[0][0]
                expected_slots = {(rank, lane) for lane in range(cta_threads)}
                if set(participant_slots) != expected_slots:
                    raise PGIRError(
                        "CTA_PARTICIPANT_SET_INCOMPLETE",
                        "CTA barrier must cover every lane in one CTA exactly once",
                        {"resource": resource.id, "participant_slots": participant_slots, "cta_threads": cta_threads},
                    )
            if resource.kind == "cluster_barrier":
                expected_slots = {
                    (rank, lane)
                    for rank in range(cluster_ctas)
                    for lane in range(cta_threads)
                }
                if schema_version != "pgir-v2" or resource.scope != "CLUSTER":
                    raise PGIRError(
                        "RESOURCE_SCOPE_MISMATCH",
                        "cluster barriers require PGIR v2 CLUSTER scope",
                        {"resource": resource.id, "scope": resource.scope},
                    )
                if set(participant_slots) != expected_slots:
                    raise PGIRError(
                        "CLUSTER_PARTICIPANT_SET_INCOMPLETE",
                        "cluster barrier must cover every lane in every CTA exactly once",
                        {"resource": resource.id, "participant_slots": participant_slots, "cluster_ctas": cluster_ctas, "cta_threads": cta_threads},
                    )
            assert resource.visibility_mask is not None
            mask = resource.visibility_mask
            data_visibility = any((
                mask.shared,
                mask.dsmem,
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
            if resource.kind == "cluster_barrier" and data_visibility and (
                not mask.dsmem
                or mask.shared
                or mask.global_read
                or mask.global_write
                or mask.tensor_read
                or mask.tensor_write
            ):
                raise PGIRError(
                    "UNSUPPORTED_CLUSTER_VISIBILITY",
                    "cluster barrier data visibility is limited to DSMEM in PGIR v2",
                    {"resource": resource.id, "visibility_mask": mask.artifact()},
                )
        elif resource.kind == "channel":
            missing = [
                name for name in ("capacity", "scope", "visibility_mask")
                if getattr(resource, name) is None
            ]
            if missing:
                raise PGIRError(
                    "RESOURCE_FIELDS_MISSING",
                    "channel resource lacks required metadata",
                    {"resource": resource.id, "fields": missing},
                )
            if resource.protocol is not None:
                raise PGIRError(
                    "RESOURCE_FIELDS_UNEXPECTED",
                    "channel resource cannot carry an async-group protocol",
                    {"resource": resource.id, "fields": ["protocol"]},
                )
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
                    "channel must select either control-only or at least one data visibility domain",
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
            if resource.protocol in {"CP_ASYNC", "WGMMA", "TMA_STORE"} and resource.capacity > 8:
                code = {
                    "CP_ASYNC": "CP_ASYNC_CAPACITY_UNSUPPORTED",
                    "WGMMA": "WGMMA_CAPACITY_UNSUPPORTED",
                    "TMA_STORE": "TMA_STORE_CAPACITY_UNSUPPORTED",
                }[resource.protocol]
                raise PGIRError(
                    code,
                    f"{resource.protocol.lower()} supports at most eight uncompleted committed groups",
                    {"resource": resource.id, "capacity": resource.capacity},
                )
            if resource.protocol == "WGMMA":
                participant_lanes = [
                    lane
                    for owner in resource.participants
                    for lane in actors[owner].participants
                ]
                if len(participant_lanes) != len(set(participant_lanes)):
                    raise PGIRError(
                        "OVERLAPPING_WGMMA_PARTICIPANTS",
                        "WGMMA owner warpgroups must have disjoint lane sets",
                        {"resource": resource.id},
                    )
                for owner in resource.participants:
                    actor = actors[owner]
                    lanes = tuple(sorted(actor.participants))
                    if (
                        actor.kind != "WARPGROUP"
                        or len(lanes) != 128
                        or lanes != tuple(range(lanes[0], lanes[0] + 128))
                        or lanes[0] % 128 != 0
                    ):
                        raise PGIRError(
                            "INVALID_WGMMA_PARTICIPANTS",
                            "WGMMA owners must be complete aligned 128-lane warpgroups",
                            {"resource": resource.id, "owner": owner, "lanes": list(lanes)},
                        )

    group_owners: dict[tuple[str, str], list[str]] = {}
    for resource in module.resources:
        if resource.kind == "async_group" and resource.protocol in IMPLEMENTED_ASYNC_PROTOCOLS:
            for owner in resource.participants:
                group_owners.setdefault((resource.protocol, owner), []).append(resource.id)
    ambiguous_owners = {
        f"{protocol}:{owner}": sorted(resource_ids)
        for (protocol, owner), resource_ids in group_owners.items()
        if len(resource_ids) > 1
    }
    if ambiguous_owners:
        raise PGIRError(
            "AMBIGUOUS_GROUP_OWNER_RESOURCE",
            "one protocol owner cannot span multiple async-group resources in PGIR v1",
            {"owners": ambiguous_owners},
        )

    required_fields = {
        "MemRead": ("region",), "MemWrite": ("region",),
        "BarrierArrive": ("resource", "count", "invocation_id"),
        "BarrierObserve": ("resource", "invocation_id"),
        "CpAsyncIssue": ("resource", "source_region", "destination_region", "bytes", "group_owner"),
        "CpAsyncCommitGroup": ("resource", "group_owner", "group_seq"),
        "CpAsyncWaitGroup": ("resource", "group_owner", "wait_group_count"),
        "WgmmaFenceOperand": ("resource", "group_owner", "regions"),
        "WgmmaIssue": ("resource", "group_owner", "regions", "destination_region"),
        "WgmmaCommitGroup": ("resource", "group_owner", "group_seq"),
        "WgmmaWaitGroup": ("resource", "group_owner", "wait_group_count", "completion_kind"),
        "TmaStoreIssue": ("resource", "group_owner", "source_region", "destination_region", "bytes"),
        "TmaStoreCommitGroup": ("resource", "group_owner", "group_seq"),
        "TmaStoreWaitGroup": ("resource", "group_owner", "wait_group_count", "wait_mode"),
        "ChannelCreate": ("resource",),
        "ProducerAcquire": ("resource", "slot", "logical_phase"),
        "ProducerCommit": ("resource", "slot", "logical_phase"),
        "ConsumerWait": ("resource", "slot", "logical_phase"),
        "ConsumerRelease": ("resource", "slot", "logical_phase"),
        "ProxyFence": ("from_domain", "to_domain", "scope", "regions", "visibility_mask"),
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
        "WgmmaFenceOperand": {"resource", "group_owner", "regions"},
        "WgmmaIssue": {
            "resource", "group_owner", "regions", "destination_region",
            "engine_id", "engine_seq",
        },
        "WgmmaCommitGroup": {"resource", "group_owner", "group_seq"},
        "WgmmaWaitGroup": {
            "resource", "group_owner", "wait_group_count", "completion_kind",
        },
        "TmaStoreIssue": {
            "resource", "group_owner", "source_region", "destination_region", "bytes",
            "engine_id", "engine_seq",
        },
        "TmaStoreCommitGroup": {"resource", "group_owner", "group_seq"},
        "TmaStoreWaitGroup": {
            "resource", "group_owner", "wait_group_count", "wait_mode",
        },
        "ChannelCreate": {"resource"},
        "ProducerAcquire": {"resource", "slot", "logical_phase"},
        "ProducerCommit": {"resource", "slot", "logical_phase"},
        "ConsumerWait": {"resource", "slot", "logical_phase"},
        "ConsumerRelease": {"resource", "slot", "logical_phase"},
        "ProxyFence": {"from_domain", "to_domain", "scope", "regions", "visibility_mask"},
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
        dangling_dependencies = _dangling_dependencies(event.depends_on, events)
        if dangling_dependencies:
            raise PGIRError("DANGLING_REFERENCE", "event has dangling dependencies", {"event": event.id, "depends_on": dangling_dependencies})
        if event.id in event.depends_on:
            raise PGIRError("SELF_DEPENDENCY", "event cannot depend on itself", {"event": event.id})
        if event.resource is not None and event.actor not in resources[event.resource].participants:
            raise PGIRError("UNRESOLVED_PARTICIPANTS", "event actor is outside resource participant set", {"event": event.id, "resource": event.resource})
        if event.type in {"BarrierArrive", "BarrierObserve"}:
            resource = resources[event.resource]
            if resource.kind not in {"cta_barrier", "named_barrier", "cluster_barrier"}:
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
        elif event.type in {
            "WgmmaFenceOperand",
            "WgmmaIssue",
            "WgmmaCommitGroup",
            "WgmmaWaitGroup",
        }:
            resource = resources[event.resource]
            if resource.kind != "async_group" or resource.protocol != "WGMMA":
                raise PGIRError(
                    "RESOURCE_KIND_MISMATCH",
                    "WGMMA event must reference a WGMMA group resource",
                    {"event": event.id, "resource": event.resource, "kind": resource.kind, "protocol": resource.protocol},
                )
            if event.group_owner != event.actor:
                raise PGIRError(
                    "GROUP_OWNER_MISMATCH",
                    "WGMMA groups are owned by the issuing warpgroup actor",
                    {"event": event.id, "actor": event.actor, "group_owner": event.group_owner},
                )
            if event.type in {"WgmmaFenceOperand", "WgmmaIssue"} and any(
                regions[region].address_space != "SHARED" for region in event.regions
            ):
                raise PGIRError(
                    "UNSUPPORTED_WGMMA_OPERAND_REGION",
                    "WGMMA operands must be explicit shared-memory regions",
                    {"event": event.id, "regions": list(event.regions)},
                )
            if event.type == "WgmmaIssue" and regions[event.destination_region].address_space != "REGISTER":
                raise PGIRError(
                    "UNSUPPORTED_WGMMA_ACCUMULATOR_REGION",
                    "WGMMA accumulator output must be an explicit register region",
                    {"event": event.id, "region": event.destination_region},
                )
            if event.type == "WgmmaWaitGroup":
                if event.wait_group_count >= resource.capacity:
                    raise PGIRError(
                        "WAIT_GROUP_COUNT_OUT_OF_RANGE",
                        "WGMMA wait_group count must be smaller than the group capacity",
                        {"event": event.id, "wait_group_count": event.wait_group_count, "capacity": resource.capacity},
                    )
                if event.completion_kind not in {"OPERANDS_RELEASED", "RESULT_READY"}:
                    raise PGIRError(
                        "INVALID_WGMMA_WAIT_ENDPOINT",
                        "WGMMA wait must explicitly select operands released or result ready",
                        {"event": event.id, "completion_kind": event.completion_kind},
                    )
        elif event.type in {"TmaStoreIssue", "TmaStoreCommitGroup", "TmaStoreWaitGroup"}:
            resource = resources[event.resource]
            if resource.kind != "async_group" or resource.protocol != "TMA_STORE":
                raise PGIRError(
                    "RESOURCE_KIND_MISMATCH",
                    "TMA store event must reference a TMA_STORE group resource",
                    {"event": event.id, "resource": event.resource, "kind": resource.kind, "protocol": resource.protocol},
                )
            if event.group_owner != event.actor:
                raise PGIRError(
                    "GROUP_OWNER_MISMATCH",
                    "TMA store groups are owned by the issuing program actor",
                    {"event": event.id, "actor": event.actor, "group_owner": event.group_owner},
                )
            if actors[event.actor].kind == "ASYNC_ENGINE":
                raise PGIRError(
                    "INVALID_GROUP_OWNER",
                    "an ASYNC_ENGINE actor cannot own a program-visible TMA store group",
                    {"event": event.id, "actor": event.actor},
                )
            if event.type == "TmaStoreIssue" and (
                regions[event.source_region].address_space != "SHARED"
                or regions[event.destination_region].address_space != "GLOBAL"
            ):
                raise PGIRError(
                    "UNSUPPORTED_TMA_STORE_REGION",
                    "TMA store requires an explicit shared source and global destination",
                    {"event": event.id},
                )
            if (
                event.type == "TmaStoreWaitGroup"
                and event.wait_group_count >= resource.capacity
            ):
                raise PGIRError(
                    "WAIT_GROUP_COUNT_OUT_OF_RANGE",
                    "TMA store wait_group count must be smaller than the group capacity",
                    {"event": event.id, "wait_group_count": event.wait_group_count, "capacity": resource.capacity},
                )
        elif event.type in {"ChannelCreate", "ProducerAcquire", "ProducerCommit", "ConsumerWait", "ConsumerRelease"}:
            resource = resources[event.resource]
            if resource.kind != "channel":
                raise PGIRError(
                    "RESOURCE_KIND_MISMATCH",
                    "channel event must reference a channel resource",
                    {"event": event.id, "resource": event.resource, "kind": resource.kind},
                )
            if event.type != "ChannelCreate" and event.slot >= resource.capacity:
                raise PGIRError(
                    "CHANNEL_SLOT_OUT_OF_RANGE",
                    "channel event slot exceeds the declared capacity",
                    {"event": event.id, "slot": event.slot, "capacity": resource.capacity},
                )
        elif event.type == "ProxyFence":
            assert event.visibility_mask is not None
            if event.visibility_mask.dsmem or any(
                regions[region_id].address_space == "DSMEM"
                for region_id in event.regions
            ):
                raise PGIRError(
                    "UNSUPPORTED_DSMEM_PROXY",
                    "DSMEM proxy-fence semantics are outside the supported PGIR v2 subset",
                    {"event": event.id},
                )
            if event.from_domain != "GENERIC" or event.to_domain not in {"ASYNC_COPY", "TMA", "TENSOR_CORE"}:
                raise PGIRError(
                    "UNSUPPORTED_PROXY_DIRECTION",
                    "supported ProxyFence directions are GENERIC to ASYNC_COPY, TMA, or TENSOR_CORE",
                    {"event": event.id, "from_domain": event.from_domain, "to_domain": event.to_domain},
                )
            if event.scope != "CTA":
                raise PGIRError(
                    "UNSUPPORTED_PROXY_SCOPE",
                    "only CTA-scope ProxyFence semantics are supported in PGIR v1",
                    {"event": event.id, "scope": event.scope},
                )
            mask = event.visibility_mask
            if mask.control_only:
                raise PGIRError(
                    "INVALID_PROXY_VISIBILITY_MASK",
                    "ProxyFence must publish at least one data visibility domain",
                    {"event": event.id, "visibility_mask": mask.artifact()},
                )
            uncovered = sorted(
                region_id for region_id in event.regions
                if (
                    regions[region_id].address_space == "SHARED" and not mask.shared
                    or regions[region_id].address_space == "GLOBAL" and not mask.global_write
                    or regions[region_id].address_space == "TENSOR" and not mask.tensor_write
                    or regions[region_id].address_space == "REGISTER"
                )
            )
            if uncovered:
                raise PGIRError(
                    "PROXY_REGION_MASK_MISMATCH",
                    "ProxyFence visibility mask does not publish every named region",
                    {"event": event.id, "regions": uncovered},
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
    engine_completion_streams(module)

    commit_streams: dict[tuple[str, str], list[Event]] = {}
    for event in module.events:
        if event.type in {"CpAsyncCommitGroup", "WgmmaCommitGroup", "TmaStoreCommitGroup"}:
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

    for actor, trace in sorted(module.events_by_actor.items()):
        staged_typed: dict[str, int] = {}
        for event in trace:
            if event.type in {"WgmmaIssue", "TmaStoreIssue"}:
                staged_typed[event.resource] = staged_typed.get(event.resource, 0) + 1
            elif event.type in {"WgmmaCommitGroup", "TmaStoreCommitGroup"}:
                if staged_typed.get(event.resource, 0) == 0:
                    protocol = resources[event.resource].protocol
                    raise PGIRError(
                        f"UNSUPPORTED_EMPTY_{protocol}_GROUP",
                        f"empty {protocol} commit-group semantics are outside the supported NVIDIA-valid subset",
                        {"event": event.id, "actor": actor, "resource": event.resource},
                    )
                staged_typed[event.resource] = 0

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

    if schema_version == "pgir-v2":
        cluster_resources = {
            resource.id
            for resource in module.resources
            if resource.kind == "cluster_barrier"
        }
        for resource_id in sorted(cluster_resources):
            resource = resources[resource_id]
            collective_events = [
                event
                for event in module.events
                if event.resource == resource_id
                and event.type in {"BarrierArrive", "BarrierObserve"}
            ]
            invocations = sorted({event.invocation_id for event in collective_events})
            if invocations and invocations != list(range(len(invocations))):
                raise PGIRError(
                    "CLUSTER_INVOCATION_SEQUENCE_GAP",
                    "cluster barrier invocation IDs must be gap-free and start at zero",
                    {"resource": resource_id, "invocations": invocations},
                )
            for invocation in invocations:
                arrivals = sorted(
                    event.actor
                    for event in collective_events
                    if event.invocation_id == invocation
                    and event.type == "BarrierArrive"
                    and event.predicate
                )
                observes = sorted(
                    event.actor
                    for event in collective_events
                    if event.invocation_id == invocation
                    and event.type == "BarrierObserve"
                )
                expected = sorted(resource.participants)
                if arrivals != expected or observes != expected:
                    raise PGIRError(
                        "CLUSTER_BARRIER_PARTICIPATION_INCOMPLETE",
                        "every cluster barrier participant must arrive and observe each invocation exactly once",
                        {
                            "resource": resource_id,
                            "invocation": invocation,
                            "expected": expected,
                            "arrivals": arrivals,
                            "observes": observes,
                        },
                    )
        for actor, trace in module.events_by_actor.items():
            observe_positions = [
                index
                for index, event in enumerate(trace)
                if event.type == "BarrierObserve" and event.resource in cluster_resources
            ]
            for index, event in enumerate(trace):
                if event.type not in {"MemRead", "MemWrite"}:
                    continue
                assert event.region is not None
                if regions[event.region].address_space != "DSMEM":
                    continue
                if (
                    not any(position < index for position in observe_positions)
                    or not any(position > index for position in observe_positions)
                ):
                    raise PGIRError(
                        "DSMEM_LIFETIME_UNGUARDED",
                        "DSMEM access requires cluster synchronization before access and before CTA exit",
                        {"event": event.id, "actor": actor},
                    )

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
    } | {
        stable_group_completion_id("WGMMA", event.group_owner, event.group_seq, completion_kind)
        for event in module.events if event.type == "WgmmaCommitGroup"
        for completion_kind in ("OPERANDS_RELEASED", "RESULT_READY")
    } | {
        stable_group_id("WGMMA", event.group_owner, event.group_seq)
        for event in module.events if event.type == "WgmmaCommitGroup"
    } | {
        stable_group_completion_id("TMA_STORE", event.group_owner, event.group_seq, completion_kind)
        for event in module.events if event.type == "TmaStoreCommitGroup"
        for completion_kind in ("SOURCE_RELEASED", "GLOBAL_COMPLETE")
    } | {
        stable_group_id("TMA_STORE", event.group_owner, event.group_seq)
        for event in module.events if event.type == "TmaStoreCommitGroup"
    } | {
        stable_channel_handoff_id(event.resource, event.slot, event.logical_phase)
        for event in module.events if event.type in {"ProducerCommit", "ConsumerWait"}
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
    actors = tuple(Actor(
        item["id"],
        item["kind"],
        tuple(sorted(item["participants"])),
        cta_rank=item.get("cta_rank"),
    ) for item in data["actors"])
    resources = tuple(Resource(
        item["id"],
        item["kind"],
        tuple(sorted(item["participants"])),
        capacity=item.get("capacity"),
        protocol=item.get("protocol"),
        scope=item.get("scope"),
        visibility_mask=VisibilityMask.from_mapping(item["visibility_mask"]) if "visibility_mask" in item else None,
    ) for item in data["resources"])
    regions = tuple(Region(
        item["id"],
        item["allocation_id"],
        item["address_space"],
        item["offset"],
        item["extent"],
        owner_cta_rank=item.get("owner_cta_rank"),
    ) for item in data["regions"])
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
    grouped = _bucket_events_by_actor(actors, events)
    module = Module(
        raw=data,
        actors=actors,
        resources=resources,
        regions=regions,
        events=events,
        events_by_actor=grouped,
        _event_map={event.id: event for event in events},
        _resource_map={resource.id: resource for resource in resources},
        _region_map={region.id: region for region in regions},
        _actor_map={actor.id: actor for actor in actors},
    )
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
