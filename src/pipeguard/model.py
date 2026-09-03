from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

IMPLEMENTED_EVENTS = frozenset({
    "MemRead", "MemWrite", "MBarrierInit", "MBarrierInvalidate",
    "MBarrierArrive", "MBarrierExpectTx", "MBarrierArriveExpectTx",
    "MBarrierWaitParity", "TmaLoadIssue",
})
KNOWN_UNSUPPORTED_EVENTS = frozenset({
    "AtomicReadModifyWrite", "Fork", "Join", "BarrierArrive", "BarrierRelease",
    "BarrierObserve", "MBarrierTxComplete", "MBarrierPhaseComplete",
    "CpAsyncIssue", "CpAsyncCommitGroup", "CpAsyncGroupComplete", "CpAsyncWaitGroup",
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


@dataclass(frozen=True, slots=True)
class Region:
    id: str
    allocation_id: str
    address_space: str
    offset: int
    extent: int


@dataclass(frozen=True, slots=True)
class ParityEvidence:
    kind: str
    detail: str


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
    generation: int | None = None
    raw_parity: int | None = None
    parity_evidence: ParityEvidence | None = None
    predicate: bool = True


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


def _schema() -> dict[str, Any]:
    return json.loads(files("pipeguard.specs").joinpath("pgir-v1.schema.json").read_text())


def stable_completion_id(issue_id: str, generation: int) -> str:
    return f"completion:tma-load:{issue_id}:g{generation:08d}"


def stable_phase_id(resource_id: str, generation: int) -> str:
    return f"completion:mbarrier-phase:{resource_id}:g{generation:08d}"


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
    for index, region in enumerate(data.get("regions", [])):
        for name in ("offset", "extent"):
            if name in region:
                _strict_integer(region[name], f"/regions/{index}/{name}")
    for index, event in enumerate(data.get("events", [])):
        for name in ("seq", "count", "bytes", "generation", "raw_parity"):
            if name in event:
                _strict_integer(event[name], f"/events/{index}/{name}")


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

    required_fields = {
        "MemRead": ("region",), "MemWrite": ("region",),
        "MBarrierInit": ("resource", "count"), "MBarrierInvalidate": ("resource",),
        "MBarrierArrive": ("resource", "count", "generation"),
        "MBarrierExpectTx": ("resource", "bytes", "generation"),
        "MBarrierArriveExpectTx": ("resource", "count", "bytes", "generation"),
        "MBarrierWaitParity": ("resource", "generation", "raw_parity", "parity_evidence"),
        "TmaLoadIssue": ("resource", "source_region", "destination_region", "bytes", "generation"),
    }
    common = {"id", "type", "actor", "seq", "depends_on"}
    specific = {
        "MemRead": {"region"}, "MemWrite": {"region"},
        "MBarrierInit": {"resource", "count"}, "MBarrierInvalidate": {"resource"},
        "MBarrierArrive": {"resource", "count", "generation", "predicate"},
        "MBarrierExpectTx": {"resource", "bytes", "generation"},
        "MBarrierArriveExpectTx": {"resource", "count", "bytes", "generation", "predicate"},
        "MBarrierWaitParity": {"resource", "generation", "raw_parity", "parity_evidence"},
        "TmaLoadIssue": {"resource", "source_region", "destination_region", "bytes", "generation"},
    }
    raw_events = {item["id"]: item for item in module.raw["events"]}
    for event in module.events:
        if event.id.startswith("completion:"):
            raise PGIRError("RESERVED_ID_NAMESPACE", "real event ID uses the synthetic completion namespace", {"event": event.id})
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
        referenced_regions = [value for value in (event.region, event.source_region, event.destination_region) if value is not None]
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
        if event.type == "TmaLoadIssue" and (
            regions[event.source_region].address_space != "GLOBAL"
            or regions[event.destination_region].address_space != "SHARED"
        ):
            raise PGIRError("UNSUPPORTED_TMA_REGION", "TMA load requires an explicit global source and shared destination", {"event": event.id})

    synthetic_ids = {
        stable_completion_id(event.id, event.generation)
        for event in module.events if event.type == "TmaLoadIssue"
    } | {
        stable_phase_id(event.resource, event.generation)
        for event in module.events if event.resource is not None and event.generation is not None
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
    resources = tuple(Resource(item["id"], item["kind"], tuple(sorted(item["participants"]))) for item in data["resources"])
    regions = tuple(Region(item["id"], item["allocation_id"], item["address_space"], item["offset"], item["extent"]) for item in data["regions"])
    events = tuple(Event(
        id=item["id"], type=item["type"], actor=item["actor"], seq=item["seq"],
        depends_on=tuple(sorted(item.get("depends_on", []))), resource=item.get("resource"),
        region=item.get("region"), source_region=item.get("source_region"), destination_region=item.get("destination_region"),
        count=item.get("count"), bytes=item.get("bytes"), generation=item.get("generation"), raw_parity=item.get("raw_parity"),
        parity_evidence=ParityEvidence(**item["parity_evidence"]) if "parity_evidence" in item else None,
        predicate=item.get("predicate", True),
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
    normalized["unsupported_constructs"] = sorted(normalized["unsupported_constructs"])
    return normalized
