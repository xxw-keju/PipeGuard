from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from .model import (
    IMPLEMENTED_EVENTS,
    Event,
    Module,
    PGIRError,
    event_from_dict,
    module_from_dict,
)


PARAMETRIC_SCHEMA_VERSION = "pgir-parametric-v1"
EXECUTION_SCOPES = frozenset({"INSTANCE", "ELECTED", "COLLECTIVE"})
TRANSITION_MULTIPLICITIES = frozenset({"PER_INSTANCE", "ONCE"})
RENDEZVOUS_MODES = frozenset({"NONE", "ALL_PROGRAM_INSTANCES"})
SOURCE_RESERVED_PREFIXES = ("completion:", "group:")
EVIDENCE_CLAIMS = {
    "participant_mapping": "RESOLVED",
    "lane_ownership": "RESOLVED",
    "predicate_mapping": "RESOLVED",
    "alias": "EXACT",
}
EVENT_FIELD_NAMES = frozenset({
    "depends_on",
    "resource",
    "region",
    "source_region",
    "destination_region",
    "count",
    "bytes",
    "lifecycle_epoch",
    "generation",
    "layout",
    "raw_parity",
    "parity_evidence",
    "predicate",
    "invocation_id",
    "group_owner",
    "group_seq",
    "wait_group_count",
    "completion_kind",
    "wait_mode",
    "engine_id",
    "engine_seq",
    "slot",
    "logical_phase",
    "from_domain",
    "to_domain",
    "scope",
    "regions",
    "transfers",
    "visibility_mask",
})


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    status: str
    method: str
    source: str


@dataclass(frozen=True, slots=True)
class ProgramStreamClass:
    id: str
    instances: tuple[int, ...]

    def actor_id(self, instance: int) -> str:
        if instance not in self.instances:
            raise PGIRError(
                "PARAMETRIC_INSTANCE_OUT_OF_RANGE",
                "logical instance is outside its program-stream class",
                {"stream": self.id, "instance": instance},
            )
        return stable_stream_instance_id(self.id, instance)


@dataclass(frozen=True, slots=True)
class TemplateDependency:
    template: str
    relation: str


@dataclass(frozen=True, slots=True)
class EventTemplate:
    id: str
    type: str
    stream: str
    order: int
    index: int
    program_instances: tuple[int, ...]
    executors: tuple[int, ...]
    transition: str
    rendezvous: str
    owner: str | None
    fields: dict[str, Any]
    instance_fields: dict[str, tuple[Any, ...]]
    dependencies: tuple[TemplateDependency, ...]
    instance_positions: dict[int, int]

    @property
    def instances(self) -> tuple[int, ...]:
        """Compatibility name for code that consumes logical occurrences."""

        return self.program_instances

    @property
    def execution_scope(self) -> str:
        """Derive the legacy scope without conflating its four meanings."""

        if self.transition == "PER_INSTANCE":
            return "INSTANCE"
        if self.rendezvous == "ALL_PROGRAM_INSTANCES":
            return "COLLECTIVE"
        return "ELECTED"

    def field(self, name: str, instance: int) -> Any:
        values = self.instance_fields.get(name)
        if values is not None:
            try:
                position = self.instance_positions[instance]
            except KeyError as error:
                raise PGIRError(
                    "PARAMETRIC_INSTANCE_OUT_OF_RANGE",
                    "event field requested for an inactive logical instance",
                    {"template": self.id, "instance": instance, "field": name},
                ) from error
            return values[position]
        return self.fields.get(name)

    def memory_regions(self, instance: int) -> tuple[str, ...]:
        regions = self.field("regions", instance)
        if regions is not None:
            return tuple(regions)
        region = self.field("region", instance)
        return () if region is None else (region,)

    def logical_id(self, instance: int) -> str:
        if self.rendezvous == "ALL_PROGRAM_INSTANCES":
            return stable_collective_event_id(self.id)
        return stable_event_instance_id(self.id, instance)


@dataclass(frozen=True, slots=True)
class ParametricModule:
    raw: dict[str, Any]
    stream_classes: tuple[ProgramStreamClass, ...]
    templates: tuple[EventTemplate, ...]
    evidence: dict[str, EvidenceClaim]
    lane_trace_offsets: tuple[int, ...]
    lane_trace_template_ids: tuple[int, ...]
    lane_ids: tuple[int, ...]
    template_indices: dict[str, int]

    @property
    def logical_stream_count(self) -> int:
        return sum(len(stream.instances) for stream in self.stream_classes)

    @property
    def template_count(self) -> int:
        return len(self.templates)

    @property
    def logical_event_count(self) -> int:
        return sum(len(template.program_instances) for template in self.templates)

    def lane_trace(self, lane: int) -> tuple[int, ...]:
        try:
            position = self.lane_ids.index(lane)
        except ValueError as error:
            raise PGIRError(
                "PARAMETRIC_INSTANCE_OUT_OF_RANGE",
                "lane trace requested outside the selected CTA",
                {"lane": lane},
            ) from error
        begin = self.lane_trace_offsets[position]
        end = self.lane_trace_offsets[position + 1]
        return self.lane_trace_template_ids[begin:end]

    def execution_view(self) -> "ParametricExecutionView":
        return ParametricExecutionView(self)

    def expand_reference(self) -> Module:
        """Deterministically expand compact source events into concrete PGIR.

        This is the semantic oracle, not the default large-input execution
        strategy. INSTANCE/ELECTED templates become one source event per
        logical stream. A COLLECTIVE template becomes per-stream rendezvous
        points plus exactly one owner transition on the lowest participating
        stream, so the protocol FSM is never cloned per participant.
        """

        stream_map = {stream.id: stream for stream in self.stream_classes}
        template_map = {template.id: template for template in self.templates}
        logical_ids = _logical_event_ids(self.templates)

        actors = deepcopy(self.raw["owners"])
        actor_ids = {actor["id"] for actor in actors}
        for actor in actors:
            actor["role"] = (
                "ASYNC_ENGINE"
                if actor["kind"] == "ASYNC_ENGINE"
                else "EXECUTION_OWNER"
            )
        for stream in self.stream_classes:
            for instance in stream.instances:
                actor_id = stream.actor_id(instance)
                if actor_id in actor_ids:
                    raise PGIRError(
                        "DUPLICATE_ID",
                        f"duplicate actor id: {actor_id}",
                        {"id": actor_id, "kind": "actor"},
                    )
                actor_ids.add(actor_id)
                actors.append({
                    "id": actor_id,
                    "kind": "LANE",
                    "participants": [instance],
                    "role": "PROGRAM_STREAM",
                })

        resources = [
            _expand_resource(resource, stream_map)
            for resource in self.raw["resources"]
        ]
        traces: dict[str, list[dict[str, Any]]] = {
            stream.actor_id(instance): []
            for stream in self.stream_classes
            for instance in stream.instances
        }

        for template in sorted(self.templates, key=lambda item: (item.order, item.id)):
            stream = stream_map[template.stream]
            if template.execution_scope == "COLLECTIVE":
                collective_id = logical_ids[template.id][template.instances[0]]
                enter_ids = {
                    instance: stable_collective_point_id(template.id, instance, "enter")
                    for instance in template.instances
                }
                for instance in template.instances:
                    actor = stream.actor_id(instance)
                    traces[actor].append({
                        "id": enter_ids[instance],
                        "type": "ProgramPoint",
                        "actor": actor,
                        "depends_on": _dependency_ids(
                            template,
                            instance,
                            template_map,
                            logical_ids,
                        ),
                    })
                leader = min(template.executors)
                leader_actor = stream.actor_id(leader)
                event = _expanded_source_event(
                    template,
                    leader,
                    leader_actor,
                    collective_id,
                )
                event["depends_on"] = sorted(enter_ids.values())
                traces[leader_actor].append(event)
                for instance in template.instances:
                    actor = stream.actor_id(instance)
                    traces[actor].append({
                        "id": stable_collective_point_id(template.id, instance, "exit"),
                        "type": "ProgramPoint",
                        "actor": actor,
                        "depends_on": [collective_id],
                    })
                continue

            if template.transition == "ONCE":
                executor = min(template.executors)
                for instance in template.program_instances:
                    actor = stream.actor_id(instance)
                    dependencies = _dependency_ids(
                        template,
                        instance,
                        template_map,
                        logical_ids,
                    )
                    if instance == executor:
                        event = _expanded_source_event(
                            template,
                            instance,
                            actor,
                            logical_ids[template.id][instance],
                        )
                    else:
                        event = {
                            "id": stable_collective_point_id(
                                template.id, instance, "pass"
                            ),
                            "type": "ProgramPoint",
                            "actor": actor,
                        }
                    if dependencies:
                        event["depends_on"] = dependencies
                    traces[actor].append(event)
                continue

            for instance in template.program_instances:
                actor = stream.actor_id(instance)
                identifier = logical_ids[template.id][instance]
                event = _expanded_source_event(
                    template,
                    instance,
                    actor,
                    identifier,
                )
                dependencies = _dependency_ids(
                    template,
                    instance,
                    template_map,
                    logical_ids,
                )
                if dependencies:
                    event["depends_on"] = dependencies
                traces[actor].append(event)

        events: list[dict[str, Any]] = []
        for actor in sorted(traces):
            for sequence, event in enumerate(traces[actor]):
                event["seq"] = sequence
                events.append(event)

        expanded = {
            "schema_version": "pgir-v1",
            "producer": deepcopy(self.raw["producer"]),
            "target": deepcopy(self.raw["target"]),
            "assumptions": deepcopy(self.raw["assumptions"]),
            "control_structure": deepcopy(self.raw["control_structure"]),
            "actors": actors,
            "resources": resources,
            "regions": deepcopy(self.raw["regions"]),
            "events": events,
            "frontend_provenance": deepcopy(self.raw["frontend_provenance"]),
            "unsupported_constructs": deepcopy(self.raw["unsupported_constructs"]),
        }
        module = module_from_dict(expanded)
        return replace(module, _parametric=self)


def stable_stream_instance_id(stream_id: str, instance: int) -> str:
    return f"stream:{len(stream_id.encode('utf-8')):08d}:{stream_id}:i{instance:08d}"


def stable_event_instance_id(template_id: str, instance: int) -> str:
    return f"source:event:{len(template_id.encode('utf-8')):08d}:{template_id}:i{instance:08d}"


def stable_collective_event_id(template_id: str) -> str:
    return f"source:collective:{len(template_id.encode('utf-8')):08d}:{template_id}"


def stable_collective_point_id(template_id: str, instance: int, position: str) -> str:
    return (
        f"source:rendezvous:{position}:{len(template_id.encode('utf-8')):08d}:"
        f"{template_id}:i{instance:08d}"
    )


_SOURCE = 0
_COLLECTIVE_ENTER = 1
_COLLECTIVE_SOURCE = 2
_COLLECTIVE_EXIT = 3


def _compact_dependency_ids(
    template: EventTemplate,
    instance: int,
    template_map: dict[str, EventTemplate],
) -> tuple[str, ...]:
    dependencies: set[str] = set()
    for dependency in template.dependencies:
        target = template_map[dependency.template]
        if dependency.relation == "SAME_INSTANCE":
            if instance not in target.instances:
                raise PGIRError(
                    "PARAMETRIC_DEPENDENCY_INSTANCE_MISMATCH",
                    "SAME_INSTANCE dependency has no event for this logical stream",
                    {
                        "template": template.id,
                        "instance": instance,
                        "depends_on": target.id,
                    },
                )
            dependencies.add(target.logical_id(instance))
        elif dependency.relation == "ALL_INSTANCES":
            dependencies.update(target.logical_id(item) for item in target.instances)
        else:
            dependencies.add(target.logical_id(min(target.executors)))
    return tuple(sorted(dependencies))


class ParametricActorTrace:
    __slots__ = ("view", "stream", "instance", "actor", "codes")

    def __init__(
        self,
        view: "ParametricExecutionView",
        stream: ProgramStreamClass,
        instance: int,
    ) -> None:
        self.view = view
        self.stream = stream
        self.instance = instance
        self.actor = stream.actor_id(instance)
        codes = array("I")
        for index, template in enumerate(view.module.templates):
            if template.stream != stream.id or instance not in template.program_instances:
                continue
            if template.transition == "PER_INSTANCE":
                codes.append(index << 2 | _SOURCE)
                continue
            if template.rendezvous == "ALL_PROGRAM_INSTANCES":
                codes.append(index << 2 | _COLLECTIVE_ENTER)
                if instance == min(template.executors):
                    codes.append(index << 2 | _COLLECTIVE_SOURCE)
                codes.append(index << 2 | _COLLECTIVE_EXIT)
                continue
            codes.append(
                index << 2
                | (_SOURCE if instance == min(template.executors) else _COLLECTIVE_ENTER)
            )
        self.codes = codes

    def __len__(self) -> int:
        return len(self.codes)

    def __iter__(self) -> Iterator[Event]:
        for sequence in range(len(self.codes)):
            yield self[sequence]

    def __getitem__(self, sequence: int | slice) -> Event | tuple[Event, ...]:
        if isinstance(sequence, slice):
            return tuple(self[index] for index in range(*sequence.indices(len(self))))
        if sequence < 0:
            sequence += len(self.codes)
        if not 0 <= sequence < len(self.codes):
            raise IndexError(sequence)
        return self.view.event_from_code(
            self.actor,
            self.instance,
            int(self.codes[sequence]),
            sequence,
        )

    def sequence_of(self, code: int) -> int:
        position = bisect_left(self.codes, code)
        if position >= len(self.codes) or self.codes[position] != code:
            raise KeyError(code)
        return position


class ParametricEventMap(Mapping[str, Event]):
    def __init__(self, view: "ParametricExecutionView") -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view.events)

    def __iter__(self) -> Iterator[str]:
        return (event.id for event in self.view.events)

    @staticmethod
    def _template_and_instance(identifier: str, prefix: str) -> tuple[str, int]:
        remainder = identifier[len(prefix):]
        if len(remainder) < 10 or remainder[8] != ":":
            raise KeyError(identifier)
        template, marker, raw_instance = remainder[9:].rpartition(":i")
        if not marker:
            raise KeyError(identifier)
        try:
            instance = int(raw_instance)
        except ValueError as error:
            raise KeyError(identifier) from error
        return template, instance

    def decode(self, identifier: str) -> tuple[EventTemplate, int, int]:
        if identifier.startswith("source:event:"):
            template_id, instance = self._template_and_instance(
                identifier, "source:event:"
            )
            kind = _SOURCE
        elif identifier.startswith("source:collective:"):
            remainder = identifier[len("source:collective:"):]
            if len(remainder) < 9 or remainder[8] != ":":
                raise KeyError(identifier)
            template_id = remainder[9:]
            template = self.view.template_map.get(template_id)
            if template is None or template.execution_scope != "COLLECTIVE":
                raise KeyError(identifier)
            instance = min(template.instances)
            kind = _COLLECTIVE_SOURCE
        else:
            for position, prefix in (
                (_COLLECTIVE_ENTER, "source:rendezvous:enter:"),
                (_COLLECTIVE_EXIT, "source:rendezvous:exit:"),
            ):
                if identifier.startswith(prefix):
                    template_id, instance = self._template_and_instance(
                        identifier, prefix
                    )
                    kind = position
                    break
            else:
                raise KeyError(identifier)
        template = self.view.template_map.get(template_id)
        if template is None or instance not in self.view.template_instance_sets[template_id]:
            raise KeyError(identifier)
        return template, instance, kind

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str):
            return False
        try:
            self.decode(identifier)
        except KeyError:
            return False
        return True

    def __getitem__(self, identifier: str) -> Event:
        template, instance, kind = self.decode(identifier)
        return self.view.event_for(template.id, instance, kind)



class ParametricEvents:
    def __init__(self, view: "ParametricExecutionView") -> None:
        self.view = view
        self._length = sum(len(trace) for trace in view.events_by_actor.values())

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Event]:
        for actor in sorted(self.view.events_by_actor):
            yield from self.view.events_by_actor[actor]


class ParametricExecutionView:
    """Non-resident logical-event view consumed by the protocol engine.

    Actor traces store four-byte template codes. Event objects and stable IDs are
    reconstructed only for the current frontier, PDG query, or diagnostic.
    """

    implicit_program_completion = True
    record_trace = False

    def __init__(self, module: ParametricModule) -> None:
        self.parametric = module
        self.module = module
        self.template_map = {template.id: template for template in module.templates}
        self.template_indices = {
            template.id: index for index, template in enumerate(module.templates)
        }
        self.template_instance_sets = {
            template.id: frozenset(template.instances)
            for template in module.templates
        }
        self.template_instance_positions = {
            template.id: {
                instance: position
                for position, instance in enumerate(template.instances)
            }
            for template in module.templates
        }
        self.stream_map = {
            stream.id: stream for stream in module.stream_classes
        }
        owners = deepcopy(module.raw["owners"])
        for owner in owners:
            owner["role"] = (
                "ASYNC_ENGINE"
                if owner["kind"] == "ASYNC_ENGINE"
                else "EXECUTION_OWNER"
            )
        actors = list(owners)
        for stream in module.stream_classes:
            actors.extend({
                "id": stream.actor_id(instance),
                "kind": "LANE",
                "participants": [instance],
                "role": "PROGRAM_STREAM",
            } for instance in stream.instances)
        raw = {
            "schema_version": "pgir-v1",
            "producer": deepcopy(module.raw["producer"]),
            "target": deepcopy(module.raw["target"]),
            "assumptions": deepcopy(module.raw["assumptions"]),
            "control_structure": deepcopy(module.raw["control_structure"]),
            "actors": actors,
            "resources": [
                _expand_resource(resource, self.stream_map)
                for resource in module.raw["resources"]
            ],
            "regions": deepcopy(module.raw["regions"]),
            "events": [],
            "frontend_provenance": deepcopy(module.raw["frontend_provenance"]),
            "unsupported_constructs": deepcopy(module.raw["unsupported_constructs"]),
        }
        base = module_from_dict(raw)
        self.raw = raw
        self.actors = base.actors
        self.resources = base.resources
        self.regions = base.regions
        self._resource_map = base.resource_map
        self._region_map = base.region_map
        self._actor_map = base.actor_map
        self.events_by_actor: dict[str, ParametricActorTrace | tuple[Event, ...]] = {
            actor.id: () for actor in self.actors
        }
        self.actor_instances: dict[str, tuple[ProgramStreamClass, int]] = {}
        for stream in module.stream_classes:
            for instance in stream.instances:
                actor = stream.actor_id(instance)
                self.actor_instances[actor] = (stream, instance)
                self.events_by_actor[actor] = ParametricActorTrace(
                    self, stream, instance
                )
        self.events = ParametricEvents(self)
        self._event_map = ParametricEventMap(self)

    @property
    def event_map(self) -> Mapping[str, Event]:
        return self._event_map

    @property
    def resource_map(self) -> dict[str, Any]:
        return self._resource_map

    @property
    def region_map(self) -> dict[str, Any]:
        return self._region_map

    @property
    def actor_map(self) -> dict[str, Any]:
        return self._actor_map

    @property
    def required_logical_waits(self) -> dict[tuple[str, int, int], frozenset[str]]:
        required: dict[tuple[str, int, int], set[str]] = {}
        for event in self.events:
            if event.type != "MBarrierWaitParity":
                continue
            assert event.resource is not None
            assert event.lifecycle_epoch is not None
            assert event.generation is not None
            required.setdefault(
                (event.resource, event.lifecycle_epoch, event.generation), set()
            ).add(event.id)
        return {key: frozenset(value) for key, value in required.items()}

    def _trace_for(self, template: EventTemplate, instance: int) -> ParametricActorTrace:
        stream = self.stream_map[template.stream]
        actor = stream.actor_id(instance)
        trace = self.events_by_actor[actor]
        assert isinstance(trace, ParametricActorTrace)
        return trace

    def event_for(self, template_id: str, instance: int, kind: int) -> Event:
        template = self.template_map.get(template_id)
        if template is None or instance not in template.instances:
            raise KeyError((template_id, instance))
        index = self.template_indices[template_id]
        code = index << 2 | kind
        trace = self._trace_for(template, instance)
        sequence = trace.sequence_of(code)
        return self.event_from_code(trace.actor, instance, code, sequence)

    def event_from_code(
        self,
        actor: str,
        instance: int,
        code: int,
        sequence: int,
    ) -> Event:
        template = self.module.templates[code >> 2]
        kind = code & 3
        if kind == _SOURCE:
            item = _expanded_source_event(
                template,
                instance,
                actor,
                template.logical_id(instance),
                position=self.template_instance_positions[template.id][instance],
            )
            item["depends_on"] = list(
                _compact_dependency_ids(template, instance, self.template_map)
            )
        elif kind == _COLLECTIVE_ENTER:
            item = {
                "id": stable_collective_point_id(template.id, instance, "enter"),
                "type": "ProgramPoint",
                "actor": actor,
                "depends_on": list(
                    _compact_dependency_ids(template, instance, self.template_map)
                ),
            }
        elif kind == _COLLECTIVE_SOURCE:
            leader = min(template.executors)
            if instance != leader:
                raise KeyError((template.id, instance, "collective-source"))
            item = _expanded_source_event(
                template,
                leader,
                actor,
                stable_collective_event_id(template.id),
                position=self.template_instance_positions[template.id][leader],
            )
            item["depends_on"] = [
                stable_collective_point_id(template.id, item_instance, "enter")
                for item_instance in template.instances
            ]
        elif kind == _COLLECTIVE_EXIT:
            item = {
                "id": stable_collective_point_id(template.id, instance, "exit"),
                "type": "ProgramPoint",
                "actor": actor,
                "depends_on": [stable_collective_event_id(template.id)],
            }
        else:
            raise KeyError(code)
        item["seq"] = sequence
        return event_from_dict(item)

    def program_event_position(self, identifier: str) -> tuple[str, int] | None:
        try:
            template, instance, kind = self._event_map.decode(identifier)
        except KeyError:
            return None
        stream = self.stream_map[template.stream]
        actor = stream.actor_id(instance)
        trace = self.events_by_actor[actor]
        assert isinstance(trace, ParametricActorTrace)
        code = self.template_indices[template.id] << 2 | kind
        try:
            sequence = trace.sequence_of(code)
        except KeyError:
            return None
        return actor, sequence


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "parameterized PGIR value must be an object",
            {"location": location},
        )
    return value


def _require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise PGIRError(
            "PARAMETRIC_FIELDS_MISSING",
            "parameterized PGIR object is missing required fields",
            {"location": location, "fields": missing},
        )
    if unknown:
        raise PGIRError(
            "PARAMETRIC_FIELDS_UNKNOWN",
            "parameterized PGIR object contains unknown fields",
            {"location": location, "fields": unknown},
        )


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "parameterized PGIR field must be a nonempty string",
            {"location": location},
        )
    return value


def _require_nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PGIRError(
            "STRICT_INTEGER_REQUIRED",
            "JSON integer must decode as a nonnegative Python int",
            {"path": location, "value": value},
        )
    return value


def _require_instances(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "logical instance set must be a nonempty array",
            {"location": location},
        )
    instances = tuple(
        _require_nonnegative_integer(instance, f"{location}/{index}")
        for index, instance in enumerate(value)
    )
    if len(instances) != len(set(instances)):
        raise PGIRError(
            "PARAMETRIC_DUPLICATE_INSTANCE",
            "logical instance set contains duplicates",
            {"location": location, "instances": list(instances)},
        )
    return tuple(sorted(instances))


def _parse_evidence(raw: Any) -> dict[str, EvidenceClaim]:
    evidence = _require_object(raw, "/frontend_evidence")
    _require_exact_keys(
        evidence,
        set(EVIDENCE_CLAIMS),
        set(),
        "/frontend_evidence",
    )
    parsed: dict[str, EvidenceClaim] = {}
    for name, expected in EVIDENCE_CLAIMS.items():
        claim = _require_object(evidence[name], f"/frontend_evidence/{name}")
        _require_exact_keys(
            claim,
            {"status", "method", "source"},
            set(),
            f"/frontend_evidence/{name}",
        )
        status = _require_string(
            claim["status"], f"/frontend_evidence/{name}/status"
        )
        method = _require_string(
            claim["method"], f"/frontend_evidence/{name}/method"
        )
        source = _require_string(
            claim["source"], f"/frontend_evidence/{name}/source"
        )
        if status != expected:
            raise PGIRError(
                "PARAMETRIC_EVIDENCE_UNRESOLVED",
                "frontend certainty claim lacks the required structured evidence status",
                {"claim": name, "status": status, "required": expected},
            )
        parsed[name] = EvidenceClaim(status, method, source)
    return parsed


def _parse_streams(raw_streams: Any, cta_threads: int) -> tuple[ProgramStreamClass, ...]:
    if not isinstance(raw_streams, list) or not raw_streams:
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "parameterized PGIR requires at least one program-stream class",
            {"location": "/program_streams"},
        )
    streams: list[ProgramStreamClass] = []
    identifiers: set[str] = set()
    represented_instances: set[int] = set()
    for index, raw in enumerate(raw_streams):
        location = f"/program_streams/{index}"
        item = _require_object(raw, location)
        _require_exact_keys(item, {"id", "kind", "instances"}, set(), location)
        identifier = _require_string(item["id"], f"{location}/id")
        if identifier in identifiers:
            raise PGIRError(
                "DUPLICATE_ID",
                f"duplicate program stream id: {identifier}",
                {"id": identifier, "kind": "program_stream"},
            )
        identifiers.add(identifier)
        if item["kind"] != "LANE":
            raise PGIRError(
                "PARAMETRIC_STREAM_KIND_UNSUPPORTED",
                "the initial parameterized profile requires logical lane streams",
                {"stream": identifier, "kind": item["kind"]},
            )
        instances = _require_instances(item["instances"], f"{location}/instances")
        invalid = [instance for instance in instances if instance >= cta_threads]
        if invalid:
            raise PGIRError(
                "PARTICIPANT_OUT_OF_RANGE",
                "logical stream instance exceeds CTA bounds",
                {"stream": identifier, "instances": invalid, "cta_threads": cta_threads},
            )
        overlap = sorted(represented_instances.intersection(instances))
        if overlap:
            raise PGIRError(
                "PARAMETRIC_STREAM_INSTANCE_OVERLAP",
                "one logical lane cannot belong to multiple program-stream classes",
                {"stream": identifier, "instances": overlap},
            )
        represented_instances.update(instances)
        streams.append(ProgramStreamClass(identifier, instances))
    return tuple(streams)


def _parse_dependencies(raw: Any, location: str) -> tuple[TemplateDependency, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "template dependencies must be an array",
            {"location": location},
        )
    result: list[TemplateDependency] = []
    for index, value in enumerate(raw):
        item_location = f"{location}/{index}"
        item = _require_object(value, item_location)
        _require_exact_keys(item, {"template", "relation"}, set(), item_location)
        template = _require_string(item["template"], f"{item_location}/template")
        relation = _require_string(item["relation"], f"{item_location}/relation")
        if relation not in {"SAME_INSTANCE", "ALL_INSTANCES", "COLLECTIVE"}:
            raise PGIRError(
                "PARAMETRIC_DEPENDENCY_RELATION_UNSUPPORTED",
                "template dependency relation is unsupported",
                {"location": item_location, "relation": relation},
            )
        result.append(TemplateDependency(template, relation))
    return tuple(result)


def _parse_templates(
    raw_templates: Any,
    streams: tuple[ProgramStreamClass, ...],
    owner_map: dict[str, dict[str, Any]],
) -> tuple[EventTemplate, ...]:
    if not isinstance(raw_templates, list):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "event_templates must be an array",
            {"location": "/event_templates"},
        )
    stream_map = {stream.id: stream for stream in streams}
    identifiers: set[str] = set()
    orders: set[int] = set()
    templates: list[EventTemplate] = []
    for index, raw in enumerate(raw_templates):
        location = f"/event_templates/{index}"
        item = _require_object(raw, location)
        common_required = {"id", "type", "stream", "order", "fields"}
        legacy_shape = "execution_scope" in item or "instances" in item
        if legacy_shape:
            _require_exact_keys(
                item,
                common_required | {"execution_scope", "instances"},
                {"owner", "instance_fields", "dependencies"},
                location,
            )
        else:
            _require_exact_keys(
                item,
                common_required
                | {"program_instances", "executors", "transition", "rendezvous"},
                {"owner", "instance_fields", "dependencies"},
                location,
            )
        identifier = _require_string(item["id"], f"{location}/id")
        if identifier.startswith(SOURCE_RESERVED_PREFIXES):
            raise PGIRError(
                "RESERVED_ID_NAMESPACE",
                "source template ID uses a synthetic completion namespace",
                {"template": identifier},
            )
        if identifier in identifiers:
            raise PGIRError(
                "DUPLICATE_ID",
                f"duplicate event template id: {identifier}",
                {"id": identifier, "kind": "event_template"},
            )
        identifiers.add(identifier)
        order = _require_nonnegative_integer(item["order"], f"{location}/order")
        if order in orders:
            raise PGIRError(
                "PARAMETRIC_TEMPLATE_ORDER_DUPLICATE",
                "event template order must be globally unique",
                {"order": order},
            )
        orders.add(order)
        stream_id = _require_string(item["stream"], f"{location}/stream")
        stream = stream_map.get(stream_id)
        if stream is None:
            raise PGIRError(
                "DANGLING_REFERENCE",
                "event template references an unknown program-stream class",
                {"template": identifier, "stream": stream_id},
            )
        if legacy_shape:
            program_instances = _require_instances(
                item["instances"], f"{location}/instances"
            )
            scope = _require_string(
                item["execution_scope"], f"{location}/execution_scope"
            )
            if scope not in EXECUTION_SCOPES:
                raise PGIRError(
                    "PARAMETRIC_EXECUTION_SCOPE_UNSUPPORTED",
                    "event template execution scope is unsupported",
                    {"template": identifier, "execution_scope": scope},
                )
            executors = program_instances
            transition = "PER_INSTANCE" if scope == "INSTANCE" else "ONCE"
            rendezvous = (
                "ALL_PROGRAM_INSTANCES" if scope == "COLLECTIVE" else "NONE"
            )
        else:
            program_instances = _require_instances(
                item["program_instances"], f"{location}/program_instances"
            )
            executors = _require_instances(
                item["executors"], f"{location}/executors"
            )
            transition = _require_string(
                item["transition"], f"{location}/transition"
            )
            rendezvous = _require_string(
                item["rendezvous"], f"{location}/rendezvous"
            )
            if transition not in TRANSITION_MULTIPLICITIES:
                raise PGIRError(
                    "PARAMETRIC_TRANSITION_UNSUPPORTED",
                    "template transition multiplicity is unsupported",
                    {"template": identifier, "transition": transition},
                )
            if rendezvous not in RENDEZVOUS_MODES:
                raise PGIRError(
                    "PARAMETRIC_RENDEZVOUS_UNSUPPORTED",
                    "template rendezvous mode is unsupported",
                    {"template": identifier, "rendezvous": rendezvous},
                )
            if transition == "PER_INSTANCE" and executors != program_instances:
                raise PGIRError(
                    "PARAMETRIC_EXECUTOR_MISMATCH",
                    "per-instance transitions require one executor per program instance",
                    {"template": identifier},
                )
            if transition == "ONCE" and not executors:
                raise PGIRError(
                    "PARAMETRIC_EXECUTOR_MISMATCH",
                    "once transitions require at least one statically exact executor",
                    {"template": identifier},
                )
            if rendezvous == "ALL_PROGRAM_INSTANCES" and transition != "ONCE":
                raise PGIRError(
                    "PARAMETRIC_RENDEZVOUS_INVALID",
                    "all-participant rendezvous requires one owner transition",
                    {"template": identifier},
                )
        unknown_instances = sorted(set(program_instances) - set(stream.instances))
        unknown_executors = sorted(set(executors) - set(stream.instances))
        if unknown_instances or unknown_executors:
            raise PGIRError(
                "PARAMETRIC_INSTANCE_OUT_OF_RANGE",
                "event template references lanes outside its stream class",
                {
                    "template": identifier,
                    "program_instances": unknown_instances,
                    "executors": unknown_executors,
                },
            )
        if not set(executors).issubset(program_instances):
            raise PGIRError(
                "PARAMETRIC_EXECUTOR_MISMATCH",
                "instruction executors must be program participants",
                {"template": identifier},
            )
        scope = (
            "INSTANCE"
            if transition == "PER_INSTANCE"
            else "COLLECTIVE"
            if rendezvous == "ALL_PROGRAM_INSTANCES"
            else "ELECTED"
        )
        event_type = _require_string(item["type"], f"{location}/type")
        owner = item.get("owner")
        if owner is not None:
            owner = _require_string(owner, f"{location}/owner")
            if owner not in owner_map:
                raise PGIRError(
                    "DANGLING_REFERENCE",
                    "event template references an unknown execution owner",
                    {"template": identifier, "owner": owner},
                )
        if event_type.startswith("Wgmma") and owner is None:
            raise PGIRError(
                "PARAMETRIC_COLLECTIVE_OWNER_REQUIRED",
                "WGMMA event template requires an explicit warpgroup owner",
                {"template": identifier},
            )
        if owner is not None:
            owner_lanes = set(owner_map[owner]["participants"])
            uncovered = sorted(set(program_instances) - owner_lanes)
            if uncovered:
                raise PGIRError(
                    "PARAMETRIC_OWNER_COVERAGE_MISMATCH",
                    "execution owner does not cover every participating logical stream",
                    {"template": identifier, "owner": owner, "instances": uncovered},
                )
        if event_type == "ProgramPoint":
            raise PGIRError(
                "PARAMETRIC_INTERNAL_EVENT_FORBIDDEN",
                "ProgramPoint is reserved for deterministic reference expansion",
                {"template": identifier},
            )
        if event_type not in IMPLEMENTED_EVENTS:
            code = (
                "UNSUPPORTED_ATOMIC"
                if event_type == "AtomicReadModifyWrite"
                else "UNSUPPORTED_PROTOCOL_OPERATION"
            )
            raise PGIRError(
                code,
                "event type is outside the frozen direct-checker subdomain",
                {"template": identifier, "type": event_type},
            )
        if event_type in {"MemRead", "MemWrite"} and scope != "INSTANCE":
            raise PGIRError(
                "PARAMETRIC_MEMORY_SCOPE_INVALID",
                "ordinary memory operations must execute as logical stream instances",
                {"template": identifier, "execution_scope": scope},
            )
        if event_type.startswith("Wgmma") and scope != "COLLECTIVE":
            raise PGIRError(
                "PARAMETRIC_WGMMA_SCOPE_INVALID",
                "WGMMA source operations require collective warpgroup execution",
                {"template": identifier, "execution_scope": scope},
            )
        if event_type == "TmaLoadIssue" and scope != "ELECTED":
            raise PGIRError(
                "PARAMETRIC_TMA_SCOPE_INVALID",
                "TMA load issue requires one statically elected program stream",
                {"template": identifier, "execution_scope": scope},
            )

        fields = _require_object(item["fields"], f"{location}/fields")
        instance_fields = _require_object(
            item.get("instance_fields", {}), f"{location}/instance_fields"
        )
        invalid_fields = sorted((set(fields) | set(instance_fields)) - EVENT_FIELD_NAMES)
        if invalid_fields:
            raise PGIRError(
                "PARAMETRIC_FIELDS_UNKNOWN",
                "event template contains unsupported event fields",
                {"template": identifier, "fields": invalid_fields},
            )
        overlap = sorted(set(fields).intersection(instance_fields))
        if overlap:
            raise PGIRError(
                "PARAMETRIC_FIELD_BINDING_AMBIGUOUS",
                "event field cannot be both constant and instance-indexed",
                {"template": identifier, "fields": overlap},
            )
        if "depends_on" in fields or "depends_on" in instance_fields:
            raise PGIRError(
                "PARAMETRIC_DEPENDENCY_ENCODING_INVALID",
                "template dependencies must use the structured dependencies field",
                {"template": identifier},
            )
        parsed_instance_fields: dict[str, tuple[Any, ...]] = {}
        for field_name, values in instance_fields.items():
            if not isinstance(values, list) or len(values) != len(program_instances):
                raise PGIRError(
                    "PARAMETRIC_BINDING_CARDINALITY_MISMATCH",
                    "instance field values must align one-to-one with program instances",
                    {
                        "template": identifier,
                        "field": field_name,
                        "instances": len(program_instances),
                        "values": len(values) if isinstance(values, list) else None,
                    },
                )
            parsed_instance_fields[field_name] = tuple(deepcopy(values))
        dependencies = _parse_dependencies(
            item.get("dependencies"), f"{location}/dependencies"
        )
        templates.append(EventTemplate(
            identifier,
            event_type,
            stream_id,
            order,
            len(templates),
            program_instances,
            executors,
            transition,
            rendezvous,
            owner,
            deepcopy(fields),
            parsed_instance_fields,
            dependencies,
            {
                instance: position
                for position, instance in enumerate(program_instances)
            },
        ))

    template_map = {template.id: template for template in templates}
    for template in templates:
        for dependency in template.dependencies:
            target = template_map.get(dependency.template)
            if target is None:
                raise PGIRError(
                    "DANGLING_REFERENCE",
                    "event template dependency references an unknown template",
                    {"template": template.id, "depends_on": dependency.template},
                )
            if target.id == template.id:
                raise PGIRError(
                    "SELF_DEPENDENCY",
                    "event template cannot depend on itself",
                    {"template": template.id},
                )
            if dependency.relation == "COLLECTIVE" and target.transition != "ONCE":
                raise PGIRError(
                    "PARAMETRIC_DEPENDENCY_RELATION_INVALID",
                    "COLLECTIVE dependency must reference a once-transition template",
                    {"template": template.id, "depends_on": target.id},
                )
    ordered = sorted(templates, key=lambda item: (item.order, item.id))
    return tuple(
        replace(template, index=index)
        for index, template in enumerate(ordered)
    )


def _validate_owners(raw_owners: Any, cta_threads: int) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_owners, list):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "owners must be an array",
            {"location": "/owners"},
        )
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_owners):
        location = f"/owners/{index}"
        item = _require_object(raw, location)
        _require_exact_keys(item, {"id", "kind", "participants"}, set(), location)
        identifier = _require_string(item["id"], f"{location}/id")
        if identifier in result:
            raise PGIRError(
                "DUPLICATE_ID",
                f"duplicate execution owner id: {identifier}",
                {"id": identifier, "kind": "owner"},
            )
        kind = _require_string(item["kind"], f"{location}/kind")
        if kind not in {"LANE", "WARP", "WARPGROUP", "PARTITION", "ASYNC_ENGINE"}:
            raise PGIRError(
                "PARAMETRIC_OWNER_KIND_UNSUPPORTED",
                "execution owner kind is unsupported",
                {"owner": identifier, "kind": kind},
            )
        participants = _require_instances(item["participants"], f"{location}/participants")
        invalid = [participant for participant in participants if participant >= cta_threads]
        if invalid:
            raise PGIRError(
                "PARTICIPANT_OUT_OF_RANGE",
                "execution owner participant exceeds CTA bounds",
                {"owner": identifier, "participants": invalid, "cta_threads": cta_threads},
            )
        result[identifier] = {
            "id": identifier,
            "kind": kind,
            "participants": list(participants),
        }
    return result


def _validate_regions(raw_regions: Any) -> set[str]:
    if not isinstance(raw_regions, list):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "regions must be an array",
            {"location": "/regions"},
        )
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_regions):
        location = f"/regions/{index}"
        region = _require_object(raw, location)
        _require_exact_keys(
            region,
            {"id", "allocation_id", "address_space", "offset", "extent"},
            set(),
            location,
        )
        identifier = _require_string(region["id"], f"{location}/id")
        if identifier in identifiers:
            raise PGIRError(
                "DUPLICATE_ID",
                f"duplicate region id: {identifier}",
                {"id": identifier, "kind": "region"},
            )
        identifiers.add(identifier)
        _require_string(region["allocation_id"], f"{location}/allocation_id")
        if region["address_space"] not in {"SHARED", "GLOBAL", "REGISTER", "TENSOR"}:
            raise PGIRError(
                "PARAMETRIC_FIELD_TYPE",
                "region address space is unsupported",
                {"location": f"{location}/address_space"},
            )
        offset = _require_nonnegative_integer(
            region["offset"], f"{location}/offset"
        )
        extent = _require_nonnegative_integer(region["extent"], f"{location}/extent")
        if extent == 0:
            raise PGIRError(
                "PARAMETRIC_FIELD_TYPE",
                "region extent must be positive",
                {"location": f"{location}/extent"},
            )
        if offset + extent > 1 << 64:
            raise PGIRError(
                "PARAMETRIC_REGION_ADDRESS_OUT_OF_RANGE",
                "region byte interval exceeds the 64-bit address domain",
                {"location": location, "offset": offset, "extent": extent},
            )
    return identifiers


def _validate_memory_template_regions(
    templates: tuple[EventTemplate, ...],
    region_ids: set[str],
) -> None:
    for template in templates:
        if template.type not in {"MemRead", "MemWrite"}:
            continue
        for instance in template.program_instances:
            regions = template.memory_regions(instance)
            if not regions:
                raise PGIRError(
                    "UNRESOLVED_MEMORY_REGION",
                    "parameterized memory event has no exact byte segment",
                    {"template": template.id, "instance": instance},
                )
            unknown = [
                region for region in regions
                if not isinstance(region, str) or region not in region_ids
            ]
            if unknown:
                raise PGIRError(
                    "DANGLING_REFERENCE",
                    "parameterized memory event references an unknown region",
                    {"template": template.id, "instance": instance, "regions": unknown},
                )


def _validate_resource_participants(
    resources: Any,
    owner_map: dict[str, dict[str, Any]],
    stream_map: dict[str, ProgramStreamClass],
) -> None:
    if not isinstance(resources, list):
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "resources must be an array",
            {"location": "/resources"},
        )
    for resource_index, resource in enumerate(resources):
        item = _require_object(resource, f"/resources/{resource_index}")
        participants = item.get("participants")
        if not isinstance(participants, list) or not participants:
            raise PGIRError(
                "PARAMETRIC_FIELD_TYPE",
                "resource participants must be a nonempty array",
                {"location": f"/resources/{resource_index}/participants"},
            )
        for participant_index, participant in enumerate(participants):
            location = f"/resources/{resource_index}/participants/{participant_index}"
            if isinstance(participant, str):
                if participant not in owner_map:
                    raise PGIRError(
                        "UNRESOLVED_PARTICIPANTS",
                        "resource references an unknown execution owner",
                        {"location": location, "participant": participant},
                    )
                continue
            reference = _require_object(participant, location)
            _require_exact_keys(reference, {"stream"}, {"instances"}, location)
            stream = _require_string(reference["stream"], f"{location}/stream")
            stream_class = stream_map.get(stream)
            if stream_class is None:
                raise PGIRError(
                    "UNRESOLVED_PARTICIPANTS",
                    "resource references an unknown program-stream class",
                    {"location": location, "stream": stream},
                )
            if "instances" in reference:
                instances = _require_instances(
                    reference["instances"], f"{location}/instances"
                )
                unknown = sorted(set(instances) - set(stream_class.instances))
                if unknown:
                    raise PGIRError(
                        "PARAMETRIC_INSTANCE_OUT_OF_RANGE",
                        "resource participant subset is outside its stream class",
                        {"location": location, "stream": stream, "instances": unknown},
                    )


def _validate_expanded_envelope(
    root: dict[str, Any],
    owner_map: dict[str, dict[str, Any]],
    streams: tuple[ProgramStreamClass, ...],
) -> None:
    actors = []
    for owner in owner_map.values():
        actors.append({
            **deepcopy(owner),
            "role": (
                "ASYNC_ENGINE"
                if owner["kind"] == "ASYNC_ENGINE"
                else "EXECUTION_OWNER"
            ),
        })
    for stream in streams:
        actors.extend(
            {
                "id": stream.actor_id(instance),
                "kind": "LANE",
                "participants": [instance],
                "role": "PROGRAM_STREAM",
            }
            for instance in stream.instances
        )
    stream_map = {stream.id: stream for stream in streams}
    module_from_dict({
        "schema_version": "pgir-v1",
        "producer": deepcopy(root["producer"]),
        "target": deepcopy(root["target"]),
        "assumptions": deepcopy(root["assumptions"]),
        "control_structure": deepcopy(root["control_structure"]),
        "actors": actors,
        "resources": [
            _expand_resource(resource, stream_map)
            for resource in root["resources"]
        ],
        "regions": deepcopy(root["regions"]),
        "events": [],
        "frontend_provenance": deepcopy(root["frontend_provenance"]),
        "unsupported_constructs": deepcopy(root["unsupported_constructs"]),
    })


def parametric_module_from_dict(data: dict[str, Any]) -> ParametricModule:
    root = _require_object(data, "/")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "producer",
            "target",
            "assumptions",
            "control_structure",
            "owners",
            "program_streams",
            "resources",
            "regions",
            "event_templates",
            "frontend_provenance",
            "frontend_evidence",
            "unsupported_constructs",
        },
        set(),
        "/",
    )
    if root["schema_version"] != PARAMETRIC_SCHEMA_VERSION:
        raise PGIRError(
            "UNKNOWN_SCHEMA_VERSION",
            "parameterized PGIR schema version is not supported",
            {"schema_version": root["schema_version"]},
        )
    target = _require_object(root["target"], "/target")
    cta_threads = _require_nonnegative_integer(
        target.get("cta_threads"), "/target/cta_threads"
    )
    if cta_threads == 0:
        raise PGIRError(
            "PARAMETRIC_FIELD_TYPE",
            "CTA thread count must be positive",
            {"location": "/target/cta_threads"},
        )
    evidence = _parse_evidence(root["frontend_evidence"])
    assumptions = _require_object(root["assumptions"], "/assumptions")
    if assumptions.get("statically_known_participants") is not True:
        raise PGIRError(
            "UNRESOLVED_ASSUMPTIONS",
            "parameterized logical streams require statically known participants",
            {"assumptions": ["statically_known_participants"]},
        )
    if assumptions.get("alias_precision") != "EXACT":
        raise PGIRError(
            "UNRESOLVED_ASSUMPTIONS",
            "parameterized memory instances require exact alias evidence",
            {"assumptions": ["alias_precision"]},
        )
    provenance = _require_object(root["frontend_provenance"], "/frontend_provenance")
    if provenance.get("extraction_complete") is not True:
        raise PGIRError(
            "PARAMETRIC_EVIDENCE_UNRESOLVED",
            "parameterized PGIR requires complete frontend extraction evidence",
            {"claim": "extraction_complete"},
        )
    owner_map = _validate_owners(root["owners"], cta_threads)
    streams = _parse_streams(root["program_streams"], cta_threads)
    stream_map = {stream.id: stream for stream in streams}
    generated_actor_ids = {
        stream.actor_id(instance)
        for stream in streams
        for instance in stream.instances
    }
    collisions = sorted(generated_actor_ids.intersection(owner_map))
    if collisions:
        raise PGIRError(
            "DUPLICATE_ID",
            "generated program-stream actors collide with execution owners",
            {"ids": collisions, "kind": "actor"},
        )
    _validate_resource_participants(root["resources"], owner_map, stream_map)
    region_ids = _validate_regions(root["regions"])
    templates = _parse_templates(root["event_templates"], streams, owner_map)
    _validate_memory_template_regions(templates, region_ids)
    _validate_expanded_envelope(root, owner_map, streams)
    template_indices = {
        template.id: index for index, template in enumerate(templates)
    }
    lane_ids = tuple(sorted(
        instance for stream in streams for instance in stream.instances
    ))
    traces_by_lane: dict[int, list[int]] = {lane: [] for lane in lane_ids}
    for index, template in enumerate(templates):
        for lane in template.program_instances:
            traces_by_lane[lane].append(index)
    lane_trace_offsets = [0]
    lane_trace_template_ids: list[int] = []
    for lane in lane_ids:
        lane_trace_template_ids.extend(traces_by_lane[lane])
        lane_trace_offsets.append(len(lane_trace_template_ids))
    return ParametricModule(
        deepcopy(root),
        streams,
        templates,
        evidence,
        tuple(lane_trace_offsets),
        tuple(lane_trace_template_ids),
        lane_ids,
        template_indices,
    )


def normalize_parametric_module(module: ParametricModule) -> dict[str, Any]:
    normalized = deepcopy(module.raw)
    normalized["owners"] = sorted(normalized["owners"], key=lambda item: item["id"])
    normalized["program_streams"] = sorted(
        normalized["program_streams"], key=lambda item: item["id"]
    )
    normalized["resources"] = sorted(
        normalized["resources"], key=lambda item: item["id"]
    )
    normalized["regions"] = sorted(normalized["regions"], key=lambda item: item["id"])
    normalized["event_templates"] = sorted(
        normalized["event_templates"], key=lambda item: (item["order"], item["id"])
    )
    for owner in normalized["owners"]:
        owner["participants"] = sorted(owner["participants"])
    for stream in normalized["program_streams"]:
        stream["instances"] = sorted(stream["instances"])
    for template in normalized["event_templates"]:
        if "instances" in template:
            template["instances"] = sorted(template["instances"])
        else:
            template["program_instances"] = sorted(template["program_instances"])
            template["executors"] = sorted(template["executors"])
        if "dependencies" in template:
            template["dependencies"] = sorted(
                template["dependencies"],
                key=lambda item: (item["template"], item["relation"]),
            )
    normalized["unsupported_constructs"] = sorted(normalized["unsupported_constructs"])
    return normalized


def _expand_resource(
    resource: dict[str, Any],
    stream_map: dict[str, ProgramStreamClass],
) -> dict[str, Any]:
    expanded = deepcopy(resource)
    participants: list[str] = []
    for participant in resource["participants"]:
        if isinstance(participant, str):
            participants.append(participant)
        else:
            stream = stream_map[participant["stream"]]
            instances = participant.get("instances", stream.instances)
            participants.extend(
                stream.actor_id(instance)
                for instance in instances
            )
    if len(participants) != len(set(participants)):
        raise PGIRError(
            "UNRESOLVED_PARTICIPANTS",
            "expanded resource participant set contains duplicates",
            {"resource": resource.get("id"), "participants": participants},
        )
    expanded["participants"] = participants
    return expanded


def _logical_event_ids(
    templates: tuple[EventTemplate, ...],
) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for template in templates:
        if template.execution_scope == "COLLECTIVE":
            identifier = stable_collective_event_id(template.id)
            result[template.id] = {
                instance: identifier
                for instance in template.instances
            }
        else:
            result[template.id] = {
                instance: stable_event_instance_id(template.id, instance)
                for instance in template.instances
            }
    return result


def _resolve_binding(value: Any, actor: str) -> Any:
    if value == "$self":
        return actor
    if isinstance(value, list):
        return [_resolve_binding(item, actor) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_binding(item, actor) for key, item in value.items()}
    return deepcopy(value)


def _expanded_source_event(
    template: EventTemplate,
    instance: int,
    actor: str,
    identifier: str,
    *,
    position: int | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "id": identifier,
        "type": template.type,
        "actor": actor,
    }
    if template.owner is not None:
        event["owner"] = template.owner
    event.update({
        field_name: _resolve_binding(value, actor)
        for field_name, value in template.fields.items()
    })
    if position is None:
        position = template.instances.index(instance)
    event.update({
        field_name: _resolve_binding(values[position], actor)
        for field_name, values in template.instance_fields.items()
    })
    return event


def _dependency_ids(
    template: EventTemplate,
    instance: int,
    template_map: dict[str, EventTemplate],
    logical_ids: dict[str, dict[int, str]],
) -> list[str]:
    dependencies: set[str] = set()
    for dependency in template.dependencies:
        target = template_map[dependency.template]
        target_ids = logical_ids[target.id]
        if dependency.relation == "SAME_INSTANCE":
            identifier = target_ids.get(instance)
            if identifier is None:
                raise PGIRError(
                    "PARAMETRIC_DEPENDENCY_INSTANCE_MISMATCH",
                    "SAME_INSTANCE dependency has no event for this logical stream",
                    {
                        "template": template.id,
                        "instance": instance,
                        "depends_on": target.id,
                    },
                )
            dependencies.add(identifier)
        elif dependency.relation == "ALL_INSTANCES":
            dependencies.update(target_ids.values())
        else:
            dependencies.add(target.logical_id(min(target.executors)))
    return sorted(dependencies)
