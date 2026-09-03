from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from .engine import Executor, ExplorationLimit, ProtocolError
from .model import Event, Module, normalize_module, stable_completion_id, stable_digest, stable_phase_id


@dataclass(frozen=True, slots=True)
class Effect:
    id: str
    actor: str
    mode: str
    region: str
    begin: str
    end: str
    async_window: bool


class PDG:
    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: set[tuple[str, str, str]] = set()

    def add_edge(self, source: str, target: str, kind: str) -> None:
        self.nodes.update((source, target))
        self.edges.add((source, target, kind))

    def reachable(self, source: str, target: str) -> bool:
        if source == target:
            return True
        adjacency: dict[str, list[str]] = {}
        for left, right, _kind in sorted(self.edges):
            adjacency.setdefault(left, []).append(right)
        queue = deque([source])
        seen = {source}
        while queue:
            node = queue.popleft()
            for next_node in adjacency.get(node, []):
                if next_node == target:
                    return True
                if next_node not in seen:
                    seen.add(next_node)
                    queue.append(next_node)
        return False

    def artifact(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [{"from": source, "to": target, "kind": kind} for source, target, kind in sorted(self.edges)],
        }


def _base_pdg(module: Module) -> PDG:
    graph = PDG()
    graph.nodes.update(event.id for event in module.events)
    for actor in sorted(module.events_by_actor):
        trace = module.events_by_actor[actor]
        for first, second in zip(trace, trace[1:]):
            graph.add_edge(first.id, second.id, "PROGRAM_ORDER")
    for event in module.events:
        for dependency in event.depends_on:
            graph.add_edge(dependency, event.id, "EXPLICIT_DEPENDENCY")
        if event.resource is None or event.generation is None:
            continue
        phase = stable_phase_id(event.resource, event.generation)
        if event.type == "MBarrierArrive" and event.predicate:
            graph.add_edge(event.id, phase, "BARRIER_ARRIVE")
        elif event.type == "MBarrierArriveExpectTx" and event.predicate:
            graph.add_edge(event.id, phase, "BARRIER_ARRIVE")
            graph.add_edge(event.id, phase, "EXPECT_TO_PHASE")
        elif event.type == "MBarrierExpectTx":
            graph.add_edge(event.id, phase, "EXPECT_TO_PHASE")
        elif event.type == "MBarrierWaitParity":
            graph.add_edge(phase, event.id, "COMPLETION_TO_WAIT")
        elif event.type == "TmaLoadIssue":
            completion = stable_completion_id(event.id, event.generation)
            graph.add_edge(event.id, completion, "ISSUE_TO_COMPLETION")
            graph.add_edge(completion, phase, "GROUP_TO_COMPLETION")
    return graph


def _anchor_ids(module: Module) -> list[str]:
    identifiers = {event.id for event in module.events}
    for event in module.events:
        if event.type == "TmaLoadIssue":
            identifiers.add(stable_completion_id(event.id, event.generation))
        if event.resource is not None and event.generation is not None:
            identifiers.add(stable_phase_id(event.resource, event.generation))
    return sorted(identifiers)


def _effects(module: Module) -> list[Effect]:
    result: list[Effect] = []
    for event in module.events:
        if event.type in {"MemRead", "MemWrite"}:
            result.append(Effect(event.id, event.actor, "READ" if event.type == "MemRead" else "WRITE", event.region, event.id, event.id, False))
        elif event.type == "TmaLoadIssue":
            completion = stable_completion_id(event.id, event.generation)
            result.extend((
                Effect(f"effect:{event.id}:source", event.actor, "READ", event.source_region, event.id, completion, True),
                Effect(f"effect:{event.id}:destination", event.actor, "WRITE", event.destination_region, event.id, completion, True),
            ))
    return sorted(result, key=lambda effect: effect.id)


def _may_alias(module: Module, left: str, right: str) -> bool:
    a = module.region_map[left]
    b = module.region_map[right]
    return a.address_space == b.address_space and a.allocation_id == b.allocation_id and a.offset < b.offset + b.extent and b.offset < a.offset + a.extent


def _arrivals(module: Module, resource: str, generation: int) -> list[Event]:
    return [event for event in module.events if event.resource == resource and event.generation == generation and event.predicate and event.type in {"MBarrierArrive", "MBarrierArriveExpectTx"}]


def _waits(module: Module, resource: str, generation: int) -> list[Event]:
    return [event for event in module.events if event.resource == resource and event.generation == generation and event.type == "MBarrierWaitParity"]


def _mbarrier_visibility(module: Module, graph: PDG, source: Effect, target: Effect) -> bool:
    if module.region_map[source.region].address_space != "SHARED":
        return False
    source_event = module.event_map[source.end]
    target_event = module.event_map[target.begin]
    for arrival in module.events:
        if arrival.type not in {"MBarrierArrive", "MBarrierArriveExpectTx"} or not arrival.predicate:
            continue
        if arrival.actor != source_event.actor or not graph.reachable(source.end, arrival.id):
            continue
        for wait in _waits(module, arrival.resource, arrival.generation):
            if wait.actor == target_event.actor and graph.reachable(wait.id, target.begin):
                return True
    return False


def _completion_observed(module: Module, graph: PDG, window: Effect, point: Effect) -> bool:
    issue = module.event_map[window.begin]
    point_event = module.event_map[point.begin]
    return any(wait.actor == point_event.actor and graph.reachable(wait.id, point.begin) for wait in _waits(module, issue.resource, issue.generation))


def _data_hazard(module: Module, graph: PDG) -> dict[str, Any] | None:
    effects = _effects(module)
    for index, left in enumerate(effects):
        for right in effects[index + 1:]:
            if left.actor == right.actor and not left.async_window and not right.async_window:
                continue
            if left.mode == right.mode == "READ" or not _may_alias(module, left.region, right.region):
                continue
            safe = False
            query = ""
            if not left.async_window and not right.async_window:
                if left.actor == right.actor:
                    safe = graph.reachable(left.end, right.begin) or graph.reachable(right.end, left.begin)
                else:
                    safe = _mbarrier_visibility(module, graph, left, right) or _mbarrier_visibility(module, graph, right, left)
                query = "point conflict requires same-actor order or a scoped shared mbarrier publication"
            elif left.async_window and right.async_window:
                safe = graph.reachable(left.end, right.begin) or graph.reachable(right.end, left.begin)
                query = "async windows must be non-overlapping"
            else:
                window, point = (left, right) if left.async_window else (right, left)
                if window.mode == "READ":
                    # generic WRITE -> TMA READ requires a matching generic->TMA proxy
                    # fence, which is deliberately unsupported in this core.
                    point_before = False
                    completion_before = _completion_observed(module, graph, window, point)
                    query = "generic write before TMA read requires unsupported proxy visibility, or overwrite must follow observed completion"
                else:
                    point_before = graph.reachable(point.end, window.begin)
                    completion_before = _completion_observed(module, graph, window, point)
                    query = "TMA destination conflict must precede issue or follow observed completion"
                safe = point_before or completion_before
            if not safe:
                return {
                    "category": "UNORDERED_CONFLICT",
                    "events": sorted({left.begin, right.begin}),
                    "effects": [
                        {"id": effect.id, "actor": effect.actor, "mode": effect.mode, "region": effect.region, "window": [effect.begin, effect.end]}
                        for effect in (left, right)
                    ],
                    "missing_path": query,
                    "repair": "add a supported, region-scoped mbarrier handoff; generic-to-TMA source publication additionally requires proxy support",
                }
    return None


def _statistics(module: Module, engine: str) -> dict[str, Any]:
    return {
        "actors": len(module.actors),
        "events": len(module.events),
        "resources": len(module.resources),
        "regions": len(module.regions),
        "explored_steps": 0,
        "explored_states": 0,
        "engine": engine,
    }


def _protocol_report(
    error: ProtocolError,
    trace: list[dict[str, Any]],
    graph: PDG,
    boundaries: list[dict[str, Any]],
    digest: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = error.diagnostic()
    return {
        "verdict": "SYNC_HAZARD", "checker_version": "0.2.0", "summary": diagnostic["message"],
        "diagnostics": [diagnostic],
        "witness": {
            "category": diagnostic["code"], "event": diagnostic.get("event"),
            "protocol_state": diagnostic.get("protocol_state", {}), "reachable_execution_prefix": trace,
            "repair": "restore lifecycle/generation/count/transaction invariants before this event",
        },
        "statistics": statistics,
        "artifacts": {"normalized_sha256": digest, "pdg": graph.artifact(), "boundaries": boundaries},
    }


def _inconclusive(
    code: str,
    message: str,
    digest: str,
    graph: PDG,
    boundaries: list[dict[str, Any]],
    statistics: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        diagnostic["details"] = details
    return {
        "verdict": "INCONCLUSIVE", "checker_version": "0.2.0", "summary": message,
        "diagnostics": [diagnostic], "statistics": statistics,
        "artifacts": {"normalized_sha256": digest, "pdg": graph.artifact(), "boundaries": boundaries},
    }


def _first_error(errors: list[tuple[ProtocolError, Any]]) -> tuple[ProtocolError, Any]:
    return sorted(errors, key=lambda item: (item[0].code, item[0].event or "", tuple(step["transition"] for step in item[1].trace)))[0]


def _add_dynamic_edges(graph: PDG, state: Any) -> None:
    graph.edges.update(state.dynamic_edges)
    graph.nodes.update(node for edge in state.dynamic_edges for node in edge[:2])


def _stuck_report(
    executor: Executor,
    state: Any,
    graph: PDG,
    boundaries: list[dict[str, Any]],
    digest: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    detail = executor.stuck_diagnostic(state)
    error = ProtocolError(detail["code"], detail["message"], state=detail.get("protocol_state", detail))
    return _protocol_report(error, state.trace, graph, boundaries, digest, statistics)


def _run_engine(module: Module, engine: Literal["maximal", "exhaustive"]) -> dict[str, Any]:
    normalized = normalize_module(module)
    digest = stable_digest(normalized)
    executor = Executor(module)
    graph = _base_pdg(module)
    canonical = executor.explore()
    boundaries: list[dict[str, Any]] = []
    statistics = _statistics(module, engine)
    statistics["explored_steps"] += len(canonical.state.trace)

    if engine == "maximal":
        if canonical.error is not None:
            return _protocol_report(canonical.error, canonical.state.trace, graph, boundaries, digest, statistics)
        if canonical.stuck:
            return _stuck_report(executor, canonical.state, graph, boundaries, digest, statistics)
    else:
        try:
            global_outcomes = executor.explore_all()
        except ExplorationLimit as error:
            return _inconclusive("EXPLORATION_LIMIT", str(error), digest, graph, boundaries, statistics)
        statistics["explored_states"] += global_outcomes.visited_states
        if global_outcomes.errors:
            error, state = _first_error(global_outcomes.errors)
            return _protocol_report(error, state.trace, graph, boundaries, digest, statistics)
        if global_outcomes.stuck:
            state = sorted(global_outcomes.stuck, key=executor.fingerprint)[0]
            return _stuck_report(executor, state, graph, boundaries, digest, statistics)

    for anchor in _anchor_ids(module):
        if engine == "maximal":
            outcome = executor.explore(hold=anchor)
            statistics["explored_steps"] += len(outcome.state.trace)
            if outcome.error is not None:
                return _protocol_report(outcome.error, outcome.state.trace, graph, boundaries, digest, statistics)
            states = [outcome.state]
            boundary = outcome.boundary
        else:
            try:
                outcomes = executor.explore_all(hold=anchor)
            except ExplorationLimit as error:
                return _inconclusive("EXPLORATION_LIMIT", str(error), digest, graph, boundaries, statistics)
            statistics["explored_states"] += outcomes.visited_states
            if outcomes.errors:
                error, state = _first_error(outcomes.errors)
                return _protocol_report(error, state.trace, graph, boundaries, digest, statistics)
            boundary_set = {executor.boundary(state) for state in outcomes.terminals}
            if len(boundary_set) != 1:
                return _inconclusive(
                    "NONCONFLUENT_BOUNDARY",
                    "hold-anchor maximal outcomes have different full boundaries",
                    digest,
                    graph,
                    boundaries,
                    statistics,
                )
            states = outcomes.terminals
            boundary = next(iter(boundary_set), ())

        boundaries.append({"anchor": anchor, "boundary": list(boundary)})
        for state in states:
            probe_error, probed = executor.probe(state, anchor)
            if probe_error is not None:
                return _protocol_report(probe_error, state.trace, graph, boundaries, digest, statistics)
            if probed is None:
                return _inconclusive(
                    "ANCHOR_NOT_REACHED",
                    "held anchor was not enabled at its terminal boundary",
                    digest,
                    graph,
                    boundaries,
                    statistics,
                    {"anchor": anchor, "boundary": list(executor.boundary(state))},
                )
            _add_dynamic_edges(graph, state)
        for event_id in boundary:
            if event_id != anchor:
                graph.add_edge(anchor, event_id, "BOUNDARY_ORDER")

    _add_dynamic_edges(graph, canonical.state)
    hazard = _data_hazard(module, graph)
    artifacts = {
        "normalized_sha256": digest, "canonical_trace": canonical.state.trace,
        "boundaries": sorted(boundaries, key=lambda item: item["anchor"]),
        "pdg": graph.artifact(),
    }
    counts = {key: statistics[key] for key in ("actors", "events", "resources", "regions")}
    if hazard is not None:
        hazard["reachable_execution_prefix"] = canonical.state.trace
        hazard["protocol_state"] = {key: value.snapshot() for key, value in sorted(canonical.state.barriers.items())}
        return {
            "verdict": "DATA_HAZARD", "checker_version": "0.2.0",
            "summary": "a conflicting memory access pair lacks supported execution and visibility ordering",
            "diagnostics": [{"code": hazard["category"], "message": hazard["missing_path"], "events": hazard["events"]}],
            "witness": hazard, "statistics": statistics, "artifacts": artifacts,
        }
    return {
        "verdict": "SAFE", "checker_version": "0.2.0", "summary": "all supported protocol and data checks passed",
        "diagnostics": [], "statistics": statistics,
        "manifest": {
            "scope": "finite statically-resolved sm90 single_cta mbarrier/TMA-load core",
            "assumptions": dict(sorted(module.raw["assumptions"].items())), "producer": module.raw["producer"],
            "frontend_provenance": module.raw["frontend_provenance"], "counts": counts,
        },
        "artifacts": artifacts,
    }


def _core_result(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": report["verdict"],
        "primary_diagnostic": report.get("diagnostics", [{}])[0].get("code") if report.get("diagnostics") else None,
        "boundaries": report.get("artifacts", {}).get("boundaries", []),
    }


def check_module(
    module: Module,
    engine: Literal["maximal", "exhaustive", "crosscheck"] = "maximal",
) -> dict[str, Any]:
    if engine == "maximal" or engine == "exhaustive":
        return _run_engine(module, engine)
    if engine != "crosscheck":
        raise ValueError(f"unknown checker engine: {engine}")

    maximal = _run_engine(module, "maximal")
    exhaustive = _run_engine(module, "exhaustive")
    maximal_core = _core_result(maximal)
    exhaustive_core = _core_result(exhaustive)
    statistics = _statistics(module, "crosscheck")
    statistics["explored_steps"] = maximal["statistics"]["explored_steps"] + exhaustive["statistics"]["explored_steps"]
    statistics["explored_states"] = exhaustive["statistics"]["explored_states"]
    statistics["maximal_steps"] = maximal["statistics"]["explored_steps"]
    statistics["exhaustive_states"] = exhaustive["statistics"]["explored_states"]

    # A protocol-risk run returns at its first reachable witness, so the two
    # engines may legitimately have collected different boundary *prefixes*.
    # Boundary equality is meaningful only after both engines complete all
    # anchor passes (SAFE or DATA_HAZARD). For early protocol verdicts, compare
    # the verdict itself; witness selection is not part of the Boolean theorem.
    comparison = {"maximal": maximal_core, "exhaustive": exhaustive_core}
    complete_verdicts = {"SAFE", "DATA_HAZARD"}
    mismatch = maximal_core["verdict"] != exhaustive_core["verdict"]
    if not mismatch and maximal_core["verdict"] in complete_verdicts:
        mismatch = maximal_core["boundaries"] != exhaustive_core["boundaries"]
    if mismatch:
        digest = stable_digest(normalize_module(module))
        return {
            "verdict": "INCONCLUSIVE", "checker_version": "0.2.0",
            "summary": "maximal and exhaustive engines disagree on the core verdict or held-anchor boundaries",
            "diagnostics": [{
                "code": "ENGINE_CROSSCHECK_MISMATCH",
                "message": "maximal and exhaustive engines produced different core results",
                "details": comparison,
            }],
            "statistics": statistics,
            "artifacts": {
                "normalized_sha256": digest,
                "pdg": maximal.get("artifacts", {}).get("pdg", {"nodes": [], "edges": []}),
                "boundaries": maximal_core["boundaries"],
                "crosscheck": comparison,
            },
        }

    result = dict(maximal)
    result["statistics"] = statistics
    artifacts = dict(result.get("artifacts", {}))
    artifacts["crosscheck"] = comparison
    result["artifacts"] = artifacts
    return result
