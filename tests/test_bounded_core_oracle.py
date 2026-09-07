from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

from pipeguard.checker import PDG, check_module
from pipeguard.model import module_from_dict


@dataclass(frozen=True, slots=True)
class CoreEvent:
    id: str
    actor: int
    kind: Literal["ARRIVE", "WAIT"]
    generation: int


@dataclass(frozen=True, slots=True)
class CoreState:
    pcs: tuple[int, ...]
    generation: int
    pending: int
    generation_started: bool
    publications: frozenset[int]
    last_successfully_observed_generation: int | None
    remaining: tuple[tuple[int, frozenset[str]], ...]
    executed: frozenset[str]


@dataclass(frozen=True, slots=True)
class OracleResult:
    states: frozenset[CoreState]
    invalid_risk: bool
    stuck_risk: bool

    @property
    def risk(self) -> bool:
        return self.invalid_risk or self.stuck_risk


def _remaining_map(state: CoreState) -> dict[int, set[str]]:
    return {generation: set(events) for generation, events in state.remaining}


def _remaining_tuple(remaining: dict[int, set[str]]) -> tuple[tuple[int, frozenset[str]], ...]:
    return tuple(
        (generation, frozenset(events))
        for generation, events in sorted(remaining.items())
        if events
    )


def _wait_status(
    state: CoreState,
    event: CoreEvent,
    semantics: Literal["raw", "exact"],
) -> Literal["READY", "BLOCKED", "INVALID"]:
    target = event.generation
    if semantics == "raw":
        if target == state.generation - 1 and target in state.publications:
            return "READY"
    elif state.generation == target + 1 and target in state.publications:
        return "READY"
    if target >= state.generation:
        return "BLOCKED"
    return "INVALID"


def _core_oracle(
    traces: tuple[tuple[CoreEvent, ...], ...],
    capacity: int,
    semantics: Literal["raw", "exact"],
) -> OracleResult:
    required: dict[int, set[str]] = {}
    for trace in traces:
        for event in trace:
            if event.kind == "WAIT":
                required.setdefault(event.generation, set()).add(event.id)
    initial = CoreState(
        pcs=(0,) * len(traces),
        generation=0,
        pending=capacity,
        generation_started=False,
        publications=frozenset(),
        last_successfully_observed_generation=None,
        remaining=_remaining_tuple(required),
        executed=frozenset(),
    )
    stack = [initial]
    seen: set[CoreState] = set()
    invalid_risk = False
    stuck_risk = False
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        choices: list[tuple[str, int | None]] = []
        invalid_here = False
        remaining = _remaining_map(state)
        if state.pending == 0:
            if state.generation > 0 and remaining.get(state.generation - 1):
                invalid_here = True
                invalid_risk = True
            else:
                choices.append(("PHASE", None))
        for actor, trace in enumerate(traces):
            if state.pcs[actor] == len(trace):
                continue
            event = trace[state.pcs[actor]]
            if event.kind == "WAIT":
                status = _wait_status(state, event, semantics)
                if status == "READY":
                    choices.append(("PROGRAM", actor))
                elif status == "INVALID":
                    invalid_here = True
                    invalid_risk = True
                continue
            if event.generation == state.generation and state.pending > 0:
                if (
                    state.generation > 0
                    and state.last_successfully_observed_generation != state.generation - 1
                ):
                    invalid_here = True
                    invalid_risk = True
                else:
                    choices.append(("PROGRAM", actor))
            elif event.generation == state.generation + 1 and state.pending == 0:
                pass
            else:
                invalid_here = True
                invalid_risk = True
        if not choices:
            all_program_events_finished = all(
                pc == len(trace) for pc, trace in zip(state.pcs, traces)
            )
            quiescent = (
                all_program_events_finished
                and not state.generation_started
                and state.pending == capacity
            )
            unobserved = (
                state.generation > 0
                and state.last_successfully_observed_generation != state.generation - 1
            )
            if quiescent and unobserved:
                invalid_here = True
                invalid_risk = True
            elif not quiescent and not invalid_here:
                stuck_risk = True
            continue
        for kind, actor in choices:
            if kind == "PHASE":
                publications = set(state.publications)
                publications.add(state.generation)
                stack.append(CoreState(
                    pcs=state.pcs,
                    generation=state.generation + 1,
                    pending=capacity,
                    generation_started=False,
                    publications=frozenset(publications),
                    last_successfully_observed_generation=state.last_successfully_observed_generation,
                    remaining=state.remaining,
                    executed=state.executed,
                ))
                continue
            assert actor is not None
            event = traces[actor][state.pcs[actor]]
            pcs = list(state.pcs)
            pcs[actor] += 1
            next_remaining = _remaining_map(state)
            pending = state.pending
            generation_started = state.generation_started
            observed_generation = state.last_successfully_observed_generation
            if event.kind == "ARRIVE":
                pending -= 1
                generation_started = True
            else:
                next_remaining.get(event.generation, set()).discard(event.id)
                observed_generation = event.generation
            stack.append(CoreState(
                pcs=tuple(pcs),
                generation=state.generation,
                pending=pending,
                generation_started=generation_started,
                publications=state.publications,
                last_successfully_observed_generation=observed_generation,
                remaining=_remaining_tuple(next_remaining),
                executed=state.executed | {event.id},
            ))
    return OracleResult(frozenset(seen), invalid_risk, stuck_risk)


def _abstract_program(
    capacity: int,
    actor_count: int,
    assignment: tuple[int, ...],
) -> tuple[tuple[CoreEvent, ...], ...]:
    specifications = [
        *(("ARRIVE", 0) for _ in range(capacity)),
        ("WAIT", 0),
        *(("ARRIVE", 1) for _ in range(capacity)),
    ]
    traces: list[list[CoreEvent]] = [[] for _ in range(actor_count)]
    counters = {("ARRIVE", 0): 0, ("WAIT", 0): 0, ("ARRIVE", 1): 0}
    for actor, (kind, generation) in zip(assignment, specifications):
        index = counters[(kind, generation)]
        counters[(kind, generation)] += 1
        prefix = "a" if kind == "ARRIVE" else "w"
        event = CoreEvent(f"{prefix}{generation}_{index}", actor, kind, generation)
        traces[actor].append(event)
    wait_actor = assignment[capacity]
    traces[wait_actor].append(CoreEvent("w1_0", wait_actor, "WAIT", 1))
    return tuple(tuple(trace) for trace in traces)


def _underfilled_program(
    capacity: int,
    actor_count: int,
    arrival_count: int,
    assignment: tuple[int, ...],
) -> tuple[tuple[CoreEvent, ...], ...]:
    specifications = [
        *(("ARRIVE", 0) for _ in range(arrival_count)),
        ("WAIT", 0),
    ]
    traces: list[list[CoreEvent]] = [[] for _ in range(actor_count)]
    arrive_index = 0
    for actor, (kind, generation) in zip(assignment, specifications):
        if kind == "ARRIVE":
            identifier = f"a0_{arrive_index}"
            arrive_index += 1
        else:
            identifier = "w0_0"
        traces[actor].append(CoreEvent(identifier, actor, kind, generation))
    return tuple(tuple(trace) for trace in traces)


def _pgir_for_core(traces: tuple[tuple[CoreEvent, ...], ...], capacity: int):
    actors = [
        {"id": "setup", "kind": "LANE", "participants": [0]},
        *[
            {"id": f"actor_{actor}", "kind": "LANE", "participants": [actor + 1]}
            for actor in range(len(traces))
        ],
    ]
    events: list[dict] = [{
        "id": "init",
        "type": "MBarrierInit",
        "actor": "setup",
        "seq": 0,
        "resource": "bar",
        "lifecycle_epoch": 0,
        "count": capacity,
    }]
    for actor, trace in enumerate(traces):
        for sequence, event in enumerate(trace):
            raw = {
                "id": event.id,
                "type": "MBarrierArrive" if event.kind == "ARRIVE" else "MBarrierWaitParity",
                "actor": f"actor_{actor}",
                "seq": sequence,
                "resource": "bar",
                "lifecycle_epoch": 0,
                "generation": event.generation,
                "depends_on": ["init"],
            }
            if event.kind == "ARRIVE":
                raw["count"] = 1
            else:
                raw.update({
                    "raw_parity": event.generation % 2,
                    "parity_evidence": {
                        "derivation_kind": "BARRIER_HISTORY",
                        "logical_stream_id": f"actor_{actor}:bar",
                        "lifecycle_epoch": 0,
                        "stream_rank": event.generation,
                        "polling_coalesced": True,
                    },
                })
            events.append(raw)
    raw_module = {
        "schema_version": "pgir-v1",
        "producer": {
            "dsl": "TILELANG",
            "dsl_version": "bounded-oracle",
            "dsl_commit": "test",
            "backend_commit": "test",
            "adapter_version": "bounded-oracle",
            "command": "pytest",
        },
        "target": {
            "architecture": "sm90",
            "warp_size": 32,
            "cta_threads": len(actors),
            "shared_memory_bytes": 65536,
            "grid_scope": "single_cta",
        },
        "assumptions": {
            "finite_control_flow": True,
            "statically_known_participants": True,
            "fair_async_completion": True,
            "no_cluster_effects": True,
            "alias_precision": "EXACT",
            "parity_normalization_status": "RESOLVED",
        },
        "actors": actors,
        "resources": [{
            "id": "bar",
            "kind": "mbarrier",
            "participants": [actor["id"] for actor in actors],
        }],
        "regions": [],
        "control_structure": {"finite_event_graph": True},
        "events": events,
        "frontend_provenance": {
            "source": "bounded-core-oracle",
            "extraction_complete": True,
        },
        "unsupported_constructs": [],
    }
    return module_from_dict(raw_module)


def _graph_reachable(report: dict, source: str, target: str) -> bool:
    adjacency: dict[str, set[str]] = {}
    for edge in report["artifacts"]["pdg"]["edges"]:
        adjacency.setdefault(edge["from"], set()).add(edge["to"])
    stack = [source]
    seen = {source}
    while stack:
        node = stack.pop()
        for successor in adjacency.get(node, set()):
            if successor == target:
                return True
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return False


def _oracle_must_order(states: frozenset[CoreState], source: str, target: str) -> bool:
    return not any(target in state.executed and source not in state.executed for state in states)


def test_bounded_raw_oracle_matches_checker_lts_and_safe_pdg():
    checked = 0
    safe = 0
    risk = 0
    stuck = 0
    lts_checked = 0
    pdg_pairs = 0
    for capacity, actor_count in ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2)):
        width = 2 * capacity + 1
        for assignment in product(range(actor_count), repeat=width):
            traces = _abstract_program(capacity, actor_count, assignment)
            raw = _core_oracle(traces, capacity, "raw")
            exact = _core_oracle(traces, capacity, "exact")
            module = _pgir_for_core(traces, capacity)
            report = check_module(module, engine="crosscheck")
            checked += 1
            if raw.risk:
                risk += 1
                stuck += raw.stuck_risk
                assert report["verdict"] == "SYNC_HAZARD", (
                    capacity,
                    actor_count,
                    assignment,
                    report,
                )
            else:
                safe += 1
                assert report["verdict"] == "SAFE", (
                    capacity,
                    actor_count,
                    assignment,
                    report,
                )
            if not raw.invalid_risk:
                lts_checked += 1
                assert raw.states == exact.states
            if report["verdict"] == "SAFE":
                event_ids = [event.id for trace in traces for event in trace]
                for source, target in product(event_ids, repeat=2):
                    if source == target:
                        continue
                    pdg_pairs += 1
                    assert _graph_reachable(report, source, target) == _oracle_must_order(
                        raw.states,
                        source,
                        target,
                    ), (capacity, actor_count, assignment, source, target)
    for capacity in (1, 2, 3):
        actor_count = 2
        for arrival_count in range(capacity):
            width = arrival_count + 1
            for assignment in product(range(actor_count), repeat=width):
                traces = _underfilled_program(
                    capacity,
                    actor_count,
                    arrival_count,
                    assignment,
                )
                raw = _core_oracle(traces, capacity, "raw")
                exact = _core_oracle(traces, capacity, "exact")
                report = check_module(_pgir_for_core(traces, capacity), engine="crosscheck")
                checked += 1
                risk += raw.risk
                stuck += raw.stuck_risk
                lts_checked += not raw.invalid_risk
                assert not raw.invalid_risk
                assert raw.stuck_risk
                assert raw.states == exact.states
                assert report["verdict"] == "SYNC_HAZARD", (
                    capacity,
                    arrival_count,
                    assignment,
                    report,
                )
    assert checked == 460
    assert safe == 64
    assert risk == 396
    assert stuck == 22
    assert lts_checked == 86
    assert pdg_pairs == 2102


def _linear_observed_mbarrier(generations: int):
    trace = []
    for generation in range(generations):
        trace.extend((
            CoreEvent(f"arrive_{generation}", 0, "ARRIVE", generation),
            CoreEvent(f"wait_{generation}", 0, "WAIT", generation),
        ))
    return _pgir_for_core((tuple(trace),), 1)


def test_successful_wait_guard_preserves_t_eff_n_ext_bound():
    reports = {
        size: check_module(_linear_observed_mbarrier(size), engine="maximal")
        for size in (16, 32)
    }
    for result in reports.values():
        statistics = result["statistics"]
        assert result["verdict"] == "SAFE"
        assert statistics["scheduler_work_units"] <= 12 * statistics["t_eff"] * statistics["n_ext"]
        assert statistics["pdg_edges"] <= 2 * statistics["n_ext"]
        assert statistics["full_enabled_scans"] == 0
        assert statistics["state_clones"] == 0
        assert statistics["replayed_prefix_steps"] == 0
    small = reports[16]["statistics"]
    large = reports[32]["statistics"]
    assert large["n_ext"] == 2 * small["n_ext"] - 1
    assert large["scheduler_work_units"] <= 2 * small["scheduler_work_units"] + 32


def _independent_reads(actor_count: int, events_per_actor: int):
    actors = [
        {"id": f"actor_{actor}", "kind": "LANE", "participants": [actor]}
        for actor in range(actor_count)
    ]
    return module_from_dict({
        "schema_version": "pgir-v1",
        "producer": {
            "dsl": "TRITON",
            "dsl_version": "bounded-oracle",
            "dsl_commit": "test",
            "backend_commit": "test",
            "adapter_version": "bounded-oracle",
            "command": "pytest",
        },
        "target": {
            "architecture": "sm90",
            "warp_size": 32,
            "cta_threads": actor_count,
            "shared_memory_bytes": 65536,
            "grid_scope": "single_cta",
        },
        "assumptions": {
            "finite_control_flow": True,
            "statically_known_participants": True,
            "fair_async_completion": True,
            "no_cluster_effects": True,
            "alias_precision": "EXACT",
            "parity_normalization_status": "RESOLVED",
        },
        "actors": actors,
        "resources": [],
        "regions": [{
            "id": "tile",
            "allocation_id": "smem",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 16,
        }],
        "control_structure": {"finite_event_graph": True},
        "events": [
            {
                "id": f"read_{actor}_{sequence}",
                "type": "MemRead",
                "actor": f"actor_{actor}",
                "seq": sequence,
                "region": "tile",
            }
            for actor in range(actor_count)
            for sequence in range(events_per_actor)
        ],
        "frontend_provenance": {
            "source": "pdg-work-bound",
            "extraction_complete": True,
        },
        "unsupported_constructs": [],
    })


def test_pdg_reachability_scans_each_source_once_after_graph_is_frozen():
    graph = PDG()
    graph.add_edge("a", "b", "PROGRAM_ORDER")
    graph.add_edge("b", "c", "PROGRAM_ORDER")
    graph.add_edge("c", "d", "PROGRAM_ORDER")
    assert graph.reachable("a", "c")
    assert graph.reachable("a", "d")
    assert not graph.reachable("a", "missing")
    statistics = graph.statistics()
    assert statistics["pdg_reachability_queries"] == 3
    assert statistics["pdg_reachability_source_scans"] == 1
    assert statistics["pdg_reachability_edge_visits"] == 3


def test_incremental_and_direct_pdg_work_have_t_eff_n_ext_certificate():
    for actor_count in (4, 8):
        report = check_module(_independent_reads(actor_count, 4), engine="maximal")
        statistics = report["statistics"]
        assert report["verdict"] == "SAFE"
        assert statistics["n_ext"] == actor_count * 4
        assert statistics["t_eff"] == actor_count
        assert statistics["scheduler_work_units"] <= 12 * statistics["t_eff"] * statistics["n_ext"]
        assert statistics["pdg_edge_insertions"] == statistics["pdg_edges"]
        assert statistics["pdg_edges_by_kind"]["PROGRAM_ORDER"] == actor_count * 3
        assert statistics["pdg_edges_by_kind"].get("BOUNDARY_ORDER", 0) <= (
            statistics["anchors"] * actor_count
        )
        assert statistics["pdg_edges"] <= statistics["n_ext"] * (statistics["t_eff"] + 4)
        assert statistics["full_enabled_scans"] == 0
        assert statistics["state_clones"] == 0
        assert statistics["replayed_prefix_steps"] == 0
