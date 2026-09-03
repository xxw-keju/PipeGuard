from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from .model import Event, Module, stable_completion_id, stable_phase_id


class ProtocolError(Exception):
    def __init__(self, code: str, message: str, event: str | None = None, state: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.event = event
        self.state = state or {}

    def diagnostic(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.event is not None:
            result["event"] = self.event
        if self.state:
            result["protocol_state"] = self.state
        return result


class ExplorationLimit(Exception):
    pass


@dataclass(slots=True)
class AsyncOp:
    completion_id: str
    issue_id: str
    actor: str
    resource: str
    destination_region: str
    generation: int
    bytes: int


@dataclass(slots=True)
class BarrierState:
    lifecycle: Literal["UNINITIALIZED", "LIVE", "INVALIDATED"] = "UNINITIALIZED"
    generation: int = 0
    parity: int = 0
    base_count: int = 0
    pending: int = 0
    expected_tx: int = 0
    completed_tx: int = 0
    generation_started: bool = False
    can_invalidate: bool = False
    publications: set[int] = field(default_factory=set)

    def snapshot(self) -> dict[str, Any]:
        return {
            "lifecycle": self.lifecycle,
            "logical_generation": self.generation,
            "hardware_parity": self.parity,
            "pending_arrivals_current": self.pending,
            "expected_tx_bytes_current": self.expected_tx,
            "completed_tx_bytes_current": self.completed_tx,
            "can_invalidate": self.can_invalidate,
            "publications": sorted(self.publications),
        }


@dataclass(frozen=True, slots=True)
class Transition:
    id: str
    kind: Literal["PROGRAM", "TMA_COMPLETION", "PHASE_COMPLETION"]
    event: Event | None = None
    resource: str | None = None

    def order_key(self) -> tuple[Any, ...]:
        if self.kind != "PROGRAM":
            return (0, self.id)
        assert self.event is not None
        return (1, self.event.actor, self.event.seq, self.event.id)


@dataclass(slots=True)
class ExecutionState:
    frontiers: dict[str, int]
    executed: set[str]
    barriers: dict[str, BarrierState]
    pending_completions: dict[str, AsyncOp] = field(default_factory=dict)
    completed_transitions: set[str] = field(default_factory=set)
    trace: list[dict[str, Any]] = field(default_factory=list)
    dynamic_edges: set[tuple[str, str, str]] = field(default_factory=set)

    def clone(self) -> "ExecutionState":
        return deepcopy(self)


@dataclass(slots=True)
class ExplorationResult:
    state: ExecutionState
    boundary: tuple[str, ...]
    error: ProtocolError | None
    stuck: bool
    held: str | None


@dataclass(slots=True)
class ExhaustiveResult:
    terminals: list[ExecutionState]
    errors: list[tuple[ProtocolError, ExecutionState]]
    stuck: list[ExecutionState]
    visited_states: int


class Executor:
    """Reference scheduler and finite exhaustive oracle.

    Deterministic maximal execution always selects the minimum stable
    transition. ``explore_all`` remains a finite small-model oracle.
    Synthetic TMA and phase completions remain independent transitions.
    """

    MAX_STATES = 100_000

    def __init__(self, module: Module):
        self.module = module
        self.events = module.event_map

    def initial_state(self) -> ExecutionState:
        return ExecutionState(
            frontiers={actor.id: 0 for actor in self.module.actors},
            executed=set(),
            barriers={resource.id: BarrierState() for resource in self.module.resources},
        )

    def _frontier_event(self, state: ExecutionState, actor: str) -> Event | None:
        index = state.frontiers[actor]
        trace = self.module.events_by_actor[actor]
        return trace[index] if index < len(trace) else None

    def _phase_transition(self, resource_id: str, barrier: BarrierState) -> Transition | None:
        if barrier.lifecycle == "LIVE" and barrier.generation_started and barrier.pending == 0 and barrier.completed_tx == barrier.expected_tx:
            return Transition(stable_phase_id(resource_id, barrier.generation), "PHASE_COMPLETION", resource=resource_id)
        return None

    def _wait_status(self, event: Event, barrier: BarrierState) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.generation is not None and event.raw_parity is not None
        if barrier.lifecycle != "LIVE" or event.raw_parity != event.generation % 2:
            return "INVALID"
        if event.generation > barrier.generation + 1:
            return "INVALID"
        if event.generation >= barrier.generation:
            return "BLOCKED"
        if event.generation < barrier.generation - 1 or event.generation not in barrier.publications:
            return "INVALID"
        return "READY"

    def enabled(self, state: ExecutionState, hold: str | None = None) -> list[Transition]:
        result: list[Transition] = []
        for completion_id in sorted(state.pending_completions):
            if completion_id != hold:
                result.append(Transition(completion_id, "TMA_COMPLETION"))
        for resource_id in sorted(state.barriers):
            transition = self._phase_transition(resource_id, state.barriers[resource_id])
            if transition is not None and transition.id != hold:
                result.append(transition)
        for actor in sorted(state.frontiers):
            event = self._frontier_event(state, actor)
            if event is None or event.id == hold or not set(event.depends_on).issubset(state.executed):
                continue
            if event.type == "MBarrierWaitParity" and self._wait_status(event, state.barriers[event.resource]) == "BLOCKED":
                continue
            if event.resource is not None and event.generation is not None:
                barrier = state.barriers[event.resource]
                phase = self._phase_transition(event.resource, barrier)
                held_op = state.pending_completions.get(hold) if hold is not None else None
                held_completion_enables_phase = (
                    held_op is not None
                    and held_op.resource == event.resource
                    and held_op.generation == barrier.generation
                    and barrier.pending == 0
                    and barrier.completed_tx + held_op.bytes == barrier.expected_tx
                )
                if event.generation == barrier.generation + 1 and (phase is not None or held_completion_enables_phase):
                    # A normalized future-generation event denotes an instruction
                    # after the hardware phase flip. It cannot execute before the
                    # synthetic phase transition. A deliberately held completion
                    # that alone enables the phase also forms an analysis boundary.
                    continue
            result.append(Transition(event.id, "PROGRAM", event=event))
        return sorted(result, key=Transition.order_key)

    def _barrier(self, state: ExecutionState, event: Event) -> BarrierState:
        assert event.resource is not None
        barrier = state.barriers[event.resource]
        if barrier.lifecycle == "UNINITIALIZED":
            raise ProtocolError("USE_BEFORE_INIT", "mbarrier used before initialization", event.id, barrier.snapshot())
        if barrier.lifecycle == "INVALIDATED":
            raise ProtocolError("USE_AFTER_INVALIDATE", "mbarrier used after invalidation", event.id, barrier.snapshot())
        return barrier

    def _require_current_generation(self, barrier: BarrierState, event: Event) -> None:
        assert event.generation is not None
        if event.generation < barrier.generation:
            raise ProtocolError("STALE_GENERATION", "event targets an already retired generation", event.id, barrier.snapshot())
        if event.generation > barrier.generation + 1:
            raise ProtocolError("PHASE_LAP", "event is more than one generation ahead", event.id, barrier.snapshot())
        if event.generation == barrier.generation + 1:
            raise ProtocolError(
                "EARLY_NEXT_GENERATION",
                "next-generation event reached the barrier before the current phase completed",
                event.id,
                barrier.snapshot(),
            )

    def _arrive(self, barrier: BarrierState, event: Event) -> None:
        assert event.count is not None
        if not event.predicate:
            return
        self._require_current_generation(barrier, event)
        barrier.can_invalidate = False
        barrier.generation_started = True
        barrier.pending -= event.count
        if barrier.pending < 0:
            raise ProtocolError("ARRIVAL_UNDERFLOW", "arrival count underflow", event.id, barrier.snapshot())

    def _expect(self, barrier: BarrierState, event: Event) -> None:
        assert event.bytes is not None
        self._require_current_generation(barrier, event)
        barrier.can_invalidate = False
        barrier.generation_started = True
        barrier.expected_tx += event.bytes

    def _advance_program(self, state: ExecutionState, event: Event) -> None:
        state.executed.add(event.id)
        state.frontiers[event.actor] += 1
        state.completed_transitions.add(event.id)

    def _record(self, state: ExecutionState, transition: Transition) -> None:
        state.trace.append({
            "step": len(state.trace), "transition": transition.id, "kind": transition.kind,
            "barriers": {key: state.barriers[key].snapshot() for key in sorted(state.barriers)},
        })

    def apply(self, state: ExecutionState, transition: Transition) -> None:
        if transition.kind == "TMA_COMPLETION":
            op = state.pending_completions.pop(transition.id)
            barrier = state.barriers[op.resource]
            if op.generation < barrier.generation:
                raise ProtocolError("STALE_COMPLETION", "transaction completed after its generation retired", op.issue_id, barrier.snapshot())
            if op.generation > barrier.generation:
                raise ProtocolError("EARLY_NEXT_GENERATION", "transaction completed before its generation became current", op.issue_id, barrier.snapshot())
            barrier.completed_tx += op.bytes
            if barrier.completed_tx > barrier.expected_tx:
                raise ProtocolError("TX_BYTES_EXCEED_EXPECTED", "completed transaction bytes exceed expected bytes", op.issue_id, barrier.snapshot())
            state.completed_transitions.add(transition.id)
            state.dynamic_edges.add((op.issue_id, transition.id, "ISSUE_TO_COMPLETION"))
            self._record(state, transition)
            return

        if transition.kind == "PHASE_COMPLETION":
            assert transition.resource is not None
            barrier = state.barriers[transition.resource]
            generation = barrier.generation
            barrier.publications.add(generation)
            barrier.publications = {item for item in barrier.publications if item >= generation - 1}
            barrier.generation += 1
            barrier.parity = barrier.generation % 2
            barrier.pending = barrier.base_count
            barrier.expected_tx = 0
            barrier.completed_tx = 0
            barrier.generation_started = False
            barrier.can_invalidate = True
            state.completed_transitions.add(transition.id)
            self._record(state, transition)
            return

        event = transition.event
        assert event is not None
        if event.type == "MBarrierInit":
            barrier = state.barriers[event.resource]
            if barrier.lifecycle == "LIVE":
                raise ProtocolError("REINIT_LIVE", "live mbarrier cannot be reinitialized", event.id, barrier.snapshot())
            assert event.count is not None
            resource = self.module.resource_map[event.resource]
            participant_lanes = {lane for actor_id in resource.participants for lane in self.module.actor_map[actor_id].participants}
            if event.count > len(participant_lanes):
                raise ProtocolError("PARTICIPANT_CONTRACT_UNSATISFIABLE", "mbarrier init count exceeds the statically known participant set", event.id, {"init_count": event.count, "participant_lanes": sorted(participant_lanes)})
            barrier.lifecycle = "LIVE"
            barrier.generation = barrier.parity = 0
            barrier.base_count = barrier.pending = event.count
            barrier.expected_tx = barrier.completed_tx = 0
            barrier.generation_started = False
            barrier.can_invalidate = False
            barrier.publications.clear()
        elif event.type == "MBarrierInvalidate":
            barrier = self._barrier(state, event)
            in_flight = any(op.resource == event.resource for op in state.pending_completions.values())
            if not barrier.can_invalidate or barrier.generation_started or in_flight:
                raise ProtocolError("INVALIDATE_INCOMPLETE", "cannot invalidate before an observed completed phase or with in-flight work", event.id, barrier.snapshot())
            barrier.lifecycle = "INVALIDATED"
            barrier.can_invalidate = False
            barrier.publications.clear()
        elif event.type == "MBarrierArrive":
            self._arrive(self._barrier(state, event), event)
        elif event.type == "MBarrierExpectTx":
            self._expect(self._barrier(state, event), event)
        elif event.type == "MBarrierArriveExpectTx":
            barrier = self._barrier(state, event)
            if event.predicate:
                self._expect(barrier, event)
                self._arrive(barrier, event)
        elif event.type == "MBarrierWaitParity":
            barrier = self._barrier(state, event)
            status = self._wait_status(event, barrier)
            if status == "INVALID":
                expected = event.generation % 2 if event.generation is not None else None
                code = "WRONG_PARITY" if event.raw_parity != expected else "STALE_GENERATION"
                raise ProtocolError(code, "wait parity/generation does not name a live publication", event.id, barrier.snapshot())
            if status == "BLOCKED":
                raise AssertionError("blocked wait was applied")
            state.dynamic_edges.add((stable_phase_id(event.resource, event.generation), event.id, "COMPLETION_TO_WAIT"))
        elif event.type == "TmaLoadIssue":
            barrier = self._barrier(state, event)
            self._require_current_generation(barrier, event)
            barrier.can_invalidate = False
            barrier.generation_started = True
            completion_id = stable_completion_id(event.id, event.generation)
            if completion_id in state.pending_completions or completion_id in state.completed_transitions:
                raise ProtocolError("DUPLICATE_COMPLETION", "completion identity is not unique", event.id)
            state.pending_completions[completion_id] = AsyncOp(completion_id, event.id, event.actor, event.resource, event.destination_region, event.generation, event.bytes)
        self._advance_program(state, event)
        self._record(state, transition)

    def boundary(self, state: ExecutionState) -> tuple[str, ...]:
        return tuple(event.id for actor in sorted(state.frontiers) if (event := self._frontier_event(state, actor)) is not None)

    def unfinished(self, state: ExecutionState) -> bool:
        if any(self._frontier_event(state, actor) is not None for actor in state.frontiers) or state.pending_completions:
            return True
        if any(self._phase_transition(resource, barrier) is not None for resource, barrier in state.barriers.items()):
            return True
        return any(barrier.generation_started for barrier in state.barriers.values() if barrier.lifecycle == "LIVE")

    def fingerprint(self, state: ExecutionState) -> tuple[Any, ...]:
        barriers = tuple((resource, barrier.lifecycle, barrier.generation, barrier.parity, barrier.base_count, barrier.pending,
                          barrier.expected_tx, barrier.completed_tx, barrier.generation_started, barrier.can_invalidate,
                          tuple(sorted(barrier.publications))) for resource, barrier in sorted(state.barriers.items()))
        pending = tuple((key, op.issue_id, op.actor, op.resource, op.destination_region, op.generation, op.bytes)
                        for key, op in sorted(state.pending_completions.items()))
        return (tuple(sorted(state.frontiers.items())), tuple(sorted(state.executed)), barriers, pending, tuple(sorted(state.completed_transitions)))

    def explore(self, hold: str | None = None) -> ExplorationResult:
        state = self.initial_state()
        error: ProtocolError | None = None
        while choices := self.enabled(state, hold):
            try:
                self.apply(state, choices[0])
            except ProtocolError as caught:
                error = caught
                break
        return ExplorationResult(state, self.boundary(state), error, error is None and hold is None and self.unfinished(state), hold)

    def explore_all(self, hold: str | None = None) -> ExhaustiveResult:
        stack = [self.initial_state()]
        seen: set[tuple[Any, ...]] = set()
        terminals: list[ExecutionState] = []
        errors: list[tuple[ProtocolError, ExecutionState]] = []
        stuck: list[ExecutionState] = []
        while stack:
            state = stack.pop()
            marker = self.fingerprint(state)
            if marker in seen:
                continue
            seen.add(marker)
            if len(seen) > self.MAX_STATES:
                raise ExplorationLimit(f"state exploration exceeded {self.MAX_STATES} states")
            choices = self.enabled(state, hold)
            if not choices:
                terminals.append(state)
                if hold is None and self.unfinished(state):
                    stuck.append(state)
                continue
            for transition in reversed(choices):
                successor = state.clone()
                try:
                    self.apply(successor, transition)
                except ProtocolError as error:
                    errors.append((error, successor))
                else:
                    stack.append(successor)
        return ExhaustiveResult(terminals, errors, stuck, len(seen))

    def probe(self, state: ExecutionState, anchor: str) -> tuple[ProtocolError | None, ExecutionState | None]:
        transition = next((item for item in self.enabled(state) if item.id == anchor), None)
        if transition is None:
            return None, None
        successor = state.clone()
        try:
            self.apply(successor, transition)
        except ProtocolError as error:
            return error, successor
        return None, successor

    def stuck_diagnostic(self, state: ExecutionState) -> dict[str, Any]:
        for resource_id in sorted(state.barriers):
            barrier = state.barriers[resource_id]
            if barrier.generation_started:
                if barrier.pending > 0:
                    return {"code": "MISSING_ARRIVAL", "message": "mbarrier phase cannot complete: arrivals are missing", "resource": resource_id, "protocol_state": barrier.snapshot()}
                pending_bytes = sum(op.bytes for op in state.pending_completions.values() if op.resource == resource_id and op.generation == barrier.generation)
                if barrier.completed_tx + pending_bytes != barrier.expected_tx:
                    return {"code": "TX_BYTES_MISMATCH", "message": "mbarrier expected/completable transaction bytes do not match", "resource": resource_id, "protocol_state": barrier.snapshot()}
        waits = [event.id for actor in sorted(state.frontiers) if (event := self._frontier_event(state, actor)) is not None and event.type == "MBarrierWaitParity"]
        if waits:
            return {"code": "STUCK_WAIT", "message": "no transition can satisfy one or more waits", "events": waits}
        return {"code": "STUCK_DEPENDENCY", "message": "no program or completion transition is enabled", "boundary": list(self.boundary(state))}
