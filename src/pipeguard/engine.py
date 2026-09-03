from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from .model import Event, Module, stable_collective_release_id, stable_completion_id, stable_phase_id
from .protocols import CollectiveBarrierState


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
    lifecycle_epoch: int
    generation: int
    bytes: int


@dataclass(slots=True)
class BarrierState:
    lifecycle: Literal["UNINITIALIZED", "LIVE", "INVALIDATED"] = "UNINITIALIZED"
    lifecycle_epoch: int = -1
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
            "lifecycle_epoch": self.lifecycle_epoch,
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
    kind: Literal["PROGRAM", "TMA_COMPLETION", "PHASE_COMPLETION", "BARRIER_RELEASE"]
    event: Event | None = None
    resource: str | None = None

    def order_key(self) -> tuple[Any, ...]:
        if self.kind != "PROGRAM":
            return (0, self.id)
        assert self.event is not None
        return (1, self.event.actor, self.event.seq, self.event.id)


@dataclass(frozen=True, slots=True)
class TransitionDelta:
    """Committed state changes used by notification-driven schedulers.

    The delta is returned only after a transition succeeds. Reference callers
    may ignore it; incremental callers use it to update readiness without a
    global enabled-transition scan.
    """

    transition_id: str
    kind: Literal["PROGRAM", "TMA_COMPLETION", "PHASE_COMPLETION", "BARRIER_RELEASE"]
    advanced_actor: str | None = None
    executed_event: str | None = None
    changed_resources: frozenset[str] = frozenset()
    changed_collectives: frozenset[str] = frozenset()
    spawned_completions: tuple[str, ...] = ()
    retired_completions: tuple[str, ...] = ()
    consumed_waits: tuple[str, ...] = ()
    dynamic_edges: tuple[tuple[str, str, str], ...] = ()


@dataclass(slots=True)
class ExecutionState:
    frontiers: dict[str, int]
    executed: set[str]
    barriers: dict[str, BarrierState]
    collectives: dict[str, CollectiveBarrierState] = field(default_factory=dict)
    remaining: dict[tuple[str, int, int], set[str]] = field(default_factory=dict)
    observed: set[str] = field(default_factory=set)
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
        self.required_logical_waits = module.required_logical_waits

    def initial_state(self) -> ExecutionState:
        return ExecutionState(
            frontiers={actor.id: 0 for actor in self.module.actors},
            executed=set(),
            barriers={
                resource.id: BarrierState()
                for resource in self.module.resources if resource.kind == "mbarrier"
            },
            collectives={
                resource.id: CollectiveBarrierState(resource.participants, resource.capacity)
                for resource in self.module.resources
                if resource.kind in {"cta_barrier", "named_barrier"}
            },
            remaining={key: set(event_ids) for key, event_ids in self.required_logical_waits.items()},
            observed=set(),
        )

    def _frontier_event(self, state: ExecutionState, actor: str) -> Event | None:
        index = state.frontiers[actor]
        trace = self.module.events_by_actor[actor]
        return trace[index] if index < len(trace) else None

    def _phase_transition(self, resource_id: str, barrier: BarrierState) -> Transition | None:
        if barrier.lifecycle == "LIVE" and barrier.generation_started and barrier.pending == 0 and barrier.completed_tx == barrier.expected_tx:
            return Transition(stable_phase_id(resource_id, barrier.lifecycle_epoch, barrier.generation), "PHASE_COMPLETION", resource=resource_id)
        return None

    def _collective_transition(
        self,
        resource_id: str,
        collective: CollectiveBarrierState,
    ) -> Transition | None:
        if collective.arrived_count == collective.capacity:
            return Transition(
                stable_collective_release_id(resource_id, collective.invocation),
                "BARRIER_RELEASE",
                resource=resource_id,
            )
        return None

    def _collective_observe_status(
        self,
        event: Event,
        collective: CollectiveBarrierState,
    ) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.invocation_id is not None
        if event.invocation_id in collective.publications:
            return "READY"
        if event.invocation_id >= collective.invocation:
            return "BLOCKED"
        return "INVALID"

    def _wait_status(self, event: Event, barrier: BarrierState) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.generation is not None and event.raw_parity is not None and event.parity_evidence is not None
        if barrier.lifecycle != "LIVE" or event.parity_evidence.lifecycle_epoch != barrier.lifecycle_epoch or event.raw_parity != event.generation % 2:
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
        for resource_id in sorted(state.collectives):
            transition = self._collective_transition(resource_id, state.collectives[resource_id])
            if transition is not None and transition.id != hold:
                result.append(transition)
        for actor in sorted(state.frontiers):
            event = self._frontier_event(state, actor)
            if event is None or event.id == hold or not set(event.depends_on).issubset(state.executed):
                continue
            if event.type == "MBarrierWaitParity" and self._wait_status(event, state.barriers[event.resource]) == "BLOCKED":
                continue
            if event.type == "BarrierObserve" and self._collective_observe_status(
                event, state.collectives[event.resource]
            ) == "BLOCKED":
                continue
            if event.resource is not None and event.generation is not None:
                barrier = state.barriers[event.resource]
                phase = self._phase_transition(event.resource, barrier)
                if event.generation == barrier.generation + 1 and phase is not None:
                    # A normalized future-generation event denotes an instruction
                    # after the hardware phase flip. It cannot execute before an
                    # already-enabled synthetic phase transition. A merely held tx
                    # completion has not flipped the phase, so an independent early
                    # Arrive must remain explorable and fail the pending-count check.
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
        assert event.lifecycle_epoch is not None
        if event.lifecycle_epoch != barrier.lifecycle_epoch:
            code = "STALE_LIFECYCLE" if event.lifecycle_epoch < barrier.lifecycle_epoch else "FUTURE_LIFECYCLE"
            raise ProtocolError(code, "event lifecycle epoch does not match the live mbarrier lifecycle", event.id, barrier.snapshot())
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

    def _validate_arrive(self, barrier: BarrierState, event: Event) -> None:
        assert event.count is not None and event.generation is not None
        if event.generation < barrier.generation or event.generation > barrier.generation + 1:
            self._require_current_generation(barrier, event)
        if barrier.pending == 0:
            raise ProtocolError(
                "ARRIVE_WITH_ZERO_PENDING_COUNT",
                "mbarrier arrive requires a nonzero pending arrival count",
                event.id,
                barrier.snapshot(),
            )
        self._require_current_generation(barrier, event)
        if event.count > barrier.pending:
            snapshot = barrier.snapshot()
            snapshot["arrival_count"] = event.count
            raise ProtocolError("ARRIVAL_UNDERFLOW", "arrival count underflow", event.id, snapshot)

    def _arrive(self, state: ExecutionState, barrier: BarrierState, event: Event) -> None:
        assert event.count is not None
        if not event.predicate:
            return
        self._validate_arrive(barrier, event)
        barrier.can_invalidate = False
        barrier.generation_started = True
        barrier.pending -= event.count

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
        })

    def apply(self, state: ExecutionState, transition: Transition) -> TransitionDelta:
        if transition.kind == "TMA_COMPLETION":
            op = state.pending_completions.get(transition.id)
            if op is None:
                raise ProtocolError(
                    "INVALID_TMA_COMPLETION_TRANSITION",
                    "TMA completion is not pending in the current execution state",
                    transition.id,
                )
            barrier = state.barriers[op.resource]
            if op.lifecycle_epoch != barrier.lifecycle_epoch:
                raise ProtocolError("STALE_COMPLETION", "transaction completed outside its mbarrier lifecycle", op.issue_id, barrier.snapshot())
            if op.generation < barrier.generation:
                raise ProtocolError("STALE_COMPLETION", "transaction completed after its generation retired", op.issue_id, barrier.snapshot())
            if op.generation > barrier.generation:
                raise ProtocolError("EARLY_NEXT_GENERATION", "transaction completed before its generation became current", op.issue_id, barrier.snapshot())
            completed_tx = barrier.completed_tx + op.bytes
            if completed_tx > barrier.expected_tx:
                snapshot = barrier.snapshot()
                snapshot["completion_bytes"] = op.bytes
                snapshot["completed_tx_bytes_after"] = completed_tx
                raise ProtocolError("TX_BYTES_EXCEED_EXPECTED", "completed transaction bytes exceed expected bytes", op.issue_id, snapshot)
            edge = (op.issue_id, transition.id, "ISSUE_TO_COMPLETION")
            state.pending_completions.pop(transition.id)
            barrier.completed_tx = completed_tx
            state.completed_transitions.add(transition.id)
            state.dynamic_edges.add(edge)
            self._record(state, transition)
            return TransitionDelta(
                transition.id,
                transition.kind,
                changed_resources=frozenset({op.resource}),
                retired_completions=(transition.id,),
                dynamic_edges=(edge,),
            )

        if transition.kind == "BARRIER_RELEASE":
            assert transition.resource is not None
            collective = state.collectives[transition.resource]
            enabled_transition = self._collective_transition(transition.resource, collective)
            if enabled_transition is None or enabled_transition.id != transition.id:
                snapshot = collective.snapshot()
                snapshot["expected_barrier_release"] = (
                    enabled_transition.id if enabled_transition is not None else None
                )
                raise ProtocolError(
                    "INVALID_BARRIER_RELEASE_TRANSITION",
                    "collective barrier release is not enabled for this invocation",
                    transition.id,
                    snapshot,
                )
            invocation = collective.invocation
            edges = tuple(
                (event_id, transition.id, "BARRIER_ARRIVE")
                for event_id, _count in sorted(collective.arrivals.values())
            )
            collective.publications.add(invocation)
            collective.observations.setdefault(invocation, set())
            collective.invocation += 1
            collective.arrivals.clear()
            state.completed_transitions.add(transition.id)
            state.dynamic_edges.update(edges)
            self._record(state, transition)
            return TransitionDelta(
                transition.id,
                transition.kind,
                changed_collectives=frozenset({transition.resource}),
                dynamic_edges=edges,
            )

        if transition.kind == "PHASE_COMPLETION":
            assert transition.resource is not None
            barrier = state.barriers[transition.resource]
            enabled_transition = self._phase_transition(transition.resource, barrier)
            if enabled_transition is None or enabled_transition.id != transition.id:
                snapshot = barrier.snapshot()
                snapshot["expected_phase_completion"] = enabled_transition.id if enabled_transition is not None else None
                raise ProtocolError(
                    "INVALID_PHASE_COMPLETION_TRANSITION",
                    "phase completion is not enabled for the current barrier generation",
                    transition.id,
                    snapshot,
                )
            generation = barrier.generation
            if generation > 0:
                key = (transition.resource, barrier.lifecycle_epoch, generation - 1)
                outstanding = state.remaining.get(key, set())
                if outstanding:
                    snapshot = barrier.snapshot()
                    snapshot["stale_target_generation"] = generation - 1
                    snapshot["remaining_logical_waits"] = sorted(outstanding)
                    raise ProtocolError(
                        "PROTOCOL_INVALID_PHASE_ADVANCE",
                        "phase completion would retire an unobserved required LogicalWait target",
                        transition.id,
                        snapshot,
                    )
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
            return TransitionDelta(
                transition.id,
                transition.kind,
                changed_resources=frozenset({transition.resource}),
            )

        event = transition.event
        assert event is not None
        changed_resources: set[str] = set()
        changed_collectives: set[str] = set()
        spawned_completions: tuple[str, ...] = ()
        consumed_waits: tuple[str, ...] = ()
        added_edges: tuple[tuple[str, str, str], ...] = ()
        if event.type == "BarrierArrive":
            assert event.resource is not None and event.invocation_id is not None and event.count is not None
            collective = state.collectives[event.resource]
            if event.predicate:
                if event.invocation_id < collective.invocation:
                    raise ProtocolError(
                        "STALE_BARRIER_INVOCATION",
                        "collective arrival targets an already released invocation",
                        event.id,
                        collective.snapshot(),
                    )
                if event.invocation_id > collective.invocation:
                    raise ProtocolError(
                        "FUTURE_BARRIER_INVOCATION",
                        "collective arrival skipped the current invocation",
                        event.id,
                        collective.snapshot(),
                    )
                if event.actor in collective.arrivals:
                    raise ProtocolError(
                        "DUPLICATE_BARRIER_ARRIVAL",
                        "actor arrived more than once for one collective invocation",
                        event.id,
                        collective.snapshot(),
                    )
                if collective.arrived_count + event.count > collective.capacity:
                    snapshot = collective.snapshot()
                    snapshot["arrival_count"] = event.count
                    raise ProtocolError(
                        "BARRIER_ARRIVAL_OVERFLOW",
                        "collective arrival exceeds the participant capacity",
                        event.id,
                        snapshot,
                    )
                collective.arrivals[event.actor] = (event.id, event.count)
                changed_collectives.add(event.resource)
        elif event.type == "BarrierObserve":
            assert event.resource is not None and event.invocation_id is not None
            collective = state.collectives[event.resource]
            status = self._collective_observe_status(event, collective)
            if status == "INVALID":
                raise ProtocolError(
                    "STALE_BARRIER_OBSERVE",
                    "collective observe targets an invocation without a retained release",
                    event.id,
                    collective.snapshot(),
                )
            if status == "BLOCKED":
                raise AssertionError("blocked collective observe was applied")
            observed = collective.observations.setdefault(event.invocation_id, set())
            if event.actor in observed:
                raise ProtocolError(
                    "DUPLICATE_BARRIER_OBSERVE",
                    "actor observed one collective invocation more than once",
                    event.id,
                    collective.snapshot(),
                )
            observed.add(event.actor)
            release = stable_collective_release_id(event.resource, event.invocation_id)
            edge = (release, event.id, "BARRIER_RELEASE")
            state.dynamic_edges.add(edge)
            added_edges = (edge,)
            changed_collectives.add(event.resource)
        elif event.type == "MBarrierInit":
            barrier = state.barriers[event.resource]
            if barrier.lifecycle == "LIVE":
                raise ProtocolError("REINIT_LIVE", "live mbarrier cannot be reinitialized", event.id, barrier.snapshot())
            assert event.count is not None
            resource = self.module.resource_map[event.resource]
            participant_lanes = {lane for actor_id in resource.participants for lane in self.module.actor_map[actor_id].participants}
            if event.count > len(participant_lanes):
                raise ProtocolError("PARTICIPANT_CONTRACT_UNSATISFIABLE", "mbarrier init count exceeds the statically known participant set", event.id, {"init_count": event.count, "participant_lanes": sorted(participant_lanes)})
            assert event.lifecycle_epoch is not None
            expected_lifecycle = barrier.lifecycle_epoch + 1
            if event.lifecycle_epoch != expected_lifecycle:
                raise ProtocolError("LIFECYCLE_EPOCH_MISMATCH", "mbarrier init must start the next lifecycle epoch", event.id, {"expected_lifecycle_epoch": expected_lifecycle, "event_lifecycle_epoch": event.lifecycle_epoch})
            barrier.lifecycle = "LIVE"
            barrier.lifecycle_epoch = event.lifecycle_epoch
            barrier.generation = barrier.parity = 0
            barrier.base_count = barrier.pending = event.count
            barrier.expected_tx = barrier.completed_tx = 0
            barrier.generation_started = False
            barrier.can_invalidate = False
            barrier.publications.clear()
            changed_resources.add(event.resource)
        elif event.type == "MBarrierInvalidate":
            barrier = self._barrier(state, event)
            in_flight = any(op.resource == event.resource for op in state.pending_completions.values())
            completed_target = barrier.generation - 1
            outstanding = state.remaining.get((event.resource, barrier.lifecycle_epoch, completed_target), set()) if completed_target >= 0 else set()
            if not barrier.can_invalidate or barrier.generation_started or in_flight or outstanding:
                snapshot = barrier.snapshot()
                if outstanding:
                    snapshot["remaining_logical_waits"] = sorted(outstanding)
                raise ProtocolError("INVALIDATE_INCOMPLETE", "cannot invalidate before an observed completed phase or with in-flight work", event.id, snapshot)
            barrier.lifecycle = "INVALIDATED"
            barrier.can_invalidate = False
            barrier.publications.clear()
            changed_resources.add(event.resource)
        elif event.type == "MBarrierArrive":
            self._arrive(state, self._barrier(state, event), event)
            if event.predicate:
                changed_resources.add(event.resource)
        elif event.type == "MBarrierExpectTx":
            self._expect(self._barrier(state, event), event)
            changed_resources.add(event.resource)
        elif event.type == "MBarrierArriveExpectTx":
            barrier = self._barrier(state, event)
            if event.predicate:
                self._validate_arrive(barrier, event)
                self._expect(barrier, event)
                self._arrive(state, barrier, event)
                changed_resources.add(event.resource)
        elif event.type == "MBarrierWaitParity":
            barrier = self._barrier(state, event)
            status = self._wait_status(event, barrier)
            if status == "INVALID":
                assert event.parity_evidence is not None and event.generation is not None
                expected = event.generation % 2
                if event.raw_parity != expected:
                    code = "WRONG_PARITY"
                elif event.parity_evidence.lifecycle_epoch != barrier.lifecycle_epoch:
                    code = "STALE_LIFECYCLE"
                elif event.generation < barrier.generation - 1:
                    code = "STALE_PHASE_WAIT"
                else:
                    code = "INVALID_WAIT_PHASE"
                raise ProtocolError(code, "wait parity/generation does not name a current or immediately preceding phase", event.id, barrier.snapshot())
            if status == "BLOCKED":
                raise AssertionError("blocked wait was applied")
            assert event.resource is not None and event.lifecycle_epoch is not None and event.generation is not None
            key = (event.resource, event.lifecycle_epoch, event.generation)
            bucket = state.remaining.get(key)
            if event.id in state.observed or bucket is None or event.id not in bucket:
                raise ProtocolError("DUPLICATE_LOGICAL_WAIT_CONSUMPTION", "logical wait was already consumed or is not required by the completed phase", event.id, barrier.snapshot())
            bucket.remove(event.id)
            if not bucket:
                state.remaining.pop(key)
            state.observed.add(event.id)
            edge = (stable_phase_id(event.resource, event.lifecycle_epoch, event.generation), event.id, "COMPLETION_TO_WAIT")
            state.dynamic_edges.add(edge)
            consumed_waits = (event.id,)
            added_edges = (edge,)
        elif event.type == "TmaLoadIssue":
            barrier = self._barrier(state, event)
            self._require_current_generation(barrier, event)
            completion_id = stable_completion_id(event.id, event.generation)
            if completion_id in state.pending_completions or completion_id in state.completed_transitions:
                raise ProtocolError("DUPLICATE_COMPLETION", "completion identity is not unique", event.id)
            barrier.can_invalidate = False
            barrier.generation_started = True
            state.pending_completions[completion_id] = AsyncOp(completion_id, event.id, event.actor, event.resource, event.destination_region, event.lifecycle_epoch, event.generation, event.bytes)
            changed_resources.add(event.resource)
            spawned_completions = (completion_id,)
        self._advance_program(state, event)
        self._record(state, transition)
        return TransitionDelta(
            transition.id,
            transition.kind,
            advanced_actor=event.actor,
            executed_event=event.id,
            changed_resources=frozenset(changed_resources),
            changed_collectives=frozenset(changed_collectives),
            spawned_completions=spawned_completions,
            consumed_waits=consumed_waits,
            dynamic_edges=added_edges,
        )

    def boundary(self, state: ExecutionState) -> tuple[str, ...]:
        return tuple(event.id for actor in sorted(state.frontiers) if (event := self._frontier_event(state, actor)) is not None)

    def unfinished(self, state: ExecutionState) -> bool:
        if any(self._frontier_event(state, actor) is not None for actor in state.frontiers) or state.pending_completions:
            return True
        if any(self._phase_transition(resource, barrier) is not None for resource, barrier in state.barriers.items()):
            return True
        if any(
            collective.arrivals or self._collective_transition(resource, collective) is not None
            for resource, collective in state.collectives.items()
        ):
            return True
        return any(barrier.generation_started for barrier in state.barriers.values() if barrier.lifecycle == "LIVE")

    def fingerprint(self, state: ExecutionState) -> tuple[Any, ...]:
        barriers = tuple((resource, barrier.lifecycle, barrier.lifecycle_epoch, barrier.generation, barrier.parity, barrier.base_count, barrier.pending,
                          barrier.expected_tx, barrier.completed_tx, barrier.generation_started, barrier.can_invalidate,
                          tuple(sorted(barrier.publications))) for resource, barrier in sorted(state.barriers.items()))
        collectives = tuple(
            (
                resource,
                collective.participants,
                collective.capacity,
                collective.invocation,
                tuple(sorted((actor, event, count) for actor, (event, count) in collective.arrivals.items())),
                tuple(sorted(collective.publications)),
                tuple(
                    (invocation, tuple(sorted(actors)))
                    for invocation, actors in sorted(collective.observations.items())
                ),
            )
            for resource, collective in sorted(state.collectives.items())
        )
        remaining = tuple((key, tuple(sorted(event_ids))) for key, event_ids in sorted(state.remaining.items()))
        pending = tuple((key, op.issue_id, op.actor, op.resource, op.destination_region, op.lifecycle_epoch, op.generation, op.bytes)
                        for key, op in sorted(state.pending_completions.items()))
        return (tuple(sorted(state.frontiers.items())), tuple(sorted(state.executed)), barriers, collectives, remaining,
                tuple(sorted(state.observed)), pending, tuple(sorted(state.completed_transitions)))

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
        for resource_id in sorted(state.collectives):
            collective = state.collectives[resource_id]
            if collective.arrivals:
                return {
                    "code": "MISSING_BARRIER_PARTICIPANT",
                    "message": "collective barrier invocation cannot release because participants are missing",
                    "resource": resource_id,
                    "protocol_state": collective.snapshot(),
                }
        waits = [event.id for actor in sorted(state.frontiers) if (event := self._frontier_event(state, actor)) is not None and event.type == "MBarrierWaitParity"]
        if waits:
            return {"code": "STUCK_WAIT", "message": "no transition can satisfy one or more waits", "events": waits}
        observes = [
            event.id for actor in sorted(state.frontiers)
            if (event := self._frontier_event(state, actor)) is not None and event.type == "BarrierObserve"
        ]
        if observes:
            return {
                "code": "STUCK_BARRIER_OBSERVE",
                "message": "no collective release can satisfy one or more barrier observes",
                "events": observes,
            }
        return {"code": "STUCK_DEPENDENCY", "message": "no program or completion transition is enabled", "boundary": list(self.boundary(state))}
