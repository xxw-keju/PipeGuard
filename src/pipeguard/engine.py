from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from .model import (
    Event,
    Module,
    engine_completion_streams,
    stable_channel_handoff_id,
    stable_collective_release_id,
    stable_completion_id,
    stable_group_completion_id,
    stable_group_id,
    stable_phase_id,
)
from .protocols import (
    CollectiveBarrierState,
    CommittedGroup,
    CommitGroupState,
    ProducerConsumerChannelState,
    VisibilityTransfer,
)


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
    kind: Literal["PROGRAM", "TMA_COMPLETION", "PHASE_COMPLETION", "BARRIER_RELEASE", "GROUP_COMPLETION", "CHANNEL_HANDOFF"]
    event: Event | None = None
    resource: str | None = None
    owner: str | None = None
    completion_kind: str | None = None
    slot: int | None = None

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
    kind: Literal["PROGRAM", "TMA_COMPLETION", "PHASE_COMPLETION", "BARRIER_RELEASE", "GROUP_COMPLETION", "CHANNEL_HANDOFF"]
    advanced_actor: str | None = None
    executed_event: str | None = None
    changed_resources: frozenset[str] = frozenset()
    changed_collectives: frozenset[str] = frozenset()
    changed_groups: frozenset[tuple[str, str]] = frozenset()
    changed_channels: frozenset[tuple[str, int]] = frozenset()
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
    groups: dict[str, CommitGroupState] = field(default_factory=dict)
    channels: dict[str, ProducerConsumerChannelState] = field(default_factory=dict)
    visibility_transfers: list[VisibilityTransfer] = field(default_factory=list)
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
        self.engine_streams = engine_completion_streams(module)
        self.engine_predecessors: dict[str, str] = {}
        self.engine_successors: dict[str, tuple[str, ...]] = {}
        successors: dict[str, list[str]] = {}
        for endpoints in self.engine_streams.values():
            for predecessor, successor in zip(endpoints, endpoints[1:]):
                self.engine_predecessors[successor] = predecessor
                successors.setdefault(predecessor, []).append(successor)
        self.engine_successors = {
            predecessor: tuple(sorted(items))
            for predecessor, items in sorted(successors.items())
        }

    def _engine_completion_ready(self, state: ExecutionState, completion: str) -> bool:
        predecessor = self.engine_predecessors.get(completion)
        return predecessor is None or predecessor in state.completed_transitions

    def _engine_order_state(self, state: ExecutionState, completion: str) -> dict[str, Any]:
        predecessor = self.engine_predecessors.get(completion)
        return {
            "completion": completion,
            "required_predecessor": predecessor,
            "predecessor_completed": predecessor in state.completed_transitions if predecessor else True,
        }

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
                if resource.kind in {"cta_barrier", "named_barrier", "cluster_barrier"}
            },
            groups={
                resource.id: CommitGroupState(
                    resource.protocol,
                    resource.participants,
                    resource.capacity,
                    {
                        "CP_ASYNC": ("GROUP_COMPLETE",),
                        "WGMMA": ("OPERANDS_RELEASED", "RESULT_READY"),
                        "TMA_STORE": ("SOURCE_RELEASED", "GLOBAL_COMPLETE"),
                    }[resource.protocol],
                )
                for resource in self.module.resources
                if resource.kind == "async_group"
                and resource.protocol in {"CP_ASYNC", "WGMMA", "TMA_STORE"}
            },
            channels={
                resource.id: ProducerConsumerChannelState(
                    resource.participants,
                    resource.capacity,
                )
                for resource in self.module.resources
                if resource.kind == "channel"
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

    def _channel_transition(
        self,
        resource_id: str,
        slot: int,
        channel: ProducerConsumerChannelState,
    ) -> Transition | None:
        if not channel.initialized:
            return None
        slot_state = channel.slots[slot]
        if slot_state.state != "COMMITTED":
            return None
        return Transition(
            stable_channel_handoff_id(resource_id, slot, slot_state.logical_phase),
            "CHANNEL_HANDOFF",
            resource=resource_id,
            slot=slot,
        )

    def _producer_acquire_status(
        self,
        event: Event,
        channel: ProducerConsumerChannelState,
    ) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.slot is not None and event.logical_phase is not None
        if not channel.initialized:
            return "BLOCKED"
        slot_state = channel.slots[event.slot]
        if event.logical_phase < slot_state.logical_phase:
            return "INVALID"
        if event.logical_phase > slot_state.logical_phase + 1:
            return "INVALID"
        if event.logical_phase == slot_state.logical_phase + 1:
            return "BLOCKED"
        return "READY" if slot_state.state == "PRODUCER_AVAILABLE" else "INVALID"

    def _consumer_wait_status(
        self,
        event: Event,
        channel: ProducerConsumerChannelState,
    ) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.slot is not None and event.logical_phase is not None
        if not channel.initialized:
            return "BLOCKED"
        slot_state = channel.slots[event.slot]
        if event.logical_phase < slot_state.logical_phase:
            return "INVALID"
        if event.logical_phase > slot_state.logical_phase + 1:
            return "INVALID"
        if event.logical_phase > slot_state.logical_phase:
            return "BLOCKED"
        if slot_state.state == "CONSUMER_AVAILABLE":
            return "READY"
        if slot_state.state in {"PRODUCER_AVAILABLE", "PRODUCER_OWNED", "COMMITTED"}:
            return "BLOCKED"
        return "INVALID"

    def _group_transition(
        self,
        resource_id: str,
        owner: str,
        completion_kind: str,
        group_state: CommitGroupState,
        state: ExecutionState,
    ) -> Transition | None:
        pending = group_state.pending(owner, completion_kind)
        if not pending:
            return None
        group = pending[0]
        if (
            group_state.protocol == "TMA_STORE"
            and completion_kind == "GLOBAL_COMPLETE"
            and not group.is_completed("SOURCE_RELEASED")
        ):
            return None
        completion = group.completion_id(completion_kind)
        if not self._engine_completion_ready(state, completion):
            return None
        return Transition(
            completion,
            "GROUP_COMPLETION",
            resource=resource_id,
            owner=owner,
            completion_kind=completion_kind,
        )

    def _group_wait_status(
        self,
        event: Event,
        group_state: CommitGroupState,
        completion_kind: str,
    ) -> Literal["READY", "BLOCKED", "INVALID"]:
        assert event.group_owner is not None and event.wait_group_count is not None
        if group_state.staged.get(event.group_owner):
            return "INVALID"
        required = group_state.required_for_wait(
            event.group_owner,
            event.wait_group_count,
            completion_kind,
        )
        return (
            "READY"
            if all(group.is_completed(completion_kind) for group in required)
            else "BLOCKED"
        )

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
            if completion_id != hold and self._engine_completion_ready(state, completion_id):
                result.append(Transition(completion_id, "TMA_COMPLETION"))
        for resource_id in sorted(state.barriers):
            transition = self._phase_transition(resource_id, state.barriers[resource_id])
            if transition is not None and transition.id != hold:
                result.append(transition)
        for resource_id in sorted(state.collectives):
            transition = self._collective_transition(resource_id, state.collectives[resource_id])
            if transition is not None and transition.id != hold:
                result.append(transition)
        for resource_id in sorted(state.groups):
            group_state = state.groups[resource_id]
            for owner in sorted(group_state.participants):
                for completion_kind in group_state.completion_kinds:
                    transition = self._group_transition(
                        resource_id,
                        owner,
                        completion_kind,
                        group_state,
                        state,
                    )
                    if transition is not None and transition.id != hold:
                        result.append(transition)
        for resource_id in sorted(state.channels):
            channel = state.channels[resource_id]
            if channel.initialized:
                for slot in sorted(channel.slots):
                    transition = self._channel_transition(resource_id, slot, channel)
                    if transition is not None and transition.id != hold:
                        result.append(transition)
        for actor in sorted(state.frontiers):
            event = self._frontier_event(state, actor)
            if event is None or event.id == hold or not set(event.depends_on).issubset(state.executed):
                continue
            if (
                event.type == "MBarrierInvalidate"
                and self._phase_transition(event.resource, state.barriers[event.resource]) is not None
            ):
                continue
            if event.type == "MBarrierWaitParity" and self._wait_status(event, state.barriers[event.resource]) == "BLOCKED":
                continue
            if event.type == "BarrierObserve" and self._collective_observe_status(
                event, state.collectives[event.resource]
            ) == "BLOCKED":
                continue
            if event.type == "CpAsyncWaitGroup" and self._group_wait_status(
                event,
                state.groups[event.resource],
                "GROUP_COMPLETE",
            ) == "BLOCKED":
                continue
            if event.type == "WgmmaWaitGroup" and self._group_wait_status(
                event,
                state.groups[event.resource],
                event.completion_kind,
            ) == "BLOCKED":
                continue
            if event.type == "TmaStoreWaitGroup" and self._group_wait_status(
                event,
                state.groups[event.resource],
                "SOURCE_RELEASED" if event.wait_mode == "READ_ONLY" else "GLOBAL_COMPLETE",
            ) == "BLOCKED":
                continue
            if event.type == "ProducerAcquire" and self._producer_acquire_status(
                event,
                state.channels[event.resource],
            ) == "BLOCKED":
                continue
            if event.type == "ConsumerWait" and self._consumer_wait_status(
                event,
                state.channels[event.resource],
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
            if not self._engine_completion_ready(state, transition.id):
                raise ProtocolError(
                    "ENGINE_ORDER_VIOLATION",
                    "async completion occurred before its explicit engine predecessor",
                    transition.id,
                    self._engine_order_state(state, transition.id),
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

        if transition.kind == "GROUP_COMPLETION":
            if (
                transition.resource is None
                or transition.owner is None
                or transition.completion_kind is None
            ):
                raise ProtocolError(
                    "INVALID_GROUP_COMPLETION_TRANSITION",
                    "group completion lacks its resource, owner, or endpoint identity",
                    transition.id,
                )
            if not self._engine_completion_ready(state, transition.id):
                raise ProtocolError(
                    "ENGINE_ORDER_VIOLATION",
                    "async completion occurred before its explicit engine predecessor",
                    transition.id,
                    self._engine_order_state(state, transition.id),
                )
            group_state = state.groups.get(transition.resource)
            enabled_transition = (
                self._group_transition(
                    transition.resource,
                    transition.owner,
                    transition.completion_kind,
                    group_state,
                    state,
                )
                if group_state is not None
                else None
            )
            if enabled_transition is None or enabled_transition.id != transition.id:
                snapshot = group_state.snapshot(transition.owner) if group_state is not None else {}
                snapshot["expected_group_completion"] = (
                    enabled_transition.id if enabled_transition is not None else None
                )
                raise ProtocolError(
                    "INVALID_GROUP_COMPLETION_TRANSITION",
                    "group completion is not the next enabled endpoint for this owner",
                    transition.id,
                    snapshot,
                )
            group = group_state.pending(
                transition.owner,
                transition.completion_kind,
            )[0]
            edges = [(group.group_id, transition.id, "GROUP_TO_COMPLETION")]
            if (
                group_state.protocol == "TMA_STORE"
                and transition.completion_kind == "GLOBAL_COMPLETE"
            ):
                edges.append((
                    group.completion_id("SOURCE_RELEASED"),
                    transition.id,
                    "ENDPOINT_ORDER",
                ))
            group.mark_completed(transition.completion_kind)
            state.completed_transitions.add(transition.id)
            state.dynamic_edges.update(edges)
            self._record(state, transition)
            return TransitionDelta(
                transition.id,
                transition.kind,
                changed_groups=frozenset({(transition.resource, transition.owner)}),
                retired_completions=(transition.id,),
                dynamic_edges=tuple(edges),
            )

        if transition.kind == "CHANNEL_HANDOFF":
            if transition.resource is None or transition.slot is None:
                raise ProtocolError(
                    "INVALID_CHANNEL_HANDOFF_TRANSITION",
                    "channel handoff lacks its resource or slot identity",
                    transition.id,
                )
            channel = state.channels.get(transition.resource)
            expected = (
                self._channel_transition(transition.resource, transition.slot, channel)
                if channel is not None and channel.initialized and transition.slot in channel.slots
                else None
            )
            if expected is None or expected.id != transition.id:
                snapshot = channel.snapshot(transition.slot) if channel is not None else {}
                snapshot["expected_channel_handoff"] = expected.id if expected is not None else None
                raise ProtocolError(
                    "INVALID_CHANNEL_HANDOFF_TRANSITION",
                    "channel handoff is not enabled for this slot and phase",
                    transition.id,
                    snapshot,
                )
            slot_state = channel.slots[transition.slot]
            edge = (slot_state.commit_event, transition.id, "CHANNEL_COMMIT")
            slot_state.state = "CONSUMER_AVAILABLE"
            slot_state.handoff_id = transition.id
            channel.handoffs[(transition.slot, slot_state.logical_phase)] = transition.id
            state.completed_transitions.add(transition.id)
            state.dynamic_edges.add(edge)
            self._record(state, transition)
            return TransitionDelta(
                transition.id,
                transition.kind,
                changed_channels=frozenset({(transition.resource, transition.slot)}),
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
        changed_groups: set[tuple[str, str]] = set()
        changed_channels: set[tuple[str, int]] = set()
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
        elif event.type == "CpAsyncIssue":
            assert event.resource is not None and event.group_owner is not None
            group_state = state.groups[event.resource]
            if event.predicate:
                group_state.staged_operations(event.group_owner).append(event.id)
                changed_groups.add((event.resource, event.group_owner))
        elif event.type == "CpAsyncCommitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.group_seq is not None
            group_state = state.groups[event.resource]
            expected = group_state.next_group_seq.get(event.group_owner, 0)
            if event.group_seq != expected:
                code = "STALE_COMMIT_GROUP" if event.group_seq < expected else "FUTURE_COMMIT_GROUP"
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["event_group_seq"] = event.group_seq
                raise ProtocolError(
                    code,
                    "cp.async commit sequence does not name the next group",
                    event.id,
                    snapshot,
                )
            if len(group_state.pending_groups(event.group_owner)) >= group_state.capacity:
                raise ProtocolError(
                    "COMMIT_GROUP_CAPACITY_EXCEEDED",
                    "cp.async cannot retain more uncompleted groups than its resource capacity",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            group_id = stable_group_id(group_state.protocol, event.group_owner, event.group_seq)
            completion_id = stable_group_completion_id(
                group_state.protocol,
                event.group_owner,
                event.group_seq,
                "GROUP_COMPLETE",
            )
            if (
                completion_id in state.completed_transitions
                or any(
                    completion_id in group.completion_ids.values()
                    for group in group_state.groups(event.group_owner)
                )
            ):
                raise ProtocolError(
                    "DUPLICATE_GROUP_COMPLETION",
                    "cp.async group completion identity is not unique",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            issues = tuple(group_state.staged.get(event.group_owner, ()))
            group = CommittedGroup(
                event.group_seq,
                group_id,
                event.id,
                issues,
                {"GROUP_COMPLETE": completion_id},
                completed_endpoints={"GROUP_COMPLETE"} if not issues else set(),
            )
            edges = tuple(
                [(event.id, group_id, "COMMIT_TO_GROUP")]
                + [(issue, group_id, "ISSUE_TO_GROUP") for issue in issues]
                + ([(group_id, completion_id, "GROUP_TO_COMPLETION")] if not issues else [])
            )
            group_state.groups(event.group_owner).append(group)
            group_state.staged[event.group_owner] = []
            group_state.next_group_seq[event.group_owner] = expected + 1
            if not issues:
                state.completed_transitions.add(completion_id)
            state.dynamic_edges.update(edges)
            added_edges = edges
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "CpAsyncWaitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.wait_group_count is not None
            group_state = state.groups[event.resource]
            status = self._group_wait_status(
                event,
                group_state,
                "GROUP_COMPLETE",
            )
            if status == "INVALID":
                raise ProtocolError(
                    "WAIT_BEFORE_COMMIT",
                    "cp.async wait reached staged operations that have not been committed",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            if status == "BLOCKED":
                raise AssertionError("blocked cp.async wait was applied")
            completion_kind = "GROUP_COMPLETE"
            old_frontier = group_state.observed_frontier(
                event.group_owner,
                completion_kind,
            )
            required = group_state.required_for_wait(
                event.group_owner,
                event.wait_group_count,
                completion_kind,
            )
            newly_observed = [group for group in required if group.sequence > old_frontier]
            edges = tuple(
                (group.completion_id(completion_kind), event.id, "COMPLETION_TO_WAIT")
                for group in newly_observed
            )
            if required:
                group_state.set_observed_frontier(
                    event.group_owner,
                    completion_kind,
                    max(old_frontier, required[-1].sequence),
                )
            state.dynamic_edges.update(edges)
            added_edges = edges
        elif event.type == "WgmmaFenceOperand":
            assert event.resource is not None and event.group_owner is not None
            group_state = state.groups[event.resource]
            group_state.operand_fences.setdefault(event.group_owner, set()).update(event.regions)
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "WgmmaIssue":
            assert event.resource is not None and event.group_owner is not None
            group_state = state.groups[event.resource]
            covered = group_state.operand_fences.get(event.group_owner, set())
            missing = sorted(set(event.regions) - covered)
            if missing:
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["missing_operand_fence_regions"] = missing
                raise ProtocolError(
                    "WGMMA_OPERAND_FENCE_MISSING",
                    "WGMMA issue requires a prior operand fence covering every shared operand",
                    event.id,
                    snapshot,
                )
            group_state.staged_operations(event.group_owner).append(event.id)
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "WgmmaCommitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.group_seq is not None
            group_state = state.groups[event.resource]
            expected = group_state.next_group_seq.get(event.group_owner, 0)
            if event.group_seq != expected:
                code = "STALE_COMMIT_GROUP" if event.group_seq < expected else "FUTURE_COMMIT_GROUP"
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["event_group_seq"] = event.group_seq
                raise ProtocolError(
                    code,
                    "WGMMA commit sequence does not name the next group",
                    event.id,
                    snapshot,
                )
            if len(group_state.pending_groups(event.group_owner)) >= group_state.capacity:
                raise ProtocolError(
                    "COMMIT_GROUP_CAPACITY_EXCEEDED",
                    "WGMMA cannot retain more result-pending groups than its resource capacity",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            group_id = stable_group_id(group_state.protocol, event.group_owner, event.group_seq)
            completion_ids = {
                kind: stable_group_completion_id(
                    group_state.protocol,
                    event.group_owner,
                    event.group_seq,
                    kind,
                )
                for kind in group_state.completion_kinds
            }
            known_ids = {
                identifier
                for group in group_state.committed.get(event.group_owner, ())
                for identifier in group.completion_ids.values()
            }
            duplicates = sorted(
                identifier for identifier in completion_ids.values()
                if identifier in state.completed_transitions or identifier in known_ids
            )
            if duplicates:
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["duplicate_completion_ids"] = duplicates
                raise ProtocolError(
                    "DUPLICATE_GROUP_COMPLETION",
                    "WGMMA group completion identity is not unique",
                    event.id,
                    snapshot,
                )
            issues = tuple(group_state.staged.get(event.group_owner, ()))
            completed_endpoints = set(group_state.completion_kinds) if not issues else set()
            group = CommittedGroup(
                event.group_seq,
                group_id,
                event.id,
                issues,
                completion_ids,
                completed_endpoints=completed_endpoints,
            )
            edges = tuple(
                [(event.id, group_id, "COMMIT_TO_GROUP")]
                + [(issue, group_id, "ISSUE_TO_GROUP") for issue in issues]
                + (
                    [
                        (group_id, completion_ids[kind], "GROUP_TO_COMPLETION")
                        for kind in group_state.completion_kinds
                    ]
                    if not issues else []
                )
            )
            group_state.groups(event.group_owner).append(group)
            group_state.staged[event.group_owner] = []
            group_state.next_group_seq[event.group_owner] = expected + 1
            if not issues:
                state.completed_transitions.update(completion_ids.values())
            state.dynamic_edges.update(edges)
            added_edges = edges
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "WgmmaWaitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.wait_group_count is not None and event.completion_kind is not None
            group_state = state.groups[event.resource]
            completion_kind = event.completion_kind
            status = self._group_wait_status(event, group_state, completion_kind)
            if status == "INVALID":
                raise ProtocolError(
                    "WAIT_BEFORE_COMMIT",
                    "WGMMA wait reached staged operations that have not been committed",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            if status == "BLOCKED":
                raise AssertionError("blocked WGMMA wait was applied")
            old_frontier = group_state.observed_frontier(event.group_owner, completion_kind)
            required = group_state.required_for_wait(
                event.group_owner,
                event.wait_group_count,
                completion_kind,
            )
            newly_observed = [group for group in required if group.sequence > old_frontier]
            edges = tuple(
                (group.completion_id(completion_kind), event.id, "COMPLETION_TO_WAIT")
                for group in newly_observed
            )
            if required:
                group_state.set_observed_frontier(
                    event.group_owner,
                    completion_kind,
                    max(old_frontier, required[-1].sequence),
                )
            state.dynamic_edges.update(edges)
            added_edges = edges
        elif event.type == "TmaStoreIssue":
            assert event.resource is not None and event.group_owner is not None
            group_state = state.groups[event.resource]
            group_state.staged_operations(event.group_owner).append(event.id)
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "TmaStoreCommitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.group_seq is not None
            group_state = state.groups[event.resource]
            expected = group_state.next_group_seq.get(event.group_owner, 0)
            if event.group_seq != expected:
                code = "STALE_COMMIT_GROUP" if event.group_seq < expected else "FUTURE_COMMIT_GROUP"
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["event_group_seq"] = event.group_seq
                raise ProtocolError(
                    code,
                    "TMA store commit sequence does not name the next group",
                    event.id,
                    snapshot,
                )
            if len(group_state.pending_groups(event.group_owner)) >= group_state.capacity:
                raise ProtocolError(
                    "COMMIT_GROUP_CAPACITY_EXCEEDED",
                    "TMA store cannot retain more unfinished groups than its resource capacity",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            group_id = stable_group_id(group_state.protocol, event.group_owner, event.group_seq)
            completion_ids = {
                kind: stable_group_completion_id(
                    group_state.protocol,
                    event.group_owner,
                    event.group_seq,
                    kind,
                )
                for kind in group_state.completion_kinds
            }
            known_ids = {
                identifier
                for group in group_state.committed.get(event.group_owner, ())
                for identifier in group.completion_ids.values()
            }
            duplicates = sorted(
                identifier for identifier in completion_ids.values()
                if identifier in state.completed_transitions or identifier in known_ids
            )
            if duplicates:
                snapshot = group_state.snapshot(event.group_owner)
                snapshot["duplicate_completion_ids"] = duplicates
                raise ProtocolError(
                    "DUPLICATE_GROUP_COMPLETION",
                    "TMA store group completion identity is not unique",
                    event.id,
                    snapshot,
                )
            issues = tuple(group_state.staged.get(event.group_owner, ()))
            group = CommittedGroup(
                event.group_seq,
                group_id,
                event.id,
                issues,
                completion_ids,
            )
            edges = tuple(
                [(event.id, group_id, "COMMIT_TO_GROUP")]
                + [(issue, group_id, "ISSUE_TO_GROUP") for issue in issues]
            )
            group_state.groups(event.group_owner).append(group)
            group_state.staged[event.group_owner] = []
            group_state.next_group_seq[event.group_owner] = expected + 1
            state.dynamic_edges.update(edges)
            added_edges = edges
            changed_groups.add((event.resource, event.group_owner))
        elif event.type == "TmaStoreWaitGroup":
            assert event.resource is not None and event.group_owner is not None
            assert event.wait_group_count is not None and event.wait_mode is not None
            group_state = state.groups[event.resource]
            completion_kind = (
                "SOURCE_RELEASED"
                if event.wait_mode == "READ_ONLY"
                else "GLOBAL_COMPLETE"
            )
            status = self._group_wait_status(event, group_state, completion_kind)
            if status == "INVALID":
                raise ProtocolError(
                    "WAIT_BEFORE_COMMIT",
                    "TMA store wait reached staged operations that have not been committed",
                    event.id,
                    group_state.snapshot(event.group_owner),
                )
            if status == "BLOCKED":
                raise AssertionError("blocked TMA store wait was applied")
            old_frontier = group_state.observed_frontier(event.group_owner, completion_kind)
            required = group_state.required_for_wait(
                event.group_owner,
                event.wait_group_count,
                completion_kind,
            )
            newly_observed = [group for group in required if group.sequence > old_frontier]
            edges = tuple(
                (group.completion_id(completion_kind), event.id, "COMPLETION_TO_WAIT")
                for group in newly_observed
            )
            if required:
                group_state.set_observed_frontier(
                    event.group_owner,
                    completion_kind,
                    max(old_frontier, required[-1].sequence),
                )
            state.dynamic_edges.update(edges)
            added_edges = edges
        elif event.type == "ChannelCreate":
            assert event.resource is not None
            channel = state.channels[event.resource]
            if channel.initialized:
                raise ProtocolError(
                    "CHANNEL_REINITIALIZED",
                    "producer-consumer channel cannot be initialized more than once",
                    event.id,
                    channel.snapshot(),
                )
            channel.initialize(event.id)
            changed_channels.update((event.resource, slot) for slot in channel.slots)
        elif event.type == "ProducerAcquire":
            assert event.resource is not None and event.slot is not None and event.logical_phase is not None
            channel = state.channels[event.resource]
            if not channel.initialized:
                raise ProtocolError(
                    "CHANNEL_USE_BEFORE_CREATE",
                    "producer acquire used a channel before ChannelCreate",
                    event.id,
                    channel.snapshot(),
                )
            slot_state = channel.slots[event.slot]
            if event.logical_phase != slot_state.logical_phase:
                code = "STALE_CHANNEL_PHASE" if event.logical_phase < slot_state.logical_phase else "FUTURE_CHANNEL_PHASE"
                raise ProtocolError(code, "producer acquire phase does not match the slot", event.id, channel.snapshot(event.slot))
            if slot_state.state != "PRODUCER_AVAILABLE":
                raise ProtocolError(
                    "CHANNEL_SLOT_NOT_PRODUCER_AVAILABLE",
                    "producer acquire requires a released producer-available slot",
                    event.id,
                    channel.snapshot(event.slot),
                )
            edges: list[tuple[str, str, str]] = []
            if event.logical_phase > 0:
                release = channel.releases.get((event.slot, event.logical_phase - 1))
                if release is None:
                    raise ProtocolError(
                        "CHANNEL_RELEASE_EVIDENCE_MISSING",
                        "reused channel slot lacks the prior consumer release",
                        event.id,
                        channel.snapshot(event.slot),
                    )
                edges.append((release, event.id, "CHANNEL_RELEASE"))
            slot_state.state = "PRODUCER_OWNED"
            slot_state.producer = event.actor
            slot_state.consumer = None
            slot_state.acquire_event = event.id
            slot_state.commit_event = None
            slot_state.handoff_id = None
            slot_state.wait_event = None
            state.dynamic_edges.update(edges)
            added_edges = tuple(edges)
            changed_channels.add((event.resource, event.slot))
        elif event.type == "ProducerCommit":
            assert event.resource is not None and event.slot is not None and event.logical_phase is not None
            channel = state.channels[event.resource]
            if not channel.initialized:
                raise ProtocolError("CHANNEL_USE_BEFORE_CREATE", "producer commit used a channel before ChannelCreate", event.id, channel.snapshot())
            slot_state = channel.slots[event.slot]
            if event.logical_phase != slot_state.logical_phase:
                code = "STALE_CHANNEL_PHASE" if event.logical_phase < slot_state.logical_phase else "FUTURE_CHANNEL_PHASE"
                raise ProtocolError(code, "producer commit phase does not match the slot", event.id, channel.snapshot(event.slot))
            if slot_state.state != "PRODUCER_OWNED" or slot_state.producer != event.actor:
                raise ProtocolError(
                    "CHANNEL_PRODUCER_OWNERSHIP",
                    "only the actor owning a producer-acquired slot may commit it",
                    event.id,
                    channel.snapshot(event.slot),
                )
            slot_state.state = "COMMITTED"
            slot_state.commit_event = event.id
            changed_channels.add((event.resource, event.slot))
        elif event.type == "ConsumerWait":
            assert event.resource is not None and event.slot is not None and event.logical_phase is not None
            channel = state.channels[event.resource]
            status = self._consumer_wait_status(event, channel)
            if status == "BLOCKED":
                raise AssertionError("blocked channel consumer wait was applied")
            if status == "INVALID":
                raise ProtocolError(
                    "INVALID_CHANNEL_CONSUMER_WAIT",
                    "consumer wait does not name a published current slot phase",
                    event.id,
                    channel.snapshot(event.slot),
                )
            slot_state = channel.slots[event.slot]
            slot_state.state = "CONSUMER_OWNED"
            slot_state.consumer = event.actor
            slot_state.wait_event = event.id
            edge = (slot_state.handoff_id, event.id, "CHANNEL_HANDOFF")
            state.dynamic_edges.add(edge)
            added_edges = (edge,)
            changed_channels.add((event.resource, event.slot))
        elif event.type == "ConsumerRelease":
            assert event.resource is not None and event.slot is not None and event.logical_phase is not None
            channel = state.channels[event.resource]
            if not channel.initialized:
                raise ProtocolError("CHANNEL_USE_BEFORE_CREATE", "consumer release used a channel before ChannelCreate", event.id, channel.snapshot())
            slot_state = channel.slots[event.slot]
            if event.logical_phase != slot_state.logical_phase:
                code = "STALE_CHANNEL_PHASE" if event.logical_phase < slot_state.logical_phase else "FUTURE_CHANNEL_PHASE"
                raise ProtocolError(code, "consumer release phase does not match the slot", event.id, channel.snapshot(event.slot))
            if slot_state.state != "CONSUMER_OWNED" or slot_state.consumer != event.actor:
                raise ProtocolError(
                    "CHANNEL_CONSUMER_OWNERSHIP",
                    "only the actor owning a consumer-waited slot may release it",
                    event.id,
                    channel.snapshot(event.slot),
                )
            edge = (slot_state.wait_event, event.id, "CHANNEL_RELEASE")
            channel.releases[(event.slot, event.logical_phase)] = event.id
            slot_state.logical_phase += 1
            slot_state.state = "PRODUCER_AVAILABLE"
            slot_state.producer = None
            slot_state.consumer = None
            slot_state.acquire_event = None
            slot_state.commit_event = None
            slot_state.handoff_id = None
            slot_state.wait_event = None
            state.dynamic_edges.add(edge)
            added_edges = (edge,)
            changed_channels.add((event.resource, event.slot))
        elif event.type == "ProxyFence":
            assert event.from_domain is not None and event.to_domain is not None
            assert event.scope is not None and event.visibility_mask is not None
            state.visibility_transfers.append(VisibilityTransfer(
                event.id,
                event.actor,
                event.from_domain,
                event.to_domain,
                event.scope,
                event.regions,
                event.visibility_mask,
            ))
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
            phase_incomplete = barrier.generation_started and not (
                barrier.pending == 0
                and barrier.completed_tx == barrier.expected_tx
            )
            if phase_incomplete or in_flight or outstanding:
                snapshot = barrier.snapshot()
                if outstanding:
                    snapshot["remaining_logical_waits"] = sorted(outstanding)
                raise ProtocolError(
                    "INVALIDATE_INCOMPLETE",
                    "cannot invalidate with a started phase, in-flight work, or outstanding wait users",
                    event.id,
                    snapshot,
                )
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
            changed_groups=frozenset(changed_groups),
            changed_channels=frozenset(changed_channels),
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
        if any(
            bool(group_state.staged.get(owner))
            or any(
                group_state.pending(owner, completion_kind)
                for completion_kind in group_state.completion_kinds
            )
            for group_state in state.groups.values()
            for owner in group_state.participants
        ):
            return True
        if any(
            channel.initialized
            and any(slot.state != "PRODUCER_AVAILABLE" for slot in channel.slots.values())
            for channel in state.channels.values()
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
        groups = tuple(
            (
                resource,
                group_state.protocol,
                group_state.participants,
                group_state.capacity,
                group_state.completion_kinds,
                tuple(
                    (
                        owner,
                        group_state.next_group_seq.get(owner, 0),
                        tuple(group_state.staged.get(owner, ())),
                        tuple(sorted(group_state.operand_fences.get(owner, set()))),
                        tuple(
                            (
                                group.sequence,
                                group.group_id,
                                group.commit_event,
                                group.issue_events,
                                tuple(sorted(group.completion_ids.items())),
                                tuple(sorted(group.completed_endpoints)),
                            )
                            for group in group_state.committed.get(owner, ())
                        ),
                        tuple(sorted(
                            group_state.observed_frontiers.get(owner, {}).items()
                        )),
                    )
                    for owner in sorted(group_state.participants)
                ),
            )
            for resource, group_state in sorted(state.groups.items())
        )
        channels = tuple(
            (
                resource,
                channel.participants,
                channel.capacity,
                channel.initialized,
                channel.create_event,
                tuple(
                    (
                        slot,
                        item.logical_phase,
                        item.state,
                        item.producer,
                        item.consumer,
                        item.acquire_event,
                        item.commit_event,
                        item.handoff_id,
                        item.wait_event,
                    )
                    for slot, item in sorted(channel.slots.items())
                ),
                tuple(sorted(channel.handoffs.items())),
                tuple(sorted(channel.releases.items())),
            )
            for resource, channel in sorted(state.channels.items())
        )
        transfers = tuple(
            (
                transfer.event,
                transfer.actor,
                transfer.from_domain,
                transfer.to_domain,
                transfer.scope,
                transfer.regions,
                tuple(sorted(transfer.visibility_mask.artifact().items())),
            )
            for transfer in sorted(state.visibility_transfers, key=lambda item: item.event)
        )
        remaining = tuple((key, tuple(sorted(event_ids))) for key, event_ids in sorted(state.remaining.items()))
        pending = tuple((key, op.issue_id, op.actor, op.resource, op.destination_region, op.lifecycle_epoch, op.generation, op.bytes)
                        for key, op in sorted(state.pending_completions.items()))
        return (tuple(sorted(state.frontiers.items())), tuple(sorted(state.executed)), barriers, collectives, groups, channels, transfers, remaining,
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
        for resource_id in sorted(state.groups):
            group_state = state.groups[resource_id]
            for owner in sorted(group_state.participants):
                if group_state.staged.get(owner):
                    return {
                        "code": {
                            "CP_ASYNC": "MISSING_CP_ASYNC_COMMIT",
                            "WGMMA": "MISSING_WGMMA_COMMIT",
                            "TMA_STORE": "MISSING_TMA_STORE_COMMIT",
                        }[group_state.protocol],
                        "message": f"{group_state.protocol.lower()} operations remain staged without a commit_group",
                        "resource": resource_id,
                        "owner": owner,
                        "protocol_state": group_state.snapshot(owner),
                    }
                if any(
                    group_state.pending(owner, completion_kind)
                    for completion_kind in group_state.completion_kinds
                ):
                    return {
                        "code": "STUCK_GROUP_COMPLETION",
                        "message": "a committed async group cannot reach all synthetic completions",
                        "resource": resource_id,
                        "owner": owner,
                        "protocol_state": group_state.snapshot(owner),
                    }
        for resource_id in sorted(state.channels):
            channel = state.channels[resource_id]
            if not channel.initialized:
                continue
            for slot, slot_state in sorted(channel.slots.items()):
                if slot_state.state == "PRODUCER_OWNED":
                    return {
                        "code": "MISSING_PRODUCER_COMMIT",
                        "message": "channel producer owns a slot but never commits it",
                        "resource": resource_id,
                        "slot": slot,
                        "protocol_state": channel.snapshot(slot),
                    }
                if slot_state.state == "CONSUMER_AVAILABLE":
                    return {
                        "code": "MISSING_CONSUMER_WAIT",
                        "message": "published channel slot is never acquired by a consumer",
                        "resource": resource_id,
                        "slot": slot,
                        "protocol_state": channel.snapshot(slot),
                    }
                if slot_state.state == "CONSUMER_OWNED":
                    return {
                        "code": "MISSING_CONSUMER_RELEASE",
                        "message": "channel consumer owns a slot but never releases it",
                        "resource": resource_id,
                        "slot": slot,
                        "protocol_state": channel.snapshot(slot),
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
        channel_waits = [
            event.id for actor in sorted(state.frontiers)
            if (event := self._frontier_event(state, actor)) is not None
            and event.type == "ConsumerWait"
        ]
        if channel_waits:
            return {
                "code": "STUCK_CHANNEL_WAIT",
                "message": "no producer handoff can satisfy one or more consumer waits",
                "events": channel_waits,
            }
        group_waits = [
            event for actor in sorted(state.frontiers)
            if (event := self._frontier_event(state, actor)) is not None
            and event.type in {"CpAsyncWaitGroup", "WgmmaWaitGroup", "TmaStoreWaitGroup"}
        ]
        if group_waits:
            protocols = sorted({event.type for event in group_waits})
            return {
                "code": "STUCK_GROUP_WAIT",
                "message": "no typed group completion can satisfy one or more waits",
                "events": [event.id for event in group_waits],
                "protocols": protocols,
            }
        return {"code": "STUCK_DEPENDENCY", "message": "no program or completion transition is enabled", "boundary": list(self.boundary(state))}
