from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .engine import (
    ExecutionState,
    Executor,
    ExplorationResult,
    ProtocolError,
    Transition,
    TransitionDelta,
)
from .model import Module, stable_completion_id, stable_phase_id


@dataclass(frozen=True, slots=True)
class EndAnchor:
    """Typed stream terminator used to flush the final held transition."""


END = EndAnchor()
AnchorToken = str | EndAnchor


@dataclass(frozen=True, slots=True)
class AnchorStream:
    kind: Literal["PROGRAM", "PHASE_COMPLETION", "TMA_COMPLETION"]
    name: str
    anchors: tuple[str, ...]

    @property
    def tokens(self) -> tuple[AnchorToken, ...]:
        return (*self.anchors, END)


class StableReadySet:
    """Minimum-rank set backed by a hierarchical 64-bit bitmap."""

    WORD_BITS = 64

    def __init__(self, size: int) -> None:
        self.size = size
        width = max(1, (size + self.WORD_BITS - 1) // self.WORD_BITS)
        self.levels: list[list[int]] = []
        while True:
            self.levels.append([0] * width)
            if width == 1:
                break
            width = (width + self.WORD_BITS - 1) // self.WORD_BITS

    def add(self, rank: int) -> bool:
        if rank < 0 or rank >= self.size:
            raise IndexError(f"ready rank out of range: {rank}")
        word_index, bit_index = divmod(rank, self.WORD_BITS)
        mask = 1 << bit_index
        if self.levels[0][word_index] & mask:
            return False
        self.levels[0][word_index] |= mask

        child_word = word_index
        for level in range(1, len(self.levels)):
            word_index, bit_index = divmod(child_word, self.WORD_BITS)
            mask = 1 << bit_index
            if self.levels[level][word_index] & mask:
                break
            self.levels[level][word_index] |= mask
            child_word = word_index
        return True

    def discard(self, rank: int) -> bool:
        if rank < 0 or rank >= self.size:
            return False
        word_index, bit_index = divmod(rank, self.WORD_BITS)
        mask = 1 << bit_index
        if not self.levels[0][word_index] & mask:
            return False
        self.levels[0][word_index] &= ~mask
        if self.levels[0][word_index]:
            return True

        child_word = word_index
        for level in range(1, len(self.levels)):
            word_index, bit_index = divmod(child_word, self.WORD_BITS)
            self.levels[level][word_index] &= ~(1 << bit_index)
            if self.levels[level][word_index]:
                break
            child_word = word_index
        return True

    def pop_min(self) -> int | None:
        if not self:
            return None
        child_word = 0
        for level in range(len(self.levels) - 1, -1, -1):
            word = self.levels[level][child_word]
            bit_index = (word & -word).bit_length() - 1
            child_word = child_word * self.WORD_BITS + bit_index
        rank = child_word
        if rank >= self.size:
            raise AssertionError(f"ready bitmap produced invalid rank: {rank}")
        self.discard(rank)
        return rank

    def __bool__(self) -> bool:
        return bool(self.levels[-1][0])


@dataclass(slots=True)
class IncrementalCounters:
    program_streams: int = 0
    phase_streams: int = 0
    tma_completion_streams: int = 0
    anchor_streams: int = 0
    anchors: int = 0
    n_ext: int = 0
    t_eff: int = 0
    state_initializations: int = 0
    completed_passes: int = 0
    anchor_boundaries: int = 0
    transitions_by_kind: dict[str, int] = field(default_factory=lambda: {
        "PROGRAM": 0,
        "TMA_COMPLETION": 0,
        "PHASE_COMPLETION": 0,
    })
    transition_attempts: int = 0
    worklist_pushes: int = 0
    worklist_pops: int = 0
    stale_worklist_pops: int = 0
    dependency_registrations: int = 0
    dependency_notifications: int = 0
    wait_registrations: int = 0
    wait_notifications: int = 0
    resource_notifications: int = 0
    frontier_exposures: int = 0
    phase_enable_checks: int = 0
    remaining_removals: int = 0
    boundary_evaluations: int = 0
    full_enabled_scans: int = 0
    state_clones: int = 0
    replayed_prefix_steps: int = 0

    @property
    def successful_transitions(self) -> int:
        return sum(self.transitions_by_kind.values())

    @property
    def scheduler_work_units(self) -> int:
        return sum((
            self.transition_attempts,
            self.worklist_pushes,
            self.worklist_pops,
            self.dependency_registrations,
            self.dependency_notifications,
            self.wait_registrations,
            self.wait_notifications,
            self.resource_notifications,
            self.frontier_exposures,
            self.phase_enable_checks,
            self.remaining_removals,
            self.boundary_evaluations,
        ))

    def certificate(self) -> dict[str, Any]:
        return {
            "stream_counts": {
                "program": self.program_streams,
                "phase_completion": self.phase_streams,
                "tma_completion": self.tma_completion_streams,
                "total": self.anchor_streams,
            },
            "anchors": self.anchors,
            "n_ext": self.n_ext,
            "t_eff": self.t_eff,
            "state_initializations": self.state_initializations,
            "completed_passes": self.completed_passes,
            "anchor_boundaries": self.anchor_boundaries,
            "transitions_by_kind": dict(self.transitions_by_kind),
            "transition_attempts": self.transition_attempts,
            "scheduler_work_units": self.scheduler_work_units,
            "worklist_pushes": self.worklist_pushes,
            "worklist_pops": self.worklist_pops,
            "stale_worklist_pops": self.stale_worklist_pops,
            "dependency_registrations": self.dependency_registrations,
            "dependency_notifications": self.dependency_notifications,
            "wait_registrations": self.wait_registrations,
            "wait_notifications": self.wait_notifications,
            "resource_notifications": self.resource_notifications,
            "frontier_exposures": self.frontier_exposures,
            "phase_enable_checks": self.phase_enable_checks,
            "remaining_removals": self.remaining_removals,
            "boundary_evaluations": self.boundary_evaluations,
            "full_enabled_scans": self.full_enabled_scans,
            "state_clones": self.state_clones,
            "replayed_prefix_steps": self.replayed_prefix_steps,
        }


def build_anchor_streams(module: Module, anchor_ids: list[str]) -> tuple[AnchorStream, ...]:
    """Partition every checker anchor into its naturally ordered stream."""

    streams: list[AnchorStream] = []
    for actor in sorted(module.events_by_actor):
        anchors = tuple(event.id for event in module.events_by_actor[actor])
        if anchors:
            streams.append(AnchorStream("PROGRAM", f"program:{actor}", anchors))

    phases: dict[str, dict[str, tuple[int, int, str]]] = {
        resource.id: {} for resource in module.resources
    }
    for event in module.events:
        if event.resource is None or event.generation is None or event.lifecycle_epoch is None:
            continue
        identifier = stable_phase_id(event.resource, event.lifecycle_epoch, event.generation)
        phases[event.resource][identifier] = (event.lifecycle_epoch, event.generation, identifier)
    for resource in sorted(phases):
        ordered = tuple(item[2] for item in sorted(phases[resource].values()))
        if ordered:
            streams.append(AnchorStream("PHASE_COMPLETION", f"phase:{resource}", ordered))

    for event in sorted(module.events, key=lambda item: stable_completion_id(item.id, item.generation or 0)):
        if event.type != "TmaLoadIssue":
            continue
        identifier = stable_completion_id(event.id, event.generation)
        streams.append(AnchorStream("TMA_COMPLETION", f"tma:{identifier}", (identifier,)))

    covered = [anchor for stream in streams for anchor in stream.anchors]
    if len(covered) != len(set(covered)) or set(covered) != set(anchor_ids):
        missing = sorted(set(anchor_ids) - set(covered))
        extra = sorted(set(covered) - set(anchor_ids))
        raise AssertionError(f"anchor stream partition mismatch: missing={missing}, extra={extra}")
    return tuple(sorted(streams, key=lambda stream: (stream.anchors[0] if stream.anchors else "", stream.kind, stream.name)))


class ReadyIndex:
    """Delta-maintained candidate index with static transition ranks."""

    def __init__(
        self,
        executor: Executor,
        state: ExecutionState,
        anchor_ids: list[str],
        counters: IncrementalCounters,
        hold: AnchorToken | None = None,
    ) -> None:
        self.executor = executor
        self.state = state
        self.counters = counters
        self.hold: AnchorToken | None = hold
        self.active: dict[str, Transition] = {}
        self.held_candidates: dict[str, Transition] = {}
        self.phase_ready: dict[str, str] = {}
        self.actor_dependencies: dict[str, set[str]] = {}
        self.dependency_waiters: dict[str, set[str]] = {}
        self.actor_resource: dict[str, str] = {}
        self.resource_followers: dict[str, set[str]] = {}
        self.blocked_waits: dict[str, set[str]] = {}
        self.future_phase_followers: dict[str, set[str]] = {}

        tma_ids = {
            stable_completion_id(event.id, event.generation)
            for event in executor.module.events if event.type == "TmaLoadIssue"
        }
        phase_resources: dict[str, str] = {}
        for event in executor.module.events:
            if event.resource is not None and event.generation is not None:
                phase_resources[stable_phase_id(event.resource, event.lifecycle_epoch, event.generation)] = event.resource
        transitions: list[Transition] = [
            Transition(event.id, "PROGRAM", event=event) for event in executor.module.events
        ]
        for identifier in anchor_ids:
            if identifier in executor.events:
                continue
            if identifier in tma_ids:
                transitions.append(Transition(identifier, "TMA_COMPLETION"))
            else:
                transitions.append(Transition(identifier, "PHASE_COMPLETION", resource=phase_resources[identifier]))
        ordered_transitions = sorted(transitions, key=Transition.order_key)
        self.rank = {
            transition.id: rank
            for rank, transition in enumerate(ordered_transitions)
        }
        self.identifiers = tuple(transition.id for transition in ordered_transitions)
        self.ready = StableReadySet(len(ordered_transitions))

        for actor in sorted(state.frontiers):
            self._assess_actor(actor)
        for resource in sorted(state.barriers):
            self._refresh_phase(resource)

    def _hold_id(self) -> str | None:
        return self.hold if isinstance(self.hold, str) else None

    def _queue(self, transition: Transition) -> None:
        if transition.id in self.active:
            return
        if transition.id == self._hold_id():
            self.held_candidates[transition.id] = transition
            return
        self.held_candidates.pop(transition.id, None)
        self.active[transition.id] = transition
        if not self.ready.add(self.rank[transition.id]):
            raise AssertionError(f"active transition already present in ready bitmap: {transition.id}")
        self.counters.worklist_pushes += 1

    def _deactivate(self, identifier: str) -> None:
        if self.active.pop(identifier, None) is not None:
            self.ready.discard(self.rank[identifier])
        self.held_candidates.pop(identifier, None)

    def _clear_actor(self, actor: str) -> None:
        event = self.executor._frontier_event(self.state, actor)
        if event is not None:
            self._deactivate(event.id)
        for dependency in self.actor_dependencies.pop(actor, set()):
            waiters = self.dependency_waiters.get(dependency)
            if waiters is not None:
                waiters.discard(actor)
                if not waiters:
                    self.dependency_waiters.pop(dependency, None)
        resource = self.actor_resource.pop(actor, None)
        if resource is not None:
            self.resource_followers[resource].discard(actor)
            self.blocked_waits.get(resource, set()).discard(actor)
            self.future_phase_followers.get(resource, set()).discard(actor)

    def _assess_actor(self, actor: str) -> None:
        self._clear_actor(actor)
        event = self.executor._frontier_event(self.state, actor)
        if event is None:
            return
        self.counters.frontier_exposures += 1
        missing = {dependency for dependency in event.depends_on if dependency not in self.state.executed}
        if missing:
            self.actor_dependencies[actor] = missing
            for dependency in missing:
                self.dependency_waiters.setdefault(dependency, set()).add(actor)
                self.counters.dependency_registrations += 1
            return

        if event.resource is not None:
            self.actor_resource[actor] = event.resource
            self.resource_followers.setdefault(event.resource, set()).add(actor)
        if event.type == "MBarrierWaitParity":
            status = self.executor._wait_status(event, self.state.barriers[event.resource])
            if status == "BLOCKED":
                self.blocked_waits.setdefault(event.resource, set()).add(actor)
                self.counters.wait_registrations += 1
                return
        if event.resource is not None and event.generation is not None:
            barrier = self.state.barriers[event.resource]
            phase = self.executor._phase_transition(event.resource, barrier)
            if event.generation == barrier.generation + 1 and phase is not None:
                self.future_phase_followers.setdefault(event.resource, set()).add(actor)
                return
        self._queue(Transition(event.id, "PROGRAM", event=event))

    def _refresh_phase(self, resource: str) -> None:
        self.counters.phase_enable_checks += 1
        old = self.phase_ready.pop(resource, None)
        if old is not None:
            self._deactivate(old)
        transition = self.executor._phase_transition(resource, self.state.barriers[resource])
        if transition is not None:
            self.phase_ready[resource] = transition.id
            self._queue(transition)

    def _refresh_resource(self, resource: str) -> None:
        self.counters.resource_notifications += 1
        self._refresh_phase(resource)
        followers = tuple(sorted(self.resource_followers.get(resource, set())))
        for actor in followers:
            event = self.executor._frontier_event(self.state, actor)
            if event is not None and event.type == "MBarrierWaitParity":
                self.counters.wait_notifications += 1
            self._assess_actor(actor)

    def pop(self) -> Transition | None:
        rank = self.ready.pop_min()
        if rank is None:
            return None
        self.counters.worklist_pops += 1
        identifier = self.identifiers[rank]
        transition = self.active.pop(identifier, None)
        if transition is None:
            raise AssertionError(f"ready bitmap lost active transition: {identifier}")
        return transition

    def held_ready(self, anchor: str) -> bool:
        return anchor in self.held_candidates

    def install_hold(self, hold: AnchorToken) -> None:
        self.hold = hold
        if isinstance(hold, str):
            transition = self.active.pop(hold, None)
            if transition is not None:
                self.ready.discard(self.rank[hold])
                self.held_candidates[hold] = transition

    def release(self, anchor: str) -> None:
        transition = self.held_candidates.pop(anchor, None)
        if transition is None:
            raise AssertionError(f"released anchor is not candidate-ready: {anchor}")
        self._queue(transition)

    def notify(self, delta: TransitionDelta) -> None:
        self.counters.remaining_removals += len(delta.consumed_waits)
        if delta.advanced_actor is not None:
            self._assess_actor(delta.advanced_actor)
        if delta.executed_event is not None:
            waiters = tuple(sorted(self.dependency_waiters.get(delta.executed_event, set())))
            self.counters.dependency_notifications += len(waiters)
            for actor in waiters:
                self._assess_actor(actor)
        for completion in delta.spawned_completions:
            self._queue(Transition(completion, "TMA_COMPLETION"))
        for resource in sorted(delta.changed_resources):
            self._refresh_resource(resource)


@dataclass(slots=True)
class DriveResult:
    error: ProtocolError | None = None


@dataclass(slots=True)
class AnchorAnalysis:
    boundaries: list[dict[str, Any]]
    dynamic_edges: set[tuple[str, str, str]]
    error: ProtocolError | None = None
    error_state: ExecutionState | None = None
    stuck_state: ExecutionState | None = None
    blocked_anchor: str | None = None
    unreachable_anchor: str | None = None
    unreachable_state: ExecutionState | None = None


class IncrementalEngine:
    """Canonical and delayed-anchor maximal execution without full scans."""

    def __init__(self, executor: Executor, anchor_ids: list[str]):
        self.executor = executor
        self.anchor_ids = anchor_ids
        self.streams = build_anchor_streams(executor.module, anchor_ids)
        self.counters = IncrementalCounters(
            program_streams=sum(stream.kind == "PROGRAM" for stream in self.streams),
            phase_streams=sum(stream.kind == "PHASE_COMPLETION" for stream in self.streams),
            tma_completion_streams=sum(stream.kind == "TMA_COMPLETION" for stream in self.streams),
            anchor_streams=len(self.streams),
            anchors=len(anchor_ids),
            n_ext=len(anchor_ids) + sum(len(event.depends_on) for event in executor.module.events),
            t_eff=len(self.streams),
        )

    def _new(self, hold: AnchorToken | None) -> tuple[ExecutionState, ReadyIndex]:
        state = self.executor.initial_state()
        self.counters.state_initializations += 1
        return state, ReadyIndex(self.executor, state, self.anchor_ids, self.counters, hold)

    def _drive(self, state: ExecutionState, ready: ReadyIndex) -> DriveResult:
        while (transition := ready.pop()) is not None:
            self.counters.transition_attempts += 1
            try:
                delta = self.executor.apply(state, transition)
            except ProtocolError as error:
                return DriveResult(error)
            self.counters.transitions_by_kind[transition.kind] += 1
            ready.notify(delta)
        return DriveResult()

    def canonical(self) -> ExplorationResult:
        state, ready = self._new(None)
        driven = self._drive(state, ready)
        stuck = driven.error is None and self.executor.unfinished(state)
        if driven.error is None and not stuck:
            self.counters.completed_passes += 1
        return ExplorationResult(state, self.executor.boundary(state), driven.error, stuck, None)

    def analyze_streams(self) -> AnchorAnalysis:
        boundaries: list[dict[str, Any]] = []
        dynamic_edges: set[tuple[str, str, str]] = set()
        for stream in self.streams:
            first: AnchorToken = stream.anchors[0] if stream.anchors else END
            state, ready = self._new(first)
            for index, anchor in enumerate(stream.anchors):
                driven = self._drive(state, ready)
                dynamic_edges.update(state.dynamic_edges)
                if driven.error is not None:
                    return AnchorAnalysis(boundaries, dynamic_edges, driven.error, state)
                self.counters.boundary_evaluations += 1
                if not ready.held_ready(anchor):
                    if self.executor.unfinished(state):
                        return AnchorAnalysis(
                            boundaries,
                            dynamic_edges,
                            stuck_state=state,
                            blocked_anchor=anchor,
                        )
                    return AnchorAnalysis(
                        boundaries,
                        dynamic_edges,
                        unreachable_anchor=anchor,
                        unreachable_state=state,
                    )
                boundary = self.executor.boundary(state)
                boundaries.append({"anchor": anchor, "boundary": list(boundary)})
                self.counters.anchor_boundaries += 1
                next_hold: AnchorToken = stream.anchors[index + 1] if index + 1 < len(stream.anchors) else END
                ready.install_hold(next_hold)
                ready.release(anchor)

            # Typed END releases the final anchor and flushes its continuation.
            driven = self._drive(state, ready)
            dynamic_edges.update(state.dynamic_edges)
            if driven.error is not None:
                return AnchorAnalysis(boundaries, dynamic_edges, driven.error, state)
            if self.executor.unfinished(state):
                return AnchorAnalysis(boundaries, dynamic_edges, stuck_state=state, blocked_anchor=None)
            self.counters.completed_passes += 1

        return AnchorAnalysis(sorted(boundaries, key=lambda item: item["anchor"]), dynamic_edges)
