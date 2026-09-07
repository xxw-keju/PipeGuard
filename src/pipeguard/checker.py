from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from heapq import heappop, heappush, merge
from itertools import islice
from typing import Any, Literal

from .engine import Executor, ExplorationLimit, ProtocolError
from .incremental import IncrementalCounters, IncrementalEngine
from .model import (
    Event,
    Module,
    engine_completion_streams,
    normalize_module,
    stable_channel_handoff_id,
    stable_collective_release_id,
    stable_digest,
    stable_group_completion_id,
    stable_group_id,
    stable_phase_id,
)


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
        self._adjacency: dict[str, set[str]] = {}
        self._reachable_by_source: dict[str, frozenset[str]] = {}
        self.edge_insertions = 0
        self.reachability_queries = 0
        self.reachability_source_scans = 0
        self.reachability_edge_visits = 0

    def add_edge(self, source: str, target: str, kind: str) -> None:
        edge = (source, target, kind)
        if edge in self.edges:
            return
        self.nodes.update((source, target))
        self.edges.add(edge)
        self._adjacency.setdefault(source, set()).add(target)
        self._reachable_by_source.clear()
        self.edge_insertions += 1

    def add_edges(self, edges: set[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]) -> None:
        for source, target, kind in edges:
            self.add_edge(source, target, kind)

    def reachable(self, source: str, target: str) -> bool:
        self.reachability_queries += 1
        if source == target:
            return True
        reachable = self._reachable_by_source.get(source)
        if reachable is None:
            self.reachability_source_scans += 1
            queue = deque([source])
            seen = {source}
            while queue:
                node = queue.popleft()
                successors = self._adjacency.get(node, set())
                self.reachability_edge_visits += len(successors)
                for next_node in successors:
                    if next_node not in seen:
                        seen.add(next_node)
                        queue.append(next_node)
            seen.discard(source)
            reachable = frozenset(seen)
            self._reachable_by_source[source] = reachable
        return target in reachable

    def statistics(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for _source, _target, kind in self.edges:
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "pdg_nodes": len(self.nodes),
            "pdg_edges": len(self.edges),
            "pdg_edge_insertions": self.edge_insertions,
            "pdg_edges_by_kind": dict(sorted(kinds.items())),
            "pdg_reachability_queries": self.reachability_queries,
            "pdg_reachability_source_scans": self.reachability_source_scans,
            "pdg_reachability_edge_visits": self.reachability_edge_visits,
        }

    def artifact(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [{"from": source, "to": target, "kind": kind} for source, target, kind in sorted(self.edges)],
        }


def _cp_async_relations(
    module: Module,
) -> tuple[
    dict[str, tuple[str, str, tuple[str, ...]]],
    dict[str, tuple[str, str]],
    dict[str, tuple[str, ...]],
]:
    commits: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    issues: dict[str, tuple[str, str]] = {}
    waits: dict[str, tuple[str, ...]] = {}
    for actor in sorted(module.events_by_actor):
        staged: dict[str, list[str]] = {}
        committed: dict[str, list[str]] = {}
        for event in module.events_by_actor[actor]:
            if event.type == "CpAsyncIssue" and event.predicate:
                staged.setdefault(event.resource, []).append(event.id)
            elif event.type == "CpAsyncCommitGroup":
                group_id = stable_group_id("CP_ASYNC", event.group_owner, event.group_seq)
                completion_id = stable_group_completion_id(
                    "CP_ASYNC",
                    event.group_owner,
                    event.group_seq,
                    "GROUP_COMPLETE",
                )
                staged_issues = tuple(staged.get(event.resource, ()))
                commits[event.id] = (group_id, completion_id, staged_issues)
                for issue in staged_issues:
                    issues[issue] = (group_id, completion_id)
                staged[event.resource] = []
                committed.setdefault(event.resource, []).append(completion_id)
            elif event.type == "CpAsyncWaitGroup":
                groups = committed.get(event.resource, [])
                cutoff = max(0, len(groups) - event.wait_group_count)
                waits[event.id] = tuple(groups[cutoff - 1:cutoff])
    return commits, issues, waits


def _wgmma_relations(
    module: Module,
) -> tuple[
    dict[str, tuple[str, dict[str, str], tuple[str, ...]]],
    dict[str, dict[str, str]],
    dict[str, tuple[str, ...]],
]:
    commits: dict[str, tuple[str, dict[str, str], tuple[str, ...]]] = {}
    issues: dict[str, dict[str, str]] = {}
    waits: dict[str, tuple[str, ...]] = {}
    for actor in sorted(module.events_by_actor):
        staged: dict[str, list[str]] = {}
        committed: dict[tuple[str, str], list[str]] = {}
        for event in module.events_by_actor[actor]:
            if event.type == "WgmmaIssue":
                staged.setdefault(event.resource, []).append(event.id)
            elif event.type == "WgmmaCommitGroup":
                group_id = stable_group_id("WGMMA", event.group_owner, event.group_seq)
                completion_ids = {
                    kind: stable_group_completion_id(
                        "WGMMA",
                        event.group_owner,
                        event.group_seq,
                        kind,
                    )
                    for kind in ("OPERANDS_RELEASED", "RESULT_READY")
                }
                staged_issues = tuple(staged.get(event.resource, ()))
                commits[event.id] = (group_id, completion_ids, staged_issues)
                for issue in staged_issues:
                    issues[issue] = completion_ids
                staged[event.resource] = []
                for kind, completion_id in completion_ids.items():
                    committed.setdefault((event.resource, kind), []).append(completion_id)
            elif event.type == "WgmmaWaitGroup":
                groups = committed.get((event.resource, event.completion_kind), [])
                cutoff = max(0, len(groups) - event.wait_group_count)
                waits[event.id] = tuple(groups[cutoff - 1:cutoff])
    return commits, issues, waits


def _tma_store_relations(
    module: Module,
) -> tuple[
    dict[str, tuple[str, dict[str, str], tuple[str, ...]]],
    dict[str, dict[str, str]],
    dict[str, tuple[str, ...]],
]:
    commits: dict[str, tuple[str, dict[str, str], tuple[str, ...]]] = {}
    issues: dict[str, dict[str, str]] = {}
    waits: dict[str, tuple[str, ...]] = {}
    for actor in sorted(module.events_by_actor):
        staged: dict[str, list[str]] = {}
        committed: dict[tuple[str, str], list[str]] = {}
        for event in module.events_by_actor[actor]:
            if event.type == "TmaStoreIssue":
                staged.setdefault(event.resource, []).append(event.id)
            elif event.type == "TmaStoreCommitGroup":
                group_id = stable_group_id("TMA_STORE", event.group_owner, event.group_seq)
                completion_ids = {
                    kind: stable_group_completion_id(
                        "TMA_STORE",
                        event.group_owner,
                        event.group_seq,
                        kind,
                    )
                    for kind in ("SOURCE_RELEASED", "GLOBAL_COMPLETE")
                }
                staged_issues = tuple(staged.get(event.resource, ()))
                commits[event.id] = (group_id, completion_ids, staged_issues)
                for issue in staged_issues:
                    issues[issue] = completion_ids
                staged[event.resource] = []
                for kind, completion_id in completion_ids.items():
                    committed.setdefault((event.resource, kind), []).append(completion_id)
            elif event.type == "TmaStoreWaitGroup":
                kind = (
                    "SOURCE_RELEASED"
                    if event.wait_mode == "READ_ONLY"
                    else "GLOBAL_COMPLETE"
                )
                groups = committed.get((event.resource, kind), [])
                cutoff = max(0, len(groups) - event.wait_group_count)
                waits[event.id] = tuple(groups[cutoff - 1:cutoff])
    return commits, issues, waits


def _base_pdg(module: Module) -> PDG:
    graph = PDG()
    cp_commits, _cp_issues, cp_waits = _cp_async_relations(module)
    wgmma_commits, _wgmma_issues, wgmma_waits = _wgmma_relations(module)
    tma_store_commits, _tma_store_issues, tma_store_waits = _tma_store_relations(module)
    last_group_completion: dict[tuple[str, str, str, str], str] = {}
    graph.nodes.update(event.id for event in module.events)
    for actor in sorted(module.events_by_actor):
        trace = module.events_by_actor[actor]
        for first, second in zip(trace, trace[1:]):
            graph.add_edge(first.id, second.id, "PROGRAM_ORDER")
    for endpoints in engine_completion_streams(module).values():
        for predecessor, successor in zip(endpoints, endpoints[1:]):
            graph.add_edge(predecessor, successor, "ENGINE_ORDER")
    for event in module.events:
        for dependency in event.depends_on:
            graph.add_edge(dependency, event.id, "EXPLICIT_DEPENDENCY")
        if event.type in {"BarrierArrive", "BarrierObserve"}:
            release = stable_collective_release_id(event.resource, event.invocation_id)
            if event.type == "BarrierArrive" and event.predicate:
                graph.add_edge(event.id, release, "BARRIER_ARRIVE")
            elif event.type == "BarrierObserve":
                graph.add_edge(release, event.id, "BARRIER_RELEASE")
            continue
        if event.type in {"ProducerCommit", "ConsumerWait"}:
            handoff = stable_channel_handoff_id(
                event.resource,
                event.slot,
                event.logical_phase,
            )
            if event.type == "ProducerCommit":
                graph.add_edge(event.id, handoff, "CHANNEL_COMMIT")
            else:
                graph.add_edge(handoff, event.id, "CHANNEL_HANDOFF")
            continue
        if event.type == "ProducerAcquire" and event.logical_phase > 0:
            releases = [
                candidate for candidate in module.events
                if candidate.type == "ConsumerRelease"
                and candidate.resource == event.resource
                and candidate.slot == event.slot
                and candidate.logical_phase == event.logical_phase - 1
            ]
            for release in releases:
                graph.add_edge(release.id, event.id, "CHANNEL_RELEASE")
            continue
        if event.type == "CpAsyncCommitGroup":
            group, completion, issues = cp_commits[event.id]
            graph.add_edge(event.id, group, "COMMIT_TO_GROUP")
            for issue in issues:
                graph.add_edge(issue, group, "ISSUE_TO_GROUP")
            graph.add_edge(group, completion, "GROUP_TO_COMPLETION")
            completion_key = ("CP_ASYNC", event.resource, event.group_owner, "GROUP_COMPLETE")
            if previous := last_group_completion.get(completion_key):
                graph.add_edge(previous, completion, "GROUP_COMPLETION_ORDER")
            last_group_completion[completion_key] = completion
            continue
        if event.type == "CpAsyncWaitGroup":
            for completion in cp_waits[event.id]:
                graph.add_edge(completion, event.id, "COMPLETION_TO_WAIT")
            continue
        if event.type == "WgmmaCommitGroup":
            group, completions, issues = wgmma_commits[event.id]
            graph.add_edge(event.id, group, "COMMIT_TO_GROUP")
            for issue in issues:
                graph.add_edge(issue, group, "ISSUE_TO_GROUP")
            for completion_kind, completion in completions.items():
                graph.add_edge(group, completion, "GROUP_TO_COMPLETION")
                completion_key = ("WGMMA", event.resource, event.group_owner, completion_kind)
                if previous := last_group_completion.get(completion_key):
                    graph.add_edge(previous, completion, "GROUP_COMPLETION_ORDER")
                last_group_completion[completion_key] = completion
            continue
        if event.type == "WgmmaWaitGroup":
            for completion in wgmma_waits[event.id]:
                graph.add_edge(completion, event.id, "COMPLETION_TO_WAIT")
            continue
        if event.type == "TmaStoreCommitGroup":
            group, completions, issues = tma_store_commits[event.id]
            graph.add_edge(event.id, group, "COMMIT_TO_GROUP")
            for issue in issues:
                graph.add_edge(issue, group, "ISSUE_TO_GROUP")
            for completion_kind, completion in completions.items():
                graph.add_edge(group, completion, "GROUP_TO_COMPLETION")
                completion_key = ("TMA_STORE", event.resource, event.group_owner, completion_kind)
                if previous := last_group_completion.get(completion_key):
                    graph.add_edge(previous, completion, "GROUP_COMPLETION_ORDER")
                last_group_completion[completion_key] = completion
            graph.add_edge(
                completions["SOURCE_RELEASED"],
                completions["GLOBAL_COMPLETE"],
                "ENDPOINT_ORDER",
            )
            continue
        if event.type == "TmaStoreWaitGroup":
            for completion in tma_store_waits[event.id]:
                graph.add_edge(completion, event.id, "COMPLETION_TO_WAIT")
            continue
        if event.resource is None or event.generation is None:
            continue
        phase = stable_phase_id(event.resource, event.lifecycle_epoch, event.generation)
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
            graph.add_edge(event.id, phase, "ISSUE_TO_PHASE")
    return graph


def _anchor_ids(module: Module) -> list[str]:
    cp_commits, _cp_issues, _cp_waits = _cp_async_relations(module)
    wgmma_commits, _wgmma_issues, _wgmma_waits = _wgmma_relations(module)
    tma_store_commits, _tma_store_issues, _tma_store_waits = _tma_store_relations(module)
    identifiers = {event.id for event in module.events}
    for event in module.events:
        if event.type in {"BarrierArrive", "BarrierObserve"}:
            identifiers.add(stable_collective_release_id(event.resource, event.invocation_id))
        if event.type == "CpAsyncCommitGroup" and cp_commits[event.id][2]:
            identifiers.add(cp_commits[event.id][1])
        if event.type == "ProducerCommit":
            identifiers.add(stable_channel_handoff_id(
                event.resource,
                event.slot,
                event.logical_phase,
            ))
        if event.type == "WgmmaCommitGroup" and wgmma_commits[event.id][2]:
            identifiers.update(wgmma_commits[event.id][1].values())
        if event.type == "TmaStoreCommitGroup" and tma_store_commits[event.id][2]:
            identifiers.update(tma_store_commits[event.id][1].values())
        if (
            event.resource is not None
            and event.generation is not None
            and not (
                event.type in {"MBarrierArrive", "MBarrierArriveExpectTx"}
                and not event.predicate
            )
        ):
            identifiers.add(stable_phase_id(event.resource, event.lifecycle_epoch, event.generation))
    return sorted(identifiers)


def _effects(module: Module) -> list[Effect]:
    result: list[Effect] = []
    _commits, cp_issues, _waits_by_event = _cp_async_relations(module)
    _wgmma_commits, wgmma_issues, _wgmma_waits = _wgmma_relations(module)
    _tma_store_commits, tma_store_issues, _tma_store_waits = _tma_store_relations(module)
    for event in module.events:
        if event.type in {"MemRead", "MemWrite"}:
            result.append(Effect(event.id, event.actor, "READ" if event.type == "MemRead" else "WRITE", event.region, event.id, event.id, False))
        elif event.type == "TmaLoadIssue":
            phase = stable_phase_id(
                event.resource,
                event.lifecycle_epoch,
                event.generation,
            )
            for index, transfer in enumerate(event.tma_transfers):
                result.extend((
                    Effect(
                        f"effect:{event.id}:transfer:{index}:source",
                        event.actor,
                        "READ",
                        transfer.source_region,
                        event.id,
                        phase,
                        True,
                    ),
                    Effect(
                        f"effect:{event.id}:transfer:{index}:destination",
                        event.actor,
                        "WRITE",
                        transfer.destination_region,
                        event.id,
                        phase,
                        True,
                    ),
                ))
        elif event.type == "CpAsyncIssue" and event.predicate and event.id in cp_issues:
            _group, completion = cp_issues[event.id]
            result.extend((
                Effect(f"effect:{event.id}:source", event.actor, "READ", event.source_region, event.id, completion, True),
                Effect(f"effect:{event.id}:destination", event.actor, "WRITE", event.destination_region, event.id, completion, True),
            ))
        elif event.type == "WgmmaIssue" and event.id in wgmma_issues:
            completions = wgmma_issues[event.id]
            result.extend(
                Effect(
                    f"effect:{event.id}:operand:{index}",
                    event.actor,
                    "READ",
                    region,
                    event.id,
                    completions["OPERANDS_RELEASED"],
                    True,
                )
                for index, region in enumerate(event.regions)
            )
            result.append(Effect(
                f"effect:{event.id}:accumulator",
                event.actor,
                "WRITE",
                event.destination_region,
                event.id,
                completions["RESULT_READY"],
                True,
            ))
        elif event.type == "TmaStoreIssue" and event.id in tma_store_issues:
            completions = tma_store_issues[event.id]
            result.extend((
                Effect(
                    f"effect:{event.id}:source",
                    event.actor,
                    "READ",
                    event.source_region,
                    event.id,
                    completions["SOURCE_RELEASED"],
                    True,
                ),
                Effect(
                    f"effect:{event.id}:destination",
                    event.actor,
                    "WRITE",
                    event.destination_region,
                    event.id,
                    completions["GLOBAL_COMPLETE"],
                    True,
                ),
            ))
    return sorted(result, key=lambda effect: effect.id)


def _overlapping_effect_pairs(
    module: Module,
    effects: list[Effect],
) -> Iterator[tuple[Effect, Effect]]:
    effects_by_region: dict[str, list[int]] = defaultdict(list)
    for index, effect in enumerate(effects):
        effects_by_region[effect.region].append(index)

    regions_by_allocation: dict[tuple[str, int | None, str], list[Any]] = defaultdict(list)
    for region_id in effects_by_region:
        region = module.region_map[region_id]
        regions_by_allocation[
            (region.address_space, region.owner_cta_rank, region.allocation_id)
        ].append(region)

    overlapping_regions: dict[str, set[str]] = {
        region_id: {region_id}
        for region_id in effects_by_region
    }
    for regions in regions_by_allocation.values():
        active: dict[str, Any] = {}
        ends: list[tuple[int, str]] = []
        for region in sorted(regions, key=lambda item: (item.offset, item.id)):
            while ends and ends[0][0] <= region.offset:
                _end, expired = heappop(ends)
                active.pop(expired, None)
            for other in active.values():
                overlapping_regions[region.id].add(other.id)
                overlapping_regions[other.id].add(region.id)
            active[region.id] = region
            heappush(ends, (region.offset + region.extent, region.id))

    for left_index, left in enumerate(effects):
        streams = []
        for region_id in overlapping_regions[left.region]:
            indices = effects_by_region[region_id]
            position = bisect_right(indices, left_index)
            if position < len(indices):
                streams.append(islice(indices, position, None))
        for right_index in merge(*streams):
            yield left, effects[right_index]


def _arrivals(module: Module, resource: str, lifecycle_epoch: int, generation: int) -> list[Event]:
    return [event for event in module.events if event.resource == resource and event.lifecycle_epoch == lifecycle_epoch and event.generation == generation and event.predicate and event.type in {"MBarrierArrive", "MBarrierArriveExpectTx"}]


def _waits(module: Module, resource: str, lifecycle_epoch: int, generation: int) -> list[Event]:
    return [event for event in module.events if event.resource == resource and event.lifecycle_epoch == lifecycle_epoch and event.generation == generation and event.type == "MBarrierWaitParity"]


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
        for wait in _waits(module, arrival.resource, arrival.lifecycle_epoch, arrival.generation):
            if wait.actor == target_event.actor and graph.reachable(wait.id, target.begin):
                return True
    return False


def _collective_visibility(module: Module, graph: PDG, source: Effect, target: Effect) -> bool:
    region = module.region_map[source.region]
    if region.address_space not in {"SHARED", "DSMEM", "GLOBAL"}:
        return False
    source_event = module.event_map[source.end]
    target_event = module.event_map[target.begin]
    for resource in module.resources:
        if resource.kind not in {"cta_barrier", "named_barrier", "cluster_barrier"}:
            continue
        mask = resource.visibility_mask
        assert mask is not None
        if mask.control_only:
            continue
        if region.address_space == "SHARED" and not mask.shared:
            continue
        if region.address_space == "DSMEM" and (
            resource.kind != "cluster_barrier" or not mask.dsmem
        ):
            continue
        if region.address_space == "GLOBAL" and not (
            mask.global_write and (target.mode != "READ" or mask.global_read)
        ):
            continue
        arrivals = [
            event for event in module.events
            if event.type == "BarrierArrive"
            and event.resource == resource.id
            and event.actor == source_event.actor
            and event.predicate
            and graph.reachable(source.end, event.id)
        ]
        for arrival in arrivals:
            for observe in module.events:
                if (
                    observe.type == "BarrierObserve"
                    and observe.resource == resource.id
                    and observe.invocation_id == arrival.invocation_id
                    and observe.actor == target_event.actor
                    and graph.reachable(observe.id, target.begin)
                ):
                    return True
    return False


def _channel_visibility(module: Module, graph: PDG, source: Effect, target: Effect) -> bool:
    region = module.region_map[source.region]
    source_event = module.event_map[source.end]
    target_event = module.event_map[target.begin]
    for resource in module.resources:
        if resource.kind != "channel":
            continue
        mask = resource.visibility_mask
        assert mask is not None
        if mask.control_only:
            continue
        if region.address_space == "SHARED" and not mask.shared:
            continue
        if region.address_space == "DSMEM" and not mask.dsmem:
            continue
        if region.address_space == "GLOBAL" and not (
            (source.mode == "READ" and mask.global_read)
            or (source.mode == "WRITE" and mask.global_write)
        ):
            continue
        if region.address_space == "TENSOR" and not (
            (source.mode == "READ" and mask.tensor_read)
            or (source.mode == "WRITE" and mask.tensor_write)
        ):
            continue
        if region.address_space == "REGISTER":
            continue
        commits = [
            event for event in module.events
            if event.type == "ProducerCommit"
            and event.resource == resource.id
            and event.actor == source_event.actor
            and graph.reachable(source.end, event.id)
        ]
        for commit in commits:
            for wait in module.events:
                if (
                    wait.type == "ConsumerWait"
                    and wait.resource == resource.id
                    and wait.slot == commit.slot
                    and wait.logical_phase == commit.logical_phase
                    and wait.actor == target_event.actor
                    and graph.reachable(wait.id, target.begin)
                ):
                    return True
        releases = [
            event for event in module.events
            if event.type == "ConsumerRelease"
            and event.resource == resource.id
            and event.actor == source_event.actor
            and graph.reachable(source.end, event.id)
        ]
        for release in releases:
            for acquire in module.events:
                if (
                    acquire.type == "ProducerAcquire"
                    and acquire.resource == resource.id
                    and acquire.slot == release.slot
                    and acquire.logical_phase == release.logical_phase + 1
                    and acquire.actor == target_event.actor
                    and graph.reachable(acquire.id, target.begin)
                ):
                    return True
    return False


def _region_covers(module: Module, covering: str, covered: str) -> bool:
    outer = module.region_map[covering]
    inner = module.region_map[covered]
    return (
        outer.address_space == inner.address_space
        and outer.owner_cta_rank == inner.owner_cta_rank
        and outer.allocation_id == inner.allocation_id
        and outer.offset <= inner.offset
        and outer.offset + outer.extent >= inner.offset + inner.extent
    )


def _proxy_visible(module: Module, graph: PDG, point: Effect, window: Effect) -> bool:
    if point.mode != "WRITE" or window.mode != "READ":
        return False
    issue = module.event_map[window.begin]
    target_domain = {
        "CpAsyncIssue": "ASYNC_COPY",
        "TmaLoadIssue": "TMA",
        "TmaStoreIssue": "TMA",
        "WgmmaIssue": "TENSOR_CORE",
    }.get(issue.type)
    if target_domain is None:
        return False
    point_event = module.event_map[point.begin]
    source_region = module.region_map[window.region]
    for fence in module.events:
        if (
            fence.type != "ProxyFence"
            or fence.from_domain != "GENERIC"
            or fence.to_domain != target_domain
            or fence.scope != "CTA"
            or not graph.reachable(point.end, fence.id)
            or not graph.reachable(fence.id, window.begin)
            or not any(_region_covers(module, region, window.region) for region in fence.regions)
        ):
            continue
        if fence.actor != point_event.actor:
            fence_observation = Effect(
                id=f"proxy-source:{point.id}:{fence.id}",
                actor=fence.actor,
                mode="READ",
                region=point.region,
                begin=fence.id,
                end=fence.id,
                async_window=False,
            )
            if not _collective_visibility(module, graph, point, fence_observation):
                continue
        mask = fence.visibility_mask
        assert mask is not None
        if source_region.address_space == "SHARED" and mask.shared:
            return True
        if source_region.address_space == "GLOBAL" and mask.global_write:
            return True
        if source_region.address_space == "TENSOR" and mask.tensor_write:
            return True
    return False


def _completion_observed(module: Module, graph: PDG, window: Effect, point: Effect) -> bool:
    issue = module.event_map[window.begin]
    if issue.type in {"CpAsyncIssue", "WgmmaIssue", "TmaStoreIssue"}:
        point_event = module.event_map[point.begin]
        same_owner = point_event.actor == issue.actor
        if issue.type == "WgmmaIssue" and not same_owner:
            actors = module.actor_map
            owner = actors[issue.actor]
            consumer = actors[point_event.actor]
            same_owner = (
                owner.kind == "WARPGROUP"
                and set(consumer.participants).issubset(owner.participants)
            )
        return same_owner and graph.reachable(window.end, point.begin)
    point_event = module.event_map[point.begin]
    return any(
        wait.type == "MBarrierWaitParity"
        and wait.actor == point_event.actor
        and graph.reachable(window.end, wait.id)
        and graph.reachable(wait.id, point.begin)
        for wait in module.events
    )


def _ordered_wgmma_accumulations(module: Module, left: Effect, right: Effect) -> bool:
    if left.mode != "WRITE" or right.mode != "WRITE" or left.region != right.region:
        return False
    region = module.region_map[left.region]
    left_issue = module.event_map[left.begin]
    right_issue = module.event_map[right.begin]
    return (
        region.address_space == "REGISTER"
        and left_issue.type == "WgmmaIssue"
        and right_issue.type == "WgmmaIssue"
        and left_issue.resource == right_issue.resource
        and left_issue.group_owner == right_issue.group_owner
        and left_issue.engine_id is not None
        and left_issue.engine_id == right_issue.engine_id
        and left_issue.engine_seq is not None
        and right_issue.engine_seq is not None
        and left_issue.engine_seq != right_issue.engine_seq
    )


def _data_hazard(module: Module, graph: PDG) -> dict[str, Any] | None:
    effects = _effects(module)
    for left, right in _overlapping_effect_pairs(module, effects):
        if left.actor == right.actor and not left.async_window and not right.async_window:
            continue
        if left.mode == right.mode == "READ":
            continue
        safe = False
        query = ""
        if not left.async_window and not right.async_window:
            if left.actor == right.actor:
                safe = graph.reachable(left.end, right.begin) or graph.reachable(right.end, left.begin)
            else:
                safe = (
                    _mbarrier_visibility(module, graph, left, right)
                    or _mbarrier_visibility(module, graph, right, left)
                    or _collective_visibility(module, graph, left, right)
                    or _collective_visibility(module, graph, right, left)
                    or _channel_visibility(module, graph, left, right)
                    or _channel_visibility(module, graph, right, left)
                )
            query = "point conflict requires same-actor order or a typed mbarrier/collective/channel visibility transfer"
        elif left.async_window and right.async_window:
            safe = (
                graph.reachable(left.end, right.begin)
                or graph.reachable(right.end, left.begin)
                or _ordered_wgmma_accumulations(module, left, right)
            )
            query = "async windows must be non-overlapping, except ordered WGMMA updates to one register accumulator"
        else:
            window, point = (left, right) if left.async_window else (right, left)
            if window.mode == "READ":
                # A generic WRITE feeding an async READ requires a matching
                # generic-to-async proxy fence.
                point_before = _proxy_visible(module, graph, point, window)
                completion_before = _completion_observed(module, graph, window, point)
                query = "generic write before async read requires a direction/scope/region-compatible proxy fence, or overwrite must follow observed completion"
            else:
                point_before = graph.reachable(point.end, window.begin)
                completion_before = _completion_observed(module, graph, window, point)
                query = "async-copy destination conflict must precede issue or follow observed completion"
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


def _finish_incremental_statistics(statistics: dict[str, Any], counters: IncrementalCounters) -> None:
    certificate = counters.certificate()
    statistics["explored_steps"] = counters.successful_transitions
    statistics["program_streams"] = counters.program_streams
    statistics["phase_streams"] = counters.phase_streams
    statistics["barrier_release_streams"] = counters.barrier_release_streams
    statistics["group_completion_streams"] = counters.group_completion_streams
    statistics["channel_handoff_streams"] = counters.channel_handoff_streams
    statistics["engine_completion_streams"] = counters.engine_completion_streams
    statistics["anchor_streams"] = counters.anchor_streams
    statistics["anchors"] = counters.anchors
    for key in (
        "n_ext", "t_eff", "state_initializations", "completed_passes", "anchor_boundaries",
        "transition_attempts", "scheduler_work_units", "worklist_pushes", "worklist_pops",
        "stale_worklist_pops",
        "dependency_registrations", "dependency_notifications", "wait_registrations",
        "wait_notifications", "resource_notifications", "collective_notifications", "group_notifications", "channel_notifications", "engine_notifications", "frontier_exposures",
        "phase_enable_checks", "remaining_removals", "boundary_evaluations",
        "full_enabled_scans", "state_clones", "replayed_prefix_steps",
    ):
        statistics[key] = certificate[key]
    statistics["transitions_by_kind"] = certificate["transitions_by_kind"]
    statistics["certificate"] = certificate


def _finish_pdg_statistics(statistics: dict[str, Any], graph: PDG) -> None:
    statistics.update(graph.statistics())


def _protocol_report(
    error: ProtocolError,
    trace: list[dict[str, Any]],
    graph: PDG,
    boundaries: list[dict[str, Any]],
    digest: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    _finish_pdg_statistics(statistics, graph)
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
    _finish_pdg_statistics(statistics, graph)
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
    graph.add_edges(state.dynamic_edges)


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


def _independent_program_memory_only(module: Module) -> bool:
    return all(
        event.type in {"MemRead", "MemWrite"} and not event.depends_on
        for event in module.events
    )


def _run_engine(module: Module, engine: Literal["maximal", "exhaustive"]) -> dict[str, Any]:
    normalized = normalize_module(module)
    digest = stable_digest(normalized)
    executor = Executor(module)
    graph = _base_pdg(module)
    anchors = _anchor_ids(module)
    incremental = IncrementalEngine(executor, anchors) if engine == "maximal" else None
    canonical = incremental.canonical() if incremental is not None else executor.explore()
    boundaries: list[dict[str, Any]] = []
    statistics = _statistics(module, engine)
    if incremental is None:
        statistics["explored_steps"] += len(canonical.state.trace)
    else:
        _finish_incremental_statistics(statistics, incremental.counters)

    if engine == "maximal":
        assert incremental is not None
        if canonical.error is not None:
            return _protocol_report(canonical.error, canonical.state.trace, graph, boundaries, digest, statistics)
        if canonical.stuck:
            return _stuck_report(executor, canonical.state, graph, boundaries, digest, statistics)

        if _independent_program_memory_only(module):
            boundaries = [
                {"anchor": anchor, "boundary": [anchor]}
                for anchor in anchors
            ]
            incremental.counters.anchor_boundaries = len(boundaries)
            _finish_incremental_statistics(statistics, incremental.counters)
            statistics["analytical_program_only_boundaries"] = len(boundaries)
        else:
            outcomes = incremental.analyze_streams()
            boundaries = sorted(outcomes.boundaries, key=lambda item: item["anchor"])
            graph.add_edges(outcomes.dynamic_edges)
            _finish_incremental_statistics(statistics, incremental.counters)
            if outcomes.error is not None:
                assert outcomes.error_state is not None
                return _protocol_report(outcomes.error, outcomes.error_state.trace, graph, boundaries, digest, statistics)
            if outcomes.stuck_state is not None:
                return _stuck_report(executor, outcomes.stuck_state, graph, boundaries, digest, statistics)
            if outcomes.unreachable_anchor is not None:
                assert outcomes.unreachable_state is not None
                return _inconclusive(
                    "ANCHOR_NOT_REACHED",
                    "held anchor was not enabled at its terminal boundary",
                    digest,
                    graph,
                    boundaries,
                    statistics,
                    {
                        "anchor": outcomes.unreachable_anchor,
                        "boundary": list(executor.boundary(outcomes.unreachable_state)),
                    },
                )
            for item in boundaries:
                anchor = item["anchor"]
                for event_id in item["boundary"]:
                    if event_id != anchor:
                        graph.add_edge(anchor, event_id, "BOUNDARY_ORDER")
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

        for anchor in anchors:
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
    _finish_pdg_statistics(statistics, graph)
    engine_streams = engine_completion_streams(module)
    artifacts = {
        "normalized_sha256": digest, "canonical_trace": canonical.state.trace,
        "boundaries": sorted(boundaries, key=lambda item: item["anchor"]),
        "engine_completion_streams": {
            engine: list(endpoints)
            for engine, endpoints in sorted(engine_streams.items())
        },
        "pdg": graph.artifact(),
    }
    counts = {key: statistics[key] for key in ("actors", "events", "resources", "regions")}
    if hazard is not None:
        hazard["reachable_execution_prefix"] = canonical.state.trace
        hazard["protocol_state"] = {
            **{key: value.snapshot() for key, value in sorted(canonical.state.barriers.items())},
            **{
                f"collective:{key}": value.snapshot()
                for key, value in sorted(canonical.state.collectives.items())
            },
            **{
                f"group:{key}": value.snapshot()
                for key, value in sorted(canonical.state.groups.items())
            },
            **{
                f"channel:{key}": value.snapshot()
                for key, value in sorted(canonical.state.channels.items())
            },
            "visibility_transfers": [
                transfer.snapshot()
                for transfer in sorted(canonical.state.visibility_transfers, key=lambda item: item.event)
            ],
        }
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
            "scope": (
                "finite statically-resolved sm90 single_cluster PGIR-v2 cluster-barrier/DSMEM core"
                if module.raw["schema_version"] == "pgir-v2"
                else (
                    "finite statically-resolved sm90 selected CTA "
                    f"{module.raw['target']['selected_cta']} from launch grid "
                    f"{module.raw['target']['launch_grid']} mbarrier/TMA-load/TMA-store/"
                    "collective-barrier/cp.async/WGMMA/proxy-visibility/explicit-engine-order core"
                    if "selected_cta" in module.raw["target"]
                    else "finite statically-resolved sm90 single_cta mbarrier/TMA-load/TMA-store/collective-barrier/cp.async/WGMMA/proxy-visibility/explicit-engine-order core"
                )
            ),
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
    base_fields = {"actors", "events", "resources", "regions", "explored_steps", "explored_states", "engine"}
    for key, value in maximal["statistics"].items():
        if key not in base_fields:
            statistics[key] = value
    statistics["explored_steps"] = maximal["statistics"]["explored_steps"] + exhaustive["statistics"]["explored_steps"]
    statistics["explored_states"] = exhaustive["statistics"]["explored_states"]
    statistics["maximal_steps"] = maximal["statistics"]["explored_steps"]
    statistics["exhaustive_states"] = exhaustive["statistics"]["explored_states"]
    statistics["maximal_certificate"] = maximal["statistics"].get("certificate", {})

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
