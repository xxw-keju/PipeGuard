from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .engine import ProtocolError
from .model import MEMORY_EVENTS
from .parametric import (
    EventTemplate,
    ParametricModule,
    stable_collective_event_id,
    stable_stream_instance_id,
)


def _atomic_pair(left: Effect, right: Effect) -> bool:
    # Read-modify-writes are atomic with respect to each other.
    return all(e.kind == "AtomicReadModifyWrite" or e.atomic for e in (left, right))


@dataclass(slots=True)
class MBarrierState:
    lifecycle: str = "UNINITIALIZED"
    lifecycle_epoch: int = -1
    generation: int = 0
    base_count: int = 0
    pending: int = 0
    expected_tx: int = 0
    issued_tx: int = 0
    completed_tx: int = 0
    pending_tx_completions: int = 0
    deferred_arrivals: int = 0
    has_deferred_arrival: bool = False
    started: bool = False
    publications: set[int] = field(default_factory=set)
    observed_generation: int | None = None
    contributors: list[int] = field(default_factory=list)
    phase_sources: dict[int, dict[int, int]] = field(default_factory=dict)
    last_phase_contributors: tuple[int, ...] = ()
    last_phase_transactional: bool = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "lifecycle": self.lifecycle,
            "lifecycle_epoch": self.lifecycle_epoch,
            "logical_generation": self.generation,
            "hardware_parity": self.generation & 1,
            "pending_arrivals_current": self.pending,
            "expected_tx_bytes_current": self.expected_tx,
            "issued_tx_bytes_current": self.issued_tx,
            "completed_tx_bytes_current": self.completed_tx,
            "pending_tma_completions": self.pending_tx_completions,
            "publications": sorted(self.publications),
            "last_successfully_observed_generation": self.observed_generation,
        }


@dataclass(slots=True)
class CollectiveState:
    capacity: int
    invocation: int = 0
    arrived: int = 0
    arrivals: list[int] = field(default_factory=list)
    releases: dict[int, int] = field(default_factory=dict)
    release_sources: dict[int, dict[int, int]] = field(default_factory=dict)


@dataclass(slots=True)
class Group:
    sequence: int
    commit: int
    issues: tuple[int, ...]
    endpoints: dict[str, int]
    completed: set[str] = field(default_factory=set)


@dataclass(slots=True)
class GroupState:
    protocol: str
    completion_kinds: tuple[str, ...]
    staged: dict[int, list[int]] = field(default_factory=dict)
    next_sequence: dict[int, int] = field(default_factory=dict)
    groups: dict[int, list[Group]] = field(default_factory=dict)
    completed_frontiers: dict[tuple[int, str], int] = field(default_factory=dict)
    observation_frontiers: dict[tuple[int, str, int], int] = field(default_factory=dict)
    fences: dict[int, set[int]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        # JSON-safe witness form; keys are stringified and sets ordered so the
        # diagnostic report stays serializable (see _error in this module).
        return {
            "protocol": self.protocol,
            "completion_kinds": list(self.completion_kinds),
            "staged": {str(k): list(v) for k, v in self.staged.items()},
            "next_sequence": {str(k): v for k, v in self.next_sequence.items()},
            "groups": {
                str(owner): [{
                    "sequence": g.sequence,
                    "commit": g.commit,
                    "issues": list(g.issues),
                    "endpoints": dict(g.endpoints),
                    "completed": sorted(g.completed),
                } for g in items]
                for owner, items in self.groups.items()
            },
            "completed_frontiers": {str(k): v for k, v in self.completed_frontiers.items()},
            "observation_frontiers": {str(k): v for k, v in self.observation_frontiers.items()},
            "fences": {str(k): sorted(v) for k, v in self.fences.items()},
        }


@dataclass(frozen=True, slots=True)
class VisibilityCut:
    source_positions: dict[int, int]
    destination_positions: dict[int, int]
    shared: bool
    global_read: bool
    global_write: bool


@dataclass(frozen=True, slots=True)
class Effect:
    node: int
    lane: int
    mode: str
    region: int
    end: int
    asynchronous: bool
    kind: str
    atomic: bool = False


class NumericPDG:
    """Implicit lane-order graph with explicit typed boundary nodes."""

    def __init__(self, module: ParametricModule) -> None:
        self.module = module
        self.logical_nodes = len(module.lane_trace_template_ids)
        self.next_node = self.logical_nodes
        self.adjacency: dict[int, set[int]] = defaultdict(set)
        self.edge_kinds: dict[tuple[int, int], str] = {}
        self.synthetic_labels: dict[int, str] = {}
        self.transition_nodes: dict[int, int] = {}
        self.transition_templates: dict[int, int] = {}
        self.occurrence_lane: list[int] = [0] * self.logical_nodes
        self.occurrence_position: list[int] = [0] * self.logical_nodes
        self.occurrence_template: list[int] = list(module.lane_trace_template_ids)
        self.occurrence_by_lane_template: dict[tuple[int, int], int] = {}
        self.lane_slot = {lane: slot for slot, lane in enumerate(module.lane_ids)}
        for slot, lane in enumerate(module.lane_ids):
            begin = module.lane_trace_offsets[slot]
            end = module.lane_trace_offsets[slot + 1]
            for position, node in enumerate(range(begin, end)):
                template = module.lane_trace_template_ids[node]
                self.occurrence_lane[node] = lane
                self.occurrence_position[node] = position
                self.occurrence_by_lane_template[(lane, template)] = node
        for template_id, template in enumerate(module.templates):
            if template.rendezvous != "ALL_PROGRAM_INSTANCES":
                continue
            transition = self.synthetic(f"transition:{template.id}")
            self.transition_nodes[template_id] = transition
            self.transition_templates[transition] = template_id
            for lane in template.program_instances:
                occurrence = self.occurrence_by_lane_template[(lane, template_id)]
                self.add_edge(occurrence, transition, "RENDEZVOUS_ARRIVE")
                successor = self.lane_successor(occurrence, cross_rendezvous=True)
                if successor is not None:
                    self.add_edge(transition, successor, "RENDEZVOUS_RELEASE")
        self.reachability_queries = 0
        self.reachability_edge_visits = 0
        self.reachability_cache_hits = 0
        self._synthetic_indices: dict[int, int] | None = None
        self._synthetic_ancestors: tuple[int, ...] | None = None
        self._synthetic_descendants: tuple[int, ...] | None = None
        self._logical_cross_lane_edges = True
        self._topological_positions: tuple[int, ...] | None = None
        self._predecessors: dict[int, list[int]] | None = None

    def synthetic(self, label: str) -> int:
        node = self.next_node
        self.next_node += 1
        self.synthetic_labels[node] = label
        return node

    def add_edge(self, source: int, target: int, kind: str) -> None:
        if target not in self.adjacency[source]:
            self.adjacency[source].add(target)
            self.edge_kinds[(source, target)] = kind
            self._synthetic_indices = None
            self._synthetic_ancestors = None
            self._synthetic_descendants = None
            self._topological_positions = None
            self._predecessors = None

    def lane_successor(self, node: int, *, cross_rendezvous: bool = False) -> int | None:
        if node >= self.logical_nodes:
            return None
        lane = self.occurrence_lane[node]
        slot = self.lane_slot[lane]
        position = self.occurrence_position[node]
        begin = self.module.lane_trace_offsets[slot]
        end = self.module.lane_trace_offsets[slot + 1]
        template_id = self.occurrence_template[node]
        template = self.module.templates[template_id]
        if template.rendezvous == "ALL_PROGRAM_INSTANCES" and not cross_rendezvous:
            return self.transition_nodes[template_id]
        candidate = begin + position + 1
        return candidate if candidate < end else None

    def successors(self, node: int) -> tuple[int, ...]:
        result = set(self.adjacency.get(node, ()))
        successor = self.lane_successor(node)
        if successor is not None:
            result.add(successor)
        return tuple(result)

    def _build_synthetic_reachability(self) -> None:
        node_count = self.next_node
        synthetic_nodes = tuple(range(self.logical_nodes, node_count))
        synthetic_indices = {
            node: index for index, node in enumerate(synthetic_nodes)
        }
        successors = [self.successors(node) for node in range(node_count)]
        indegree = [0] * node_count
        for targets in successors:
            for target in targets:
                indegree[target] += 1
        queue = deque(node for node, degree in enumerate(indegree) if degree == 0)
        ancestors = [0] * node_count
        for node, index in synthetic_indices.items():
            ancestors[node] = 1 << index
        topological_positions = [0] * node_count
        topological_order = []
        visited = 0
        while queue:
            node = queue.popleft()
            topological_positions[node] = visited
            topological_order.append(node)
            visited += 1
            source_bits = ancestors[node]
            targets = successors[node]
            self.reachability_edge_visits += len(targets)
            for target in targets:
                ancestors[target] |= source_bits
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != node_count:
            raise RuntimeError("numeric PDG contains a cycle")
        descendants = [0] * node_count
        for node in reversed(topological_order):
            bits = 1 << synthetic_indices[node] if node in synthetic_indices else 0
            for target in successors[node]:
                bits |= descendants[target]
            descendants[node] = bits
        self._synthetic_indices = synthetic_indices
        self._synthetic_ancestors = tuple(ancestors)
        self._synthetic_descendants = tuple(descendants)
        self._logical_cross_lane_edges = any(
            target < self.logical_nodes
            and self.occurrence_lane[source] != self.occurrence_lane[target]
            for source, targets in self.adjacency.items()
            if source < self.logical_nodes
            for target in targets
        )
        self._topological_positions = tuple(topological_positions)

    def reachable(self, source: int, target: int) -> bool:
        self.reachability_queries += 1
        if source == target:
            return True
        if source < self.logical_nodes and target < self.logical_nodes:
            if self.occurrence_lane[source] == self.occurrence_lane[target]:
                return self.occurrence_position[source] <= self.occurrence_position[target]
        if source < self.logical_nodes:
            transition_template = self.transition_templates.get(target)
            if transition_template is not None:
                lane = self.occurrence_lane[source]
                occurrence = self.occurrence_by_lane_template.get(
                    (lane, transition_template)
                )
                if (
                    occurrence is not None
                    and self.occurrence_position[source]
                    <= self.occurrence_position[occurrence]
                ):
                    return True
        if self._synthetic_ancestors is None:
            self._build_synthetic_reachability()
        else:
            self.reachability_cache_hits += 1
        assert self._synthetic_indices is not None
        assert self._synthetic_ancestors is not None
        assert self._topological_positions is not None
        if self._topological_positions[source] >= self._topological_positions[target]:
            return False
        if source >= self.logical_nodes:
            index = self._synthetic_indices[source]
            return bool(self._synthetic_ancestors[target] & (1 << index))
        if target >= self.logical_nodes or not self._logical_cross_lane_edges:
            # Lane order is handled above, so without logical edges between
            # lanes every remaining path passes through a synthetic node.
            assert self._synthetic_descendants is not None
            return bool(
                self._synthetic_descendants[source] & self._synthetic_ancestors[target]
            )
        queue = deque([source])
        seen = {source}
        while queue:
            node = queue.popleft()
            successors = self.successors(node)
            self.reachability_edge_visits += len(successors)
            for successor in successors:
                if successor == target:
                    return True
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        return False

    def _dependency_sources(self, node: int) -> tuple[int, ...]:
        template_id = self.occurrence_template[node]
        template = self.module.templates[template_id]
        if not template.dependencies:
            return ()
        lane = self.occurrence_lane[node]
        sources: list[int] = []
        for dependency in template.dependencies:
            source_id = self.module.template_indices[dependency.template]
            source = self.module.templates[source_id]
            if source_id in self.transition_nodes:
                sources.append(self.transition_nodes[source_id])
            elif dependency.relation == "SAME_INSTANCE":
                sources.append(self.occurrence_by_lane_template[(lane, source_id)])
            elif dependency.relation == "ALL_INSTANCES":
                sources.extend(
                    self.occurrence_by_lane_template[(item, source_id)]
                    for item in source.program_instances
                )
            else:
                sources.append(
                    self.occurrence_by_lane_template[(min(source.executors), source_id)]
                )
        return tuple(sources)

    def executes_before_all(self, sources: tuple[int, ...], target: int) -> bool:
        """Whether every source node executes before ``target`` (one backward walk).

        This is execution order, not memory visibility: unlike the PDG it also
        follows explicit ``depends_on`` references. Ancestry within a lane is
        prefix-closed, so each lane is tracked by the deepest position reached.
        """
        logical: dict[int, int] = {}
        synthetic: set[int] = set()
        for source in sources:
            if source == target:
                continue
            if source < self.logical_nodes:
                lane = self.occurrence_lane[source]
                logical[lane] = max(logical.get(lane, -1), self.occurrence_position[source])
            else:
                synthetic.add(source)
        if not logical and not synthetic:
            return True
        self.reachability_queries += 1
        if self._predecessors is None:
            predecessors: dict[int, list[int]] = defaultdict(list)
            for source, targets in self.adjacency.items():
                for successor in targets:
                    predecessors[successor].append(source)
            self._predecessors = predecessors
        predecessors = self._predecessors
        depth: dict[int, int] = {}
        seen: set[int] = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node < self.logical_nodes:
                lane = self.occurrence_lane[node]
                position = self.occurrence_position[node]
                previous = depth.get(lane, -1)
                if position <= previous:
                    continue
                depth[lane] = position
                if logical.get(lane, position + 1) <= position:
                    del logical[lane]
                begin = self.module.lane_trace_offsets[self.lane_slot[lane]]
                for ancestor in range(begin + previous + 1, begin + position + 1):
                    self.reachability_edge_visits += 1
                    stack.extend(predecessors.get(ancestor, ()))
                    stack.extend(self._dependency_sources(ancestor))
            else:
                if node in seen:
                    continue
                seen.add(node)
                synthetic.discard(node)
                stack.extend(predecessors.get(node, ()))
            if not logical and not synthetic:
                return True
        return False

    def label(self, node: int) -> str:
        if node >= self.logical_nodes:
            return self.synthetic_labels[node]
        lane = self.occurrence_lane[node]
        template = self.module.templates[self.occurrence_template[node]]
        return f"{template.id}:lane:{lane}"

    def artifact(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for kind in self.edge_kinds.values():
            counts[kind] += 1
        return {
            "representation": "parametric",
            "implementation": "numeric-csr",
            "logical_program_nodes": self.logical_nodes,
            "implicit_program_edges": max(0, self.logical_nodes - len(self.module.lane_ids)),
            "synthetic_nodes": len(self.synthetic_labels),
            "explicit_boundary_edges": len(self.edge_kinds),
            "edges_by_kind": dict(sorted(counts.items())),
        }


class DirectParametricChecker:
    def __init__(self, module: ParametricModule) -> None:
        self.module = module
        self.graph = NumericPDG(module)
        self.resource_ids = {
            resource["id"]: index
            for index, resource in enumerate(module.raw["resources"])
        }
        self.resources = module.raw["resources"]
        self.region_ids = {
            region["id"]: index for index, region in enumerate(module.raw["regions"])
        }
        self.regions = module.raw["regions"]
        self.owner_ids = {
            owner["id"]: index for index, owner in enumerate(module.raw["owners"])
        }
        self.owner_names = tuple(owner["id"] for owner in module.raw["owners"])
        self.lane_owner_base = len(self.owner_ids)
        self.template_resources: tuple[tuple[int, ...] | None, ...] = tuple(
            (
                tuple(
                    self.resource_ids[template.field("resource", lane)]
                    for lane in template.program_instances
                )
                if "resource" in template.fields
                or "resource" in template.instance_fields
                else None
            )
            for template in module.templates
        )
        self.template_group_owners: tuple[tuple[int, ...] | None, ...] = tuple(
            (
                tuple(
                    self.lane_owner_base + lane
                    if template.field("group_owner", lane) in {None, "$self"}
                    else self.owner_ids[template.field("group_owner", lane)]
                    for lane in template.program_instances
                )
                if "group_owner" in template.fields
                or "group_owner" in template.instance_fields
                else None
            )
            for template in module.templates
        )
        self.frontiers = [0] * len(module.lane_ids)
        self.collective_arrived_bits: dict[int, int] = {}
        self.collective_ready_counts = [0] * len(module.templates)
        self.crossed: set[int] = set()
        self.crossed_counts = [0] * len(module.templates)
        self.once_executed: set[int] = set()
        self.mbarriers: dict[int, MBarrierState] = {}
        self.collectives: dict[int, CollectiveState] = {}
        self.groups: dict[int, GroupState] = {}
        self.completed: set[int] = set()
        self.deferred_cp = any(t.type == "CpAsyncMBarrierArrive" for t in module.templates)
        self.cp_copy_endpoints: dict[int, int] = {}
        self.cp_issues: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.cp_tracked: set[int] = set()
        self.endpoint_nodes: dict[tuple[Any, ...], int] = {}
        self.effects: list[Effect] = []
        self.pending_effects: dict[int, list[tuple[int, int, str, int, str]]] = defaultdict(list)
        self.engine_endpoints: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        self.visibility_cuts: list[VisibilityCut] = []
        self.completion_observations: dict[int, dict[int, int]] = defaultdict(dict)
        self.immediate_notifications: set[tuple[Any, ...]] = set()
        self.completion_queue: deque[tuple[Any, ...]] = deque()
        self.queued_completions: set[tuple[Any, ...]] = set()
        self.completion_queue_pushes = 0
        self.completion_queue_pops = 0
        self.proxy_fences: list[tuple[
            int,
            str,
            str,
            str,
            dict[
                tuple[str, int | None, str],
                tuple[tuple[int, ...], tuple[int, ...]],
            ],
        ]] = []
        self.proxy_fence_ancestor_masks: tuple[int, ...] = ()
        self.proxy_fence_descendant_masks: tuple[int, ...] = ()
        self.proxy_region_masks: dict[tuple[str, int], int] = {}
        self.proxy_fence_index_built = False
        self.proxy_fence_index_build_visits = 0
        self.proxy_fence_index_queries = 0
        self.executed_transitions = 0
        self.frontier_visits = 0
        self.collective_arrivals = 0
        self.deferred_waits = 0
        self.completion_transitions = 0
        self.group_wait_readiness_checks = 0
        self.group_completion_frontier_advances = 0
        self.group_wait_observation_advances = 0
        self.protocol_seconds = 0.0
        self.boundary_finalize_seconds = 0.0
        self.race_seconds = 0.0
        self.race_ordering_effect_visits = 0
        self.race_sweep_insertions = 0
        self.race_sweep_expirations = 0
        self.race_sweep_candidate_visits = 0
        self._initialize_resources()
        self._collect_point_effects()

    def _initialize_resources(self) -> None:
        for resource_id, raw in enumerate(self.resources):
            kind = raw["kind"]
            if kind == "mbarrier":
                self.mbarriers[resource_id] = MBarrierState()
            elif kind in {"cta_barrier", "named_barrier"}:
                self.collectives[resource_id] = CollectiveState(raw["capacity"])
            elif kind == "async_group":
                protocol = raw["protocol"]
                kinds = {
                    "CP_ASYNC": ("GROUP_COMPLETE",),
                    "WGMMA": ("OPERANDS_RELEASED", "RESULT_READY"),
                    "TMA_STORE": ("SOURCE_RELEASED", "GLOBAL_COMPLETE"),
                }[protocol]
                self.groups[resource_id] = GroupState(protocol, kinds)

    def _collect_point_effects(self) -> None:
        for template_id, template in enumerate(self.module.templates):
            if template.type not in MEMORY_EVENTS:
                continue
            mode = "READ" if template.type == "MemRead" else "WRITE"
            for lane in template.program_instances:
                node = self.graph.occurrence_by_lane_template[(lane, template_id)]
                for region in template.memory_regions(lane):
                    self.effects.append(Effect(
                        node, lane, mode, self.region_ids[region], node, False, template.type
                    ))

    def _declared_effects(self) -> list[Effect]:
        effects = [
            effect
            for effect in self.effects
            if effect.kind in MEMORY_EVENTS
        ]
        for template_id, template in enumerate(self.module.templates):
            kind = template.type
            if kind not in {
                "TmaLoadIssue", "CpAsyncIssue", "WgmmaIssue", "TmaStoreIssue",
            }:
                continue
            lanes = (
                template.program_instances
                if template.transition == "PER_INSTANCE"
                else (min(template.executors),)
            )
            for lane in lanes:
                node = self.graph.occurrence_by_lane_template[(lane, template_id)]
                if kind == "TmaLoadIssue":
                    transfers = self._field(template, "transfers", lane, [])
                    if not transfers:
                        source = self._field(template, "source_region", lane)
                        destination = self._field(template, "destination_region", lane)
                        if source is not None and destination is not None:
                            transfers = [{
                                "source_region": source,
                                "destination_region": destination,
                            }]
                    for transfer in transfers:
                        effects.extend((
                            Effect(
                                node,
                                lane,
                                "READ",
                                self.region_ids[transfer["source_region"]],
                                node,
                                True,
                                kind,
                            ),
                            Effect(
                                node,
                                lane,
                                "WRITE",
                                self.region_ids[transfer["destination_region"]],
                                node,
                                True,
                                kind,
                            ),
                        ))
                elif kind == "CpAsyncIssue":
                    if not self._field(template, "predicate", lane, True):
                        continue
                    effects.extend((
                        Effect(
                            node,
                            lane,
                            "READ",
                            self.region_ids[
                                self._field(template, "source_region", lane)
                            ],
                            node,
                            True,
                            kind,
                        ),
                        Effect(
                            node,
                            lane,
                            "WRITE",
                            self.region_ids[
                                self._field(template, "destination_region", lane)
                            ],
                            node,
                            True,
                            kind,
                        ),
                    ))
                elif kind == "WgmmaIssue":
                    for region in self._field(template, "regions", lane, []):
                        effects.append(Effect(
                            node,
                            lane,
                            "READ",
                            self.region_ids[region],
                            node,
                            True,
                            kind,
                        ))
                    effects.append(Effect(
                        node,
                        lane,
                        "WRITE",
                        self.region_ids[
                            self._field(template, "destination_region", lane)
                        ],
                        node,
                        True,
                        kind,
                    ))
                else:
                    effects.extend((
                        Effect(
                            node,
                            lane,
                            "READ",
                            self.region_ids[
                                self._field(template, "source_region", lane)
                            ],
                            node,
                            True,
                            kind,
                        ),
                        Effect(
                            node,
                            lane,
                            "WRITE",
                            self.region_ids[
                                self._field(template, "destination_region", lane)
                            ],
                            node,
                            True,
                            kind,
                            bool(self._field(template, "reduction", lane, False)),
                        ),
                    ))
        return effects

    def _resource(self, template: EventTemplate, lane: int) -> int:
        bindings = self.template_resources[template.index]
        if bindings is None:
            raise KeyError((template.id, "resource"))
        return bindings[template.instance_positions[lane]]

    def _owner(self, template: EventTemplate, lane: int) -> int:
        bindings = self.template_group_owners[template.index]
        if bindings is None:
            return self.lane_owner_base + lane
        return bindings[template.instance_positions[lane]]

    def _source_node(self, template_id: int, lane: int) -> int:
        template = self.module.templates[template_id]
        if template.rendezvous == "ALL_PROGRAM_INSTANCES":
            return self.graph.transition_nodes[template_id]
        executor = lane if template.transition == "PER_INSTANCE" else min(template.executors)
        return self.graph.occurrence_by_lane_template[(executor, template_id)]

    def _endpoint(self, key: tuple[Any, ...], label: str) -> int:
        node = self.endpoint_nodes.get(key)
        if node is None:
            node = self.graph.synthetic(label)
            self.endpoint_nodes[key] = node
        return node

    def _record_engine_endpoints(
        self,
        template: EventTemplate,
        lane: int,
        endpoints: tuple[int, ...],
    ) -> None:
        engine = self._field(template, "engine_id", lane)
        sequence = self._field(template, "engine_seq", lane)
        if engine is None or sequence is None:
            return
        self.engine_endpoints[self.owner_ids[engine]].append((sequence, endpoints))

    def _field(self, template: EventTemplate, name: str, lane: int, default: Any = None) -> Any:
        value = template.field(name, lane)
        return default if value is None else value

    def _error(self, code: str, message: str, template: EventTemplate, state: Any) -> ProtocolError:
        snapshot = state.snapshot() if hasattr(state, "snapshot") else state
        return ProtocolError(code, message, template.id, snapshot)

    def _barrier(self, template: EventTemplate, lane: int) -> tuple[int, MBarrierState]:
        resource = self._resource(template, lane)
        state = self.mbarriers[resource]
        if state.lifecycle == "UNINITIALIZED":
            raise self._error("USE_BEFORE_INIT", "mbarrier used before initialization", template, state)
        if state.lifecycle == "INVALIDATED":
            raise self._error("USE_AFTER_INVALIDATE", "mbarrier used after invalidation", template, state)
        epoch = self._field(template, "lifecycle_epoch", lane)
        if epoch != state.lifecycle_epoch:
            code = "STALE_LIFECYCLE" if epoch < state.lifecycle_epoch else "FUTURE_LIFECYCLE"
            raise self._error(code, "event lifecycle does not match the live mbarrier", template, state)
        return resource, state

    def _retired_without_wait(self, state: MBarrierState, node: int) -> bool:
        # PTX forbids invalidating an in-progress phase, not an unobserved
        # completed one. Without a wait, the completion is ordered before the
        # invalidation only if it was arrival-driven (no transaction bytes) and
        # every contributing arrival executes before the invalidation.
        if state.last_phase_transactional:
            return False
        return self.graph.executes_before_all(state.last_phase_contributors, node)

    def _require_generation(self, template: EventTemplate, lane: int, state: MBarrierState) -> None:
        generation = self._field(template, "generation", lane)
        if generation < state.generation:
            raise self._error("STALE_GENERATION", "event targets a retired generation", template, state)
        if generation > state.generation:
            code = "PHASE_LAP" if generation > state.generation + 1 else "EARLY_NEXT_GENERATION"
            raise self._error(code, "event reached a generation before the current phase completed", template, state)

    def _group_wait_ready(self, template: EventTemplate, lane: int, kind: str) -> bool:
        resource = self._resource(template, lane)
        owner = self._owner(template, lane)
        state = self.groups[resource]
        if set(state.staged.get(owner, ())) - self.cp_tracked:
            raise self._error("WAIT_BEFORE_COMMIT", "group wait reached uncommitted operations", template, state)
        groups = state.groups.get(owner, [])
        threshold = self._field(template, "wait_group_count", lane)
        cutoff = max(0, len(groups) - threshold)
        self.group_wait_readiness_checks += 1
        return state.completed_frontiers.get((owner, kind), 0) >= cutoff

    def _dependency_blockers(self, template: EventTemplate, lane: int) -> tuple[tuple[Any, ...], ...]:
        blockers: list[tuple[Any, ...]] = []
        for dependency in template.dependencies:
            source_id = self.module.template_indices[dependency.template]
            source = self.module.templates[source_id]
            if dependency.relation == "SAME_INSTANCE":
                node = self.graph.occurrence_by_lane_template.get((lane, source_id))
                if node is None or node not in self.crossed:
                    blockers.append(("same", source_id, lane))
            elif dependency.relation == "ALL_INSTANCES":
                if self.crossed_counts[source_id] != len(source.program_instances):
                    blockers.append(("all", source_id))
            elif source_id not in self.once_executed:
                blockers.append(("once", source_id))
        return tuple(blockers)

    def _protocol_blocker(self, template: EventTemplate, lane: int) -> tuple[Any, ...] | None:
        if template.type == "MBarrierWaitParity":
            resource, state = self._barrier(template, lane)
            generation = self._field(template, "generation", lane)
            parity = self._field(template, "raw_parity", lane)
            if parity != generation % 2:
                raise self._error("WRONG_PARITY", "wait parity disagrees with its logical generation", template, state)
            if generation < state.generation - 1 or generation > state.generation:
                raise self._error(
                    "PROTOCOL_INVALID_PHASE_ADVANCE",
                    "wait does not name a current or preceding phase",
                    template,
                    state,
                )
            if generation >= state.generation or generation not in state.publications:
                return ("resource", resource)
        elif template.type == "BarrierObserve":
            resource = self._resource(template, lane)
            invocation = self._field(template, "invocation_id", lane)
            if invocation not in self.collectives[resource].releases:
                return ("resource", resource)
        elif template.type in {"CpAsyncWaitGroup", "WgmmaWaitGroup", "TmaStoreWaitGroup"}:
            kind = (
                "GROUP_COMPLETE"
                if template.type == "CpAsyncWaitGroup"
                else self._field(template, "completion_kind", lane)
                if template.type == "WgmmaWaitGroup"
                else "SOURCE_RELEASED"
                if self._field(template, "wait_mode", lane) == "READ_ONLY"
                else "GLOBAL_COMPLETE"
            )
            if not self._group_wait_ready(template, lane, kind):
                return ("resource", self._resource(template, lane))
        return None

    def _apply_group(self, template: EventTemplate, lane: int, node: int) -> None:
        resource = self._resource(template, lane)
        owner = self._owner(template, lane)
        state = self.groups[resource]
        kind = template.type
        staged = state.staged.setdefault(owner, [])
        if kind in {"CpAsyncIssue", "WgmmaIssue", "TmaStoreIssue"}:
            if kind == "WgmmaIssue":
                shared = {
                    self.region_ids[region]
                    for region in self._field(template, "regions", lane, [])
                    if self.regions[self.region_ids[region]]["address_space"] == "SHARED"
                }
                missing = sorted(shared - state.fences.get(owner, set()))
                if missing:
                    raise self._error(
                        "WGMMA_OPERAND_FENCE_MISSING",
                        "WGMMA issue lacks a covering operand fence",
                        template,
                        {"missing_regions": [self.regions[index]["id"] for index in missing]},
                    )
            staged.append(node)
            return
        if kind == "WgmmaFenceOperand":
            state.fences.setdefault(owner, set()).update(
                self.region_ids[region]
                for region in self._field(template, "regions", lane, [])
            )
            return
        if kind.endswith("CommitGroup"):
            sequence = self._field(template, "group_seq", lane)
            expected = state.next_sequence.get(owner, 0)
            if sequence != expected:
                code = "STALE_COMMIT_GROUP" if sequence < expected else "FUTURE_COMMIT_GROUP"
                raise self._error(code, "commit sequence is not the next group", template, {"expected": expected, "actual": sequence})
            groups = state.groups.setdefault(owner, [])
            endpoints = {
                endpoint_kind: self._endpoint(
                    ("group", resource, owner, sequence, endpoint_kind),
                    f"group:{resource}:{owner}:{sequence}:{endpoint_kind}",
                )
                for endpoint_kind in state.completion_kinds
            }
            group = Group(sequence, node, tuple(staged), endpoints)
            if state.protocol == "TMA_STORE":
                self.graph.add_edge(
                    endpoints["SOURCE_RELEASED"],
                    endpoints["GLOBAL_COMPLETE"],
                    "ENDPOINT_ORDER",
                )
            groups.append(group)
            state.staged[owner] = []
            state.next_sequence[owner] = expected + 1
            ordered_endpoints = tuple(
                endpoints[kind] for kind in state.completion_kinds
            )
            for issue in group.issues:
                if issue in self.cp_copy_endpoints:
                    self.graph.add_edge(self.cp_copy_endpoints[issue], endpoints["GROUP_COMPLETE"], "COPY_TO_GROUP")
                self.graph.add_edge(issue, node, "ISSUE_TO_GROUP")
                issue_template = self._template_for_node(issue)
                if issue_template is not None:
                    self._record_engine_endpoints(
                        issue_template,
                        min(issue_template.executors),
                        ordered_endpoints,
                    )
            self._enqueue_completion((
                "group-endpoint", resource, owner, sequence,
                state.completion_kinds[0],
            ))
            return
        if kind.endswith("WaitGroup"):
            completion_kind = (
                "GROUP_COMPLETE"
                if kind == "CpAsyncWaitGroup"
                else self._field(template, "completion_kind", lane)
                if kind == "WgmmaWaitGroup"
                else "SOURCE_RELEASED"
                if self._field(template, "wait_mode", lane) == "READ_ONLY"
                else "GLOBAL_COMPLETE"
            )
            groups = state.groups.get(owner, [])
            cutoff = max(0, len(groups) - self._field(template, "wait_group_count", lane))
            if set(state.staged.get(owner, ())) - self.cp_tracked:
                raise self._error("WAIT_BEFORE_COMMIT", "group wait reached uncommitted operations", template, {})
            destination_lanes = (
                template.program_instances
                if (
                    template.rendezvous == "ALL_PROGRAM_INSTANCES"
                    or kind == "TmaStoreWaitGroup"
                )
                else (lane,)
            )
            destination_positions = {
                participant: self.graph.occurrence_position[
                    self.graph.occurrence_by_lane_template[
                        (participant, template.index)
                    ]
                ]
                for participant in destination_lanes
            }
            for participant, position in destination_positions.items():
                frontier_key = (owner, completion_kind, participant)
                observed = state.observation_frontiers.get(frontier_key, 0)
                for group in groups[observed:cutoff]:
                    endpoint = group.endpoints[completion_kind]
                    self.graph.add_edge(endpoint, node, "COMPLETION_TO_WAIT")
                    observations = self.completion_observations[endpoint]
                    observations[participant] = min(
                        observations.get(participant, position), position
                    )
                    self.group_wait_observation_advances += 1
                state.observation_frontiers[frontier_key] = max(
                    observed, cutoff
                )

    def _apply_protocol(self, template_id: int, lane: int) -> None:
        template = self.module.templates[template_id]
        node = self._source_node(template_id, lane)
        kind = template.type
        if kind in MEMORY_EVENTS or kind == "ProgramPoint":
            return
        if kind == "MBarrierInit":
            resource = self._resource(template, lane)
            state = self.mbarriers[resource]
            if self._field(template, "layout", lane, "V0") != "V0":
                raise self._error("UNSUPPORTED_MBARRIER_LAYOUT_V1", "only mbarrier layout V0 is supported", template, state)
            if state.lifecycle == "LIVE":
                raise self._error("REINIT_LIVE", "live mbarrier cannot be reinitialized", template, state)
            epoch = self._field(template, "lifecycle_epoch", lane)
            if epoch != state.lifecycle_epoch + 1:
                raise self._error("LIFECYCLE_EPOCH_MISMATCH", "mbarrier lifecycle is not consecutive", template, state)
            count = self._field(template, "count", lane)
            state.lifecycle = "LIVE"
            state.lifecycle_epoch = epoch
            state.generation = 0
            state.base_count = state.pending = count
            state.expected_tx = state.issued_tx = state.completed_tx = 0
            state.pending_tx_completions = 0
            state.started = False
            state.publications.clear()
            state.observed_generation = None
            state.contributors.clear()
            state.phase_sources.clear()
            state.last_phase_contributors = ()
            state.last_phase_transactional = True
        elif kind == "MBarrierInvalidate":
            _resource, state = self._barrier(template, lane)
            if state.started:
                # With every arrival in and every expected byte issued, the
                # phase completes asynchronously; the missing step is the wait.
                if state.pending == 0 and state.issued_tx == state.expected_tx:
                    raise self._error("INVALID_MBARRIER_UNOBSERVED_PHASE", "invalidation requires a successful wait", template, state)
                raise self._error("INVALIDATE_INCOMPLETE", "cannot invalidate an incomplete phase", template, state)
            completed = state.generation - 1
            if (
                completed >= 0
                and state.observed_generation != completed
                and not self._retired_without_wait(state, node)
            ):
                raise self._error("INVALID_MBARRIER_UNOBSERVED_PHASE", "invalidation requires a successful wait", template, state)
            state.lifecycle = "INVALIDATED"
        elif kind == "CpAsyncMBarrierArrive":
            resource, state = self._barrier(template, lane)
            self._require_generation(template, lane, state)
            generation = self._field(template, "generation", lane)
            if generation > 0 and state.observed_generation != generation - 1:
                raise self._error("INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE", "deferred arrive requires observation of the preceding phase", template, state)
            if self._field(template, "predicate", lane, True):
                if state.pending <= state.deferred_arrivals:
                    raise self._error("ARRIVAL_UNDERFLOW", "deferred arrivals exceed the pending count", template, state)
                group_resource = self.resource_ids[self._field(template, "async_resource", lane)]
                issues = tuple(self.cp_issues[(group_resource, self._owner(template, lane))])
                endpoint = self._endpoint(("deferred-arrival", node), f"deferred-arrival:{node}")
                # Deliberately no program-order edge to this endpoint. noinc
                # publishes prior cp.async operations, not generic accesses.
                for issue in issues:
                    self.graph.add_edge(self.cp_copy_endpoints[issue], endpoint, "COPY_TO_DEFERRED_ARRIVAL")
                self.cp_tracked.update(issues)
                state.deferred_arrivals += 1
                state.has_deferred_arrival = state.started = True
                self._enqueue_completion(("deferred-arrival", resource, state.lifecycle_epoch,
                                          generation, endpoint, issues, template.index, lane))
        elif kind in {"MBarrierArrive", "MBarrierArriveExpectTx", "MBarrierExpectTx", "TmaLoadIssue"}:
            resource, state = self._barrier(template, lane)
            target_generation = self._field(template, "generation", lane)
            if (
                kind in {"MBarrierArrive", "MBarrierArriveExpectTx"}
                and self._field(template, "predicate", lane, True)
                and target_generation > 0
                and state.observed_generation != target_generation - 1
            ):
                raise self._error("INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE", "subsequent-phase arrive requires a successful preceding wait", template, state)
            self._require_generation(template, lane, state)
            if kind in {"MBarrierArrive", "MBarrierArriveExpectTx"} and self._field(template, "predicate", lane, True):
                count = self._field(template, "count", lane)
                if count > state.pending - state.deferred_arrivals:
                    raise self._error("ARRIVAL_UNDERFLOW", "mbarrier arrival count underflow", template, state)
                state.pending -= count
                state.started = True
                state.contributors.append(node)
            if kind == "MBarrierExpectTx" or (
                kind == "MBarrierArriveExpectTx"
                and self._field(template, "predicate", lane, True)
            ):
                state.expected_tx += self._field(template, "bytes", lane)
                state.started = True
                state.contributors.append(node)
            if kind == "TmaLoadIssue":
                byte_count = self._field(template, "bytes", lane)
                state.issued_tx += byte_count
                state.pending_tx_completions += 1
                self._enqueue_completion((
                    "tma-transaction", resource, state.lifecycle_epoch,
                    state.generation, byte_count, node,
                ))
                state.started = True
                state.contributors.append(node)
                endpoint = self._endpoint(
                    ("phase", resource, state.lifecycle_epoch, state.generation),
                    f"phase:{resource}:{state.lifecycle_epoch}:{state.generation}",
                )
                self._record_engine_endpoints(template, lane, (endpoint,))
                transfers = self._field(template, "transfers", lane, [])
                if not transfers:
                    source = self._field(template, "source_region", lane)
                    destination = self._field(template, "destination_region", lane)
                    if source is not None and destination is not None:
                        transfers = [{
                            "source_region": source,
                            "destination_region": destination,
                        }]
                for transfer in transfers:
                    self.effects.extend((
                        Effect(node, lane, "READ", self.region_ids[transfer["source_region"]], endpoint, True, kind),
                        Effect(node, lane, "WRITE", self.region_ids[transfer["destination_region"]], endpoint, True, kind),
                    ))
            if (
                kind in {"MBarrierArrive", "MBarrierArriveExpectTx"}
                and self._field(template, "predicate", lane, True)
                and state.pending == 0
                and state.pending_tx_completions == 0
                and state.completed_tx == state.expected_tx
            ):
                self.immediate_notifications.update(
                    self._publish_mbarrier_phase(resource, state)
                )
        elif kind == "MBarrierWaitParity":
            resource, state = self._barrier(template, lane)
            generation = self._field(template, "generation", lane)
            endpoint = self.endpoint_nodes[("phase", resource, state.lifecycle_epoch, generation)]
            self.graph.add_edge(endpoint, node, "COMPLETION_TO_WAIT")
            destination_positions = {
                participant: self.graph.occurrence_position[
                    self.graph.occurrence_by_lane_template[
                        (participant, template.index)
                    ]
                ]
                for participant in template.program_instances
            }
            self.visibility_cuts.append(VisibilityCut(
                source_positions=dict(state.phase_sources.get(generation, {})),
                destination_positions=destination_positions,
                shared=True,
                global_read=False,
                global_write=False,
            ))
            observations = self.completion_observations[endpoint]
            for participant, position in destination_positions.items():
                observations[participant] = min(
                    observations.get(participant, position), position
                )
            state.observed_generation = generation
        elif kind == "BarrierArrive":
            resource = self._resource(template, lane)
            state = self.collectives[resource]
            invocation = self._field(template, "invocation_id", lane)
            if invocation != state.invocation:
                code = "STALE_BARRIER_INVOCATION" if invocation < state.invocation else "FUTURE_BARRIER_INVOCATION"
                raise self._error(code, "barrier invocation is not current", template, {"invocation": state.invocation})
            count = self._field(template, "count", lane)
            if state.arrived + count > state.capacity:
                raise self._error("BARRIER_ARRIVAL_OVERFLOW", "barrier arrival exceeds capacity", template, {"arrived": state.arrived, "count": count})
            state.arrived += count
            state.arrivals.append(node)
            if state.arrived == state.capacity:
                self._enqueue_completion(("barrier-release", resource, state.invocation))
        elif kind == "BarrierObserve":
            resource = self._resource(template, lane)
            invocation = self._field(template, "invocation_id", lane)
            collective = self.collectives[resource]
            release = collective.releases[invocation]
            self.graph.add_edge(release, node, "BARRIER_RELEASE")
            destination_positions = {
                participant: self.graph.occurrence_position[
                    self.graph.occurrence_by_lane_template[
                        (participant, template.index)
                    ]
                ]
                for participant in template.program_instances
            }
            mask = self.resources[resource]["visibility_mask"]
            if not mask.get("control_only", False):
                self.visibility_cuts.append(VisibilityCut(
                    source_positions=dict(collective.release_sources[invocation]),
                    destination_positions=destination_positions,
                    shared=bool(mask.get("shared")),
                    global_read=bool(mask.get("global_read")),
                    global_write=bool(mask.get("global_write")),
                ))
        elif kind in {
            "CpAsyncIssue", "CpAsyncCommitGroup", "CpAsyncWaitGroup",
            "WgmmaFenceOperand", "WgmmaIssue", "WgmmaCommitGroup", "WgmmaWaitGroup",
            "TmaStoreIssue", "TmaStoreCommitGroup", "TmaStoreWaitGroup",
        }:
            self._apply_group(template, lane, node)
            if kind == "CpAsyncIssue" and self._field(template, "predicate", lane, True):
                self.pending_effects[node].extend((
                    (lane, self.region_ids[self._field(template, "source_region", lane)], "READ", node, kind),
                    (lane, self.region_ids[self._field(template, "destination_region", lane)], "WRITE", node, kind),
                ))
                if self.deferred_cp:
                    endpoint = self._endpoint(("cp-copy", node), f"cp-copy:{node}")
                    self.cp_copy_endpoints[node] = endpoint
                    self.cp_issues[(self._resource(template, lane), self._owner(template, lane))].append(node)
                    for actor, region, mode, begin, issue_kind in self.pending_effects.pop(node):
                        self.effects.append(Effect(begin, actor, mode, region, endpoint, True, issue_kind))
                    self._enqueue_completion(("cp-copy", endpoint))
            elif kind == "WgmmaIssue":
                for region in self._field(template, "regions", lane, []):
                    self.pending_effects[node].append((lane, self.region_ids[region], "READ", node, kind))
                self.pending_effects[node].append((lane, self.region_ids[self._field(template, "destination_region", lane)], "WRITE", node, kind))
            elif kind == "TmaStoreIssue":
                self.pending_effects[node].extend((
                    (lane, self.region_ids[self._field(template, "source_region", lane)], "READ", node, kind),
                    (lane, self.region_ids[self._field(template, "destination_region", lane)], "WRITE", node, kind),
                ))
        elif kind == "ProxyFence":
            regions = tuple(
                self.region_ids[region]
                for region in self._field(template, "regions", lane, [])
            )
            self.proxy_fences.append((
                node,
                self._field(template, "from_domain", lane),
                self._field(template, "to_domain", lane),
                self._field(template, "scope", lane),
                self._proxy_region_coverage(regions),
            ))
        self.executed_transitions += 1

    def _enqueue_completion(self, action: tuple[Any, ...]) -> None:
        if action in self.queued_completions:
            return
        self.completion_queue.append(action)
        self.queued_completions.add(action)
        self.completion_queue_pushes += 1

    def _enqueue_mbarrier_phase_if_ready(
        self, resource: int, state: MBarrierState,
    ) -> None:
        if (
            state.lifecycle == "LIVE"
            and state.started
            and state.pending == 0
            and state.pending_tx_completions == 0
            and state.completed_tx == state.expected_tx
        ):
            self._enqueue_completion((
                "mbarrier-phase", resource,
                state.lifecycle_epoch, state.generation,
            ))

    def _publish_mbarrier_phase(
        self,
        resource: int,
        state: MBarrierState,
    ) -> set[tuple[Any, ...]]:
        generation = state.generation
        endpoint = self._endpoint(
            ("phase", resource, state.lifecycle_epoch, generation),
            f"phase:{resource}:{state.lifecycle_epoch}:{generation}",
        )
        source_positions: dict[int, int] = {}
        for contributor in state.contributors:
            self.graph.add_edge(contributor, endpoint, "PHASE_COMPLETION")
            template_id = self.graph.transition_templates.get(contributor)
            if template_id is not None:
                template = self.module.templates[template_id]
                for participant in template.program_instances:
                    occurrence = self.graph.occurrence_by_lane_template[
                        (participant, template_id)
                    ]
                    source_positions[participant] = max(
                        source_positions.get(participant, -1),
                        self.graph.occurrence_position[occurrence],
                    )
            elif contributor < self.graph.logical_nodes:
                participant = self.graph.occurrence_lane[contributor]
                source_positions[participant] = max(
                    source_positions.get(participant, -1),
                    self.graph.occurrence_position[contributor],
                )
        self.completed.add(endpoint)
        state.publications.add(generation)
        state.phase_sources[generation] = source_positions
        state.last_phase_contributors = tuple(state.contributors)
        state.last_phase_transactional = state.expected_tx > 0 or state.issued_tx > 0 or state.has_deferred_arrival
        state.has_deferred_arrival = False
        state.generation += 1
        state.pending = state.base_count
        state.expected_tx = state.issued_tx = state.completed_tx = 0
        state.pending_tx_completions = 0
        state.started = False
        state.contributors = []
        self.completion_transitions += 1
        return {("resource", resource)}

    def _drain_completions(self) -> set[tuple[Any, ...]]:
        if not self.completion_queue:
            return set()
        action = self.completion_queue.popleft()
        self.queued_completions.remove(action)
        self.completion_queue_pops += 1
        kind = action[0]
        if kind == "cp-copy":
            self.completed.add(action[1])
            self.completion_transitions += 1
            return {("copy-complete", action[1])}
        if kind == "deferred-arrival":
            _, resource, epoch, generation, endpoint, issues, template_id, lane = action
            state = self.mbarriers[resource]
            template = self.module.templates[template_id]
            if state.lifecycle_epoch != epoch or state.generation != generation:
                raise self._error("STALE_GENERATION", "deferred arrival outlived its barrier phase", template, state)
            assert all(self.cp_copy_endpoints[issue] in self.completed for issue in issues)
            state.deferred_arrivals -= 1
            state.pending -= 1
            state.contributors.append(endpoint)
            self.completed.add(endpoint)
            self.completion_transitions += 1
            self._enqueue_mbarrier_phase_if_ready(resource, state)
            return {("completion-progress", resource)}
        if kind == "tma-transaction":
            _, resource, epoch, generation, byte_count, _issue = action
            state = self.mbarriers[resource]
            if state.lifecycle_epoch != epoch or state.generation != generation:
                return {("completion-progress", resource)}
            state.pending_tx_completions -= 1
            state.completed_tx += byte_count
            self.completion_transitions += 1
            self._enqueue_mbarrier_phase_if_ready(resource, state)
            return {("completion-progress", resource)}
        if kind == "mbarrier-phase":
            _, resource, epoch, generation = action
            state = self.mbarriers[resource]
            if state.lifecycle_epoch != epoch or state.generation != generation:
                return {("completion-progress", resource)}
            return self._publish_mbarrier_phase(resource, state)
        if kind == "barrier-release":
            _, resource, invocation = action
            state = self.collectives[resource]
            if state.invocation != invocation or state.arrived != state.capacity:
                return {("completion-progress", resource)}
            release = self._endpoint(
                ("barrier", resource, invocation),
                f"barrier:{resource}:{invocation}",
            )
            source_positions: dict[int, int] = {}
            for arrival in state.arrivals:
                self.graph.add_edge(arrival, release, "BARRIER_ARRIVE")
                template_id = self.graph.transition_templates.get(arrival)
                if template_id is not None:
                    template = self.module.templates[template_id]
                    for lane in template.program_instances:
                        occurrence = self.graph.occurrence_by_lane_template[(lane, template_id)]
                        source_positions[lane] = self.graph.occurrence_position[occurrence]
                elif arrival < self.graph.logical_nodes:
                    source_positions[self.graph.occurrence_lane[arrival]] = self.graph.occurrence_position[arrival]
            self.completed.add(release)
            state.releases[invocation] = release
            state.release_sources[invocation] = source_positions
            state.invocation += 1
            state.arrived = 0
            state.arrivals = []
            self.completion_transitions += 1
            return {("resource", resource)}
        if kind == "group-endpoint":
            _, resource, owner, sequence, endpoint_kind = action
            state = self.groups[resource]
            group = state.groups[owner][sequence]
            endpoint = group.endpoints[endpoint_kind]
            self.graph.add_edge(group.commit, endpoint, "GROUP_TO_COMPLETION")
            for issue in group.issues:
                self.graph.add_edge(issue, endpoint, "ISSUE_TO_COMPLETION")
                for lane, region, mode, begin, issue_kind in self.pending_effects.get(issue, []):
                    end_kind = (
                        "OPERANDS_RELEASED" if issue_kind == "WgmmaIssue" and mode == "READ"
                        else "RESULT_READY" if issue_kind == "WgmmaIssue"
                        else "SOURCE_RELEASED" if issue_kind == "TmaStoreIssue" and mode == "READ"
                        else "GLOBAL_COMPLETE" if issue_kind == "TmaStoreIssue"
                        else "GROUP_COMPLETE"
                    )
                    if end_kind == endpoint_kind:
                        atomic = (issue_kind == "TmaStoreIssue" and mode == "WRITE"
                                  and self._field(self._template_for_node(begin), "reduction", lane, False))
                        self.effects.append(Effect(begin, lane, mode, region, endpoint, True, issue_kind, atomic))
            group.completed.add(endpoint_kind)
            frontier_key = (owner, endpoint_kind)
            frontier = state.completed_frontiers.get(frontier_key, 0)
            groups = state.groups[owner]
            while (
                frontier < len(groups)
                and endpoint_kind in groups[frontier].completed
            ):
                frontier += 1
                self.group_completion_frontier_advances += 1
            state.completed_frontiers[frontier_key] = frontier
            self.completed.add(endpoint)
            self.completion_transitions += 1
            endpoint_index = state.completion_kinds.index(endpoint_kind)
            if endpoint_index + 1 < len(state.completion_kinds):
                self._enqueue_completion((
                    "group-endpoint", resource, owner, sequence,
                    state.completion_kinds[endpoint_index + 1],
                ))
            return {("resource", resource)}
        raise AssertionError(f"unknown completion action: {action!r}")

    def _advance(self, lane: int, template_id: int) -> None:
        slot = self.graph.lane_slot[lane]
        node = self.graph.occurrence_by_lane_template[(lane, template_id)]
        self.crossed.add(node)
        self.crossed_counts[template_id] += 1
        self.frontiers[slot] += 1

    def run(self) -> dict[str, Any]:
        protocol_start = perf_counter()
        if all(
            template.type in MEMORY_EVENTS and not template.dependencies
            for template in self.module.templates
        ):
            self.frontier_visits = len(self.module.lane_trace_template_ids)
            self.protocol_seconds = perf_counter() - protocol_start
            return self._memory_report()
        queue = deque(self.module.lane_ids)
        queued = set(self.module.lane_ids)
        blocked: dict[tuple[Any, ...], set[int]] = defaultdict(set)
        deferred_waits: deque[int] = deque()
        deferred_wait_set: set[int] = set()
        error: ProtocolError | None = None
        while error is None:
            force_deferred_wait = False
            if queue:
                lane = queue.popleft()
                queued.discard(lane)
            elif deferred_waits:
                lane = deferred_waits.popleft()
                deferred_wait_set.discard(lane)
                force_deferred_wait = True
            else:
                try:
                    changed = self._drain_completions()
                except ProtocolError as caught:
                    error = caught
                    break
                if not changed:
                    break
                for key in changed:
                    for waiter in blocked.pop(key, ()):
                        if waiter not in queued:
                            queue.append(waiter)
                            queued.add(waiter)
                continue
            slot = self.graph.lane_slot[lane]
            trace_begin = self.module.lane_trace_offsets[slot]
            trace_end = self.module.lane_trace_offsets[slot + 1]
            if trace_begin + self.frontiers[slot] >= trace_end:
                continue
            self.frontier_visits += 1
            template_id = self.module.lane_trace_template_ids[
                trace_begin + self.frontiers[slot]
            ]
            template = self.module.templates[template_id]
            blockers = self._dependency_blockers(template, lane)
            if blockers:
                for key in blockers:
                    blocked[key].add(lane)
                continue
            try:
                protocol_blocker = self._protocol_blocker(template, lane)
            except ProtocolError as caught:
                error = caught
                break
            if protocol_blocker is not None:
                blocked[protocol_blocker].add(lane)
                continue
            if template.type == "MBarrierWaitParity" and not force_deferred_wait:
                if lane not in deferred_wait_set:
                    deferred_waits.append(lane)
                    deferred_wait_set.add(lane)
                    self.deferred_waits += 1
                continue
            advanced_lanes: list[int] = []
            if template.rendezvous == "ALL_PROGRAM_INSTANCES":
                arrivals = self.collective_arrived_bits.get(template_id, 0)
                bit = 1 << lane
                if not arrivals & bit:
                    arrivals |= bit
                    self.collective_arrived_bits[template_id] = arrivals
                    self.collective_ready_counts[template_id] += 1
                    self.collective_arrivals += 1
                if self.collective_ready_counts[template_id] != len(template.program_instances):
                    continue
                try:
                    self._apply_protocol(template_id, min(template.executors))
                except ProtocolError as caught:
                    error = caught
                    break
                self.once_executed.add(template_id)
                advanced_lanes = list(template.program_instances)
                for participant in advanced_lanes:
                    self._advance(participant, template_id)
                    if participant not in queued:
                        queue.append(participant)
                        queued.add(participant)
                self.collective_arrived_bits.pop(template_id, None)
                self.collective_ready_counts[template_id] = 0
            elif template.transition == "ONCE":
                if lane == min(template.executors) and template_id not in self.once_executed:
                    try:
                        self._apply_protocol(template_id, lane)
                    except ProtocolError as caught:
                        error = caught
                        break
                    self.once_executed.add(template_id)
                self._advance(lane, template_id)
                advanced_lanes = [lane]
                if lane not in queued:
                    queue.append(lane)
                    queued.add(lane)
            else:
                try:
                    self._apply_protocol(template_id, lane)
                except ProtocolError as caught:
                    error = caught
                    break
                self._advance(lane, template_id)
                advanced_lanes = [lane]
                if lane not in queued:
                    queue.append(lane)
                    queued.add(lane)

            notification_keys = {
                *(("same", template_id, advanced) for advanced in advanced_lanes),
                *self.immediate_notifications,
            }
            self.immediate_notifications.clear()
            if self.crossed_counts[template_id] == len(template.program_instances):
                notification_keys.add(("all", template_id))
            if template_id in self.once_executed:
                notification_keys.add(("once", template_id))
            for key in notification_keys:
                for waiter in blocked.pop(key, ()):
                    if waiter not in queued:
                        queue.append(waiter)
                        queued.add(waiter)

        unfinished = [
            lane
            for lane in self.module.lane_ids
            if self.frontiers[self.graph.lane_slot[lane]]
            < (
                self.module.lane_trace_offsets[self.graph.lane_slot[lane] + 1]
                - self.module.lane_trace_offsets[self.graph.lane_slot[lane]]
            )
        ]
        self.protocol_seconds = perf_counter() - protocol_start
        if error is not None:
            return self._protocol_report(error)
        for state in self.mbarriers.values():
            if not state.started:
                continue
            if state.pending == 0 and state.completed_tx != state.expected_tx:
                return self._protocol_report(ProtocolError(
                    "TX_BYTES_MISMATCH",
                    "mbarrier transaction completion bytes do not match the expected byte count",
                    state=state.snapshot(),
                ))
            if state.pending != 0 or state.pending_tx_completions != 0:
                return self._protocol_report(ProtocolError(
                    "STUCK_PROTOCOL_STATE",
                    "kernel terminates with an incomplete mbarrier phase",
                    state=state.snapshot(),
                ))
        if unfinished:
            return self._protocol_report(ProtocolError(
                "STUCK_PROTOCOL_STATE",
                "no enabled lane or asynchronous completion can advance the program",
                state={"lanes": unfinished},
            ))
        for resource, state in self.collectives.items():
            if state.arrived:
                return self._protocol_report(ProtocolError(
                    "MISSING_BARRIER_PARTICIPANT",
                    "kernel terminates with an incomplete barrier invocation",
                    state={
                        "resource": self.resources[resource]["id"],
                        "invocation_id": state.invocation,
                        "arrived": state.arrived,
                        "capacity": state.capacity,
                    },
                ))
        for resource, state in self.groups.items():
            for owner, staged in state.staged.items():
                staged = [issue for issue in staged if issue not in self.cp_tracked]
                if staged:
                    protocol = self.resources[resource]["protocol"]
                    return self._protocol_report(ProtocolError(
                        {"CP_ASYNC": "MISSING_CP_ASYNC_COMMIT", "WGMMA": "MISSING_WGMMA_COMMIT",
                         "TMA_STORE": "MISSING_TMA_STORE_COMMIT"}[protocol],
                        "async operations remain staged without a commit_group",
                        state={"resource": self.resources[resource]["id"], "owner": owner,
                               "staged": list(staged)},
                    ))
        boundary_start = perf_counter()
        self._add_engine_order_edges()
        self._build_proxy_fence_index()
        self.boundary_finalize_seconds = perf_counter() - boundary_start
        return self._memory_report()

    def _add_engine_order_edges(self) -> None:
        for records in self.engine_endpoints.values():
            ordered = sorted(set(records), key=lambda item: item[0])
            previous: int | None = None
            previous_endpoints: tuple[int, ...] | None = None
            for _sequence, endpoints in ordered:
                if endpoints == previous_endpoints:
                    continue
                for endpoint in endpoints:
                    if previous is not None:
                        self.graph.add_edge(previous, endpoint, "ENGINE_ORDER")
                    previous = endpoint
                previous_endpoints = endpoints

    def _region_overlap(self, left: int, right: int) -> bool:
        first = self.regions[left]
        second = self.regions[right]
        return (
            first["allocation_id"] == second["allocation_id"]
            and first["address_space"] == second["address_space"]
            and first.get("owner_cta_rank") == second.get("owner_cta_rank")
            and first["offset"] < second["offset"] + second["extent"]
            and second["offset"] < first["offset"] + first["extent"]
        )

    def _proxy_region_coverage(
        self,
        region_ids: tuple[int, ...],
    ) -> dict[
        tuple[str, int | None, str],
        tuple[tuple[int, ...], tuple[int, ...]],
    ]:
        grouped: dict[tuple[str, int | None, str], list[int]] = defaultdict(list)
        for region_id in region_ids:
            region = self.regions[region_id]
            grouped[(
                region["address_space"],
                region.get("owner_cta_rank"),
                region["allocation_id"],
            )].append(region_id)

        coverage = {}
        for key, identifiers in grouped.items():
            ordered = self._proxy_radix_order(
                identifiers,
                lambda identifier: self.regions[identifier]["offset"],
            )
            merged: list[tuple[int, int]] = []
            for identifier in ordered:
                region = self.regions[identifier]
                start = region["offset"]
                end = start + region["extent"]
                if merged and start <= merged[-1][1]:
                    merged[-1] = (
                        merged[-1][0],
                        max(merged[-1][1], end),
                    )
                else:
                    merged.append((start, end))
            coverage[key] = (
                tuple(start for start, _end in merged),
                tuple(end for _start, end in merged),
            )
        return coverage

    def _proxy_radix_order(self, values: list[Any], key_function: Any) -> list[Any]:
        ordered = list(values)
        for shift in range(0, 65, 8):
            buckets: list[list[Any]] = [[] for _ in range(256)]
            for value in ordered:
                buckets[(key_function(value) >> shift) & 0xFF].append(value)
            ordered = [value for bucket in buckets for value in bucket]
            self.proxy_fence_index_build_visits += 2 * len(ordered)
        return ordered

    def _build_proxy_fence_index(self) -> None:
        if self.proxy_fence_index_built:
            return
        self.proxy_fence_index_built = True
        node_count = self.graph.next_node
        if not self.proxy_fences:
            self.proxy_fence_ancestor_masks = (0,) * node_count
            self.proxy_fence_descendant_masks = (0,) * node_count
            return

        fence_bits_by_node: dict[int, int] = defaultdict(int)
        dependency_fence_bits_by_node: dict[int, int] = defaultdict(int)
        intervals_by_key: dict[
            tuple[str, str, int | None, str],
            list[tuple[int, int, int]],
        ] = defaultdict(list)
        for index, (
            node,
            source_domain,
            target_domain,
            scope,
            coverage,
        ) in enumerate(self.proxy_fences):
            bit = 1 << index
            fence_bits_by_node[node] |= bit
            template = self._template_for_node(node)
            if template is not None:
                for dependency in template.dependencies:
                    source_id = self.module.template_indices[dependency.template]
                    source = self.module.templates[source_id]
                    if dependency.relation == "SAME_INSTANCE":
                        source_nodes = (
                            self.graph.occurrence_by_lane_template[(lane, source_id)]
                            for lane in template.program_instances
                        )
                    elif dependency.relation == "ALL_INSTANCES":
                        source_nodes = (
                            self.graph.occurrence_by_lane_template[(lane, source_id)]
                            for lane in source.program_instances
                        )
                    else:
                        source_nodes = (
                            self._source_node(source_id, min(source.executors)),
                        )
                    for source_node in source_nodes:
                        dependency_fence_bits_by_node[source_node] |= bit
                        self.proxy_fence_index_build_visits += 1
            if source_domain != "GENERIC" or scope != "CTA":
                continue
            for (
                address_space,
                owner_cta_rank,
                allocation_id,
            ), (starts, ends) in coverage.items():
                key = (
                    target_domain,
                    address_space,
                    owner_cta_rank,
                    allocation_id,
                )
                intervals_by_key[key].extend(
                    (start, end, bit)
                    for start, end in zip(starts, ends, strict=True)
                )

        successors = [self.graph.successors(node) for node in range(node_count)]
        indegree = [0] * node_count
        for targets in successors:
            self.proxy_fence_index_build_visits += len(targets)
            for target in targets:
                indegree[target] += 1
        queue = deque(node for node, degree in enumerate(indegree) if degree == 0)
        topological: list[int] = []
        while queue:
            node = queue.popleft()
            topological.append(node)
            for target in successors[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(topological) != node_count:
            raise RuntimeError("numeric PDG contains a cycle")

        ancestors = [fence_bits_by_node.get(node, 0) for node in range(node_count)]
        for node in topological:
            for target in successors[node]:
                ancestors[target] |= ancestors[node]
                self.proxy_fence_index_build_visits += 1
        descendants = [
            fence_bits_by_node.get(node, 0)
            | dependency_fence_bits_by_node.get(node, 0)
            for node in range(node_count)
        ]
        for node in reversed(topological):
            for target in successors[node]:
                descendants[node] |= descendants[target]
                self.proxy_fence_index_build_visits += 1
        self.proxy_fence_ancestor_masks = tuple(ancestors)
        self.proxy_fence_descendant_masks = tuple(descendants)

        targets_by_key: dict[
            tuple[str, str, int | None, str],
            dict[int, None],
        ] = defaultdict(dict)
        domains = {
            "CpAsyncIssue": "ASYNC_COPY",
            "TmaLoadIssue": "TMA",
            "TmaStoreIssue": "TMA",
            "WgmmaIssue": "TENSOR_CORE",
        }
        for effect in self.effects:
            if not effect.asynchronous or effect.mode != "READ":
                continue
            domain = domains.get(effect.kind)
            if domain is None:
                continue
            region = self.regions[effect.region]
            key = (
                domain,
                region["address_space"],
                region.get("owner_cta_rank"),
                region["allocation_id"],
            )
            targets_by_key[key][effect.region] = None

        for key, target_ids in targets_by_key.items():
            intervals = self._proxy_radix_order(
                intervals_by_key.get(key, []),
                lambda interval: interval[0],
            )
            ordered_targets = self._proxy_radix_order(
                list(target_ids),
                lambda region_id: self.regions[region_id]["offset"],
            )
            children: list[list[int]] = [[-1, -1]]
            masks = [0]

            def insert_endpoint(endpoint: int, fence_bit: int) -> None:
                node = 0
                masks[node] |= fence_bit
                for shift in range(64, -1, -1):
                    digit = (endpoint >> shift) & 1
                    child = children[node][digit]
                    if child == -1:
                        child = len(children)
                        children[node][digit] = child
                        children.append([-1, -1])
                        masks.append(0)
                    node = child
                    masks[node] |= fence_bit
                    self.proxy_fence_index_build_visits += 1

            def suffix_mask(endpoint: int) -> int:
                node = 0
                result = 0
                for shift in range(64, -1, -1):
                    if node == -1:
                        break
                    digit = (endpoint >> shift) & 1
                    if digit == 0:
                        greater = children[node][1]
                        if greater != -1:
                            result |= masks[greater]
                    node = children[node][digit]
                    self.proxy_fence_index_build_visits += 1
                if node != -1:
                    result |= masks[node]
                return result

            interval_index = 0
            for region_id in ordered_targets:
                region = self.regions[region_id]
                while (
                    interval_index < len(intervals)
                    and intervals[interval_index][0] <= region["offset"]
                ):
                    _start, end, fence_bit = intervals[interval_index]
                    insert_endpoint(end, fence_bit)
                    interval_index += 1
                self.proxy_region_masks[(key[0], region_id)] = suffix_mask(
                    region["offset"] + region["extent"]
                )

    def _proxy_visible(self, point: Effect, window: Effect) -> bool:
        if point.mode != "WRITE" or window.mode != "READ":
            return False
        domain = {
            "CpAsyncIssue": "ASYNC_COPY",
            "TmaLoadIssue": "TMA",
            "TmaStoreIssue": "TMA",
            "WgmmaIssue": "TENSOR_CORE",
        }.get(window.kind)
        if domain is None:
            return False
        if not self.proxy_fence_index_built:
            self._build_proxy_fence_index()
        self.proxy_fence_index_queries += 1
        coverage = self.proxy_region_masks.get((domain, window.region), 0)
        return bool(
            coverage
            & self.proxy_fence_descendant_masks[point.end]
            & self.proxy_fence_ancestor_masks[window.node]
        )

    def _template_for_node(self, node: int) -> EventTemplate | None:
        if node < self.graph.logical_nodes:
            return self.module.templates[self.graph.occurrence_template[node]]
        template_id = self.graph.transition_templates.get(node)
        return self.module.templates[template_id] if template_id is not None else None

    def _ordered_wgmma_accumulations(self, left: Effect, right: Effect) -> bool:
        if (
            left.kind != "WgmmaIssue"
            or right.kind != "WgmmaIssue"
            or left.mode != "WRITE"
            or right.mode != "WRITE"
            or left.region != right.region
            or self.regions[left.region]["address_space"] != "REGISTER"
        ):
            return False
        first = self._template_for_node(left.node)
        second = self._template_for_node(right.node)
        if first is None or second is None:
            return False
        first_lane = min(first.executors)
        second_lane = min(second.executors)
        return (
            self._field(first, "resource", first_lane)
            == self._field(second, "resource", second_lane)
            and self._field(first, "group_owner", first_lane)
            == self._field(second, "group_owner", second_lane)
            and self._field(first, "engine_id", first_lane)
            == self._field(second, "engine_id", second_lane)
            and self._field(first, "engine_seq", first_lane)
            != self._field(second, "engine_seq", second_lane)
        )

    def _ordered_by_visibility_cut(self, source: Effect, target: Effect) -> bool:
        if source.node >= self.graph.logical_nodes or target.node >= self.graph.logical_nodes:
            return False
        region = self.regions[source.region]
        source_position = self.graph.occurrence_position[source.node]
        target_position = self.graph.occurrence_position[target.node]
        for cut in self.visibility_cuts:
            source_cut = cut.source_positions.get(source.lane)
            target_cut = cut.destination_positions.get(target.lane)
            if (
                source_cut is None
                or target_cut is None
                or source_position >= source_cut
                or target_position <= target_cut
            ):
                continue
            if region["address_space"] == "SHARED" and cut.shared:
                return True
            if (
                region["address_space"] == "GLOBAL"
                and cut.global_write
                and (target.mode != "READ" or cut.global_read)
            ):
                return True
        return False

    def _completion_observed_by_cut(self, window: Effect, point: Effect) -> bool:
        observed_at = self.completion_observations.get(window.end, {}).get(
            point.lane
        )
        if observed_at is None:
            return False
        if point.node < self.graph.logical_nodes:
            point_position = self.graph.occurrence_position[point.node]
        else:
            template_id = self.graph.transition_templates.get(point.node)
            occurrence = (
                self.graph.occurrence_by_lane_template.get(
                    (point.lane, template_id)
                )
                if template_id is not None
                else None
            )
            if occurrence is None:
                return False
            point_position = self.graph.occurrence_position[occurrence]
        return point_position > observed_at

    def _ordered(self, left: Effect, right: Effect) -> bool:
        if left.lane == right.lane and not left.asynchronous and not right.asynchronous:
            return True
        if not left.asynchronous and not right.asynchronous:
            return (
                self._ordered_by_visibility_cut(left, right)
                or self._ordered_by_visibility_cut(right, left)
                or self.graph.reachable(left.end, right.node)
                or self.graph.reachable(right.end, left.node)
            )
        if left.asynchronous and right.asynchronous:
            return (
                self._ordered_wgmma_accumulations(left, right)
                or self._completion_observed_by_cut(left, right)
                or self._completion_observed_by_cut(right, left)
                or self.graph.reachable(left.end, right.node)
                or self.graph.reachable(right.end, left.node)
            )
        window, point = (left, right) if left.asynchronous else (right, left)
        if self._completion_observed_by_cut(window, point):
            return True
        if window.mode == "READ":
            # Non-bulk cp.async reads its source in the generic proxy, so
            # ordinary happens-before orders an earlier write.
            before = self._proxy_visible(point, window) or (
                window.kind == "CpAsyncIssue"
                and self.graph.reachable(point.end, window.node)
            )
        else:
            before = self.graph.reachable(point.end, window.node)
        return before or self.graph.reachable(window.end, point.node)

    def _radix_order(
        self,
        values: list[Any],
        key_functions: tuple[Any, ...],
    ) -> list[Any]:
        ordered = list(values)
        for key_function in reversed(key_functions):
            for shift in range(0, 65, 8):
                counts = [0] * 256
                for value in ordered:
                    counts[(key_function(value) >> shift) & 0xFF] += 1
                positions = [0] * 256
                total = 0
                for index, count in enumerate(counts):
                    positions[index] = total
                    total += count
                next_order: list[Any] = [None] * len(ordered)
                for value in ordered:
                    digit = (key_function(value) >> shift) & 0xFF
                    next_order[positions[digit]] = value
                    positions[digit] += 1
                self.race_ordering_effect_visits += 2 * len(ordered)
                ordered = next_order
        return ordered

    def _candidate_conflicts(self, effects: list[Effect]):
        by_allocation: dict[tuple[Any, ...], list[Effect]] = defaultdict(list)
        for effect in effects:
            region = self.regions[effect.region]
            by_allocation[(
                region["address_space"],
                region.get("owner_cta_rank"),
                region["allocation_id"],
            )].append(effect)
        for allocation, allocation_effects in by_allocation.items():
            if all(effect.mode == "READ" for effect in allocation_effects):
                continue
            if (
                len({effect.lane for effect in allocation_effects}) == 1
                and all(not effect.asynchronous for effect in allocation_effects)
            ):
                continue
            ordered = self._radix_order(
                allocation_effects,
                (
                    lambda effect: self.regions[effect.region]["offset"],
                    lambda effect: self.regions[effect.region]["extent"],
                    lambda effect: effect.node,
                ),
            )
            indexes = list(range(len(ordered)))
            endings = self._radix_order(
                indexes,
                (
                    lambda index: (
                        self.regions[ordered[index].region]["offset"]
                        + self.regions[ordered[index].region]["extent"]
                    ),
                    lambda index: index,
                ),
            )
            active: dict[int, Effect] = {}
            active_by_lane: dict[int, dict[int, None]] = {}
            active_writes_by_lane: dict[int, dict[int, None]] = {}
            active_async_by_lane: dict[int, dict[int, None]] = {}
            active_async_writes_by_lane: dict[int, dict[int, None]] = {}
            end_index = 0
            for record_index, right in enumerate(ordered):
                start = self.regions[right.region]["offset"]
                while end_index < len(endings):
                    expired_index = endings[end_index]
                    expired = ordered[expired_index]
                    expired_region = self.regions[expired.region]
                    if expired_region["offset"] + expired_region["extent"] > start:
                        break
                    end_index += 1
                    if expired_index not in active:
                        continue
                    del active[expired_index]
                    bucket_sets = [active_by_lane]
                    if expired.mode == "WRITE":
                        bucket_sets.append(active_writes_by_lane)
                    if expired.asynchronous:
                        bucket_sets.append(active_async_by_lane)
                        if expired.mode == "WRITE":
                            bucket_sets.append(active_async_writes_by_lane)
                    for buckets in bucket_sets:
                        bucket = buckets[expired.lane]
                        del bucket[expired_index]
                        if not bucket:
                            del buckets[expired.lane]
                    self.race_sweep_expirations += 1
                candidate_buckets = (
                    active_writes_by_lane if right.mode == "READ"
                    else active_by_lane
                )
                same_lane_buckets = (
                    candidate_buckets if right.asynchronous
                    else active_async_writes_by_lane if right.mode == "READ"
                    else active_async_by_lane
                )
                for candidate_index in same_lane_buckets.get(right.lane, {}):
                    self.race_sweep_candidate_visits += 1
                    if not _atomic_pair(active[candidate_index], right):
                        yield active[candidate_index], right, allocation
                for candidate_lane, bucket in candidate_buckets.items():
                    if candidate_lane == right.lane:
                        continue
                    for candidate_index in bucket:
                        self.race_sweep_candidate_visits += 1
                        if not _atomic_pair(active[candidate_index], right):
                            yield active[candidate_index], right, allocation
                active[record_index] = right
                bucket_sets = [active_by_lane]
                if right.mode == "WRITE":
                    bucket_sets.append(active_writes_by_lane)
                if right.asynchronous:
                    bucket_sets.append(active_async_by_lane)
                    if right.mode == "WRITE":
                        bucket_sets.append(active_async_writes_by_lane)
                for buckets in bucket_sets:
                    buckets.setdefault(right.lane, {})[record_index] = None
                self.race_sweep_insertions += 1

    def _declared_conflict_count(self) -> int:
        race_start = perf_counter()
        exact_pairs = sum(
            1 for _left, _right, _allocation
            in self._candidate_conflicts(self._declared_effects())
        )
        self.race_seconds = perf_counter() - race_start
        return exact_pairs

    def _conflict_witness(
        self,
        left: Effect,
        right: Effect,
        allocation: tuple[Any, ...],
    ) -> dict[str, Any]:
        return {
            "category": "UNORDERED_CONFLICT",
            "events": sorted({
                self._event_identifier(left.node),
                self._event_identifier(right.node),
            }),
            "effects": [self._effect_witness(left), self._effect_witness(right)],
            "allocation": allocation[2],
            "overlap": {
                "offset": max(
                    self.regions[left.region]["offset"],
                    self.regions[right.region]["offset"],
                ),
                "extent": min(
                    self.regions[left.region]["offset"]
                    + self.regions[left.region]["extent"],
                    self.regions[right.region]["offset"]
                    + self.regions[right.region]["extent"],
                ) - max(
                    self.regions[left.region]["offset"],
                    self.regions[right.region]["offset"],
                ),
            },
            "missing_path": (
                "neither program order nor a typed visibility boundary "
                "orders the conflict"
            ),
        }

    def _wgmma_result_not_ready(self, left: Effect, right: Effect) -> bool:
        if left.asynchronous == right.asynchronous:
            return False
        window, point = (left, right) if left.asynchronous else (right, left)
        return (
            window.kind == "WgmmaIssue"
            and window.mode == "WRITE"
            and point.mode == "READ"
            and self.regions[window.region]["address_space"] == "REGISTER"
            and self.graph.label(window.end).endswith(":RESULT_READY")
        )


    def _memory_report(self) -> dict[str, Any]:
        race_start = perf_counter()
        exact_pairs = 0
        examined_pairs = 0
        first_hazard: tuple[Effect, Effect, tuple[Any, ...]] | None = None
        for left, right, allocation in self._candidate_conflicts(self.effects):
            exact_pairs += 1
            if first_hazard is not None:
                continue
            examined_pairs += 1
            if not self._ordered(left, right):
                first_hazard = (left, right, allocation)
        self.race_seconds = perf_counter() - race_start
        if first_hazard is not None:
            left, right, allocation = first_hazard
            data_witness = self._conflict_witness(left, right, allocation)
            if self._wgmma_result_not_ready(left, right):
                sync_witness = {
                    **data_witness,
                    "category": "WGMMA_RESULT_NOT_READY",
                    "endpoint": "RESULT_READY",
                    "missing_path": (
                        "the accumulator is read before a matching WGMMA "
                        "RESULT_READY observation"
                    ),
                }
                return self._report(
                    "SYNC_HAZARD",
                    "WGMMA_RESULT_NOT_READY",
                    "WGMMA accumulator is read before RESULT_READY",
                    witness=sync_witness,
                    all_findings=[
                        {
                            "class": "SYNC_HAZARD",
                            "code": "WGMMA_RESULT_NOT_READY",
                            "endpoint": "RESULT_READY",
                            "witness": sync_witness,
                        },
                        {
                            "class": "DATA_HAZARD",
                            "code": "UNORDERED_CONFLICT",
                            "witness": data_witness,
                        },
                    ],
                    race_pairs=exact_pairs,
                    examined_pairs=examined_pairs,
                )
            return self._report(
                "DATA_HAZARD",
                "UNORDERED_CONFLICT",
                "a conflicting memory access pair lacks supported ordering",
                witness=data_witness,
                race_pairs=exact_pairs,
                examined_pairs=examined_pairs,
            )
        return self._report(
            "SAFE",
            None,
            "all supported protocol and data checks passed",
            race_pairs=exact_pairs,
            examined_pairs=examined_pairs,
        )

    def _event_identifier(self, node: int) -> str:
        if node < self.graph.logical_nodes:
            lane = self.graph.occurrence_lane[node]
            template = self.module.templates[self.graph.occurrence_template[node]]
            return template.logical_id(lane)
        for template_id, transition in self.graph.transition_nodes.items():
            if transition == node:
                return stable_collective_event_id(self.module.templates[template_id].id)
        return self.graph.label(node)

    def _effect_witness(self, effect: Effect) -> dict[str, Any]:
        event = self._event_identifier(effect.node)
        template = self.module.templates[
            self.graph.occurrence_template[effect.node]
        ] if effect.node < self.graph.logical_nodes else None
        actor = stable_stream_instance_id(
            template.stream if template is not None else self.module.stream_classes[0].id,
            effect.lane,
        )
        region = self.regions[effect.region]["id"]
        return {
            "id": f"effect:{event}:{effect.mode.lower()}:{region}",
            "actor": actor,
            "event": event,
            "lane": effect.lane,
            "mode": effect.mode,
            "region": region,
            "window": [event, self.graph.label(effect.end)],
        }

    def _protocol_report(self, error: ProtocolError) -> dict[str, Any]:
        diagnostic = error.diagnostic()
        race_pairs = self._declared_conflict_count()
        return self._report(
            "SYNC_HAZARD",
            diagnostic["code"],
            diagnostic["message"],
            witness={
                "category": diagnostic["code"],
                "event": diagnostic.get("event"),
                "protocol_state": diagnostic.get("protocol_state", {}),
                "reachable_frontier": {
                    str(lane): self.frontiers[self.graph.lane_slot[lane]]
                    for lane in self.module.lane_ids
                },
            },
            race_pairs=race_pairs,
            examined_pairs=0,
            race_scan_scope="DECLARED_ACCESS_CONFLICT_ENUMERATION",
            race_ordering_performed=False,
        )

    def _report(
        self,
        verdict: str,
        code: str | None,
        summary: str,
        *,
        witness: dict[str, Any] | None = None,
        all_findings: list[dict[str, Any]] | None = None,
        race_pairs: int = 0,
        examined_pairs: int | None = None,
        race_scan_scope: str = "DYNAMIC_EFFECT_ORDERING",
        race_ordering_performed: bool = True,
    ) -> dict[str, Any]:
        diagnostics = [] if code is None else [{"code": code, "message": summary}]
        if examined_pairs is None:
            examined_pairs = race_pairs
        statistics = {
            "representation": "parametric-protocol",
            "actors": len(self.module.lane_ids),
            "events": len(self.module.lane_trace_template_ids),
            "resources": len(self.resources),
            "regions": len(self.regions),
            "logical_program_streams": len(self.module.lane_ids),
            "event_templates": len(self.module.templates),
            "logical_events": len(self.module.lane_trace_template_ids),
            "materialized_event_objects": 0,
            "materialized_artifact_events": 0,
            "frontier_visits": self.frontier_visits,
            "full_enabled_scans": 0,
            "state_clones": 0,
            "replayed_prefix_steps": 0,
            "protocol_transitions": self.executed_transitions,
            "collective_arrivals": self.collective_arrivals,
            "deferred_waits": self.deferred_waits,
            "completion_transitions": self.completion_transitions,
            "completion_queue_pushes": self.completion_queue_pushes,
            "completion_queue_pops": self.completion_queue_pops,
            "group_wait_readiness_checks": self.group_wait_readiness_checks,
            "group_completion_frontier_advances": (
                self.group_completion_frontier_advances
            ),
            "group_wait_observation_advances": (
                self.group_wait_observation_advances
            ),
            "implicit_pdg_nodes": self.graph.logical_nodes,
            "pdg_resident_program_nodes": 0,
            "pdg_implicit_program_edges": max(
                0, self.graph.logical_nodes - len(self.module.lane_ids)
            ),
            "pdg_implicit_boundary_edges": sum(
                kind.startswith("RENDEZVOUS")
                for kind in self.graph.edge_kinds.values()
            ),
            "explicit_boundary_edges": len(self.graph.edge_kinds),
            "reachability_queries": self.graph.reachability_queries,
            "reachability_edge_visits": self.graph.reachability_edge_visits,
            "reachability_cache_entries": (
                len(self.graph._synthetic_indices or {})
            ),
            "reachability_cache_hits": self.graph.reachability_cache_hits,
            "race_candidate_pairs": race_pairs,
            "exact_conflicting_access_pairs": race_pairs,
            "pairs_examined_before_termination": examined_pairs,
            "exact_r_complete": True,
            "race_scan_scope": race_scan_scope,
            "race_ordering_performed": race_ordering_performed,
            "race_ordering_effect_visits": self.race_ordering_effect_visits,
            "race_sweep_insertions": self.race_sweep_insertions,
            "race_sweep_expirations": self.race_sweep_expirations,
            "race_sweep_candidate_visits": self.race_sweep_candidate_visits,
            "proxy_fence_index_build_visits": self.proxy_fence_index_build_visits,
            "proxy_fence_index_queries": self.proxy_fence_index_queries,
            "protocol_schedule_s": self.protocol_seconds,
            "boundary_finalize_s": self.boundary_finalize_seconds,
            "race_scan_s": self.race_seconds,
            "engine": "maximal",
        }
        result: dict[str, Any] = {
            "verdict": verdict,
            "checker_version": "0.3.0",
            "summary": summary,
            "diagnostics": diagnostics,
            "statistics": statistics,
            "artifacts": {
                "pdg": self.graph.artifact(),
                "boundaries": [],
                "parametric_program": {
                    "logical_streams": len(self.module.lane_ids),
                    "event_templates": len(self.module.templates),
                    "logical_events": len(self.module.lane_trace_template_ids),
                    "logical_boundaries": (
                        len(self.graph.transition_nodes) + len(self.endpoint_nodes)
                    ),
                    "materialized_boundaries": 0,
                },
            },
        }
        if witness is not None:
            result["witness"] = witness
        if all_findings is not None:
            result["all_findings"] = all_findings
        if verdict == "SAFE":
            result["manifest"] = {
                "scope": "sm90/sm90a selected-single-CTA unified lane-parameterized core",
                "assumptions": dict(sorted(self.module.raw["assumptions"].items())),
                "producer": self.module.raw["producer"],
                "frontend_provenance": self.module.raw["frontend_provenance"],
                "complexity": {
                    "protocol_and_implicit_pdg": "O(TN)",
                    "memory_conflicts": "O(TN + R)",
                },
            }
        return result


def check_parametric_module(module: ParametricModule) -> dict[str, Any]:
    total_start = perf_counter()
    checker = DirectParametricChecker(module)
    setup_seconds = perf_counter() - total_start
    report = checker.run()
    statistics = report["statistics"]
    statistics["initial_pdg_setup_s"] = setup_seconds
    statistics["implicit_pdg_build_s"] = (
        setup_seconds
        + statistics["protocol_schedule_s"]
        + statistics["boundary_finalize_s"]
    )
    statistics["checker_total_s"] = perf_counter() - total_start
    return report
