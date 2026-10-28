from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EffectExpectation:
    id: str
    actor: str
    mode: str
    region: str
    window: tuple[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor": self.actor,
            "mode": self.mode,
            "region": self.region,
            "window": list(self.window),
        }


@dataclass(frozen=True)
class EdgeExpectation:
    source: str
    target: str
    kind: str

    def as_dict(self) -> dict[str, str]:
        return {"from": self.source, "to": self.target, "kind": self.kind}


@dataclass(frozen=True)
class RuleCase:
    label: str
    pair_id: str
    family: str
    scenario_id: str
    variant: int
    rule_ids: tuple[str, ...]
    polarity: str
    expected_stage: str
    expected_verdict: str | None
    expected_diagnostic: str | None
    diagnostic_events: tuple[str, ...]
    expected_effects: tuple[EffectExpectation, ...]
    protocol_state: dict[str, Any]
    required_pdg_edges: tuple[EdgeExpectation, ...]
    forbidden_pdg_edges: tuple[EdgeExpectation, ...]
    semantic_dimensions: tuple[str, ...]
    repair: str | None
    possible_bad_execution: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def rule_id(self) -> str:
        # Suite 2 only consumes the legacy single-rule cases. Keep its public
        # compatibility accessor while Suite 1 records all applicable rules.
        return self.rule_ids[0]


def _lane(identifier: str, participant: int) -> dict[str, Any]:
    return {"id": identifier, "kind": "LANE", "participants": [participant]}


def _region(
    identifier: str,
    allocation: str,
    *,
    address_space: str = "SHARED",
    offset: int = 0,
    extent: int = 4,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "allocation_id": allocation,
        "address_space": address_space,
        "offset": offset,
        "extent": extent,
    }


def _parity(stream: str, generation: int) -> dict[str, Any]:
    return {
        "derivation_kind": "BARRIER_HISTORY",
        "logical_stream_id": stream,
        "lifecycle_epoch": 0,
        "stream_rank": generation,
        "polling_coalesced": True,
    }


def _visibility(*, shared: bool = True, global_memory: bool = True) -> dict[str, bool]:
    return {
        "shared": shared,
        "global_read": global_memory,
        "global_write": global_memory,
        "tensor_read": True,
        "tensor_write": True,
        "control_only": False,
    }


def _raw(
    label: str,
    *,
    actors: list[dict[str, Any]],
    events: list[dict[str, Any]],
    resources: list[dict[str, Any]] | None = None,
    regions: list[dict[str, Any]] | None = None,
    cta_threads: int | None = None,
) -> dict[str, Any]:
    participants = [
        lane
        for actor in actors
        if actor["kind"] != "ASYNC_ENGINE"
        for lane in actor["participants"]
    ]
    return {
        "schema_version": "pgir-v1",
        "producer": {
            # PGIR v1 currently has no DIRECT_TEST producer enum. The adapter
            # identity and source URI are the authoritative provenance here.
            "dsl": "TILELANG",
            "dsl_version": "direct-pgir-rule-catalog",
            "dsl_commit": "test",
            "backend_commit": "test",
            "adapter_version": "direct-pgir-rule-catalog-v1",
            "command": "pytest tests/suite/pgir_direct/cases",
        },
        "target": {
            "architecture": "sm90",
            "warp_size": 32,
            "cta_threads": cta_threads or max(participants, default=0) + 1,
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
        "control_structure": {"finite_event_graph": True},
        "actors": actors,
        "resources": resources or [],
        "regions": regions or [],
        "events": events,
        "frontend_provenance": {
            "extraction_complete": True,
            "source": f"direct-pgir://{label}",
        },
        "unsupported_constructs": [],
    }


_DATA_REPAIR = (
    "add a supported, region-scoped mbarrier handoff; generic-to-TMA source "
    "publication additionally requires proxy support"
)
_PROTOCOL_REPAIR = "restore lifecycle/generation/count/transaction invariants before this event"


def _phase(resource: str, generation: int) -> str:
    return f"completion:mbarrier-phase:{resource}:l00000000:g{generation:08d}"


def _release(resource: str, invocation: int = 0) -> str:
    return f"completion:barrier-release:{len(resource):08d}:{resource}:i{invocation:08d}"


def _group_completion(protocol: str, owner: str, kind: str, group: int = 0) -> str:
    return (
        f"completion:group:{protocol.lower()}:{len(owner):08d}:{owner}:"
        f"g{group:08d}:{kind.lower()}"
    )


def _effect(
    identifier: str,
    actor: str,
    mode: str,
    region: str,
    begin: str,
    end: str,
) -> EffectExpectation:
    return EffectExpectation(identifier, actor, mode, region, (begin, end))


def _edge(source: str, target: str, kind: str) -> EdgeExpectation:
    return EdgeExpectation(source, target, kind)


def _legacy_expectations(
    family: str,
    variant: int,
    buggy: bool,
) -> dict[str, Any]:
    dimensions = {
        "MEM-RAW": ("memory", "cross-actor", "raw", "aliasing"),
        "MEM-WAW": ("memory", "cross-actor", "waw", "aliasing"),
        "CTA-HANDOFF": ("cta-barrier", "collective", "shared-visibility"),
        "MB-REUSE": ("mbarrier", "phase-reuse", "successful-observation"),
        "TMA-OBSERVE": ("tma-load", "destination-window", "mbarrier-wait"),
        "CP-WAIT": ("cp.async", "destination-window", "wait-frontier"),
        "CP-OVERLAP": ("cp.async", "same-group", "destination-overlap"),
        "TMA-PROXY": ("tma-load", "source-window", "proxy-visibility"),
        "WGMMA-ENDPOINT": ("wgmma", "typed-completion-endpoint"),
        "TMA-STORE-ENDPOINT": ("tma-store", "typed-completion-endpoint"),
    }[family]
    if not buggy:
        required: tuple[EdgeExpectation, ...] = ()
        if family == "CTA-HANDOFF":
            release = _release("sync")
            required = (
                _edge("writer_arrive", release, "BARRIER_ARRIVE"),
                _edge(release, "reader_observe", "BARRIER_RELEASE"),
            )
        elif family == "MB-REUSE":
            required = (
                _edge(_phase("bar", 0), "wait0", "COMPLETION_TO_WAIT"),
                _edge("wait0", "arrive1", "PROGRAM_ORDER"),
            )
        elif family in {"TMA-OBSERVE", "TMA-PROXY"}:
            required = (_edge(_phase("bar", 0), "wait", "COMPLETION_TO_WAIT"),)
        elif family in {"CP-WAIT", "CP-OVERLAP"}:
            completion = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE")
            required = (_edge(completion, "wait", "COMPLETION_TO_WAIT"),)
        elif family == "WGMMA-ENDPOINT":
            kind = "OPERANDS_RELEASED" if variant % 2 == 0 else "RESULT_READY"
            wait = "wait_operand" if variant % 2 == 0 else "wait_result"
            required = (_edge(_group_completion("WGMMA", "wg", kind), wait, "COMPLETION_TO_WAIT"),)
        elif family == "TMA-STORE-ENDPOINT":
            kind = "SOURCE_RELEASED" if variant % 2 == 0 else "GLOBAL_COMPLETE"
            wait = "wait_source" if variant % 2 == 0 else "wait_global"
            required = (_edge(_group_completion("TMA_STORE", "owner", kind), wait, "COMPLETION_TO_WAIT"),)
        return {
            "diagnostic_events": (),
            "expected_effects": (),
            "protocol_state": {},
            "required_pdg_edges": required,
            "forbidden_pdg_edges": (),
            "semantic_dimensions": dimensions,
            "repair": None,
        }

    events: tuple[str, ...]
    effects: tuple[EffectExpectation, ...]
    required = ()
    forbidden = ()
    protocol_state: dict[str, Any] = {"visibility_transfers": []}
    repair = _DATA_REPAIR
    if family == "MEM-RAW":
        events = ("left_write", "right_access")
        effects = (
            _effect("left_write", "left", "WRITE", "left_region", "left_write", "left_write"),
            _effect("right_access", "right", "READ", "right_region", "right_access", "right_access"),
        )
        forbidden = (_edge("left_write", "right_access", "BOUNDARY_ORDER"),)
    elif family == "MEM-WAW":
        events = ("left_write", "right_access")
        effects = (
            _effect("left_write", "left", "WRITE", "left_region", "left_write", "left_write"),
            _effect("right_access", "right", "WRITE", "right_region", "right_access", "right_access"),
        )
        forbidden = (_edge("left_write", "right_access", "BOUNDARY_ORDER"),)
    elif family == "CTA-HANDOFF":
        events = ("read", "write")
        effects = (
            _effect("read", "reader", "READ", "tile", "read", "read"),
            _effect("write", "writer", "WRITE", "tile", "write", "write"),
        )
    elif family == "MB-REUSE":
        events = ("arrive1",)
        effects = ()
        protocol_state = {
            "lifecycle": "LIVE",
            "lifecycle_epoch": 0,
            "logical_generation": 1,
            "last_successfully_observed_generation": None,
            "required_observed_generation": 0,
        }
        repair = _PROTOCOL_REPAIR
        required = (_edge("arrive0", _phase("bar", 0), "BARRIER_ARRIVE"),)
    elif family == "TMA-OBSERVE":
        completion = _phase("bar", 0)
        events = ("issue", "read")
        effects = (
            _effect("effect:issue:transfer:0:destination", "owner", "WRITE", "tile", "issue", completion),
            _effect("read", "owner", "READ", "tile", "read", "read"),
        )
        required = (_edge(completion, "wait", "COMPLETION_TO_WAIT"),)
        forbidden = (_edge(completion, "read", "BOUNDARY_ORDER"),)
    elif family == "CP-WAIT":
        completion = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE")
        events = ("issue", "overwrite")
        effects = (
            _effect("effect:issue:destination", "owner", "WRITE", "tile", "issue", completion),
            _effect("overwrite", "owner", "WRITE", "tile", "overwrite", "overwrite"),
        )
        required = (_edge(completion, "wait", "COMPLETION_TO_WAIT"),)
        forbidden = (_edge(completion, "overwrite", "BOUNDARY_ORDER"),)
    elif family == "CP-OVERLAP":
        completion = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE")
        events = ("issue0", "issue1")
        effects = (
            _effect("effect:issue0:destination", "owner", "WRITE", "tile0", "issue0", completion),
            _effect("effect:issue1:destination", "owner", "WRITE", "tile1", "issue1", completion),
        )
        required = (_edge(completion, "wait", "COMPLETION_TO_WAIT"),)
    elif family == "TMA-PROXY":
        completion = _phase("bar", 0)
        events = ("issue", "source_write")
        effects = (
            _effect("effect:issue:transfer:0:source", "owner", "READ", "source", "issue", completion),
            _effect("source_write", "owner", "WRITE", "source", "source_write", "source_write"),
        )
        required = (_edge(completion, "wait", "COMPLETION_TO_WAIT"),)
        forbidden = (_edge("source_write", "issue", "BOUNDARY_ORDER"),)
    elif family == "WGMMA-ENDPOINT":
        if variant % 2 == 0:
            completion = _group_completion("WGMMA", "wg", "OPERANDS_RELEASED")
            events = ("issue", "overwrite")
            effects = (
                _effect("effect:issue:operand:0", "wg", "READ", "operand", "issue", completion),
                _effect("overwrite", "wg", "WRITE", "operand", "overwrite", "overwrite"),
            )
            forbidden = (_edge(completion, "overwrite", "BOUNDARY_ORDER"),)
        else:
            completion = _group_completion("WGMMA", "wg", "RESULT_READY")
            events = ("issue", "read_accumulator")
            effects = (
                _effect("effect:issue:accumulator", "wg", "WRITE", "accumulator", "issue", completion),
                _effect("read_accumulator", "wg", "READ", "accumulator", "read_accumulator", "read_accumulator"),
            )
            forbidden = (_edge(completion, "read_accumulator", "BOUNDARY_ORDER"),)
            # The missing RESULT_READY observation is reported as a
            # SYNC_HAZARD, so the witness carries the protocol repair.
            repair = _PROTOCOL_REPAIR
        required = (
            _edge(_group_completion("WGMMA", "wg", "OPERANDS_RELEASED"), "wait_operand", "COMPLETION_TO_WAIT"),
            _edge(_group_completion("WGMMA", "wg", "RESULT_READY"), "wait_result", "COMPLETION_TO_WAIT"),
        )
    elif family == "TMA-STORE-ENDPOINT":
        if variant % 2 == 0:
            completion = _group_completion("TMA_STORE", "owner", "SOURCE_RELEASED")
            events = ("issue", "overwrite")
            effects = (
                _effect("effect:issue:source", "owner", "READ", "source", "issue", completion),
                _effect("overwrite", "owner", "WRITE", "source", "overwrite", "overwrite"),
            )
            forbidden = (_edge(completion, "overwrite", "BOUNDARY_ORDER"),)
        else:
            completion = _group_completion("TMA_STORE", "owner", "GLOBAL_COMPLETE")
            events = ("issue", "read_destination")
            effects = (
                _effect("effect:issue:destination", "owner", "WRITE", "destination", "issue", completion),
                _effect("read_destination", "owner", "READ", "destination", "read_destination", "read_destination"),
            )
            forbidden = (_edge(completion, "read_destination", "BOUNDARY_ORDER"),)
        required = (_edge(completion, "wait_source" if variant % 2 == 0 else "wait_global", "COMPLETION_TO_WAIT"),)
    else:  # pragma: no cover - construction-time guard
        raise AssertionError(f"unknown legacy family: {family}")
    return {
        "diagnostic_events": events,
        "expected_effects": effects,
        "protocol_state": protocol_state,
        "required_pdg_edges": required,
        "forbidden_pdg_edges": forbidden,
        "semantic_dimensions": dimensions,
        "repair": repair,
    }


def _case(
    *,
    family: str,
    rule_id: str,
    variant: int,
    buggy: bool,
    raw: dict[str, Any],
    buggy_verdict: str,
    buggy_diagnostic: str,
    bad_execution: tuple[str, ...],
) -> RuleCase:
    polarity = "BUG" if buggy else "CORRECT"
    pair_id = f"{family}-V{variant:02d}"
    label = f"PGIR-{rule_id}-{pair_id}-{polarity}"
    raw["frontend_provenance"]["source"] = f"direct-pgir://{label}"
    expected = _legacy_expectations(family, variant, buggy)
    scenario_id = family
    if family == "WGMMA-ENDPOINT":
        scenario_id += "-OPERAND" if variant % 2 == 0 else "-RESULT"
    elif family == "TMA-STORE-ENDPOINT":
        scenario_id += "-SOURCE" if variant % 2 == 0 else "-GLOBAL"
    return RuleCase(
        label=label,
        pair_id=pair_id,
        family=family,
        scenario_id=scenario_id,
        variant=variant,
        rule_ids=(rule_id,),
        polarity=polarity,
        expected_stage="CHECKER",
        expected_verdict=buggy_verdict if buggy else "SAFE",
        expected_diagnostic=buggy_diagnostic if buggy else None,
        diagnostic_events=expected["diagnostic_events"],
        expected_effects=expected["expected_effects"],
        protocol_state=expected["protocol_state"],
        required_pdg_edges=expected["required_pdg_edges"],
        forbidden_pdg_edges=expected["forbidden_pdg_edges"],
        semantic_dimensions=expected["semantic_dimensions"],
        repair=expected["repair"],
        possible_bad_execution=bad_execution if buggy else (),
        raw=raw,
    )


def _memory_pair(variant: int, *, right_type: str, family: str) -> tuple[RuleCase, RuleCase]:
    extent = (1, 2, 4, 8, 16, 32, 3, 5, 7, 12, 24, 64)[variant]
    actors = [_lane("left", 0), _lane("right", 1)]
    events = [
        {"id": "left_write", "type": "MemWrite", "actor": "left", "seq": 0, "region": "left_region"},
        {"id": "right_access", "type": right_type, "actor": "right", "seq": 0, "region": "right_region"},
    ]
    buggy = _raw(
        "temporary",
        actors=deepcopy(actors),
        regions=[
            _region("left_region", "shared", offset=0, extent=extent),
            _region("right_region", "shared", offset=0, extent=extent),
        ],
        events=deepcopy(events),
    )
    correct = _raw(
        "temporary",
        actors=deepcopy(actors),
        regions=[
            _region("left_region", "shared", offset=0, extent=extent),
            _region("right_region", "shared", offset=extent, extent=extent),
        ],
        events=deepcopy(events),
    )
    bad = ("right:right_access(shared[0:n])", "left:left_write(shared[0:n])")
    return (
        _case(family=family, rule_id="MEM-02", variant=variant, buggy=True, raw=buggy,
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family=family, rule_id="MEM-02", variant=variant, buggy=False, raw=correct,
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _cta_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    extent = variant + 1
    actors = [_lane("writer", 0), _lane("reader", 1)]
    regions = [_region("tile", "shared", extent=extent)]
    buggy_events = [
        {"id": "write", "type": "MemWrite", "actor": "writer", "seq": 0, "region": "tile"},
        {"id": "read", "type": "MemRead", "actor": "reader", "seq": 0, "region": "tile"},
    ]
    correct_events = [
        {"id": "write", "type": "MemWrite", "actor": "writer", "seq": 0, "region": "tile"},
        {"id": "writer_arrive", "type": "BarrierArrive", "actor": "writer", "seq": 1, "resource": "sync", "count": 1, "invocation_id": 0},
        {"id": "writer_observe", "type": "BarrierObserve", "actor": "writer", "seq": 2, "resource": "sync", "invocation_id": 0},
        {"id": "reader_arrive", "type": "BarrierArrive", "actor": "reader", "seq": 0, "resource": "sync", "count": 1, "invocation_id": 0},
        {"id": "reader_observe", "type": "BarrierObserve", "actor": "reader", "seq": 1, "resource": "sync", "invocation_id": 0},
        {"id": "read", "type": "MemRead", "actor": "reader", "seq": 2, "region": "tile"},
    ]
    resource = [{
        "id": "sync", "kind": "cta_barrier", "participants": ["writer", "reader"],
        "capacity": 2, "scope": "CTA", "visibility_mask": _visibility(global_memory=False),
    }]
    bad = ("reader:read(tile)", "writer:write(tile)", "no completed CTA barrier between them")
    return (
        _case(family="CTA-HANDOFF", rule_id="BAR-04", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="CTA-HANDOFF", rule_id="BAR-04", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=resource, regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _mbarrier_reuse_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    count = 1
    actor = [_lane("owner", variant)]
    resource = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    init = {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": count}
    arrive0 = {"id": "arrive0", "type": "MBarrierArrive", "actor": "owner", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": count}
    arrive1 = {"id": "arrive1", "type": "MBarrierArrive", "actor": "owner", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "generation": 1, "count": count}
    wait0 = {"id": "wait0", "type": "MBarrierWaitParity", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": _parity("owner:bar", 0)}
    wait1 = {"id": "wait1", "type": "MBarrierWaitParity", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 1, "raw_parity": 1, "parity_evidence": _parity("owner:bar", 1)}
    buggy_events = [deepcopy(init), deepcopy(arrive0), deepcopy(arrive1), deepcopy(wait0), deepcopy(wait1)]
    correct_events = [deepcopy(init), deepcopy(arrive0), deepcopy(wait0), deepcopy(arrive1), deepcopy(wait1)]
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    bad = ("owner:arrive0 completes g0", "owner:arrive1 starts g1", "owner:wait0 observes g0 too late")
    return (
        _case(family="MB-REUSE", rule_id="MB-08", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actor), resources=deepcopy(resource), events=buggy_events),
              buggy_verdict="SYNC_HAZARD", buggy_diagnostic="INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE", bad_execution=bad),
        _case(family="MB-REUSE", rule_id="MB-08", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actor), resources=deepcopy(resource), events=correct_events),
              buggy_verdict="SYNC_HAZARD", buggy_diagnostic="INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE", bad_execution=bad),
    )


def _tma_observation_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    byte_count = 16 * (variant + 1)
    actors = [_lane("owner", 0)]
    resources = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    regions = [
        _region("source", "global", address_space="GLOBAL", extent=byte_count),
        _region("tile", "shared", extent=byte_count),
    ]
    prefix = [
        {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "expect", "type": "MBarrierExpectTx", "actor": "owner", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": byte_count},
        {"id": "issue", "type": "TmaLoadIssue", "actor": "owner", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "source_region": "source", "destination_region": "tile", "bytes": byte_count},
        {"id": "arrive", "type": "MBarrierArrive", "actor": "owner", "seq": 3, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1},
    ]
    wait = {"id": "wait", "type": "MBarrierWaitParity", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": _parity("owner:bar", 0)}
    read = {"id": "read", "type": "MemRead", "actor": "owner", "seq": 0, "region": "tile"}
    buggy_events = deepcopy(prefix) + [deepcopy(read), deepcopy(wait)]
    correct_events = deepcopy(prefix) + [deepcopy(wait), deepcopy(read)]
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    bad = ("owner:issue TMA write(tile)", "owner:read(tile)", "TMA completion and successful wait may occur later")
    return (
        _case(family="TMA-OBSERVE", rule_id="TMA-04", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="TMA-OBSERVE", rule_id="TMA-04", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _cp_wait_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    byte_count = (4, 8, 16)[variant % 3]
    offset = (variant // 3) * 16
    actors = [_lane("owner", 0)]
    resources = [{"id": "copies", "kind": "async_group", "participants": ["owner"], "capacity": 8, "protocol": "CP_ASYNC", "scope": "CTA"}]
    regions = [
        _region("source", "global", address_space="GLOBAL", offset=offset, extent=byte_count),
        _region("tile", "shared", offset=offset, extent=byte_count),
    ]
    prefix = [
        {"id": "issue", "type": "CpAsyncIssue", "actor": "owner", "seq": 0, "resource": "copies", "source_region": "source", "destination_region": "tile", "bytes": byte_count, "group_owner": "owner"},
        {"id": "commit", "type": "CpAsyncCommitGroup", "actor": "owner", "seq": 1, "resource": "copies", "group_owner": "owner", "group_seq": 0},
    ]
    wait = {"id": "wait", "type": "CpAsyncWaitGroup", "actor": "owner", "seq": 0, "resource": "copies", "group_owner": "owner", "wait_group_count": 0}
    overwrite = {"id": "overwrite", "type": "MemWrite", "actor": "owner", "seq": 0, "region": "tile"}
    buggy_events = deepcopy(prefix) + [deepcopy(overwrite), deepcopy(wait)]
    correct_events = deepcopy(prefix) + [deepcopy(wait), deepcopy(overwrite)]
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    bad = ("owner:cp.async writes tile", "owner:overwrite(tile)", "owner:wait_group occurs too late")
    return (
        _case(family="CP-WAIT", rule_id="CP-04", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="CP-WAIT", rule_id="CP-04", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _cp_overlap_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    byte_count = (4, 8, 16)[variant % 3]
    base = (variant // 3) * 32
    actors = [_lane("owner", 0)]
    resources = [{"id": "copies", "kind": "async_group", "participants": ["owner"], "capacity": 8, "protocol": "CP_ASYNC", "scope": "CTA"}]
    events = [
        {"id": "issue0", "type": "CpAsyncIssue", "actor": "owner", "seq": 0, "resource": "copies", "source_region": "source0", "destination_region": "tile0", "bytes": byte_count, "group_owner": "owner"},
        {"id": "issue1", "type": "CpAsyncIssue", "actor": "owner", "seq": 1, "resource": "copies", "source_region": "source1", "destination_region": "tile1", "bytes": byte_count, "group_owner": "owner"},
        {"id": "commit", "type": "CpAsyncCommitGroup", "actor": "owner", "seq": 2, "resource": "copies", "group_owner": "owner", "group_seq": 0},
        {"id": "wait", "type": "CpAsyncWaitGroup", "actor": "owner", "seq": 3, "resource": "copies", "group_owner": "owner", "wait_group_count": 0},
    ]
    common_regions = [
        _region("source0", "global", address_space="GLOBAL", offset=base, extent=byte_count),
        _region("source1", "global", address_space="GLOBAL", offset=base + byte_count, extent=byte_count),
        _region("tile0", "shared", offset=base, extent=byte_count),
    ]
    buggy_regions = common_regions + [_region("tile1", "shared", offset=base, extent=byte_count)]
    correct_regions = common_regions + [_region("tile1", "shared", offset=base + byte_count, extent=byte_count)]
    bad = ("owner:issue0 writes tile", "owner:issue1 writes same tile in the same cp.async group")
    return (
        _case(family="CP-OVERLAP", rule_id="CP-03", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=buggy_regions, events=deepcopy(events)),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="CP-OVERLAP", rule_id="CP-03", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=correct_regions, events=deepcopy(events)),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _tma_proxy_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    byte_count = 16 * (variant + 1)
    actors = [_lane("owner", 0)]
    resources = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    regions = [
        _region("source", "global", address_space="GLOBAL", extent=byte_count),
        _region("tile", "shared", extent=byte_count),
    ]
    base = [
        {"id": "source_write", "type": "MemWrite", "actor": "owner", "seq": 0, "region": "source"},
        {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "expect", "type": "MBarrierExpectTx", "actor": "owner", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": byte_count},
    ]
    fence = {"id": "proxy", "type": "ProxyFence", "actor": "owner", "seq": 0, "from_domain": "GENERIC", "to_domain": "TMA", "scope": "CTA", "regions": ["source"], "visibility_mask": _visibility()}
    suffix = [
        {"id": "issue", "type": "TmaLoadIssue", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "source_region": "source", "destination_region": "tile", "bytes": byte_count},
        {"id": "arrive", "type": "MBarrierArrive", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1},
        {"id": "wait", "type": "MBarrierWaitParity", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": _parity("owner:bar", 0)},
    ]
    buggy_events = deepcopy(base) + deepcopy(suffix)
    correct_events = deepcopy(base) + [deepcopy(fence)] + deepcopy(suffix)
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    bad = ("owner:generic write(source)", "owner:TMA async-proxy read(source)", "no generic-to-TMA proxy fence")
    return (
        _case(family="TMA-PROXY", rule_id="TMA-06", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="TMA-PROXY", rule_id="TMA-06", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


def _wgmma_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    extent = 16 * (variant + 1)
    actors = [{"id": "wg", "kind": "WARPGROUP", "participants": list(range(128))}]
    resources = [{"id": "mma", "kind": "async_group", "participants": ["wg"], "capacity": 8, "protocol": "WGMMA", "scope": "CTA"}]
    regions = [
        _region("operand", "shared", extent=extent),
        _region("accumulator", "register", address_space="REGISTER", extent=extent),
    ]
    prefix = [
        {"id": "fence", "type": "WgmmaFenceOperand", "actor": "wg", "seq": 0, "resource": "mma", "group_owner": "wg", "regions": ["operand"]},
        {"id": "issue", "type": "WgmmaIssue", "actor": "wg", "seq": 1, "resource": "mma", "group_owner": "wg", "regions": ["operand"], "destination_region": "accumulator"},
        {"id": "commit", "type": "WgmmaCommitGroup", "actor": "wg", "seq": 2, "resource": "mma", "group_owner": "wg", "group_seq": 0},
    ]
    wait_operand = {"id": "wait_operand", "type": "WgmmaWaitGroup", "actor": "wg", "seq": 0, "resource": "mma", "group_owner": "wg", "wait_group_count": 0, "completion_kind": "OPERANDS_RELEASED"}
    wait_result = {"id": "wait_result", "type": "WgmmaWaitGroup", "actor": "wg", "seq": 0, "resource": "mma", "group_owner": "wg", "wait_group_count": 0, "completion_kind": "RESULT_READY"}
    if variant % 2 == 0:
        access = {"id": "overwrite", "type": "MemWrite", "actor": "wg", "seq": 0, "region": "operand"}
        buggy_events = deepcopy(prefix) + [deepcopy(access), deepcopy(wait_operand), deepcopy(wait_result)]
        correct_events = deepcopy(prefix) + [deepcopy(wait_operand), deepcopy(access), deepcopy(wait_result)]
        rule = "WG-08"
        bad = ("wg:WGMMA async read(operand)", "wg:overwrite(operand)", "operand release occurs later")
    else:
        access = {"id": "read_accumulator", "type": "MemRead", "actor": "wg", "seq": 0, "region": "accumulator"}
        buggy_events = deepcopy(prefix) + [deepcopy(wait_operand), deepcopy(access), deepcopy(wait_result)]
        correct_events = deepcopy(prefix) + [deepcopy(wait_operand), deepcopy(wait_result), deepcopy(access)]
        rule = "WG-07"
        bad = ("wg:WGMMA writes accumulator asynchronously", "wg:read(accumulator)", "result-ready wait occurs later")
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    buggy_verdict = "DATA_HAZARD" if variant % 2 == 0 else "SYNC_HAZARD"
    buggy_diagnostic = (
        "UNORDERED_CONFLICT" if variant % 2 == 0 else "WGMMA_RESULT_NOT_READY"
    )
    return (
        _case(family="WGMMA-ENDPOINT", rule_id=rule, variant=variant, buggy=True,
              raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict=buggy_verdict, buggy_diagnostic=buggy_diagnostic, bad_execution=bad),
        _case(family="WGMMA-ENDPOINT", rule_id=rule, variant=variant, buggy=False,
              raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict=buggy_verdict, buggy_diagnostic=buggy_diagnostic, bad_execution=bad),
    )


def _tma_store_pair(variant: int) -> tuple[RuleCase, RuleCase]:
    extent = 16 * (variant + 1)
    actors = [_lane("owner", 0)]
    resources = [{"id": "stores", "kind": "async_group", "participants": ["owner"], "capacity": 8, "protocol": "TMA_STORE", "scope": "CTA"}]
    regions = [
        _region("source", "shared", extent=extent),
        _region("destination", "global", address_space="GLOBAL", extent=extent),
    ]
    prefix = [
        {"id": "issue", "type": "TmaStoreIssue", "actor": "owner", "seq": 0, "resource": "stores", "group_owner": "owner", "source_region": "source", "destination_region": "destination", "bytes": extent},
        {"id": "commit", "type": "TmaStoreCommitGroup", "actor": "owner", "seq": 1, "resource": "stores", "group_owner": "owner", "group_seq": 0},
    ]
    wait_source = {"id": "wait_source", "type": "TmaStoreWaitGroup", "actor": "owner", "seq": 0, "resource": "stores", "group_owner": "owner", "wait_group_count": 0, "wait_mode": "READ_ONLY"}
    wait_global = {"id": "wait_global", "type": "TmaStoreWaitGroup", "actor": "owner", "seq": 0, "resource": "stores", "group_owner": "owner", "wait_group_count": 0, "wait_mode": "GLOBAL"}
    if variant % 2 == 0:
        access = {"id": "overwrite", "type": "MemWrite", "actor": "owner", "seq": 0, "region": "source"}
        buggy_events = deepcopy(prefix) + [deepcopy(access), deepcopy(wait_source)]
        correct_events = deepcopy(prefix) + [deepcopy(wait_source), deepcopy(access)]
        bad = ("owner:TMA-store reads source asynchronously", "owner:overwrite(source)", "source-release wait occurs later")
    else:
        access = {"id": "read_destination", "type": "MemRead", "actor": "owner", "seq": 0, "region": "destination"}
        buggy_events = deepcopy(prefix) + [deepcopy(wait_source), deepcopy(access), deepcopy(wait_global)]
        correct_events = deepcopy(prefix) + [deepcopy(wait_source), deepcopy(wait_global), deepcopy(access)]
        bad = ("owner:TMA-store writes destination asynchronously", "owner:read(destination)", "global-completion wait occurs later")
    for events in (buggy_events, correct_events):
        for seq, event in enumerate(events):
            event["seq"] = seq
    return (
        _case(family="TMA-STORE-ENDPOINT", rule_id="TMA-05", variant=variant, buggy=True,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="TMA-STORE-ENDPOINT", rule_id="TMA-05", variant=variant, buggy=False,
              raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
    )


CASES: list[RuleCase] = []
for variant in range(12):
    CASES.extend(_memory_pair(variant, right_type="MemRead", family="MEM-RAW"))
    CASES.extend(_memory_pair(variant, right_type="MemWrite", family="MEM-WAW"))
    CASES.extend(_cta_pair(variant))
    CASES.extend(_mbarrier_reuse_pair(variant))
    CASES.extend(_tma_observation_pair(variant))
    CASES.extend(_cp_wait_pair(variant))
    CASES.extend(_cp_overlap_pair(variant))
    CASES.extend(_tma_proxy_pair(variant))
    CASES.extend(_wgmma_pair(variant))
    CASES.extend(_tma_store_pair(variant))


def _named_case(
    *,
    label: str,
    family: str,
    scenario_id: str,
    variant: int,
    rule_ids: tuple[str, ...],
    polarity: str,
    expected_stage: str,
    expected_verdict: str | None,
    expected_diagnostic: str | None,
    diagnostic_events: tuple[str, ...],
    expected_effects: tuple[EffectExpectation, ...],
    protocol_state: dict[str, Any],
    required_pdg_edges: tuple[EdgeExpectation, ...],
    forbidden_pdg_edges: tuple[EdgeExpectation, ...],
    semantic_dimensions: tuple[str, ...],
    repair: str | None,
    possible_bad_execution: tuple[str, ...],
    raw: dict[str, Any],
) -> RuleCase:
    raw["frontend_provenance"]["source"] = f"direct-pgir://{label}"
    return RuleCase(
        label=label,
        pair_id=f"NAMED-{family}-{scenario_id}",
        family=family,
        scenario_id=f"{family}:{scenario_id}",
        variant=variant,
        rule_ids=rule_ids,
        polarity=polarity,
        expected_stage=expected_stage,
        expected_verdict=expected_verdict,
        expected_diagnostic=expected_diagnostic,
        diagnostic_events=diagnostic_events,
        expected_effects=expected_effects,
        protocol_state=protocol_state,
        required_pdg_edges=required_pdg_edges,
        forbidden_pdg_edges=forbidden_pdg_edges,
        semantic_dimensions=semantic_dimensions,
        repair=repair,
        possible_bad_execution=possible_bad_execution,
        raw=raw,
    )


def _legacy_raw(label: str) -> dict[str, Any]:
    return deepcopy(next(case.raw for case in CASES if case.label == label))


def _resequence(raw: dict[str, Any]) -> None:
    for actor in {event["actor"] for event in raw["events"]}:
        actor_events = sorted(
            (event for event in raw["events"] if event["actor"] == actor),
            key=lambda event: (event["seq"], event["id"]),
        )
        for seq, event in enumerate(actor_events):
            event["seq"] = seq


def _append_pair(
    cases: list[RuleCase],
    *,
    family: str,
    scenario_id: str,
    rule_ids: tuple[str, ...],
    dimensions: tuple[str, ...],
    buggy_raw: dict[str, Any],
    correct_raw: dict[str, Any],
    buggy_verdict: str | None,
    buggy_diagnostic: str,
    bad_execution: tuple[str, ...],
    diagnostic_events: tuple[str, ...] = (),
    expected_effects: tuple[EffectExpectation, ...] = (),
    protocol_state: dict[str, Any] | None = None,
    required_bug_edges: tuple[EdgeExpectation, ...] = (),
    forbidden_bug_edges: tuple[EdgeExpectation, ...] = (),
    required_correct_edges: tuple[EdgeExpectation, ...] = (),
    forbidden_correct_edges: tuple[EdgeExpectation, ...] = (),
    buggy_stage: str = "CHECKER",
) -> None:
    primary = rule_ids[0]
    common = {
        "family": family,
        "scenario_id": scenario_id,
        "variant": 0,
        "rule_ids": rule_ids,
        "semantic_dimensions": dimensions,
    }
    cases.append(_named_case(
        label=f"PGIR-{primary}-NAMED-{scenario_id}-BUG",
        polarity="BUG",
        expected_stage=buggy_stage,
        expected_verdict=buggy_verdict,
        expected_diagnostic=buggy_diagnostic,
        diagnostic_events=diagnostic_events,
        expected_effects=expected_effects,
        protocol_state=protocol_state or {},
        required_pdg_edges=required_bug_edges,
        forbidden_pdg_edges=forbidden_bug_edges,
        repair=_PROTOCOL_REPAIR if buggy_verdict == "SYNC_HAZARD" else _DATA_REPAIR if buggy_verdict == "DATA_HAZARD" else None,
        possible_bad_execution=bad_execution,
        raw=buggy_raw,
        **common,
    ))
    cases.append(_named_case(
        label=f"PGIR-{primary}-NAMED-{scenario_id}-CORRECT",
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=required_correct_edges,
        forbidden_pdg_edges=forbidden_correct_edges,
        repair=None,
        possible_bad_execution=(),
        raw=correct_raw,
        **common,
    ))


def _add_named_cases() -> None:
    named: list[RuleCase] = []

    raw = _legacy_raw("PGIR-MEM-02-MEM-RAW-V02-BUG")
    raw["events"][0]["type"] = "MemRead"
    raw["events"][1]["type"] = "MemWrite"
    raw["regions"][1]["offset"] = 2
    named.append(_named_case(
        label="PGIR-MEM-02-NAMED-PARTIAL-WAR-BUG",
        family="MEMORY",
        scenario_id="PARTIAL-WAR",
        variant=0,
        rule_ids=("MEM-02",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="DATA_HAZARD",
        expected_diagnostic="UNORDERED_CONFLICT",
        diagnostic_events=("left_write", "right_access"),
        expected_effects=(
            _effect("left_write", "left", "READ", "left_region", "left_write", "left_write"),
            _effect("right_access", "right", "WRITE", "right_region", "right_access", "right_access"),
        ),
        protocol_state={"visibility_transfers": []},
        required_pdg_edges=(),
        forbidden_pdg_edges=(_edge("left_write", "right_access", "BOUNDARY_ORDER"),),
        semantic_dimensions=("memory", "cross-actor", "war", "partial-overlap"),
        repair=_DATA_REPAIR,
        possible_bad_execution=("left reads bytes [0,4)", "right writes overlapping bytes [2,6)", "no visibility handoff"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-MEM-02-MEM-RAW-V11-CORRECT")
    named.append(_named_case(
        label="PGIR-MEM-02-NAMED-ADJACENT-DISJOINT-CORRECT",
        family="MEMORY",
        scenario_id="ADJACENT-DISJOINT",
        variant=0,
        rule_ids=("MEM-02",),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("memory", "cross-actor", "exact-alias", "adjacent-regions"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-MEM-02-MEM-RAW-V00-BUG")
    raw["events"][1]["region"] = "missing_region"
    named.append(_named_case(
        label="PGIR-MEM-02-NAMED-DANGLING-REGION-BUG",
        family="MEMORY",
        scenario_id="DANGLING-REGION",
        variant=0,
        rule_ids=("MEM-02",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="DANGLING_REFERENCE",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("memory", "model-validation", "region-reference"),
        repair=None,
        possible_bad_execution=("right access names a region absent from the PGIR region table",),
        raw=raw,
    ))

    release = _release("sync")
    raw = _legacy_raw("PGIR-BAR-04-CTA-HANDOFF-V00-CORRECT")
    mask = raw["resources"][0]["visibility_mask"]
    for key in mask:
        mask[key] = False
    mask["control_only"] = True
    named.append(_named_case(
        label="PGIR-BAR-04-NAMED-CONTROL-ONLY-HANDOFF-BUG",
        family="CTA-BARRIER",
        scenario_id="CONTROL-ONLY-HANDOFF",
        variant=0,
        rule_ids=("BAR-04", "MEM-02"),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="DATA_HAZARD",
        expected_diagnostic="UNORDERED_CONFLICT",
        diagnostic_events=("read", "write"),
        expected_effects=(
            _effect("read", "reader", "READ", "tile", "read", "read"),
            _effect("write", "writer", "WRITE", "tile", "write", "write"),
        ),
        protocol_state={"visibility_transfers": []},
        required_pdg_edges=(
            _edge("writer_arrive", release, "BARRIER_ARRIVE"),
            _edge(release, "reader_observe", "BARRIER_RELEASE"),
        ),
        forbidden_pdg_edges=(_edge("write", "read", "BOUNDARY_ORDER"),),
        semantic_dimensions=("cta-barrier", "control-only", "shared-visibility"),
        repair=_DATA_REPAIR,
        possible_bad_execution=("writer publishes only control", "reader observes release", "shared tile remains unpublished"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-BAR-04-CTA-HANDOFF-V00-CORRECT")
    rename = {"writer": "a_writer", "reader": "z_reader"}
    for actor in raw["actors"]:
        actor["id"] = rename[actor["id"]]
    raw["resources"][0]["participants"] = ["a_writer", "z_reader"]
    for event in raw["events"]:
        event["actor"] = rename[event["actor"]]
    duplicate = deepcopy(next(event for event in raw["events"] if event["id"] == "writer_arrive"))
    duplicate.update({"id": "writer_arrive_again", "seq": 2})
    raw["events"].append(duplicate)
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG",
        family="CTA-BARRIER",
        scenario_id="DUPLICATE-ARRIVAL",
        variant=0,
        rule_ids=("BAR-03",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="DUPLICATE_BARRIER_ARRIVAL",
        diagnostic_events=("writer_arrive_again",),
        expected_effects=(),
        protocol_state={
            "invocation": 0,
            "arrived_count": 1,
            "arrived_actors": ["a_writer"],
            "missing_actors": ["z_reader"],
        },
        required_pdg_edges=(_edge("writer_arrive", "writer_arrive_again", "PROGRAM_ORDER"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cta-barrier", "collective", "single-arrival-per-owner"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("a_writer arrives at invocation 0", "a_writer arrives at invocation 0 again"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-BAR-04-CTA-HANDOFF-V07-CORRECT")
    named.append(_named_case(
        label="PGIR-BAR-04-NAMED-COMPLETE-HANDOFF-CORRECT",
        family="CTA-BARRIER",
        scenario_id="COMPLETE-HANDOFF",
        variant=0,
        rule_ids=("BAR-04",),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge("writer_arrive", release, "BARRIER_ARRIVE"),
            _edge(release, "reader_observe", "BARRIER_RELEASE"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cta-barrier", "collective", "shared-visibility", "complete-observation"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-BAR-04-CTA-HANDOFF-V00-CORRECT")
    raw["resources"][0].update({"participants": ["writer"], "capacity": 1})
    named.append(_named_case(
        label="PGIR-BAR-01-NAMED-INCOMPLETE-CTA-PARTICIPANTS-BUG",
        family="CTA-BARRIER",
        scenario_id="INCOMPLETE-CTA-PARTICIPANTS",
        variant=0,
        rule_ids=("BAR-01",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="CTA_PARTICIPANT_SET_INCOMPLETE",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cta-barrier", "model-validation", "participant-coverage"),
        repair=None,
        possible_bad_execution=("CTA barrier omits the reader participant slot",),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-MB-08-MB-REUSE-V00-CORRECT")
    raw["events"] = [event for event in raw["events"] if event["id"] != "init"]
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-MB-01-NAMED-USE-BEFORE-INIT-BUG",
        family="MBARRIER",
        scenario_id="USE-BEFORE-INIT",
        variant=0,
        rule_ids=("MB-01",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="USE_BEFORE_INIT",
        diagnostic_events=("arrive0",),
        expected_effects=(),
        protocol_state={"lifecycle": "UNINITIALIZED", "lifecycle_epoch": -1, "can_invalidate": False},
        required_pdg_edges=(_edge("arrive0", _phase("bar", 0), "BARRIER_ARRIVE"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("mbarrier", "lifecycle", "initialization"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("arrive0 executes while bar is UNINITIALIZED",),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-04-TMA-OBSERVE-V00-CORRECT")
    next(event for event in raw["events"] if event["id"] == "expect")["bytes"] = 32
    named.append(_named_case(
        label="PGIR-MB-06-NAMED-TX-BYTES-MISMATCH-BUG",
        family="MBARRIER",
        scenario_id="TX-BYTES-MISMATCH",
        variant=0,
        rule_ids=("MB-06", "TMA-03"),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="TX_BYTES_MISMATCH",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={"expected_tx_bytes_current": 32, "issued_tx_bytes_current": 16, "can_invalidate": False},
        required_pdg_edges=(
            _edge("expect", _phase("bar", 0), "EXPECT_TO_PHASE"),
            _edge("issue", _phase("bar", 0), "ISSUE_TO_PHASE"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("mbarrier", "transaction-accounting", "tma-load"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("bar expects 32 bytes", "TMA issues 16 bytes", "phase cannot complete"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-MB-08-MB-REUSE-V09-CORRECT")
    named.append(_named_case(
        label="PGIR-MB-08-NAMED-TWO-GENERATION-HANDOFF-CORRECT",
        family="MBARRIER",
        scenario_id="TWO-GENERATION-HANDOFF",
        variant=0,
        rule_ids=("MB-08",),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge(_phase("bar", 0), "wait0", "COMPLETION_TO_WAIT"),
            _edge("wait0", "arrive1", "PROGRAM_ORDER"),
            _edge(_phase("bar", 1), "wait1", "COMPLETION_TO_WAIT"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("mbarrier", "phase-reuse", "two-generations", "successful-observation"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-MB-08-MB-REUSE-V09-BUG")
    named.append(_named_case(
        label="PGIR-MB-08-NAMED-NEXT-ARRIVE-BEFORE-WAIT-BUG",
        family="MBARRIER",
        scenario_id="NEXT-ARRIVE-BEFORE-WAIT",
        variant=0,
        rule_ids=("MB-08",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE",
        diagnostic_events=("arrive1",),
        expected_effects=(),
        protocol_state={"logical_generation": 1, "last_successfully_observed_generation": None, "required_observed_generation": 0},
        required_pdg_edges=(_edge("arrive0", _phase("bar", 0), "BARRIER_ARRIVE"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("mbarrier", "phase-reuse", "wait-before-next-arrive"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("generation 0 completes", "generation 1 arrive-on executes", "generation 0 has not been successfully waited"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-CP-04-CP-WAIT-V00-CORRECT")
    raw["events"] = [event for event in raw["events"] if event["id"] != "commit"]
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-CP-02-NAMED-WAIT-BEFORE-COMMIT-BUG",
        family="CP-ASYNC",
        scenario_id="WAIT-BEFORE-COMMIT",
        variant=0,
        rule_ids=("CP-02", "CP-04"),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="WAIT_BEFORE_COMMIT",
        diagnostic_events=("wait",),
        expected_effects=(),
        protocol_state={
            "protocol": "CP_ASYNC",
            "owners": {"owner": {"next_group_seq": 0, "staged_operations": ["issue"], "committed_groups": []}},
        },
        required_pdg_edges=(_edge("issue", "wait", "PROGRAM_ORDER"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cp.async", "commit-group", "wait-before-commit"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("cp.async issue remains staged", "wait_group reaches the uncommitted operation"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-CP-04-CP-WAIT-V00-CORRECT")
    next(region for region in raw["regions"] if region["id"] == "tile")["address_space"] = "GLOBAL"
    named.append(_named_case(
        label="PGIR-CP-01-NAMED-INVALID-DESTINATION-SPACE-BUG",
        family="CP-ASYNC",
        scenario_id="INVALID-DESTINATION-SPACE",
        variant=0,
        rule_ids=("CP-01",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="UNSUPPORTED_CP_ASYNC_REGION",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cp.async", "model-validation", "address-space"),
        repair=None,
        possible_bad_execution=("cp.async destination is GLOBAL instead of SHARED",),
        raw=raw,
    ))

    cp_completion = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE")
    raw = _legacy_raw("PGIR-CP-04-CP-WAIT-V08-CORRECT")
    named.append(_named_case(
        label="PGIR-CP-04-NAMED-WAIT-FRONTIER-CORRECT",
        family="CP-ASYNC",
        scenario_id="WAIT-FRONTIER",
        variant=0,
        rule_ids=("CP-04",),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge("commit", "group:cp_async:00000005:owner:g00000000", "COMMIT_TO_GROUP"),
            _edge(cp_completion, "wait", "COMPLETION_TO_WAIT"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("cp.async", "commit-group", "wait-frontier", "destination-reuse"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-04-TMA-OBSERVE-V00-CORRECT")
    issue = next(event for event in raw["events"] if event["id"] == "issue")
    issue["transfers"] = [{
        "source_region": issue.pop("source_region"),
        "destination_region": issue.pop("destination_region"),
        "bytes": 8,
    }]
    named.append(_named_case(
        label="PGIR-TMA-03-NAMED-GROUPED-BYTES-MISMATCH-BUG",
        family="TMA-LOAD",
        scenario_id="GROUPED-BYTES-MISMATCH",
        variant=0,
        rule_ids=("TMA-03",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="TMA_TRANSFER_BYTES_MISMATCH",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("tma-load", "model-validation", "grouped-transfer-bytes"),
        repair=None,
        possible_bad_execution=("TMA issue declares 16 bytes", "grouped transfer accounts for only 8 bytes"),
        raw=raw,
    ))

    tma_completion = _phase("bar", 0)
    raw = _legacy_raw("PGIR-TMA-04-TMA-OBSERVE-V04-BUG")
    named.append(_named_case(
        label="PGIR-TMA-04-NAMED-DESTINATION-BEFORE-WAIT-BUG",
        family="TMA-LOAD",
        scenario_id="DESTINATION-BEFORE-WAIT",
        variant=0,
        rule_ids=("TMA-04",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="DATA_HAZARD",
        expected_diagnostic="UNORDERED_CONFLICT",
        diagnostic_events=("issue", "read"),
        expected_effects=(
            _effect("effect:issue:transfer:0:destination", "owner", "WRITE", "tile", "issue", tma_completion),
            _effect("read", "owner", "READ", "tile", "read", "read"),
        ),
        protocol_state={"visibility_transfers": []},
        required_pdg_edges=(_edge(tma_completion, "wait", "COMPLETION_TO_WAIT"),),
        forbidden_pdg_edges=(_edge(tma_completion, "read", "BOUNDARY_ORDER"),),
        semantic_dimensions=("tma-load", "destination-window", "wait-observation"),
        repair=_DATA_REPAIR,
        possible_bad_execution=("TMA destination write remains in flight", "owner reads tile before successful mbarrier wait"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-06-TMA-PROXY-V03-BUG")
    named.append(_named_case(
        label="PGIR-TMA-06-NAMED-GENERIC-SOURCE-NO-PROXY-BUG",
        family="TMA-LOAD",
        scenario_id="GENERIC-SOURCE-NO-PROXY",
        variant=0,
        rule_ids=("TMA-06",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="DATA_HAZARD",
        expected_diagnostic="UNORDERED_CONFLICT",
        diagnostic_events=("issue", "source_write"),
        expected_effects=(
            _effect("effect:issue:transfer:0:source", "owner", "READ", "source", "issue", tma_completion),
            _effect("source_write", "owner", "WRITE", "source", "source_write", "source_write"),
        ),
        protocol_state={"visibility_transfers": []},
        required_pdg_edges=(_edge(tma_completion, "wait", "COMPLETION_TO_WAIT"),),
        forbidden_pdg_edges=(_edge("source_write", "issue", "BOUNDARY_ORDER"),),
        semantic_dimensions=("tma-load", "source-window", "generic-to-tma-proxy"),
        repair=_DATA_REPAIR,
        possible_bad_execution=("generic source write executes", "TMA reads source without a compatible proxy fence"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-06-TMA-PROXY-V06-CORRECT")
    named.append(_named_case(
        label="PGIR-TMA-06-NAMED-PROXY-AND-WAIT-CORRECT",
        family="TMA-LOAD",
        scenario_id="PROXY-AND-WAIT",
        variant=0,
        rule_ids=("TMA-04", "TMA-06"),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge("issue", tma_completion, "ISSUE_TO_PHASE"),
            _edge(tma_completion, "wait", "COMPLETION_TO_WAIT"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("tma-load", "source-proxy", "destination-wait", "full-handoff"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-WG-08-WGMMA-ENDPOINT-V00-CORRECT")
    raw["events"] = [event for event in raw["events"] if event["id"] != "fence"]
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-WG-03-NAMED-MISSING-OPERAND-FENCE-BUG",
        family="WGMMA",
        scenario_id="MISSING-OPERAND-FENCE",
        variant=0,
        rule_ids=("WG-03",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="WGMMA_OPERAND_FENCE_MISSING",
        diagnostic_events=("issue",),
        expected_effects=(),
        protocol_state={"protocol": "WGMMA", "missing_operand_fence_regions": ["operand"]},
        required_pdg_edges=(_edge("issue", "group:wgmma:00000002:wg:g00000000", "ISSUE_TO_GROUP"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("wgmma", "operand-fence", "shared-operand"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("WGMMA issue consumes shared operand without a covering operand fence",),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-WG-08-WGMMA-ENDPOINT-V00-CORRECT")
    raw["actors"][0]["participants"] = list(range(127))
    named.append(_named_case(
        label="PGIR-WG-01-NAMED-INVALID-WARPGROUP-SHAPE-BUG",
        family="WGMMA",
        scenario_id="INVALID-WARPGROUP-SHAPE",
        variant=0,
        rule_ids=("WG-01",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="INVALID_WGMMA_PARTICIPANTS",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("wgmma", "model-validation", "warpgroup-participants"),
        repair=None,
        possible_bad_execution=("WGMMA owner contains 127 lanes rather than one complete 128-lane warpgroup",),
        raw=raw,
    ))

    wg_operand = _group_completion("WGMMA", "wg", "OPERANDS_RELEASED")
    wg_result = _group_completion("WGMMA", "wg", "RESULT_READY")
    raw = _legacy_raw("PGIR-WG-08-WGMMA-ENDPOINT-V00-CORRECT")
    raw["events"].append({
        "id": "read_accumulator", "type": "MemRead", "actor": "wg",
        "seq": 6, "region": "accumulator",
    })
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-WG-07-WG-08-NAMED-DUAL-ENDPOINT-CORRECT",
        family="WGMMA",
        scenario_id="DUAL-ENDPOINT",
        variant=0,
        rule_ids=("WG-07", "WG-08"),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge(wg_operand, "wait_operand", "COMPLETION_TO_WAIT"),
            _edge(wg_result, "wait_result", "COMPLETION_TO_WAIT"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("wgmma", "operand-release", "result-ready", "typed-endpoints"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-05-TMA-STORE-ENDPOINT-V00-CORRECT")
    raw["events"] = [event for event in raw["events"] if event["id"] != "commit"]
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG",
        family="TMA-STORE",
        scenario_id="STORE-WAIT-BEFORE-COMMIT",
        variant=0,
        rule_ids=("TMA-05",),
        polarity="BUG",
        expected_stage="CHECKER",
        expected_verdict="SYNC_HAZARD",
        expected_diagnostic="WAIT_BEFORE_COMMIT",
        diagnostic_events=("wait_source",),
        expected_effects=(),
        protocol_state={
            "protocol": "TMA_STORE",
            "owners": {"owner": {"next_group_seq": 0, "staged_operations": ["issue"], "committed_groups": []}},
        },
        required_pdg_edges=(_edge("issue", "wait_source", "PROGRAM_ORDER"),),
        forbidden_pdg_edges=(),
        semantic_dimensions=("tma-store", "commit-group", "wait-before-commit"),
        repair=_PROTOCOL_REPAIR,
        possible_bad_execution=("TMA store issue remains staged", "source wait reaches it before commit"),
        raw=raw,
    ))

    raw = _legacy_raw("PGIR-TMA-05-TMA-STORE-ENDPOINT-V00-CORRECT")
    next(region for region in raw["regions"] if region["id"] == "source")["address_space"] = "GLOBAL"
    named.append(_named_case(
        label="PGIR-TMA-05-NAMED-STORE-INVALID-SOURCE-BUG",
        family="TMA-STORE",
        scenario_id="STORE-INVALID-SOURCE",
        variant=0,
        rule_ids=("TMA-05",),
        polarity="BUG",
        expected_stage="MODEL",
        expected_verdict=None,
        expected_diagnostic="UNSUPPORTED_TMA_STORE_REGION",
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(),
        forbidden_pdg_edges=(),
        semantic_dimensions=("tma-store", "model-validation", "source-address-space"),
        repair=None,
        possible_bad_execution=("TMA store source is GLOBAL rather than SHARED",),
        raw=raw,
    ))

    store_source = _group_completion("TMA_STORE", "owner", "SOURCE_RELEASED")
    store_global = _group_completion("TMA_STORE", "owner", "GLOBAL_COMPLETE")
    raw = _legacy_raw("PGIR-TMA-05-TMA-STORE-ENDPOINT-V01-CORRECT")
    raw["events"].append({
        "id": "overwrite", "type": "MemWrite", "actor": "owner",
        "seq": 3, "region": "source",
    })
    _resequence(raw)
    named.append(_named_case(
        label="PGIR-TMA-05-NAMED-STORE-DUAL-ENDPOINT-CORRECT",
        family="TMA-STORE",
        scenario_id="STORE-DUAL-ENDPOINT",
        variant=0,
        rule_ids=("TMA-05",),
        polarity="CORRECT",
        expected_stage="CHECKER",
        expected_verdict="SAFE",
        expected_diagnostic=None,
        diagnostic_events=(),
        expected_effects=(),
        protocol_state={},
        required_pdg_edges=(
            _edge(store_source, "wait_source", "COMPLETION_TO_WAIT"),
            _edge(store_global, "wait_global", "COMPLETION_TO_WAIT"),
            _edge(store_source, store_global, "ENDPOINT_ORDER"),
        ),
        forbidden_pdg_edges=(),
        semantic_dimensions=("tma-store", "source-release", "global-complete", "typed-endpoints"),
        repair=None,
        possible_bad_execution=(),
        raw=raw,
    ))

    # Region geometry: containment must conflict; adjacency repairs it.
    actors = [_lane("left", 0), _lane("right", 1)]
    events = [
        {"id": "left_access", "type": "MemWrite", "actor": "left", "seq": 0, "region": "left_region"},
        {"id": "right_access", "type": "MemWrite", "actor": "right", "seq": 0, "region": "right_region"},
    ]
    buggy = _raw("temporary", actors=deepcopy(actors), events=deepcopy(events), regions=[
        _region("left_region", "shared", offset=0, extent=16),
        _region("right_region", "shared", offset=4, extent=4),
    ])
    correct = _raw("temporary", actors=deepcopy(actors), events=deepcopy(events), regions=[
        _region("left_region", "shared", offset=0, extent=16),
        _region("right_region", "shared", offset=16, extent=4),
    ])
    _append_pair(
        named, family="MEMORY", scenario_id="CONTAINED-WAW", rule_ids=("MEM-01", "MEM-02"),
        dimensions=("memory", "waw", "containment", "touching-boundary"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict="DATA_HAZARD",
        buggy_diagnostic="UNORDERED_CONFLICT", diagnostic_events=("left_access", "right_access"),
        expected_effects=(
            _effect("left_access", "left", "WRITE", "left_region", "left_access", "left_access"),
            _effect("right_access", "right", "WRITE", "right_region", "right_access", "right_access"),
        ), protocol_state={"visibility_transfers": []},
        forbidden_bug_edges=(_edge("left_access", "right_access", "BOUNDARY_ORDER"),),
        bad_execution=("left writes [0,16)", "right writes contained interval [4,8)", "no typed ordering exists"),
    )

    # Allocation identity is part of overlap, not merely equal numeric offsets.
    buggy = _raw("temporary", actors=deepcopy(actors), events=deepcopy(events), regions=[
        _region("left_region", "shared", offset=0, extent=8),
        _region("right_region", "shared", offset=0, extent=8),
    ])
    correct = _raw("temporary", actors=deepcopy(actors), events=deepcopy(events), regions=[
        _region("left_region", "shared-a", offset=0, extent=8),
        _region("right_region", "shared-b", offset=0, extent=8),
    ])
    _append_pair(
        named, family="MEMORY", scenario_id="ALLOCATION-IDENTITY", rule_ids=("MEM-01", "MEM-02"),
        dimensions=("memory", "waw", "allocation-identity", "equal-offset"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict="DATA_HAZARD",
        buggy_diagnostic="UNORDERED_CONFLICT", diagnostic_events=("left_access", "right_access"),
        expected_effects=(
            _effect("left_access", "left", "WRITE", "left_region", "left_access", "left_access"),
            _effect("right_access", "right", "WRITE", "right_region", "right_access", "right_access"),
        ), protocol_state={"visibility_transfers": []},
        bad_execution=("two lanes write equal offsets in the same allocation", "no typed ordering exists"),
    )

    # A collective cannot release until every declared participant has arrived.
    actors = [_lane("a", 0), _lane("b", 1)]
    resource = [{
        "id": "sync", "kind": "cta_barrier", "participants": ["a", "b"],
        "capacity": 2, "scope": "CTA", "visibility_mask": _visibility(global_memory=False),
    }]
    buggy_events = [{"id": "arrive_a", "type": "BarrierArrive", "actor": "a", "seq": 0,
                     "resource": "sync", "count": 1, "invocation_id": 0}]
    correct_events = buggy_events + [{"id": "arrive_b", "type": "BarrierArrive", "actor": "b", "seq": 0,
                                      "resource": "sync", "count": 1, "invocation_id": 0}]
    _append_pair(
        named, family="CTA-BARRIER", scenario_id="MISSING-PARTICIPANT", rule_ids=("BAR-01", "BAR-04"),
        dimensions=("cta-barrier", "participant-completeness", "terminal-stuck"),
        buggy_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), events=buggy_events),
        correct_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), events=correct_events),
        buggy_verdict="SYNC_HAZARD", buggy_diagnostic="MISSING_BARRIER_PARTICIPANT",
        protocol_state={"invocation": 0, "arrived_count": 1, "missing_actors": ["b"]},
        bad_execution=("a arrives at invocation 0", "b never arrives", "collective cannot release"),
    )

    # A transaction-completed phase must be observed before invalidation; only
    # arrival-driven phases whose arrivals precede the invalidation are retired.
    actors = [_lane("owner", 0)]
    resource = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    regions = [
        _region("source", "global", address_space="GLOBAL", extent=16),
        _region("tile", "shared", extent=16),
    ]
    init = {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 0,
            "resource": "bar", "lifecycle_epoch": 0, "count": 1}
    expect = {"id": "expect", "type": "MBarrierExpectTx", "actor": "owner", "seq": 1,
              "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": 16}
    tx_arrive = {"id": "arrive", "type": "MBarrierArrive", "actor": "owner", "seq": 2,
                 "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1}
    issue = {"id": "issue", "type": "TmaLoadIssue", "actor": "owner", "seq": 3,
             "resource": "bar", "lifecycle_epoch": 0, "generation": 0,
             "source_region": "source", "destination_region": "tile", "bytes": 16}
    tx_wait = {"id": "wait", "type": "MBarrierWaitParity", "actor": "owner", "seq": 4,
               "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0,
               "parity_evidence": _parity("owner:bar", 0)}
    invalidate = {"id": "invalidate", "type": "MBarrierInvalidate", "actor": "owner", "seq": 4,
                  "resource": "bar", "lifecycle_epoch": 0}
    prefix = [deepcopy(init), deepcopy(expect), deepcopy(tx_arrive), deepcopy(issue)]
    buggy_events = deepcopy(prefix) + [deepcopy(invalidate)]
    correct_events = deepcopy(prefix) + [deepcopy(tx_wait), deepcopy(invalidate)]
    correct_events[-1]["seq"] = 5
    _append_pair(
        named, family="MBARRIER", scenario_id="INVALIDATE-REQUIRES-OBSERVATION", rule_ids=("MB-03", "MB-08"),
        dimensions=("mbarrier", "lifecycle", "terminal-observation", "invalidate"),
        buggy_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=buggy_events),
        correct_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=correct_events),
        buggy_verdict="SYNC_HAZARD", buggy_diagnostic="INVALID_MBARRIER_UNOBSERVED_PHASE",
        diagnostic_events=("invalidate",), protocol_state={"required_observed_generation": 0},
        required_correct_edges=(_edge(_phase("bar", 0), "wait", "COMPLETION_TO_WAIT"),),
        bad_execution=("generation 0 completes through TMA transaction bytes", "owner invalidates bar without a successful wait"),
    )

    # Live reinitialization is invalid; wait+invalidate starts a fresh lifecycle epoch.
    arrive = {**tx_arrive, "seq": 1}
    wait = {**tx_wait, "seq": 2}
    buggy_events = [deepcopy(init), {**deepcopy(init), "id": "init_again", "seq": 1}]
    correct_events = [deepcopy(init), deepcopy(arrive), deepcopy(wait),
                      {**deepcopy(invalidate), "seq": 3},
                      {**deepcopy(init), "id": "init_epoch1", "seq": 4, "lifecycle_epoch": 1}]
    _append_pair(
        named, family="MBARRIER", scenario_id="REINITIALIZE-LIFECYCLE", rule_ids=("MB-01", "MB-03"),
        dimensions=("mbarrier", "lifecycle", "reinitialize", "epoch"),
        buggy_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), events=buggy_events),
        correct_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), events=correct_events),
        buggy_verdict="SYNC_HAZARD", buggy_diagnostic="REINIT_LIVE", diagnostic_events=("init_again",),
        protocol_state={"lifecycle": "LIVE", "lifecycle_epoch": 0},
        bad_execution=("bar is initialized and still live", "init_again attempts to reset the same lifecycle"),
    )

    # Frontends must coalesce polling through the successful acquire observation.
    correct = _legacy_raw("PGIR-MB-08-MB-REUSE-V00-CORRECT")
    buggy = deepcopy(correct)
    next(event for event in buggy["events"] if event["id"] == "wait0")["parity_evidence"]["polling_coalesced"] = False
    _append_pair(
        named, family="MBARRIER", scenario_id="NONCOALESCED-POLL", rule_ids=("MB-09", "MB-10"),
        dimensions=("mbarrier", "parity-evidence", "polling", "acquire-until-true"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict=None,
        buggy_diagnostic="SCHEMA_INVALID", buggy_stage="MODEL",
        bad_execution=("frontend exposes a one-shot/non-coalesced poll", "successful acquire observation is not established"),
    )

    # wait_group 1 closes the older of two groups but leaves the newest open.
    actors = [_lane("owner", 0)]
    resource = [{"id": "copies", "kind": "async_group", "participants": ["owner"],
                 "capacity": 8, "protocol": "CP_ASYNC", "scope": "CTA"}]
    regions = [
        _region("source0", "global", address_space="GLOBAL", offset=0, extent=16),
        _region("source1", "global", address_space="GLOBAL", offset=16, extent=16),
        _region("tile0", "shared", offset=0, extent=16),
        _region("tile1", "shared", offset=16, extent=16),
    ]
    prefix = [
        {"id": "issue0", "type": "CpAsyncIssue", "actor": "owner", "seq": 0, "resource": "copies", "source_region": "source0", "destination_region": "tile0", "bytes": 16, "group_owner": "owner"},
        {"id": "commit0", "type": "CpAsyncCommitGroup", "actor": "owner", "seq": 1, "resource": "copies", "group_owner": "owner", "group_seq": 0},
        {"id": "issue1", "type": "CpAsyncIssue", "actor": "owner", "seq": 2, "resource": "copies", "source_region": "source1", "destination_region": "tile1", "bytes": 16, "group_owner": "owner"},
        {"id": "commit1", "type": "CpAsyncCommitGroup", "actor": "owner", "seq": 3, "resource": "copies", "group_owner": "owner", "group_seq": 1},
        {"id": "wait1", "type": "CpAsyncWaitGroup", "actor": "owner", "seq": 4, "resource": "copies", "group_owner": "owner", "wait_group_count": 1},
    ]
    final_wait = {"id": "wait0", "type": "CpAsyncWaitGroup", "actor": "owner", "seq": 6,
                  "resource": "copies", "group_owner": "owner", "wait_group_count": 0}
    buggy_events = deepcopy(prefix) + [{"id": "overwrite", "type": "MemWrite", "actor": "owner", "seq": 5, "region": "tile1"}, deepcopy(final_wait)]
    correct_events = deepcopy(prefix) + [{"id": "overwrite", "type": "MemWrite", "actor": "owner", "seq": 5, "region": "tile0"}, deepcopy(final_wait)]
    cp0 = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE", 0)
    cp1 = _group_completion("CP_ASYNC", "owner", "GROUP_COMPLETE", 1)
    _append_pair(
        named, family="CP-ASYNC", scenario_id="TWO-GROUP-WAIT-ONE", rule_ids=("CP-01", "CP-02", "CP-04", "MEM-04"),
        dimensions=("cp.async", "two-groups", "wait-group-1", "oldest-vs-newest"),
        buggy_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=buggy_events),
        correct_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=correct_events),
        buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", diagnostic_events=("issue1", "overwrite"),
        expected_effects=(
            _effect("effect:issue1:destination", "owner", "WRITE", "tile1", "issue1", cp1),
            _effect("overwrite", "owner", "WRITE", "tile1", "overwrite", "overwrite"),
        ), protocol_state={"visibility_transfers": []},
        required_bug_edges=(_edge(cp0, "wait1", "COMPLETION_TO_WAIT"),),
        forbidden_bug_edges=(_edge(cp1, "wait1", "COMPLETION_TO_WAIT"),),
        required_correct_edges=(_edge(cp0, "wait1", "COMPLETION_TO_WAIT"),),
        bad_execution=("groups 0 and 1 are pending", "wait_group 1 closes only group 0", "tile1 is overwritten while group 1 remains open"),
    )

    # One logical TMA issue may carry multiple exact transfers but one completion contribution.
    actors = [_lane("owner", 0)]
    resource = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    regions = [
        _region("source0", "global", address_space="GLOBAL", offset=0, extent=8),
        _region("source1", "global", address_space="GLOBAL", offset=8, extent=8),
        _region("tile0", "shared", offset=0, extent=8),
        _region("tile1", "shared", offset=8, extent=8),
    ]
    events = [
        {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "expect", "type": "MBarrierExpectTx", "actor": "owner", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": 16},
        {"id": "issue", "type": "TmaLoadIssue", "actor": "owner", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": 16,
         "transfers": [{"source_region": "source0", "destination_region": "tile0", "bytes": 8},
                       {"source_region": "source1", "destination_region": "tile1", "bytes": 8}]},
        {"id": "arrive", "type": "MBarrierArrive", "actor": "owner", "seq": 3, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1},
        {"id": "wait", "type": "MBarrierWaitParity", "actor": "owner", "seq": 4, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": _parity("owner:bar", 0)},
    ]
    correct = _raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=deepcopy(events))
    buggy = deepcopy(correct)
    next(event for event in buggy["events"] if event["id"] == "issue")["transfers"][1]["bytes"] = 4
    _append_pair(
        named, family="TMA-LOAD", scenario_id="MULTI-TRANSFER-ONE-COMPLETION", rule_ids=("TMA-03", "MB-07"),
        dimensions=("tma-load", "multi-transfer", "byte-accounting", "single-completion"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict=None,
        buggy_diagnostic="TMA_TRANSFER_BYTES_MISMATCH", buggy_stage="MODEL",
        required_correct_edges=(_edge("issue", _phase("bar", 0), "ISSUE_TO_PHASE"),),
        bad_execution=("logical issue declares 16 bytes", "its transfer list accounts for only 12 bytes"),
    )

    # WGMMA wait_group 1 makes only the older result ready.
    actors = [{"id": "wg", "kind": "WARPGROUP", "participants": list(range(128))}]
    resource = [{"id": "mma", "kind": "async_group", "participants": ["wg"],
                 "capacity": 8, "protocol": "WGMMA", "scope": "CTA"}]
    regions = [
        _region("operand0", "shared", offset=0, extent=16),
        _region("operand1", "shared", offset=16, extent=16),
        _region("acc0", "reg0", address_space="REGISTER", extent=16),
        _region("acc1", "reg1", address_space="REGISTER", extent=16),
    ]
    prefix = [
        {"id": "fence", "type": "WgmmaFenceOperand", "actor": "wg", "seq": 0, "resource": "mma", "group_owner": "wg", "regions": ["operand0", "operand1"]},
        {"id": "issue0", "type": "WgmmaIssue", "actor": "wg", "seq": 1, "resource": "mma", "group_owner": "wg", "regions": ["operand0"], "destination_region": "acc0"},
        {"id": "commit0", "type": "WgmmaCommitGroup", "actor": "wg", "seq": 2, "resource": "mma", "group_owner": "wg", "group_seq": 0},
        {"id": "issue1", "type": "WgmmaIssue", "actor": "wg", "seq": 3, "resource": "mma", "group_owner": "wg", "regions": ["operand1"], "destination_region": "acc1"},
        {"id": "commit1", "type": "WgmmaCommitGroup", "actor": "wg", "seq": 4, "resource": "mma", "group_owner": "wg", "group_seq": 1},
        {"id": "wait1", "type": "WgmmaWaitGroup", "actor": "wg", "seq": 5, "resource": "mma", "group_owner": "wg", "wait_group_count": 1, "completion_kind": "RESULT_READY"},
    ]
    final_wait = {"id": "wait0", "type": "WgmmaWaitGroup", "actor": "wg", "seq": 7,
                  "resource": "mma", "group_owner": "wg", "wait_group_count": 0, "completion_kind": "RESULT_READY"}
    buggy_events = deepcopy(prefix) + [{"id": "read", "type": "MemRead", "actor": "wg", "seq": 6, "region": "acc1"}, deepcopy(final_wait)]
    correct_events = deepcopy(prefix) + [{"id": "read", "type": "MemRead", "actor": "wg", "seq": 6, "region": "acc0"}, deepcopy(final_wait)]
    wg0 = _group_completion("WGMMA", "wg", "RESULT_READY", 0)
    wg1 = _group_completion("WGMMA", "wg", "RESULT_READY", 1)
    _append_pair(
        named, family="WGMMA", scenario_id="TWO-GROUP-WAIT-ONE", rule_ids=("WG-06", "WG-07", "MEM-04"),
        dimensions=("wgmma", "two-groups", "wait-group-1", "result-frontier"),
        buggy_raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=buggy_events),
        correct_raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=correct_events),
        buggy_verdict="SYNC_HAZARD", buggy_diagnostic="WGMMA_RESULT_NOT_READY", diagnostic_events=("issue1", "read"),
        expected_effects=(
            _effect("effect:issue1:accumulator", "wg", "WRITE", "acc1", "issue1", wg1),
            _effect("read", "wg", "READ", "acc1", "read", "read"),
        ), protocol_state={"visibility_transfers": []},
        required_bug_edges=(_edge(wg0, "wait1", "COMPLETION_TO_WAIT"),),
        forbidden_bug_edges=(_edge(wg1, "wait1", "COMPLETION_TO_WAIT"),),
        required_correct_edges=(_edge(wg0, "wait1", "COMPLETION_TO_WAIT"),),
        bad_execution=("WGMMA groups 0 and 1 are pending", "wait_group 1 closes only result 0", "accumulator 1 is read early"),
    )

    # TMA-store wait_group 1 likewise leaves the newest global destination open.
    actors = [_lane("owner", 0)]
    resource = [{"id": "stores", "kind": "async_group", "participants": ["owner"],
                 "capacity": 8, "protocol": "TMA_STORE", "scope": "CTA"}]
    regions = [
        _region("source0", "shared", offset=0, extent=16),
        _region("source1", "shared", offset=16, extent=16),
        _region("destination0", "global", address_space="GLOBAL", offset=0, extent=16),
        _region("destination1", "global", address_space="GLOBAL", offset=16, extent=16),
    ]
    prefix = [
        {"id": "issue0", "type": "TmaStoreIssue", "actor": "owner", "seq": 0, "resource": "stores", "group_owner": "owner", "source_region": "source0", "destination_region": "destination0", "bytes": 16},
        {"id": "commit0", "type": "TmaStoreCommitGroup", "actor": "owner", "seq": 1, "resource": "stores", "group_owner": "owner", "group_seq": 0},
        {"id": "issue1", "type": "TmaStoreIssue", "actor": "owner", "seq": 2, "resource": "stores", "group_owner": "owner", "source_region": "source1", "destination_region": "destination1", "bytes": 16},
        {"id": "commit1", "type": "TmaStoreCommitGroup", "actor": "owner", "seq": 3, "resource": "stores", "group_owner": "owner", "group_seq": 1},
        {"id": "wait1", "type": "TmaStoreWaitGroup", "actor": "owner", "seq": 4, "resource": "stores", "group_owner": "owner", "wait_group_count": 1, "wait_mode": "GLOBAL"},
    ]
    final_wait = {"id": "wait0", "type": "TmaStoreWaitGroup", "actor": "owner", "seq": 6,
                  "resource": "stores", "group_owner": "owner", "wait_group_count": 0, "wait_mode": "GLOBAL"}
    buggy_events = deepcopy(prefix) + [{"id": "read", "type": "MemRead", "actor": "owner", "seq": 5, "region": "destination1"}, deepcopy(final_wait)]
    correct_events = deepcopy(prefix) + [{"id": "read", "type": "MemRead", "actor": "owner", "seq": 5, "region": "destination0"}, deepcopy(final_wait)]
    store0 = _group_completion("TMA_STORE", "owner", "GLOBAL_COMPLETE", 0)
    store1 = _group_completion("TMA_STORE", "owner", "GLOBAL_COMPLETE", 1)
    _append_pair(
        named, family="TMA-STORE", scenario_id="TWO-GROUP-WAIT-ONE", rule_ids=("TMA-05", "MEM-04"),
        dimensions=("tma-store", "two-groups", "wait-group-1", "global-frontier"),
        buggy_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=buggy_events),
        correct_raw=_raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), regions=deepcopy(regions), events=correct_events),
        buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", diagnostic_events=("issue1", "read"),
        expected_effects=(
            _effect("effect:issue1:destination", "owner", "WRITE", "destination1", "issue1", store1),
            _effect("read", "owner", "READ", "destination1", "read", "read"),
        ), protocol_state={"visibility_transfers": []},
        required_bug_edges=(_edge(store0, "wait1", "COMPLETION_TO_WAIT"),),
        forbidden_bug_edges=(_edge(store1, "wait1", "COMPLETION_TO_WAIT"),),
        required_correct_edges=(_edge(store0, "wait1", "COMPLETION_TO_WAIT"),),
        bad_execution=("TMA-store groups 0 and 1 are pending", "wait_group 1 closes only destination 0", "destination 1 is read early"),
    )

    # arrive_drop is deliberately outside the supported operational subset.
    actors = [_lane("owner", 0)]
    resource = [{"id": "bar", "kind": "mbarrier", "participants": ["owner"]}]
    init = {"id": "init", "type": "MBarrierInit", "actor": "owner", "seq": 0,
            "resource": "bar", "lifecycle_epoch": 0, "count": 1}
    arrive = {"id": "arrive", "type": "MBarrierArrive", "actor": "owner", "seq": 1,
              "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1}
    wait = {"id": "wait", "type": "MBarrierWaitParity", "actor": "owner", "seq": 2,
            "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0,
            "parity_evidence": _parity("owner:bar", 0)}
    correct = _raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource),
                   events=[deepcopy(init), deepcopy(arrive), deepcopy(wait)])
    buggy = deepcopy(correct)
    next(event for event in buggy["events"] if event["id"] == "arrive")["type"] = "MBarrierArriveDrop"
    _append_pair(
        named, family="MBARRIER", scenario_id="ARRIVE-DROP-FAIL-CLOSED", rule_ids=("MB-05",),
        dimensions=("mbarrier", "unsupported-event", "arrive-drop", "fail-closed"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict=None,
        buggy_diagnostic="KNOWN_EVENT_UNSUPPORTED", buggy_stage="MODEL",
        bad_execution=("arrive_drop changes future phase participation", "the current PGIR subset cannot model that state change"),
    )

    # Hardware parity wraps after two generations; logical generation remains exact.
    events = [deepcopy(init)]
    for generation in range(3):
        events.extend([
            {"id": f"arrive{generation}", "type": "MBarrierArrive", "actor": "owner", "seq": len(events),
             "resource": "bar", "lifecycle_epoch": 0, "generation": generation, "count": 1},
            {"id": f"wait{generation}", "type": "MBarrierWaitParity", "actor": "owner", "seq": len(events) + 1,
             "resource": "bar", "lifecycle_epoch": 0, "generation": generation, "raw_parity": generation % 2,
             "parity_evidence": _parity("owner:bar", generation)},
        ])
    correct = _raw("temporary", actors=deepcopy(actors), resources=deepcopy(resource), events=deepcopy(events))
    buggy = deepcopy(correct)
    next(event for event in buggy["events"] if event["id"] == "wait2")["raw_parity"] = 1
    _append_pair(
        named, family="MBARRIER", scenario_id="PARITY-WRAP", rule_ids=("MB-09",),
        dimensions=("mbarrier", "three-generations", "parity-wrap", "logical-generation"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict=None,
        buggy_diagnostic="MALFORMED_STREAM_PARITY", buggy_stage="MODEL",
        required_correct_edges=(_edge(_phase("bar", 2), "wait2", "COMPLETION_TO_WAIT"),),
        bad_execution=("generation 2 wraps to hardware parity 0", "wait2 incorrectly supplies parity 1"),
    )

    # A wait for a generation older than the immediately previous generation is stale.
    stale_actors = [deepcopy(actors[0]), _lane("late", 1)]
    stale_resource = deepcopy(resource)
    stale_resource[0]["participants"] = ["owner", "late"]
    correct = _raw("temporary", actors=deepcopy(stale_actors), resources=stale_resource, events=deepcopy(events), cta_threads=2)
    buggy = deepcopy(correct)
    buggy["events"].append({
        "id": "stale_wait", "type": "MBarrierWaitParity", "actor": "late", "seq": 0,
        "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0,
        "parity_evidence": _parity("late:bar", 0), "depends_on": ["wait2"],
    })
    _append_pair(
        named, family="MBARRIER", scenario_id="STALE-GENERATION-WAIT", rule_ids=("MB-09",),
        dimensions=("mbarrier", "three-generations", "stale-wait", "logical-generation"),
        buggy_raw=buggy, correct_raw=correct, buggy_verdict="SYNC_HAZARD",
        buggy_diagnostic="PROTOCOL_INVALID_PHASE_ADVANCE", diagnostic_events=(_phase("bar", 1),),
        protocol_state={"logical_generation": 1, "stale_target_generation": 0, "remaining_logical_waits": ["stale_wait"]},
        required_correct_edges=(_edge(_phase("bar", 2), "wait2", "COMPLETION_TO_WAIT"),),
        bad_execution=("generations 0, 1, and 2 retire", "a later wait targets stale generation 0"),
    )

    assert len(named) == 50
    CASES.extend(named)


_add_named_cases()
