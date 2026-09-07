from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RuleCase:
    label: str
    pair_id: str
    rule_id: str
    polarity: str
    expected_verdict: str
    expected_diagnostic: str | None
    possible_bad_execution: tuple[str, ...]
    raw: dict[str, Any]


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
    return RuleCase(
        label=label,
        pair_id=pair_id,
        rule_id=rule_id,
        polarity=polarity,
        expected_verdict=buggy_verdict if buggy else "SAFE",
        expected_diagnostic=buggy_diagnostic if buggy else None,
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
    return (
        _case(family="WGMMA-ENDPOINT", rule_id=rule, variant=variant, buggy=True,
              raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=buggy_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
        _case(family="WGMMA-ENDPOINT", rule_id=rule, variant=variant, buggy=False,
              raw=_raw("temporary", cta_threads=128, actors=deepcopy(actors), resources=deepcopy(resources), regions=deepcopy(regions), events=correct_events),
              buggy_verdict="DATA_HAZARD", buggy_diagnostic="UNORDERED_CONFLICT", bad_execution=bad),
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
