from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fastjsonschema
import pytest

import pipeguard.checker as checker_module
from pipeguard.checker import (
    Effect,
    _anchor_ids,
    _data_hazard,
    _effects,
    _overlapping_effect_pairs,
    check_module,
)
from pipeguard.engine import ExecutionState, Executor, ProtocolError, Transition
from pipeguard.incremental import END, IncrementalEngine, StableReadySet, build_anchor_streams
from pipeguard.model import (
    Actor,
    Event,
    PGIRError,
    _bucket_events_by_actor,
    _dangling_dependencies,
    _diagnostic_schema_validator,
    _fast_schema_validator,
    load_module,
    module_from_dict,
    stable_channel_handoff_id,
    stable_collective_release_id,
    stable_completion_id,
    stable_group_completion_id,
    stable_group_id,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return load_module(FIXTURES / name)


def verdict(name: str) -> dict:
    return check_module(load(name))


@pytest.mark.parametrize("fixture", ["safe_handoff.json", "cluster_dsmem_handoff.json"])
def test_empty_finite_event_graph_is_safe_for_both_pgir_versions(fixture: str):
    raw = json.loads((FIXTURES / fixture).read_text())
    raw["events"] = []
    module = module_from_dict(raw)
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"
    assert check_module(module, engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.parametrize("fixture", ["safe_handoff.json", "missing_sync_data_hazard.json"])
def test_overlapping_effect_index_preserves_naive_pair_order(fixture: str):
    module = load(fixture)
    effects = _effects(module)
    expected = []
    for left_index, left in enumerate(effects):
        left_region = module.region_map[left.region]
        for right in effects[left_index + 1:]:
            right_region = module.region_map[right.region]
            if (
                left_region.address_space == right_region.address_space
                and left_region.owner_cta_rank == right_region.owner_cta_rank
                and left_region.allocation_id == right_region.allocation_id
                and left_region.offset < right_region.offset + right_region.extent
                and right_region.offset < left_region.offset + left_region.extent
            ):
                expected.append((left.id, right.id))
    actual = [
        (left.id, right.id)
        for left, right in _overlapping_effect_pairs(module, effects)
    ]
    assert actual == expected


def test_data_hazard_disjoint_intervals_do_not_scan_all_effect_pairs(monkeypatch):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["resources"] = []
    raw["regions"] = [
        {
            "id": f"region-{index}",
            "allocation_id": "array",
            "address_space": "GLOBAL",
            "offset": index * 4,
            "extent": 4,
        }
        for index in range(4096)
    ]
    actor_sequences = {actor["id"]: 0 for actor in raw["actors"]}
    raw["events"] = []
    for index, region in enumerate(raw["regions"]):
        actor = raw["actors"][index % len(raw["actors"])]["id"]
        sequence = actor_sequences[actor]
        actor_sequences[actor] += 1
        raw["events"].append({
            "id": f"write-{index}",
            "type": "MemWrite",
            "actor": actor,
            "seq": sequence,
            "region": region["id"],
        })
    module = module_from_dict(raw)

    class NoQuadraticSlice(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("data hazard scan sliced the complete effect suffix")
            return super().__getitem__(key)

    effects = NoQuadraticSlice(
        Effect(
            id=f"effect-{event.id}",
            actor=event.actor,
            mode="WRITE",
            region=event.region,
            begin=event.id,
            end=event.id,
            async_window=False,
        )
        for event in module.events
    )
    monkeypatch.setattr(checker_module, "_effects", lambda _module: effects)
    assert _data_hazard(module, object()) is None


def test_module_indexes_are_constructed_once_and_reused():
    module = load("safe_handoff.json")
    assert module.event_map is module.event_map
    assert module.resource_map is module.resource_map
    assert module.region_map is module.region_map
    assert module.actor_map is module.actor_map


def test_events_by_actor_buckets_once_in_actor_order_and_sorts_each_trace():
    class SinglePassEvents(tuple):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("events were scanned more than once")
            return super().__iter__()

    actors = (
        Actor("z_actor", "LANE", (0,)),
        Actor("a_actor", "LANE", (1,)),
        Actor("empty_actor", "LANE", (2,)),
    )
    events = SinglePassEvents((
        Event("a_late", "MemRead", "a_actor", 1),
        Event("z_late", "MemRead", "z_actor", 1),
        Event("a_tie_b", "MemRead", "a_actor", 0),
        Event("unknown", "MemRead", "missing_actor", 0),
        Event("z_early", "MemRead", "z_actor", 0),
        Event("a_tie_a", "MemRead", "a_actor", 0),
    ))

    grouped = _bucket_events_by_actor(actors, events)

    assert events.iterations == 1
    assert list(grouped) == ["z_actor", "a_actor", "empty_actor"]
    assert [[event.id for event in trace] for trace in grouped.values()] == [
        ["z_early", "z_late"],
        ["a_tie_a", "a_tie_b", "a_late"],
        [],
    ]


@pytest.mark.parametrize(
    ("mutation", "code", "details"),
    [
        (
            lambda raw: raw["events"][0].update(actor="missing"),
            "DANGLING_REFERENCE",
            {"event": "init", "actor": "missing"},
        ),
        (
            lambda raw: raw["events"][1].update(id=raw["events"][0]["id"]),
            "DUPLICATE_ID",
            {"id": "init", "kind": "event"},
        ),
        (
            lambda raw: raw["events"][1].update(seq=3),
            "NONCONTIGUOUS_SEQUENCE",
            {"actor": "a_producer", "sequence": [0, 2, 3]},
        ),
    ],
)
def test_events_by_actor_bucketing_preserves_validation_errors(mutation, code, details):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    mutation(raw)

    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)

    assert caught.value.code == code
    assert caught.value.details == details


def test_dangling_dependencies_scan_only_dependency_ids_and_preserve_details():
    class MembershipOnlyEvents(dict):
        def __iter__(self):
            raise AssertionError("event table must not be scanned")

        def keys(self):
            raise AssertionError("event keys must not be scanned")

    events = MembershipOnlyEvents(known=object())

    assert _dangling_dependencies(("missing-z", "known", "missing-a", "missing-z"), events) == [
        "missing-a",
        "missing-z",
    ]

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"][3]["depends_on"] = ["missing-z", "init", "missing-a"]
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "DANGLING_REFERENCE"
    assert caught.value.details == {"event": "wait", "depends_on": ["missing-a", "missing-z"]}


def test_self_dependency_remains_a_distinct_validation_error():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"][3]["depends_on"] = ["wait"]

    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)

    assert caught.value.code == "SELF_DEPENDENCY"
    assert caught.value.details == {"event": "wait"}


@pytest.mark.parametrize("fixture", ["safe_handoff.json", "cluster_dsmem_handoff.json"])
def test_fast_schema_acceptance_matches_canonical_validator(fixture: str):
    valid = json.loads((FIXTURES / fixture).read_text())
    documents = [valid]
    for mutation in (
        lambda raw: raw.update(unexpected=True),
        lambda raw: raw["events"][0].update(unexpected=True),
        lambda raw: raw["events"][0].update(id=""),
        lambda raw: raw["events"][0].update(seq=-1),
        lambda raw: raw["events"][0].update(depends_on=["duplicate", "duplicate"]),
    ):
        raw = json.loads(json.dumps(valid))
        mutation(raw)
        documents.append(raw)

    fast = _fast_schema_validator(valid["schema_version"])
    canonical = _diagnostic_schema_validator(valid["schema_version"])
    for raw in documents:
        try:
            fast(raw)
            fast_accepts = True
        except fastjsonschema.JsonSchemaException:
            fast_accepts = False
        canonical_accepts = next(canonical.iter_errors(raw), None) is None
        assert fast_accepts == canonical_accepts


def test_safe_handoff_has_visibility_path_and_manifest():
    result = verdict("safe_handoff.json")
    assert result["verdict"] == "SAFE"
    assert result["manifest"]["counts"] == {"actors": 2, "events": 5, "regions": 1, "resources": 1}
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": "completion:mbarrier-phase:bar:l00000000:g00000000", "to": "wait", "kind": "COMPLETION_TO_WAIT"} in edges


@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("missing_arrive.json", "MISSING_ARRIVAL"),
        ("stale_generation.json", "PROTOCOL_INVALID_PHASE_ADVANCE"),
        ("phase_lap.json", "PHASE_LAP"),
        ("use_before_init.json", "USE_BEFORE_INIT"),
        ("tma_bytes_mismatch.json", "TX_BYTES_MISMATCH"),
    ],
)
def test_sync_hazards(fixture: str, code: str):
    result = verdict(fixture)
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == code
    assert result["witness"]["reachable_execution_prefix"] is not None


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_partial_next_phase_before_old_wait_is_safe(engine):
    result = check_module(load("partial_next_phase.json"), engine=engine)
    assert result["verdict"] == "SAFE"
    if engine == "crosscheck":
        comparison = result["artifacts"]["crosscheck"]
        assert comparison["maximal"]["verdict"] == comparison["exhaustive"]["verdict"] == "SAFE"
        assert comparison["maximal"]["boundaries"] == comparison["exhaustive"]["boundaries"]


def test_initial_remaining_tracks_future_obligations_and_partial_arrival_is_allowed():
    executor = Executor(load("partial_next_phase.json"))
    state = executor.initial_state()
    assert state.remaining == {
        ("bar", 0, 0): {"wait0"},
        ("bar", 0, 1): {"wait1"},
    }
    assert state.observed == set()
    initial_fingerprint = executor.fingerprint(state)
    without_future = state.clone()
    without_future.remaining.pop(("bar", 0, 1))
    assert executor.fingerprint(without_future) != initial_fingerprint

    for identifier in ("init", "arrive0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION"))
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive1a"))
    assert state.barriers["bar"].generation == 1
    assert state.barriers["bar"].pending == 1
    assert state.remaining[("bar", 0, 0)] == {"wait0"}

    wait = next(item for item in executor.enabled(state) if item.id == "wait0")
    executor.apply(state, wait)
    assert "wait0" in state.observed
    assert ("bar", 0, 0) not in state.remaining
    before_duplicate = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, wait)
    assert caught.value.code == "DUPLICATE_LOGICAL_WAIT_CONSUMPTION"
    assert executor.fingerprint(state) == before_duplicate

    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive1b"))
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION"))
    assert state.barriers["bar"].generation == 2


def test_tma_issue_does_not_credit_completion_bytes():
    module = load("tma_completion_before_arrival.json")
    executor = Executor(module)
    state = executor.initial_state()
    for identifier in ("init", "expect", "issue"):
        transition = next(item for item in executor.enabled(state) if item.id == identifier)
        executor.apply(state, transition)
    barrier = state.barriers["bar"]
    completion = stable_completion_id("issue", 0)
    assert barrier.expected_tx == 4
    assert barrier.completed_tx == 0
    assert completion in state.pending_completions
    assert not any(item["kind"] == "PHASE_COMPLETION" for item in state.trace)


@pytest.mark.parametrize(
    ("fixture", "relation"),
    [
        ("tma_completion_before_arrival.json", "before"),
        ("tma_completion_after_arrival.json", "after"),
    ],
)
def test_tma_completion_is_independent_and_can_precede_or_follow_arrival(fixture: str, relation: str):
    result = verdict(fixture)
    assert result["verdict"] == "SAFE"
    trace = [item["transition"] for item in result["artifacts"]["canonical_trace"]]
    completion_index = next(index for index, item in enumerate(trace) if item.startswith("completion:tma-load:"))
    arrival_index = trace.index("arrive")
    assert (completion_index < arrival_index) if relation == "before" else (completion_index > arrival_index)
    phase_index = next(index for index, item in enumerate(trace) if item.startswith("completion:mbarrier-phase:"))
    assert phase_index > max(completion_index, arrival_index)


def test_missing_synchronization_is_data_hazard_not_canonical_order():
    result = verdict("missing_sync_data_hazard.json")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["witness"]["events"] == ["read", "write"]


def test_tma_destination_access_window_requires_completion_observation():
    result = verdict("tma_window_hazard.json")
    assert result["verdict"] == "DATA_HAZARD"
    window = next(effect for effect in result["witness"]["effects"] if effect["id"] == "effect:issue:destination")
    assert window["window"] == ["issue", stable_completion_id("issue", 0)]


def test_tma_source_read_window_rejects_unordered_overwrite():
    raw = json.loads((FIXTURES / "tma_window_hazard.json").read_text())
    read = next(event for event in raw["events"] if event["id"] == "read")
    read.update({"type": "MemWrite", "region": "source"})
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "DATA_HAZARD"
    source = next(effect for effect in result["witness"]["effects"] if effect["id"] == "effect:issue:source")
    assert source["mode"] == "READ"
    assert source["window"] == ["issue", stable_completion_id("issue", 0)]


def test_tma_load_missing_source_is_fail_closed():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    next(event for event in raw["events"] if event["type"] == "TmaLoadIssue").pop("source_region")
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "EVENT_FIELDS_MISSING"


def test_unknown_and_known_unimplemented_events_are_distinct_and_fail_closed():
    with pytest.raises(PGIRError, match="MysteryOp") as unknown:
        load("unknown_event.json")
    assert unknown.value.code == "UNKNOWN_EVENT"

    raw = json.loads((FIXTURES / "unknown_event.json").read_text())
    raw["events"][0]["type"] = "AtomicReadModifyWrite"
    with pytest.raises(PGIRError) as known:
        module_from_dict(raw)
    assert known.value.code == "KNOWN_EVENT_UNSUPPORTED"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda raw: raw["events"][0].update({"surprise": 1}), "SCHEMA_INVALID"),
        (lambda raw: raw["events"].__setitem__(1, {**raw["events"][1], "id": raw["events"][0]["id"]}), "DUPLICATE_ID"),
        (lambda raw: raw["events"][0].update({"actor": "missing"}), "DANGLING_REFERENCE"),
        (lambda raw: raw["assumptions"].update({"parity_normalization_status": "UNRESOLVED"}), "INCONCLUSIVE_PARITY"),
        (lambda raw: raw["resources"][0].update({"participants": ["missing"]}), "UNRESOLVED_PARTICIPANTS"),
    ],
)
def test_loader_fail_closed(mutation, code: str):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    mutation(raw)
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == code


def test_arrival_underflow_is_rejected():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"][2]["count"] = 2
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "ARRIVAL_UNDERFLOW"


def test_init_count_must_fit_static_participant_contract():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"][0]["count"] = 3
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "PARTICIPANT_CONTRACT_UNSATISFIABLE"


def lifecycle_module(*event_types: str):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [raw["actors"][0]]
    raw["resources"][0]["participants"] = ["a_producer"]
    events = []
    for seq, event_type in enumerate(event_types):
        event = {"id": f"e{seq}", "type": event_type, "actor": "a_producer", "seq": seq, "resource": "bar", "lifecycle_epoch": 0}
        if event_type == "MBarrierInit":
            event["count"] = 1
        elif event_type == "MBarrierArrive":
            event.update({"count": 1, "generation": 0})
        elif event_type == "MBarrierWaitParity":
            event.update({
                "generation": 0,
                "raw_parity": 0,
                "parity_evidence": {"derivation_kind": "BARRIER_HISTORY", "logical_stream_id": "lifecycle", "lifecycle_epoch": 0, "stream_rank": 0, "polling_coalesced": True},
            })
        events.append(event)
    raw["events"] = events
    return module_from_dict(raw)


def test_invalidate_after_completed_phase_is_legal():
    result = check_module(lifecycle_module("MBarrierInit", "MBarrierArrive", "MBarrierWaitParity", "MBarrierInvalidate"))
    assert result["verdict"] == "SAFE"


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_invalidate_after_phase_conditions_reach_zero_needs_no_wait(engine):
    result = check_module(
        lifecycle_module("MBarrierInit", "MBarrierArrive", "MBarrierInvalidate"),
        engine=engine,
    )
    assert result["verdict"] == "SAFE"


def test_use_after_invalidate_and_live_reinit_are_rejected():
    use_after = check_module(lifecycle_module("MBarrierInit", "MBarrierArrive", "MBarrierWaitParity", "MBarrierInvalidate", "MBarrierArrive"))
    assert use_after["diagnostics"][0]["code"] == "USE_AFTER_INVALIDATE"
    reinit = check_module(lifecycle_module("MBarrierInit", "MBarrierInit"))
    assert reinit["diagnostics"][0]["code"] == "REINIT_LIVE"


def scheduler_race(main_actor: str, arrival_actor: str):
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    raw["actors"] = [
        {"id": main_actor, "kind": "PARTITION", "participants": [0]},
        {"id": arrival_actor, "kind": "PARTITION", "participants": [1]},
    ]
    raw["target"]["cta_threads"] = 2
    raw["resources"][0]["participants"] = [main_actor, arrival_actor]
    for event in raw["events"]:
        event["actor"] = main_actor
    init = next(event for event in raw["events"] if event["id"] == "init")
    expect = next(event for event in raw["events"] if event["id"] == "expect")
    issue = next(event for event in raw["events"] if event["id"] == "issue")
    wait = next(event for event in raw["events"] if event["id"] == "wait")
    init["seq"], expect["seq"], issue["seq"], wait["seq"] = 0, 1, 2, 3
    arrive = next(event for event in raw["events"] if event["id"] == "arrive")
    arrive.update({"actor": arrival_actor, "seq": 0, "depends_on": ["init"]})
    return module_from_dict(raw)


def test_all_schedule_exploration_is_actor_rename_independent():
    first = check_module(scheduler_race("a_main", "z_arrival"), engine="exhaustive")
    second = check_module(scheduler_race("z_main", "a_arrival"), engine="exhaustive")
    assert first["verdict"] == second["verdict"] == "SYNC_HAZARD"
    assert first["diagnostics"][0]["code"] == second["diagnostics"][0]["code"] == "STALE_GENERATION"


def test_hold_anchor_terminal_is_actually_probed():
    executor = Executor(scheduler_race("a_main", "z_arrival"))
    outcomes = executor.explore_all(hold="expect")
    probes = [executor.probe(state, "expect")[0] for state in outcomes.terminals]
    assert any(error is not None and error.code == "STALE_GENERATION" for error in probes)


def test_predicated_false_arrival_does_not_publish_visibility():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"].append({"id": "c_arrival", "kind": "PARTITION", "participants": [2]})
    raw["target"]["cta_threads"] = 3
    raw["resources"][0]["participants"].append("c_arrival")
    false_arrival = next(event for event in raw["events"] if event["id"] == "arrive")
    false_arrival["predicate"] = False
    raw["events"].append({
        "id": "real_arrive", "type": "MBarrierArrive", "actor": "c_arrival", "seq": 0,
        "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 0, "depends_on": ["init"],
    })
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "DATA_HAZARD"
    assert not any(edge["kind"] == "BARRIER_ARRIVE" and edge["from"] == "arrive" for edge in result["artifacts"]["pdg"]["edges"])


def test_cross_actor_global_dependency_is_not_visibility():
    raw = json.loads((FIXTURES / "missing_sync_data_hazard.json").read_text())
    raw["regions"][0].update({"address_space": "GLOBAL", "allocation_id": "global"})
    next(event for event in raw["events"] if event["id"] == "read")["depends_on"] = ["write"]
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "DATA_HAZARD"


def test_generic_write_to_tma_source_requires_proxy_support():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    for event in raw["events"]:
        event["seq"] += 1
    raw["events"].append({"id": "source_write", "type": "MemWrite", "actor": "a", "seq": 0, "region": "source"})
    result = check_module(module_from_dict(raw))
    assert result["verdict"] == "DATA_HAZARD"
    assert "proxy" in result["witness"]["missing_path"]


def test_synthetic_id_namespace_collision_is_rejected():
    raw = json.loads((FIXTURES / "tma_window_hazard.json").read_text())
    next(event for event in raw["events"] if event["id"] == "read")["id"] = stable_completion_id("issue", 0)
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "RESERVED_ID_NAMESPACE"


def test_extraction_evidence_and_lane_bounds_are_required():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = []
    raw["events"] = []
    with pytest.raises(PGIRError) as empty:
        module_from_dict(raw)
    assert empty.value.code == "SCHEMA_INVALID"

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"][0]["participants"] = [999]
    with pytest.raises(PGIRError) as lane:
        module_from_dict(raw)
    assert lane.value.code == "PARTICIPANT_OUT_OF_RANGE"


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_unused_initialized_barrier_can_be_invalidated_without_a_completed_phase(engine):
    result = check_module(
        lifecycle_module("MBarrierInit", "MBarrierInvalidate"),
        engine=engine,
    )
    assert result["verdict"] == "SAFE"


def test_integral_float_and_missing_per_wait_evidence_fail_closed():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    next(event for event in raw["events"] if event["id"] == "wait")["generation"] = 0.0
    with pytest.raises(PGIRError) as number:
        module_from_dict(raw)
    assert number.value.code == "STRICT_INTEGER_REQUIRED"

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    next(event for event in raw["events"] if event["id"] == "wait").pop("parity_evidence")
    with pytest.raises(PGIRError) as evidence:
        module_from_dict(raw)
    assert evidence.value.code == "INCONCLUSIVE_PARITY"


def test_event_type_specific_fields_and_false_arrive_expect_tx():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    next(event for event in raw["events"] if event["id"] == "read")["bytes"] = 4
    with pytest.raises(PGIRError) as fields:
        module_from_dict(raw)
    assert fields.value.code == "EVENT_FIELDS_UNEXPECTED"

    raw = json.loads((FIXTURES / "missing_arrive.json").read_text())
    raw["events"] = [
        raw["events"][0],
        {"id": "false", "type": "MBarrierArriveExpectTx", "actor": "a", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "bytes": 8, "generation": 0, "predicate": False},
        {"id": "real", "type": "MBarrierArrive", "actor": "a", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "count": 2, "generation": 0},
        {"id": "wait", "type": "MBarrierWaitParity", "actor": "b", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": {"derivation_kind": "BARRIER_HISTORY", "logical_stream_id": "false-arrive-test", "lifecycle_epoch": 0, "stream_rank": 0, "polling_coalesced": True}, "depends_on": ["init"]},
    ]
    assert check_module(module_from_dict(raw))["verdict"] == "SAFE"

    raw["events"].insert(3, {
        "id": "false_next", "type": "MBarrierArriveExpectTx",
        "actor": "a", "seq": 3, "resource": "bar", "lifecycle_epoch": 0,
        "count": 1, "bytes": 8, "generation": 1, "predicate": False,
    })
    module = module_from_dict(raw)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert not any(
        "mbarrier-phase" in anchor and "g00000001" in anchor
        for anchor in _anchor_ids(module)
    )



def structured_evidence(stream: str, rank: int, lifecycle: int = 0) -> dict:
    return {
        "derivation_kind": "BARRIER_HISTORY",
        "logical_stream_id": stream,
        "lifecycle_epoch": lifecycle,
        "stream_rank": rank,
        "polling_coalesced": True,
    }


def strict_multi_observer_module(require_both: bool):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    names = ("a_producer", "b_observer", "c_observer")
    raw["actors"] = [
        {"id": name, "kind": "PARTITION", "participants": [index]}
        for index, name in enumerate(names)
    ]
    raw["target"]["cta_threads"] = 3
    raw["resources"][0]["participants"] = list(names)
    raw["regions"] = []
    dependencies = ["wait_b", "wait_c"] if require_both else ["wait_b"]
    raw["events"] = [
        {"id": "init", "type": "MBarrierInit", "actor": "a_producer", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "arrive0", "type": "MBarrierArrive", "actor": "a_producer", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 0},
        {"id": "arrive1", "type": "MBarrierArrive", "actor": "a_producer", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 1, "depends_on": dependencies},
        {"id": "wait_b", "type": "MBarrierWaitParity", "actor": "b_observer", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("b", 0), "depends_on": ["init"]},
        {"id": "wait_c", "type": "MBarrierWaitParity", "actor": "c_observer", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("c", 0), "depends_on": ["init"]},
    ]
    return module_from_dict(raw)


def test_required_and_multiple_observers():
    module = strict_multi_observer_module(True)
    assert module.required_logical_waits == {("bar", 0, 0): frozenset({"wait_b", "wait_c"})}
    for engine in ("maximal", "exhaustive", "crosscheck"):
        assert check_module(module, engine=engine)["verdict"] == "SAFE"


def test_missing_observer_blocks_only_phase_completion_all_engines():
    module = strict_multi_observer_module(False)
    for engine in ("maximal", "exhaustive", "crosscheck"):
        result = check_module(module, engine=engine)
        assert result["verdict"] == "SYNC_HAZARD"
        assert result["diagnostics"][0]["code"] == "PROTOCOL_INVALID_PHASE_ADVANCE"


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_invalidate_cannot_retire_unobserved_required_waits(engine):
    raw = json.loads(json.dumps(strict_multi_observer_module(True).raw))
    invalidate = next(event for event in raw["events"] if event["id"] == "arrive1")
    invalidate.clear()
    invalidate.update({
        "id": "invalidate",
        "type": "MBarrierInvalidate",
        "actor": "a_producer",
        "seq": 2,
        "resource": "bar",
        "lifecycle_epoch": 0,
    })

    result = check_module(module_from_dict(raw), engine=engine)
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "INVALIDATE_INCOMPLETE"
    remaining = result["witness"]["protocol_state"].get("remaining_logical_waits")
    if remaining is not None:
        assert remaining and set(remaining).issubset({"wait_b", "wait_c"})


def test_duplicate_rank_gap_wrong_generation_and_parity_fail_closed():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"] = [event for event in raw["events"] if event["id"] != "read"]
    wait = next(event for event in raw["events"] if event["id"] == "wait")
    duplicate = dict(wait, id="dup", seq=1)
    duplicate["parity_evidence"] = structured_evidence("b_consumer:bar", 1)
    raw["events"].append(duplicate)
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "DUPLICATE_LOGICAL_WAIT"

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    next(event for event in raw["events"] if event["id"] == "wait")["parity_evidence"]["stream_rank"] = 1
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "LOGICAL_WAIT_RANK_GAP"

    raw = json.loads((FIXTURES / "partial_next_phase.json").read_text())
    wait = next(event for event in raw["events"] if event["id"] == "wait1")
    wait.update(generation=2, raw_parity=0)
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "LOGICAL_WAIT_GENERATION_MISMATCH"

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    next(event for event in raw["events"] if event["id"] == "wait")["raw_parity"] = 1
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "MALFORMED_STREAM_PARITY"


def test_canonical_wait_stream_cannot_be_split_by_external_stream_id():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"] = [event for event in raw["events"] if event["id"] != "read"]
    wait = next(event for event in raw["events"] if event["id"] == "wait")
    duplicate = dict(wait, id="split", seq=1)
    duplicate["parity_evidence"] = structured_evidence("forged-split", 0)
    raw["events"].append(duplicate)

    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "LOGICAL_WAIT_STREAM_ID_MISMATCH"


@pytest.mark.parametrize("field",["derivation_kind","logical_stream_id","lifecycle_epoch","stream_rank","polling_coalesced"])
def test_structured_fields_required(field):
    r=json.loads((FIXTURES/"safe_handoff.json").read_text()); next(e for e in r["events"] if e["id"]=="wait")["parity_evidence"].pop(field)
    with pytest.raises(PGIRError) as e: module_from_dict(r)
    assert e.value.code=="SCHEMA_INVALID"


def test_unknown_derivation_fails_closed():
    r=json.loads((FIXTURES/"safe_handoff.json").read_text()); next(e for e in r["events"] if e["id"]=="wait")["parity_evidence"]["derivation_kind"]="UNKNOWN"
    with pytest.raises(PGIRError) as e: module_from_dict(r)
    assert e.value.code=="SCHEMA_INVALID"


@pytest.mark.parametrize("value",[False,None])
def test_polling_evidence_fail_closed(value):
    r=json.loads((FIXTURES/"safe_handoff.json").read_text()); evidence=next(e for e in r["events"] if e["id"]=="wait")["parity_evidence"]
    evidence.pop("polling_coalesced") if value is None else evidence.update(polling_coalesced=False)
    with pytest.raises(PGIRError) as e: module_from_dict(r)
    assert e.value.code=="SCHEMA_INVALID"


def test_phase_completion_guard_is_atomic_after_one_of_multiple_waits():
    executor = Executor(strict_multi_observer_module(False))
    state = executor.initial_state()
    for identifier in ("init", "arrive0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION"))
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "wait_b"))
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive1"))
    completion = next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION")
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, completion)
    assert caught.value.code == "PROTOCOL_INVALID_PHASE_ADVANCE"
    assert caught.value.event == completion.id
    assert caught.value.state["remaining_logical_waits"] == ["wait_c"]
    assert executor.fingerprint(state) == before


def test_rejected_tma_completion_is_failure_atomic():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    next(event for event in raw["events"] if event["id"] == "issue")["bytes"] = 8
    executor = Executor(module_from_dict(raw))
    state = executor.initial_state()
    for identifier in ("init", "expect", "issue"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    completion = next(item for item in executor.enabled(state) if item.kind == "TMA_COMPLETION")
    before = executor.fingerprint(state)

    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, completion)
    assert caught.value.code == "TX_BYTES_EXCEED_EXPECTED"
    assert executor.fingerprint(state) == before


def test_stale_phase_completion_transition_is_rejected_atomically():
    executor = Executor(load("safe_handoff.json"))
    state = executor.initial_state()
    for identifier in ("init", "write", "arrive"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    completion = next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION")
    executor.apply(state, completion)
    before = executor.fingerprint(state)

    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, completion)
    assert caught.value.code == "INVALID_PHASE_COMPLETION_TRANSITION"
    assert executor.fingerprint(state) == before


def test_missing_covered_observe_group_is_unsupported_not_sync_hazard():
    raw = json.loads((FIXTURES / "stale_generation.json").read_text())
    raw["events"] = [event for event in raw["events"] if event["type"] != "MBarrierWaitParity"]
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "UNSUPPORTED_MISSING_OBSERVE_GROUP"
    assert caught.value.details["target_generation"] == 0


def tx_delayed_phase_module():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    raw["events"] = [
        {"id": "init", "type": "MBarrierInit", "actor": "a", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "arrive0", "type": "MBarrierArrive", "actor": "a", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 0},
        {"id": "expect1", "type": "MBarrierExpectTx", "actor": "a", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "bytes": 4, "generation": 1},
        {"id": "issue1", "type": "TmaLoadIssue", "actor": "a", "seq": 3, "resource": "bar", "lifecycle_epoch": 0, "bytes": 4, "generation": 1, "source_region": "source", "destination_region": "tile"},
        {"id": "arrive1", "type": "MBarrierArrive", "actor": "a", "seq": 4, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 1},
        {"id": "wait0", "type": "MBarrierWaitParity", "actor": "a", "seq": 5, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("a:bar", 0)},
    ]
    return module_from_dict(raw)


def test_tx_delay_allows_old_wait_after_final_arrive_before_phase_completion():
    executor = Executor(tx_delayed_phase_module())
    state = executor.initial_state()
    for identifier in ("init", "arrive0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION"))
    for identifier in ("expect1", "issue1", "arrive1", "wait0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    barrier = state.barriers["bar"]
    assert barrier.generation == 1 and barrier.pending == 0
    assert barrier.completed_tx == 0 and barrier.expected_tx == 4
    assert state.observed == {"wait0"}
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "TMA_COMPLETION"))
    executor.apply(state, next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION"))
    assert state.barriers["bar"].generation == 2


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_tx_delay_schedule_that_completes_before_old_wait_is_risky(engine):
    result = check_module(tx_delayed_phase_module(), engine=engine)
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "PROTOCOL_INVALID_PHASE_ADVANCE"


def tx_pending_early_next_arrival_module():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    raw["actors"] = [
        {"id": "a", "kind": "PARTITION", "participants": [0]},
        {"id": "b", "kind": "PARTITION", "participants": [1]},
    ]
    raw["target"]["cta_threads"] = 2
    raw["resources"][0]["participants"] = ["a", "b"]
    raw["events"] = [
        {"id": "init", "type": "MBarrierInit", "actor": "a", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "expect0", "type": "MBarrierExpectTx", "actor": "a", "seq": 1, "resource": "bar", "lifecycle_epoch": 0, "bytes": 4, "generation": 0},
        {"id": "issue0", "type": "TmaLoadIssue", "actor": "a", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "bytes": 4, "generation": 0, "source_region": "source", "destination_region": "tile"},
        {"id": "arrive0", "type": "MBarrierArrive", "actor": "a", "seq": 3, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 0},
        {"id": "wait0", "type": "MBarrierWaitParity", "actor": "a", "seq": 4, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("a:bar", 0)},
        {"id": "arrive1", "type": "MBarrierArrive", "actor": "b", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 1, "depends_on": ["arrive0"]},
    ]
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_held_tx_completion_does_not_hide_zero_pending_early_arrival(engine):
    result = check_module(tx_pending_early_next_arrival_module(), engine=engine)
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "ARRIVE_WITH_ZERO_PENDING_COUNT"


def test_phase_lap_precedes_zero_pending_diagnostic():
    raw = json.loads(json.dumps(tx_pending_early_next_arrival_module().raw))
    next(event for event in raw["events"] if event["id"] == "arrive1")["generation"] = 2
    executor = Executor(module_from_dict(raw))
    state = executor.initial_state()
    for identifier in ("init", "expect0", "issue0", "arrive0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    arrive = next(item for item in executor.enabled(state) if item.id == "arrive1")

    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, arrive)
    assert caught.value.code == "PHASE_LAP"


def test_truly_stale_wait_has_defensive_fallback():
    executor = Executor(load("safe_handoff.json"))
    state = executor.initial_state()
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "init"))
    state.barriers["bar"].generation = 2
    state.barriers["bar"].publications = {1}
    wait = next(item for item in executor.enabled(state) if item.id == "wait")
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, wait)
    assert caught.value.code == "STALE_PHASE_WAIT"


def test_reinit_phase_ids_and_visibility_are_lifecycle_scoped():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    names = ("a_writer", "b_reader", "c_arrival")
    raw["actors"] = [
        {"id": name, "kind": "PARTITION", "participants": [index]}
        for index, name in enumerate(names)
    ]
    raw["target"]["cta_threads"] = 3
    raw["resources"][0]["participants"] = list(names)
    raw["events"] = [
        {"id": "init0", "type": "MBarrierInit", "actor": "a_writer", "seq": 0, "resource": "bar", "lifecycle_epoch": 0, "count": 1},
        {"id": "write", "type": "MemWrite", "actor": "a_writer", "seq": 1, "region": "tile"},
        {"id": "arrive0", "type": "MBarrierArrive", "actor": "a_writer", "seq": 2, "resource": "bar", "lifecycle_epoch": 0, "count": 1, "generation": 0},
        {"id": "wait0", "type": "MBarrierWaitParity", "actor": "a_writer", "seq": 3, "resource": "bar", "lifecycle_epoch": 0, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("writer", 0, 0)},
        {"id": "invalidate", "type": "MBarrierInvalidate", "actor": "a_writer", "seq": 4, "resource": "bar", "lifecycle_epoch": 0},
        {"id": "init1", "type": "MBarrierInit", "actor": "a_writer", "seq": 5, "resource": "bar", "lifecycle_epoch": 1, "count": 1},
        {"id": "arrive1", "type": "MBarrierArrive", "actor": "c_arrival", "seq": 0, "resource": "bar", "lifecycle_epoch": 1, "count": 1, "generation": 0, "depends_on": ["init1"]},
        {"id": "wait1", "type": "MBarrierWaitParity", "actor": "b_reader", "seq": 0, "resource": "bar", "lifecycle_epoch": 1, "generation": 0, "raw_parity": 0, "parity_evidence": structured_evidence("reader", 0, 1), "depends_on": ["init1"]},
        {"id": "read", "type": "MemRead", "actor": "b_reader", "seq": 1, "region": "tile"},
    ]
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    phases = [
        step["transition"]
        for step in result["artifacts"]["canonical_trace"]
        if step["kind"] == "PHASE_COMPLETION"
    ]
    assert phases == [
        "completion:mbarrier-phase:bar:l00000000:g00000000",
        "completion:mbarrier-phase:bar:l00000001:g00000000",
    ]


def test_default_checker_engine_is_maximal(monkeypatch):
    def exhaustive_must_not_run(*_args, **_kwargs):
        raise AssertionError("default maximal engine called explore_all")

    monkeypatch.setattr(Executor, "explore_all", exhaustive_must_not_run)
    result = check_module(load("safe_handoff.json"))
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["engine"] == "maximal"
    assert result["statistics"]["explored_steps"] > 0
    assert result["statistics"]["explored_states"] == 0


@pytest.mark.parametrize("engine", ["exhaustive", "crosscheck"])
def test_oracle_engines_are_available(engine: str):
    result = check_module(load("safe_handoff.json"), engine=engine)
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["engine"] == engine
    assert result["statistics"]["explored_states"] > 0


def test_crosscheck_accepts_different_boundary_prefixes_for_same_sync_risk():
    result = check_module(scheduler_race("a_main", "z_arrival"), engine="crosscheck")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "STALE_GENERATION"
    comparison = result["artifacts"]["crosscheck"]
    assert comparison["maximal"]["verdict"] == "SYNC_HAZARD"
    assert comparison["exhaustive"]["verdict"] == "SYNC_HAZARD"
    assert comparison["maximal"]["boundaries"] != comparison["exhaustive"]["boundaries"]

def test_cli_output_is_byte_deterministic():
    path = FIXTURES / "safe_handoff.json"
    command = [sys.executable, "-m", "pipeguard", "check", str(path)]
    first = subprocess.run(command, check=True, capture_output=True).stdout
    second = subprocess.run(command, check=True, capture_output=True).stdout
    assert first == second
    document = json.loads(first)
    assert document["verdict"] == "SAFE"


@pytest.mark.parametrize("command", ["validate", "normalize", "check", "explain"])
def test_all_cli_commands_emit_json(command: str):
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", command, str(FIXTURES / "safe_handoff.json")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert isinstance(json.loads(process.stdout), dict)



def test_cli_pretty_and_statistics_are_real_interface_options():
    path = FIXTURES / "safe_handoff.json"
    base = [sys.executable, "-m", "pipeguard", "check", str(path)]
    compact = subprocess.run(base, check=True, capture_output=True, text=True).stdout
    pretty = subprocess.run(base + ["--pretty"], check=True, capture_output=True, text=True).stdout
    compact_document = json.loads(compact)
    pretty_document = json.loads(pretty)

    assert compact_document == pretty_document
    assert "\n  \"" not in compact
    assert "\n  \"" in pretty
    statistics = pretty_document["statistics"]
    assert statistics["engine"] == "maximal"
    assert {key: statistics[key] for key in ("actors", "events", "resources", "regions")} == {
        "actors": 2, "events": 5, "resources": 1, "regions": 1,
    }
    assert statistics["explored_steps"] > 0


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_cli_check_engine_modes(engine: str):
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", "check", str(FIXTURES / "safe_handoff.json"), "--engine", engine],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    assert document["verdict"] == "SAFE"
    assert document["statistics"]["engine"] == engine


# Incremental held-anchor engine coverage.
def independent_read_module(actor_count: int, events_per_actor: int = 1):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [
        {"id": f"actor_{index:02d}", "kind": "LANE", "participants": [index]}
        for index in range(actor_count)
    ]
    raw["target"]["cta_threads"] = 32
    raw["resources"] = []
    raw["events"] = [
        {
            "id": f"read_{actor:02d}_{seq:02d}",
            "type": "MemRead",
            "actor": f"actor_{actor:02d}",
            "seq": seq,
            "region": "tile",
        }
        for actor in range(actor_count)
        for seq in range(events_per_actor)
    ]
    return module_from_dict(raw)


def test_stable_ready_set_orders_multiple_bitmap_levels():
    ready = StableReadySet(257)
    for rank in (256, 64, 129, 0, 63, 128):
        assert ready.add(rank)
    assert not ready.add(64)
    assert ready.discard(129)
    assert [ready.pop_min() for _ in range(5)] == [0, 63, 64, 128, 256]
    assert ready.pop_min() is None


def generated_barrier_module(capacity: int, observer_count: int):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    observers = [f"observer_{index}" for index in range(observer_count)]
    raw["actors"] = [
        {"id": "producer", "kind": "PARTITION", "participants": list(range(capacity))},
        *[
            {
                "id": actor,
                "kind": "PARTITION",
                "participants": [capacity + index],
            }
            for index, actor in enumerate(observers)
        ],
    ]
    raw["target"]["cta_threads"] = capacity + observer_count
    raw["resources"][0]["participants"] = ["producer", *observers]
    raw["regions"] = []
    raw["events"] = [
        {
            "id": "init",
            "type": "MBarrierInit",
            "actor": "producer",
            "seq": 0,
            "resource": "bar",
            "lifecycle_epoch": 0,
            "count": capacity,
        },
        {
            "id": "arrive",
            "type": "MBarrierArrive",
            "actor": "producer",
            "seq": 1,
            "resource": "bar",
            "lifecycle_epoch": 0,
            "generation": 0,
            "count": capacity,
        },
        *[
            {
                "id": f"wait_{index}",
                "type": "MBarrierWaitParity",
                "actor": actor,
                "seq": 0,
                "resource": "bar",
                "lifecycle_epoch": 0,
                "generation": 0,
                "raw_parity": 0,
                "parity_evidence": structured_evidence(f"{actor}:bar", 0),
                "depends_on": ["init"],
            }
            for index, actor in enumerate(observers)
        ],
    ]
    return module_from_dict(raw)


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("observer_count", [1, 2])
def test_generated_barrier_models_match_exhaustive_boundaries(capacity, observer_count):
    module = generated_barrier_module(capacity, observer_count)
    maximal = check_module(module, engine="maximal")
    exhaustive = check_module(module, engine="exhaustive")
    assert maximal["verdict"] == exhaustive["verdict"] == "SAFE"
    assert maximal["artifacts"]["boundaries"] == exhaustive["artifacts"]["boundaries"]


def test_transition_delta_and_compact_trace_are_notification_ready():
    module = independent_read_module(1)
    executor = Executor(module)
    state = executor.initial_state()
    transition = executor.enabled(state)[0]
    delta = executor.apply(state, transition)
    assert delta.transition_id == "read_00_00"
    assert delta.kind == "PROGRAM"
    assert delta.advanced_actor == "actor_00"
    assert delta.executed_event == "read_00_00"
    assert delta.changed_resources == frozenset()
    assert state.trace == [{"step": 0, "transition": "read_00_00", "kind": "PROGRAM"}]


def test_incremental_program_stream_first_middle_final_and_end_flush():
    module = independent_read_module(1, events_per_actor=3)
    anchors = _anchor_ids(module)
    engine = IncrementalEngine(Executor(module), anchors)
    streams = build_anchor_streams(module, anchors)
    assert len(streams) == 1
    assert streams[0].anchors == ("read_00_00", "read_00_01", "read_00_02")
    assert streams[0].tokens[-1] is END
    assert not isinstance(END, str)

    canonical = engine.canonical()
    outcomes = engine.analyze_streams()
    assert canonical.error is None and not canonical.stuck
    assert outcomes.error is None and outcomes.stuck_state is None
    assert outcomes.boundaries == [
        {"anchor": "read_00_00", "boundary": ["read_00_00"]},
        {"anchor": "read_00_01", "boundary": ["read_00_01"]},
        {"anchor": "read_00_02", "boundary": ["read_00_02"]},
    ]
    # Three canonical applications plus three releases in the one stream;
    # the final application proves END flushed the last held anchor.
    assert engine.counters.transitions_by_kind == {
        "PROGRAM": 6,
        "TMA_COMPLETION": 0,
        "PHASE_COMPLETION": 0,
        "BARRIER_RELEASE": 0,
        "GROUP_COMPLETION": 0,
        "CHANNEL_HANDOFF": 0,
    }
    assert engine.counters.state_initializations == 2


def test_end_flush_reports_invalid_final_anchor():
    module = lifecycle_module("MBarrierInit", "MBarrierInit")
    engine = IncrementalEngine(Executor(module), _anchor_ids(module))
    outcomes = engine.analyze_streams()
    assert outcomes.error is not None
    assert outcomes.error.code == "REINIT_LIVE"
    assert outcomes.error.event == "e1"
    assert outcomes.boundaries == [
        {"anchor": "e0", "boundary": ["e0"]},
        {"anchor": "e1", "boundary": ["e1"]},
    ]


def test_maximal_uses_no_enabled_scan_or_state_clone(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(load("partial_next_phase.json"), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["artifacts"]["canonical_trace"]
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0
    assert result["statistics"]["replayed_prefix_steps"] == 0


def test_every_real_and_synthetic_anchor_has_one_incremental_boundary():
    module = load("tma_completion_before_arrival.json")
    expected = _anchor_ids(module)
    result = check_module(module, engine="maximal")
    anchors = [item["anchor"] for item in result["artifacts"]["boundaries"]]
    assert result["verdict"] == "SAFE"
    assert anchors == expected
    assert any(anchor.startswith("completion:tma-load:") for anchor in anchors)
    assert any(anchor.startswith("completion:mbarrier-phase:") for anchor in anchors)
    assert len(anchors) == len(set(anchors))


def test_held_tma_completion_exposes_early_next_generation_hazard():
    module = tx_pending_early_next_arrival_module()
    completion = stable_completion_id("issue0", 0)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "ARRIVE_WITH_ZERO_PENDING_COUNT"
    assert result["diagnostics"][0]["event"] == "arrive1"
    prefix = [step["transition"] for step in result["witness"]["reachable_execution_prefix"]]
    assert "issue0" in prefix
    assert completion not in prefix


@pytest.mark.parametrize(("actors", "events_per_actor"), [(1, 3), (2, 2), (3, 1)])
def test_generated_small_models_have_exact_maximal_exhaustive_boundaries(actors, events_per_actor):
    module = independent_read_module(actors, events_per_actor)
    maximal = check_module(module, engine="maximal")
    exhaustive = check_module(module, engine="exhaustive")
    assert maximal["verdict"] == exhaustive["verdict"] == "SAFE"
    assert maximal["artifacts"]["boundaries"] == exhaustive["artifacts"]["boundaries"]


def test_incremental_operation_counts_scale_by_work_not_timing():
    reports = {
        size: check_module(independent_read_module(1, events_per_actor=size), engine="maximal")
        for size in (16, 32)
    }
    for size, result in reports.items():
        statistics = result["statistics"]
        assert result["verdict"] == "SAFE"
        assert statistics["n_ext"] == size
        assert statistics["t_eff"] == 1
        assert statistics["state_initializations"] == 1
        assert statistics["completed_passes"] == 1
        assert statistics["anchor_boundaries"] == size
        assert statistics["analytical_program_only_boundaries"] == size
        assert statistics["transitions_by_kind"]["PROGRAM"] == size
        assert statistics["transition_attempts"] == size
        assert statistics["scheduler_work_units"] <= 10 * size
        assert statistics["worklist_pops"] == statistics["transition_attempts"]
        assert statistics["stale_worklist_pops"] == 0
        assert statistics["full_enabled_scans"] == 0
        assert statistics["state_clones"] == 0
        assert statistics["replayed_prefix_steps"] == 0
    small = reports[16]["statistics"]
    large = reports[32]["statistics"]
    assert large["transition_attempts"] == 2 * small["transition_attempts"]
    assert large["scheduler_work_units"] == 2 * small["scheduler_work_units"]


def test_incremental_statistics_certificate_is_deterministic():
    module = independent_read_module(3, 2)
    first = check_module(module, engine="maximal")["statistics"]
    second = check_module(module, engine="maximal")["statistics"]
    assert first == second
    assert first["certificate"]["stream_counts"] == {
        "program": 3,
        "phase_completion": 0,
        "tma_completion": 0,
        "barrier_release": 0,
        "group_completion": 0,
        "channel_handoff": 0,
        "engine_completion": 0,
        "total": 3,
    }
    assert first["boundary_evaluations"] == 0
    assert first["analytical_program_only_boundaries"] == 6


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_predicated_false_arrival_creates_no_synthetic_phase_anchor(engine):
    module = lifecycle_module("MBarrierInit", "MBarrierArrive")
    raw = json.loads(json.dumps(module.raw))
    raw["events"][1]["predicate"] = False
    module = module_from_dict(raw)
    assert not any("mbarrier-phase" in anchor for anchor in _anchor_ids(module))
    result = check_module(module, engine=engine)
    assert result["verdict"] == "SAFE"
    if engine == "crosscheck":
        assert result["artifacts"]["crosscheck"]["maximal"]["verdict"] == "SAFE"
        assert result["artifacts"]["crosscheck"]["exhaustive"]["verdict"] == "SAFE"


def test_crosscheck_retains_maximal_incremental_certificate():
    result = check_module(load("safe_handoff.json"), engine="crosscheck")
    statistics = result["statistics"]
    assert statistics["certificate"] == statistics["maximal_certificate"]
    assert statistics["full_enabled_scans"] == 0
    assert statistics["state_clones"] == 0
    assert statistics["replayed_prefix_steps"] == 0


# Shared SM90 protocol substrate. New protocol events remain fail-closed until
# their FSM checkpoint moves them into IMPLEMENTED_EVENTS.
def full_visibility_mask() -> dict:
    return {
        "shared": True,
        "global_read": True,
        "global_write": True,
        "tensor_read": True,
        "tensor_write": True,
        "control_only": False,
    }


def test_proxy_protocol_fields_are_structured_and_validated():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    fence = next(event for event in raw["events"] if event["id"] == "read")
    fence.clear()
    fence.update({
        "id": "proxy",
        "type": "ProxyFence",
        "actor": "b_consumer",
        "seq": 1,
        "from_domain": "GENERIC",
        "to_domain": "TMA",
        "scope": "CTA",
        "regions": ["tile"],
        "visibility_mask": full_visibility_mask(),
    })
    module = module_from_dict(raw)
    assert check_module(module)["verdict"] == "SAFE"

    fence["visibility_mask"].pop("tensor_write")
    with pytest.raises(PGIRError) as malformed:
        module_from_dict(raw)
    assert malformed.value.code == "SCHEMA_INVALID"


def test_channel_and_mbarrier_resource_metadata_fail_closed():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["resources"][0].update({
        "kind": "channel",
        "capacity": 2,
        "scope": "CTA",
    })
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "RESOURCE_FIELDS_MISSING"

    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["resources"][0]["capacity"] = 2
    with pytest.raises(PGIRError) as unexpected:
        module_from_dict(raw)
    assert unexpected.value.code == "RESOURCE_FIELDS_UNEXPECTED"


def test_protocol_synthetic_ids_are_stable_and_component_safe():
    identifiers = {
        stable_collective_release_id("bar:a", 3),
        stable_collective_release_id("bar", 3),
        stable_group_id("CP_ASYNC", "owner:a", 2),
        stable_group_id("CP_ASYNC", "owner", 2),
        stable_group_completion_id("WGMMA", "wg:0", 4, "RESULT_READY"),
        stable_channel_handoff_id("channel:0", 1, 5),
    }
    assert len(identifiers) == 6
    assert stable_group_id("CP_ASYNC", "owner:a", 2) in identifiers
    assert all(identifier.startswith(("completion:", "group:")) for identifier in identifiers)


def test_tma_load_accepts_only_complete_gap_free_async_engine_evidence():
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    raw["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    issue = next(event for event in raw["events"] if event["id"] == "issue")
    issue.update({"engine_id": "dma", "engine_seq": 0})
    module = module_from_dict(raw)
    parsed = module.event_map["issue"]
    assert (parsed.engine_id, parsed.engine_seq) == ("dma", 0)
    assert check_module(module)["verdict"] == "SAFE"

    incomplete = json.loads(json.dumps(raw))
    next(event for event in incomplete["events"] if event["id"] == "issue").pop("engine_seq")
    with pytest.raises(PGIRError) as missing:
        module_from_dict(incomplete)
    assert missing.value.code == "ASYNC_ENGINE_EVIDENCE_INCOMPLETE"

    gap = json.loads(json.dumps(raw))
    next(event for event in gap["events"] if event["id"] == "issue")["engine_seq"] = 1
    with pytest.raises(PGIRError) as nonzero:
        module_from_dict(gap)
    assert nonzero.value.code == "ASYNC_ENGINE_SEQUENCE_GAP"

    wrong_kind = json.loads(json.dumps(raw))
    next(event for event in wrong_kind["events"] if event["id"] == "issue")["engine_id"] = "a"
    with pytest.raises(PGIRError) as invalid:
        module_from_dict(wrong_kind)
    assert invalid.value.code == "INVALID_ASYNC_ENGINE"


def collective_handoff_module(kind: str = "cta_barrier", mask: dict | None = None):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [
        {"id": "a_writer", "kind": "PARTITION", "participants": [0]},
        {"id": "b_reader", "kind": "PARTITION", "participants": [1]},
    ]
    raw["target"]["cta_threads"] = 2
    raw["resources"] = [{
        "id": "sync",
        "kind": kind,
        "participants": ["a_writer", "b_reader"],
        "capacity": 2,
        "scope": "CTA",
        "visibility_mask": mask or {
            **full_visibility_mask(),
            "global_read": False,
            "global_write": False,
            "tensor_read": False,
            "tensor_write": False,
        },
    }]
    raw["events"] = [
        {"id": "write", "type": "MemWrite", "actor": "a_writer", "seq": 0, "region": "tile"},
        {"id": "arrive_writer", "type": "BarrierArrive", "actor": "a_writer", "seq": 1, "resource": "sync", "count": 1, "invocation_id": 0},
        {"id": "observe_writer", "type": "BarrierObserve", "actor": "a_writer", "seq": 2, "resource": "sync", "invocation_id": 0},
        {"id": "arrive_reader", "type": "BarrierArrive", "actor": "b_reader", "seq": 0, "resource": "sync", "count": 1, "invocation_id": 0},
        {"id": "observe_reader", "type": "BarrierObserve", "actor": "b_reader", "seq": 1, "resource": "sync", "invocation_id": 0},
        {"id": "read", "type": "MemRead", "actor": "b_reader", "seq": 2, "region": "tile"},
    ]
    return module_from_dict(raw)


@pytest.mark.parametrize("kind", ["cta_barrier", "named_barrier"])
def test_collective_barrier_handoff_is_safe_and_crosschecked(kind):
    module = collective_handoff_module(kind)
    result = check_module(module, engine="crosscheck")
    assert result["verdict"] == "SAFE"
    release = stable_collective_release_id("sync", 0)
    anchors = {item["anchor"] for item in result["artifacts"]["boundaries"]}
    assert release in anchors
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": "arrive_writer", "to": release, "kind": "BARRIER_ARRIVE"} in edges
    assert {"from": release, "to": "observe_reader", "kind": "BARRIER_RELEASE"} in edges
    assert result["statistics"]["barrier_release_streams"] == 1


def test_control_only_collective_does_not_publish_shared_data():
    mask = {name: False for name in full_visibility_mask()}
    mask["control_only"] = True
    result = check_module(collective_handoff_module(mask=mask), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda raw: raw["events"].pop(3),
            "MISSING_BARRIER_PARTICIPANT",
        ),
        (
            lambda raw: raw["events"].insert(
                2,
                {"id": "arrive_again", "type": "BarrierArrive", "actor": "a_writer", "seq": 2, "resource": "sync", "count": 1, "invocation_id": 0},
            ),
            "DUPLICATE_BARRIER_ARRIVAL",
        ),
    ],
)
def test_collective_protocol_hazards_have_structured_witness(mutation, code):
    raw = json.loads(json.dumps(collective_handoff_module().raw))
    mutation(raw)
    for actor in {event["actor"] for event in raw["events"]}:
        actor_events = sorted(
            (event for event in raw["events"] if event["actor"] == actor),
            key=lambda event: (event["seq"], event["id"]),
        )
        for seq, event in enumerate(actor_events):
            event["seq"] = seq
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == code
    assert result["witness"]["protocol_state"]


def test_collective_two_invocations_have_ordered_release_stream():
    raw = json.loads(json.dumps(collective_handoff_module().raw))
    for actor in ("a_writer", "b_reader"):
        events = [event for event in raw["events"] if event["actor"] == actor]
        tail = max(event["seq"] for event in events) + 1
        raw["events"].extend([
            {"id": f"arrive2_{actor}", "type": "BarrierArrive", "actor": actor, "seq": tail, "resource": "sync", "count": 1, "invocation_id": 1},
            {"id": f"observe2_{actor}", "type": "BarrierObserve", "actor": actor, "seq": tail + 1, "resource": "sync", "invocation_id": 1},
        ])
    module = module_from_dict(raw)
    result = check_module(module, engine="crosscheck")
    assert result["verdict"] == "SAFE"
    releases = [
        item["anchor"] for item in result["artifacts"]["boundaries"]
        if item["anchor"].startswith("completion:barrier-release:")
    ]
    assert releases == [
        stable_collective_release_id("sync", 0),
        stable_collective_release_id("sync", 1),
    ]


def test_collective_maximal_uses_delta_ready_index(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from collective maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(collective_handoff_module(), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


def test_collective_release_replay_is_failure_atomic():
    executor = Executor(collective_handoff_module())
    state = executor.initial_state()
    for identifier in ("write", "arrive_writer", "arrive_reader"):
        transition = next(item for item in executor.enabled(state) if item.id == identifier)
        executor.apply(state, transition)
    release = next(item for item in executor.enabled(state) if item.kind == "BARRIER_RELEASE")
    executor.apply(state, release)
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, release)
    assert caught.value.code == "INVALID_BARRIER_RELEASE_TRANSITION"
    assert executor.fingerprint(state) == before


def test_collective_future_invocation_is_rejected():
    raw = json.loads(json.dumps(collective_handoff_module().raw))
    next(event for event in raw["events"] if event["id"] == "arrive_writer")["invocation_id"] = 1
    for engine in ("maximal", "exhaustive", "crosscheck"):
        result = check_module(module_from_dict(raw), engine=engine)
        assert result["verdict"] == "SYNC_HAZARD"
        assert result["diagnostics"][0]["code"] == "FUTURE_BARRIER_INVOCATION"


def test_collective_resource_and_arrival_require_exact_lane_contract():
    raw = json.loads(json.dumps(collective_handoff_module().raw))
    raw["resources"][0]["capacity"] = 1
    with pytest.raises(PGIRError) as capacity:
        module_from_dict(raw)
    assert capacity.value.code == "COLLECTIVE_CAPACITY_MISMATCH"

    raw = json.loads(json.dumps(collective_handoff_module().raw))
    next(event for event in raw["events"] if event["id"] == "arrive_writer")["count"] = 2
    with pytest.raises(PGIRError) as arrival:
        module_from_dict(raw)
    assert arrival.value.code == "COLLECTIVE_ARRIVAL_COUNT_MISMATCH"

    raw = json.loads(json.dumps(collective_handoff_module().raw))
    raw["target"]["cta_threads"] = 3
    with pytest.raises(PGIRError) as incomplete:
        module_from_dict(raw)
    assert incomplete.value.code == "CTA_PARTICIPANT_SET_INCOMPLETE"

    raw = json.loads(json.dumps(collective_handoff_module().raw))
    raw["resources"][0]["visibility_mask"]["control_only"] = True
    with pytest.raises(PGIRError) as mixed:
        module_from_dict(raw)
    assert mixed.value.code == "INVALID_VISIBILITY_MASK"


def cp_async_module(
    group_count: int = 1,
    wait_group_count: int = 0,
    overwrite_group: int | None = 0,
    empty_groups: bool = False,
):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [
        {"id": "owner", "kind": "PARTITION", "participants": [0]},
    ]
    raw["target"]["cta_threads"] = 1
    raw["resources"] = [{
        "id": "copies",
        "kind": "async_group",
        "participants": ["owner"],
        "capacity": 8,
        "protocol": "CP_ASYNC",
        "scope": "CTA",
    }]
    raw["regions"] = [{
        "id": "source",
        "allocation_id": "global-source",
        "address_space": "GLOBAL",
        "offset": 0,
        "extent": 64,
    }] + [
        {
            "id": f"tile{group}",
            "allocation_id": "shared",
            "address_space": "SHARED",
            "offset": group * 16,
            "extent": 16,
        }
        for group in range(group_count)
    ]
    events = []
    for group in range(group_count):
        if not empty_groups:
            events.append({
                "id": f"issue{group}",
                "type": "CpAsyncIssue",
                "actor": "owner",
                "seq": len(events),
                "resource": "copies",
                "source_region": "source",
                "destination_region": f"tile{group}",
                "bytes": 16,
                "group_owner": "owner",
            })
        events.append({
            "id": f"commit{group}",
            "type": "CpAsyncCommitGroup",
            "actor": "owner",
            "seq": len(events),
            "resource": "copies",
            "group_owner": "owner",
            "group_seq": group,
        })
    events.append({
        "id": "wait",
        "type": "CpAsyncWaitGroup",
        "actor": "owner",
        "seq": len(events),
        "resource": "copies",
        "group_owner": "owner",
        "wait_group_count": wait_group_count,
    })
    if overwrite_group is not None:
        events.append({
            "id": "overwrite",
            "type": "MemWrite",
            "actor": "owner",
            "seq": len(events),
            "region": f"tile{overwrite_group}",
        })
    raw["events"] = events
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_cp_async_wait_zero_closes_copy_window(engine):
    module = cp_async_module()
    result = check_module(module, engine=engine)
    completion = stable_group_completion_id("CP_ASYNC", "owner", 0, "GROUP_COMPLETE")
    group = stable_group_id("CP_ASYNC", "owner", 0)
    assert result["verdict"] == "SAFE"
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": "issue0", "to": group, "kind": "ISSUE_TO_GROUP"} in edges
    assert {"from": group, "to": completion, "kind": "GROUP_TO_COMPLETION"} in edges
    assert {"from": completion, "to": "wait", "kind": "COMPLETION_TO_WAIT"} in edges
    assert completion in {item["anchor"] for item in result["artifacts"]["boundaries"]}


def test_cp_async_empty_commit_is_trivially_complete_without_async_anchor():
    module = cp_async_module(empty_groups=True, overwrite_group=None)
    result = check_module(module, engine="crosscheck")
    completion = stable_group_completion_id("CP_ASYNC", "owner", 0, "GROUP_COMPLETE")
    group = stable_group_id("CP_ASYNC", "owner", 0)
    assert result["verdict"] == "SAFE"
    assert completion not in {item["anchor"] for item in result["artifacts"]["boundaries"]}
    assert {"from": group, "to": completion, "kind": "GROUP_TO_COMPLETION"} in result["artifacts"]["pdg"]["edges"]
    assert result["statistics"]["group_completion_streams"] == 0


def test_cp_async_wait_group_one_observes_only_older_groups():
    safe = check_module(cp_async_module(2, wait_group_count=1, overwrite_group=0), engine="crosscheck")
    assert safe["verdict"] == "SAFE"

    hazard = check_module(cp_async_module(2, wait_group_count=1, overwrite_group=1), engine="crosscheck")
    assert hazard["verdict"] == "DATA_HAZARD"
    latest = stable_group_completion_id("CP_ASYNC", "owner", 1, "GROUP_COMPLETE")
    assert not any(
        edge["from"] == latest and edge["to"] == "wait"
        for edge in hazard["artifacts"]["pdg"]["edges"]
    )


def test_cp_async_missing_commit_and_wait_before_commit_fail_closed():
    raw = json.loads(json.dumps(cp_async_module().raw))
    raw["events"] = [raw["events"][0]]
    missing = check_module(module_from_dict(raw))
    assert missing["verdict"] == "SYNC_HAZARD"
    assert missing["diagnostics"][0]["code"] == "MISSING_CP_ASYNC_COMMIT"

    raw = json.loads(json.dumps(cp_async_module().raw))
    raw["events"] = [raw["events"][0], raw["events"][2]]
    raw["events"][1]["seq"] = 1
    early = check_module(module_from_dict(raw), engine="crosscheck")
    assert early["verdict"] == "SYNC_HAZARD"
    assert early["diagnostics"][0]["code"] == "WAIT_BEFORE_COMMIT"


def test_cp_async_group_validation_rejects_owner_sequence_and_capacity_ambiguity():
    raw = json.loads(json.dumps(cp_async_module().raw))
    raw["actors"].append({"id": "other", "kind": "PARTITION", "participants": [0]})
    next(event for event in raw["events"] if event["id"] == "issue0")["group_owner"] = "other"
    with pytest.raises(PGIRError) as owner:
        module_from_dict(raw)
    assert owner.value.code == "GROUP_OWNER_MISMATCH"

    raw = json.loads(json.dumps(cp_async_module().raw))
    next(event for event in raw["events"] if event["id"] == "commit0")["group_seq"] = 1
    with pytest.raises(PGIRError) as sequence:
        module_from_dict(raw)
    assert sequence.value.code == "COMMIT_GROUP_SEQUENCE_GAP"

    raw = json.loads(json.dumps(cp_async_module().raw))
    raw["resources"][0]["capacity"] = 9
    with pytest.raises(PGIRError) as capacity:
        module_from_dict(raw)
    assert capacity.value.code == "CP_ASYNC_CAPACITY_UNSUPPORTED"

    raw = json.loads(json.dumps(cp_async_module().raw))
    raw["events"][-2]["wait_group_count"] = 8
    with pytest.raises(PGIRError) as wait_count:
        module_from_dict(raw)
    assert wait_count.value.code == "WAIT_GROUP_COUNT_OUT_OF_RANGE"


def test_cp_async_completion_replay_is_failure_atomic():
    executor = Executor(cp_async_module(overwrite_group=None))
    state = executor.initial_state()
    for identifier in ("issue0", "commit0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    completion = next(item for item in executor.enabled(state) if item.kind == "GROUP_COMPLETION")
    executor.apply(state, completion)
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, completion)
    assert caught.value.code == "INVALID_GROUP_COMPLETION_TRANSITION"
    assert executor.fingerprint(state) == before


def test_cp_async_group_completion_stream_is_ordered_and_delta_driven(monkeypatch):
    module = cp_async_module(2, wait_group_count=0, overwrite_group=1)
    completion_ids = [
        stable_group_completion_id("CP_ASYNC", "owner", group, "GROUP_COMPLETE")
        for group in range(2)
    ]
    stream = next(
        item for item in build_anchor_streams(module, _anchor_ids(module))
        if item.kind == "GROUP_COMPLETION"
    )
    assert stream.anchors == tuple(completion_ids)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from cp.async maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["group_completion_streams"] == 1
    assert result["statistics"]["group_notifications"] > 0
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


def test_held_cp_async_completion_exposes_group_capacity_violation():
    raw = json.loads(json.dumps(cp_async_module(2, wait_group_count=0, overwrite_group=None).raw))
    raw["resources"][0]["capacity"] = 1
    module = module_from_dict(raw)
    for engine in ("maximal", "exhaustive", "crosscheck"):
        result = check_module(module, engine=engine)
        assert result["verdict"] == "SYNC_HAZARD"
        assert result["diagnostics"][0]["code"] == "COMMIT_GROUP_CAPACITY_EXCEEDED"


def test_cp_async_wait_does_not_publish_to_another_actor_via_dependency():
    raw = json.loads(json.dumps(cp_async_module(overwrite_group=None).raw))
    raw["actors"].append({"id": "reader", "kind": "PARTITION", "participants": [1]})
    raw["target"]["cta_threads"] = 2
    raw["resources"][0]["participants"].append("reader")
    raw["events"].append({
        "id": "cross_actor_read",
        "type": "MemRead",
        "actor": "reader",
        "seq": 0,
        "depends_on": ["wait"],
        "region": "tile0",
    })
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"


def cp_async_linear_module(size: int):
    raw = json.loads(json.dumps(cp_async_module(size, overwrite_group=None).raw))
    raw["resources"][0]["capacity"] = 1
    original = {event["id"]: event for event in raw["events"]}
    raw["events"] = []
    for group in range(size):
        issue = original[f"issue{group}"]
        commit = original[f"commit{group}"]
        issue["seq"] = 3 * group
        commit["seq"] = 3 * group + 1
        raw["events"].extend([
            issue,
            commit,
            {
                "id": f"wait{group}",
                "type": "CpAsyncWaitGroup",
                "actor": "owner",
                "seq": 3 * group + 2,
                "resource": "copies",
                "group_owner": "owner",
                "wait_group_count": 0,
            },
        ])
    return module_from_dict(raw)


def test_cp_async_incremental_work_scales_linearly_by_count():
    reports = {
        size: check_module(cp_async_linear_module(size), engine="maximal")
        for size in (16, 32)
    }
    for size, result in reports.items():
        statistics = result["statistics"]
        assert result["verdict"] == "SAFE"
        assert statistics["n_ext"] == 4 * size
        assert statistics["t_eff"] == 2
        assert statistics["state_initializations"] == 3
        assert statistics["completed_passes"] == 3
        assert statistics["transition_attempts"] == 12 * size
        assert statistics["full_enabled_scans"] == 0
        assert statistics["state_clones"] == 0
        assert statistics["replayed_prefix_steps"] == 0
    small = reports[16]["statistics"]
    large = reports[32]["statistics"]
    assert large["transition_attempts"] == 2 * small["transition_attempts"]
    assert large["scheduler_work_units"] <= 2 * small["scheduler_work_units"] + 16


def wgmma_module(
    wait_operands: bool = True,
    wait_result: bool = True,
    overwrite_operand: bool = True,
    read_accumulator: bool = True,
):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [{
        "id": "wg",
        "kind": "WARPGROUP",
        "participants": list(range(128)),
    }]
    raw["target"]["cta_threads"] = 128
    raw["resources"] = [{
        "id": "mma",
        "kind": "async_group",
        "participants": ["wg"],
        "capacity": 8,
        "protocol": "WGMMA",
        "scope": "CTA",
    }]
    raw["regions"] = [
        {
            "id": "operand",
            "allocation_id": "shared-operand",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 64,
        },
        {
            "id": "accumulator",
            "allocation_id": "register-accumulator",
            "address_space": "REGISTER",
            "offset": 0,
            "extent": 64,
        },
    ]
    events = [
        {
            "id": "fence",
            "type": "WgmmaFenceOperand",
            "actor": "wg",
            "seq": 0,
            "resource": "mma",
            "group_owner": "wg",
            "regions": ["operand"],
        },
        {
            "id": "mma_issue",
            "type": "WgmmaIssue",
            "actor": "wg",
            "seq": 1,
            "resource": "mma",
            "group_owner": "wg",
            "regions": ["operand"],
            "destination_region": "accumulator",
        },
        {
            "id": "mma_commit",
            "type": "WgmmaCommitGroup",
            "actor": "wg",
            "seq": 2,
            "resource": "mma",
            "group_owner": "wg",
            "group_seq": 0,
        },
    ]
    if wait_operands:
        events.append({
            "id": "wait_operands",
            "type": "WgmmaWaitGroup",
            "actor": "wg",
            "seq": len(events),
            "resource": "mma",
            "group_owner": "wg",
            "wait_group_count": 0,
            "completion_kind": "OPERANDS_RELEASED",
        })
    if overwrite_operand:
        events.append({
            "id": "operand_overwrite",
            "type": "MemWrite",
            "actor": "wg",
            "seq": len(events),
            "region": "operand",
        })
    if wait_result:
        events.append({
            "id": "wait_result",
            "type": "WgmmaWaitGroup",
            "actor": "wg",
            "seq": len(events),
            "resource": "mma",
            "group_owner": "wg",
            "wait_group_count": 0,
            "completion_kind": "RESULT_READY",
        })
    if read_accumulator:
        events.append({
            "id": "accumulator_read",
            "type": "MemRead",
            "actor": "wg",
            "seq": len(events),
            "region": "accumulator",
        })
    raw["events"] = events
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_wgmma_typed_endpoints_close_their_own_windows(engine):
    result = check_module(wgmma_module(), engine=engine)
    assert result["verdict"] == "SAFE"
    operands = stable_group_completion_id("WGMMA", "wg", 0, "OPERANDS_RELEASED")
    result_ready = stable_group_completion_id("WGMMA", "wg", 0, "RESULT_READY")
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": operands, "to": "wait_operands", "kind": "COMPLETION_TO_WAIT"} in edges
    assert {"from": result_ready, "to": "wait_result", "kind": "COMPLETION_TO_WAIT"} in edges
    anchors = {item["anchor"] for item in result["artifacts"]["boundaries"]}
    assert {operands, result_ready}.issubset(anchors)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_wgmma_result_wait_publishes_to_member_lane_actor(engine):
    raw = json.loads(json.dumps(wgmma_module(overwrite_operand=False).raw))
    raw["actors"].append({"id": "lane0", "kind": "LANE", "participants": [0]})
    read = next(event for event in raw["events"] if event["id"] == "accumulator_read")
    read.update({"actor": "lane0", "seq": 0, "depends_on": ["wait_result"]})
    assert check_module(module_from_dict(raw), engine=engine)["verdict"] == "SAFE"


def test_wgmma_result_wait_does_not_publish_outside_owner_warpgroup():
    raw = json.loads(json.dumps(wgmma_module(overwrite_operand=False).raw))
    raw["target"]["cta_threads"] = 129
    raw["actors"].append({"id": "outsider", "kind": "LANE", "participants": [128]})
    read = next(event for event in raw["events"] if event["id"] == "accumulator_read")
    read.update({"actor": "outsider", "seq": 0, "depends_on": ["wait_result"]})
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"


def test_wgmma_result_ready_does_not_release_shared_operands():
    result = check_module(
        wgmma_module(wait_operands=False, wait_result=True, read_accumulator=False),
        engine="crosscheck",
    )
    assert result["verdict"] == "DATA_HAZARD"
    operand_effect = next(
        effect for effect in result["witness"]["effects"]
        if effect["id"] == "effect:mma_issue:operand:0"
    )
    assert operand_effect["window"][1].endswith("operands_released")


def test_wgmma_operands_released_does_not_make_result_ready():
    result = check_module(
        wgmma_module(wait_result=False, overwrite_operand=False, read_accumulator=True),
        engine="crosscheck",
    )
    assert result["verdict"] == "DATA_HAZARD"
    accumulator = next(
        effect for effect in result["witness"]["effects"]
        if effect["id"] == "effect:mma_issue:accumulator"
    )
    assert accumulator["window"][1].endswith("result_ready")


def test_wgmma_missing_operand_fence_has_protocol_witness():
    raw = json.loads(json.dumps(wgmma_module().raw))
    raw["events"] = [event for event in raw["events"] if event["id"] != "fence"]
    for seq, event in enumerate(raw["events"]):
        event["seq"] = seq
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "WGMMA_OPERAND_FENCE_MISSING"
    assert result["witness"]["protocol_state"]["missing_operand_fence_regions"] == ["operand"]


def test_wgmma_requires_complete_warpgroup_and_typed_wait_endpoint():
    raw = json.loads(json.dumps(wgmma_module().raw))
    raw["actors"][0]["participants"] = list(range(127))
    with pytest.raises(PGIRError) as participants:
        module_from_dict(raw)
    assert participants.value.code == "INVALID_WGMMA_PARTICIPANTS"

    raw = json.loads(json.dumps(wgmma_module().raw))
    next(event for event in raw["events"] if event["id"] == "wait_result")["completion_kind"] = "GROUP_COMPLETE"
    with pytest.raises(PGIRError) as endpoint:
        module_from_dict(raw)
    assert endpoint.value.code == "INVALID_WGMMA_WAIT_ENDPOINT"


def test_wgmma_issue_accepts_register_operand_without_proxy_fencing_it():
    raw = json.loads(json.dumps(wgmma_module().raw))
    raw["regions"].append({
        "id": "register_operand",
        "allocation_id": "register-input",
        "address_space": "REGISTER",
        "offset": 0,
        "extent": 64,
    })
    next(event for event in raw["events"] if event["id"] == "mma_issue")["regions"].append("register_operand")
    assert check_module(module_from_dict(raw), engine="crosscheck")["verdict"] == "SAFE"


def test_wgmma_accepts_aligned_multi_warpgroup_partition_owner():
    raw = json.loads(json.dumps(wgmma_module().raw))
    raw["target"]["cta_threads"] = 256
    raw["actors"][0].update({
        "kind": "PARTITION",
        "participants": list(range(256)),
    })
    assert check_module(module_from_dict(raw), engine="crosscheck")["verdict"] == "SAFE"


def test_wgmma_empty_group_is_fail_closed_without_ptx_contract():
    raw = json.loads(json.dumps(wgmma_module().raw))
    raw["events"] = [
        next(event for event in raw["events"] if event["id"] == "mma_commit"),
        next(event for event in raw["events"] if event["id"] == "wait_result"),
    ]
    for seq, event in enumerate(raw["events"]):
        event["seq"] = seq
    with pytest.raises(PGIRError) as empty:
        module_from_dict(raw)
    assert empty.value.code == "UNSUPPORTED_EMPTY_WGMMA_GROUP"


def test_wgmma_endpoint_replay_is_failure_atomic():
    executor = Executor(wgmma_module(
        wait_operands=False,
        wait_result=False,
        overwrite_operand=False,
        read_accumulator=False,
    ))
    state = executor.initial_state()
    for identifier in ("fence", "mma_issue", "mma_commit"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    endpoint = next(
        item for item in executor.enabled(state)
        if item.kind == "GROUP_COMPLETION" and item.completion_kind == "RESULT_READY"
    )
    executor.apply(state, endpoint)
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, endpoint)
    assert caught.value.code == "INVALID_GROUP_COMPLETION_TRANSITION"
    assert executor.fingerprint(state) == before


def test_wgmma_maximal_uses_two_delta_driven_endpoint_streams(monkeypatch):
    module = wgmma_module()
    streams = [
        stream for stream in build_anchor_streams(module, _anchor_ids(module))
        if stream.kind == "GROUP_COMPLETION"
    ]
    assert len(streams) == 2

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from WGMMA maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["group_completion_streams"] == 2
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


def test_wgmma_capacity_counts_every_unfinished_endpoint():
    raw = json.loads(json.dumps(wgmma_module(
        wait_operands=False,
        overwrite_operand=False,
        read_accumulator=False,
    ).raw))
    raw["resources"][0]["capacity"] = 1
    wait_result = next(event for event in raw["events"] if event["id"] == "wait_result")
    wait_result["seq"] = 3
    raw["events"].extend([
        {
            "id": "mma_issue1",
            "type": "WgmmaIssue",
            "actor": "wg",
            "seq": 4,
            "resource": "mma",
            "group_owner": "wg",
            "regions": ["operand"],
            "destination_region": "accumulator",
        },
        {
            "id": "mma_commit1",
            "type": "WgmmaCommitGroup",
            "actor": "wg",
            "seq": 5,
            "resource": "mma",
            "group_owner": "wg",
            "group_seq": 1,
        },
    ])
    module = module_from_dict(raw)
    for engine in ("maximal", "exhaustive", "crosscheck"):
        result = check_module(module, engine=engine)
        assert result["verdict"] == "SYNC_HAZARD"
        assert result["diagnostics"][0]["code"] == "COMMIT_GROUP_CAPACITY_EXCEEDED"


def tma_store_module(
    read_only_wait: bool = True,
    global_wait: bool = True,
    overwrite_source: bool = True,
    read_destination: bool = True,
):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [{"id": "store_owner", "kind": "PARTITION", "participants": [0]}]
    raw["target"]["cta_threads"] = 1
    raw["resources"] = [{
        "id": "stores", "kind": "async_group", "participants": ["store_owner"],
        "capacity": 8, "protocol": "TMA_STORE", "scope": "CTA",
    }]
    raw["regions"] = [
        {"id": "store_source", "allocation_id": "shared-source", "address_space": "SHARED", "offset": 0, "extent": 64},
        {"id": "store_destination", "allocation_id": "global-destination", "address_space": "GLOBAL", "offset": 0, "extent": 64},
    ]
    events = [
        {
            "id": "store_issue", "type": "TmaStoreIssue", "actor": "store_owner", "seq": 0,
            "resource": "stores", "group_owner": "store_owner", "source_region": "store_source",
            "destination_region": "store_destination", "bytes": 64,
        },
        {
            "id": "store_commit", "type": "TmaStoreCommitGroup", "actor": "store_owner", "seq": 1,
            "resource": "stores", "group_owner": "store_owner", "group_seq": 0,
        },
    ]
    if read_only_wait:
        events.append({
            "id": "store_wait_read", "type": "TmaStoreWaitGroup", "actor": "store_owner", "seq": len(events),
            "resource": "stores", "group_owner": "store_owner", "wait_group_count": 0, "wait_mode": "READ_ONLY",
        })
    if global_wait and not read_only_wait:
        events.append({
            "id": "store_wait_global", "type": "TmaStoreWaitGroup", "actor": "store_owner", "seq": len(events),
            "resource": "stores", "group_owner": "store_owner", "wait_group_count": 0, "wait_mode": "GLOBAL",
        })
    if overwrite_source:
        events.append({"id": "store_source_overwrite", "type": "MemWrite", "actor": "store_owner", "seq": len(events), "region": "store_source"})
    if global_wait and read_only_wait:
        events.append({
            "id": "store_wait_global", "type": "TmaStoreWaitGroup", "actor": "store_owner", "seq": len(events),
            "resource": "stores", "group_owner": "store_owner", "wait_group_count": 0, "wait_mode": "GLOBAL",
        })
    if read_destination:
        events.append({"id": "store_destination_read", "type": "MemRead", "actor": "store_owner", "seq": len(events), "region": "store_destination"})
    raw["events"] = events
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_tma_store_typed_waits_close_source_and_global_windows(engine):
    result = check_module(tma_store_module(), engine=engine)
    assert result["verdict"] == "SAFE"
    source = stable_group_completion_id("TMA_STORE", "store_owner", 0, "SOURCE_RELEASED")
    complete = stable_group_completion_id("TMA_STORE", "store_owner", 0, "GLOBAL_COMPLETE")
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": source, "to": complete, "kind": "ENDPOINT_ORDER"} in edges
    assert {"from": source, "to": "store_wait_read", "kind": "COMPLETION_TO_WAIT"} in edges
    assert {"from": complete, "to": "store_wait_global", "kind": "COMPLETION_TO_WAIT"} in edges


def test_tma_store_read_only_wait_does_not_imply_global_completion():
    result = check_module(tma_store_module(global_wait=False), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    destination = next(effect for effect in result["witness"]["effects"] if effect["id"] == "effect:store_issue:destination")
    assert destination["window"][1].endswith("global_complete")


def test_tma_store_global_wait_also_follows_source_release():
    result = check_module(tma_store_module(read_only_wait=False), engine="crosscheck")
    assert result["verdict"] == "SAFE"


def test_tma_store_missing_and_empty_commit_fail_closed():
    raw = json.loads(json.dumps(tma_store_module().raw))
    raw["events"] = [raw["events"][0]]
    missing = check_module(module_from_dict(raw))
    assert missing["verdict"] == "SYNC_HAZARD"
    assert missing["diagnostics"][0]["code"] == "MISSING_TMA_STORE_COMMIT"

    raw = json.loads(json.dumps(tma_store_module().raw))
    raw["events"] = [raw["events"][1]]
    raw["events"][0]["seq"] = 0
    with pytest.raises(PGIRError) as empty:
        module_from_dict(raw)
    assert empty.value.code == "UNSUPPORTED_EMPTY_TMA_STORE_GROUP"


def test_tma_store_requires_shared_to_global_regions():
    raw = json.loads(json.dumps(tma_store_module().raw))
    next(region for region in raw["regions"] if region["id"] == "store_source")["address_space"] = "GLOBAL"
    with pytest.raises(PGIRError) as region:
        module_from_dict(raw)
    assert region.value.code == "UNSUPPORTED_TMA_STORE_REGION"


def test_tma_store_global_completion_replay_is_failure_atomic():
    executor = Executor(tma_store_module(overwrite_source=False, read_destination=False))
    state = executor.initial_state()
    for identifier in ("store_issue", "store_commit"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    source = next(item for item in executor.enabled(state) if item.kind == "GROUP_COMPLETION" and item.completion_kind == "SOURCE_RELEASED")
    executor.apply(state, source)
    complete = next(item for item in executor.enabled(state) if item.kind == "GROUP_COMPLETION" and item.completion_kind == "GLOBAL_COMPLETE")
    executor.apply(state, complete)
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as replay:
        executor.apply(state, complete)
    assert replay.value.code == "INVALID_GROUP_COMPLETION_TRANSITION"
    assert executor.fingerprint(state) == before


def test_tma_store_maximal_uses_two_delta_driven_endpoint_streams(monkeypatch):
    module = tma_store_module()
    streams = [stream for stream in build_anchor_streams(module, _anchor_ids(module)) if stream.kind == "GROUP_COMPLETION"]
    assert len(streams) == 2

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from TMA-store maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["group_completion_streams"] == 2
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


def channel_module(phases: int = 1, control_only: bool = False):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [
        {"id": "producer", "kind": "PARTITION", "participants": [0]},
        {"id": "consumer", "kind": "PARTITION", "participants": [1]},
    ]
    raw["target"]["cta_threads"] = 2
    mask = {name: False for name in full_visibility_mask()}
    mask["control_only" if control_only else "shared"] = True
    raw["resources"] = [{
        "id": "pipe", "kind": "channel", "participants": ["producer", "consumer"],
        "capacity": 1, "scope": "CTA", "visibility_mask": mask,
    }]
    raw["regions"] = [{"id": "tile", "allocation_id": "shared-tile", "address_space": "SHARED", "offset": 0, "extent": 64}]
    producer_events = [{"id": "create", "type": "ChannelCreate", "actor": "producer", "seq": 0, "resource": "pipe"}]
    consumer_events = []
    for phase in range(phases):
        producer_events.extend([
            {"id": f"acquire{phase}", "type": "ProducerAcquire", "actor": "producer", "seq": 1 + 3 * phase, "resource": "pipe", "slot": 0, "logical_phase": phase},
            {"id": f"produce{phase}", "type": "MemWrite", "actor": "producer", "seq": 2 + 3 * phase, "region": "tile"},
            {"id": f"commit{phase}", "type": "ProducerCommit", "actor": "producer", "seq": 3 + 3 * phase, "resource": "pipe", "slot": 0, "logical_phase": phase},
        ])
        consumer_events.extend([
            {"id": f"consume_wait{phase}", "type": "ConsumerWait", "actor": "consumer", "seq": 3 * phase, "resource": "pipe", "slot": 0, "logical_phase": phase},
            {"id": f"consume{phase}", "type": "MemRead", "actor": "consumer", "seq": 1 + 3 * phase, "region": "tile"},
            {"id": f"release{phase}", "type": "ConsumerRelease", "actor": "consumer", "seq": 2 + 3 * phase, "resource": "pipe", "slot": 0, "logical_phase": phase},
        ])
    raw["events"] = producer_events + consumer_events
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_channel_handoff_and_two_sided_reuse_are_safe(engine):
    module = channel_module(phases=2)
    result = check_module(module, engine=engine)
    assert result["verdict"] == "SAFE"
    handoffs = [stable_channel_handoff_id("pipe", 0, phase) for phase in range(2)]
    anchors = {item["anchor"] for item in result["artifacts"]["boundaries"]}
    assert set(handoffs).issubset(anchors)
    edges = result["artifacts"]["pdg"]["edges"]
    assert {"from": "commit0", "to": handoffs[0], "kind": "CHANNEL_COMMIT"} in edges
    assert {"from": handoffs[0], "to": "consume_wait0", "kind": "CHANNEL_HANDOFF"} in edges
    assert {"from": "release0", "to": "acquire1", "kind": "CHANNEL_RELEASE"} in edges


def test_control_only_channel_orders_execution_but_not_shared_visibility():
    result = check_module(channel_module(control_only=True), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"


def gsan_cross_sm_reduced_module(producer_sem: str, consumer_sem: str, scope: str):
    raw = cluster_dsmem_module(cluster_ctas=2)
    raw["actors"] = [
        {"id": "producer", "kind": "PARTITION", "participants": [0], "cta_rank": 0},
        {"id": "consumer", "kind": "PARTITION", "participants": [0], "cta_rank": 1},
    ]
    raw["regions"] = [{
        "id": "payload", "allocation_id": "global:payload",
        "address_space": "GLOBAL", "offset": 0, "extent": 4,
    }]
    synchronizes = (
        producer_sem in {"release", "acq_rel"}
        and consumer_sem in {"acquire", "acq_rel"}
        and scope in {"gpu", "sys"}
    )
    if not synchronizes:
        raw["resources"] = []
        raw["events"] = [
            {"id": "payload_write", "type": "MemWrite", "actor": "producer", "seq": 0, "region": "payload"},
            {"id": "payload_read", "type": "MemRead", "actor": "consumer", "seq": 0, "region": "payload"},
        ]
        return module_from_dict(raw)
    mask = {name: False for name in full_visibility_mask()}
    mask.update({"dsmem": False, "global_read": True, "global_write": True})
    raw["resources"] = [{
        "id": "atomic_flag", "kind": "channel",
        "participants": ["producer", "consumer"],
        "capacity": 1, "scope": "GPU", "visibility_mask": mask,
    }]
    raw["events"] = [
        {"id": "flag_create", "type": "ChannelCreate", "actor": "producer", "seq": 0, "resource": "atomic_flag"},
        {"id": "flag_acquire_slot", "type": "ProducerAcquire", "actor": "producer", "seq": 1, "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
        {"id": "payload_write", "type": "MemWrite", "actor": "producer", "seq": 2, "region": "payload"},
        {"id": "flag_release", "type": "ProducerCommit", "actor": "producer", "seq": 3, "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
        {"id": "flag_poll", "type": "ConsumerWait", "actor": "consumer", "seq": 0, "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
        {"id": "payload_read", "type": "MemRead", "actor": "consumer", "seq": 1, "region": "payload"},
        {"id": "flag_done", "type": "ConsumerRelease", "actor": "consumer", "seq": 2, "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
    ]
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_gsan_release_acquire_cross_sm_reduced_case_is_safe(engine):
    result = check_module(
        gsan_cross_sm_reduced_module("release", "acquire", "gpu"),
        engine=engine,
    )
    assert result["verdict"] == "SAFE"
    assert any(edge["kind"] == "CHANNEL_HANDOFF" for edge in result["artifacts"]["pdg"]["edges"])


def test_gpu_channel_does_not_publish_dsmem():
    raw = cluster_dsmem_module(control_only=True)
    mask = {name: False for name in full_visibility_mask()}
    mask.update({"dsmem": False, "global_read": True, "global_write": True})
    raw["resources"].append({
        "id": "atomic_flag", "kind": "channel",
        "participants": ["a0", "a1"],
        "capacity": 1, "scope": "GPU", "visibility_mask": mask,
    })
    additions = {
        "a0": [
            {"id": "flag_create", "type": "ChannelCreate", "resource": "atomic_flag"},
            {"id": "flag_acquire", "type": "ProducerAcquire", "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
            {"id": "flag_commit", "type": "ProducerCommit", "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
        ],
        "a1": [
            {"id": "flag_wait", "type": "ConsumerWait", "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
            {"id": "flag_release", "type": "ConsumerRelease", "resource": "atomic_flag", "slot": 0, "logical_phase": 0},
        ],
    }
    reordered = []
    for actor in ("a0", "a1"):
        events = [event for event in raw["events"] if event["actor"] == actor]
        memory_index = next(
            index for index, event in enumerate(events)
            if event["type"] in {"MemRead", "MemWrite"}
        )
        before, after = additions[actor][:-1], additions[actor][-1:]
        events[memory_index:memory_index] = before
        events[memory_index + len(before) + 1:memory_index + len(before) + 1] = after
        for seq, event in enumerate(events):
            event.update({"actor": actor, "seq": seq})
        reordered.extend(events)
    raw["events"] = reordered
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.parametrize(
    ("producer_sem", "consumer_sem", "scope"),
    [
        ("relaxed", "acquire", "gpu"),
        ("relaxed", "acquire", "sys"),
        ("release", "relaxed", "gpu"),
        ("release", "relaxed", "sys"),
        ("release", "acquire", "cta"),
        ("release", "acq_rel", "cta"),
        ("acq_rel", "acquire", "cta"),
        ("acq_rel", "acq_rel", "cta"),
    ],
)
def test_gsan_cross_sm_semantic_and_scope_mismatch_reduced_cases_report_race(
    producer_sem,
    consumer_sem,
    scope,
):
    result = check_module(
        gsan_cross_sm_reduced_module(producer_sem, consumer_sem, scope),
        engine="crosscheck",
    )
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.parametrize(
    ("left", "right"),
    [("MemWrite", "MemRead"), ("MemRead", "MemWrite"), ("MemWrite", "MemWrite")],
    ids=["raw", "war", "waw"],
)
def test_gsan_basic_cross_sm_race_reduced_cases_crosscheck(left, right):
    raw = cluster_dsmem_module(cluster_ctas=2)
    raw["actors"] = [
        {"id": "cta0", "kind": "PARTITION", "participants": [0], "cta_rank": 0},
        {"id": "cta1", "kind": "PARTITION", "participants": [0], "cta_rank": 1},
    ]
    raw["resources"] = []
    raw["regions"] = [{
        "id": "target", "allocation_id": "global:target",
        "address_space": "GLOBAL", "offset": 0, "extent": 4,
    }]
    raw["events"] = [
        {"id": "left", "type": left, "actor": "cta0", "seq": 0, "region": "target"},
        {"id": "right", "type": right, "actor": "cta1", "seq": 0, "region": "target"},
    ]
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


def test_channel_missing_commit_and_release_have_structured_witnesses():
    raw = json.loads(json.dumps(channel_module().raw))
    raw["events"] = [event for event in raw["events"] if event["id"] != "commit0"]
    for actor in ("producer", "consumer"):
        actor_events = [event for event in raw["events"] if event["actor"] == actor]
        for seq, event in enumerate(actor_events):
            event["seq"] = seq
    missing_commit = check_module(module_from_dict(raw))
    assert missing_commit["verdict"] == "SYNC_HAZARD"
    assert missing_commit["diagnostics"][0]["code"] == "MISSING_PRODUCER_COMMIT"

    raw = json.loads(json.dumps(channel_module().raw))
    raw["events"] = [event for event in raw["events"] if event["id"] != "release0"]
    missing_release = check_module(module_from_dict(raw))
    assert missing_release["verdict"] == "SYNC_HAZARD"
    assert missing_release["diagnostics"][0]["code"] == "MISSING_CONSUMER_RELEASE"


def test_channel_wrong_owner_and_slot_fail_closed():
    raw = json.loads(json.dumps(channel_module().raw))
    commit = next(event for event in raw["events"] if event["id"] == "commit0")
    commit.update({"actor": "consumer", "seq": 0, "depends_on": ["produce0"]})
    producer_events = [event for event in raw["events"] if event["actor"] == "producer"]
    for seq, event in enumerate(producer_events):
        event["seq"] = seq
    consumer_events = [commit] + [
        event for event in raw["events"]
        if event["actor"] == "consumer" and event["id"] != "commit0"
    ]
    for seq, event in enumerate(consumer_events):
        event["seq"] = seq
    wrong_owner = check_module(module_from_dict(raw))
    assert wrong_owner["verdict"] == "SYNC_HAZARD"
    assert wrong_owner["diagnostics"][0]["code"] == "CHANNEL_PRODUCER_OWNERSHIP"

    raw = json.loads(json.dumps(channel_module().raw))
    next(event for event in raw["events"] if event["id"] == "acquire0")["slot"] = 1
    with pytest.raises(PGIRError) as slot:
        module_from_dict(raw)
    assert slot.value.code == "CHANNEL_SLOT_OUT_OF_RANGE"


def test_channel_handoff_replay_is_failure_atomic():
    executor = Executor(channel_module())
    state = executor.initial_state()
    for identifier in ("create", "acquire0", "produce0", "commit0"):
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == identifier))
    handoff = next(item for item in executor.enabled(state) if item.kind == "CHANNEL_HANDOFF")
    executor.apply(state, handoff)
    before = executor.fingerprint(state)
    with pytest.raises(ProtocolError) as replay:
        executor.apply(state, handoff)
    assert replay.value.code == "INVALID_CHANNEL_HANDOFF_TRANSITION"
    assert executor.fingerprint(state) == before


def test_channel_maximal_uses_delta_driven_handoff_stream(monkeypatch):
    module = channel_module(phases=2)
    streams = [stream for stream in build_anchor_streams(module, _anchor_ids(module)) if stream.kind == "CHANNEL_HANDOFF"]
    assert len(streams) == 1
    assert streams[0].anchors == tuple(stable_channel_handoff_id("pipe", 0, phase) for phase in range(2))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference scan/clone reached from channel maximal engine")

    monkeypatch.setattr(Executor, "enabled", forbidden)
    monkeypatch.setattr(ExecutionState, "clone", forbidden)
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["channel_handoff_streams"] == 1
    assert result["statistics"]["channel_notifications"] > 0
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


def tma_load_proxy_module(to_domain: str = "TMA", fence_after_issue: bool = False, fence_region: str = "source"):
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    original = sorted(raw["events"], key=lambda event: event["seq"])
    source_write = {"id": "source_write", "type": "MemWrite", "actor": "a", "seq": 0, "region": "source"}
    proxy = {
        "id": "proxy", "type": "ProxyFence", "actor": "a", "seq": 0,
        "from_domain": "GENERIC", "to_domain": to_domain, "scope": "CTA",
        "regions": [fence_region], "visibility_mask": full_visibility_mask(),
    }
    events = [source_write]
    for event in original:
        events.append(event)
        if fence_after_issue and event["id"] == "issue":
            events.append(proxy)
    if not fence_after_issue:
        issue_index = next(index for index, event in enumerate(events) if event["id"] == "issue")
        events.insert(issue_index, proxy)
    for seq, event in enumerate(events):
        event["seq"] = seq
    raw["events"] = events
    return raw


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_directional_proxy_fence_publishes_generic_write_to_tma(engine):
    module = module_from_dict(tma_load_proxy_module())
    result = check_module(module, engine=engine)
    assert result["verdict"] == "SAFE"
    executor = Executor(module)
    state = executor.explore().state
    assert [transfer.to_domain for transfer in state.visibility_transfers] == ["TMA"]


def elected_proxy_tma_store_module(control_only: bool = False):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["target"]["cta_threads"] = 2
    raw["actors"] = [
        {"id": "writer", "kind": "LANE", "participants": [0]},
        {"id": "fencer", "kind": "LANE", "participants": [1]},
    ]
    visibility = full_visibility_mask()
    visibility["control_only"] = control_only
    if control_only:
        visibility.update({
            "shared": False,
            "global_read": False,
            "global_write": False,
            "tensor_read": False,
            "tensor_write": False,
        })
    raw["resources"] = [
        {
            "id": "cta", "kind": "cta_barrier", "participants": ["writer", "fencer"],
            "capacity": 2, "scope": "CTA", "visibility_mask": visibility,
        },
        {
            "id": "stores", "kind": "async_group", "participants": ["fencer"],
            "capacity": 8, "protocol": "TMA_STORE", "scope": "CTA",
        },
    ]
    raw["regions"] = [
        {
            "id": "shared", "allocation_id": "shared", "address_space": "SHARED",
            "offset": 0, "extent": 64,
        },
        {
            "id": "global", "allocation_id": "global", "address_space": "GLOBAL",
            "offset": 0, "extent": 64,
        },
    ]
    raw["events"] = [
        {"id": "write", "type": "MemWrite", "actor": "writer", "seq": 0, "region": "shared"},
        {
            "id": "writer_arrive", "type": "BarrierArrive", "actor": "writer", "seq": 1,
            "resource": "cta", "invocation_id": 0, "count": 1,
        },
        {
            "id": "writer_observe", "type": "BarrierObserve", "actor": "writer", "seq": 2,
            "resource": "cta", "invocation_id": 0,
        },
        {
            "id": "fencer_arrive", "type": "BarrierArrive", "actor": "fencer", "seq": 0,
            "resource": "cta", "invocation_id": 0, "count": 1,
        },
        {
            "id": "fencer_observe", "type": "BarrierObserve", "actor": "fencer", "seq": 1,
            "resource": "cta", "invocation_id": 0,
        },
        {
            "id": "proxy", "type": "ProxyFence", "actor": "fencer", "seq": 2,
            "from_domain": "GENERIC", "to_domain": "TMA", "scope": "CTA",
            "regions": ["shared"], "visibility_mask": full_visibility_mask(),
        },
        {
            "id": "store", "type": "TmaStoreIssue", "actor": "fencer", "seq": 3,
            "resource": "stores", "group_owner": "fencer", "source_region": "shared",
            "destination_region": "global", "bytes": 64,
        },
        {
            "id": "commit", "type": "TmaStoreCommitGroup", "actor": "fencer", "seq": 4,
            "resource": "stores", "group_owner": "fencer", "group_seq": 0,
        },
        {
            "id": "wait", "type": "TmaStoreWaitGroup", "actor": "fencer", "seq": 5,
            "resource": "stores", "group_owner": "fencer", "wait_group_count": 0,
            "wait_mode": "GLOBAL",
        },
    ]
    return module_from_dict(raw)


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_elected_proxy_fence_publishes_collective_shared_writes(engine):
    assert check_module(elected_proxy_tma_store_module(), engine=engine)["verdict"] == "SAFE"


def test_elected_proxy_fence_requires_typed_collective_visibility():
    result = check_module(elected_proxy_tma_store_module(control_only=True), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


def test_proxy_wrong_domain_does_not_publish():
    raw = tma_load_proxy_module()
    next(event for event in raw["events"] if event["id"] == "proxy")["to_domain"] = "ASYNC_COPY"
    assert check_module(module_from_dict(raw), engine="crosscheck")["verdict"] == "DATA_HAZARD"


def test_proxy_after_issue_does_not_publish_retroactively():
    raw = tma_load_proxy_module(fence_after_issue=True)
    assert check_module(module_from_dict(raw), engine="crosscheck")["verdict"] == "DATA_HAZARD"


def test_proxy_region_must_cover_async_source():
    raw = tma_load_proxy_module(fence_region="source_slice")
    source = next(region for region in raw["regions"] if region["id"] == "source")
    raw["regions"].append({
        **source,
        "id": "source_slice",
        "extent": max(1, source["extent"] // 2),
    })
    result = check_module(module_from_dict(raw), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"


def test_proxy_reverse_direction_and_mask_mismatch_fail_closed():
    raw = tma_load_proxy_module()
    proxy = next(event for event in raw["events"] if event["id"] == "proxy")
    proxy.update({"from_domain": "TMA", "to_domain": "GENERIC"})
    with pytest.raises(PGIRError) as direction:
        module_from_dict(raw)
    assert direction.value.code == "UNSUPPORTED_PROXY_DIRECTION"

    raw = tma_load_proxy_module()
    proxy = next(event for event in raw["events"] if event["id"] == "proxy")
    proxy["scope"] = "CLUSTER"
    with pytest.raises(PGIRError) as scope:
        module_from_dict(raw)
    assert scope.value.code == "SCHEMA_INVALID"

    raw = tma_load_proxy_module()
    proxy = next(event for event in raw["events"] if event["id"] == "proxy")
    proxy["visibility_mask"]["global_write"] = False
    with pytest.raises(PGIRError) as mask:
        module_from_dict(raw)
    assert mask.value.code == "PROXY_REGION_MASK_MISMATCH"


def tma_engine_order_module(order: tuple[int, int] | None):
    raw = json.loads((FIXTURES / "tma_completion_before_arrival.json").read_text())
    raw["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    raw["resources"] = [
        {"id": f"bar{index}", "kind": "mbarrier", "participants": ["a"]}
        for index in range(2)
    ]
    raw["regions"] = [
        {
            "id": f"source{index}", "allocation_id": f"global-input-{index}",
            "address_space": "GLOBAL", "offset": 0, "extent": 16,
        }
        for index in range(2)
    ] + [
        {
            "id": f"tile{index}", "allocation_id": f"shared-{index}",
            "address_space": "SHARED", "offset": 0, "extent": 16,
        }
        for index in range(2)
    ]
    events = []
    for index in range(2):
        events.extend([
            {
                "id": f"init{index}", "type": "MBarrierInit", "actor": "a",
                "seq": len(events), "resource": f"bar{index}", "count": 1,
                "lifecycle_epoch": 0,
            },
            {
                "id": f"expect{index}", "type": "MBarrierExpectTx", "actor": "a",
                "seq": len(events) + 1, "resource": f"bar{index}", "bytes": 4,
                "lifecycle_epoch": 0, "generation": 0,
            },
            {
                "id": f"issue{index}", "type": "TmaLoadIssue", "actor": "a",
                "seq": len(events) + 2, "resource": f"bar{index}", "bytes": 4,
                "source_region": f"source{index}", "destination_region": f"tile{index}",
                "lifecycle_epoch": 0, "generation": 0,
            },
            {
                "id": f"arrive{index}", "type": "MBarrierArrive", "actor": "a",
                "seq": len(events) + 3, "resource": f"bar{index}", "count": 1,
                "lifecycle_epoch": 0, "generation": 0,
            },
        ])
    if order is not None:
        for index, engine_seq in enumerate(order):
            next(event for event in events if event["id"] == f"issue{index}").update({
                "engine_id": "dma",
                "engine_seq": engine_seq,
            })
    events.extend([
        {
            "id": "wait1", "type": "MBarrierWaitParity", "actor": "a",
            "seq": len(events), "resource": "bar1", "lifecycle_epoch": 0,
            "generation": 0, "raw_parity": 0,
            "parity_evidence": {
                "derivation_kind": "BARRIER_HISTORY", "logical_stream_id": "a:bar1",
                "lifecycle_epoch": 0, "stream_rank": 0, "polling_coalesced": True,
            },
        },
        {
            "id": "overwrite0", "type": "MemWrite", "actor": "a",
            "seq": len(events) + 1, "region": "source0",
        },
    ])
    raw["events"] = events
    return raw


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_explicit_engine_order_makes_later_observation_cover_prior_completion(engine):
    result = check_module(module_from_dict(tma_engine_order_module((0, 1))), engine=engine)
    first = stable_completion_id("issue0", 0)
    second = stable_completion_id("issue1", 0)
    assert result["verdict"] == "SAFE"
    assert result["artifacts"]["engine_completion_streams"] == {"dma": [first, second]}
    assert {"from": first, "to": second, "kind": "ENGINE_ORDER"} in result["artifacts"]["pdg"]["edges"]
    if engine == "maximal":
        assert result["statistics"]["engine_completion_streams"] == 1
        assert result["statistics"]["tma_completion_streams"] == 0
        assert result["statistics"]["engine_notifications"] > 0
        assert result["statistics"]["full_enabled_scans"] == 0


@pytest.mark.parametrize(
    ("module", "issue_id", "completion"),
    [
        (
            cp_async_module(overwrite_group=None),
            "issue0",
            stable_group_completion_id("CP_ASYNC", "owner", 0, "GROUP_COMPLETE"),
        ),
        (
            wgmma_module(),
            "mma_issue",
            stable_group_completion_id("WGMMA", "wg", 0, "RESULT_READY"),
        ),
        (
            tma_store_module(),
            "store_issue",
            stable_group_completion_id("TMA_STORE", "store_owner", 0, "GLOBAL_COMPLETE"),
        ),
    ],
)
def test_group_protocols_lift_engine_order_evidence_to_terminal_endpoint(module, issue_id, completion):
    raw = json.loads(json.dumps(module.raw))
    raw["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    next(event for event in raw["events"] if event["id"] == issue_id).update({
        "engine_id": "dma",
        "engine_seq": 0,
    })
    result = check_module(module_from_dict(raw), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["artifacts"]["engine_completion_streams"] == {"dma": [completion]}
    assert result["statistics"]["engine_completion_streams"] == 1
    assert result["statistics"]["full_enabled_scans"] == 0


def test_missing_or_reverse_engine_order_preserves_completion_race():
    assert check_module(module_from_dict(tma_engine_order_module(None)), engine="crosscheck")["verdict"] == "DATA_HAZARD"
    assert check_module(module_from_dict(tma_engine_order_module((1, 0))), engine="crosscheck")["verdict"] == "DATA_HAZARD"


def test_engine_order_blocks_forced_out_of_order_completion_atomically():
    module = module_from_dict(tma_engine_order_module((0, 1)))
    executor = Executor(module)
    state = executor.initial_state()
    for event in module.events_by_actor["a"]:
        if event.id == "wait1":
            break
        executor.apply(state, Transition(event.id, "PROGRAM", event=event))
    before = executor.fingerprint(state)
    second = stable_completion_id("issue1", 0)
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, Transition(second, "TMA_COMPLETION"))
    assert caught.value.code == "ENGINE_ORDER_VIOLATION"
    assert executor.fingerprint(state) == before


def test_engine_order_requires_resolved_acyclic_typed_endpoints():
    unresolved = json.loads(json.dumps(cp_async_module(overwrite_group=None).raw))
    unresolved["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    issue = next(event for event in unresolved["events"] if event["id"] == "issue0")
    issue.update({"engine_id": "dma", "engine_seq": 0})
    unresolved["events"] = [event for event in unresolved["events"] if event["id"] != "commit0"]
    for seq, event in enumerate(unresolved["events"]):
        event["seq"] = seq
    with pytest.raises(PGIRError) as missing:
        module_from_dict(unresolved)
    assert missing.value.code == "ASYNC_ENGINE_COMPLETION_UNRESOLVED"

    predicated_off = json.loads(json.dumps(cp_async_module(overwrite_group=None).raw))
    predicated_off["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    issue = next(event for event in predicated_off["events"] if event["id"] == "issue0")
    issue.update({"engine_id": "dma", "engine_seq": 0, "predicate": False})
    with pytest.raises(PGIRError) as inactive:
        module_from_dict(predicated_off)
    assert inactive.value.code == "ASYNC_ENGINE_COMPLETION_UNRESOLVED"

    ambiguous = json.loads(json.dumps(cp_async_module(group_count=2, overwrite_group=None).raw))
    ambiguous["actors"].append({"id": "dma", "kind": "ASYNC_ENGINE", "participants": [0]})
    issue0 = next(event for event in ambiguous["events"] if event["id"] == "issue0")
    issue1 = next(event for event in ambiguous["events"] if event["id"] == "issue1")
    duplicate = {**issue0, "id": "issue0b", "engine_id": "dma", "engine_seq": 2}
    issue0.update({"engine_id": "dma", "engine_seq": 0})
    issue1.update({"engine_id": "dma", "engine_seq": 1})
    commit_index = next(index for index, event in enumerate(ambiguous["events"]) if event["id"] == "commit0")
    ambiguous["events"].insert(commit_index, duplicate)
    for seq, event in enumerate(ambiguous["events"]):
        event["seq"] = seq
    with pytest.raises(PGIRError) as cycle:
        module_from_dict(ambiguous)
    assert cycle.value.code == "AMBIGUOUS_ASYNC_ENGINE_COMPLETION"


def cluster_dsmem_module(control_only: bool = False, cluster_ctas: int = 2):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["schema_version"] = "pgir-v2"
    raw["target"] = {
        "architecture": "sm90", "warp_size": 32, "cta_threads": 1,
        "shared_memory_bytes": 65536, "grid_scope": "single_cluster",
        "cluster_shape": {"x": cluster_ctas, "y": 1, "z": 1},
    }
    raw["assumptions"].update({
        "no_cluster_effects": False,
        "cluster_topology_resolved": True,
        "remote_alias_precision": "EXACT",
    })
    raw["actors"] = [
        {"id": f"a{rank}", "kind": "PARTITION", "participants": [0], "cta_rank": rank}
        for rank in range(cluster_ctas)
    ]
    raw["resources"] = [{
        "id": "cluster", "kind": "cluster_barrier",
        "participants": [f"a{rank}" for rank in range(cluster_ctas)],
        "capacity": cluster_ctas, "scope": "CLUSTER",
        "visibility_mask": {
            "shared": False, "dsmem": not control_only,
            "global_read": False, "global_write": False,
            "tensor_read": False, "tensor_write": False,
            "control_only": control_only,
        },
    }]
    raw["regions"] = [{
        "id": "remote_tile", "allocation_id": "cluster-tile", "address_space": "DSMEM",
        "owner_cta_rank": 0, "offset": 0, "extent": 16,
    }]
    actor_events = {}
    for rank in range(cluster_ctas):
        actor = f"a{rank}"
        items = [
            (f"arrive0_{actor}", "BarrierArrive", 0),
            (f"observe0_{actor}", "BarrierObserve", 0),
        ]
        if rank == 0:
            items.append(("write_a0", "MemWrite", None))
        items.extend([
            (f"arrive1_{actor}", "BarrierArrive", 1),
            (f"observe1_{actor}", "BarrierObserve", 1),
        ])
        if rank == 1:
            items.append(("read_a1", "MemRead", None))
        items.extend([
            (f"arrive2_{actor}", "BarrierArrive", 2),
            (f"observe2_{actor}", "BarrierObserve", 2),
        ])
        actor_events[actor] = items
    events = []
    for actor, items in actor_events.items():
        for seq, (identifier, kind, invocation) in enumerate(items):
            event = {"id": identifier, "type": kind, "actor": actor, "seq": seq}
            if kind in {"BarrierArrive", "BarrierObserve"}:
                event.update({"resource": "cluster", "invocation_id": invocation})
                if kind == "BarrierArrive":
                    event["count"] = 1
            else:
                event["region"] = "remote_tile"
            events.append(event)
    raw["events"] = events
    return raw


@pytest.mark.parametrize("cluster_ctas", [2, 3, 4])
@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_pgir_v2_cluster_barrier_publishes_dsmem(engine, cluster_ctas):
    result = check_module(
        module_from_dict(cluster_dsmem_module(cluster_ctas=cluster_ctas)),
        engine=engine,
    )
    assert result["verdict"] == "SAFE"
    assert "single_cluster" in result["manifest"]["scope"]
    assert any(edge["kind"] == "BARRIER_RELEASE" for edge in result["artifacts"]["pdg"]["edges"])
    if engine == "maximal":
        assert result["statistics"]["barrier_release_streams"] == 1
        assert sum(
            item["anchor"].startswith("completion:barrier-release:")
            for item in result["artifacts"]["boundaries"]
        ) == 3
        assert result["statistics"]["collective_notifications"] > 0
        assert result["statistics"]["full_enabled_scans"] == 0
        assert result["statistics"]["state_clones"] == 0


def test_cluster_control_only_does_not_publish_dsmem():
    result = check_module(module_from_dict(cluster_dsmem_module(control_only=True)), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"


def test_dsmem_alias_identity_includes_owner_cta_rank():
    raw = cluster_dsmem_module(control_only=True)
    alternate = {**raw["regions"][0], "id": "other_tile", "owner_cta_rank": 1}
    raw["regions"].append(alternate)
    next(event for event in raw["events"] if event["id"] == "read_a1")["region"] = "other_tile"
    assert check_module(module_from_dict(raw), engine="crosscheck")["verdict"] == "SAFE"


def test_pgir_v2_cluster_topology_and_lifetime_fail_closed():
    with pytest.raises(PGIRError) as size:
        module_from_dict(cluster_dsmem_module(cluster_ctas=9))
    assert size.value.code == "NONPORTABLE_CLUSTER_SIZE"

    raw = cluster_dsmem_module()
    raw["regions"][0]["owner_cta_rank"] = 2
    with pytest.raises(PGIRError) as owner:
        module_from_dict(raw)
    assert owner.value.code == "DSMEM_OWNER_OUT_OF_RANGE"

    raw = cluster_dsmem_module()
    raw["resources"][0].update({"participants": ["a0"], "capacity": 1})
    with pytest.raises(PGIRError) as participants:
        module_from_dict(raw)
    assert participants.value.code == "CLUSTER_PARTICIPANT_SET_INCOMPLETE"

    raw = cluster_dsmem_module()
    raw["events"] = [event for event in raw["events"] if event["id"] != "observe2_a0"]
    for actor in ("a0", "a1"):
        for seq, event in enumerate(item for item in raw["events"] if item["actor"] == actor):
            event["seq"] = seq
    with pytest.raises(PGIRError) as participation:
        module_from_dict(raw)
    assert participation.value.code == "CLUSTER_BARRIER_PARTICIPATION_INCOMPLETE"

    raw = cluster_dsmem_module()
    write = next(event for event in raw["events"] if event["id"] == "write_a0")
    for event in raw["events"]:
        if event["actor"] == "a0" and event is not write:
            event["seq"] += 1
    write["seq"] = 0
    with pytest.raises(PGIRError) as lifetime:
        module_from_dict(raw)
    assert lifetime.value.code == "DSMEM_LIFETIME_UNGUARDED"


def test_pgir_v1_cannot_silently_accept_cluster_fields():
    raw = cluster_dsmem_module()
    raw["schema_version"] = "pgir-v1"
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "SCHEMA_INVALID"


def test_pgir_v2_cluster_fixture_runs_through_cli():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeguard",
            "check",
            str(FIXTURES / "cluster_dsmem_handoff.json"),
            "--engine",
            "crosscheck",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert result["verdict"] == "SAFE"
    assert "single_cluster" in result["manifest"]["scope"]
