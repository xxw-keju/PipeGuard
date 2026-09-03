from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeguard.checker import _anchor_ids, check_module
from pipeguard.engine import ExecutionState, Executor, ProtocolError
from pipeguard.incremental import END, IncrementalEngine, StableReadySet, build_anchor_streams
from pipeguard.model import (
    PGIRError,
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
    raw["events"][0]["type"] = "CpAsyncIssue"
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


def test_init_cannot_be_invalidated_before_a_completed_phase():
    result = check_module(lifecycle_module("MBarrierInit", "MBarrierInvalidate"))
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "INVALIDATE_INCOMPLETE"


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
        assert statistics["state_initializations"] == 2
        assert statistics["completed_passes"] == 2
        assert statistics["anchor_boundaries"] == size
        assert statistics["transitions_by_kind"]["PROGRAM"] == 2 * size
        assert statistics["transition_attempts"] == 2 * size
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
        "total": 3,
    }
    assert first["boundary_evaluations"] == 6


@pytest.mark.parametrize("engine", ["maximal", "exhaustive", "crosscheck"])
def test_never_spawned_synthetic_phase_anchor_is_unreachable_not_stuck(engine):
    module = lifecycle_module("MBarrierInit", "MBarrierArrive")
    raw = json.loads(json.dumps(module.raw))
    raw["events"][1]["predicate"] = False
    result = check_module(module_from_dict(raw), engine=engine)
    assert result["verdict"] == "INCONCLUSIVE"
    if engine == "crosscheck":
        assert result["artifacts"]["crosscheck"]["maximal"]["verdict"] == "INCONCLUSIVE"
        assert result["artifacts"]["crosscheck"]["exhaustive"]["verdict"] == "INCONCLUSIVE"
    else:
        assert result["diagnostics"][0]["code"] == "ANCHOR_NOT_REACHED"


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


def test_future_protocol_fields_are_structured_but_still_unsupported():
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
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "KNOWN_EVENT_UNSUPPORTED"

    fence["visibility_mask"].pop("tensor_write")
    with pytest.raises(PGIRError) as malformed:
        module_from_dict(raw)
    assert malformed.value.code == "SCHEMA_INVALID"


def test_future_resource_kinds_cannot_expand_safe_scope_early():
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["resources"][0].update({
        "kind": "channel",
        "capacity": 2,
        "scope": "CTA",
    })
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "KNOWN_RESOURCE_UNSUPPORTED"

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
