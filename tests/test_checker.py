from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.engine import Executor, ProtocolError
from pipeguard.model import PGIRError, load_module, module_from_dict, stable_completion_id

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
    assert {"from": "completion:mbarrier-phase:bar:g00000000", "to": "wait", "kind": "COMPLETION_TO_WAIT"} in edges


@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("missing_arrive.json", "MISSING_ARRIVAL"),
        ("wrong_generation.json", "WRONG_PARITY"),
        ("stale_generation.json", "STALE_GENERATION"),
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


def test_next_phase_waits_at_held_phase_boundary_then_runs_as_current():
    module = load("partial_next_phase.json")
    result = check_module(module)
    assert result["verdict"] == "SAFE"
    phase_anchor = next(item for item in result["artifacts"]["boundaries"] if item["anchor"].endswith("bar:g00000000"))
    assert phase_anchor["boundary"] == ["arrive1a"]



def test_future_generation_is_not_buffered_while_tx_is_outstanding():
    executor = Executor(load("partial_next_phase.json"))
    state = executor.initial_state()
    for identifier in ("init", "expect", "issue", "arrive0"):
        transition = next(item for item in executor.enabled(state) if item.id == identifier)
        executor.apply(state, transition)

    barrier = state.barriers["bar"]
    assert barrier.pending == 0
    assert barrier.completed_tx < barrier.expected_tx
    future = next(item for item in executor.enabled(state) if item.id == "arrive1a")
    with pytest.raises(ProtocolError) as caught:
        executor.apply(state, future)
    assert caught.value.code == "EARLY_NEXT_GENERATION"
    assert barrier.generation == 0
    assert not any(name.startswith("next_") for name in barrier.__dataclass_fields__)


@pytest.mark.parametrize("capacity", [2, 3])
def test_partial_arrival_is_legal_after_phase_advances(capacity: int):
    raw = json.loads((FIXTURES / "partial_next_phase.json").read_text())
    raw["actors"][0]["participants"] = list(range(capacity))
    raw["target"]["cta_threads"] = capacity
    next(event for event in raw["events"] if event["id"] == "init")["count"] = capacity
    next(event for event in raw["events"] if event["id"] == "arrive0")["count"] = capacity
    executor = Executor(module_from_dict(raw))
    state = executor.initial_state()
    for identifier in ("init", "expect", "issue"):
        transition = next(item for item in executor.enabled(state) if item.id == identifier)
        executor.apply(state, transition)
    completion = next(item for item in executor.enabled(state) if item.kind == "TMA_COMPLETION")
    executor.apply(state, completion)
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive0"))
    phase = next(item for item in executor.enabled(state) if item.kind == "PHASE_COMPLETION")
    executor.apply(state, phase)
    executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive1a"))
    if capacity == 3:
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == "wait0"))
        executor.apply(state, next(item for item in executor.enabled(state) if item.id == "arrive1b"))

    barrier = state.barriers["bar"]
    assert barrier.generation == 1
    assert barrier.pending == 1
    assert barrier.generation_started

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
        event = {"id": f"e{seq}", "type": event_type, "actor": "a_producer", "seq": seq, "resource": "bar"}
        if event_type == "MBarrierInit":
            event["count"] = 1
        elif event_type == "MBarrierArrive":
            event.update({"count": 1, "generation": 0})
        elif event_type == "MBarrierWaitParity":
            event.update({
                "generation": 0,
                "raw_parity": 0,
                "parity_evidence": {"kind": "BARRIER_HISTORY", "detail": "lifecycle fixture generation 0"},
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
        "resource": "bar", "count": 1, "generation": 0, "depends_on": ["init"],
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
        {"id": "false", "type": "MBarrierArriveExpectTx", "actor": "a", "seq": 1, "resource": "bar", "count": 1, "bytes": 8, "generation": 0, "predicate": False},
        {"id": "real", "type": "MBarrierArrive", "actor": "a", "seq": 2, "resource": "bar", "count": 2, "generation": 0},
        {"id": "wait", "type": "MBarrierWaitParity", "actor": "b", "seq": 0, "resource": "bar", "generation": 0, "raw_parity": 0, "parity_evidence": {"kind": "BARRIER_HISTORY", "detail": "generation zero"}, "depends_on": ["init"]},
    ]
    assert check_module(module_from_dict(raw))["verdict"] == "SAFE"



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
