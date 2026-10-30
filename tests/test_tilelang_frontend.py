from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import FrontendError, load_tilelang_snapshot, tilelang_snapshot_to_module
from pipeguard.model import PGIRError, normalize_module

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "tilelang_safe_handoff.snapshot.json"
CAPTURED_SNAPSHOT = FIXTURES / "tilelang_captured_handoff.snapshot.json"
CAPTURED_ASYNC_SNAPSHOT = FIXTURES / "tilelang_captured_cp_async.snapshot.json"


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def _parametric_snapshot() -> dict:
    digest = "0" * 64
    return {
        "snapshot_version": "tilelang-snapshot-v3",
        "profile": {
            "id": "sm90-single-cta-static-v1",
            "source_ir_sha256": digest,
            "compiler_ir_stage": "post-inject-fence-proxy",
            "target_rule": "tilelang-cuda-sm_90a-v1",
        },
        "producer": {
            "dsl_version": "test",
            "dsl_commit": "test",
            "backend_commit": "test",
            "adapter_version": "tilelang-pgir-v3",
            "command": "compile-only-test",
        },
        "target": {
            "architecture": "sm90",
            "warp_size": 32,
            "cta_threads": 2,
            "shared_memory_bytes": 0,
            "grid_scope": "single_cta",
            "launch_grid": [1, 1, 1],
            "selected_cta": [0, 0, 0],
        },
        "extraction": {
            "complete": True,
            "stage": "post-inject-fence-proxy",
            "source": "parametric-tilelang-test",
            "finite_control_flow": True,
            "statically_known_participants": True,
            "fair_async_completion": True,
            "no_cluster_effects": True,
            "alias_precision": "EXACT",
            "parity_normalization_status": "RESOLVED",
            "static_expansions": 0,
            "executable_nodes_seen": 1,
            "classified_nodes": 1,
            "emitting_nodes": 1,
            "elided_nodes": 0,
            "operation_records": 1,
            "logical_events_emitted": 2,
            "allocation_identity_evidence": "tir-buffer-data-object-and-actor-identity-v1",
            "alias_evidence": "tirx.noalias-v1",
            "participant_evidence": "primfunc-thread-extent-lane-enumeration-v1",
            "lane_ownership_evidence": "tir-thread-binding-active-lane-evaluation-v1",
            "predicate_evidence": "tir-static-predicate-active-lane-splitting-v1",
        },
        "actors": [
            {"id": "lane:0000", "kind": "LANE", "participants": [0]},
            {"id": "lane:0001", "kind": "LANE", "participants": [1]},
        ],
        "resources": [],
        "regions": [
            {"id": "a:0", "allocation_id": "A", "address_space": "GLOBAL", "offset": 0, "extent": 4},
            {"id": "a:1", "allocation_id": "A", "address_space": "GLOBAL", "offset": 4, "extent": 4},
        ],
        "operations": [{
            "id": "tl:template:0",
            "op": "tilelang.mem.write",
            "order": 0,
            "execution_scope": "INSTANCE",
            "instances": [0, 1],
            "fields": {},
            "instance_fields": {"region": ["a:0", "a:1"]},
            "evidence": {
                "source_location": "root/store",
                "activation_path": ["root", "store"],
                "recognizer_rule": "BufferStore->tilelang.mem.write:v1",
                "ir_operation": "BufferStore",
            },
        }],
        "unsupported_constructs": [],
    }


def test_tilelang_snapshot_normalizes_to_dsl_neutral_pgir():
    module = load_tilelang_snapshot(SNAPSHOT)
    normalized = normalize_module(module)
    assert normalized["schema_version"] == "pgir-v1"
    assert normalized["producer"]["dsl"] == "TILELANG"
    assert normalized["frontend_provenance"] == {
        "extraction_complete": True,
        "source": "tilelang_safe_handoff.tirx",
    }
    assert [event["type"] for event in normalized["events"]] == [
        "BarrierArrive",
        "BarrierObserve",
        "MBarrierWaitParity",
        "MemRead",
        "MBarrierInit",
        "BarrierArrive",
        "BarrierObserve",
        "MemWrite",
        "MBarrierArrive",
    ]
    wait = next(event for event in normalized["events"] if event["id"] == "tl:wait")
    assert wait["parity_evidence"] == {
        "derivation_kind": "PARITY_EXPRESSION",
        "logical_stream_id": "consumer:mbarrier:smem:0",
        "lifecycle_epoch": 0,
        "stream_rank": 0,
        "polling_coalesced": True,
    }


def test_tilelang_v3_snapshot_preserves_parameterized_lanes():
    module = tilelang_snapshot_to_module(_parametric_snapshot())
    normalized = normalize_module(module)
    assert normalized["schema_version"] == "pgir-parametric-v1"
    assert normalized["program_streams"] == [{
        "id": "program:tilelang:lanes",
        "kind": "LANE",
        "instances": [0, 1],
    }]
    template = normalized["event_templates"][0]
    assert template["program_instances"] == [0, 1]
    assert template["executors"] == [0, 1]
    assert template["transition"] == "PER_INSTANCE"
    assert template["rendezvous"] == "NONE"
    assert template["instance_fields"] == {
        "region": ["a:0", "a:1"],
    }
    result = check_module(module, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["representation"] == "parametric-protocol"
    assert result["statistics"]["events"] == 2
    assert result["statistics"]["materialized_event_objects"] == 0


def test_tilelang_v3_rejects_misaligned_instance_fields():
    raw = _parametric_snapshot()
    raw["operations"][0]["instance_fields"]["region"] = ["a:0"]
    with pytest.raises(FrontendError) as caught:
        tilelang_snapshot_to_module(raw)
    assert caught.value.code == "FRONTEND_INSTANCE_FIELDS_INVALID"


def test_tilelang_snapshot_runs_through_crosschecked_checker():
    result = check_module(load_tilelang_snapshot(SNAPSHOT), engine="crosscheck")
    assert result["verdict"] == "SAFE"
    assert result["manifest"]["producer"]["dsl"] == "TILELANG"
    assert result["artifacts"]["crosscheck"]["maximal"] == result["artifacts"]["crosscheck"]["exhaustive"]


def test_live_tilelang_tirx_capture_fixture_is_safe():
    snapshot = json.loads(CAPTURED_SNAPSHOT.read_text())
    assert snapshot["snapshot_version"] == "tilelang-snapshot-v2"
    assert snapshot["extraction"]["events_emitted"] == len(snapshot["operations"])
    assert all("evidence" in operation for operation in snapshot["operations"])
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "tilelang.mbarrier.init",
        "tilelang.barrier.arrive",
        "tilelang.barrier.observe",
        "tilelang.mem.write",
        "tilelang.mbarrier.arrive",
        "tilelang.barrier.arrive",
        "tilelang.barrier.observe",
        "tilelang.mbarrier.wait_parity",
        "tilelang.mem.read",
        "tilelang.mem.write",
    ]
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="crosscheck")
    assert result["verdict"] == "SAFE"
    assert result["artifacts"]["normalized_sha256"] == (
        "806a9c417eaeba4d30316cf7b7b1e3e45d5f06a155280d16a8b688ea2974f825"
    )


def test_live_tilelang_cp_async_capture_fixture_is_safe():
    snapshot = json.loads(CAPTURED_ASYNC_SNAPSHOT.read_text())
    assert snapshot["snapshot_version"] == "tilelang-snapshot-v2"
    assert snapshot["extraction"]["events_emitted"] == len(snapshot["operations"])
    assert all("evidence" in operation for operation in snapshot["operations"])
    operations = [operation["op"] for operation in snapshot["operations"]]
    assert operations[:5] == [
        "tilelang.mem.write",
        "tilelang.proxy_fence",
        "tilelang.cp_async.issue",
        "tilelang.cp_async.commit_group",
        "tilelang.cp_async.wait_group",
    ]
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="crosscheck")
    assert result["verdict"] == "SAFE"
    # Normalization drops the ignored legacy async-group capacity field.
    assert result["artifacts"]["normalized_sha256"] == (
        "3141dc2b59167dcc81dadd9330247cc1771520a642dd8b040c39b5b64f8f37f2"
    )
    assert result["statistics"]["n_ext"] == 11
    assert result["statistics"]["t_eff"] == 3


@pytest.mark.parametrize("command", ["validate", "normalize", "check", "explain"])
def test_cli_accepts_tilelang_frontend_snapshot(command: str):
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", command, str(SNAPSHOT), "--frontend", "tilelang"],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    if command == "validate":
        assert document["valid"] is True
    elif command == "normalize":
        assert document["producer"]["dsl"] == "TILELANG"
    else:
        assert document["verdict"] == "SAFE"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda raw: raw["extraction"].update({"complete": False}), "FRONTEND_EXTRACTION_INCOMPLETE"),
        (lambda raw: raw["extraction"].update({"stage": "pre-lowering"}), "FRONTEND_STAGE_UNSUPPORTED"),
        (lambda raw: raw["unsupported_constructs"].append("dynamic_loop"), "FRONTEND_UNSUPPORTED_CONSTRUCTS"),
        (lambda raw: raw["operations"][0].update({"op": "tilelang.unknown"}), "FRONTEND_UNKNOWN_OPERATION"),
        (lambda raw: raw["operations"][0].update({"guessed": True}), "FRONTEND_FIELDS_UNKNOWN"),
        (lambda raw: raw.update({"producer": []}), "FRONTEND_FIELD_TYPE"),
    ],
)
def test_tilelang_frontend_fails_closed(mutation, code: str):
    raw = _snapshot()
    mutation(raw)
    with pytest.raises(FrontendError) as caught:
        tilelang_snapshot_to_module(raw)
    assert caught.value.code == code


def test_tilelang_frontend_does_not_repair_wrong_parity():
    raw = _snapshot()
    wait = next(operation for operation in raw["operations"] if operation["id"] == "tl:wait")
    wait["raw_parity"] = 1
    with pytest.raises(PGIRError) as caught:
        tilelang_snapshot_to_module(raw)
    assert caught.value.code == "MALFORMED_STREAM_PARITY"


def test_cli_reports_frontend_failure_as_unsupported(tmp_path: Path):
    raw = _snapshot()
    raw["operations"][0]["op"] = "tilelang.unknown"
    path = tmp_path / "unknown.snapshot.json"
    path.write_text(json.dumps(raw))
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", "check", str(path), "--frontend", "tilelang"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    document = json.loads(process.stdout)
    assert document["verdict"] == "UNSUPPORTED"
    assert document["diagnostics"][0]["code"] == "FRONTEND_UNKNOWN_OPERATION"
