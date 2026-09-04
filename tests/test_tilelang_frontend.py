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
    assert result["artifacts"]["normalized_sha256"] == (
        "707d03034a828a7e54d8cd865e87f88c21e6afc53f2f7f094ee4fc78f1f65b03"
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
