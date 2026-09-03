from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import FrontendError, load_triton_snapshot, triton_snapshot_to_module
from pipeguard.model import PGIRError, normalize_module

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "triton_safe_handoff.snapshot.json"
CAPTURED_SNAPSHOT = FIXTURES / "triton_captured_mbarrier.snapshot.json"


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def test_triton_snapshot_normalizes_to_dsl_neutral_pgir():
    module = load_triton_snapshot(SNAPSHOT)
    normalized = normalize_module(module)
    assert normalized["schema_version"] == "pgir-v1"
    assert normalized["producer"]["dsl"] == "TRITON"
    assert normalized["frontend_provenance"] == {
        "extraction_complete": True,
        "source": "triton_safe_handoff.ttgir",
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
    wait = next(event for event in normalized["events"] if event["id"] == "triton:wait")
    assert wait["parity_evidence"]["derivation_kind"] == "PHASE_SSA"
    assert wait["parity_evidence"]["stream_rank"] == 0


def test_triton_snapshot_runs_through_crosschecked_checker():
    result = check_module(load_triton_snapshot(SNAPSHOT), engine="crosscheck")
    assert result["verdict"] == "SAFE"
    assert result["manifest"]["producer"]["dsl"] == "TRITON"
    assert result["artifacts"]["crosscheck"]["maximal"] == result["artifacts"]["crosscheck"]["exhaustive"]


def test_live_triton_ttgir_capture_fixture_is_safe():
    snapshot = json.loads(CAPTURED_SNAPSHOT.read_text())
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mbarrier.init",
        "triton.mbarrier.arrive",
        "triton.mbarrier.wait_parity",
    ]
    result = check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")
    assert result["verdict"] == "SAFE"
    assert result["artifacts"]["normalized_sha256"] == (
        "eb604645de9b36b7968d9845904b9b880b85bdd37ca0f0de3bf119c4c809932d"
    )


@pytest.mark.parametrize("command", ["validate", "normalize", "check", "explain"])
def test_cli_accepts_triton_frontend_snapshot(command: str):
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", command, str(SNAPSHOT), "--frontend", "triton"],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    if command == "validate":
        assert document["valid"] is True
    elif command == "normalize":
        assert document["producer"]["dsl"] == "TRITON"
    else:
        assert document["verdict"] == "SAFE"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda raw: raw["extraction"].update({"complete": False}), "FRONTEND_EXTRACTION_INCOMPLETE"),
        (lambda raw: raw["extraction"].update({"stage": "ttir"}), "FRONTEND_STAGE_UNSUPPORTED"),
        (lambda raw: raw["unsupported_constructs"].append("scf.while"), "FRONTEND_UNSUPPORTED_CONSTRUCTS"),
        (lambda raw: raw["operations"][0].update({"op": "triton.unknown"}), "FRONTEND_UNKNOWN_OPERATION"),
        (lambda raw: raw["operations"][0].update({"guessed": True}), "FRONTEND_FIELDS_UNKNOWN"),
        (lambda raw: raw.update({"producer": []}), "FRONTEND_FIELD_TYPE"),
    ],
)
def test_triton_frontend_fails_closed(mutation, code: str):
    raw = _snapshot()
    mutation(raw)
    with pytest.raises(FrontendError) as caught:
        triton_snapshot_to_module(raw)
    assert caught.value.code == code


def test_triton_frontend_does_not_repair_wrong_phase_ssa():
    raw = _snapshot()
    wait = next(operation for operation in raw["operations"] if operation["id"] == "triton:wait")
    wait["raw_parity"] = 1
    with pytest.raises(PGIRError) as caught:
        triton_snapshot_to_module(raw)
    assert caught.value.code == "MALFORMED_STREAM_PARITY"


def test_cli_reports_triton_frontend_failure_as_unsupported(tmp_path: Path):
    raw = _snapshot()
    raw["operations"][0]["op"] = "triton.unknown"
    path = tmp_path / "unknown.snapshot.json"
    path.write_text(json.dumps(raw))
    process = subprocess.run(
        [sys.executable, "-m", "pipeguard", "check", str(path), "--frontend", "triton"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    document = json.loads(process.stdout)
    assert document["verdict"] == "UNSUPPORTED"
    assert document["diagnostics"][0]["code"] == "FRONTEND_UNKNOWN_OPERATION"
