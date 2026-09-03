from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pipeguard.checker import check_module
from pipeguard.frontends import (
    semantic_digest,
    semantic_signature,
    tilelang_snapshot_to_module,
    triton_snapshot_to_module,
)

FIXTURES = Path(__file__).parent / "fixtures"
TILELANG = FIXTURES / "tilelang_safe_handoff.snapshot.json"
TRITON = FIXTURES / "triton_safe_handoff.snapshot.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _without_wait(snapshot: dict) -> dict:
    snapshot["operations"] = [
        operation
        for operation in snapshot["operations"]
        if not operation["op"].endswith("mbarrier.wait_parity")
    ]
    read = next(
        operation
        for operation in snapshot["operations"]
        if operation["op"].endswith("mem.read") and operation["actor"] == "consumer"
    )
    read["seq"] = 2
    return snapshot


def _witness_shape(result: dict) -> dict:
    witness = result["witness"]
    return {
        "category": witness["category"],
        "effects": sorted(
            (effect["mode"], effect["region"], effect["window"][0] == effect["window"][1])
            for effect in witness["effects"]
        ),
        "missing_path": witness["missing_path"],
        "prefix_kinds": [step["kind"] for step in witness["reachable_execution_prefix"]],
    }


def test_equivalent_tilelang_and_triton_normalize_to_one_semantic_signature():
    tilelang = tilelang_snapshot_to_module(_load(TILELANG))
    triton = triton_snapshot_to_module(_load(TRITON))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    assert semantic_digest(tilelang) == semantic_digest(triton)


def test_equivalent_safe_frontends_have_same_verdict_and_work_certificate():
    reports = [
        check_module(tilelang_snapshot_to_module(_load(TILELANG)), engine="crosscheck"),
        check_module(triton_snapshot_to_module(_load(TRITON)), engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["SAFE", "SAFE"]
    comparable = (
        "n_ext",
        "t_eff",
        "scheduler_work_units",
        "transition_attempts",
        "anchor_boundaries",
        "pdg_nodes",
        "pdg_edges",
        "pdg_edges_by_kind",
    )
    assert {
        key: reports[0]["statistics"][key]
        for key in comparable
    } == {
        key: reports[1]["statistics"][key]
        for key in comparable
    }


def test_equivalent_missing_wait_frontends_have_same_hazard_witness_shape():
    tilelang = tilelang_snapshot_to_module(_without_wait(_load(TILELANG)))
    triton = triton_snapshot_to_module(_without_wait(_load(TRITON)))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    reports = [
        check_module(tilelang, engine="crosscheck"),
        check_module(triton, engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["DATA_HAZARD", "DATA_HAZARD"]
    assert _witness_shape(reports[0]) == _witness_shape(reports[1])


def test_compare_cli_reports_cross_dsl_equivalence():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeguard",
            "compare",
            str(TILELANG),
            str(TRITON),
            "--left-frontend",
            "tilelang",
            "--right-frontend",
            "triton",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    assert document["equivalent"] is True
    assert document["semantic_equivalent"] is True
    assert document["checker_equivalent"] is True
    assert document["left"]["semantic_sha256"] == document["right"]["semantic_sha256"]
    assert document["left"]["verdict"] == document["right"]["verdict"] == "SAFE"


def test_cross_dsl_signature_and_cli_are_deterministic():
    command = [
        sys.executable,
        "-m",
        "pipeguard",
        "compare",
        str(TILELANG),
        str(TRITON),
        "--left-frontend",
        "tilelang",
        "--right-frontend",
        "triton",
    ]
    first = subprocess.run(command, check=True, capture_output=True).stdout
    second = subprocess.run(command, check=True, capture_output=True).stdout
    assert first == second
