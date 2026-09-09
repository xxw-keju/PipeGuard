from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TESTS))

from pipeguard.checker import check_module  # noqa: E402
from pipeguard.model import module_from_dict  # noqa: E402
from suite.pgir_direct.cases.catalog import CASES  # noqa: E402
from test_parametric import _single_stream_parametric, parametric_memory_module  # noqa: E402


def _case(label: str) -> dict[str, Any]:
    return deepcopy(next(case.raw for case in CASES if case.label == label))


def _as_parametric(raw: dict[str, Any]) -> dict[str, Any]:
    if len(raw["actors"]) == 1:
        return _single_stream_parametric(raw)
    actors = {actor["id"]: actor for actor in raw["actors"]}
    lanes = sorted({
        lane
        for actor in raw["actors"]
        for lane in actor["participants"]
    })
    templates = []
    for order, event in enumerate(sorted(
        raw["events"], key=lambda item: (item["seq"], item["actor"], item["id"])
    )):
        actor = actors[event["actor"]]
        fields = {
            key: deepcopy(value)
            for key, value in event.items()
            if key not in {"id", "type", "actor", "seq", "depends_on", "owner"}
        }
        instances = list(actor["participants"])
        scope = (
            "COLLECTIVE" if event["type"].startswith("Wgmma")
            else "ELECTED" if event["type"] == "TmaLoadIssue"
            else "INSTANCE"
        )
        template = {
            "id": event["id"],
            "type": event["type"],
            "stream": "lane",
            "order": order,
            "execution_scope": scope,
            "instances": instances,
            "fields": fields,
        }
        if event["type"] not in {"MemRead", "MemWrite"}:
            template["owner"] = event["actor"]
        if event.get("depends_on"):
            template["dependencies"] = [
                {"template": dependency, "relation": "ALL_INSTANCES"}
                for dependency in event["depends_on"]
            ]
        templates.append(template)
    return {
        "schema_version": "pgir-parametric-v1",
        "producer": deepcopy(raw["producer"]),
        "target": deepcopy(raw["target"]),
        "assumptions": deepcopy(raw["assumptions"]),
        "control_structure": deepcopy(raw["control_structure"]),
        "owners": deepcopy(raw["actors"]),
        "program_streams": [{"id": "lane", "kind": "LANE", "instances": lanes}],
        "resources": deepcopy(raw["resources"]),
        "regions": deepcopy(raw["regions"]),
        "event_templates": templates,
        "frontend_provenance": {
            "extraction_complete": True,
            "source": "semantic-gate-multi-actor-adapter",
        },
        "frontend_evidence": {
            claim: {
                "status": "EXACT" if claim == "alias" else "RESOLVED",
                "method": "direct-case-multi-actor",
                "source": "fixture",
            }
            for claim in (
                "participant_mapping", "lane_ownership", "predicate_mapping", "alias"
            )
        },
        "unsupported_constructs": [],
    }


def _events(raw: dict[str, Any], *event_ids: str) -> dict[str, Any]:
    wanted = set(event_ids)
    selected = [event for event in raw["events"] if event["id"] in wanted]
    if {event["id"] for event in selected} != wanted:
        raise AssertionError(f"missing event in semantic gate: {wanted}")
    selected.sort(key=lambda event: event["seq"])
    for sequence, event in enumerate(selected):
        event["seq"] = sequence
        event.pop("depends_on", None)
    raw["events"] = selected
    return raw


def _barrier_pair() -> dict[str, Any]:
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "sync",
        "kind": "cta_barrier",
        "participants": ["partition:0"],
        "capacity": 2,
        "scope": "CTA",
        "visibility_mask": {
            "shared": True,
            "global_read": False,
            "global_write": False,
            "tensor_read": False,
            "tensor_write": False,
            "control_only": False,
        },
    }]
    raw["event_templates"] = [
        raw["event_templates"][0],
        {
            "id": "arrive",
            "type": "BarrierArrive",
            "stream": "lane",
            "order": 1,
            "program_instances": [0, 1],
            "executors": [0, 1],
            "transition": "ONCE",
            "rendezvous": "ALL_PROGRAM_INSTANCES",
            "owner": "partition:0",
            "fields": {"resource": "sync", "count": 2, "invocation_id": 0},
        },
        {
            "id": "observe",
            "type": "BarrierObserve",
            "stream": "lane",
            "order": 2,
            "program_instances": [0, 1],
            "executors": [0, 1],
            "transition": "ONCE",
            "rendezvous": "ALL_PROGRAM_INSTANCES",
            "owner": "partition:0",
            "fields": {"resource": "sync", "invocation_id": 0},
        },
        {**raw["event_templates"][1], "order": 3},
    ]
    return raw


def _barrier_participants(*, complete: bool) -> dict[str, Any]:
    raw = parametric_memory_module()
    raw["owners"] = [
        {"id": "owner:0", "kind": "LANE", "participants": [0]},
        {"id": "owner:1", "kind": "LANE", "participants": [1]},
    ]
    raw["resources"] = [{
        "id": "sync",
        "kind": "cta_barrier",
        "participants": ["owner:0", "owner:1"],
        "capacity": 2,
        "scope": "CTA",
        "visibility_mask": {
            "shared": True,
            "global_read": False,
            "global_write": False,
            "tensor_read": False,
            "tensor_write": False,
            "control_only": False,
        },
    }]
    arrivals = [{
        "id": "arrive_0",
        "type": "BarrierArrive",
        "stream": "lane",
        "order": 0,
        "program_instances": [0],
        "executors": [0],
        "transition": "PER_INSTANCE",
        "rendezvous": "NONE",
        "owner": "owner:0",
        "fields": {"resource": "sync", "count": 1, "invocation_id": 0},
    }]
    if complete:
        arrivals.append({
            **arrivals[0],
            "id": "arrive_1",
            "order": 1,
            "program_instances": [1],
            "executors": [1],
            "owner": "owner:1",
        })
    raw["event_templates"] = arrivals
    return raw


def _cp_capacity(*, wait_before_ninth: bool) -> dict[str, Any]:
    raw = _case("PGIR-CP-04-NAMED-WAIT-FRONTIER-CORRECT")
    issue = next(event for event in raw["events"] if event["id"] == "issue")
    commit = next(event for event in raw["events"] if event["id"] == "commit")
    wait = next(event for event in raw["events"] if event["id"] == "wait")
    events = []
    regions = []
    source = next(region for region in raw["regions"] if region["id"] == "source")
    tile = next(region for region in raw["regions"] if region["id"] == "tile")
    for sequence in range(9):
        if sequence == 8 and wait_before_ninth:
            current = deepcopy(wait)
            current["id"] = "wait_before_ninth"
            current["seq"] = len(events)
            events.append(current)
        source_region = deepcopy(source)
        source_region["id"] = f"source_{sequence}"
        source_region["offset"] = sequence * source["extent"]
        tile_region = deepcopy(tile)
        tile_region["id"] = f"tile_{sequence}"
        tile_region["offset"] = sequence * tile["extent"]
        regions.extend((source_region, tile_region))
        current_issue = deepcopy(issue)
        current_issue["id"] = f"issue_{sequence}"
        current_issue["source_region"] = source_region["id"]
        current_issue["destination_region"] = tile_region["id"]
        current_issue["seq"] = len(events)
        events.append(current_issue)
        current_commit = deepcopy(commit)
        current_commit["id"] = f"commit_{sequence}"
        current_commit["group_seq"] = sequence
        current_commit["seq"] = len(events)
        events.append(current_commit)
    raw["events"] = events
    raw["regions"] = regions
    return raw


def _diagnostic(report: dict[str, Any]) -> str | None:
    return report.get("diagnostics", [{}])[0].get("code") if report.get("diagnostics") else None


def _effects(report: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        (effect["mode"], effect["region"])
        for effect in report.get("witness", {}).get("effects", [])
    )


def gate_definitions() -> list[tuple[str, dict[str, Any], str]]:
    wgmma = "PGIR-WG-07-WG-08-NAMED-DUAL-ENDPOINT-CORRECT"
    tma = "PGIR-TMA-05-NAMED-STORE-DUAL-ENDPOINT-CORRECT"
    cp_safe = "PGIR-CP-04-NAMED-WAIT-FRONTIER-CORRECT"
    cp_bug = "PGIR-CP-04-CP-WAIT-V00-BUG"

    independent = parametric_memory_module()
    dependency = parametric_memory_module()
    dependency["event_templates"][1]["dependencies"] = [{
        "template": "write", "relation": "ALL_INSTANCES",
    }]

    return [
        ("independent-lanes-canonical-order-is-not-visibility", independent, "DATA_HAZARD"),
        ("generic-dependency-is-not-visibility", dependency, "DATA_HAZARD"),
        ("cta-barrier-typed-cut-orders-cross-lane-access", _barrier_pair(), "SAFE"),
        ("wgmma-eager-operand-completion-without-wait-stays-open", _single_stream_parametric(_events(_case(wgmma), "fence", "issue", "commit", "overwrite")), "DATA_HAZARD"),
        ("wgmma-operands-released-closes-only-operand-window", _single_stream_parametric(_events(_case(wgmma), "fence", "issue", "commit", "wait_operand", "overwrite")), "SAFE"),
        ("wgmma-operand-wait-does-not-release-result", _single_stream_parametric(_events(_case(wgmma), "fence", "issue", "commit", "wait_operand", "read_accumulator")), "SYNC_HAZARD"),
        ("wgmma-result-ready-releases-result", _single_stream_parametric(_events(_case(wgmma), "fence", "issue", "commit", "wait_result", "read_accumulator")), "SAFE"),
        ("tma-store-eager-source-completion-without-wait-stays-open", _single_stream_parametric(_events(_case(tma), "issue", "commit", "overwrite")), "DATA_HAZARD"),
        ("tma-store-source-released-allows-source-reuse", _single_stream_parametric(_events(_case(tma), "issue", "commit", "wait_source", "overwrite")), "SAFE"),
        ("tma-store-read-only-wait-does-not-make-global-visible", _single_stream_parametric(_events(_case(tma), "issue", "commit", "wait_source", "read_destination")), "DATA_HAZARD"),
        ("tma-store-global-complete-orders-global-read", _single_stream_parametric(_events(_case(tma), "issue", "commit", "wait_global", "read_destination")), "SAFE"),
        ("cp-async-mandated-wait-frontier", _single_stream_parametric(_case(cp_safe)), "SAFE"),
        ("cp-async-non-mandated-recent-group-stays-open", _single_stream_parametric(_case(cp_bug)), "DATA_HAZARD"),
        ("cp-async-held-completions-expose-capacity-overflow", _single_stream_parametric(_cp_capacity(wait_before_ninth=False)), "SYNC_HAZARD"),
        ("cp-async-final-endpoint-wait-retires-capacity", _single_stream_parametric(_cp_capacity(wait_before_ninth=True)), "SAFE"),
        ("missing-barrier-participant-is-not-released", _barrier_participants(complete=False), "SYNC_HAZARD"),
        ("complete-barrier-participant-set-releases", _barrier_participants(complete=True), "SAFE"),
        ("stale-mbarrier-generation-does-not-release", _as_parametric(_case("PGIR-MB-09-NAMED-STALE-GENERATION-WAIT-BUG")), "SYNC_HAZARD"),
        ("matching-mbarrier-generation-releases", _as_parametric(_case("PGIR-MB-09-NAMED-STALE-GENERATION-WAIT-CORRECT")), "SAFE"),
        ("missing-proxy-direction-does-not-create-visibility", _single_stream_parametric(_case("PGIR-TMA-06-NAMED-GENERIC-SOURCE-NO-PROXY-BUG")), "DATA_HAZARD"),
        ("matching-proxy-direction-and-wait-create-visibility", _single_stream_parametric(_case("PGIR-TMA-06-NAMED-PROXY-AND-WAIT-CORRECT")), "SAFE"),
    ]


def evaluate_gates() -> dict[str, Any]:
    records = []
    for gate_id, raw, expected in gate_definitions():
        module = module_from_dict(raw)
        direct = check_module(module, engine="maximal")
        reference = check_module(module.expand_reference(), engine="crosscheck")
        record = {
            "gate_id": gate_id,
            "expected_verdict": expected,
            "direct_verdict": direct["verdict"],
            "reference_verdict": reference["verdict"],
            "direct_diagnostic": _diagnostic(direct),
            "reference_diagnostic": _diagnostic(reference),
            "direct_effects": _effects(direct),
            "reference_effects": _effects(reference),
            "reference_engine": reference["statistics"].get("engine"),
            "direct_representation": direct["statistics"].get("representation"),
            "materialized_event_objects": direct["statistics"].get(
                "materialized_event_objects"
            ),
            "race_candidate_pairs": direct["statistics"].get(
                "race_candidate_pairs"
            ),
        }
        record["equivalent"] = (
            direct["verdict"] == expected
            and reference["verdict"] == expected
            and record["direct_diagnostic"] == record["reference_diagnostic"]
            and record["direct_effects"] == record["reference_effects"]
        )
        records.append(record)
    return {
        "schema_version": "pipeguard-semantic-gates-v1",
        "reference_method": "expanded PGIR maximal-plus-exhaustive crosscheck",
        "gate_count": len(records),
        "all_equivalent": all(record["equivalent"] for record in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = evaluate_gates()
    payload["git_head"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    payload["tracked_diff_sha256"] = hashlib.sha256(subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--binary"],
        capture_output=True,
        check=True,
    ).stdout).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "gate_count": payload["gate_count"],
        "all_equivalent": payload["all_equivalent"],
    }, sort_keys=True))
    return 0 if payload["all_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
