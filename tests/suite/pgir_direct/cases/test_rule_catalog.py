from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from pipeguard.checker import check_module
from pipeguard.model import PGIRError, module_from_dict
from suite.pgir_direct.cases.catalog import CASES, RuleCase



def _assert_subset(actual: Any, expected: Any, path: str = "protocol_state") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), {"path": path, "actual": actual, "expected": expected}
        for key, value in expected.items():
            assert key in actual, {"path": path, "missing": key, "actual": actual}
            _assert_subset(actual[key], value, f"{path}.{key}")
        return
    assert actual == expected, {"path": path, "actual": actual, "expected": expected}


def _diagnostic_events(diagnostic: dict[str, Any]) -> tuple[str, ...]:
    if "events" in diagnostic:
        return tuple(diagnostic["events"])
    event = diagnostic.get("event")
    return () if event is None else (event,)


def _witness_events(witness: dict[str, Any]) -> tuple[str, ...]:
    if "events" in witness:
        return tuple(witness["events"])
    event = witness.get("event")
    return () if event is None else (event,)


def test_rule_catalog_is_unified_labeled_and_auditable():
    assert len(CASES) == 290
    assert len({case.label for case in CASES}) == len(CASES)
    assert all(case.label.startswith("PGIR-") for case in CASES)
    assert {case.expected_stage for case in CASES} == {"MODEL", "CHECKER"}
    assert {case.polarity for case in CASES} == {"BUG", "CORRECT"}

    assert sum(case.expected_stage == "MODEL" for case in CASES) == 10
    assert sum(case.expected_verdict == "SAFE" for case in CASES) == 140
    assert sum(case.expected_verdict == "DATA_HAZARD" for case in CASES) == 117
    assert sum(case.expected_verdict == "SYNC_HAZARD" for case in CASES) == 23
    assert len({case.scenario_id for case in CASES}) == 49

    pair_members: dict[str, list[RuleCase]] = {}
    for case in CASES:
        pair_members.setdefault(case.pair_id, []).append(case)
    complete_pairs = [members for members in pair_members.values() if len(members) == 2]
    assert len(complete_pairs) == 133
    assert all({case.polarity for case in members} == {"BUG", "CORRECT"} for members in complete_pairs)

    for case in CASES:
        assert case.family
        assert case.scenario_id
        assert isinstance(case.variant, int) and case.variant >= 0
        assert case.rule_ids and all(case.rule_ids)
        assert case.semantic_dimensions
        assert case.raw["frontend_provenance"]["source"] == f"direct-pgir://{case.label}"
        if case.polarity == "BUG":
            assert case.possible_bad_execution
        else:
            assert not case.possible_bad_execution
        if case.expected_stage == "MODEL":
            assert case.expected_verdict is None
            assert case.expected_diagnostic
            assert not case.diagnostic_events
            assert not case.expected_effects
            assert case.repair is None
        elif case.expected_verdict == "SAFE":
            assert case.expected_diagnostic is None
            assert not case.diagnostic_events
            assert not case.expected_effects
            assert case.repair is None
        else:
            assert case.expected_diagnostic
            assert case.repair
            if case.expected_diagnostic == "UNORDERED_CONFLICT":
                assert case.diagnostic_events
                assert case.expected_effects
            else:
                assert case.protocol_state


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_direct_pgir_nvidia_rule_case(case: RuleCase):
    raw = deepcopy(case.raw)
    if case.expected_stage == "MODEL":
        with pytest.raises(PGIRError) as caught:
            module_from_dict(raw)
        assert caught.value.code == case.expected_diagnostic, {
            "label": case.label,
            "rules": case.rule_ids,
            "expected_stage": case.expected_stage,
            "possible_bad_execution": list(case.possible_bad_execution),
            "error": caught.value.diagnostic(),
        }
        return

    result = check_module(module_from_dict(raw), engine="crosscheck")
    context = {
        "label": case.label,
        "rules": case.rule_ids,
        "semantic_dimensions": case.semantic_dimensions,
        "possible_bad_execution": list(case.possible_bad_execution),
        "result": result,
    }
    assert result["verdict"] == case.expected_verdict, context

    diagnostics = result["diagnostics"]
    if case.expected_verdict == "SAFE":
        assert diagnostics == [], context
        assert "witness" not in result, context
    else:
        assert len(diagnostics) == 1, context
        diagnostic = diagnostics[0]
        assert diagnostic["code"] == case.expected_diagnostic, context
        assert _diagnostic_events(diagnostic) == case.diagnostic_events, context

        witness = result["witness"]
        assert witness["category"] == case.expected_diagnostic, context
        assert _witness_events(witness) == case.diagnostic_events, context
        assert witness.get("effects", []) == [effect.as_dict() for effect in case.expected_effects], context
        _assert_subset(witness.get("protocol_state", {}), case.protocol_state)
        assert witness["repair"] == case.repair, context

    crosscheck = result["artifacts"]["crosscheck"]
    for engine in ("maximal", "exhaustive"):
        assert crosscheck[engine]["verdict"] == case.expected_verdict, context
        assert crosscheck[engine]["primary_diagnostic"] == case.expected_diagnostic, context

    edges = result["artifacts"]["pdg"]["edges"]
    for edge in case.required_pdg_edges:
        assert edge.as_dict() in edges, {**context, "missing_edge": edge.as_dict()}
    for edge in case.forbidden_pdg_edges:
        assert edge.as_dict() not in edges, {**context, "forbidden_edge": edge.as_dict()}
