from __future__ import annotations

from copy import deepcopy

import pytest

from pipeguard.checker import check_module
from pipeguard.model import module_from_dict
from suite.pgir_direct.cases.catalog import CASES, RuleCase


def test_rule_catalog_is_balanced_labeled_and_auditable():
    assert len(CASES) == 240
    assert len({case.label for case in CASES}) == len(CASES)
    assert {case.polarity for case in CASES} == {"BUG", "CORRECT"}
    assert sum(case.polarity == "BUG" for case in CASES) == 120
    assert sum(case.polarity == "CORRECT" for case in CASES) == 120
    assert len({case.pair_id for case in CASES}) == 120
    assert all(case.possible_bad_execution for case in CASES if case.polarity == "BUG")
    assert all(not case.possible_bad_execution for case in CASES if case.polarity == "CORRECT")
    for pair_id in {case.pair_id for case in CASES}:
        pair = [case for case in CASES if case.pair_id == pair_id]
        assert {case.polarity for case in pair} == {"BUG", "CORRECT"}
        assert len(pair) == 2
    for case in CASES:
        assert case.raw["frontend_provenance"]["source"] == f"direct-pgir://{case.label}"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_direct_pgir_nvidia_rule_case(case: RuleCase):
    result = check_module(module_from_dict(deepcopy(case.raw)), engine="crosscheck")
    assert result["verdict"] == case.expected_verdict, {
        "label": case.label,
        "rule": case.rule_id,
        "possible_bad_execution": list(case.possible_bad_execution),
        "result": result,
    }
    if case.expected_diagnostic is not None:
        assert result["diagnostics"][0]["code"] == case.expected_diagnostic, {
            "label": case.label,
            "rule": case.rule_id,
            "possible_bad_execution": list(case.possible_bad_execution),
            "result": result,
        }
