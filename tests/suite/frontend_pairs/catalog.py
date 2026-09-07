from __future__ import annotations

from dataclasses import dataclass

from suite.pgir_direct.cases.catalog import CASES


@dataclass(frozen=True)
class FrontendPairSpec:
    label: str
    pair_id: str
    rule_id: str
    family: str
    variant: int
    polarity: str
    expected_verdict: str
    expected_diagnostic: str | None
    possible_bad_execution: tuple[str, ...]
    shape: dict[str, int]
    tilelang_entry: str
    triton_entry: str


def _family_and_variant(pair_id: str) -> tuple[str, int]:
    family, encoded_variant = pair_id.rsplit("-V", 1)
    return family, int(encoded_variant)


def _shape(family: str, variant: int) -> dict[str, int]:
    # Keep compile-only inputs intentionally small. Variants change layout and
    # protocol depth without producing the million-event stress configurations.
    elements = (1, 2, 4, 8)[variant % 4]
    stages = (2, 3, 4)[variant % 3]
    threads = 128 if family == "WGMMA-ENDPOINT" else (32, 64)[variant % 2]
    return {
        "elements": elements,
        "bytes": 4 * elements,
        "stages": stages,
        "threads": threads,
    }


PAIRS: tuple[FrontendPairSpec, ...] = tuple(
    FrontendPairSpec(
        label=case.label,
        pair_id=case.pair_id,
        rule_id=case.rule_id,
        family=_family_and_variant(case.pair_id)[0],
        variant=_family_and_variant(case.pair_id)[1],
        polarity=case.polarity,
        expected_verdict=case.expected_verdict,
        expected_diagnostic=case.expected_diagnostic,
        possible_bad_execution=case.possible_bad_execution,
        shape=_shape(*_family_and_variant(case.pair_id)),
        tilelang_entry="build_tilelang_case",
        triton_entry="build_triton_case",
    )
    for case in CASES
)
