from __future__ import annotations

from dataclasses import dataclass

from suite.pgir_direct.cases.catalog import CASES


@dataclass(frozen=True)
class FrontendPairSpec:
    label: str
    pair_id: str
    rule_id: str
    rule_ids: tuple[str, ...]
    family: str
    scenario_id: str
    variant: int
    polarity: str
    expected_stage: str
    expected_verdict: str | None
    expected_diagnostic: str | None
    possible_bad_execution: tuple[str, ...]
    shape: dict[str, int]
    tilelang_entry: str
    triton_entry: str
    tilelang_source_limitation: str | None
    triton_source_limitation: str | None


_TILELANG_SOURCE_LIMITATIONS = {
    "PGIR-MEM-02-NAMED-DANGLING-REGION-BUG": "TileLang rejects the out-of-bounds access before a dangling PGIR region can be formed.",
    "PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG": "T.sync_threads emits a complete new invocation rather than a duplicate arrival to one invocation.",
    "PGIR-BAR-01-NAMED-INCOMPLETE-CTA-PARTICIPANTS-BUG": "The high-level CTA barrier cannot declare incomplete participant metadata.",
    "PGIR-TMA-03-NAMED-GROUPED-BYTES-MISMATCH-BUG": "Capture derives issue bytes and transfer bytes from the same typed access and cannot produce contradictory PGIR metadata.",
    "PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG": "T.tma_copy emits issue and commit atomically, so source cannot insert a wait between them.",
    "PGIR-MB-01-NAMED-REINITIALIZE-LIFECYCLE-BUG": "TileLang exposes allocation-time initialization but no source-level live-barrier reinit primitive.",
    "PGIR-MB-09-NAMED-NONCOALESCED-POLL-BUG": "The blocking TileLang wait API does not expose non-coalesced polling evidence.",
    "PGIR-TMA-03-NAMED-MULTI-TRANSFER-ONE-COMPLETION-BUG": "Capture derives transfer totals from typed TMA accesses and cannot emit contradictory grouped-byte metadata.",
    "PGIR-MB-05-NAMED-ARRIVE-DROP-FAIL-CLOSED-BUG": "TileLang has no public mbarrier arrive_drop primitive.",
    "PGIR-WG-01-NAMED-INVALID-WARPGROUP-SHAPE-BUG": "Native lowering rejects a 64-thread WGMMA before capture.",
    "PGIR-TMA-05-NAMED-STORE-INVALID-SOURCE-BUG": "Native lowering rejects a non-shared TMA-store source before capture.",
}

_TRITON_SOURCE_LIMITATIONS = {
    "PGIR-MEM-02-NAMED-DANGLING-REGION-BUG": "Pointer provenance has no allocation bound from which to construct a dangling PGIR region.",
    "PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG": "ttgl.barrier emits a complete new invocation rather than a duplicate arrival to one invocation.",
    "PGIR-BAR-01-NAMED-INCOMPLETE-CTA-PARTICIPANTS-BUG": "The high-level CTA barrier cannot declare incomplete participant metadata.",
    "PGIR-CP-01-NAMED-INVALID-DESTINATION-SPACE-BUG": "The async-copy API does not accept a non-shared destination descriptor.",
    "PGIR-TMA-03-NAMED-GROUPED-BYTES-MISMATCH-BUG": "Capture derives issue bytes and transfer bytes from the same descriptor and cannot produce contradictory PGIR metadata.",
    "PGIR-WG-03-NAMED-MISSING-OPERAND-FENCE-BUG": "Triton inserts the required WGMMA operand fence during lowering.",
    "PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG": "The TMA-store API lowers issue and commit together, so source cannot insert a wait between them.",
    "PGIR-BAR-01-NAMED-MISSING-PARTICIPANT-BUG": "ttgl.barrier does not expose a partial-participant CTA invocation.",
    "PGIR-MB-09-NAMED-NONCOALESCED-POLL-BUG": "The blocking mbarrier wait API does not expose non-coalesced polling evidence.",
    "PGIR-TMA-03-NAMED-MULTI-TRANSFER-ONE-COMPLETION-BUG": "Capture derives transfer totals from the descriptor and cannot emit contradictory grouped-byte metadata.",
    "PGIR-MB-05-NAMED-ARRIVE-DROP-FAIL-CLOSED-BUG": "Gluon exposes no mbarrier arrive_drop primitive.",
    "PGIR-MB-09-NAMED-STALE-GENERATION-WAIT-BUG": "One straight-line Gluon actor cannot reproduce the late actor's independent logical wait stream.",
    "PGIR-WG-01-NAMED-INVALID-WARPGROUP-SHAPE-BUG": "Native lowering rejects a WGMMA layout whose warp count disagrees with the launch.",
    "PGIR-TMA-05-NAMED-STORE-INVALID-SOURCE-BUG": "Native lowering rejects a pointer in place of a shared-memory descriptor.",
}


def _shape(family: str, variant: int) -> dict[str, int]:
    # Keep compile-only inputs intentionally small. Variants change layout and
    # protocol depth without producing the million-event stress configurations.
    elements = (1, 2, 4, 8)[variant % 4]
    stages = (2, 3, 4)[variant % 3]
    threads = 128 if family in {"WGMMA", "WGMMA-ENDPOINT"} else (32, 64)[variant % 2]
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
        rule_ids=case.rule_ids,
        family=case.family,
        scenario_id=case.scenario_id,
        variant=case.variant,
        polarity=case.polarity,
        expected_stage=case.expected_stage,
        expected_verdict=case.expected_verdict,
        expected_diagnostic=case.expected_diagnostic,
        possible_bad_execution=case.possible_bad_execution,
        shape=_shape(case.family, case.variant),
        tilelang_entry="build_tilelang_case",
        triton_entry="build_triton_case",
        tilelang_source_limitation=_TILELANG_SOURCE_LIMITATIONS.get(case.label),
        triton_source_limitation=_TRITON_SOURCE_LIMITATIONS.get(case.label),
    )
    for case in CASES
)
