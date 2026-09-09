from __future__ import annotations

from suite.frontend_pairs.catalog import PAIRS
from suite.pgir_direct.cases.catalog import CASES


def test_frontend_pair_catalog_covers_every_direct_case_twice():
    assert len(PAIRS) == 290
    assert len({pair.label for pair in PAIRS}) == 290
    assert {pair.label for pair in PAIRS} == {case.label for case in CASES}
    assert {pair.expected_stage for pair in PAIRS} == {"CHECKER", "MODEL"}
    assert sum(pair.expected_stage == "MODEL" for pair in PAIRS) == 10
    assert sum(pair.polarity == "BUG" for pair in PAIRS) == 150
    assert sum(pair.polarity == "CORRECT" for pair in PAIRS) == 140
    assert len(PAIRS) * 2 == 580
    assert all(pair.tilelang_entry == "build_tilelang_case" for pair in PAIRS)
    assert all(pair.triton_entry == "build_triton_case" for pair in PAIRS)
    assert sum(pair.tilelang_source_limitation is not None for pair in PAIRS) == 11
    assert sum(pair.triton_source_limitation is not None for pair in PAIRS) == 14
    assert all(
        pair.polarity == "BUG"
        for pair in PAIRS
        if pair.tilelang_source_limitation is not None
        or pair.triton_source_limitation is not None
    )


def test_frontend_pair_shapes_stay_small_and_wgmma_keeps_a_full_warpgroup():
    for pair in PAIRS:
        assert pair.shape["elements"] in {1, 2, 4, 8}
        assert pair.shape["stages"] in {2, 3, 4}
        assert pair.shape["threads"] in {32, 64, 128}
        if pair.family in {"WGMMA", "WGMMA-ENDPOINT"}:
            assert pair.shape["threads"] == 128
        else:
            assert pair.shape["threads"] <= 64
