from __future__ import annotations

from suite.frontend_pairs.catalog import PAIRS


def test_frontend_pair_catalog_covers_every_direct_case_twice():
    assert len(PAIRS) == 240
    assert len({pair.label for pair in PAIRS}) == 240
    assert sum(pair.polarity == "BUG" for pair in PAIRS) == 120
    assert sum(pair.polarity == "CORRECT" for pair in PAIRS) == 120
    assert len(PAIRS) * 2 == 480
    assert all(pair.tilelang_entry == "build_tilelang_case" for pair in PAIRS)
    assert all(pair.triton_entry == "build_triton_case" for pair in PAIRS)


def test_frontend_pair_shapes_stay_small_and_wgmma_keeps_a_full_warpgroup():
    for pair in PAIRS:
        assert pair.shape["elements"] in {1, 2, 4, 8}
        assert pair.shape["stages"] in {2, 3, 4}
        assert pair.shape["threads"] in {32, 64, 128}
        if pair.family == "WGMMA-ENDPOINT":
            assert pair.shape["threads"] == 128
        else:
            assert pair.shape["threads"] <= 64
