# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {
  "admitted": false,
  "case_id": "s2x-tilelang-data-hazard-tma-04",
  "dsl": "tilelang",
  "oracle": {
    "basis": "independent paired protocol template plus variant-specific participants/region/depth",
    "verdict": "DATA_HAZARD"
  },
  "polarity": "BUG",
  "primitive": "TMA",
  "result_class": "DATA_HAZARD",
  "scenario_id": "TMA-OBSERVE",
  "semantic_knobs": {
    "elements": 5,
    "protocol_depth": 5,
    "region_offset_bytes": 64,
    "stages": 3,
    "threads": 32
  },
  "template_family": "TMA-OBSERVE",
  "variant": 4
}

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
