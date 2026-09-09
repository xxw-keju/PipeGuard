# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {
  "admitted": false,
  "case_id": "s2x-tilelang-safe-wgmma-01",
  "dsl": "tilelang",
  "oracle": {
    "basis": "independent paired protocol template plus variant-specific participants/region/depth",
    "verdict": "SAFE"
  },
  "polarity": "CORRECT",
  "primitive": "WGMMA",
  "result_class": "SAFE",
  "scenario_id": "WGMMA-ENDPOINT",
  "semantic_knobs": {
    "elements": 2,
    "protocol_depth": 2,
    "region_offset_bytes": 16,
    "stages": 3,
    "threads": 64
  },
  "template_family": "WGMMA-ENDPOINT",
  "variant": 1
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
