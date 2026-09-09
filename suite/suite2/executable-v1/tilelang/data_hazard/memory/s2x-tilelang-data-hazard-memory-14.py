# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {
  "admitted": false,
  "case_id": "s2x-tilelang-data-hazard-memory-14",
  "dsl": "tilelang",
  "oracle": {
    "basis": "independent paired protocol template plus variant-specific participants/region/depth",
    "verdict": "DATA_HAZARD"
  },
  "polarity": "BUG",
  "primitive": "Memory",
  "result_class": "DATA_HAZARD",
  "scenario_id": "MEM-WAW",
  "semantic_knobs": {
    "elements": 15,
    "protocol_depth": 15,
    "region_offset_bytes": 224,
    "stages": 4,
    "threads": 96
  },
  "template_family": "MEM-WAW",
  "variant": 14
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
