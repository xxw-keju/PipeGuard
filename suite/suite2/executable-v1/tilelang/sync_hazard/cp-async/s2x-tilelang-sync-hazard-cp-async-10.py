# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {
  "admitted": false,
  "case_id": "s2x-tilelang-sync-hazard-cp-async-10",
  "dsl": "tilelang",
  "oracle": {
    "basis": "independent paired protocol template plus variant-specific participants/region/depth",
    "verdict": "SYNC_HAZARD"
  },
  "polarity": "BUG",
  "primitive": "cp.async",
  "result_class": "SYNC_HAZARD",
  "scenario_id": "CP-ASYNC:WAIT-BEFORE-COMMIT",
  "semantic_knobs": {
    "elements": 11,
    "protocol_depth": 11,
    "region_offset_bytes": 160,
    "stages": 3,
    "threads": 96
  },
  "template_family": "CP-ASYNC",
  "variant": 10
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
