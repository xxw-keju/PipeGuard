# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = {
  "admitted": false,
  "case_id": "s2x-triton-sync-hazard-proxy-06",
  "dsl": "triton",
  "oracle": {
    "basis": "independent paired protocol template plus variant-specific participants/region/depth",
    "verdict": "SYNC_HAZARD"
  },
  "polarity": "BUG",
  "primitive": "Proxy",
  "result_class": "SYNC_HAZARD",
  "scenario_id": "WGMMA:MISSING-OPERAND-FENCE",
  "semantic_knobs": {
    "elements": 7,
    "protocol_depth": 7,
    "region_offset_bytes": 96,
    "stages": 2,
    "threads": 96
  },
  "template_family": "WGMMA",
  "variant": 6
}

def pipeguard_resolve():
    s = CASE_SPEC
    family, scenario = s["template_family"], s["scenario_id"]
    elements = s["semantic_knobs"]["elements"]
    threads = s["semantic_knobs"]["threads"]
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        return _cases._memory_kernel(family, elements, threads)
    if family == "MB-REUSE": return _cases._mbarrier_kernel()
    if family in {"CP-WAIT", "CP-OVERLAP"}: return _cases._cp_async_kernel(family, threads)
    if family in {"TMA-OBSERVE", "TMA-PROXY"}: return _cases._tma_load_kernel(family)
    if family == "WGMMA-ENDPOINT": return _cases._wgmma_kernel()
    if family == "TMA-STORE-ENDPOINT": return _cases._tma_store_kernel()
    if family == "CTA-BARRIER": return _cases._named_cta_kernel(scenario, threads)
    if family == "MBARRIER": return _cases._named_mbarrier_kernel(scenario)
    if family == "CP-ASYNC": return _cases._named_cp_async_kernel(scenario, threads)
    if family == "TMA-LOAD": return _cases._named_tma_load_kernel(scenario)
    if family == "WGMMA": return _cases._named_wgmma_kernel(scenario)
    if family == "TMA-STORE": return _cases._named_tma_store_kernel(scenario)
    raise KeyError(f"unsupported template family: {family}")
