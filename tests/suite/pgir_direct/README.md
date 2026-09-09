# Direct PGIR NVIDIA-rule suite

This is one unified suite of small, directly constructed PGIR kernels. It tests
checker semantics without invoking either the TileLang or Triton frontend. The
older 573-case bounded oracle remains independent supporting evidence; it is not
the source of this catalog.

## Runner and catalog

`cases/catalog.py` exports the single public `CASES` registry. The one
parameterized runner in `cases/test_rule_catalog.py` executes every case. A case
may expect rejection by `module_from_dict` or a verdict from `check_module`, but
these are execution stages inside the same suite, not separate suites.

```bash
PYTHONPATH=src:tests /usr/bin/python3 -m pytest -q tests/suite/pgir_direct/cases
```

## Size versus semantic diversity

The catalog contains:

```text
290 direct PGIR cases
133 complete BUG/CORRECT pairs
24 additional focused BUG or CORRECT controls
49 distinct semantic scenarios
27 associated NVIDIA rule IDs
280 checker-stage cases
10 model-stage fail-closed cases
```

The original 240 cases and labels are retained. They contain 120 pairs but only
12 distinct protocol structures; their repeated variants remain useful for
extent, offset, byte-count, and participant regression. Fifty newer cases add 37
named state-machine scenarios, including 13 additional complete pairs. Raw case
count is therefore never reported as a substitute for semantic coverage.

The current expected outcomes are:

```text
140 SAFE
117 DATA_HAZARD
23 SYNC_HAZARD
10 model-stage PGIRError
```

## Added semantic scenarios

| Domain | Distinct named scenarios | Main dimensions |
|---|---:|---|
| Memory | 5 | partial WAR, containment WAW, adjacency, allocation identity, dangling region |
| CTA barrier | 5 | complete handoff, control-only handoff, duplicate arrival, missing/incomplete participants |
| mbarrier | 10 | initialization, transaction accounting, two-generation reuse, invalidate/reinit, acquire-until-true evidence, parity wrap, stale generation, arrive-drop fail-closed |
| cp.async | 4 | wait-before-commit, destination-space validation, one-group frontier, two-group `wait_group 1` frontier |
| TMA load | 6 | destination observation, source proxy, grouped-byte validation, multi-transfer one-completion accounting, complete handoff |
| WGMMA | 4 | participant validation, shared operand-fence model, dual endpoints, two-group result frontier |
| TMA store | 4 | wait-before-commit, source-space validation, dual endpoints, two-group global frontier |
| **Named total** | **37** | plus 12 preserved protocol structures = 49 total |

Associated rule IDs are:

```text
BAR-01 BAR-03 BAR-04
MB-01 MB-03 MB-05 MB-06 MB-07 MB-08 MB-09 MB-10
CP-01 CP-02 CP-03 CP-04
TMA-03 TMA-04 TMA-05 TMA-06
WG-01 WG-03 WG-06 WG-07 WG-08
MEM-01 MEM-02 MEM-04
```

A rule ID means that the case exercises the explicitly represented subset of
that rule. It does not imply that all PTX forms of the rule are modeled.

## Case contract

Every `RuleCase` records:

```text
label, pair_id, family, scenario_id, variant, rule_ids, polarity
expected_stage, expected_verdict, expected_diagnostic
diagnostic_events, expected_effects, protocol_state
required_pdg_edges, forbidden_pdg_edges
semantic_dimensions, repair
possible_bad_execution, raw PGIR
```

The label is the pytest ID, in-memory identity, and
`frontend_provenance.source = direct-pgir://<label>`. BUG cases include an
auditable bad-execution sequence. BUG/CORRECT pairs share a scenario identity and
the CORRECT member repairs the rule-relevant region, order, wait, fence,
participant, or lifecycle fact.

## Assertion contract

The unified runner enforces:

- exact model-stage `PGIRError.code` for unsupported or malformed evidence;
- independently declared checker verdict and primary diagnostic under
  `engine="crosscheck"`;
- exact diagnostic and witness event IDs;
- exact hazard effect IDs, actors, modes, regions, and begin/end windows;
- selected protocol-state fields and required/forbidden typed PDG edges;
- empty diagnostics and no witness for every SAFE result;
- rule-specific structured evidence for every checker-stage BUG, rather than
  accepting an unrelated `UNORDERED_CONFLICT`.

Maximal/exhaustive agreement is an implementation crosscheck, not NVIDIA ground
truth. Expected results come from the NVIDIA rule ledger, the independent bounded
oracle where applicable, and manually auditable protocol traces.

## Trust boundary

- Program order is per PGIR actor; ordinary examples use explicit lane actors.
- Cross-actor visibility requires a typed synchronization or proxy relation.
- Async effects remain open until their typed completion endpoint.
- One logical TMA load with multiple transfer regions contributes one phase
  completion while accounting for every transfer byte.
- Hardware parity does not replace logical mbarrier generation identity.
- `UNSUPPORTED` or schema/model rejection is never counted as SAFE.

The following remain intentionally limited:

- BAR-02 aligned convergence and WG-02 aligned convergence are not represented.
- MB-05 `arrive_drop` is tested only as fail-closed, not semantically executed.
- TMA-02 full descriptor/range legality remains frontend evidence, not core PGIR.
- The current `WgmmaFenceOperand` checks shared-region coverage; it is not a full
  NVIDIA WG-03 register-fence model.
- New WGMMA correctness cases use full result completion conservatively; they do
  not claim that an earlier independent operand-release endpoint is established
  by the NVIDIA evidence.
- MEM-02 covers exact-region conflict/ordering subsets, not full PTX
  morally-strong strength/scope semantics.

All 290 labels are consumed by Suite 2 through explicit `family` and
`scenario_id` metadata; there is no direct-only frontend coverage flag.
