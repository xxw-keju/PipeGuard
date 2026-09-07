# Direct PGIR NVIDIA-rule suite

This directory is the primary rule-level checker suite. Every case is constructed
directly as PGIR and does not invoke either the TileLang or Triton frontend.
The older 573-case bounded oracle remains independent supporting evidence; it is
not the source of this catalog.

## Layout

```text
pgir_direct/
  README.md
  cases/
    test_rule_catalog.py
```

Additional suites can be added as sibling directories under `tests/suite/`.

## Catalog size and balance

The catalog contains 120 BUG/CORRECT pairs, for 240 direct PGIR cases:

| Family | NVIDIA rule | Pairs | Cases |
|---|---|---:|---:|
| Cross-lane RAW | MEM-02 | 12 | 24 |
| Cross-lane WAW | MEM-02 | 12 | 24 |
| CTA-barrier handoff | BAR-04 | 12 | 24 |
| Unobserved mbarrier reuse | MB-08 | 12 | 24 |
| TMA completion observation | TMA-04 | 12 | 24 |
| cp.async wait frontier | CP-04 | 12 | 24 |
| Same-group cp.async destination overlap | CP-03 | 12 | 24 |
| Generic-to-TMA proxy handoff | TMA-06 | 12 | 24 |
| WGMMA operand/result endpoints | WG-07/WG-08 | 12 | 24 |
| TMA-store source/global endpoints | TMA-05 | 12 | 24 |
| **Total** |  | **120** | **240** |

For each pair, the CORRECT case differs in the rule-relevant ordering, region,
wait, or fence fact. The BUG case directly violates the listed rule; it is not
merely a randomly malformed PGIR document.

## Mandatory case metadata

Every `RuleCase` is constructed with:

```text
label
pair_id
rule_id
polarity = BUG | CORRECT
expected_verdict
expected_diagnostic
possible_bad_execution
raw PGIR
```

The unique label has the form:

```text
PGIR-<RULE>-<FAMILY>-V<NN>-BUG
PGIR-<RULE>-<FAMILY>-V<NN>-CORRECT
```

For example:

```text
PGIR-MB-08-MB-REUSE-V03-BUG
PGIR-MB-08-MB-REUSE-V03-CORRECT
```

The same label is used as:

1. the pytest parameter ID;
2. the in-memory case identity;
3. `frontend_provenance.source = direct-pgir://<label>` in the constructed PGIR.

Every BUG case includes a short `possible_bad_execution` sequence for human
explanation and failure output. CORRECT cases intentionally carry no bad trace.

## Trust boundary

- Cases are generated from the rule catalog, not from current checker behavior.
- Ordinary program actors are explicit `LANE` actors.
- `(actor, seq)` supplies only same-actor program order.
- Cross-actor order must come from a typed synchronization relation.
- Async completions are synthesized from issue/resource/group identities.
- `engine="crosscheck"` compares optimized maximal exploration with exhaustive
  exploration for every small case.
- The test still asserts an independently declared verdict and diagnostic;
  crosscheck agreement alone is not treated as NVIDIA ground truth.
- Frontend lowering and future compact/parameterized PGIR are tested separately
  against these direct reference cases.

## Future compact representation comparison

Once parameterized logical per-thread streams are integrated, each applicable
pair will gain a compact counterpart. Compact and explicit forms must agree on:

- verdict and primary diagnostic;
- per-anchor boundary sets;
- PDG reachability for named event pairs;
- hazard effects, actors, and regions;
- witness rule/resource identity.

The direct labeled cases remain the reference and must not be weakened to make a
compact implementation pass.
