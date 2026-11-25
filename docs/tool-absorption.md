# Absorbed tooling from `Pang-Ke/wqb-agent-tools`

Audited 2026-11-25 against the reference repository (28 Python files, ~5.7k lines)
and this repository at `46d6447`. The reference is a thin CLI over the BRAIN
API; most of it duplicates commands we already own (`probe`, `campaign`,
`triage`, `promote`, `alpha check`, `docs`). Four designs were kept because they
answer questions our code answered only indirectly or not at all.

Nothing was ported. Each module below is written against our own AST, our own
evidence rules and our own tests.

| Design | Reference | Ours before | Gap | Absorbed as |
| --- | --- | --- | --- | --- |
| Power Pool budget | `pp.py`: `operator_count`, `pp_budget` | `pp_eligibility.py` only read platform classifications; `expression_lint.estimate_operator_count` was approximate and had no exemption for `ts_backfill`/`group_backfill` | no structural budget, no field/dataset gate | `research_app/pp_budget.py` |
| Submission planner | `corr.py` + `readiness.py` (correlation helpers, no selection) | `independence.py` measured marginal Sharpe and calm/stressed correlation; no ordering or subset choice | no independent-set selection over a conflict graph | `research_app/submission_planner.py` |
| Description compiler | `readiness.py`: `draft_pp_description`, `check_description` | only a length gate in the buffer/seed store | no assembly from catalog text, no duplication check | `research_app/description.py` |
| Rule snapshot | `readiness.py`: `active_rules` (read at call time) | `check_evidence.py` normalised one payload; nothing kept thresholds, quotas or theme multipliers with provenance | rules were re-read and forgotten | `research_app/rules_snapshot.py` |

## What each module refuses to do

All four follow the same rule: deterministic computation, explicit evidence,
no scoring, and a missing input stays unknown.

* `pp_budget.pp_budget` counts calls and infix operators from the AST, exempts
  `ts_backfill`/`group_backfill`, excludes grouping keys from the field count,
  and reports a field whose dataset is unknown as an issue rather than assuming
  another field's dataset. It prefers the platform's `regular.operatorCount`
  when it has one and says so in `operator_count_source`.
* `submission_planner.plan` maximises total supplied weight over a conflict
  graph defined by `|rho| >= threshold` (a strong negative correlation is the
  same position flipped). A candidate without a weight, or a pair without a
  measured correlation, is reported as unknown and never treated as independent.
  The output is a suggestion with its evidence, not an authorization.
* `description.compose` assembles `IDEA / DATA / RATIONALE` only from catalog
  descriptions the caller supplies. A field or operator without one is reported
  in `missing` and no draft is produced; `check` then verifies length (100),
  both sections, and near-duplicate similarity against existing descriptions.
* `rules_snapshot` records every threshold, quota and theme multiplier with
  `source`, `observed_at` and `scope`, and `lookup` returns `None` or a `stale`
  flag instead of a default. `official_rule` and `measured_rule` keep the
  difference between a documented rule and an inference with a sample size.

## Evidence rules

* Platform check values are authoritative when present; local AST counts are
  labelled `ast` and never presented as the platform's number.
* Absence is reported as `unknown` and blocks `gate_summary`, matching the
  existing rule that missing evidence is not a pass.
* No train/test mixing: these modules never read test-period PnL. The planner
  takes correlations the caller measured; `triage --show-test` remains the only
  explicit test view.

## Not wired into the CLI yet

The four modules are library-level. They are imported by tests and are intended
to be called from `promote`, `triage` and the campaign review path once their
inputs (catalog descriptions, measured correlations) are wired. Adding commands
would change the CLI contract, and `tests/test_contract.py` is the gate for
that; that change is deliberately not part of this absorption.