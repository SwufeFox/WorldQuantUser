# Candidate Gates, Submission Planning, Descriptions and Rule Snapshots

Four modules under `research_app/` handle structural Power Pool checks,
multi-candidate submission ordering, structured description assembly and
observed platform-rule persistence. Each module is built directly on the
repository's FASTEXPR AST, check normalization (`check_evidence.py`) and atomic
cache primitives (`cache_store.py`).

| Capability | Module | Responsibility | Live Integration |
| --- | --- | --- | --- |
| Power Pool structural budget | `research_app/pp_budget.py` | Count charged vs. exempt operators (`ts_backfill`, `group_backfill`), data fields (excluding grouping keys) and dataset provenance; combine with observed platform checks. | `alpha get` (`power_pool_budget`), `promote` (`power_pool_budget`), `tools/atom_runner.py` (`pp_budget`) |
| Independent subset & order | `research_app/submission_planner.py` | Solve exact branch-and-bound maximum-weight independent set over a correlation conflict graph (`\|rho\| >= threshold`). | `triage` (`submission_plan` on multi-candidate train screening), `buffer list` (`submission_plan` on shelved candidates) |
| Description assembly & check | `research_app/description.py` | Assemble `IDEA:` / `DATA:` / `RATIONALE:` drafts strictly from supplied catalog notes; verify length, required headers and near-duplicate ratio against existing seeds. | `buffer add` (`description_review`), `promote` (`description_review`) |
| Scoped rule snapshots | `research_app/rules_snapshot.py` | Persist observed check limits, submission quotas and theme multipliers with `source`, UTC `observed_at` and `scope` in `rules-snapshot.jsonl`. | `promote` and `buffer add` (deduplicated updates after live checks) |

## Invariants

All four modules enforce deterministic computation and explicit uncertainty:

* **Power Pool budget (`pp_budget`)**:
  * Counts every function call and infix/unary/ternary operator from `fastexpr_ast`,
    exempting `ts_backfill` and `group_backfill` from the charged count.
  * Prefers the platform's integer `regular.operatorCount` when present (`operator_count_source: "platform"`)
    and labels local counts as `"ast"`.
  * Treats missing field-to-dataset mappings as unresolved issues rather than assuming
    single-dataset membership.
  * In `gate_summary`, formal platform `FAIL` / `ERROR` verdicts always block, Power Pool
    correlation enforces `< 0.5` strictly (`0.5` blocks even if a Regular exception passed),
    and `PENDING` or missing checks remain `unknown` rather than `pass`.
* **Submission planner (`submission_planner`)**:
  * Treats strongly negative correlations (`|rho| >= threshold`) as conflicts (same exposure with opposite sign).
  * Never treats an unmeasured pair as independent: pairs without finite measured correlations
    are listed in `unknown_pairs` and cannot be co-selected into the same independent subset.
  * Operates strictly on caller-supplied train weights and train-period daily PnL correlations.
* **Description compiler (`description`)**:
  * Refuses to produce a draft when `idea`, `fields`, or any referenced field/operator note is missing.
  * Synthesizes default `RATIONALE:` text solely from the supplied inputs without inventing
    polarity or neutralization claims.
  * In `buffer add` and `promote`, `description_review` is attached as non-blocking diagnostic
    evidence so plain-prose Regular descriptions (>= 100 chars) remain accepted while Power Pool
    section and near-duplicate warnings stay visible.
* **Rule snapshots (`rules_snapshot`)**:
  * Parses all ISO-8601 timestamps (`Z`, explicit offsets, fractional seconds) in UTC when computing
    `age_seconds` and `stale`. Invalid or missing timestamps yield `age_seconds: null, stale: null`.
  * Writes `rules-snapshot.jsonl` atomically under a file lock and deduplicates unchanged
    `(name, scope)` check observations.
