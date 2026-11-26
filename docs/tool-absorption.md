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
| Description assembly & check | `research_app/description.py` | Assemble the official three-section Power Pool description (`Idea:`, `Rationale for data used:`, `Rationale for operators used:`) strictly from supplied catalog notes; verify exact headings, `-` bullet lists, length and near-duplicate ratio against existing seeds. | `alpha readiness`, `alpha describe`, `buffer add` (`description_review`), `promote` (`description_review`) |
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
  * Uses the exact headings from `getting-started-power-pool-alphas`: `Idea:`,
    `Rationale for data used:` and `Rationale for operators used:`. Each heading must be on its
    own line and followed by at least one `- ` bullet; the old all-caps `IDEA:` / `RATIONALE:`
    form is rejected.
  * Synthesizes default data and operator rationale text solely from the supplied inputs without inventing
    polarity or neutralization claims.
  * `alpha describe`, `buffer add`, `promote`, the seed store and new submissions require a valid
    official template. `buffer add` also requires the platform `regular.description` to match
    `--description`, so writing the description remotely is a real prerequisite rather than an
    optional convention. Near-duplicate similarity remains visible in `description_review`.
* **Rule snapshots (`rules_snapshot`)**:
  * Parses all ISO-8601 timestamps (`Z`, explicit offsets, fractional seconds) in UTC when computing
    `age_seconds` and `stale`. Invalid or missing timestamps yield `age_seconds: null, stale: null`.
  * Writes `rules-snapshot.jsonl` atomically under a file lock and deduplicates unchanged
    `(name, scope)` check observations.
