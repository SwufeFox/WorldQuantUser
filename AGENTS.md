# AGENTS.md

## Start with the current question

```bash
python research_cli.py brief
```

Read `research_dir`, source existence, open questions and recent evidence before
choosing a direction. Missing or empty local records do not mean a direction is
untested. Check the workspace before drawing conclusions from an empty brief.

Discover only the interface needed next:

```bash
python research_cli.py help                 # command map, only when needed
python research_cli.py help catalog fields  # one task's options and examples
python research_cli.py schema probe         # one command's machine contract
```

The full `schema` is for tooling integrations, not routine session startup.
Do not read `research_cli.py` to learn the interface. The question-to-command map
and research workflow live in [docs/tooling.md](docs/tooling.md); open the relevant
section when needed, rather than loading every recipe.

## Research loop

- Establish the question, known evidence and the next uncertainty to resolve.
  For open exploration, consider a few distinct economic explanations before
  selecting fields or tuning a winning expression. A targeted fix does not need
  a brainstorming ceremony.
- Separate observations, interpretations and untested hypotheses. A high score
  is a lead, not a mechanism. Read the supporting record before treating a
  shortened brief preview as a conclusion.
- Use `tools/unit_report.py --dataset DATASET` and `ledger list` before calling a
  family new. Read failures as well as winners: `rejected` rejects an expression;
  `falsified` records a controlled test of a hypothesis; `no_go` records a block
  before a fair test. Preserve settings, evidence and conditions for reopening.
- Choose an experiment that distinguishes explanations. A scan declares
  `design: explore`; a controlled `probe` declares its control and arm designs,
  and reports arm-minus-control train deltas. `triage` screens on train metrics
  and train PnL; `--show-test` is the explicit validation view after freezing.
  Missing evidence stays unknown rather than being filled from the full window.
- Carry forward decisions, not the conversation. Keep `open-threads.md` in the
  research directory: one `- [open] **question** — evidence; next discriminating
  check` or `- [closed] **question** — conclusion; evidence; reopening condition`
  per line. Put active questions first; details belong in linked reports or the
  ledger. Update the relevant line when evidence changes, not a second summary.

## Execution invariants

1. Always `simulate FILE --dry-run` / `probe FILE --dry-run` before a real run.
   Simulations consume time and platform capacity; account limits are live facts.
   Reuse stored results through `alpha get` / `alpha list` instead of rerunning.
2. Check `auth check` before designing around an unfamiliar account or region.
   Reuse the session cache; use `--relogin` only when it is actually stale.
3. Prefer `catalog fields --expressions` for field wrappers. Inspect structured
   checks and comparisons when deciding what an experiment demonstrated.
4. Train data supports design and selection; view test data after freezing the
   design. Keep validation distinct from another tuning round.
5. Keep credentials out of chat, logs and commits. `.env` and session caches are
   private; the current cache path comes from the CLI, not a remembered filename.
6. stdout is exactly one JSON document, success or failure; stderr is progress
   (and JSONL events with `--stream`). Branch on `error.code`, not message text.
   Exit codes: 0 success, 2 input, 3 auth, 4 service/rate limit, 5 missing,
   6 all simulations failed, 7 submission disabled, 130 interrupted.
7. Code lives in the repo; results, evidence, caches and throwaway scripts live
   in the research directory. Do not add a new tool when an existing command can
   answer the question. Before a new entry point, explain the closest tool, why
   it cannot be extended and the otherwise unanswered recurring question.

## Changing the code

| Concern | Owner |
| --- | --- |
| Invocation, workspace, JSON envelope, saving, exit codes | `research_app/app.py` |
| Arguments, help, generated schema | `research_app/parser.py` |
| Shared errors, output, settings, sessions | `research_app/runtime.py` |
| Simulation, probe, resume, ablation, comparisons | `research_app/experiments.py` |
| Catalog and validation | `research_app/catalog.py`, `research_app/validation.py` |
| Platform alphas and local ledger | `research_app/alphas.py`, `research_app/ledger.py` |
| Briefs, account, literature | `research_app/campaign.py`, `research_app/account.py`, `research_app/literature.py` |
| Research-file contract and BRAIN client | `research_config.py`, `machine_lib.py` |

Put behavior in its owning module and bind it explicitly in `parser.py`. Tests
import that module; `research_cli.py` is executable only. Core libraries must not
import `research_app`. `machine_lib` raises `BrainError` subclasses; never `print()`
on CLI paths or call `input()`. Defaults and allowed settings live in
`machine_lib` and are exposed through schema, with no second copy.

Preserve the JSON envelope and documented error/exit codes. Update meaningful
behavioral coverage in `tests/test_contract.py` for contract changes, and run the
offline suite without credentials:

```bash
python -m unittest discover -s tests -v
```

Update the relevant usage documentation; keep [docs/tooling.md](docs/tooling.md)
accurate when a tool is added, moved or removed.
