# AGENTS.md

## Follow one execution path

```bash
python research_cli.py brief
python research_cli.py campaign status --limit 10
```

Read `research_dir`, source existence, open questions and recent evidence before
choosing a direction. Missing or empty local records do not mean a direction is
untested. Use the same research directory for every step.

| Step | Command and decision |
| --- | --- |
| Inspect | `brief` and `campaign status` identify existing work and its canonical batch. Recover that work before drafting a replacement. |
| Prepare | Put the question in the spec's `idea`, declare its mechanism and stable arm labels, then `campaign preview SPEC.json`. Read `would_block`, `blocking`, `gate_blocking` and the returned recovery references. |
| Execute | Put the reviewed spec in `backlog/`; one owner runs `campaign run --watch --defer-quick-pnl --stream`. That owner handles submission and recovery. |
| Review | `ledger analyze --group-by scope`, then narrow to the mechanism and inspect failures as well as usable results. |
| Validate | After FULL completes, write the official-template description with `alpha describe`, then run `buffer add` (or `promote`, which calls it) with `--name`, `--category`, `--tags` and `--description`. Only then may the authorized `alpha submit` entry point be called twice. |

After execution starts, keep the spec, arm labels, settings and batch path fixed.
Use the returned checkpoint path, not a reconstructed filename. An edited
hypothesis is a new spec; a retry is the same checkpoint. Do not rename arms or
switch runners to evade a duplicate or unresolved submission.

Subagents may draft separate specs, inspect receipts and analyse results. Only
one owner advances a queue or batch. While Campaign owns it, other agents use
read-only status/inspection; they do not start another Campaign, direct Probe
resume, or compatibility runner against the same work. Hand off the checkpoint
after the existing writer exits. A lock file's presence is not proof of a live owner.

The step-by-step guide and state table live in
[docs/ledger-workflow.md](docs/ledger-workflow.md). Use `help COMMAND` for the
next command's options and examples; use `schema COMMAND` for its machine
contract. The full schema and source files are not startup reading.
For throughput, shared budgets and concurrency measurements, read the Chinese
[docs/fast-backtest.md](docs/fast-backtest.md) guide.

## Read progress literally

- `ok: true` means the command was handled. For a saved batch, start with
  Campaign's current `jobs[].diagnosis` and its `next_action.argv`; a fresh spec
  without a batch has `state: queued`. Use saved task IDs and Probe's
  `execution` / `next_actions` to inspect the underlying evidence.
- Accepted simulation: recover the original task. Unknown simulation POST:
  reconcile its identity before any new POST. Final-alpha `UNCERTAIN` is a
  separate submission state; read its saved submission outcome.
- A local wait budget ending does not cancel a platform task. A retry time is
  a reason to wait, not a reason to start another resume loop.
- Report changes since the previous observation: newly accepted arms, new
  terminal results, new recorded evidence, the blocker and the next action/time.
  Repeated deferred receipts, a live process or growing logs are not output.
- `last_attempt` and `last_runner_event` are historical observations. Check
  `matches_current_progress` before using an old attempt as current evidence.
  `progress.stalled` requests diagnosis; it does not authorize an unlock or POST.
- If receipts stop changing, say that there is no new progress and identify
  what evidence is missing. Do not silently repeat, delete locks or release
  ambiguous reservations by age. An OMP/subagent wait is host orchestration;
  report the pending agent/task separately from BRAIN and machine health.
- Terminal execution failures belong in the same ledger with their input and
  error evidence; they are not automatic scientific rejections. Check
  `recording.recorded_failures` and `recording.missing_idea_count`. An idea comes
  from the arm or spec, never from a guessed explanation of its filename.
- The compatibility Quick exploration runner checks the whole input before new
  POSTs: a nonempty `idea` on every arm, at most eight arms per AST skeleton, and
  at least three datasets supported by account/scoped field evidence. Read its
  `.admission.json` receipt when blocked; `family` labels and field prefixes are
  not catalog evidence. Existing accepted/unknown tasks and historical recording
  remain recoverable. These exploration rules do not redefine a controlled
  two-arm Probe or FULL validation. See the Quick section in `docs/tooling.md`.
- Research seeds are the ledger's experiment records, including failed attempts.
  Submission-candidate metadata belongs to the existing buffer/seed store. The
  exact order is FULL complete → `alpha describe` writes the platform description
  → `buffer add --name --category --tags --description` stores the seed and checks
  → `alpha submit` is issued twice. `buffer add` rereads the platform description
  and requires an exact match, so the remote write is not optional.
  Every candidate description must use the official Power Pool template, with the
  exact headings `Idea:`, `Rationale for data used:` and `Rationale for operators
  used:`; each heading is followed by a `- ` bullet list. The old all-caps `IDEA:`
  / `RATIONALE:` form is invalid. The overall description must contain at least
  100 characters after trimming outer whitespace. Describe the actual hypothesis
  and evidence; do not invent a rationale to fill the field or maintain another
  seed-tracking loop. A seed check with any `FAIL` blocks shelving and submission;
  for example, `LOW_FITNESS 0.63` stays out of the seed store.
  `metadata_required` means validation reached missing candidate metadata; append
  the listed `required_options` to `next_action.argv` and reuse its frozen choice.
  It does not mean the candidate was buffered or that a new simulation is needed.

## Research loop

- Establish the question, known evidence and the next uncertainty to resolve.
  For open exploration, consider a few distinct economic explanations before
  selecting fields or tuning a winning expression. A targeted fix does not need
  a brainstorming ceremony.
- Within the authorized research task, resolve reachable prerequisites and prepare
  the next discriminating check proactively. Batch independent reads; use bounded
  concurrency for retrieval. Treat `feed.next_actions` as evidence-linked leads,
  not automatic permission to simulate or submit. Reconcile contrary evidence
  before spending another round on the current winner.
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
- For Python, declare `language: PYTHON` and the required `lookback`. Declare
  BRAIN inputs in `@alpha(data=[...])`. Use the .py entry points or a
  spec-relative `source_file`. Preflight parses the source without executing it.
  Use an explore design to compare languages and align the PnL date ranges
  before deciding whether two implementations agree.
- Carry forward decisions, not the conversation. Keep `open-threads.md` in the
  research directory: one `- [open] **question** — evidence; next discriminating
  check` or `- [closed] **question** — conclusion; evidence; reopening condition`
  per line. Put active questions first; details belong in linked reports or the
  ledger. Update the relevant line when evidence changes, not a second summary.

## Execution invariants

1. Preview before a real run: `campaign preview FILE` for the default queue,
   or `simulate FILE --dry-run` / `probe FILE --dry-run` for a standalone task.
   Simulations consume time and platform capacity; account limits are live facts.
   Reuse stored results through `alpha get` / `alpha list` instead of rerunning.
   Probe preview can point to past results and a recoverable job in the
   expected batch file. References do not start jobs or substitute results.
   If a run stops, inspect its saved progress first. The execution owner uses
   `probe --resume BATCH` to advance known queued/accepted work and
   `probe --recover BATCH` for read-only platform recovery. Standalone simulation
   jobs use `resume CHECKPOINT`. To inspect a
   platform task started elsewhere, use `probe --from-simulation ID` for child
   states and Alpha IDs. Reuse saved evidence when stdout was lost.
2. Check `auth check` before designing around an unfamiliar account or region.
   Reuse the session cache; use `--relogin` only when it is actually stale.
3. Prefer `catalog fields --expressions` for field wrappers. Inspect structured
   checks and comparisons when deciding what an experiment demonstrated.
4. Train data supports design and selection; view test data after freezing the
   design. Campaign freezes its measured training gate before calling
   `promote --selection`. If required evidence is unknown, the worker defers
   the spec and does not open the test view.
5. Keep credentials out of chat, logs and commits. `.env` and session caches are
   private; the current cache path comes from the CLI, not a remembered filename.
6. Structured commands, including `help` and `schema`, write one JSON document
   to stdout. Terminal `--help` and `--version` write text. Progress and
   `--stream` events go to stderr. Branch on `error.code` for structured errors.
   See `schema` for the current error and exit codes.
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
| Read-only inspection of existing simulation tasks | `research_app/simulation_import.py` |
| Catalog and validation | `research_app/catalog.py`, `research_app/validation.py` |
| Training selection, promotion, candidate shelf | `research_app/triage.py`, `research_app/selection.py`, `research_app/promote.py`, `research_app/buffer_cmd.py` |
| Power Pool budget, submission planning, descriptions, rule snapshots | `research_app/pp_budget.py`, `research_app/submission_planner.py`, `research_app/description.py`, `research_app/rules_snapshot.py` |
| Platform alphas and local ledger | `research_app/alphas.py`, `research_app/ledger.py`, `research_app/ledger_analysis.py`, `registry_index.py` |
| Campaigns, briefs, feeds, account, literature | `research_app/campaign.py`, `research_app/feed.py`, `research_app/account.py`, `research_app/literature.py` |
| Research-file contract, check normalization, slots and BRAIN client | `research_config.py`, `check_evidence.py`, `simulation_slots.py`, `simulation_multi.py`, `machine_lib.py` |

Put behavior in its owning module and bind it explicitly in `parser.py`. Tests
import that module; `research_cli.py` is executable only. Core libraries must not
import `research_app`. `machine_lib` raises `BrainError` subclasses; never `print()`
on CLI paths or call `input()`. Defaults and allowed settings live in
`machine_lib` and are exposed through schema, with no second copy.

Preserve the JSON envelope and documented error/exit codes. Update meaningful
behavioral coverage in `tests/test_contract.py` for contract changes, and run the
offline pytest suite without credentials:

```bash
python -m pytest -q
```

Update the relevant usage documentation; keep [docs/tooling.md](docs/tooling.md)
accurate when a tool is added, moved or removed.
