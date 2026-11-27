---
name: worldquant-research-cli
description: Aligns agents with the WorldQuant BRAIN v4 research_cli resident daemon, 4-gate state machine, ledger workflow, FASTEXPR rules, and structured error recovery. Use before running research_cli.py commands, proposing specs, handling inbox gates, analyzing the ledger, or planning submissions.
---

# WorldQuant Research CLI & v4 State-Machine Workflow Alignment

Follow the repository's v4 resident daemon and 4-gate state machine instead of guessing flags, spawning competing background loops, or bypassing gates. Always stay inside the workspace root and pass the same `--research-dir PATH` across commands when working with a non-default research directory.

## 1. Pre-Flight Alignment (Before Any Spec or Command)

1. Start/check the resident daemon and inspect workspace state first:
   - `python research_cli.py daemon start`
   - `python research_cli.py brief`
   - `python research_cli.py inbox`
2. Never guess flags, payload schemas, or error meanings from memory. Run:
   - `python research_cli.py help` or `python research_cli.py help <command>` for options and examples.
   - `python research_cli.py schema <command>` for the machine contract and `ERROR_CODES` / `EXIT_CODES`.

## 2. Repository Documentation Map

Read only the document needed for the current task (all links are relative to this skill file):

| Need | Read |
| --- | --- |
| Non-negotiable execution rules & code ownership | [AGENTS.md](../../../AGENTS.md) |
| Architecture overview, install, `.env`, JSON contract | [README.md](../../../README.md) |
| v4 4-gate state machine, Train/Test firewall, Power Pool submit chain | [docs/ledger-workflow.md](../../../docs/ledger-workflow.md) |
| Full CLI command reference & companion `tools/` scripts | [docs/tooling.md](../../../docs/tooling.md) |
| Structured errors (`ERROR_CODES`), gate diagnosis & `UNCERTAIN` recovery | [docs/errors.md](../../../docs/errors.md) |
| Throughput diagnostics, shared slot budgets & concurrency benchmarks (中文) | [docs/fast-backtest.md](../../../docs/fast-backtest.md) |
| FASTEXPR AST parser, expression identity & source-analysis contract | [docs/ast-pipeline.md](../../../docs/ast-pipeline.md) |
| FASTEXPR construction units & compression report | [docs/unit-compression.md](../../../docs/unit-compression.md) |
| Power Pool structural budgets, independent-subset planning & rule snapshots | [docs/tool-absorption.md](../../../docs/tool-absorption.md) |
| Workspace backup, archive & replay verification | [docs/backup-replay.md](../../../docs/backup-replay.md) |

## 3. v4 4-Gate Execution Path & Hard Invariants

- **Canonical short non-blocking loop**:
  1. `daemon start` -> `brief` / `inbox` -> `propose SPEC.json` (offline AST/Python validation; enters `AUTO_PREFLIGHT`).
  2. `wait --max-wait 30` (wakes on condition variable when a run hits `GATE_*`) -> `inspect RUN_ID`.
  3. **Gate 1 (`GATE_TRIAGE`)**: `inspect` shows train-only metrics (protocol firewall). Call `act RUN_ID hydrate-pnl --top 3` (or `--arms ...`, excluding diagnostic/control arms) -> `AUTO_HYDRATE_PNL`.
  4. **Gate 2 (`GATE_APPROVE_FULL`)**: Require explicit passing yearly train evidence, then call `act RUN_ID approve-full [--arms LABEL]` for one arm (freezes `runs/<run_id>-selection.json` and opens `test_view_open`), `back-to-triage`, or `drop --verdict rejected|falsified|no_go`.
  5. `AUTO_FULL_VALIDATE` runs FULL sim, live checks, self-correlation, and 5/10/20 bps cost audit; failing candidates automatically transition to `REJECTED_AT_VALIDATION`.
  6. **Gate 3 (`GATE_DESCRIBE`)**: Call `act RUN_ID describe --name ... --category ... --tags ... --description ...` with exact headings `Idea:`, `Rationale for data used:`, `Rationale for operators used:`, each followed by `- ` bullets, >= 100 trimmed characters.
  7. **Gate 4 (`GATE_CONFIRM_SUBMIT`)**: Candidate is already shelved in `submit-buffer.jsonl` and `seeds.jsonl`. Call `act RUN_ID hold` (-> `SHELVED`) or `act RUN_ID submit --yes` (requires `BRAIN_ALLOW_AGENT_SUBMIT=true`).
- **Single-writer daemon**: Never spawn duplicate `daemon _serve` instances or manually edit `workspace.db`, `daemon/*.lock`, or frozen selections. Legacy v3 `backlog/` and `backlog-results/` directories can be imported via `migrate-workspace`.
- **BRAIN Labs**: For repeated simulation anomalies, uncertain field coverage or distribution, and cross-field interactions, open `labs open` or `act RUN_ID request-labs`. Specify scope, field IDs, quantitative questions, aggregate statistics and conditional next steps. An OPEN ticket blocks overlapping proposals. Receive only aggregate evidence through `labs resolve`.
- **Immutable identity once proposed**: Never rename arm labels or tweak syntax to evade `request_fingerprint` duplicates or `GATE_PREFLIGHT_BLOCKED`. Inspect `preflight_block` and either `act RUN_ID retry-preflight` or `drop`.

## 4. Error & Deferral Handling

`research_cli.py` writes one JSON object to `stdout` (`stderr` holds progress events). Check **both** `ok` and structured state fields, consulting [docs/errors.md](../../../docs/errors.md):

1. **When `ok` is `false`**:
   - Branch on `error.code`, follow `error.hint`, and inspect `error.details`.
   - Never bypass `SUBMIT_DISABLED` or `CONFIRM_REQUIRED` without explicit authorization (`--yes` + `BRAIN_ALLOW_AGENT_SUBMIT=true`).
2. **When `AUTO_*` defers or `UNCERTAIN` occurs**:
   - Check `next_wake_at`, `unchanged_defers`, and `stalled`. Do not loop rapidly or issue duplicate POSTs when `submit_attempt.state == "UNCERTAIN"`; let the daemon run its read-only readback reconciliation.

## 5. Code Modification Checklist

If changing code inside the repository (see ownership table in [AGENTS.md](../../../AGENTS.md)):
- Keep `research_app` and `tools` imports out of core modules (`*.py` at repo root and `brain_core/*.py`, excluding `research_cli.py`); keep `tools` imports out of `research_app/*.py`.
- Raise `CliError(code, message, hint=..., details=...)` with an actionable `hint` in CLI handlers and ensure every public module, class, and function has a docstring.
- Run `python -m pytest -q` (offline suite, no credentials required) after edits.
