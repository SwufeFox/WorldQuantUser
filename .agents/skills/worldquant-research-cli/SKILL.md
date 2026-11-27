---
name: worldquant-research-cli
description: Aligns agents with the WorldQuant BRAIN research_cli toolchain, ledger workflow, FASTEXPR rules, and structured error recovery. Use before running research_cli.py commands, drafting probe or campaign specs, checking campaign/probe batches, analyzing the ledger, or promoting and submitting alphas.
---

# WorldQuant Research CLI & Workflow Alignment

Follow the repository's documented toolchain instead of guessing flags, inventing parallel runners, or retrying blindly after errors. Always stay inside the workspace root and pass the same `--research-dir PATH` across commands when working with a non-default research directory.

## 1. Pre-Flight Alignment (Before Any Spec or Command)

1. Run `python research_cli.py brief` and `python research_cli.py campaign status --limit 10` first to inspect `research_dir`, ledger paths, queued specs, open questions, and recoverable batches.
2. Never guess flags, payload schemas, or error meanings from memory. Run:
   - `python research_cli.py help` or `python research_cli.py help <command>` for options and examples.
   - `python research_cli.py schema <command>` for the machine contract and `ERROR_CODES` / `EXIT_CODES`.
3. Preview every batch or standalone run before hitting BRAIN:
   - Default queue: `python research_cli.py campaign preview SPEC.json`
   - Standalone: `python research_cli.py simulate SPEC.json --dry-run` or `probe SPEC.json --dry-run`

## 2. Repository Documentation Map

Read only the document needed for the current task (all links are relative to this skill file):

| Need | Read |
| --- | --- |
| Non-negotiable execution rules & code ownership | [AGENTS.md](../../../AGENTS.md) |
| Architecture overview, install, `.env`, JSON contract | [README.md](../../../README.md) |
| Step-by-step batch research, recovery, ledger review, Power Pool submit chain | [docs/ledger-workflow.md](../../../docs/ledger-workflow.md) |
| Bulk QUICK pilot/expansion publication (`campaign publish`), `catalog inventory` & `campaign diagnose` | [docs/research-campaign-queue.md](../../../docs/research-campaign-queue.md) |
| Full CLI command reference & companion `tools/` scripts | [docs/tooling.md](../../../docs/tooling.md) |
| Structured errors (`ERROR_CODES`), `campaign status` diagnosis & safe recovery | [docs/errors.md](../../../docs/errors.md) |
| Throughput diagnostics, shared slot budgets & concurrency benchmarks (中文) | [docs/fast-backtest.md](../../../docs/fast-backtest.md) |
| FASTEXPR AST parser, expression identity & source-analysis contract | [docs/ast-pipeline.md](../../../docs/ast-pipeline.md) |
| FASTEXPR construction units & compression report | [docs/unit-compression.md](../../../docs/unit-compression.md) |
| Power Pool structural budgets, independent-subset planning & rule snapshots | [docs/tool-absorption.md](../../../docs/tool-absorption.md) |
| Workspace backup, archive & replay verification | [docs/backup-replay.md](../../../docs/backup-replay.md) |

## 3. Single Execution Path & Hard Invariants

- **Canonical sequence**: `brief` -> `campaign status` (or `campaign diagnose` for nested `--campaign-dir`) -> `catalog inventory` / `catalog find` -> `campaign preview SPEC.json` (or `campaign publish` for 16-arm QUICK pilots / `--phase expansion --pilot-batch`) -> single owner executes `campaign run --watch --defer-quick-pnl --stream` -> `ledger analyze --group-by scope` -> targeted `--yearly` PnL & FULL spec -> `alpha describe` -> `buffer add` (or `promote`) -> `alpha submit` twice.
- **Single owner per batch**: Subagents may draft separate specs and run read-only status/inspection (`campaign status`, `ledger analyze`, `probe --recover`), but never start a second writer against the same queue or checkpoint.
- **Immutable identity once launched**: Never rename arm labels, switch runners, or make cosmetic FASTEXPR edits to evade `would_block`, `gate_blocking`, or `exact_repeat`. Recover the saved checkpoint path instead.
- **Train vs. Test discipline**: Use training metrics (`triage --freeze`) before opening `--show-test` or calling `promote --selection`. Missing evidence stays unknown (`missing`/`stale`/`partial`/`unavailable`) rather than being filled with full-IS numbers.
- **Power Pool descriptions**: Write descriptions with `alpha describe` BEFORE `buffer add` or `promote`. Headings must match verbatim: `Idea:`, `Rationale for data used:`, `Rationale for operators used:`, each followed by `- ` bullets, at least 100 trimmed characters total. Any `FAIL` check blocks buffering and submission.

## 4. Error & Deferral Handling

`research_cli.py` writes one JSON object to `stdout` (`stderr` holds progress/`--stream` events). Check **both** `ok` and structured state fields, consulting [docs/errors.md](../../../docs/errors.md) whenever a command fails or pauses:

1. **When `ok` is `false`**:
   - Branch on `error.code` (never scrape prose), inspect `error.hint` for the exact corrective action, and check `error.details` (`checkpoint`, `recording`, `protected_path`, `retry_after`, `unknown_submission`).
   - Stop immediately on `submission_unknown` / `UNKNOWN_SUBMISSION` or `AUTH_CAPTCHA`; reconcile existing tasks before any new `POST`.
2. **When `ok` is `true` on `campaign status` or `probe`**:
   - Inspect `jobs[].diagnosis.state`, `jobs[].diagnosis.reason`, and follow `jobs[].diagnosis.next_action.argv` (or `next_actions[].argv`) verbatim, preserving its `--research-dir`.
   - If `diagnosis.state` is `waiting` (`platform_pending`, `preflight_backoff`, `shared_slots_occupied`, `submission_cooldown`), respect `next_poll_at` / `retry_at`; do not delete lock files or start duplicate resume loops.
   - If `promote` or `campaign run --validate` returns `outcome: "metadata_required"` (`buffered: false`), supply `required_options` via `alpha describe` and pass them into `next_action.argv` without re-simulating.

## 5. Code Modification Checklist

If changing code inside the repository (see ownership table in [AGENTS.md](../../../AGENTS.md)):
- Keep `research_app` imports out of core modules (`machine_lib.py`, `research_config.py`, etc.).
- Raise `CliError(code, message, hint=..., details=...)` with an actionable `hint` in CLI handlers (never leak raw `ValueError` or `print()` on CLI paths).
- Run `python -m pytest -q` (offline suite, no credentials required) after edits.
