# AGENTS.md

## Follow one execution path

```bash
python research_cli.py daemon start
python research_cli.py brief
python research_cli.py inbox
```

Read `research_dir`, source existence, open questions, and pending `GATE_*` items
before choosing a direction. Missing or empty local records do not mean a
direction is untested. Use the same `--research-dir` for every step.

| Step | Command and decision |
| --- | --- |
| Inspect | `brief` and `inbox` / `runs` summarize ledger evidence, pending interaction gates, and active `AUTO_*` runs. |
| Propose | Put the question in the spec's `idea` (spec or per-arm) and stable arm labels, then run `python research_cli.py propose SPEC.json`. It validates offline without waiting on BRAIN and enters `AUTO_PREFLIGHT`. |
| Wait / Wake | Run `wait --max-wait 30` (or check `inbox` later) until the resident daemon advances a run to a `GATE_*` pause point (`GATE_PREFLIGHT_BLOCKED`, `GATE_TRIAGE`, `GATE_APPROVE_FULL`, `GATE_DESCRIBE`, `GATE_CONFIRM_SUBMIT`). |
| Gate 1 & 2 | At `GATE_TRIAGE`, `inspect RUN_ID` shows train-only metrics (protocol Train/Test firewall). Call `act RUN_ID hydrate-pnl --top 3` (or `--arms ...`) -> daemon reaches `GATE_APPROVE_FULL`. Call `act RUN_ID approve-full` to freeze `runs/<run_id>-selection.json` and open the test view, or `drop --verdict falsified`. |
| Validate & Gate 3 | `AUTO_FULL_VALIDATE` runs FULL sim, live checks, self-correlation, and 5/10/20 bps cost audit, pausing at `GATE_DESCRIBE` (or `REJECTED_AT_VALIDATION`). Call `act RUN_ID describe --name ... --category ... --tags ... --description ...` with the official Power Pool template. |
| Gate 4 & Shelf | `AUTO_BUFFER` verifies remote description parity and writes `seeds.jsonl` + `submit-buffer.jsonl`. At `GATE_CONFIRM_SUBMIT`, call `act RUN_ID hold` (-> `SHELVED`) or `act RUN_ID submit --yes` (requires `BRAIN_ALLOW_AGENT_SUBMIT=true`). |
| Review | `ledger analyze --group-by scope`, then narrow to the mechanism and inspect failures (`rejected`/`falsified`/`no_go`) alongside candidate leads. `shelf list` / `shelf plan` plan independent submissions. |

Keep a single resident daemon per workspace (`daemon start` / `daemon stop` /
`daemon status`). Subagents may draft separate specs, call `propose` for new
hypotheses, inspect runs (`inspect RUN_ID`), and analyze the ledger (`ledger analyze`),
but must not bypass `daemon/state.lock`, manually edit `runs/*.json`, or rename
arms to evade duplicate detection (`request_fingerprint` / `exact_repeat`).

The detailed state transition guide lives in
[docs/ledger-workflow.md](docs/ledger-workflow.md). Use `help COMMAND` for
options and examples; use `schema COMMAND` for its machine contract.
For throughput, shared budgets and concurrency measurements, read
[docs/fast-backtest.md](docs/fast-backtest.md).

## Read progress literally

- `ok: true` means the CLI/IPC call was handled. Read `state`, `execution`,
  `unchanged_defers`, `stalled`, `next_wake_at`, and `allowed_actions` on each run.
- Repeated identical `next_wake_at` or `stalled: true` requests diagnosis via
  `inspect RUN_ID` / `last_error`; `AUTO_PREFLIGHT` automatically trips to
  `GATE_PREFLIGHT_BLOCKED` after `max_unchanged_defers` (default 3) unchanged retries.
- Terminal execution failures are recorded into `alpha-registry.jsonl` with
  `failure:` keys and original error text; they are not automatic scientific
  falsifications (`falsified`).
- `AUTO_SUBMITTING` reconciles `UNCERTAIN` submission outcomes via read-only
  `fetch_alpha_metrics` and `read_alpha_submission_status` without issuing a
  second `POST /alphas/{id}/submit`.
- Every candidate description passed to `act RUN_ID describe` must use the
  official Power Pool template: exact headings `Idea:`,
  `Rationale for data used:`, `Rationale for operators used:`, each followed by
  `- ` bullets, with at least 100 trimmed characters total.

## Research loop

- Establish the question, known evidence and the next uncertainty to resolve.
  For open exploration, consider a few distinct economic explanations before
  selecting fields or tuning an expression.
- Separate observations, interpretations and untested hypotheses. A high score
  is a lead, not a mechanism. Check `tools/unit_report.py --dataset DATASET` and
  `ledger list` before calling a family new.
- Choose an experiment that distinguishes explanations (`design: explore` vs
  controlled `add`/`drop`/`parallel`/`rescue` with a `control` arm).
- For Python, declare `language: PYTHON`, `lookback`, and `@alpha(data=[...])`.
  `propose` validates Python syntax and declarations statically before queueing.
- Keep `open-threads.md` in the research directory: one
  `- [open] **question** — evidence; next discriminating check` or
  `- [closed] **question** — conclusion; evidence; reopening condition`
  per line.

## 人类协作者的 BRAIN Labs 沙盒能力

用户随时可在隔离环境 BRAIN Labs 中执行 Python 分析（pandas/numpy/sklearn/xgboost），直接检查 In-Sample 原始 Datafield 的分布、缺失结构、覆盖率、极值、字段间相关性及滚动 ML 交互，但无法导出原始数据。

**触发委派规则**：当你遇到以下情况时，**停止继续生成盲试表达式消耗模拟额度**，改为通过 `research_cli.py labs open`（或在状态机门禁下通过 `act RUN_ID request-labs`）向用户下发一份简洁的“Labs 诊断工单”：
1. 同一数据集或字段连续多次逻辑合理但模拟结果异常（高换手、收益极端集中、子样本骤降）——可用 `labs suggest` 核查。
2. 不确定字段的真实更新频率、覆盖率、长尾极值或行业缺失特征。
3. 需要验证多字段共线性、去冗余或探索非线性/条件分段交互结构。

**工单格式**：注明 `Region/Universe/Delay`、`目标字段`、`待验证的具体定量问题`、`期望用户回传的汇总统计指标`（禁止导出原始数据），以及基于诊断结果的下一步分支方案。在 `OPEN` 工单未通过 `labs resolve` 回传统计摘要前，`propose` 会本地拦截对同作用域目标字段的盲试模拟。

## Execution invariants

1. `propose` validates syntax and spec structure locally with zero BRAIN quota
   before entering `AUTO_PREFLIGHT`; online field/operator lint and duplicate
   checks happen inside the daemon before any simulation POST.
2. Check `auth check` before designing around an unfamiliar account or region.
   Reuse the session cache; use `--relogin` only when it is actually stale.
3. Prefer `catalog fields --expressions` for field wrappers. Inspect structured
   checks and comparisons when deciding what an experiment demonstrated.
4. Train data supports design and selection; view test data ONLY after
   `act RUN_ID approve-full` freezes `runs/<run_id>-selection.json`.
   `inspect` enforces this protocol-level Train/Test firewall automatically.
5. Keep credentials out of chat, logs and commits. `.env`, session caches, and
   `daemon/endpoint.json` bearer tokens are private.
6. Structured commands, including `help` and `schema`, write one JSON document
   to stdout. Terminal `--help` and `--version` write text. Progress goes to
   stderr. Branch on `error.code` for structured errors (see `schema`).
7. Code lives in the repo; results, `runs/`, `daemon/`, evidence, caches and
   throwaway scripts live in the research directory. Legacy v3 `backlog/` and
   `backlog-results/` directories can be imported once via `migrate-workspace`.

## Changing the code

| Concern | Owner |
| --- | --- |
| Invocation, workspace, JSON envelope, saving, exit codes | `research_app/app.py` |
| Arguments, help, generated schema | `research_app/parser.py` |
| Shared errors, output, settings, sessions | `research_app/runtime.py` |
| Persistent 4-gate state machine, firewall, workspace migration | `research_app/state_machine.py` |
| SQLite WAL workspace store (`runs`, `slot_journals`, `labs_tickets`) | `workspace_db.py` |
| Delta-Parquet + SQLite WAL primary PnL store | `pnl_store.py`, `research_app/pnl_snapshot.py` |
| BRAIN Labs diagnostic tickets & blind-sim guardrails | `research_app/labs.py` |
| State-machine autonomous step workers, cost audit, slot reconciliation | `research_app/sm_workers.py` |
| Resident single-writer HTTP loopback daemon & non-blocking CLI handlers | `research_app/daemon.py` |
| Bounded workspace brief, book health, views log, breeding | `research_app/brief.py`, `research_app/breeding.py` |
| Simulation, ablation, comparisons, spec normalization | `research_app/experiments.py`, `research_app/exp_*.py`, `research_app/guarded_sim.py` |
| Read-only inspection of existing simulation tasks | `research_app/simulation_import.py` |
| Catalog, admission, validation | `research_app/catalog.py`, `research_app/batch_admission.py`, `research_app/validation.py` |
| Training selection, promotion helpers, candidate shelf | `research_app/triage.py`, `research_app/selection.py`, `research_app/promote.py`, `research_app/buffer_cmd.py` |
| Power Pool budget, submission planning, descriptions, rule snapshots | `research_app/pp_budget.py`, `research_app/submission_planner.py`, `research_app/description.py`, `research_app/rules_snapshot.py` |
| Platform alphas and local ledger | `research_app/alphas.py`, `research_app/ledger.py`, `research_app/ledger_analysis.py`, `registry_index.py` |
| Feeds, account, literature, route classification | `research_app/feed.py`, `research_app/account.py`, `research_app/literature.py`, `research_app/research_route.py` |
| Research-file contract, check normalization, slots and BRAIN client | `research_config.py`, `check_evidence.py`, `simulation_slots.py`, `simulation_multi.py`, `machine_lib.py`, `brain_core/` |

Put behavior in its owning module and bind it explicitly in `parser.py`. Core
libraries (`*.py` at repo root + `brain_core/*.py`, excluding `research_cli.py`)
must not import `research_app` or `tools`; `research_app` must not import
`tools`. `machine_lib` raises `BrainError` subclasses; every `CliError` must
provide an actionable `hint`, and every public module/class/function must have a
docstring.

Preserve the JSON envelope and documented error/exit codes. Update meaningful
behavioral coverage in `tests/test_contract.py` and `tests/test_state_machine.py`,
and run the offline pytest suite without credentials:

```bash
python -m pytest -q
```

Update the relevant usage documentation; keep [docs/tooling.md](docs/tooling.md)
and [docs/ledger-workflow.md](docs/ledger-workflow.md) accurate when tools or
states change.
