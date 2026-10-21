# AGENTS.md

Working notes for an AI agent operating this repository. Read this first; it
should save you from reading the source.

## 0. Start here

```bash
python research_cli.py schema
```

That prints the complete contract as JSON: every command, every argument with
its type and choices, the research-file format, the error codes and the exit
codes. It is generated from the live argument parser, so it cannot drift from
the implementation. **Do not read `research_cli.py` to learn the interface.**

## 1. The output contract

* `stdout` is **exactly one JSON document** per invocation, success or failure.
  `json.loads(stdout)` always works. Never scrape `stderr`.
* `stderr` carries progress and, with `--stream`, JSON Lines events
  (`submitted`, `completed`, `rate_limited`, and per-poll `polling` with
  `--verbose`).
* Failure shape:

  ```json
  {"ok": false, "command": "simulate",
   "error": {"code": "INVALID_INPUT", "message": "...", "hint": "..."}}
  ```

  Branch on `error.code`, not on the message text.
* Exit codes: `0` ok, `2` usage/invalid input, `3` auth, `4` rate limit or
  service, `5` not found, `6` every simulation failed, `130` interrupted.

## 2. Rules of engagement

1. **Always `--dry-run` before a real run.** It validates the file, resolves
   settings and prints the exact request payloads, with no credentials and no
   quota spent.
   `python research_cli.py simulate research.json --dry-run`
2. **Simulations cost quota.** `catalog ...` and `alpha ...` are free. Do not
   re-simulate something you can read with `alpha get` / `alpha list`.
3. **Check access before designing around data.** `auth check` reports the
   regions that actually return datasets. A `TUTORIAL` account sees `USA` only,
   so expressions built on other regions will fail.
4. **Prefer `catalog fields --expressions`** over hand-writing
   `winsorize(ts_backfill(...))` wrappers; it emits the expected idiom.
5. **Never re-authenticate needlessly.** The session is cached in
   `.brain_session.json`. BRAIN challenges rapid logins with a captcha
   (`AUTH_CAPTCHA`). Use `--relogin` only when the cache is actually stale.
6. **Loop on structured feedback**, not on guesses: `simulate --sort sharpe`
   returns `summary.ranking` and `summary.best`; mutate from there.
7. **Keep credentials out of chat, commits and logs.** `.env` and
   `.brain_session.json` are git-ignored; `tests/test_contract.py` guards that
   they stay that way. Never echo a credential, never write one into a file you
   intend to commit.

## 3. Recipes

Validate, then run a batch and keep the winners:

```bash
python research_cli.py simulate research.json --dry-run
python research_cli.py simulate research.json --sort sharpe --top 5 --stream
```

Explore a dataset without spending quota:

```bash
python research_cli.py catalog categories
python research_cli.py catalog fields --dataset fundamental6 --type MATRIX \
    --search liability --limit 50 --expressions --save /tmp/fnd6.json
```

Check a mechanism against the literature before spending quota on it:

```bash
python research_cli.py search "post-earnings announcement drift accruals" --min-authority 3
python research_cli.py search --fetch https://example.org/paper.pdf
```

Requires `DEEPSEEK_API_KEY` in `.env`. Results carry `rerank_score` (1.5+ is high
quality) and `authority` (3+ is authoritative), so prefer the paper over a post
about it. Cached for a week; `--no-cache` bypasses. Queries are limited to 128
characters, rejected locally.

Diagnose a single alpha:

```bash
python research_cli.py alpha get P0Mq0wEM      # metrics + per-check PASS/FAIL/limit/value
```

## 4. Where things live

| Path | Contents |
| --- | --- |
| `research_cli.py` | CLI + envelope/exit-code contract. |
| `research_config.py` | Research-file parsing and validation (`build_config`). |
| `machine_lib.py` | BRAIN API client, typed errors, FASTEXPR factories. |
| `tests/test_contract.py` | Offline contract tests — run these after any change. |

## 5. Changing the code

* Keep the contract: new commands must emit the standard envelope and map
  their failures to a documented `error.code` / exit code.
* Add or update a test in `tests/test_contract.py`; the suite must run offline
  and pass without credentials:

  ```bash
  python -m unittest discover -s tests -v
  ```
* `machine_lib` must never `print()` on a code path the CLI can reach, and must
  never call `input()`. Report failures by raising a `BrainError` subclass.
* Settings defaults and allowed values live in `machine_lib` (`DEFAULT_SETTINGS`,
  `REGIONS`, `NEUTRALIZATIONS`, `UNIVERSES`) and are re-exported through
  `research_cli.py schema`; keep a single source of truth.