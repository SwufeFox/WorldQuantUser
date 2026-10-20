# Alpha Machine

Tools for researching [WorldQuant BRAIN](https://platform.worldquantbrain.com/)
alphas: generate FASTEXPR expressions, simulate them, and read back which ones
scored well.

The repository is built around one idea: **a program should be able to drive the
whole loop without reading the source.** Every capability is reachable from
`research_cli.py`, whose contract is documented by the tool itself.

```
research_cli.py schema      # the whole contract, as JSON
```

## Layout

| File | Job |
| --- | --- |
| `research_cli.py` | The machine-readable CLI. stdout is always one JSON document. |
| `research_config.py` | Research-file format: parsing and validation, shared by CLI and GUI. |
| `machine_lib.py` | BRAIN API client + FASTEXPR factory helpers. No printing, typed errors. |
| `expression_lint.py` | Pre-flight checks: real operator names, arity, learned rejections. |
| `field_contract.py` | Field contracts: declared vs inferred period/unit/sign, verdicts. |
| `research_paths.py` | Resolves where results go -- always outside this working tree. |
| `alpha_registry.py` | Append-only ledger of alphas worth remembering. |
| `research_gui.py` | Small tkinter wrapper for interactive use. |
| `Alpha Machine.ipynb` | Worked example, first-order → second-order → third-order alphas. |
| `tools/backfill_registry.py` | One-off seed of the registry from the account and existing result files. |
| `tests/` | Offline contract tests. No account required. |

**This repository versions the tool, not the research.** Simulation results,
field-catalog snapshots, the alpha registry and the learned operator denylist
all live in a results directory outside the working tree, resolved by
`research_paths.py`:

1. `$BRAIN_RESEARCH_DIR` if set;
2. otherwise `<persistent workspace>/worldquant-research` (`/data/worldquant-research`
   on this deployment -- `$HOME` is often an ephemeral container layer, and a
   registry that evaporates is no better than one never written);
3. otherwise `$XDG_DATA_HOME/worldquant-research`, then `~/.local/share/worldquant-research`.

`research_cli.py schema` reports the resolved path under `research_paths`. A
`research/` directory inside the repo is git-ignored as a guard rail, so a stray
run cannot quietly add data to the history.

## Install

```bash
pip install -r requirements.txt      # requests (required), pandas (optional)
```

`pandas` is only needed for the notebook and `fields_to_df()`; the CLI runs
without it.

## Credentials

Copy the template and fill it in:

```bash
cp .env.example .env
```

`.env` is git-ignored. Real environment variables (`BRAIN_USERNAME`,
`BRAIN_PASSWORD`) take precedence over the file:

```bash
export BRAIN_USERNAME=you@example.com
export BRAIN_PASSWORD=...
```

`machine_lib.get_credentials()` raises `MissingCredentials` when either is
absent — nothing silently sends an empty login.

BRAIN sessions last hours but challenges rapid repeated logins with a captcha,
so the session cookie is cached and reused. It lives in
`<research_dir>/cache/session.json` — outside the git working tree, mode `0600`,
written atomically so a killed run cannot leave a truncated file behind.
Concurrent CLI calls take a file lock and share one login instead of racing
each other into a captcha. Use `--relogin` to force a new login, `--no-cache` to
ignore the cache for one call, or `auth logout` to forget it.

Catalog metadata (operators, fields, datasets, categories, operator docs) is
cached too, under `<research_dir>/cache/metadata/`. It is quota-free but slow —
the USA field catalog is 4367 rows over fourteen paginated calls — and it
changes on the order of days. Default TTL is 24 h (`--meta-max-age`, or
`BRAIN_METADATA_TTL=0` to disable); `--refresh-meta` refetches. Every catalog
response carries a `cache` report (`hit`, `age_seconds`,
`expires_in_seconds`), and if a refetch fails while an older snapshot exists the
cached copy is served with `stale: true` instead of failing the command.

```bash
python research_cli.py cache info                  # what is cached, how stale
python research_cli.py cache clear --kind metadata # drop catalog snapshots
```

## Driving it from a script or an agent

stdout is **always exactly one JSON document**, so every call is:

```python
result = json.loads(subprocess.run([...], capture_output=True).stdout)
if not result["ok"]:
    handle(result["error"]["code"], result["error"]["message"], result["error"]["hint"])
```

Progress goes to stderr, never stdout. Failures carry a stable `error.code` and
a process exit code (`USAGE`/`INVALID_INPUT` 2, `AUTH_*` 3, `RATE_LIMITED`/
`SERVICE_ERROR` 4, `NOT_FOUND` 5, `SIMULATION_FAILED` 6).

A failure can also carry `error.details`: the structured findings behind it, for
example every lint error per expression, so a caller can act without scraping the
message text.

### Discover the surface

```bash
python research_cli.py schema            # commands, arguments, formats, codes
python research_cli.py schema simulate   # one command in detail
```

### Check the account

```bash
python research_cli.py auth check
```

Returns `user_id`, `permissions` and the regions that actually have data. A
`TUTORIAL` account only sees `USA` — every other region returns zero datasets.

### Browse metadata (quota-free)

```bash
python research_cli.py catalog operators --search ts_ --group "Time Series"
python research_cli.py catalog categories                 # field/alpha counts = crowding
python research_cli.py catalog datasets --region USA
python research_cli.py catalog fields --dataset fundamental6 --type MATRIX \
    --search liability --limit 20 --expressions
```

`--expressions` also emits ready-to-paste snippets, already winsorized and
backfilled. Add `--save out.json` to keep a snapshot for offline use.

### Simulate

Always validate before spending quota:

```bash
python research_cli.py simulate --expr "rank(liabilities/assets)-0.5" --decay 4 --dry-run
```

`--dry-run` prints the exact `POST /simulations` payloads without contacting
BRAIN, so it needs no credentials.

Then run for real:

```bash
python research_cli.py simulate research.json --sort sharpe --top 5 --stream
```

`--stream` emits JSON Lines progress events on stderr (`submitted`, `completed`,
`rate_limited`) while the single JSON result stays on stdout. The result
contains one record per expression (`expression`, `decay`, `alpha_id`,
`status`, `metrics`, `checks`, `error`) plus a `summary` with counts, a
`ranking` sorted by sharpe, and `best`.

Simulation commands have a 5-minute wall-time budget by default. If a server
`Retry-After` would exceed that budget, the command returns `PENDING` rather
than sleeping past the four-hour session-cache lifetime. Accepted jobs and
unsubmitted expressions are atomically saved in a mode-0600 checkpoint under
`.brain_checkpoints/` (or the path passed to `--checkpoint`); continue later
with `python research_cli.py resume <checkpoint>`. Resume polls accepted jobs
before submitting any queued work, so it does not duplicate already accepted
simulations. `--max-wait-seconds` adjusts the budget for either command.

A failed expression is reported in its own record instead of aborting the run;
the command only fails outright when *every* expression failed.

### Inspect results

```bash
python research_cli.py alpha get P0Mq0wEM
python research_cli.py alpha list --min-sharpe 1.5 --limit 20
```

Both are quota-free.

`alpha list` filters on `status=UNSUBMITTED` by default, so it will not show
what you have already submitted. Read the live book with:

```bash
python research_cli.py alpha list --status ACTIVE --limit 30
```

Or ask for the book directly — it joins each submitted alpha with the ledger's
mechanism label, and can report how a candidate would sit against it:

```bash
python research_cli.py alpha book
python research_cli.py alpha book --check P0M7171K
```

### Check the ledger before spending a simulation

The most expensive mistake in this workflow is re-testing a field that was
already falsified. `alpha find` answers that offline, with no network call:

```bash
python research_cli.py alpha find --field anl4_cfo_number
python research_cli.py alpha find --mechanism revision --verdict rejected
```

`already_tested: true` in the output means a prior entry mentions that field;
read its verdict and metrics before writing a new expression.

Verdicts are not all failure states:

| verdict | meaning |
|---|---|
| `candidate` | passed the gates; keep it |
| `watch` | interesting but below threshold, or correlation unverified |
| `rejected` | the expression underperformed; the hypothesis was never on trial |
| `falsified` | a pre-registered hypothesis was tested with controls and the controls won — a result, not a failure |
| `no_go` | abandoned before a fair test (data blocked, unusable fields) |

`falsified` is the one worth protecting: it is what stops a later round from
re-opening a mechanism that was already ruled out, so put the deciding control
in the note.

### Read the checks that matter (including self-correlation)

`alpha get` returns the **cached** check snapshot embedded in the alpha body,
where `SELF_CORRELATION` is usually `PENDING` or missing. That is a caching
artefact, not a permission limit: `GET /alphas/{id}/check` triggers the
computation and replies with `Retry-After` until the job finishes.

```bash
python research_cli.py alpha check d5K0Pod2          # compute and print live checks
python research_cli.py alpha check d5K0Pod2 --max-wait 300
```

### Submit (and the "test submission")

There is **no separate test-submission endpoint**. The pre-submission test *is*
the check flow above; the only write action is `POST /alphas/{id}/submit`.
`alpha submit` therefore defaults to a dry-run that runs the live checks and
reports the correlation that would decide the submission, and only writes with
`--yes`:

```bash
python research_cli.py alpha submit RRA1jdmb          # dry-run, nothing written
python research_cli.py alpha submit RRA1jdmb --yes    # irreversible
```

### Lint before you spend a simulation

```bash
python research_cli.py lint 'rank(ts_delta(close, 5))'
python research_cli.py lint --file my_batch.json --dataset fundamental6
python research_cli.py lint --offline 'rank(close)'      # no account needed
```

The operator names in an expression are checked against `GET /operators`, the
authoritative list -- not the documentation site, which describes operators the
simulator does not accept. (`regression_neut` is the cautionary example: a docs
page exists, the operator does not.) `simulate` runs the same check before
submitting and refuses to spend quota on an expression that cannot work; pass
`--allow-lint-errors` to override. Exit code is `2` on a lint failure and the
error envelope carries every finding under `error.details`.

Rejections the simulator reports are written back to
`<research_dir>/operator-denylist.json` with a severity: `block` when the operator
is unknown or unavailable, `warn` when it merely rejects that kind of input (for
example `subtract` refusing event inputs). The file persists across sessions, so
the lesson survives even though it is not in git.

### Audit a field set before combining fields

```bash
python research_cli.py catalog field-audit \
    --fields fnd6_newa1v1300_capx,fnd6_newa2v1300_oancf,fnd6_newa1v1300_at \
    --operation ratio
```

The catalog tells you a field's coverage and delay; it does not tell you the
unit, the currency, the fiscal period, the sign convention or how corporate
actions are handled. Those unknowns -- not the economics -- are what sink ideas,
so they are made explicit per field as `declared`, `inferred` (with the rule that
produced it) or `unknown`, and the operation is then judged:

| verdict | meaning |
| --- | --- |
| `ready` | period and unit are known and consistent; no caveats |
| `needs_review` | usable, but fiscal alignment or sign convention is unverified |
| `blocked` | a critical attribute is unknown; do not compute on it yet |

Operations: `ts_difference` (one field over time), `ratio`, `difference`, `sum`.
Differencing a share count is blocked outright because no split basis is
declared; a sum or difference of fields with undeclared sign is blocked because
one negative stored value inverts the signal. Quota-free.

### Remember what you found

```bash
python research_cli.py simulate batch.json --record --idea "Idea 50" \
    --mechanism "external financing gap"
python research_cli.py alpha record P0Mq0wEM --idea "leverage, size-bucketed" \
    --verdict candidate --note "why it mattered"
python research_cli.py alpha registry --format markdown --min-sharpe 1.25
```

`<research_dir>/alpha-registry.jsonl` is an append-only ledger, one alpha per
line, kept outside git so findings outlive a chat window without turning the
repository into a data dump. Back that file up like any other result; it is the
durable memory of the research. Each entry is self-contained:
the expression, the full settings, the in-sample metrics, the platform checks,
the idea it belongs to, a verdict (`candidate` / `watch` / `rejected` / `no_go`)
and a note. Re-recording the same alpha replaces it on read without rewriting
history. `simulate --record` classifies automatically (clears the bar →
`candidate`, fails a check → `rejected`, otherwise `watch`); `alpha record` takes
an explicit verdict when you have actually decided something.

## Research files

```json
{
  "settings": {"region": "USA", "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 6},
  "alphas": [
    "rank(liabilities/assets)-0.5",
    {"expression": "rank(ts_delta(close, 5))", "decay": 4},
    ["ts_rank(volume, 20)", 8]
  ]
}
```

Command-line flags override the file, which overrides the defaults. A file that
does not start with `{` or `[` is read as plain text: one expression per line,
blank lines and `#` comments skipped.

## Using `machine_lib` directly

```python
import machine_lib as ml

session = ml.load_session()                     # cached session, logs in if needed
fields = ml.get_datafields(session, dataset_id="fundamental6", region="USA",
                           universe="TOP3000", delay=1, field_type="MATRIX")
print(ml.process_datafields(fields)[:3])        # FASTEXPR-ready snippets

records = ml.simulate_alphas(session, [("rank(cap)", 6)], settings={"region": "USA"})
print(records[0]["metrics"]["sharpe"])
```

Errors are typed and subclass `machine_lib.BrainError`: `MissingCredentials`,
`AuthError`, `CaptchaRequired`, `NotFound`, `RateLimited`, `ServiceError`,
`SimulationError`, each with a stable `code`.

## Notes and limits

* **Simulation quota is real.** `simulate` without `--dry-run` spends it;
  `catalog` and `alpha` never do.
* **Rate limits.** Metadata endpoints return HTTP 429 under load and
  submissions have their own limit. `machine_lib` absorbs both with the
  advertised `Retry-After` backoff and surfaces `RateLimited` only when the
  budget is exhausted.
* **Pagination.** BRAIN caps `limit` at 50 per page; `get_datafields` pages
  internally.
* **Region access is account-scoped.** Unknown region/neutralization values are
  rejected before any request is made.
* **No interactive prompts.** `login_hk()` raises with the authorization URL
  instead of blocking on `input()`.
* **Field contracts are partly inferred.** No BRAIN endpoint declares a field's
  unit, currency, fiscal period or sign convention. `field_contract` labels
  every attribute `declared` / `inferred` / `unknown` and never guesses a value
  silently, but an `inferred` period is a heuristic on the field id and should
  be checked against the economics before it is trusted.
* **Lint is not a correctness proof.** It catches unknown or unavailable
  operators, unbalanced syntax, arity mismatches and unknown field ids. It
  cannot know that a syntactically valid expression is economically meaningless.
* **The registry records alphas, not conclusions.** `verdict` and `note` are
  judgement; the expression, settings and metrics next to them are re-read from
  the platform so the numbers cannot drift.

## Tests

```bash
python -m unittest discover -s tests -v
```

Fully offline: schema, dry-run payloads, research-file parsing, credential
precedence, error envelopes, and a guard that no credential file is tracked by
git.

`tests/test_contract_v2.py` covers the v2 additions: field-contract inference and
verdicts (including the exact share-count and fiscal-alignment blockers from the
research notes), lint operator/arity/denylist behaviour, and registry
append/replace/merge semantics.
