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
| `research_gui.py` | Small tkinter wrapper for interactive use. |
| `Alpha Machine.ipynb` | Worked example, first-order → second-order → third-order alphas. |
| `tests/` | Offline contract tests. No account required. |

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

Sessions are cached in `.brain_session.json` (also git-ignored). BRAIN sessions
last hours but challenges rapid repeated logins with a captcha, so the cache is
what keeps a sequence of CLI calls cheap. Use `--relogin` to force a new login
or `--no-cache` to ignore the cache for one call.

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

## Tests

```bash
python -m unittest discover -s tests -v
```

Fully offline: schema, dry-run payloads, research-file parsing, credential
precedence, error envelopes, and a guard that no credential file is tracked by
git.
