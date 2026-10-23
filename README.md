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
| `expression_types.py` | Five field types, operator signatures, and the mismatches they imply. |
| `field_index.py` | Live field-name verification and vocabulary-free variant grouping. |
| `field_search.py` | Ranked local search over fields and operators: AND, exclusion, fuzzy. |
| `deepseek_search.py` | Literature search and page fetch, so an idea can be checked against the paper. |
| `paper_search.py` | OpenAlex paper search: journal, DOI, citations, reconstructed abstract. |
| `field_contract.py` | Field contracts: declared vs inferred period/unit/sign, verdicts. |
| `research_paths.py` | Resolves where results go -- always outside this working tree. |
| `alpha_registry.py` | Append-only ledger of alphas worth remembering. |
| `research_gui.py` | Small tkinter wrapper for interactive use. |
| `correlation.py` | Align daily PnL on shared dates, Pearson, residualise against a control. |
| `ablation.py` | Leave-one-out and solo variants, so a leg can be shown to earn its place. |
| `cache_store.py` | Small JSON cache with TTL, so repeat calls do not re-fetch. |
| `tools/expression_macros.py` | Named `def` templates, expanded in bulk, lint-gated. |
| `tools/probe_units.py` | Measure a field's unit from the simulator's own rejection. |
| `tools/pool_stats.py` | Report the pool as quartiles rather than as a best alpha. |
| `tools/cluster_ledger.py` | Group the ledger by PnL correlation or by shared datafields. |
| `tools/dedupe_ledger.py` | Find identical-PnL entries that are one alpha filed twice. |
| `tools/derive_family.py` | Derive family labels from the datafield namespace. |
| `tools/turnover_repair.py` | Break-even Sharpe for a smoothing decision. |
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

`pandas` is only needed by `fields_to_df()`; the CLI runs without it.

## Credentials

Copy the template and fill it in:

```bash
cp .env.example .env
```

`.env.example` is the single list of variables this tool reads: required credentials,
the research directory, the literature-search key and its alias, and the two tuning knobs.
It is guarded by a test that extracts environment names from the source, because that list
had already drifted once - four variables were documented nowhere, including the one that
decides where the ledger lives.

`.env` must sit next to `machine_lib.py`; the loader resolves it relative to the module,
so a `.env` elsewhere is silently ignored. It is git-ignored. Real environment variables (`BRAIN_USERNAME`,
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

`--search` is ranked, not a substring filter, and it runs locally over the
fetched catalogue so it can do what the endpoint cannot:

```bash
python research_cli.py catalog fields --dataset analyst4 --search "cfps number"
python research_cli.py catalog fields --dataset analyst4 --search "cfps -median"
python research_cli.py catalog operators --search "rank ts" --any-term
python research_cli.py catalog fields --dataset fundamental2 --search cfpps --fuzzy
```

Multi-word queries are AND, `-word` excludes, `--any-term` relaxes to OR, and
`--fuzzy` tolerates one edit. Hits carry `score` and `matched_terms`, so the
ordering is explainable. When nothing matches, the reply names the `absent_terms`
rather than quietly relaxing the query — assuming two words are synonyms is the
one thing this search refuses to do.


### Check field names and compare variants (quota-free)

```bash
python research_cli.py catalog verify --fields cfps_mean,anl4_afv4_cfps_mean --strict
python research_cli.py catalog concepts --dataset analyst4 --search cfps --min-spread 0.1
```

`catalog verify` answers one question: does this identifier exist in the live
catalog for this scope? A name typed from memory is not an operator, so the
operator catalog cannot see it, and the first symptom is a simulator error after
the submission has been spent. `simulate` now runs this check itself before
submitting anything and refuses to start on an unknown name.

`catalog concepts` groups fields that differ only in family boilerplate and shows
their coverage side by side. Two ids for the same quantity can differ by a third
of the universe, and nothing in the ids says so, so the choice is made on the
number. The grouping carries **no built-in vocabulary**: `--tail-tokens N`
compares only the trailing words, `--ignore-words` drops words you name, and
`--drop-numeric` ignores vintage numbers. Nothing about BRAIN's naming is
hard-coded, so the tool cannot rot into mis-grouping after a family is renamed.


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

### Mechanism labels overstate diversity — read the family count

`mechanism` is specific on purpose: one label per construction. That makes it a
bad diversity metric. The current book has **13 alphas, 11 mechanism labels,
but only 4 families**, because six labels are variations of the same
overnight/intraday bet.

`family` is the declared second level (`oi`, `quality`, `leverage`, `option`,
`analyst`, `event`, `reversal`, `momentum`, `attention`, `other`) and it is
never inferred: a keyword guess would label an analyst forward-EPS alpha as
`analyst` when it is really a `quality` bet.

```bash
python research_cli.py alpha book           # book_size / mechanism_count / family_count
python research_cli.py alpha find --family quality
```

Quote `family_count` when the question is "how many independent bets do I
have"; quote `mechanism` only when you need to tell two constructions apart.

### Read the checks that matter (including self-correlation)

`alpha get` returns the **cached** check snapshot embedded in the alpha body,
where `SELF_CORRELATION` is usually `PENDING` or missing. That is a caching
artefact, not a permission limit: `GET /alphas/{id}/check` triggers the
computation and replies with `Retry-After` until the job finishes.

```bash
python research_cli.py alpha check d5K0Pod2          # compute and print live checks
python research_cli.py alpha check d5K0Pod2 --max-wait 300
```

### Check self-correlation locally, and correctly

```bash
python research_cli.py alpha self-check O0J6Aeag
python research_cli.py alpha self-check gJWre33J --as-of 2026-10-22T09:15:30-04:00
```

Daily Pearson is the measure that decides, because it was calibrated against ten alphas
with a published platform value: it came within **0.006 to 0.016** and always read slightly
lower. Weekly and monthly overstated the platform number by 0.03 to 0.08, so those are
diagnostics rather than verdicts. `classify()` subtracts the measured offset, which is why
0.71 is worth submitting and 0.75 is not.

Two behaviours matter more than the arithmetic, both found by disagreeing with the
platform and then explaining why:

* **The peer set must include in-flight submissions.** `O0J6Aeag` entered the book **20
  seconds** after `gJWre33J`, and the two correlate **0.975**. Both were accepted, because
  the platform compares against a periodically refreshed snapshot rather than a live book.
  A local check that looks only at ACTIVE alphas repeats that blind spot, so recent
  submissions — including ones not yet submitted — are included by default
  (`--include-unsent-hours`, default 24).
* **A published platform value is historical.** `0mjGjJ36` was refused at 0.7005 and now
  measures 0.9991 against today's book, which is not a contradiction: the counterpart
  entered afterwards. `--as-of` restricts the peers to those submitted at or before a
  timestamp, which reproduces the original check. Timezone matters here and got it wrong
  first: `dateSubmitted` is `-04:00`, so comparing it as a string against a Beijing
  timestamp silently treats every alpha as older than it is.

The report says so explicitly when the counterpart entered after the candidate
(`counterpart_entered_after_candidate`), instead of leaving a number that looks like a
disagreement with the platform.

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

An alpha also carries metadata, and a submission is unattributable without it. Only
three properties are writable — `name`, `category` and `tags` — and `description` does
**not** exist: eighteen candidate spellings all return `Unexpected property`, so none is
invented to hold one. `alpha submit` refuses an alpha with no name or category and prints
the exact command to fix it.

```bash
python research_cli.py alpha describe ABC12345 \
    --name "Inventory stage growth with intraday position" --category FUNDAMENTAL
```

Categories are an enum discovered by probing, because an invalid value returns 400 without
listing the choices: `ANALYST`, `FUNDAMENTAL`, `PRICE_VOLUME`, `SENTIMENT`, `VOLUME`.

### What "submitted" actually means

Three outcomes look similar and are not:

| response | meaning |
| --- | --- |
| `403` with a check list | evaluated and refused; only here do you learn the real value |
| `201` with an empty body | accepted for asynchronous evaluation; **not** a submission |
| `dateSubmitted` set | the only trustworthy evidence that it went in |

A first `POST` does not return the self-correlation value at all. It comes back on a
**second** attempt once the evaluation has cached: an alpha can answer `201` and then
`403 selfCorr=0.8488` minutes later. Always confirm with `dateSubmitted`, never with the
status code.

`alpha submit` therefore reports a four-state `submission` field instead of a boolean:

| `submission` | meaning |
| --- | --- |
| `CONFIRMED` | `dateSubmitted` is set; it is in the book |
| `ACCEPTED` | the platform took the request but `dateSubmitted` is still empty - re-check before counting it |
| `REJECTED` | refused, with `failed_checks` naming what failed and its value |
| `UNCERTAIN` | accepted, but the read-back failed; the outcome is unknown |
| `NOT_SENT` | dry run, nothing was written |

`submitted` stays in the document and is true only for `CONFIRMED`. An earlier version set
it whenever the HTTP status was below 400, which contradicted this very table and produced
a false report of twenty submissions, all of which were still UNSUBMITTED.

The platform's own view names the counterpart, which is what makes the substitute route
decidable:

```bash
python research_cli.py alpha self-corr ABC12345
```

```
sharpe 2.35   max_correlation 0.8593   sharpe_needed 2.475   verdict blocked
counterparts: qMEWm7L2 (0.8593, Sharpe 2.25), XgRWJAQX (0.7896, Sharpe 2.00)
```

The margin applies to **every** counterpart above the limit, not to the most-correlated
one — taking the counterpart of the maximum understates the bar exactly when a second,
higher-Sharpe neighbour exists. Two related traps are handled rather than discovered:
`/correlations/self` answers `200` with an empty body plus `Retry-After` when it is not
ready (a blank body is retried, because reading it as "no data" reports a correlated alpha
as clear), and the recordset we use is `/recordsets/daily-pnl`, which is already daily, so
the documented requirement to difference a cumulative series does not apply here.

### The turnover floor, checked mechanically

**Disclosure data is not low-value.** Three of the book's entries are built entirely from
`fundamental6` and `analyst4` fields — inventory stages at Sharpe 2.11, receivables-payables
at 1.56, raw materials at 1.53 — because each pairs the disclosure leg with one that
reprices daily.

What cannot clear the 0.01 turnover floor is a construct with **no such leg**. Remove it and
the same data measures deferred tax **0.0077**, operating leases **0.0086**, unrecognised
tax benefits **0.009**, share-based compensation **0.0083**. No neutralisation, decay or
truncation lifted any of them, because the ranking only reorders when a filing does.

`lint` now says so before a simulation is spent:

```
turnover_floor_risk: no leg that moves daily: this construct measured 0.006-0.010 turnover
against a 0.01 floor in four accounting families ... The fields are fine - three book
entries are built from them - what is missing is the daily leg. Add one (a reversal,
position or pressure term) rather than discarding the idea.
```

The check reads **which fields carry market data**, taking that from the catalogue's
dataset rather than from a list of names. Its subject is the expression, not the field: the
finding says "add the leg", never "avoid the data". The first attempt keyed on "has a time-series
operator" and was wrong in both directions: `ts_backfill` is a fill, not movement, so a
static ratio passed silently; and `close/vwap` has no window at all, so a shape that turns
over hundreds of times a year was flagged.

It is a warning, not an error: a low-turnover signal is legal, it simply cannot pass. The
dataset split lives in one visible constant (`DISCLOSURE_DATASETS`, named for the release
cadence rather than for any judgement about value) instead of being inferred inside the
linter, so the assumption can be read and argued with.

### Types: five kinds of field, and the operators that accept them

The platform's field filter offers five types — `Matrix`, `Vector`, `Group`, `Universe`,
`Symbol` — and an operator accepts specific ones. `expression_types.py` carries the
signature table and checks arguments against it:

```python
rank(five_minute_price_change_pct)              # type_mismatch: rank expects MATRIX
rank(vec_avg(five_minute_price_change_pct))     # ok: the vector reducer returns MATRIX
multiply(five_minute_price_change_pct, 2)       # type_mismatch
group_rank(close, close)                        # type_mismatch: the group slot wants GROUP
group_rank(close, subindustry)                  # ok
group_rank(close, bucket(rank(close)))          # ok: bucket returns GROUP
densify(close)                                  # type_mismatch: densify wants GROUP
```

This replaces a hand-built denylist of "operators that reject VECTOR fields", which had
been assembled from simulator error messages and was necessarily incomplete. The rule is
now derived: only `vec_avg` and `vec_sum` accept a VECTOR, so everything else is a type
error by construction. The denylist survives only for operators the table does not cover,
so the same problem is never reported twice.

Two details worth knowing:

* `bucket` returns a **GROUP**, not a number. That is why it can be passed to a group
  operator and nowhere else, and it is the documented way to build a custom grouping.
* operators that the course material or the formulae paper documents but this account's
  catalogue lacks — `vec_choose`, `ts_max`, `ts_min`, `floor` — are reported as
  `operator_documented_unavailable` with that reason, instead of a generic unknown-operator
  error. `vec_choose` in particular appears in WorldQuant's own short-interest example and
  cannot run here.

Inference is deliberately pragmatic: a bare field resolves to its declared type, a call to
its return type, a literal to scalar, and anything else to the first typed piece inside it
or `UNKNOWN`. It stays silent rather than guessing, so a variable assigned earlier in the
expression is not mistaken for a type error.

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

### Measure a unit instead of guessing it

The catalogue does not declare units, and `field-audit` can only say `unknown`. There is
one way to find out: add the field to a pure number and read the rejection.

```bash
python tools/probe_units.py --fields fnd6_newqv1300_invfgq,fnd6_newqv1300_invwipq
```

```
fnd6_newqv1300_invfgq   Unit[TSPrice:1]   simulator named the field unit as expected
fnd6_newqv1300_invwipq  dimensionless     add with a scalar succeeded
```

Answers are cached in `<research_dir>/field-units.json` so a probe costs one simulation
only once. Two details are easy to get wrong and both are pinned by tests:

* the simulator reports `expected "Unit[TSPrice:1]", found "Unit[]"` — the **expected**
  side is the field and the **found** side is the bare scalar, so reading `found` inverts
  every answer;
* a unit mismatch can arrive as `WARNING` as well as `ERROR`, so the error text is
  inspected before the status.

This distinction is worth acting on: `unknown` is a prompt to measure, `absent` is a
reason to stop. An audit that records a block on a measurable unknown leaves a mechanism
unexamined for no reason, which is how four inventory fields sat untested for days.

## Write a template once, instantiate it in bulk

The same accounting-growth shape appeared in a dozen batches with only the field
changing, and every transcription risked a typo that only lint would sometimes catch. A
macro file makes the shape a named, parameterised thing:

```
def growth(f, w=63, b=252):
    ts_delta(ts_backfill({f}, {b}), {w}) / (abs(ts_delay(ts_backfill({f}, {b}), {w})) + 1)

def position():
    (close - low) / (high - low) - 0.5

def carrier2(f1, f2, fast, fw=0.5, w=63, b=252):
    -rank(growth({f1}, {w}, {b})) - rank(growth({f2}, {w}, {b})) - {fw}*rank({fast})
```

```bash
python tools/expression_macros.py --macros macros/carriers.macros --list
python tools/expression_macros.py --macros macros/carriers.macros \
    --expand 'carrier2(fnd6_newqv1300_invfgq,fnd6_newqv1300_invwipq,position(),0.5,63,126)'
python tools/expression_macros.py --macros macros/carriers.macros \
    --plan macros/plan-arap.json --lint --out rounds/batch.json
```

Expansion is textual, deterministic, and recursive: a macro may call another, cycles are
refused, and a macro that leaves a placeholder unresolved is an error rather than a
half-built expression. Determinism is what makes the ledger's expression dedupe
meaningful — the same macro and arguments always produce the same string, so a repeat
batch is caught instead of paid for.

`--lint` runs the same checks the CLI runs (operator catalogue, VECTOR fields read by an
operator that cannot take them, trailing tokens, operator budget) and **refuses to write
a file that has problems**. A generator that can emit unchecked expressions is a way
around every check built so far.

`build` also drops duplicates of the ledger and of the plan itself before writing, and
says which entries it dropped.

The proof that the macros are faithful is a byte comparison, not a claim: expanding
`carrier2` with the inventory fields and the `position` leg reproduces `rKd0g0bj` — the
alpha that entered the book — character for character.

### Report the pool as a distribution

A best-alpha report hides what a pool is: two pools can share a top result and differ
completely in spread, one narrow and correlated, one broad and independent.

```bash
python tools/pool_stats.py --min-fitness 1.25 --out pool.json
```

```
measure                           min       q1   median     mean       q3      max
Sharpe                         1.4100   1.7225   1.9050   1.9025   2.0700   2.5800
turnover                       0.0125   0.0934   0.1229   0.1233   0.1580   0.3793
holding days (1/turnover)      2.6364   6.3291   8.1400  13.0821  10.7131  80.0000
pairwise correlation          -0.1843   0.2731   0.4213   0.4475   0.6142   1.0000
```

The correlation row is the one that matters. A pool whose median pairwise correlation is
0.42 is one bet wearing many names, whatever its best alpha looks like. Percent-of-capital
volatility is deliberately absent: the recordset is cumulative dollar PnL and capital is
not exposed, so that figure cannot be computed here and is not faked.

#### Forecast fields: frequency is not target period

`period` answers "how often is this estimate produced". It does not answer "which fiscal
year does this observation target", and the catalog never says. The contract therefore
carries two more attributes, kept deliberately separate:

```json
"target_fiscal_period":         {"state": "unknown", "evidence": "none"},
"historical_target_alignment":  {"state": "unverified", "lookback": 63}
```

The consequence is a verdict change: `ts_difference` over an analyst forecast field now
returns `needs_review` with `target_fiscal_alignment_unverified`, where it used to return
`ready` as soon as the frequency was known. A 63-day difference of an annual estimate is
not necessarily the revision of one forecast — it can be a comparison across a fiscal-year
rollover, and reading it as a revision invents a signal.

The flag is narrow on purpose: the analyst dataset and the `anl4` prefix. Differencing a
reported accounting figure is unaffected, because blocking the whole field catalogue would
trade one mistake for a worse one.

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

## Track how alphas relate

The ledger records what each alpha is and how it scored. These commands cover the
other half — how they relate — and all of them work offline.

```bash
python research_cli.py alpha coverage                 # how much lacks a family label
python research_cli.py alpha label ABC12345 --family quality --verdict candidate
python research_cli.py alpha impact ABC12345          # who else stands on these legs
python research_cli.py alpha lineage ABC12345         # where a variant came from
```

`alpha impact` matches on the **leg body**, not the leg name: leg names are
positional, so `a` means whatever the author put there and a shared letter carries
no information. Two alphas sharing a body are the same exposure even when their
PnL is uncorrelated, and that is the question correlation cannot answer.

`alpha label` appends a new record under the same key rather than rewriting, so a
superseded judgement stays visible. `alpha coverage` exists because independence is
counted on the `family` column, and that column being partly empty is itself a
finding rather than something to paper over.

Record identity also changed: an entry with no alpha id is now keyed on the
expression **and** a fingerprint of the settings that alter results (universe,
region, delay, decay, neutralisation, truncation, test period). Keying on the
expression alone silently merged the same formula run under two universes.

## Ask whether a leg earns its place

Adding a leg felt productive and was often a loss: a five-leg version scored below
its four-leg parent. Seeing that required two batches and reading two files.
`ablate` writes every variant at once:

```bash
python research_cli.py ablate --expr "a = ...; n = ...; c = ...; a + n + c" --out /tmp/abl.json
python research_cli.py simulate /tmp/abl.json --sort fitness --stream
python research_cli.py compare <results.json> --contributions
```

The last command reports `contribution = fitness(full) - fitness(full without
this leg)`, so a leg that subtracts from the composite is named instead of
inferred. `solo` arms separate a driver from a carrier: a carrier can add value
while scoring poorly alone.

Variants are generated only when the combination is a plain sum of named legs.
A product, a gated leg like `r*(1-rank(cap))`, or a call inside the combination is
refused with the reason, because "remove a leg" has no syntactic meaning there and
a silently wrong variant costs quota and returns something that looks like
evidence.

`compare` also merges batches: several result files become one table sorted by
fitness, with failing checks and the train/test split side by side.

## Test an independence claim

`SELF_CORRELATION` reports one opaque number and never names the counterpart, so
it cannot tell you whether two candidates are two bets or one. `alpha corr` works
on the daily PnL series instead:

```bash
python research_cli.py alpha corr A B C                 # pairwise matrix
python research_cli.py alpha corr A --vs-book           # max against ACTIVE
python research_cli.py alpha corr A B --control CDE     # residual correlation
```

Every pair is reported, not just each candidate's maximum: two alphas that each
look independent of the book can still be 0.6 with each other, and that is the
pair that decides whether you have two bets. Pairs that overlap for fewer than
`--min-overlap` trading days are listed as skipped rather than dropped, because a
missing correlation and a zero correlation mean different things.

`--control` removes a shared carrier first: regress the control's PnL out of each
series, then correlate the residuals. If a resemblance disappears, the two had
the same factor and the same risk; if it survives, the resemblance is real.

Daily PnL is cached under the results tree, so re-running a comparison costs
nothing. The endpoint returns 200 with an empty body when polled too quickly;
that is retried rather than recorded as "no PnL", which would otherwise leave
correlation tables missing exactly the rows fetched fastest.

## Find the paper, not the blog post

`search` is a web search with reranking. Measured over three mechanism queries it
returned zero to one academic domain in ten; the rest were research-blog posts,
accounting explainers and brokerage pages. That is fine for looking up a definition and
useless for finding where an anomaly comes from.

`paper` asks OpenAlex instead, which is free and needs no key:

```bash
python research_cli.py paper 'net operating assets stock returns anomaly' --limit 3
python research_cli.py paper 'analyst coverage breadth' --from-year 2010 --min-citations 50
```

Every row carries a journal, a year, a DOI and a citation count, plus the abstract when
the publisher licensed it. Two details make it work:

* results are ranked by **relevance**, not citations. Sorting by citations answers a
  mechanism query with the most-cited finance papers ever written - the Fama-French
  mutual fund study among them - because citation count is not relevance;
* OpenAlex ships abstracts as an **inverted index**, `{"word": [positions]}`. It has to be
  reconstructed, otherwise every hit is a title and the tool is no better than the web
  search it replaces.

A missing abstract means the publisher withheld it; that is reported as absent rather than
guessed at.

## Check an idea against the literature

A field name tells you the data exists. It does not tell you whether the economic
claim has already been tested and rejected. `search` closes that gap without
leaving the CLI:

```bash
python research_cli.py search "equity method income persistence abnormal returns" --min-authority 3
python research_cli.py search "deferred revenue anomaly" --count 20 --content
python research_cli.py search --fetch https://www.nber.org/papers/w12345
```

Requires `DEEPSEEK_API_KEY` in `.env` (or the environment); nothing else is
needed and the key is never written to the repo. Search and Fetch are free of
charge until billing starts on 2026-11-01 Beijing time.

Results keep the provider's ranking plus `rerank_score` (1.5+ is high quality),
`authority` (3+ is authoritative) and `date`, so a journal article can be told
apart from a blog post restating it. Answers are cached for a week inside the
results tree, so re-running the same query in a later session costs nothing;
`--no-cache` bypasses that.

Query operators work as documented by the provider: `site:`, `"exact phrase"`,
`-exclude`, `lang:`, `after:`, `before:`. Queries longer than 128 characters are
rejected locally, because the server's error does not say what to shorten.

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

## License

MIT, see `LICENSE`. The tool is a client for WorldQuant BRAIN and carries no market
data; anything produced by running it belongs to whoever ran it and is subject to the
platform's own terms.

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
