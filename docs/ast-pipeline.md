# The expression tree, and what it is for

Everything below depends on one change: FASTEXPR is parsed into a tree instead of
being inspected as a string. This document records *why*, what it bought, what was
measured before each step, and what was measured and then deliberately not done —
so the same ground is not re-covered.

## Why a parser

Three failures, all of which cost real simulations:

| Failure | What the string view did |
| --- | --- |
| `historical_volatility_252` does not exist (the catalogue stops at 180) | Nothing could check a name it had never seen. One batch died on it. |
| `hv = ...; band = ...` — locals looked like fields | Regex reported the locals as unknown datafields, so readable composites were rejected by the linter. |
| `//leverage = liabilities / assets` written as a comment | Regex read the comment as code and reported `liabilities` as a field. A `/* HYPOTHESIS */` block produced 47 phantom fields in one expression. |

The tree fixes all three structurally, and it makes the next questions askable at
all: *are these two alphas the same construction*, *what does this annotation point
at*, *what happens if this leg is removed*.

## The grammar

`fastexpr_ast.parse` is a hand-written recursive-descent parser producing immutable
nodes: `Number`, `String`, `Field`, `Call`, `BinOp`, `UnaryOp`, `Ternary`,
`Annotation`, and `Program` (`assignments`, `signal`, `annotations`).

* Comments: `//`, `#`, `/* */` are blanked before tokenising, with offsets
  preserved so error messages still point at the original text.
* Precedence: `|| < && < comparisons < + - < * / < unary < ^`, with rendering that
  re-inserts parentheses only where the tree requires them.
* Round-trip: `parse(render(parse(x))) == parse(x)` for every expression in the
  ledger (1112 distinct, all parse as of 2026-10-30).

Two properties are load-bearing and are pinned by tests:

* `field_names` excludes identifiers the program binds itself, so `hv` is a local
  and `subindustry` is a field, decided by position rather than by spelling.
* `skeleton` replaces fields with `F` and literals with `L`, giving a structure key
  that is stable under renaming a field and sensitive to changing a shape.

## Sugar

`alpha_sugar` registers macros that expand at parse time, so the string handed to
the simulator contains no sugar at all. Each one encodes a decision that had already
been paid for:

| Form | Expands to | Why it exists |
| --- | --- | --- |
| `rrank(x, g)` | `group_rank(x, g) - 0.5` | Forgetting the centring only matters before a multiplication, which is exactly where it gets forgotten. |
| `band(x, w)` | `-(abs(x) - w)` | The width is a constant, and constants vanish under neutralisation. Naming it `band` stops it from looking tunable. |
| `trough(x, w)` | `abs(x) - w` | The mirror, for "both ends are good". |
| `ratio(a, b, eps)` | `a / (abs(b) + eps)` | Keeps the sign of `b` out of it, and does not double-wrap an `abs` the caller already wrote. |
| `spread(hi, lo)` | `(hi - lo) / (abs((hi+lo)/2) + eps)` | Bid-ask style spreads without repeating the midpoint. |
| `mix(a, b, w)` | `w*a + (1-w)*b` | Weight arithmetic done at compile time. |
| `combo(...)` | equal-weight sum | Hand-written coefficients drifted. |
| `gate(cond, x)` | `if_else(cond, x, 0)` | Conditional exposure as an expression, not a mask. |
| `deadband(x, cut)` | `if_else(abs(x) > cut, x, 0)` | The mask-by-multiplication version is wrong under group neutralisation: a zeroed name becomes the group mean's mirror image, not a flat position. |

## Operator knowledge: catalog versus invariants

`GET /operators` gives a name, a signature string and a level. It does **not** give
return types or cross-argument invariants, and no amount of fetching will produce
them. The split is therefore explicit:

* **Catalog** — operator names, arity, field names, field types: fetched and cached
  (`cache/metadata/`), never written into code.
* **Invariants** — `kth_element` needs `k <= d`, `bucket` takes exactly one of
  `range`/`buckets` and needs strictly increasing edges, a `bucket()` signal is
  `Group`-typed and therefore invalid: curated in `operator_rules.default.json`
  (shipped) with a cache override at `<research>/cache/metadata/operator-rules.json`.

Parameter names in constraints are read from the catalog's own `definition` string,
so a renamed parameter makes a rule skip rather than misfire.

Cross-check against FastPlus (Rust, MIT), which hardcodes 108 operators for an
Expert account: this account has 66, a strict subset. Its positional typing accepts
`group_rank(close, banana)` and `pasteurize(...)`; the linter here rejects both. The
two invariant families above were the only real gap, and they are now covered.

## Annotations

```
@mechanism("distress carrier + news impact + shape stabiliser")
@leg(name="a", role="carrier")
@ablate(leg="sy")
a = ...;
sy = ...;
```

`render()` drops annotations, so the compiled expression is byte-identical to one
written without them; `render_source()` keeps them. The vocabulary lives in
`annotation-schema.json`, so an unknown name is reported rather than ignored.

`annotations.check` decides staleness **structurally**: an annotation that names a
leg must name one the program defines. This is the failure that matters — a renamed
leg leaves the annotation rendering happily while every ablation it requests becomes
a silent no-op.

`annotations.derive_ablations` produces two kinds, because they answer different
questions:

* `zero_leg` — set the assignment to `0`, keep every weight. Isolates "does this
  leg's *content* matter" from "does its *weight* matter".
* `drop_term` — remove the terms referencing the leg and renormalise the remainder
  (`0.7a + 0.3b` → `1.0 * b`). The cruder "what if we never had it". It refuses
  when the signal is not a flat sum rather than inventing a fallback.

## Intent rewrites

`intent.plan` finds `rank(field)` where the argument is a raw level, and offers
`rank(field / size)`. A raw accounting amount is dominated by company size across the
cross-section, so ranking it mostly ranks size; dividing by total assets leaves the
composition the idea was about.

Three guards, each from a specific failure mode:

* the whole argument must be one bare field, so `group_rank((x))` and `x + y` are
  not half-matched;
* the field must be in the curated level list — a field whose unit was never
  measured cannot be classified by name without inventing knowledge;
* an already-normalised expression is left alone instead of being divided twice.

The rewrite is *offered with a reason*, never applied silently.

**Measured coverage, stated honestly: zero.** Of 3023 rank-family calls in the
ledger's 1102 distinct expressions, 177 take a genuinely bare field (locals excluded)
and none of those is a level — they are `cap`, `volume`, `close/open/low/high/vwap`,
`historical_volatility_*`, `adv15`, i.e. size and market series ranked bare on
purpose. The pattern this rewrite targets does not occur in what has already been
written, so it is a preventive check for new expressions rather than a repair pass.

A second measurement shaped the design: most accounting amounts in this catalogue are
stored as dimensionless (`assets`, `atq`, `revenue`, `cogs` all measure that way), so
`Unit[TSPrice:1]` does not identify "amountness". Levels cannot be derived from units
alone; `intent.derive_levels` adds candidates and is disjoint from the curated list by
construction, and the curated list stays the primary signal.

## The family graph

`tools/family_graph.py` builds nodes from structure and three kinds of edge:

| Edge | Test | Meaning |
| --- | --- | --- |
| `sibling` | identical skeleton | the same expression with different inputs |
| `extends` | leg set is a proper subset | "that alpha plus a leg" |
| `shares` | one common leg, no containment | weak relation, excluded from components |

Measured on the live book (2026-10-30): **48 alphas, 183 edges — 83 sibling, 36
extends, 64 shares — collapsing to 22 construction families**, largest 18, 16
singletons. `tools/family_graph_png.py` draws one component per cell, because a
single spring layout over 48 nodes is a hairball that hides the only two facts worth
seeing: how many families there are, and how lopsided the sizes are.

**Structure is not independence.** The same 48 alphas correlate at ~0.85 pairwise
while occupying 22 families, and candidates rebuilt from those families are blocked
by self-correlation at 0.84–0.87. The graph finds copies cheaply; only daily PnL
finds shared bets.

`--api-book` reads the book from the platform rather than from the ledger's `status`
column, because that column is written at submission time and never reconciled: it
marked 33 ACTIVE against 48 live.

## One entry point for simulation

`tools/guarded_sim.py` lints against the cached catalogue and raises before spending
a simulation. The lint already knew about unknown names; the research scripts simply
called `ml.simulate_alphas` directly and skipped it. A rule that can be bypassed is
not a rule.

## Measured, then not done

Two migrations were prepared and then declined, because the measurement said they
were no-ops. Recorded here so they are not retried on principle:

| Migration | Measurement | Decision |
| --- | --- | --- |
| Operator extraction → tree | 1103/1103 identical to regex | not worth the change surface |
| Arity check → tree | zero false warnings on nested calls, kwargs, deep nesting; zero findings across 1112 ledger expressions | same |

Field extraction **was** migrated: 1097/1103 identical, and all 6 differences were
the regex inventing fields out of comments or missing a CRLF-separated local.

The macro backend **was** migrated. Text substitution is correct only while every
argument is atomic; over 2728 sampled combinations the text backend leaked a nested
macro 1262 times (its call regex cannot match parentheses in an argument) and changed
meaning 101 times. Every difference is attributable to a compound argument in a
non-atomic position, and **zero** differences appear where the text output was
already correct. That zero is the acceptance test in `tests/test_ast_macros.py`.

## Ledger hygiene

The ledger is written at submission and reconciled by nothing, so it drifts in ways
that quietly change every count derived from it. Three tools now close those gaps:

| Tool | Defect it removes |
| --- | --- |
| `tools/repair_ledger.py` | 9 expressions with a variant index glued to the tail (repaired, annotated); 191 rows with no alpha id at all (marked `abandoned`, not deleted) |
| `tools/reconcile_status.py` | the `status` column lagging the platform — measured 33 ACTIVE against 48 live, i.e. 15 rows that had been submitted but still said COMPLETE or UNSUBMITTED |
| `tools/research_template.py` | a new script having to remember the guard, the sugar, the annotation check and where results go |

Consequences worth remembering: **191 of 2118 rows are shells**, so "how many alphas
do I have" needs that subtracted first; and **rows outnumber alphas**, because one
alpha can appear several times, so a line like "51 rows promoted" is not a claim
about 51 alphas.

Two audits that are now free: with 63 fields' units measured, the 48 ACTIVE
expressions contain **zero** dimensional inconsistencies (expected — the simulator
rejects those at submission — but the check costs nothing now), and the same unit
table gates the intent rewrite.

## Known limits

* **Unit coverage.** Intent rewriting wants dimensional analysis, but units can only
  be measured one simulation at a time. 48 fields are measured; the catalogue has
  thousands. Outside the curated list the rewrite stays silent, which is correct and
  also means coverage is the binding constraint on scaling it.
* **`returns` carries `Unit[TSPrice:1]`** as measured, i.e. it behaves like a price
  change rather than a ratio. Anything computing a dispersion of `returns` should be
  read with that in mind.
* The curated level/size lists in `intent-rules.json` are inputs, not truths: they
  name what is unambiguous, and absence means "not classified", not "not a level".