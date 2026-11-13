# The expression tree, and what it is for

Everything below depends on one change: FASTEXPR is parsed into a tree instead of
being inspected as a string. This document records *why*, what it bought, what was
measured before each step, and what was measured and then deliberately not done —
so the same ground is not re-covered.

## Why a parser

Three failure modes motivate structural parsing:

| Failure | What the string view did |
| --- | --- |
| An unknown field name | A parser can extract identifiers reliably; the catalog then decides whether each exists. |
| `hv = ...; band = ...` — locals look like fields | Regex can report bound locals as unknown datafields. |
| `//leverage = liabilities / assets` written as a comment | Regex can read comments as code and invent field references. |

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
* Round-trip property: `parse(render(parse(x))) == parse(x)`.

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

Operator availability is account-scoped. Another tool's hardcoded list does not
establish what this account can run. The live catalog supplies availability;
curated invariants cover constraints that signatures alone do not express.

## Annotations

```
@mechanism("sample carrier and stabiliser")
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

This is a preventive check for new expressions. Market series may be ranked bare
intentionally, so the number of bare-field rank calls is not the number of mistakes.

Accounting amounts may be stored as dimensionless, so units alone do not identify
levels. `intent.derive_levels` adds candidates disjoint from the curated list;
the curated list remains the primary signal for this rewrite.

## The family graph

`research_cli.py ledger mechanisms` builds nodes from structure and three kinds of edge:

| Edge | Test | Meaning |
| --- | --- | --- |
| `sibling` | identical skeleton | the same expression with different inputs |
| `extends` | leg set is a proper subset | "that alpha plus a leg" |
| `shares` | one common leg, no containment | weak relation, excluded from components |

Families are connected components over `sibling` edges only. `extends` is recorded as
lineage and deliberately excluded from the merge: containment is transitive through a
shared ancestor, so single linkage chains unrelated constructions together — A with one
leg, B = A + leg1, C = A + leg2 puts B and C in one "family" although they share no
construction. Current mechanism reporting groups by economic block and skeleton;
containment remains lineage evidence rather than proof of one construction.

**Structure is not independence.** Distinct constructions can share the same
underlying bet. Structural grouping finds copies cheaply; daily PnL comparison
is needed to assess shared exposure.

A platform book query supplies current membership; a local ledger's status reflects
the last recorded observation and may be stale until reconciled with `ledger sync`.

## One entry point for simulation

`tools/guarded_sim.py` lints against the cached catalogue and raises before spending
a simulation. The lint already knew about unknown names; the research scripts simply
called `ml.simulate_alphas` directly and skipped it. A rule that can be bypassed is
not a rule.

## Measured, then not done

Not every string-based operation needs migration. Operator extraction and arity
checking can remain on their existing paths when structural parsing adds no
correctness benefit. Compare behavior before expanding the change surface.

Field extraction did need the tree: comments must not invent fields, and locals
must remain locals regardless of whitespace or line-ending style.

The former macro backend needed structural substitution because compound arguments
can contain nested calls and operators. A call regex cannot reliably substitute
them without leaking macros or changing precedence. That macro system was later
removed because `alpha_unit` templates also report untried grid points.

## Ledger hygiene

Submission-time records can drift from platform state. The tools separate ongoing
reconciliation from one-off repairs:

| Tool | Defect it removes |
| --- | --- |
| `repair_ledger` (one-off, removed) | Malformed expression suffixes and records without alpha IDs required explicit repair or an `abandoned` label. |
| `ledger sync` (was `tools/reconcile_status.py`) | Refreshes local status against the platform. |
| `probe` spec files (were `tools/research_template.py`) | a new script having to remember the guard, the sugar, the annotation check and where results go |

Rows without usable identities must be distinguished from alpha records. Rows can
also outnumber alphas because one alpha may appear repeatedly; report both counts
with their units rather than treating a row update count as a book size.

The measured field-unit table supports dimensional audits as well as intent
rewrites. An unmeasured field remains unknown; absence of a finding is not proof
that its dimensions are compatible.

## Known limits

* **Unit coverage.** Intent rewriting wants dimensional analysis, but units are
  measured through simulation. Outside the curated list the rewrite stays silent;
  incomplete measurement coverage limits what it can establish.
* **`returns` carries `Unit[TSPrice:1]`** as measured, i.e. it behaves like a price
  change rather than a ratio. Anything computing a dispersion of `returns` should be
  read with that in mind.
* The curated level/size lists in `intent-rules.json` are inputs, not truths: they
  name what is unambiguous, and absence means "not classified", not "not a level".