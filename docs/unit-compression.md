# Compressing the ledger into units

The ledger stores expressions. That is the wrong granularity to think in: after a few
hundred simulations there are hundreds of strings, most of them one small step away
from a dozen others, and no way to ask *what has this construction not tried yet*.

This document records the decomposition, the measured hierarchy, and — the part that
matters most — **where compression stops and why**. All numbers were recomputed on
2026-11-01 by `python3 tools/unit_report.py`; do not trust a number quoted here after
the ledger has moved, re-run it.

## Why not sugar, and not parts

Both were measured first, and both fail at the same place.

| Attempt | Result | Why it fails |
| --- | --- | --- |
| Sugar forms (`sugar_rewrite`) | 951 expressions → **932** distinct sugar spellings (−2%) | Sugar works on shape. The variation is not in the shape. |
| Parts catalogue (`parts_catalogue`) | 951 → **424** parts | A vocabulary, not a compression: it says which ingredients exist, not how many dishes there are. |

The variation lives one level down, in **which data** and **which constants** a fixed
shape is given, and one level up, in **how many legs** are combined. Compressing on
those two axes is what `alpha_unit` does.

## The decomposition

An expression becomes five things:

| Piece | What it is | Size on the corpus |
| --- | --- | --- |
| **assembly** | How the legs are mixed, as a string over `G` (leg) and `L` (offset): `G -L -G` is "leg, minus a constant, minus another leg" | **40** |
| **legs** | The additive terms, coefficients factored out | 1–8 per expression, 1625 in total |
| **role** | A leg with its fields removed and its cleaning operators folded: `rank(ratio(F,F))` is "a ranked ratio" | **273** |
| **data class** | Dataset family of each field, from the field catalogue (4417 fields, 14 datasets, all 373 used fields covered) | 7 families |
| **fillings** | Which field in each field slot, which constant in each number slot, and the settings | the grid |

Two decisions in there were paid for by a failure:

**Fields are shared slots, numbers are not.** A field appearing twice is one slot,
because ``rank(close - low)`` and ``rank(close + low)`` are the same construction with
the same input. Numbers appearing twice are *two* slots, because sharing them by value
made ``- 0.5 * x - 0.5`` and ``- 0.5 * x - 1.5`` land in different units although only
a weight changed — a change of weight must not be a change of template.

**Identity goes through a tidy pass.** Expressions built by joining signed
coefficients with ``+`` spell ``a + -0.5`` what a human writes as ``a - 0.5``. Without
a syntactic rewrite for exactly that, every generated ablation would compare unequal
to the hand-written original and be reported as novel. Four rewrites, none algebraic:
``a + -x -> a - x``, ``1 * x -> x``, ``-1 * x -> -x``, ``x + 0 -> x``. Nothing is
commuted, factored or folded.

## The hierarchy, measured

| Level | Count | What question it answers |
| --- | --- | --- |
| Ledger rows | 1261 | what was filed |
| Expressions (after inlining) | **915** | what was run: variables are not content |
| Assemblies | **40** | how are legs mixed |
| Kinds (assembly × data family) | **129** | what kinds of alpha exist here |
| Units (exact template) | **550**, 431 of them singletons | the construction, plus its grid |
| One-step moves | **477** | what can be tried next without inventing a value |

So the compression is **1.66x at template level, 7.1x at kind level, 23x at assembly
level** — and the level that compresses best is the level that says least about any
single alpha. That is not a defect of the method, it is a fact about the corpus:

> 431 of 550 templates contain exactly one expression. The exploration was
> breadth-first: one expression per idea, then move on. 53% of expressions sit in a
> shared template, and only 477 single-slot moves are available across the whole
> corpus with values it already uses.

### The alternative that looked better and was rejected

Grouping by *leg set* instead of by template does compress further: 531 signatures
collapse to **403** maximal ones (2.27x), because an expression that is "another one
minus a leg" merges into the larger one. Measured, then rejected:

```
largest maximal unit: 8 legs, absorbs 23 signatures / 48 expressions
smallest absorbed:    group_rank(anl4_..._eps_median / close, subindustry) - 0.5
```

A single-leg expression is a subset of an eight-leg one, so containment merges
unrelated constructions. This is the same failure `family_graph` hit when it merged on
`extends` edges and produced a 467-member "family"; the fix there — merge only on
siblings — is the fix here — do not merge on containment at all. Subset derivation
stays available as a *rule* (`drop:lN`), where it is one leg and it is labelled.

## What each level is for

* **Assembly** is the cheapest reading of the corpus and the most surprising: five
  patterns cover two thirds of everything. `G` (one leg) is 280 expressions,
  `G -L` (one leg, centred) 118, `G -L -G` (carrier minus an offset minus a
  stabiliser) 93. The corpus is overwhelmingly "one ranked ratio, occasionally
  centred, occasionally with one price-based stabiliser subtracted".
* **Kind** is for reading. `fundamental` + `G` is 113 expressions; `fundamental,price`
  + `-G -L G` is 89 and holds the best single-leg-plus-stabiliser template in the
  corpus (F 1.76).
* **Unit** is for deriving. It carries the exact template and the grid; everything in
  the next section operates on it.
* **One-step moves** is for choosing what to run next.

## Derivation

`alpha_unit.derive` returns variants with a status: `observed` (this exact expression
is in the ledger), `tried` (it was run and abandoned), `novel`. Nothing writes to the
ledger; nothing is simulated.

| Rule | Where it comes from | Guard |
| --- | --- | --- |
| `zero:lN` / `drop:lN` | `annotations.derive_ablations` on the program rendered by `to_program` | none needed — asked of every leg, annotated or not |
| `data:slot` | the unit's own grid | a slot may only move to a value the corpus already put **in that slot** |
| `domain:slot` | `expand_domain`, opt-in | the slot must have shown variation, sit in one allowed family, and be the most-varied eligible one |
| `port:lN` | `crossover` between two units of one kind | same role, same data families, and one other leg already shared |

The ablation pair is deliberately *not* re-implemented here: `to_program` emits one
assignment per leg with an `@leg` annotation, so the definition of "drop a leg" stays
the one already in `annotations.py`, including its renormalisation. `annotations.check`
passes on every rendered unit, which is what makes the annotation layer and the unit
layer the same vocabulary rather than two.

`port` is the derivation the parts catalogue could not express: a part is not
attached to a position, so "put this proven leg where that one sits" had no way to
be stated. Measured on the corpus: **114 candidate ports, 72 never run**. The guards
are what make the number small — a port has to be able to say *why* the two legs are
interchangeable, and "both are ranked ratios over fundamentals" is a reason while
"both are subexpressions" is not. Loosening the family guard to the family instead of
the dataset raises the yield only to 70 novel, so the corpus, not the guard, is what
limits it.

**Measured on the book (2026-11-01):** 43 constructions, **408 derivations never run**
— 233 data moves, 100 `zero_leg`, 75 `drop_term`. The ablations dominate because the
composites were never taken apart: the 13-member balance-sheet family shares one
template and nobody removed its price-position leg.

Caveat worth keeping: many `data` moves are cosmetic — `x / atq` against
`x / ltq` is a different grid cell and barely a different bet. The count says "open",
not "promising".

## The round-trip gate

`verify` refills every template from its own filling and demands canonical equality
with the expression it came from: **915/915, 0 failures**. This is the property that
separates a compression from a summary. A template that cannot be re-instantiated
makes the coverage number say 100% while nothing was checked.

## How to recompute

```
python3 tools/unit_report.py                 # the hierarchy and the top of each level
python3 tools/unit_report.py --gaps          # book derivations never run
python3 tools/unit_report.py --unit 3f2a     # one unit in full
python3 tools/unit_report.py --derive 3f2a   # what that unit has not tried
python3 tools/unit_report.py --cross         # templates assembled from two existing ones
python3 tools/unit_report.py --json units.json
```

## Known limitations

* **The role string fragments.** 273 distinct roles for 1625 leg instances, because an
  unfamiliar operator keeps its own name in the string instead of being folded into
  `other`. That is the right trade for honesty and the wrong trade for counting; the
  top six roles cover 47% of instances, the tail is where the count goes.
* **`kind` on a single-leg assembly is coarse.** The 113-expression `fundamental` + `G`
  kind is one bucket for "rank of something fundamental", which is several different
  ideas. The kind answers *what data, how mixed*, not *which hypothesis*.
* **`expand_domain` can pick a normaliser.** If the carrier slot mixes families it is
  not eligible, and the next most-varied slot may be a denominator
  (`atq`/`ltq`/`saleq`). Pass `slot=` explicitly when that matters.
* **Settings are carried but not compressed.** Two rows of the same expression with
  different neutralisation are one unit and one point, with the settings recorded per
  point. The grid does not yet treat them as an axis.