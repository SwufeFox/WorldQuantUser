# Compressing the ledger into units

The ledger stores expressions. That is the wrong granularity to think in: after a few
hundred simulations there are hundreds of strings, most of them one small step away
from a dozen others, and no way to ask *what has this construction not tried yet*.

This document records the decomposition, the hierarchy, and **where compression
stops and why**. Run `python3 tools/unit_report.py` against the current research
directory for counts and coverage; private corpus measurements stay with that
research evidence.

## Why not sugar, and not parts

Sugar and parts answer different questions from unit compression:

| Attempt | What it captures | Why it is insufficient |
| --- | --- | --- |
| Sugar forms (`sugar_rewrite`, since removed) | Alternative spellings of a shape | Does not capture variation in fields and constants. |
| Parts catalogue (`parts_catalogue`) | Reusable ingredients | Does not identify which complete constructions or grid points have been tried. |

The variation lives one level down, in **which data** and **which constants** a fixed
shape is given, and one level up, in **how many legs** are combined. Compressing on
those two axes is what `alpha_unit` does.

## The decomposition

An expression becomes five things:

| Piece | What it is |
| --- | --- |
| **assembly** | How the legs are mixed, as a string over `G` (leg) and `L` (offset): `G -L -G` is "leg, minus a constant, minus another leg" |
| **legs** | The additive terms, coefficients factored out |
| **role** | A leg with its fields removed and its cleaning operators folded: `rank(ratio(F,F))` is "a ranked ratio" |
| **data class** | Dataset family of each field, from the field catalogue |
| **fillings** | Which field in each field slot, which constant in each number slot, and the settings |

Three decisions in there were paid for by a failure:

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

**A scale factor on the whole signal is divided out.** Under portfolio
normalisation, multiplying the whole signal by a positive scalar does not create
a new construction. Factoring it out before decomposing the sum prevents a scaled
copy from landing in its own template. Weights *inside* the sum remain grid values
and are never stripped.

## The hierarchy

| Level | What question it answers |
| --- | --- |
| Ledger rows | What was filed, including repeated records per key |
| Expressions (after inlining) | What was run: variable names are not content |
| Assemblies | How legs are mixed |
| Kinds (assembly × data family) | What combinations of shape and data families exist |
| Units (exact template) | Which constructions and grid points are represented |
| One-step moves | What can be tried next using values already observed |

Coarser levels compress more but say less about an individual alpha. Singleton
templates and sparse grids indicate limited coverage of a construction; they do
not establish that its economic hypothesis has been exhausted.

### The alternative that looked better and was rejected

Grouping by *leg set* instead of template merges an expression into any larger
expression containing it. This can compress further, but containment can chain
unrelated constructions through a shared leg.

The same issue arises when family graphs merge on `extends` edges. Keep sibling
identity separate from containment: subset derivation remains available as the
labelled rule `drop:lN`, without treating parent and child as one template.

## What each level is for

* **Assembly** describes mixing patterns, such as `G` (one leg), `G -L` (one leg,
  centred), or `G -L -G` (a carrier minus an offset and a stabiliser).
* **Kind** combines assembly with data families, allowing a reader to distinguish
  shapes that use different kinds of inputs without asserting a performance rank.
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

Ablation needs to handle subtraction as well as addition. A composite written
`-l0 - l1 - 0.5*l2` must not disappear from derivation merely because its coefficient
sum is negative. The shared annotation layer keeps subtraction and renormalisation
rules consistent with rendered units.

`port` attaches a replacement leg to a specific position. Its guards require the
same role, the same data families and another shared leg, so a proposed substitution
has an explicit structural reason. A generic parts catalogue cannot establish
that positional compatibility.

Use the report's derivation counts to find untried data moves and ablations.
Repeated settings for one construction do not constitute additional constructions,
and an untried move is a coverage gap rather than evidence of likely success.

Caveat worth keeping: many `data` moves are cosmetic — `x / atq` against
`x / ltq` is a different grid cell and barely a different bet. The count says "open",
not "promising".

## The round-trip gate

`verify` refills every template from its own filling and demands canonical equality
with the source expression. A template that cannot be re-instantiated cannot support
a trustworthy coverage claim. The current report supplies any failures.

## How to recompute

```
python3 tools/unit_report.py                 # the hierarchy and the top of each level
python3 tools/unit_report.py --gaps          # book derivations never run
python3 tools/unit_report.py --unit UNIT_KEY   # one unit in full
python3 tools/unit_report.py --derive UNIT_KEY # what that unit has not tried
python3 tools/unit_report.py --cross         # templates assembled from two existing ones
python3 tools/unit_report.py --json units.json
```

## Known limitations

* **The role string fragments.** An unfamiliar operator keeps its own name instead
  of being folded into `other`. This preserves distinctions but makes coarse role
  counts less compact.
* **`kind` on a single-leg assembly is coarse.** A `fundamental` + `G` bucket can
  contain several economic ideas. The kind answers *what data, how mixed*, not
  *which hypothesis*.
* **`expand_domain` can pick a normaliser.** If the carrier slot mixes families it is
  not eligible, and the next most-varied slot may be a denominator
  (`atq`/`ltq`/`saleq`). Pass `slot=` explicitly when that matters.
* **Settings are carried but not compressed.** Two rows of the same expression with
  different neutralisation are one unit and one point, with the settings recorded per
  point. The grid does not yet treat them as an axis.