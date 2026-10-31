#!/usr/bin/env python3
"""Compress the ledger into units: one construction, many grid points.

The ledger stores expressions. That is the wrong granularity to *think* in: 900+
strings, most of them one knight's move away from a dozen others, and no way to ask
"what has this construction not tried yet". Sugar did not fix it -- measured on the
2026-10-31 corpus, 951 expressions reduce to 932 distinct sugar forms, i.e. the
compression is 2%, because sugar works at the shape level and the variation lives
one level down, in *which data* and *which constants* a fixed shape is given.

So this module compresses on the axis that actually varies. An expression is split
into

* an **assembly** -- how the legs are combined: ``G -L -G`` is "leg, minus a constant,
  minus another leg". Only 40 of these exist in the whole corpus, and five of them
  cover two thirds of it;
* a tuple of **legs** -- the subexpressions being combined, each classified by a
  *role* (``rank(ratio(F,F))``, ``delta(F)``, ``disp(gate(...))``), which is what the
  leg *does* independent of which field it reads;
* a **data class** per leg, from the field catalogue's dataset id;
* the **fillings**: which field sits in each field slot, which constant in each
  number slot, and the settings.

Two expressions that agree on (assembly, leg shapes, slot *layout*) are one **unit**:
one template plus a grid. The template is a real tree with :class:`Slot` leaves, so a
unit can be re-instantiated at any grid point -- that is the "derivable" half. The
grid is what has actually been run -- that is the "compressible" half.

The measured hierarchy on 2026-10-31 (932 non-abandoned, parseable, deduplicated
expressions) is reported by ``python3 tools/unit_report.py``; see
``docs/unit-compression.md`` for the table and for what does *not* compress.

Design rules, each one paid for:

* **Nothing is guessed.** A field with no catalogue entry is ``unknown``, an operator
  that is not in ``unit-rules.json`` keeps its own name in the role string instead of
  being folded into ``other``. A silent collapse is invisible once stored.
* **Derivation is offered, never applied.** :func:`derive` returns variants with a
  status; nothing writes to the ledger. Expanding a slot's domain beyond the values
  the corpus already used is opt-in (:func:`expand_domain`), because the reachable
  space is where invented alphas come from.
* **A unit is only worth calling a unit if it round-trips.** :func:`verify` refills
  every point's template and demands canonical equality with the expression it came
  from. A compression that cannot reproduce its input is a summary, not a compression.
"""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field as dc_field

import annotations
import fastexpr_ast as fa

__all__ = [
    "Slot",
    "Leg",
    "Point",
    "Unit",
    "Kind",
    "Corpus",
    "Variant",
    "load_rules",
    "load_field_datasets",
    "role_of",
    "split",
    "template_of",
    "fill",
    "build",
    "derive",
    "crossover",
    "to_program",
    "expand_domain",
    "verify",
    "summary",
    "render_unit",
]

_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unit-rules.json")

#: Where the cached BRAIN field catalogue lives. Only used to answer "which dataset
#: does this field belong to"; absent is fine, every field then reads ``unknown``.
_FIELDS_CACHE = "/data/worldquant-research/cache/metadata/datafields"


# --------------------------------------------------------------------------- slots


@dataclass(frozen=True)
class Slot:
    """A hole in a template: one field, or one numeric literal.

    Slots are shared **by value**, not by position: two occurrences of ``close`` are
    one slot, and substituting it moves both. That is the assumption a slot makes --
    ``x/x`` is not ``x/y`` with a coincidence, and a weight that appears twice is one
    weight. When the assumption is wrong the filling says so, because it records the
    value for every slot name rather than re-deriving it per position.
    """

    name: str
    kind: str  # "field" | "number"

    def render(self, parent_precedence: int = 0) -> str:
        return "{%s}" % self.name


def template_of(tree):
    """Return ``(slotted_tree, filling)`` for one expression tree.

    The slotted tree is structure plus slot *layout*; the filling is the data. Two
    expressions with the same slotted tree differ only in the filling, which is
    exactly the statement "same construction, different grid point".
    """
    index, filling = {}, {}

    def slot(kind, value):
        # Fields are shared by value, numbers are not. Sharing a field is the
        # statement "the same input feeds both places", which is a property of the
        # construction; sharing a number because the digits match is a coincidence of
        # the current weights -- `- 0.5 * x - 0.5` and `- 0.5 * x - 1.5` would land in
        # different units although only a weight changed.
        key = (kind, value) if kind == "field" else (kind, len(index))
        name = index.get(key)
        if name is None:
            name = "%s%d" % ("f" if kind == "field" else "n", len(index))
            index[key] = name
            filling[name] = value
        return Slot(name, kind)

    def walk(node):
        if isinstance(node, fa.Field):
            return slot("field", node.name)
        if isinstance(node, fa.Number):
            return slot("number", node.text)
        if isinstance(node, fa.String):
            return node
        if isinstance(node, fa.Call):
            return fa.Call(node.name,
                           tuple(walk(arg) for arg in node.args),
                           tuple((key, walk(value)) for key, value in node.kwargs))
        if isinstance(node, fa.BinOp):
            return fa.BinOp(node.op, walk(node.left), walk(node.right))
        if isinstance(node, fa.UnaryOp):
            return fa.UnaryOp(node.op, walk(node.operand))
        if isinstance(node, fa.Ternary):
            return fa.Ternary(walk(node.condition), walk(node.when_true),
                              walk(node.when_false))
        return node

    return walk(tree), filling


def _positive_form(node):
    """The same value with its leading minus flipped away, or ``None``.

    A generated sum joins signed coefficients with ``+``, so a negative weight shows
    up either as a negative literal or as a product led by one. Both spell the same
    thing as a subtraction, and the identity used for dedupe has to see that.
    """
    if isinstance(node, fa.Number) and node.text.startswith("-"):
        return fa.Number(node.text[1:])
    if isinstance(node, fa.BinOp) and node.op == "*":
        for side in (0, 1):
            factor = node.left if side == 0 else node.right
            other = node.right if side == 0 else node.left
            flipped = None
            if isinstance(factor, fa.Number) and factor.text.startswith("-"):
                flipped = fa.Number(factor.text[1:])
            elif isinstance(factor, fa.UnaryOp) and factor.op == "-":
                flipped = factor.operand
            if flipped is not None:
                return fa.BinOp("*", flipped, other) if side == 0 else \
                    fa.BinOp("*", other, flipped)
    return None


def _tidy(node):
    """Rewrite spellings that differ without meaning it.

    Not algebraic: nothing is commuted, factored or folded. The three rewrites are
    ``a + -x -> a - x``, ``1 * x -> x`` / ``-1 * x -> -x`` and ``x + 0 -> x``, which
    only ever occur because a *generated* expression was built by joining signed
    coefficients with ``+``. Without them the identity used for dedupe would call
    ``a - 0.5`` and ``a + -0.5`` two different alphas and report every ablation as
    novel.
    """
    if isinstance(node, fa.Call):
        return fa.Call(node.name,
                       tuple(_tidy(arg) for arg in node.args),
                       tuple((key, _tidy(value)) for key, value in node.kwargs))
    if isinstance(node, fa.BinOp):
        left, right = _tidy(node.left), _tidy(node.right)
        if node.op == "*":
            if isinstance(left, fa.Number) and left.text in {"1", "1.0"}:
                return right
            if isinstance(right, fa.Number) and right.text in {"1", "1.0"}:
                return left
            if isinstance(left, fa.Number) and left.text in {"-1", "-1.0"}:
                return fa.UnaryOp("-", right)
            if isinstance(right, fa.Number) and right.text in {"-1", "-1.0"}:
                return fa.UnaryOp("-", left)
        if node.op == "+":
            positive = _positive_form(right)
            if positive is not None:
                return fa.BinOp("-", left, positive)
            if isinstance(right, fa.UnaryOp) and right.op == "-":
                return fa.BinOp("-", left, right.operand)
            if isinstance(left, fa.Number) and left.text == "0":
                return right
            if isinstance(right, fa.Number) and right.text == "0":
                return left
        return fa.BinOp(node.op, left, right)
    if isinstance(node, fa.UnaryOp):
        return fa.UnaryOp(node.op, _tidy(node.operand))
    if isinstance(node, fa.Ternary):
        return fa.Ternary(_tidy(node.condition), _tidy(node.when_true),
                          _tidy(node.when_false))
    return node


def fill(node, filling):
    """Instantiate a slotted tree at one grid point.

    A missing slot raises: filling a template with a partial grid would produce an
    expression that silently reads a different field, which is worse than failing.
    """
    if isinstance(node, Slot):
        if node.name not in filling:
            raise KeyError("template slot %r has no value" % node.name)
        value = filling[node.name]
        return fa.Field(value) if node.kind == "field" else fa.Number(str(value))
    if isinstance(node, fa.Call):
        return fa.Call(node.name,
                       tuple(fill(arg, filling) for arg in node.args),
                       tuple((key, fill(value, filling)) for key, value in node.kwargs))
    if isinstance(node, fa.BinOp):
        return fa.BinOp(node.op, fill(node.left, filling), fill(node.right, filling))
    if isinstance(node, fa.UnaryOp):
        return fa.UnaryOp(node.op, fill(node.operand, filling))
    if isinstance(node, fa.Ternary):
        return fa.Ternary(fill(node.condition, filling), fill(node.when_true, filling),
                          fill(node.when_false, filling))
    if isinstance(node, fa.Program):
        return fa.Program(tuple((name, fill(value, filling))
                                for name, value in node.assignments),
                          fill(node.signal, filling) if node.signal is not None else None,
                          node.annotations)
    return node


# --------------------------------------------------------------------------- rules


def load_rules(path=None):
    """Read the unit vocabulary; an unreadable file means "classify nothing"."""
    try:
        with open(path or _RULES) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"opaque": {}, "roles": {}, "binary": {}, "datasets": {}, "domains": {}}


def load_field_datasets(cache_dir=None):
    """Map every cached datafield id to its dataset id.

    Read from the on-disk metadata cache rather than the API: this is used for
    grouping, and a grouping key is not worth a request, let alone one per field.
    A missing cache returns ``{}`` -- callers then see ``unknown`` and say so.
    """
    out = {}
    directory = cache_dir or _FIELDS_CACHE
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name)) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        for record in payload.get("payload") or []:
            if isinstance(record, dict) and record.get("id"):
                out[record["id"]] = (record.get("dataset") or {}).get("id")
    return out


def _role_tables(rules):
    by_operator = {}
    for name, spec in (rules.get("roles") or {}).items():
        if not isinstance(spec, dict):
            continue
        for operator in spec.get("operators") or []:
            by_operator[operator] = spec.get("form")
    for operator in (rules.get("opaque") or {}).get("operators") or []:
        by_operator[operator] = "{0}"
    return by_operator, (rules.get("binary") or {})


def role_of(node, rules=None):
    """The *job* a subtree does, independent of which field it reads.

    ``group_rank(ts_backfill(fnd6_x, 252) / atq, subindustry)`` is
    ``rank(ratio(F,F))``: a ranked ratio. The field names are deliberately gone --
    they are the grid, not the construction. Operators that only clean a value
    (``ts_backfill``, ``abs``) are transparent, so a backfilled leg and its raw form
    are one role; without that the same construction split in two on a detail.

    An operator missing from ``unit-rules.json`` keeps its own name, so an unfamiliar
    shape is visible in the role string instead of being folded into ``other``.
    """
    _rules = rules if rules is not None else load_rules()
    by_operator, binary = _role_tables(_rules)

    def walk(current):
        if isinstance(current, fa.Number):
            return "L"
        if isinstance(current, fa.String):
            return "S"
        if isinstance(current, fa.Field):
            return "F"
        if isinstance(current, fa.BinOp):
            if current.op == "/":
                return (binary.get("/") or "ratio({0},{1})").format(
                    walk(current.left), walk(current.right))
            if current.op in {"+", "-"}:
                return (binary.get("+-") or "mix({0},{1})").format(
                    walk(current.left), walk(current.right))
            if current.op == "*":
                return (binary.get("*") or "mul({0},{1})").format(
                    walk(current.left), walk(current.right))
            return "cmp(%s,%s)" % (walk(current.left), walk(current.right))
        if isinstance(current, fa.UnaryOp):
            return "neg(%s)" % walk(current.operand)
        if isinstance(current, fa.Ternary):
            return "tern(%s,%s,%s)" % (walk(current.condition), walk(current.when_true),
                                       walk(current.when_false))
        if isinstance(current, fa.Call):
            form = by_operator.get(current.name)
            roles = [walk(arg) for arg in current.args]
            if form is not None:
                try:
                    return form.format(*roles)
                except (IndexError, KeyError):
                    pass
            return "%s(%s)" % (current.name, ",".join(roles))
        if isinstance(current, fa.Program):
            return "|".join(walk(value) for _name, value in current.assignments)
        return "?"

    return walk(node)


# ------------------------------------------------------------------------- terms


def terms(node, sign=1.0):
    """Flatten a sum into ``[(signed_weight, body), ...]``.

    Only ``+``/``-``/unary minus are flattened; a product of sums stays one term,
    because it is one decision rather than a list of legs. The coefficient is pulled
    out of ``c * body`` so the weight is a grid value and not part of the shape,
    and the sign is folded into it because ``a - 0.5*b`` and ``a + -0.5*b`` are one
    construction with one weight.
    """
    if isinstance(node, fa.BinOp) and node.op == "+":
        return terms(node.left, sign) + terms(node.right, sign)
    if isinstance(node, fa.BinOp) and node.op == "-":
        return terms(node.left, sign) + terms(node.right, -sign)
    if isinstance(node, fa.UnaryOp) and node.op == "-":
        return terms(node.operand, -sign)
    if isinstance(node, fa.BinOp) and node.op == "*":
        if isinstance(node.left, fa.Number):
            try:
                return [(sign * float(node.left.text), node.right)]
            except ValueError:
                pass
        if isinstance(node.right, fa.Number):
            try:
                return [(sign * float(node.right.text), node.left)]
            except ValueError:
                pass
    return [(sign, node)]


@dataclass(frozen=True)
class Leg:
    """One additive term of a composite: what it computes, and how much it counts."""

    body: object
    shape: str
    role: str
    fields: tuple
    families: tuple  # data families this leg reads, used to keep a port sensible
    numbers: tuple
    weight: float

    @property
    def is_constant(self):
        return isinstance(self.body, fa.Number)


def _numbers(node):
    return tuple(child.text for child in node.walk() if isinstance(child, fa.Number))


# --------------------------------------------------------------------------- point


@dataclass
class Point:
    """One expression, seen as a grid point of one unit."""

    expression: str
    flat: str
    canonical: str
    template: object
    filling: dict
    assembly: str
    terms: tuple
    legs: tuple
    settings: tuple  # every settings bundle this expression was run with, first wins
    datasets: tuple
    metrics: dict = dc_field(default_factory=dict)
    alpha_ids: tuple = ()
    statuses: tuple = ()
    unit: str = ""
    kind: str = ""

    def rebuild(self, filling=None):
        """Re-instantiate this point's template, optionally at another grid point."""
        return fill(self.template, filling if filling is not None else self.filling)

    def render(self, filling=None):
        """The expression at this grid point, with identity spellings tidied away.

        Tidy on the way out, not only on the way in: a template keeps the literal
        ``1`` of ``1 * rank(x)`` as a slot, so a refill would otherwise render a
        weight that the canonical form of the source has already dropped, and the
        round trip would fail on a difference that measures nothing.
        """
        return _tidy(self.rebuild(filling)).render()


def _inline(program):
    """Substitute every local assignment into the signal.

    Legs are then real subtrees rather than names, which is necessary because the
    same construction is written with different variable splits all over the ledger.
    """
    current = program
    for name, _value in reversed(program.assignments):
        table = dict(current.assignments)
        if name in table:
            current = fa.substitute(current, name, table[name])
    return current.signal


def _datasets_of(values, datasets_of_field, families):
    out = []
    for field in values:
        dataset = datasets_of_field.get(field)
        out.append(families.get(dataset) or dataset or "unknown")
    return tuple(sorted(set(out)))


def split(expression, rules=None, datasets=None, settings=None):
    """Decompose one expression into assembly, legs and fillings.

    Returns ``None`` for anything that does not parse or has no signal -- a damaged
    row must not stop a corpus from being built, and pretending it compressed would
    make the coverage number lie.
    """
    rules = rules if rules is not None else load_rules()
    try:
        program = fa.parse(expression)
    except fa.ParseError:
        return None
    signal = _inline(program)
    if signal is None:
        return None

    families = rules.get("datasets") or {}
    known = datasets if datasets is not None else load_field_datasets()

    legs, assembly, term_list = [], [], []
    for weight, body in terms(signal):
        if isinstance(body, fa.Number):
            assembly.append("L" if weight > 0 else "-L")
            term_list.append((weight, body))
            continue
        assembly.append("G" if weight > 0 else "-G")
        term_list.append((weight, body))
        leg_fields = tuple(fa.field_names(fa.Program((), body)))
        legs.append(Leg(body=body,
                        shape=fa.skeleton(fa.Program((), body)),
                        role=role_of(body, rules),
                        fields=leg_fields,
                        families=_datasets_of(leg_fields, known, families),
                        numbers=_numbers(body),
                        weight=weight))

    template, filling = template_of(signal)
    flat = _tidy(signal).render()
    fields = tuple(sorted({leg_field for leg in legs for leg_field in leg.fields}))
    return Point(
        expression=expression,
        flat=flat,
        canonical=fa.canonical(flat),
        template=template,
        filling=filling,
        assembly=" ".join(assembly),
        legs=tuple(legs),
        settings=(tuple(sorted((settings or {}).items())),),
        datasets=_datasets_of(fields, known, families),
        terms=tuple(term_list),
    )


# ----------------------------------------------------------------------- corpus


@dataclass
class Unit:
    """One construction: a template plus every grid point it has been run at."""

    key: str
    template: object
    points: list
    legs: tuple
    assembly: str
    datasets: tuple
    kind: str = ""

    @property
    def size(self):
        return len(self.points)

    @property
    def grid(self):
        """Slot name -> observed values, in first-seen order.

        This is the honest reachable set: a slot that only ever held one value has a
        one-point domain, and inventing more is :func:`expand_domain`'s job, which is
        explicitly opt-in.
        """
        out = collections.OrderedDict()
        for point in self.points:
            for name, value in point.filling.items():
                bucket = out.setdefault(name, [])
                if value not in bucket:
                    bucket.append(value)
        return out

    @property
    def roles(self):
        return tuple(leg.role for leg in self.legs)

    @property
    def fields(self):
        return tuple(sorted({field for leg in self.legs for field in leg.fields}))


@dataclass
class Kind:
    """Construction kinds: one assembly over one set of data families.

    This is the level that answers "what kinds of alpha exist here" -- coarse enough
    to be a handful, specific enough that the answer is not "a rank of something".
    """

    key: str
    assembly: str
    families: tuple
    units: list

    @property
    def size(self):
        return sum(unit.size for unit in self.units)

    @property
    def roles(self):
        seen, ordered = set(), []
        for unit in self.units:
            for role in unit.roles:
                if role not in seen:
                    seen.add(role)
                    ordered.append(role)
        return tuple(ordered)


@dataclass
class Corpus:
    points: list
    units: dict
    kinds: dict

    def unit_of(self, point):
        return self.units[point.unit]

    def kind_of(self, point):
        return self.kinds[point.kind]


def _kind_key(point):
    """Assembly plus data families, in a commutative order.

    Order of terms is a real difference (``a - b`` is not ``b - a``) and stays in the
    unit key. At kind level it does not matter: the question there is "how are legs
    mixed, over which kinds of data", and ``a - b`` and ``b - a`` are the same answer.
    """
    return (" ".join(sorted(point.assembly.split())), point.datasets)


def build(entries, rules=None, datasets=None):
    """Group ledger entries into points, units and kinds.

    Identity is the canonical flat tree, so a row that only differs by variable
    naming, redundant parentheses or a comment lands on the same point instead of
    inflating the count.
    """
    rules = rules if rules is not None else load_rules()
    families = rules.get("datasets") or {}
    known = datasets if datasets is not None else load_field_datasets()

    by_canonical = collections.OrderedDict()
    for entry in entries:
        expression = entry.get("expression")
        if not expression or entry.get("abandoned"):
            continue
        point = split(expression, rules, known, entry.get("settings"))
        if point is None:
            continue
        point.metrics = dict(entry.get("metrics") or {})
        existing = by_canonical.get(point.canonical)
        if existing is not None:
            existing.metrics = existing.metrics or point.metrics
            existing.alpha_ids = existing.alpha_ids + tuple(
                value for value in (entry.get("alpha_id"),) if value)
            existing.statuses = existing.statuses + tuple(
                value for value in (entry.get("status"),) if value)
            existing.settings = existing.settings + tuple(
                value for value in point.settings if value not in existing.settings)
            continue
        point.alpha_ids = tuple(value for value in (entry.get("alpha_id"),) if value)
        point.statuses = tuple(value for value in (entry.get("status"),) if value)
        by_canonical[point.canonical] = point

    units, kinds = collections.OrderedDict(), collections.OrderedDict()
    for point in by_canonical.values():
        key = point.template.render()
        unit = units.get(key)
        if unit is None:
            unit = Unit(key=key, template=point.template, points=[], legs=point.legs,
                        assembly=point.assembly, datasets=point.datasets)
            units[key] = unit
        unit.points.append(point)
        point.unit = key
        kkey = _kind_key(point)
        kind = kinds.get(kkey)
        if kind is None:
            kind = Kind(key=kkey, assembly=kkey[0], families=kkey[1], units=[])
            kinds[kkey] = kind
        if not any(existing is unit for existing in kind.units):
            kind.units.append(unit)
        point.kind = kkey
        unit.kind = kkey

    return Corpus(points=list(by_canonical.values()), units=units, kinds=kinds)


# --------------------------------------------------------------------- derivation


@dataclass(frozen=True)
class Variant:
    """One derived expression, with what produced it and whether it is new."""

    label: str
    rule: str
    expression: str
    canonical: str
    status: str  # observed | tried | novel
    detail: str = ""


def _classify(canonical, observed, tried):
    if canonical in observed:
        return "observed"
    if canonical in tried:
        return "tried"
    return "novel"


def expand_domain(unit, datasets, rules=None, families=None, slot=None, limit=None):
    """Offer fields that could fill a *carrier* slot, beyond the ones already used.

    Only offered, never applied: this is the rule that invents rather than recalls,
    and a wrong expansion is a wrong alpha with a plausible-looking template.

    Three guards, each one bought by looking at what the naive version produced:

    * a slot must have **shown variation** in this unit. `close` appearing once is not
      evidence that `close` is a free choice, and expanding it turned a coherent
      ``(close - low) / (high - low)`` position into a mix of unrelated price fields;
    * all of its values must sit in one family listed under ``domains`` -- a group key
      like ``subindustry`` is never a carrier;
    * by default only the **most-varied** eligible slot is expanded, because that is
      the slot the corpus itself has been moving. Expanding the normaliser slot next
      to it (``atq``/``ltq``/``saleq``) is allowed by passing ``slot`` explicitly, and
      ``limit`` truncates the candidate list.
    """
    rules = rules if rules is not None else load_rules()
    families = families or (rules.get("datasets") or {})
    allowed = set()
    for _name, members in (rules.get("domains") or {}).items():
        if isinstance(members, list):
            allowed |= set(members)

    eligible = collections.OrderedDict()
    for name, values in unit.grid.items():
        if not name.startswith("f") or len(values) < 2:
            continue
        used = {families.get(datasets.get(value)) or datasets.get(value) for value in values}
        if len(used) != 1 or next(iter(used)) not in allowed:
            continue
        eligible[name] = (next(iter(used)), values)

    if slot is not None:
        eligible = collections.OrderedDict(
            (name, spec) for name, spec in eligible.items() if name == slot)
    elif eligible:
        best = max(eligible, key=lambda name: len(eligible[name][1]))
        eligible = collections.OrderedDict([(best, eligible[best])])

    out = collections.OrderedDict()
    for name, (family, values) in eligible.items():
        candidates = sorted(field for field, dataset in datasets.items()
                            if (families.get(dataset) or dataset) == family
                            and field not in values)
        out[name] = candidates[:limit] if limit else candidates
    return out


def to_program(point, mechanism=None):
    """Turn a point back into a *program*: named legs, weights, annotations.

    This is what connects the unit layer to the annotation layer. A unit knows which
    legs it has and what each does; a program knows how to be ablated. Rendering the
    unit as a program with one assignment per leg and an ``@leg`` annotation for each
    means :func:`annotations.derive_ablations` can ask "does this leg earn its place"
    without the unit layer re-implementing ablation -- and there stays exactly one
    definition of what dropping a leg means.

    The signal is emitted as a ``+`` sum with signed coefficients rather than with
    ``-``, because ``annotations.drop_term`` splits sums and cannot see through a
    subtraction. Nothing about what is measured changes; only the spelling.
    """
    names = tuple("l%d" % index for index in range(len(point.legs)))
    assignments = tuple((name, leg.body) for name, leg in zip(names, point.legs))
    notes = []
    if mechanism:
        notes.append(fa.Annotation("mechanism", (),
                                   (("value", fa.String('"%s"' % mechanism)),)))
    for name, leg in zip(names, point.legs):
        notes.append(fa.Annotation("leg", (),
                                   (("name", fa.String('"%s"' % name)),
                                    ("role", fa.String('"%s"' % leg.role)))))

    terms_, leg_index = [], 0
    for weight, body in point.terms:
        if isinstance(body, fa.Number):
            terms_.append((weight, body))
        else:
            terms_.append((weight, fa.Field(names[leg_index])))
            leg_index += 1
    return fa.Program(assignments, _sum(terms_), tuple(notes))


def _sum(terms):
    """Join ``(weight, body)`` pairs as one sum with signed coefficients.

    A constant term is folded to one literal: ``- 0.5`` is the number -0.5, not a
    coefficient times 0.5. Keeping it folded is what lets ``annotations.drop_term``
    see a usable coefficient when it renormalises.
    """
    tree = None
    for weight, body in terms:
        if isinstance(body, fa.Number):
            piece = fa.Number("%.12g" % (weight * float(body.text)))
        elif weight == 1.0:
            piece = body
        else:
            piece = fa.BinOp("*", fa.Number("%.12g" % weight), body)
        tree = piece if tree is None else fa.BinOp("+", tree, piece)
    return tree


def _with_body(point, index, body):
    """Rebuild this point's sum with one leg's body replaced."""
    positions = [position for position, (_w, node) in enumerate(point.terms)
                 if not isinstance(node, fa.Number)]
    if index >= len(positions):
        return None
    target = positions[index]
    terms_ = [(weight, body if position == target else node)
              for position, (weight, node) in enumerate(point.terms)]
    return _tidy(_sum(terms_)).render()


def crossover(unit, corpus, observed=None, tried=None):
    """Templates nobody ran, assembled from legs that each already exist.

    Within one kind every unit has the same assembly and the same data families, so
    the legs are interchangeable *by construction*: taking the carrier leg from one
    construction and the stabiliser from another yields something assembled entirely
    from parts that have been simulated. That is the derivation the parts catalogue
    could not do, because a part is not attached to a position.

    Three guards, and the first one is the important one:

    * the two legs must have the same **role** -- same job, different realisation. A
      ratio leg may not be replaced by a dispersion leg;
    * they must read the same **data families**, or the port silently changes what
      the alpha is about;
    * at least one other leg must be identical, otherwise the "port" is just the
      donor construction and adds nothing.
    """
    observed = observed or set()
    tried = tried or set()
    kind = corpus.kinds.get(unit.kind)
    if kind is None or not unit.points:
        return []
    reference = unit.points[0]
    out, seen = [], set()
    for other in kind.units:
        if other is unit or len(other.legs) != len(unit.legs):
            continue
        shared = sum(1 for left, right in zip(unit.legs, other.legs)
                     if left.shape == right.shape)
        if shared == 0:
            continue
        for index, (mine, theirs) in enumerate(zip(unit.legs, other.legs)):
            if mine.shape == theirs.shape:
                continue
            if mine.role != theirs.role or mine.families != theirs.families:
                continue
            expression = _with_body(reference, index, theirs.body)
            if expression is None:
                continue
            canonical = fa.canonical(expression)
            if canonical in seen or canonical == reference.canonical:
                continue
            seen.add(canonical)
            out.append(Variant(label="port:%d" % index, rule="port",
                               expression=expression, canonical=canonical,
                               status=_classify(canonical, observed, tried),
                               detail="%s <- %s" % (mine.role, other.key[:8])))
    return out


def derive(point, grid=None, observed=None, tried=None, domains=None, mechanism=None):
    """Every variant this point's own structure asks for.

    Three rules, in increasing distance from the evidence:

    ``zero`` / ``drop``   the ablation pair, asked of every leg whether or not anybody
                          annotated it. Generated by :func:`annotations.derive_ablations`
                          on the program rendered by :func:`to_program`, so the
                          definition of "drop a leg" is the one already in the repo
                          (renormalised) and not a second one that drifts.
    ``data``              one slot moves to another value the corpus already used --
                          the unit's grid says which. Still evidence: the value is a
                          field that has filled this exact slot somewhere.
    ``domain``            one slot moves to a field that has never filled it. Only
                          produced when the caller passes ``domains``, because this is
                          the rule that invents rather than recalls.
    """
    observed = observed or set()
    tried = tried or set()
    out = []

    program = to_program(point, mechanism)
    for variant in annotations.derive_ablations(program):
        canonical = fa.canonical(_flatten(variant["expression"]))
        out.append(Variant(label=variant["label"], rule=variant["kind"],
                           expression=variant["expression"], canonical=canonical,
                           status=_classify(canonical, observed, tried),
                           detail=variant["leg"]))

    for name, current in point.filling.items():
        for value in (grid or {}).get(name, ()):
            if value == current:
                continue
            variant = _at(point, name, value)
            if variant is None:
                continue
            canonical = fa.canonical(variant)
            out.append(Variant(label="data:%s" % name, rule="data", expression=variant,
                               canonical=canonical,
                               status=_classify(canonical, observed, tried),
                               detail="%s -> %s" % (current, value)))

    for name, values in (domains or {}).items():
        current = point.filling.get(name)
        for value in values:
            if value == current:
                continue
            variant = _at(point, name, value)
            if variant is None:
                continue
            canonical = fa.canonical(variant)
            out.append(Variant(label="domain:%s" % name, rule="domain", expression=variant,
                               canonical=canonical,
                               status=_classify(canonical, observed, tried),
                               detail="%s -> %s" % (current, value)))
    return out


def _flatten(expression):
    """Inline a program and render it flat, so canonical forms are comparable.

    The unit layer keys on the flat tree: ``x = ...; group_rank(x, g)`` and the same
    thing written inline are one construction, and the ledger's own hash -- which
    keeps the assignment -- would call them two.
    """
    try:
        program = fa.parse(expression)
    except fa.ParseError:
        return expression
    signal = _inline(program)
    return _tidy(signal).render() if signal is not None else expression


def _at(point, name, value):
    """Instantiate this point with one slot moved, or ``None`` if it cannot be."""
    filling = dict(point.filling)
    filling[name] = value
    try:
        return point.render(filling)
    except (KeyError, ValueError):
        return None


def verify(corpus):
    """Refill every template and demand it reproduce the expression it came from.

    Returns ``(checked, failures)``; a failure is ``(expression, expected, got)``.
    This is the gate that keeps "compressed" from meaning "summarised": a unit whose
    template cannot be instantiated back is not a compression of anything.
    """
    failures, checked = [], 0
    for point in corpus.points:
        checked += 1
        try:
            got = fa.canonical(point.render())
        except (KeyError, ValueError) as error:
            failures.append((point.expression, point.canonical, str(error)))
            continue
        if got != point.canonical:
            failures.append((point.expression, point.canonical, got))
    return checked, failures


def summary(corpus):
    """The counts a report needs, computed from the corpus rather than remembered.

    ``one_step`` counts single-slot moves over the whole corpus: for every unit and
    every slot, the number of values that slot has held, minus one. It is the number
    of variants ``derive`` can offer without inventing a value -- deliberately *not*
    the product of the grid, which explodes combinatorially and would say "0.4%
    explored" about a corpus where most cells are different alphas only in name.
    """
    sizes = collections.Counter(unit.size for unit in corpus.units.values())
    kind_sizes = collections.Counter(kind.size for kind in corpus.kinds.values())
    return {
        "expressions": len(corpus.points),
        "units": len(corpus.units),
        "unit_singletons": sum(1 for unit in corpus.units.values() if unit.size == 1),
        "unit_sizes": dict(sorted(sizes.items())),
        "largest_unit": max((unit.size for unit in corpus.units.values()), default=0),
        "kinds": len(corpus.kinds),
        "kind_singletons": sum(1 for kind in corpus.kinds.values() if kind.size == 1),
        "kind_sizes": dict(sorted(kind_sizes.items())),
        "largest_kind": max((kind.size for kind in corpus.kinds.values()), default=0),
        "assemblies": len({point.assembly for point in corpus.points}),
        "one_step": sum(_one_step(unit) for unit in corpus.units.values()),
    }


def _one_step(unit):
    return sum(max(0, len(values) - 1) for values in unit.grid.values())


def render_unit(unit, preview=0):
    """A human-readable template: the slots, the grid, and who is in it."""
    lines = ["unit %s  (%d point%s, %d slots)"
             % (unit.key[:16], unit.size, "" if unit.size == 1 else "s",
                len(unit.grid))]
    lines.append("  assembly : %s" % unit.assembly)
    lines.append("  families : %s" % ", ".join(unit.datasets))
    for index, leg in enumerate(unit.legs):
        lines.append("  leg %-2d   : %-56s  w=%.3g" % (index, leg.role[:56], leg.weight))
    for index, leg in enumerate(unit.legs):
        lines.append("    l%d = %s" % (index, leg.body.render()[:160]))
    for name, values in unit.grid.items():
        shown = ", ".join(values[:6]) + (" ..." if len(values) > 6 else "")
        lines.append("  slot %-4s: %d value%s  %s"
                     % (name, len(values), "" if len(values) == 1 else "s", shown))
    for point in unit.points[:preview]:
        lines.append("  · %s" % point.expression.replace("\n", " ")[:150])
    return "\n".join(lines)