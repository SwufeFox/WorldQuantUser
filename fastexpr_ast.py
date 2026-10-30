#!/usr/bin/env python3
"""A real parser for FASTEXPR, plus a macro layer that compiles sugar into it.

Why a parser instead of regexes: every analysis we want downstream needs the tree,
not the string. Field extraction by regex cannot tell an assignment target from a
call argument, cannot see that ``eps`` is a local variable, and cannot tell that
two expressions differ only in a constant. All three have already caused wrong
conclusions here.

What the tree buys, in the order it matters:

* **Exact field sets.** Locals bound by ``a = ...`` are not fields. The regex path
  reports them as unknown identifiers unless a field happens to share the name.
* **Skeletons.** Replace every field with ``F`` and every literal with ``L`` and
  you get a structure key. Two alphas with the same key are the same bet with
  different inputs -- that is the family tree question, answered structurally
  instead of by correlation, which costs an API call per pair.
* **Composition.** Because a signal is a tree, assembling legs is tree surgery:
  scale a subtree, substitute one field family for another, or splice a whole
  program in as a subexpression.

Macros are the "post-compile" layer: you write ``band(r)`` and the parser expands
it to ``-(abs(r) - 0.25)`` before anything else sees it. Expansion happens on the
AST, so a macro can take a subtree, not just text. Nothing in this module knows
any BRAIN field or operator name: the macro table starts empty and is filled by
whoever imports :func:`macro`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Iterable, Sequence

__all__ = [
    "ParseError",
    "Node",
    "Number",
    "String",
    "Field",
    "Call",
    "BinOp",
    "UnaryOp",
    "Ternary",
    "Program",
    "parse",
    "parse_program",
    "macro",
    "macros",
    "expand",
    "skeleton",
    "field_names",
    "operator_names",
    "substitute",
    "map_fields",
]


class ParseError(ValueError):
    """Raised for input this grammar cannot read, with the offending offset."""

    def __init__(self, message, expression, offset):
        window = expression[max(0, offset - 24): offset + 24]
        caret = " " * min(offset, 24) + "^"
        super().__init__("%s at offset %d\n  %s\n  %s" % (message, offset, window, caret))
        self.offset = offset


# --------------------------------------------------------------------------- tree


@dataclass(frozen=True)
class Node:
    """Base class. Every node renders back to FASTEXPR with :meth:`render`."""

    def render(self, parent_precedence: int = 0) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def walk(self):
        yield self


@dataclass(frozen=True)
class Number(Node):
    text: str

    def render(self, parent_precedence: int = 0) -> str:
        return self.text


@dataclass(frozen=True)
class String(Node):
    text: str  # includes the surrounding quotes as written

    def render(self, parent_precedence: int = 0) -> str:
        return self.text


@dataclass(frozen=True)
class Field(Node):
    name: str

    def render(self, parent_precedence: int = 0) -> str:
        return self.name


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: tuple = ()
    kwargs: tuple = ()  # ((key, Node), ...) so the node stays hashable

    def render(self, parent_precedence: int = 0) -> str:
        parts = [arg.render() for arg in self.args]
        parts += ["%s=%s" % (key, value.render()) for key, value in self.kwargs]
        return "%s(%s)" % (self.name, ", ".join(parts))

    def walk(self):
        yield self
        for child in self.args:
            yield from child.walk()
        for _key, value in self.kwargs:
            yield from value.walk()


_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5,
    "^": 7,
}


@dataclass(frozen=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node

    def render(self, parent_precedence: int = 0) -> str:
        precedence = _PRECEDENCE[self.op]
        text = "%s %s %s" % (
            self.left.render(precedence),
            self.op,
            self.right.render(precedence + (1 if self.op in {"-", "/"} else 0)),
        )
        return "(%s)" % text if precedence < parent_precedence else text

    def walk(self):
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str
    operand: Node

    def render(self, parent_precedence: int = 0) -> str:
        inner = self.operand.render(6)
        text = "%s%s" % (self.op, inner)
        return "(%s)" % text if 6 < parent_precedence else text

    def walk(self):
        yield self
        yield from self.operand.walk()


@dataclass(frozen=True)
class Ternary(Node):
    condition: Node
    when_true: Node
    when_false: Node

    def render(self, parent_precedence: int = 0) -> str:
        return "%s ? %s : %s" % (
            self.condition.render(1), self.when_true.render(), self.when_false.render()
        )

    def walk(self):
        yield self
        yield from self.condition.walk()
        yield from self.when_true.walk()
        yield from self.when_false.walk()


@dataclass(frozen=True)
class Program(Node):
    """``assignments`` precede ``signal``; ``signal`` may be None if it is missing."""

    assignments: tuple = ()  # ((name, Node), ...)
    signal: object = None

    def render(self, parent_precedence: int = 0) -> str:
        chunks = ["%s = %s" % (name, value.render()) for name, value in self.assignments]
        if self.signal is not None:
            chunks.append(self.signal.render())
        return ";\n".join(chunks)

    def walk(self):
        yield self
        for _name, value in self.assignments:
            yield from value.walk()
        if self.signal is not None:
            yield from self.signal.walk()


# ------------------------------------------------------------------------ scanner

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
  | (?P<string>"[^"]*"|'[^']*')
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|==|!=|&&|\|\||[-+*/^<>!(),;=?:])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int


def _tokenize(expression: str):
    tokens, position = [], 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if not match:
            raise ParseError("unexpected character %r" % expression[position],
                             expression, position)
        kind = match.lastgroup
        if kind != "space":
            tokens.append(_Token(kind, match.group(), position))
        position = match.end()
    tokens.append(_Token("eof", "", len(expression)))
    return tokens


_COMMENT_MARKERS = ("//", "#", "/*")


def _strip_comments(expression: str) -> str:
    """Blank out comments while preserving offsets.

    The registry contains expressions carrying a leading ``//`` note, a ``/* */``
    hypothesis block, or a ``# day1`` header -- all real, all unparseable without
    this. Replacement with spaces (not deletion) keeps every offset in error
    messages pointing at the original text.
    """
    out = list(expression)
    index, quote, length = 0, None, len(expression)
    while index < length:
        char = expression[index]
        if quote:
            if char == quote and expression[index - 1] != "\\":
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            continue
        if expression.startswith("//", index) or char == "#":
            end = expression.find("\n", index)
            end = length if end < 0 else end
            for position in range(index, end):
                out[position] = " "
            index = end
            continue
        if expression.startswith("/*", index):
            end = expression.find("*/", index + 2)
            end = length if end < 0 else end + 2
            for position in range(index, end):
                if out[position] != "\n":
                    out[position] = " "
            index = end
            continue
        index += 1
    return "".join(out)


class _Parser:
    def __init__(self, expression: str):
        self.original = expression
        self.expression = _strip_comments(expression)
        self.tokens = _tokenize(self.expression)
        self.index = 0

    # -- token helpers ----------------------------------------------------
    @property
    def token(self):
        return self.tokens[self.index]

    def advance(self):
        token = self.tokens[self.index]
        self.index += 1
        return token

    def accept(self, text):
        if self.token.text == text:
            return self.advance()
        return None

    def expect(self, text):
        if self.token.text != text:
            raise ParseError("expected %r but found %r" % (text, self.token.text or "end"),
                             self.expression, self.token.start)
        return self.advance()

    # -- grammar ----------------------------------------------------------
    def program(self):
        assignments, signal = [], None
        while self.token.kind != "eof":
            if self.token.text == ";":
                self.advance()
                continue
            if (self.token.kind == "ident"
                    and self.tokens[self.index + 1].text == "="):
                name = self.advance().text
                self.expect("=")
                assignments.append((name, self.expression_node()))
            else:
                signal = self.expression_node()
            if self.token.text == ";":
                self.advance()
                continue
            if self.token.kind != "eof":
                raise ParseError("unexpected %r after a complete statement"
                                 % self.token.text, self.expression, self.token.start)
            break
        if not assignments and signal is None:
            raise ParseError("empty expression", self.expression, 0)
        return Program(tuple(assignments), signal)

    def expression_node(self):
        return self.ternary()

    def ternary(self):
        condition = self.logical_or()
        if self.accept("?"):
            when_true = self.expression_node()
            self.expect(":")
            when_false = self.expression_node()
            return Ternary(condition, when_true, when_false)
        return condition

    def _binary(self, operators, next_level):
        left = next_level()
        while self.token.text in operators:
            op = self.advance().text
            left = BinOp(op, left, next_level())
        return left

    def logical_or(self):
        return self._binary({"||"}, self.logical_and)

    def logical_and(self):
        return self._binary({"&&"}, self.comparison)

    def comparison(self):
        return self._binary({"==", "!=", "<", ">", "<=", ">="}, self.additive)

    def additive(self):
        return self._binary({"+", "-"}, self.multiplicative)

    def multiplicative(self):
        return self._binary({"*", "/"}, self.unary)

    def unary(self):
        if self.token.text in {"-", "!"}:
            op = self.advance().text
            return UnaryOp(op, self.unary())
        return self.power()

    def power(self):
        base = self.atom()
        if self.accept("^"):
            return BinOp("^", base, self.unary())
        return base

    def atom(self):
        token = self.token
        if token.kind == "number":
            self.advance()
            return Number(token.text)
        if token.kind == "string":
            self.advance()
            return String(token.text)
        if token.text == "(":
            self.advance()
            inner = self.expression_node()
            self.expect(")")
            return inner
        if token.kind == "ident":
            self.advance()
            if self.accept("("):
                args, kwargs = [], []
                if self.token.text != ")":
                    while True:
                        if (self.token.kind == "ident"
                                and self.tokens[self.index + 1].text == "="):
                            key = self.advance().text
                            self.expect("=")
                            kwargs.append((key, self.expression_node()))
                        else:
                            args.append(self.expression_node())
                        if not self.accept(","):
                            break
                self.expect(")")
                node = Call(token.text, tuple(args), tuple(kwargs))
                return _expand(node, self.expression)
            return Field(token.text)
        raise ParseError("unexpected %r" % (token.text or "end of input"),
                         self.expression, token.start)


# ------------------------------------------------------------------------ macros

_MACROS: dict = {}


def macro(name, replace=None):
    """Register a sugar function that compiles to FASTEXPR at parse time.

    The builder receives ``(args, kwargs)`` as AST nodes and must return an AST
    node. ``replace=True`` keeps the sugar name out of the final expression
    entirely (pure expansion); otherwise the builder decides what to emit.
    """
    def register(builder):
        _MACROS[name] = builder
        return builder
    return register


def macros():
    return dict(_MACROS)


def _expand(node: Call, expression: str, depth: int = 0) -> Node:
    """Expand a call if it is sugar, then expand whatever the sugar produced."""
    builder = _MACROS.get(node.name)
    if builder is None:
        return node
    if depth > 24:
        raise ParseError("macro expansion for %r is too deep" % node.name,
                         expression, 0)
    produced = builder(node.args, dict(node.kwargs))
    for child in produced.walk():
        if isinstance(child, Call) and child.name in _MACROS:
            return _expand(produced, expression, depth + 1)
    return produced


# --------------------------------------------------------------------- analyses


def parse(expression: str) -> Program:
    """Parse a FASTEXPR string into a :class:`Program`."""
    return _Parser(expression).program()


parse_program = parse


def _bound_names(program: Program):
    return {name for name, _value in program.assignments}


def field_names(program: Program):
    """Field identifiers used by the signal, excluding locals bound in-program."""
    locals_ = _bound_names(program)
    seen, ordered = set(), []
    for node in program.walk():
        if isinstance(node, Field) and node.name not in locals_:
            if node.name not in seen:
                seen.add(node.name)
                ordered.append(node.name)
    return ordered


def operator_names(program: Program):
    ordered, seen = [], set()
    for node in program.walk():
        if isinstance(node, Call) and node.name not in _MACROS:
            if node.name not in seen:
                seen.add(node.name)
                ordered.append(node.name)
    return ordered


def skeleton(program: Program, keep_literals: bool = False) -> str:
    """Structure key: fields become ``F``, literals ``L`` unless kept.

    Two alphas with the same skeleton differ only in *which* fields they read and
    which constants they use -- i.e. the same bet expressed with different inputs.
    This is the cheap structural stand-in for a correlation matrix.
    """
    def walk(node) -> str:
        if isinstance(node, Number):
            return node.text if keep_literals else "L"
        if isinstance(node, String):
            return node.text if keep_literals else "L"
        if isinstance(node, Field):
            return "F"
        if isinstance(node, Call):
            parts = [walk(arg) for arg in node.args]
            parts += ["%s=%s" % (key, walk(value)) for key, value in node.kwargs]
            return "%s(%s)" % (node.name, ",".join(parts))
        if isinstance(node, BinOp):
            return "(%s%s%s)" % (walk(node.left), node.op, walk(node.right))
        if isinstance(node, UnaryOp):
            return "(%s%s)" % (node.op, walk(node.operand))
        if isinstance(node, Ternary):
            return "(%s?%s:%s)" % (walk(node.condition), walk(node.when_true),
                                   walk(node.when_false))
        if isinstance(node, Program):
            heads = [walk(value) for _name, value in node.assignments]
            heads.append(walk(node.signal) if node.signal is not None else "-")
            return "|".join(heads)
        return "?"

    return walk(program)


# ---------------------------------------------------------------- transforms


def substitute(program: Program, name: str, replacement: Node) -> Program:
    """Inline a local variable, so a sugar-built leg can be grafted anywhere."""
    def replace(node):
        if isinstance(node, Field) and node.name == name:
            return replacement
        if isinstance(node, Call):
            return Call(node.name,
                        tuple(replace(arg) for arg in node.args),
                        tuple((key, replace(value)) for key, value in node.kwargs))
        if isinstance(node, BinOp):
            return BinOp(node.op, replace(node.left), replace(node.right))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op, replace(node.operand))
        if isinstance(node, Ternary):
            return Ternary(replace(node.condition), replace(node.when_true),
                           replace(node.when_false))
        return node

    assignments = tuple((key, replace(value))
                        for key, value in program.assignments if key != name)
    signal = replace(program.signal) if program.signal is not None else None
    return Program(assignments, signal)


def map_fields(program: Program, mapping: dict) -> Program:
    """Rename fields, which is how a leg moves between data families."""
    def replace(node):
        if isinstance(node, Field) and node.name in mapping:
            return Field(mapping[node.name])
        if isinstance(node, Call):
            return Call(node.name,
                        tuple(replace(arg) for arg in node.args),
                        tuple((key, replace(value)) for key, value in node.kwargs))
        if isinstance(node, BinOp):
            return BinOp(node.op, replace(node.left), replace(node.right))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op, replace(node.operand))
        if isinstance(node, Ternary):
            return Ternary(replace(node.condition), replace(node.when_true),
                           replace(node.when_false))
        return node

    return Program(tuple((key, replace(value)) for key, value in program.assignments),
                   replace(program.signal) if program.signal is not None else None)


def expand(expression: str) -> str:
    """Parse and render, i.e. run the sugar and return plain FASTEXPR."""
    return parse(expression).render()