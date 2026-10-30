#!/usr/bin/env python3
"""Macro expansion on the tree instead of on the text.

The text expander in ``expression_macros.py`` substitutes strings and is right for
the bodies it was written against, because those bodies put their placeholders in
atomic positions -- ``ts_backfill({f}, 252)``, ``rank({fast})``. The moment a
placeholder sits next to an operator, text stops working:

    body:  {fast} * 2
    arg:   a - b
    text:  a - b * 2          # wrong, silently
    tree:  (a - b) * 2

Nothing in the macro file records which of those two situations a given body is in,
so the safety of a template depends on the caller's discipline rather than on the
tool. This module removes the dependence: bodies are parsed once with placeholder
sentinels in place, arguments are parsed as expressions, and substitution is a
field-node replacement. Parentheses then appear where the tree says they are needed.

Placeholder syntax stays ``{name}`` so existing macro files keep working: the
braces are rewritten to ``__ph_name__`` before parsing, which is a legal
identifier, and that identifier is what gets replaced.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastexpr_ast as fa  # noqa: E402

__all__ = ["parse_macros", "prepare", "expand", "expand_ast_node"]

_SENTINEL = "__ph_%s__"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AstMacroError(ValueError):
    """A body that cannot be parsed with its placeholders in place."""


def parse_macros(text):
    """Same file format as the text expander: ``def name(params):`` then a body."""
    from tools.expression_macros import parse_macros as parse_text

    macros = parse_text(text)
    for name, spec in macros.items():
        body = _PLACEHOLDER.sub(lambda m: _SENTINEL % m.group(1), spec["body"])
        try:
            spec["ast"] = fa.parse(body)
        except fa.ParseError as exc:
            raise AstMacroError("macro %r body does not parse: %s" % (name, exc)) from exc
    return macros


def prepare(macros):
    """Fill in the parsed body for macros that came from the text parser.

    Two entry points exist because the text expander still has to work; rather than
    parse the file twice, this upgrades a text-parsed macro map in place, so a
    caller can start with ``parse_macros`` and switch backends per call.
    """
    for spec in macros.values():
        if "ast" in spec:
            continue
        body = _PLACEHOLDER.sub(lambda m: _SENTINEL % m.group(1), spec["body"])
        try:
            spec["ast"] = fa.parse(body)
        except fa.ParseError as exc:
            raise AstMacroError("macro %r body does not parse: %s"
                                % (spec.get("name"), exc)) from exc
    return macros


def _bind(spec, args):
    params = spec["params"]
    if len(args) > len(params):
        raise AstMacroError("macro %r takes %d arguments, %d given"
                            % (spec["name"], len(params), len(args)))
    bound = {}
    for index, (param, default) in enumerate(params):
        if index < len(args):
            bound[param] = args[index]
        elif default is not None:
            bound[param] = default
        else:
            raise AstMacroError("macro %r is missing argument %r" % (spec["name"], param))
    return bound


def _replace_field(node, name, replacement):
    if isinstance(node, fa.Field) and node.name == name:
        return replacement
    if isinstance(node, fa.Call):
        return fa.Call(node.name,
                       tuple(_replace_field(a, name, replacement) for a in node.args),
                       tuple((k, _replace_field(v, name, replacement)) for k, v in node.kwargs))
    if isinstance(node, fa.BinOp):
        return fa.BinOp(node.op, _replace_field(node.left, name, replacement),
                        _replace_field(node.right, name, replacement))
    if isinstance(node, fa.UnaryOp):
        return fa.UnaryOp(node.op, _replace_field(node.operand, name, replacement))
    if isinstance(node, fa.Ternary):
        return fa.Ternary(_replace_field(node.condition, name, replacement),
                          _replace_field(node.when_true, name, replacement),
                          _replace_field(node.when_false, name, replacement))
    return node


def _expand_calls(node, macros, stack):
    """Expand every call whose name is a macro, bottom-up."""
    if isinstance(node, fa.Call):
        args = tuple(_expand_calls(a, macros, stack) for a in node.args)
        kwargs = tuple((k, _expand_calls(v, macros, stack)) for k, v in node.kwargs)
        if node.name in macros:
            rendered = [a.render() for a in args]
            rendered += ["%s=%s" % (k, v.render()) for k, v in kwargs]
            text = expand(macros, node.name, rendered, stack)
            return fa.parse(text).signal or fa.Number("0")
        return fa.Call(node.name, args, kwargs)
    if isinstance(node, fa.BinOp):
        return fa.BinOp(node.op, _expand_calls(node.left, macros, stack),
                        _expand_calls(node.right, macros, stack))
    if isinstance(node, fa.UnaryOp):
        return fa.UnaryOp(node.op, _expand_calls(node.operand, macros, stack))
    if isinstance(node, fa.Ternary):
        return fa.Ternary(_expand_calls(node.condition, macros, stack),
                          _expand_calls(node.when_true, macros, stack),
                          _expand_calls(node.when_false, macros, stack))
    return node


def expand_ast_node(macros, name, args=(), _stack=()):
    """Expand one macro call into a :class:`fa.Program` (assignments allowed)."""
    if name in _stack:
        raise AstMacroError("macro cycle: %s" % " -> ".join(_stack + (name,)))
    if name not in macros:
        raise AstMacroError("unknown macro %r" % name)
    spec = macros[name]
    bound = _bind(spec, [fa.parse(arg).render() for arg in args])

    program = spec["ast"]
    # A parameter may name an intermediate variable, i.e. sit on the left of an
    # assignment (``{v} = ts_backfill(...)``). Renaming the key is as necessary as
    # substituting the body, otherwise the signal references a variable that no
    # longer exists under that name.
    key_positions = {key for key, _value in program.assignments}
    key_rename = {}
    for param, text in bound.items():
        sentinel = _SENTINEL % param
        bound_text = text.strip()
        if sentinel not in key_positions:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bound_text):
            raise AstMacroError(
                "macro %r uses parameter %r as a variable name, but was given %r"
                % (name, param, bound_text))
        key_rename[sentinel] = bound_text
    built = []
    for key, value in program.assignments:
        for param, text in bound.items():
            value = _replace_field(value, _SENTINEL % param, fa.parse(text).signal)
        built.append((key_rename.get(key, key), _expand_calls(value, macros, _stack + (name,))))
    signal = program.signal
    if signal is not None:
        for param, text in bound.items():
            signal = _replace_field(signal, _SENTINEL % param, fa.parse(text).signal)
        signal = _expand_calls(signal, macros, _stack + (name,))
    return fa.Program(tuple(built), signal)


def expand(macros, name, args=None, _stack=()):
    """Return the expanded expression as a string, exactly like the text expander."""
    return expand_ast_node(macros, name, args or [], _stack).render()