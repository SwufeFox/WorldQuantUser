"""Structural identity shared by the ledger projection and research commands."""

from __future__ import annotations

import re

import fastexpr_ast
import machine_lib as ml

#: Which information the expression reads, in the order that decides ties: the first rule
#: that matches wins, so the list runs from most specific to most generic.
#:
#: Token needles are matched only against identifiers that are **not** followed by
#: `(`, so a field fragment cannot accidentally classify an operator such as
#: `ts_std_dev`. Text needles match the raw expression and are used for shapes
#: (`open / ts_delay(close`), not for names.
BLOCK_RULES = (
    ("derivative_pricing", ("implied_volatility", "option", "straddle", "pcr_"), ()),
    ("analyst_dispersion", ("eps_high", "eps_low", "sales_high", "sales_low", "dts_spe"),
     ()),
    ("analyst_yield", ("eps_mean", "median_eps", "afv4_eps", "bvps", "cfps", "estimate"),
     ()),
    ("analyst_coverage", ("_number", "count"), ()),
    ("turnover_structure", ("sharesout", "volume"), ()),
    ("volatility_structure", ("historical_volatility",),
     ("ts_corr(abs(returns)", "ts_std_dev(ts_sum(returns", "ts_std_dev(returns",
      "ts_std_dev(if_else(returns")),
    ("price_path", ("ts_arg_max", "ts_arg_min", "close", "returns"),
     ("open / ts_delay(close", "close / open", "high / low")),
    ("debt_structure", ("debt_st", "debt_lt", "liabilities"), ()),
    ("value_quality", ("equity", "cap", "atq", "assets", "revenue", "cogs", "saleq",
                       "ebitda", "sales_growth"), ()),
    ("news_sentiment", ("news_", "scl12_", "sent"), ()),
)

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _field_tokens(text):
    """Identifiers that are used as data, not as operators: a field is never `name(`."""
    tokens = []
    for match in _TOKEN.finditer(text):
        rest = text[match.end():].lstrip()
        if rest.startswith("("):
            continue
        tokens.append(match.group(0))
    return tokens


def block_of(expression, language=None):
    """The information block an expression reads (``other`` when nothing matches)."""
    if ml.is_python_language(language):
        return "other"
    text = " ".join((expression or "").split())
    if not text:
        return "other"
    tokens = _field_tokens(text)
    for name, needles, text_needles in BLOCK_RULES:
        if any(needle in token for token in tokens for needle in needles):
            return name
        if any(needle in text for needle in text_needles):
            return name
    return "other"


def skeleton_of(expression, language=None):
    """AST skeleton with fields and literals blanked, or ``None`` when unparseable."""
    if ml.is_python_language(language):
        return None
    text = " ".join((expression or "").split())
    if not text:
        return None
    try:
        return fastexpr_ast.skeleton(fastexpr_ast.parse(text))
    except Exception:  # noqa: BLE001 - an unparseable row must not break the census
        return None


def mechanism_key(expression, language=None):
    """``(block, skeleton)``: the unit the campaign counts as one bet."""
    skeleton = skeleton_of(expression, language)
    if skeleton is None:
        return None
    return (block_of(expression), skeleton[:200])
