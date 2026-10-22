"""Pre-flight linting for FASTEXPR expressions.

The expensive failure mode this module removes: an expression passes
``simulate --dry-run`` (which only checks payload shape), gets submitted, and
only then does BRAIN reject it -- after the submit quota is spent and after the
polling wait. That is exactly what happened with ``regression_neut``, which has
a documentation page but is *not* one of the 66 operators in ``GET /operators``.

So the lint compares every operator name in the expression against the
authoritative operator list, and additionally against a small learned denylist
of operators the simulator has rejected in this config. The denylist is written
by :func:`record_rejection`, which the simulation loop calls when BRAIN returns
an operator-level error, so the same trap cannot be stepped in twice.

Checks, in the order they run:

``empty``              expression is blank
``unbalanced_parens``  parentheses do not balance
``unbalanced_string``  a quoted string literal is not closed
``unknown_operator``   identifier called as a function is not a real operator
``operator_rejected``  operator is on the learned denylist
``arity_suspicious``   argument count is outside the documented signature
``unknown_identifier`` bare identifier is neither a known field nor a literal
"""

from __future__ import annotations

import json
import os
import re

import expression_types

import research_paths
import cache_store

__all__ = [
    "build_operator_index",
    "lint_all",
    "extract_calls",
    "lint_expression",
    "load_denylist",
    "record_rejection",
    "default_denylist_path",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERALS = frozenset({"true", "false", "nan", "inf", "null", "none"})

# Variable names the simulator refuses to let an expression assign to. The
# catalog documents none of these; the entry below came from a live rejection
# ("Variable eps is reserved and cannot be assigned"), which cost one full
# simulation round because the offline lint had no way to know.
_RESERVED_IDENTIFIERS = frozenset({"eps"})

_ASSIGNMENT_TARGET = re.compile(
    r"(?:^|;)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)"
)


def _assignment_targets(masked):
    """Return assignment targets in order, without duplicates."""
    seen = []
    for match in _ASSIGNMENT_TARGET.finditer(masked or ""):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _mask_strings(expression):
    """Blank out string literals so commas/parens inside them are ignored.

    Character positions are preserved so reported offsets stay meaningful.
    """
    characters = list(expression)
    inside = False
    quote = ""
    for index, char in enumerate(expression):
        if inside:
            if char == quote:
                inside = False
            characters[index] = " "
        elif char in {'"', "'"}:
            inside = True
            quote = char
            characters[index] = " "
    return "".join(characters), inside


def _parse_signature(definition):
    """Extract ``(min_arity, max_arity, signature)`` from an operator definition.

    Definitions look like ``add(x, y, filter = false), x + y`` -- the parameter
    list is the first balanced parenthesised group. Parameters carrying ``=`` are
    optional; a trailing variadic marker means "more arguments allowed".

    The variadic marker is spelled ``..`` in some catalogue records (``max(x, y,
    ..)``) and ``...`` in others, so treat any run of two or more dots as the
    marker rather than only the three-dot form.
    """
    if not isinstance(definition, str) or "(" not in definition:
        return None, None, None
    start = definition.index("(")
    depth = 0
    end = None
    for position in range(start, len(definition)):
        char = definition[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = position
                break
    if end is None:
        return None, None, None
    inner = definition[start + 1:end].strip()
    signature = definition[:end + 1]
    if not inner:
        return 0, 0, signature
    params = [part.strip() for part in inner.split(",")]
    required = 0
    variadic = False
    for param in params:
        if not param:
            continue
        if param in {"...", "..", "*args", "*"} or (set(param) == {"."} and len(param) >= 2):
            variadic = True
            continue
        if "=" in param:
            continue
        required += 1
    maximum = None if variadic else len([p for p in params if p])
    return required, maximum, signature


def build_operator_index(operator_records):
    """Index ``GET /operators`` records by name with parsed arity.

    Returns ``{name: {category, signature, min_arity, max_arity}}``.
    """
    index = {}
    for record in operator_records or []:
        name = record.get("name")
        if not name:
            continue
        min_arity, max_arity, signature = _parse_signature(record.get("definition"))
        index[name] = {
            "category": record.get("category"),
            "signature": signature,
            "definition": record.get("definition"),
            "min_arity": min_arity,
            "max_arity": max_arity,
        }
    return index


def _count_args(text, open_index):
    """Count top-level arguments of the call whose ``(`` is at ``open_index``.

    Returns ``(arg_count, close_index)``; ``close_index`` is ``None`` when the
    group never closes.
    """
    depth = 0
    commas = 0
    has_content = False
    for position in range(open_index, len(text)):
        char = text[position]
        if char == "(":
            depth += 1
            if depth > 1:
                has_content = True
        elif char == ")":
            depth -= 1
            if depth == 0:
                return (commas + 1 if has_content else 0), position
        elif char == "," and depth == 1:
            commas += 1
            has_content = True
        elif not char.isspace() and depth == 1:
            has_content = True
    return (commas + 1 if has_content else 0), None


def _count_kwargs(text, open_index, close_index):
    """Count top-level ``name=`` keyword arguments between the call parentheses.

    The documented signature lists optional parameters too, so a call that
    supplies ``range="0.1,1,0.1"`` legitimately passes fewer positional
    arguments than the signature's parameter count. Counting keywords lets the
    arity check avoid flagging that as a missing argument.
    """
    depth = 0
    count = 0
    for position in range(open_index, close_index + 1):
        char = text[position]
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            continue
        if char != "=" or depth != 1:
            continue
        if position + 1 < len(text) and text[position + 1] in "=~":
            continue
        # Walk back over whitespace to confirm an identifier precedes the '='.
        cursor = position - 1
        while cursor > open_index and text[cursor].isspace():
            cursor -= 1
        end = cursor
        while cursor > open_index and (
            text[cursor].isalnum() or text[cursor] == "_"
        ):
            cursor -= 1
        if end > cursor:
            count += 1
    return count


def extract_calls(expression):
    """Return every function call in the expression as ``{name, arg_count, offset}``.

    Named arguments such as ``filter=true`` or ``range="0.1,1,0.1"`` are not
    calls and are skipped, as are bare literal identifiers.
    """
    masked, _ = _mask_strings(expression)
    calls = []
    for match in _IDENT.finditer(masked):
        name = match.group(0)
        position = match.end()
        while position < len(masked) and masked[position].isspace():
            position += 1
        if position >= len(masked) or masked[position] != "(":
            continue
        arg_count, close_index = _count_args(masked, position)
        kwarg_count = (
            _count_kwargs(masked, position, close_index)
            if close_index is not None
            else 0
        )
        calls.append({
            "name": name,
            "arg_count": arg_count,
            "kwarg_count": kwarg_count,
            "offset": match.start(),
        })
    return calls


def extract_identifiers(expression):
    """Return bare (non-called, non-assigned) identifiers with their offsets."""
    masked, _ = _mask_strings(expression)
    identifiers = []
    for match in _IDENT.finditer(masked):
        name = match.group(0)
        position = match.end()
        while position < len(masked) and masked[position].isspace():
            position += 1
        if position < len(masked) and masked[position] in "(=":
            continue
        identifiers.append({"name": name, "offset": match.start()})
    return identifiers


def default_denylist_path():
    """Denylist location: ``<research_dir>/operator-denylist.json``."""
    return research_paths.resolve("operator-denylist.json")


def load_denylist(path=None):
    """Load the learned denylist as ``{operator: {reason, ...}}``."""
    path = path or default_denylist_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # A corrupt denylist must not block research; degrade to empty.
        return {}
    entries = payload.get("operators") if isinstance(payload, dict) else None
    return entries or {}


def record_rejection(operator, message, context=None, path=None, severity="block"):
    """Persist an operator rejection so future lints flag it immediately.

    Called when BRAIN rejects a submitted expression with an operator-level
    error. Appends a timestamp and the rejection message; repeat rejections bump
    a counter instead of duplicating the entry.

    ``severity`` is ``block`` when the operator itself is unknown or unavailable
    (lint fails), and ``warn`` when the operator is real but rejected this
    particular input kind (lint warns). Defaulting to ``block`` is deliberate:
    silently downgrading an unknown operator is the failure this module exists
    to prevent.
    """
    import datetime

    if severity not in {"block", "warn"}:
        raise ValueError("severity must be 'block' or 'warn', got %r" % (severity,))
    path = path or default_denylist_path()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    # Read-modify-write under a lock, and replace atomically. Two concurrent
    # CLI calls used to share one fixed ".tmp" name, which lost the counter bump
    # (or worse, one writer's os.replace failing after the other moved the file).
    with cache_store.file_lock(path + ".lock", timeout=30):
        entries = load_denylist(path)
        existing = entries.get(operator)
        if existing:
            existing["count"] = int(existing.get("count", 1)) + 1
            existing["last_seen"] = stamp
            existing["message"] = message
            # A hard rejection outranks a soft one; never soften an existing block.
            if severity == "block":
                existing["severity"] = "block"
        else:
            entries[operator] = {
                "reason": "rejected by the BRAIN simulator",
                "message": message,
                "severity": severity,
                "context": context or {},
                "first_seen": stamp,
                "last_seen": stamp,
                "count": 1,
            }
        document = {
            "_comment": "Learned operator denylist. Written by expression_lint."
                        "record_rejection when the simulator rejects an operator.",
            "operators": entries,
        }
        cache_store.write_json_atomic(path, document, mode=0o644, indent=2)
    return entries[operator]


#: The simulator rejects anything above this many operator instances.
OPERATOR_LIMIT = 64
#: Warn before the wall rather than at it.
OPERATOR_WARN = 50

_CALL = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_BINARY = re.compile(r"(?<![<>!=])(?:[<>!=]=|&&|\|\||[+\-*/<>^?])")


def estimate_operator_count(expression):
    """Approximate the total number of operators, as the simulator counts them.

    ``operator_count`` elsewhere in this module is the number of *distinct*
    operators, which says nothing about the platform's limit: two alphas can use
    the same nine operators and one of them is rejected for using them ninety-five
    times. This estimate counts call sites plus binary symbols, and deliberately
    over-counts rather than under-counts, so a warning arrives early instead of a
    failed simulation arriving after the quota is spent.
    """
    text = expression or ""
    calls = [name for name in _CALL.findall(text)
             if name.lower() not in {"if", "else"}]
    binary = _BINARY.findall(text)
    return len(calls) + len(binary)


def operator_budget_findings(expression, limit=OPERATOR_LIMIT, warn=OPERATOR_WARN):
    """Findings for an expression that is likely to exceed the operator limit."""
    estimated = estimate_operator_count(expression)
    if estimated > limit:
        return [{
            "code": "operator_budget",
            "severity": "error",
            "message": "about %d operator instances, above the simulator limit of %d; "
                       "merge or drop legs before submitting" % (estimated, limit),
        }]
    if estimated > warn:
        return [{
            "code": "operator_budget",
            "severity": "warning",
            "message": "about %d operator instances, approaching the limit of %d"
                       % (estimated, limit),
        }]
    return []


#: Operators reported to reject VECTOR ("event-driven") fields outright. A forum
#: report on the platform documents rank() and ts_backfill() failing with
#: "event input" on such a field while ts_sum() works, and this account has hit
#: "Operator multiply does not support event inputs" independently. A third of the
#: field catalogue is VECTOR, so a template batch can fail en masse and silently.
VECTOR_REJECTING = frozenset({
    "rank", "ts_rank", "ts_backfill", "ts_mean", "ts_std_dev", "ts_delta",
    "ts_delay", "ts_zscore", "ts_scale", "ts_decay_linear", "ts_corr",
    "ts_regression", "zscore", "scale", "winsorize", "normalize", "quantile",
    "group_rank", "group_neutralize", "group_zscore", "group_scale",
    "group_mean", "multiply", "divide", "subtract", "add",
})
#: Operators that aggregate a vector instead, which is the documented workaround.
VECTOR_ACCEPTING = frozenset({"ts_sum", "vec_avg", "vec_sum", "ts_count_nans"})


def _call_arguments(expression):
    """Yield ``(operator, argument_text, offset)`` for every call, innermost first."""
    text = expression or ""
    found = []
    for match in _CALL.finditer(text):
        name = match.group(1)
        open_at = match.end() - 1
        depth = 0
        for index in range(open_at, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    found.append((name, text[open_at + 1:index], match.start()))
                    break
    return found


_TRAILING = re.compile(r"\)\s*([0-9A-Za-z_])")


def trailing_token_findings(expression):
    """Flag a token sitting immediately after a closing parenthesis.

    A PDF extraction glued page numbers onto eight of the 101 formulas, producing
    ``(if_else(...)) 9``. Nothing else objected: the parentheses balance, every
    operator is known, and a bare digit is not an unknown identifier, so the
    expression reached the simulator and came back as "Unexpected character '9'".

    The test is deliberately narrow - a closing parenthesis followed by whitespace
    and then a digit or letter - because that position requires an operator. An
    earlier version cut at the first balanced group and therefore flagged the
    perfectly valid ``rank(close) - 0.5``, which its own unit test caught.
    """
    found = []
    for match in _TRAILING.finditer(expression or ""):
        found.append({
            "code": "trailing_token",
            "severity": "error",
            "offset": match.start(),
            "message": ("%r follows a closing parenthesis with no operator between "
                        "them" % match.group(1)),
        })
    return found


def vector_findings(expression, field_types, strict=False):
    """Flag VECTOR fields read directly by an operator that cannot take them.

    The check is on the *innermost* enclosing call. An earlier version scanned the
    whole argument text of every call, so ``rank(vec_avg(vector_field))`` was flagged
    for the outer ``rank`` even though the aggregation in between is exactly the
    documented workaround, and four legitimate expressions were blocked.
    """
    if not field_types:
        return []
    vectors = {name for name, kind in field_types.items()
               if str(kind).upper() == "VECTOR"}
    if not vectors:
        return []
    calls = _call_arguments(expression)
    findings = []
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", expression or ""):
        token = match.group(0)
        if token not in vectors:
            continue
        # Innermost call whose argument span contains this occurrence.
        best = None
        for operator, arguments, offset in calls:
            open_at = expression.find("(", offset)
            if open_at < 0:
                continue
            close_at = open_at + 1 + len(arguments)
            if open_at < match.start() < close_at:
                if best is None or open_at > best[1]:
                    best = (operator, open_at)
        if best is None or best[0] in VECTOR_ACCEPTING:
            continue
        operator = best[0]
        severity = "error" if (strict and operator in VECTOR_REJECTING) else "warning"
        findings.append({
            "code": "vector_field_input",
            "severity": severity,
            "operator": operator,
            "identifier": token,
            "offset": match.start(),
            "message": ("%s reads VECTOR field %s; aggregate first (ts_sum, vec_avg, "
                        "vec_sum), the platform rejects event inputs here"
                        % (operator, token)),
        })
    return findings


_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_COMMENT_HASH = re.compile(r"#[^\n]*")


def strip_comments(expression):
    """Remove FASTEXPR comments before any name or type check.

    The platform accepts `/* ... */` blocks inside an expression - two alphas in the
    ledger carry a written hypothesis that way and were scored normally - but a name scan
    reads the prose as identifiers and reports every capitalised word as an unknown field.
    """
    text = _COMMENT_BLOCK.sub(" ", expression or "")
    text = _COMMENT_LINE.sub(" ", text)
    # `#` starts a line comment too: an alpha in the ledger opens with `# day1` and was
    # scored normally, so the marker is data to the platform and prose to a name scan.
    return _COMMENT_HASH.sub(" ", text)


def lint_expression(expression, operator_index=None, denylist=None,
                    known_fields=None, strict_fields=False, field_types=None):
    """Lint one expression. Returns ``{ok, errors, warnings, ...}``.

    ``operator_index`` comes from :func:`build_operator_index`; when omitted the
    operator-name check is skipped (offline use). ``known_fields`` enables the
    unknown-identifier check.
    ``field_types`` maps field id to catalog type and enables the VECTOR check.
    """
    expression = strip_comments(expression)
    errors = []
    warnings = []
    expression = expression or ""

    if not expression.strip():
        errors.append({"code": "empty", "message": "expression is blank"})
        return {
            "ok": False,
            "operator_count": 0,
            "operators": [],
            "errors": errors,
            "warnings": warnings,
        }

    masked, unterminated = _mask_strings(expression)
    if unterminated:
        errors.append({"code": "unbalanced_string",
                       "message": "a quoted string literal is never closed"})

    # Assignment targets that the simulator reserves for itself. Learned from
    # a live rejection: ``eps=ts_backfill(...)`` fails with "Variable eps is
    # reserved and cannot be assigned". Nothing in the catalog says so, so the
    # list has to be carried here.
    for name in _assignment_targets(masked):
        if name.lower() in _RESERVED_IDENTIFIERS:
            errors.append({
                "code": "reserved_identifier",
                "identifier": name,
                "message": "%r is reserved by the simulator; rename the variable"
                           % name,
            })

    depth = 0
    for char in masked:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        errors.append({"code": "unbalanced_parens",
                       "message": "parentheses do not balance (net %+d)" % depth})

    calls = extract_calls(expression)
    used = []
    for call in calls:
        name = call["name"]
        used.append(name)
        entry = {"name": name, "arg_count": call["arg_count"],
                 "offset": call["offset"]}
        if denylist and name in denylist:
            record = denylist[name] or {}
            reason = record.get("message") or record.get("reason") or \
                "previously rejected"
            severity = record.get("severity", "block")
            item = {
                "code": "operator_rejected",
                "operator": name,
                "offset": call["offset"],
                "message": "operator %r was rejected by the simulator before: %s"
                           % (name, reason),
            }
            if severity == "block":
                errors.append(item)
                entry["status"] = "denylisted"
            else:
                warnings.append(item)
                entry["status"] = "denylisted_soft"
        elif operator_index is not None:
            meta = operator_index.get(name)
            if meta is None:
                errors.append({
                    "code": "unknown_operator",
                    "operator": name,
                    "offset": call["offset"],
                    "message": "operator %r is not in the BRAIN operator list"
                               % name,
                })
                entry["status"] = "unknown"
            else:
                entry["status"] = "known"
                entry["category"] = meta.get("category")
                minimum = meta.get("min_arity")
                maximum = meta.get("max_arity")
                count = call["arg_count"]
                # The documented signature includes optional parameters, so a
                # call using keyword arguments (e.g. bucket(x, range=...)) has
                # fewer positional arguments than the parameter count without
                # being wrong. Only enforce the lower bound when no keywords
                # are present.
                if (
                    minimum is not None
                    and count < minimum
                    and not call.get("kwarg_count")
                ):
                    warnings.append({
                        "code": "arity_suspicious",
                        "operator": name,
                        "message": "%s expects at least %d argument(s) per "
                                   "`%s`, got %d"
                                   % (name, minimum, meta.get("signature"), count),
                    })
                if maximum is not None and count > maximum:
                    warnings.append({
                        "code": "arity_suspicious",
                        "operator": name,
                        "message": "%s expects at most %d argument(s) per `%s`, "
                                   "got %d"
                                   % (name, maximum, meta.get("signature"), count),
                    })
        else:
            entry["status"] = "unchecked"
        # Nested calls are visited by the same scan, so no recursion needed.

    if known_fields is not None:
        known = set(known_fields)
        # Variables assigned inside the expression are not datafields. Without
        # this, any multi-line expression that names its intermediate results --
        # which is the readable way to write a composite -- reports every local
        # name as an unknown field.
        masked = _mask_strings(expression)
        locals_ = set(_assignment_targets(masked[0] if isinstance(masked, tuple) else masked))
        seen = set()
        for identifier in extract_identifiers(expression):
            name = identifier["name"]
            if name in seen or name.lower() in _LITERALS or name in locals_:
                continue
            seen.add(name)
            if name in known:
                continue
            # Only flag things that actually look like datafield ids; a bare
            # short word is more likely a keyword we do not model.
            if "_" not in name and len(name) < 6:
                continue
            item = {
                "code": "unknown_identifier",
                "identifier": name,
                "offset": identifier["offset"],
                "message": "%r is not a known datafield in this config" % name,
            }
            if strict_fields:
                errors.append(item)
            else:
                warnings.append(item)

    # The simulator's limit is on operator instances, not distinct operators, so a
    # composite that reuses a handful of functions many times can pass every other
    # check and still be rejected ("Expression with 78 operators exceeds limit of
    # 64"). Catching it here costs nothing; catching it at the simulator costs a
    # round of quota.
    for finding in operator_budget_findings(expression):
        if finding["severity"] == "error":
            errors.append(finding)
        else:
            warnings.append(finding)

    for finding in trailing_token_findings(expression):
        errors.append(finding)

    # Types first: the signature table derives which operators accept which field type,
    # including the VECTOR rule that used to be a hand-built denylist. The denylist is
    # kept only for operators the signature table does not cover, so the same problem is
    # never reported twice.
    for finding in expression_types.type_findings(expression, field_types,
                                                  strict=strict_fields):
        if finding["severity"] == "error":
            errors.append(finding)
        else:
            warnings.append(finding)

    unsigned = [item for item in vector_findings(expression, field_types,
                                                 strict=strict_fields)
                if item.get("operator") not in expression_types.SIGNATURES]
    for finding in unsigned:
        if finding["severity"] == "error":
            errors.append(finding)
        else:
            warnings.append(finding)

    return {
        "ok": not errors,
        "operator_count": len(used),
        "operator_estimate": estimate_operator_count(expression),
        "operators": used,
        "calls": [call for call in calls],
        "details": None,
        "errors": errors,
        "warnings": warnings,
    }

def lint_all(pairs, operator_index=None, denylist=None, known_fields=None,
             strict_fields=False, field_types=None):
    """Lint a batch of ``(expression, decay)`` pairs.

    Returns ``{ok, results, errors, warnings}`` where ``errors`` and
    ``warnings`` are flattened with the offending expression attached, which is
    the shape a CLI wants to report. Per-expression reports stay in ``results``.
    """
    results = []
    errors = []
    warnings = []
    for pair in pairs or []:
        expression = pair[0] if isinstance(pair, (tuple, list)) else pair
        report = lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=strict_fields,
            field_types=field_types,
        )
        report["expression"] = expression
        results.append(report)
        for item in report["errors"]:
            errors.append(dict(item, expression=expression))
        for item in report["warnings"]:
            warnings.append(dict(item, expression=expression))
    return {
        "ok": not errors,
        "expressions": len(results),
        "results": results,
        "errors": errors,
        "warnings": warnings,
    }
