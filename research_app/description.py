"""Description compiler: assemble an official Power Pool description, then check it.

The economic hypothesis is the researcher's; this module only assembles and
verifies.  It refuses to invent a rationale for a field or an operator it has no
description for, because a fabricated sentence is worse than a missing one: the
platform re-reads these fields during review.

Platform requirements encoded here (Power Pool description):
at least 100 characters after trimming; the exact three headings from the
official ``getting-started-power-pool-alphas`` template, in order, each followed
by a Markdown bullet list; distinct from the account's other descriptions.  A
description is not a substitute for a measured check, and passing this module
does not mean the alpha is submittable.
"""

from __future__ import annotations

import difflib
import re

MIN_CHARS = 100
SECTION_IDEA = "Idea:"
SECTION_DATA = "Rationale for data used:"
SECTION_OPERATORS = "Rationale for operators used:"
SECTION_TITLES = (SECTION_IDEA, SECTION_DATA, SECTION_OPERATORS)
_HEADING_RE = re.compile(r"^(Idea:|Rationale for data used:|Rationale for operators used:)\s*$")


def _one_line(text):
    return " ".join(str(text or "").split())


def _section(title, items):
    return "%s\n%s" % (title, "\n".join("- %s" % _one_line(item) for item in items))


def _sections(text):
    """Return exact-template section payloads and structural errors."""
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = body.splitlines()
    found = []
    errors = []
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        title = match.group(1)
        if title in [item[0] for item in found]:
            errors.append('duplicate official section "%s"' % title)
        found.append((title, index))

    titles = [title for title, _index in found]
    if found and any(line.strip() for line in lines[:found[0][1]]):
        errors.append('description must start with official section "%s"' % SECTION_IDEA)
    for title in SECTION_TITLES:
        if title not in titles:
            errors.append('missing official section "%s"' % title)
    if not errors and titles != list(SECTION_TITLES):
        errors.append("official sections must appear in order: " + ", ".join(SECTION_TITLES))

    sections = {}
    for number, (title, start) in enumerate(found):
        end = found[number + 1][1] if number + 1 < len(found) else len(lines)
        content = [line.strip() for line in lines[start + 1:end] if line.strip()]
        bullets = [line[2:].strip() for line in content if line.startswith("- ")]
        non_bullets = [line for line in content if not line.startswith("- ")]
        if non_bullets:
            errors.append('section "%s" must contain only "-" bullet items' % title)
        if not bullets:
            errors.append('section "%s" must be followed by a "-" bullet list' % title)
        sections[title] = bullets
    return sections, errors


def compose(idea, fields, operators=(), field_notes=None, operator_notes=None, *,
            rationale=None, operator_rationale=None):
    """Build the text from the researcher's idea plus catalog descriptions.

    ``fields`` are data-field ids; ``field_notes`` maps each to its catalog
    description.  ``operators`` are operator names; ``operator_notes`` maps each
    to what it does in this signal.  ``rationale`` overrides the data section and
    ``operator_rationale`` overrides the operator section.  Any missing input is
    reported under ``missing`` and no draft is produced.
    """
    field_list = [f for f in (fields or ()) if isinstance(f, str) and f.strip()]
    op_list = [o for o in (operators or ()) if isinstance(o, str) and o.strip()]
    field_notes = field_notes or {}
    operator_notes = operator_notes or {}
    missing = []
    text_idea = (idea or "").strip()
    if not text_idea:
        missing.append("idea")
    if not field_list:
        missing.append("fields")
    missing.extend(f for f in field_list if not (field_notes.get(f) or "").strip())
    missing.extend("op:%s" % o for o in op_list if not (operator_notes.get(o) or "").strip())
    if missing:
        return {"ok": False, "text": None,
                "missing": missing,
                "reason": "no description evidence for %s" % ", ".join(missing)}

    if rationale is not None and str(rationale).strip():
        data_items = [str(rationale)]
    else:
        data_items = [
            "%s: %s." % (name, field_notes[name].strip().rstrip("."))
            for name in field_list
        ]
    if operator_rationale is not None and str(operator_rationale).strip():
        operator_items = [str(operator_rationale)]
    elif op_list:
        operator_items = [
            "%s: %s." % (name, operator_notes[name].strip().rstrip("."))
            for name in op_list
        ]
    else:
        operator_items = [
            "No operator-specific transformation is claimed; the documented inputs "
            "are used directly."
        ]
    text = "\n\n".join((
        _section(SECTION_IDEA, [text_idea.replace("\n", " ")]),
        _section(SECTION_DATA, data_items),
        _section(SECTION_OPERATORS, operator_items),
    ))
    checked = check(text)
    return {"ok": checked["ok"], "text": text, "check": checked}


def check(text, others=(), min_chars=MIN_CHARS, near_duplicate_ratio=0.9):
    """Validate length, required sections and duplication.

    ``others`` are existing descriptions; a similarity at or above
    ``near_duplicate_ratio`` is reported as an error, because a reworded copy of
    another alpha's rationale reads as one mechanism and invites the same
    correlation verdict.
    """
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    errors = []
    length_ok = len(body) >= min_chars
    if not length_ok:
        errors.append("length %d < %d" % (len(body), min_chars))
    sections, template_errors = _sections(body)
    errors.extend(template_errors)
    worst = None
    for other in others or ():
        candidate = (other or "").strip()
        if not candidate:
            continue
        ratio = difflib.SequenceMatcher(None, body, candidate).ratio()
        if worst is None or ratio > worst[1]:
            worst = (candidate[:60], ratio)
        if ratio >= near_duplicate_ratio:
            errors.append("near-duplicate of an existing description (similarity %.2f)" % ratio)
            break
    return {
        "ok": not errors,
        "length": len(body),
        "min_chars": min_chars,
        "errors": errors,
        "template_ok": not template_errors,
        "template_errors": template_errors,
        "required_ok": length_ok and not template_errors,
        "sections": sections,
        "closest_existing": {"preview": worst[0], "ratio": round(worst[1], 3)} if worst else None,
    }
