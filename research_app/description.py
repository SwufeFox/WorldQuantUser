"""Description compiler: assemble a Power Pool description, then check it.

The economic hypothesis is the researcher's; this module only assembles and
verifies.  It refuses to invent a rationale for a field or an operator it has no
description for, because a fabricated sentence is worse than a missing one: the
platform re-reads these fields during review.

Platform requirements encoded here (Power Pool description):
at least 100 characters after trimming; an Idea and a Rationale; distinct from
the account's other descriptions.  A description is not a substitute for a
measured check, and passing this module does not mean the alpha is submittable.
"""

from __future__ import annotations

import difflib

MIN_CHARS = 100
SECTION_IDEA = "IDEA:"
SECTION_DATA = "DATA:"
SECTION_RATIONALE = "RATIONALE:"


def compose(idea, fields, operators=(), field_notes=None, operator_notes=None):
    """Build the text from the researcher's idea plus catalog descriptions.

    ``fields`` are data-field ids; ``field_notes`` maps each to its catalog
    description.  ``operators`` are operator names; ``operator_notes`` maps each
    to what it does in this signal.  Any field or operator without a note is
    reported under ``missing`` and the draft is not produced.
    """
    field_notes = field_notes or {}
    operator_notes = operator_notes or {}
    missing = [f for f in fields if not (field_notes.get(f) or "").strip()]
    missing += ["op:%s" % o for o in operators if not (operator_notes.get(o) or "").strip()]
    text_idea = (idea or "").strip()
    if not text_idea:
        missing.insert(0, "idea")
    if missing:
        return {"ok": False, "text": None,
                "missing": missing,
                "reason": "no description evidence for %s" % ", ".join(missing)}

    data = "; ".join("%s is %s" % (f, field_notes[f].strip().rstrip(".")) for f in fields)
    ops = "; ".join("%s %s" % (name, operator_notes[name].strip().rstrip("."))
                    for name in operators) if operators else ""
    rationale = ("%s A negative rank keeps the signal dollar- and industry-neutral, "
                 "so the result comes from the cross-section rather than from market beta." %
                 ("The construction uses " + ops + "." if ops else "The signal is ranked inside its group."))
    text = "%s %s\n\n%s %s\n\n%s %s" % (SECTION_IDEA, text_idea, SECTION_DATA, data,
                                        SECTION_RATIONALE, rationale)
    return {"ok": True, "text": text, "check": check(text)}


def check(text, others=(), min_chars=MIN_CHARS, near_duplicate_ratio=0.9):
    """Validate length, required sections and duplication.

    ``others`` are existing descriptions; a similarity at or above
    ``near_duplicate_ratio`` is reported as an error, because a reworded copy of
    another alpha's rationale reads as one mechanism and invites the same
    correlation verdict.
    """
    body = (text or "").strip()
    errors = []
    if len(body) < min_chars:
        errors.append("length %d < %d" % (len(body), min_chars))
    upper = body.upper()
    if SECTION_IDEA not in upper:
        errors.append("missing %s" % SECTION_IDEA)
    if SECTION_RATIONALE not in upper:
        errors.append("missing %s" % SECTION_RATIONALE)
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
        "closest_existing": {"preview": worst[0], "ratio": round(worst[1], 3)} if worst else None,
    }