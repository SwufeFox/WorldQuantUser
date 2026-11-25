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


def compose(idea, fields, operators=(), field_notes=None, operator_notes=None, *,
            rationale=None):
    """Build the text from the researcher's idea plus catalog descriptions.

    ``fields`` are data-field ids; ``field_notes`` maps each to its catalog
    description.  ``operators`` are operator names; ``operator_notes`` maps each
    to what it does in this signal.  Any missing input is reported under
    ``missing`` and no draft is produced.
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

    data = "; ".join("%s is %s" % (f, field_notes[f].strip().rstrip(".")) for f in field_list) + "."
    ops = "; ".join("%s %s" % (name, operator_notes[name].strip().rstrip("."))
                    for name in op_list)
    if rationale is not None and str(rationale).strip():
        text_rationale = str(rationale).strip()
    elif ops:
        text_rationale = (
            "The signal combines %s through %s so cross-sectional variation "
            "in the documented inputs drives the resulting weights."
            % (", ".join(field_list), ops)
        )
    else:
        text_rationale = (
            "The signal maps %s directly to evaluate the stated hypothesis across "
            "instruments using only the documented input fields."
            % ", ".join(field_list)
        )
    text = "%s %s\n\n%s %s\n\n%s %s" % (SECTION_IDEA, text_idea, SECTION_DATA, data,
                                        SECTION_RATIONALE, text_rationale)
    checked = check(text)
    return {"ok": checked["ok"], "text": text, "check": checked}


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
