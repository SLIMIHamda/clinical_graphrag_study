"""Answer extraction & normalization.

Maps a raw model completion to a canonical label for scoring. The *same*
normalizer is applied to every arm; the answer-format audit (mgr/eval) verifies
that the resulting label schema is identical across arms before any EM/F1 is
reported — this is the control for the thesis "EM 0.00 -> 0.76" artifact, which
arose from arms being scored under different output schemas.

Returns ``None`` when no valid label can be extracted (an explicit miss, never a
silent default), so coverage is measurable.
"""

from __future__ import annotations

import re

_MCQ_LABELS = {"A", "B", "C", "D"}
_YNM = {"yes", "no", "maybe"}


def normalize(raw: str | None, answer_type: str) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if answer_type == "mcq":
        return _extract_mcq(text)
    if answer_type in ("yes_no_maybe", "yes_no"):
        return _extract_yesno(text, allow_maybe=answer_type == "yes_no_maybe")
    return text or None  # free-form: identity (scored by F1/semantic, not EM schema)


def _extract_mcq(text: str) -> str | None:
    # Prefer a lone leading letter: "B", "B.", "(B)", "Answer: B".
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    # Fallback: a parenthesized or punctuated letter anywhere.
    m = re.search(r"[\(\[]?([ABCD])[\)\].:]", text.upper())
    return m.group(1) if m else None


def _extract_yesno(text: str, allow_maybe: bool) -> str | None:
    low = text.lower()
    m = re.search(r"\b(yes|no|maybe)\b", low)
    if not m:
        return None
    label = m.group(1)
    if label == "maybe" and not allow_maybe:
        return None
    return label


def label_schema(labels: list[str | None], answer_type: str) -> set[str]:
    """The realized label vocabulary for an arm (excludes misses).

    For typed benchmarks this should equal the expected closed set; divergence
    is what the audit flags.
    """
    return {lab for lab in labels if lab is not None}


def expected_schema(answer_type: str) -> set[str] | None:
    """The closed label set for a typed benchmark, or ``None`` for free-form."""
    return {
        "mcq": set(_MCQ_LABELS),
        "yes_no_maybe": set(_YNM),
        "yes_no": {"yes", "no"},
    }.get(answer_type)
