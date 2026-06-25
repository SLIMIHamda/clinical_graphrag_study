"""Frozen prompt templates, held identical across all arms (prompt parity).

Prompt parity is a validity control (Doc 1 section 6): differences between arms
must be attributable to the manipulated factor, not to prompt wording. The
template set is versioned and hashed; the hash lands in every run-record.

Answer-type taxonomy (from the Benchmarks sheet `type` column):
  mcq            -> single letter A-D            (MMLU-Med, MedQA-US, MedMCQA)
  yes_no_maybe   -> yes | no | maybe             (PubMedQA)
  yes_no         -> yes | no                      (BioASQ-YN)
  free           -> free-form long answer         (GraphRAG-Bench, Clinical-Set)
"""

from __future__ import annotations

import hashlib
import json

SYSTEM = (
    "You are a careful medical question-answering assistant. "
    "Answer strictly in the requested format and nothing else."
)

# Output-format instruction per answer type. Kept terse and identical across
# arms so the normalized output schema is constant (kills the EM-artifact defect).
FORMAT_INSTRUCTION = {
    "mcq": "Respond with a single capital letter (A, B, C, or D) and nothing else.",
    "yes_no_maybe": "Respond with exactly one word: yes, no, or maybe.",
    "yes_no": "Respond with exactly one word: yes or no.",
    "free": "Answer concisely and ground every claim in the provided context.",
}

# Benchmark `type` strings -> internal answer type.
BENCHMARK_TYPE_TO_ANSWER = {
    "MCQ": "mcq",
    "yes/no/maybe": "yes_no_maybe",
    "yes/no": "yes_no",
    "long-form": "free",
    "mixed": "free",
}

PROMPT_SET_VERSION = "v1"


def answer_type_for(benchmark_type: str) -> str:
    return BENCHMARK_TYPE_TO_ANSWER.get(benchmark_type, "free")


def build_messages(
    question: str,
    answer_type: str,
    *,
    options: dict[str, str] | None = None,
    context: str | None = None,
) -> list[dict[str, str]]:
    """Assemble chat messages for one item. ``context`` is the retrieved passages
    block (None for No-RAG / closed-book)."""
    parts: list[str] = []
    if context:
        parts.append("Context:\n" + context.strip())
    parts.append("Question:\n" + question.strip())
    if options:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        parts.append("Options:\n" + opts)
    parts.append(FORMAT_INSTRUCTION.get(answer_type, FORMAT_INSTRUCTION["free"]))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def prompt_set_hash() -> str:
    """Stable hash of the frozen template set (system + format instructions)."""
    blob = json.dumps(
        {"version": PROMPT_SET_VERSION, "system": SYSTEM, "format": FORMAT_INSTRUCTION},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
