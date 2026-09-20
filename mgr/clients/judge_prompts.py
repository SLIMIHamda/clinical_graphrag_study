"""Shared judge prompts, parsers and token budgets.

There are two judges — :mod:`mgr.clients.nim_adapters` (hosted) and
:mod:`mgr.clients.local_adapters` (in-process, the only one permitted on chart
text). They must ask the *same question in the same words*, or a NIM-judged run
and a locally-judged run are not comparable and the RAGAS columns silently stop
meaning one thing. Prompt parity is already the rule for generation
(``configs/base.yaml``); this extends it to judging.

``MAX_TOKENS`` matters more than it looks. The generation client is configured
for MCQ answers (``max_new_tokens=8``), and a judge that inherits that budget
returns a truncated claim list and a cut-off number — which is how the first POC
produced faithfulness 0.15 and answer-relevance 0.38 from answers that had been
cut to eight tokens against a prompt missing the question. Every judge call here
sets its own budget explicitly.
"""

from __future__ import annotations

import re

JUDGE_SYSTEM = {
    "role": "system",
    "content": "You are a meticulous biomedical evaluation judge. Follow the output format exactly.",
}

#: Output budget per call kind. A yes/no answer needs a few tokens; a claim list
#: needs room to finish, or faithfulness is computed over a truncated list.
MAX_TOKENS = {
    "decompose": 256,
    "entails": 8,
    "relevant": 8,
    "relevance": 12,
    "entities": 192,
}


# -- prompts ----------------------------------------------------------------
# Wording is load-bearing: both adapters and their tests key off these strings.

def decompose_prompt(text: str) -> str:
    return "Break the following answer into atomic factual claims, one per line:\n\n" + text


def entails_prompt(hypothesis: str, premise: str) -> str:
    return (
        f"Premise:\n{premise}\n\nClaim:\n{hypothesis}\n\n"
        "Is the claim fully supported by the premise? Answer yes or no."
    )


def relevant_prompt(question: str, passage: str) -> str:
    return (
        f"Question:\n{question}\n\nPassage:\n{passage}\n\n"
        "Is this passage relevant to answering the question? Answer yes or no."
    )


def relevance_prompt(question: str, answer: str) -> str:
    return (
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        "Rate how relevant the answer is to the question, from 0.0 to 1.0. Reply with only the number."
    )


def entities_prompt(text: str) -> str:
    return (
        "List the medical entities (diseases, drugs, procedures, findings) "
        "mentioned in the text, one per line:\n\n" + text
    )


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [JUDGE_SYSTEM, {"role": "user", "content": prompt}]


# -- parsers ----------------------------------------------------------------

def parse_yes(text: str) -> bool:
    return bool(re.match(r"\s*(yes|true|1)\b", text.strip(), re.I))


def parse_float(text: str, default: float = 0.0) -> float:
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group(0)) if m else default


def parse_lines(text: str) -> list[str]:
    """One item per line, with bullets and numbering stripped."""
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-*\d.)]+)\s*", "", ln).strip()
        if ln:
            out.append(ln)
    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
