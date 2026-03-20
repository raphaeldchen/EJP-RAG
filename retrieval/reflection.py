import json
import re
from enum import Enum
from dataclasses import dataclass
from ollama import Client

from retrieval.config import OLLAMA_BASE_URL


class QueryIntent(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


@dataclass
class ReflectionResult:
    intent: QueryIntent
    reasoning: str
    rewritten_query: str | None = None


# Deterministic rewrite rules — applied before LLM classification.
# Key: regex pattern to match. Value: rewritten query string.
# Order matters — more specific patterns first.
_REWRITE_RULES: list[tuple[str, str]] = [
    (
        r"\bself.?defense\b|\bjustified.{0,20}force\b|\buse of force\b",
        "justifiable use of force elements and standards under 720 ILCS 5/7-1 Illinois",
    ),
    (
        r"\bviolat\w* probation\b|\bprobation violation\b|\bbreak\w* probation\b",
        "probation revocation hearing procedure and penalties under 730 ILCS 5/5-6-4 Illinois",
    ),
    (
        r"\bviolat\w* parole\b|\bparole violation\b",
        "parole revocation procedure and penalties under 730 ILCS 5/3-3-9 Illinois",
    ),
    (
        r"\binsanity.{0,20}defense\b|\bnot guilty.{0,20}reason\b",
        "affirmative defense of insanity elements under 720 ILCS 5/6-2 Illinois",
    ),
    (
        r"\bjuvenile.{0,30}(adult|transfer|prosecut)\b|(tried|charged).{0,20}adult.{0,20}juvenile",
        "juvenile automatic transfer discretionary transfer prosecution as adult under 705 ILCS 405/5-805 Juvenile Court Act Illinois",
    ),
    (
        r"\bexpunge\b|\bexpungement\b|\bseal.{0,10}record",
        "expungement and sealing of criminal records under 20 ILCS 2630/5.2 Illinois",
    ),
    (
        r"\bbail\b|\bbond\b.{0,20}(set|amount|post|deny)",
        "bail and pretrial detention conditions under 725 ILCS 5/110 Illinois",
    ),
]


_SYSTEM_PROMPT = """You are a query classifier for a legal research system covering Illinois criminal justice law.

Corpus:
- ILCS: criminal offenses, sentencing, definitions, affirmative defenses
- ISCR: procedural court rules (filing deadlines, appeals, discovery, jury selection)

Classify as exactly one of:
- in_scope: query is specific and uses legal terminology matching Illinois criminal law
- out_of_scope: unrelated to Illinois criminal law (other states, civil, off-topic)
- ambiguous: vague, colloquial, or jurisdiction-unclear — needs rewording

Respond ONLY with this JSON:
{
  "intent": "in_scope" | "out_of_scope" | "ambiguous",
  "reasoning": "one sentence",
  "rewritten_query": "rewritten string if ambiguous, else null"
}"""


def _apply_rewrite_rules(query: str) -> str | None:
    """Return a rewritten query if any deterministic rule matches, else None."""
    lower = query.lower()
    for pattern, rewrite in _REWRITE_RULES:
        if re.search(pattern, lower):
            return rewrite
    return None


def reflect(query: str, model: str = "llama3.2") -> ReflectionResult:
    # --- Deterministic rewrite pass (fast, no LLM) ---
    rewrite = _apply_rewrite_rules(query)
    if rewrite:
        return ReflectionResult(
            intent=QueryIntent.AMBIGUOUS,
            reasoning="Matched deterministic rewrite rule — colloquial phrasing replaced with statutory language.",
            rewritten_query=rewrite,
        )

    # --- LLM classification pass (for everything else) ---
    client = Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify: {query}"},
        ],
        options={"temperature": 0},
    )

    raw = response.message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)

    return ReflectionResult(
        intent=QueryIntent(parsed["intent"]),
        reasoning=parsed["reasoning"],
        rewritten_query=(
            parsed.get("rewritten_query")
            if parsed.get("rewritten_query") not in (None, "null", "")
            else None
        ),
    )