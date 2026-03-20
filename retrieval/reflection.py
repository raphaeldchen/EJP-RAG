import json
from enum import Enum
from dataclasses import dataclass
import anthropic

from retrieval.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


class QueryIntent(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


@dataclass
class ReflectionResult:
    intent: QueryIntent
    reasoning: str
    rewritten_query: str | None = None


_SYSTEM_PROMPT = """You are a query processor for a legal research system covering Illinois criminal justice law.

The corpus contains:
- ILCS (Illinois Compiled Statutes): criminal offenses (720 ILCS), criminal procedure (725 ILCS), corrections and sentencing (730 ILCS), juvenile justice (705 ILCS 405), mental health law (405 ILCS 5), expungement and sealing (20 ILCS 2630), and related acts
- ISCR (Illinois Supreme Court Rules): procedural court rules covering appeals, filing deadlines, discovery, and jury selection

Your job is to classify the query and, when needed, rewrite it into precise statutory language that will retrieve the most relevant chunks from the corpus.

Classify as exactly one of:
- in_scope: query is about Illinois criminal law — use this even for colloquial or vague queries that clearly relate to Illinois criminal topics; always provide a rewritten_query with the relevant ILCS citation(s) if you know them
- out_of_scope: unrelated to Illinois criminal law (federal law, other states, civil matters, completely off-topic)
- ambiguous: genuinely unclear whether the topic is within scope — needs clarification

When rewriting:
- Always include the most specific ILCS or Rule citation you know (e.g. "730 ILCS 5/5-4.5-30" for Class 1 felony sentencing, "720 ILCS 5/7-1" for self-defense, "705 ILCS 405/5-805" for juvenile transfer)
- Expand colloquial language into precise legal terminology (e.g. "beat a charge" → "affirmative defenses and grounds for suppression")
- Make the rewritten query specific enough to retrieve the right statutory sections
- Provide rewritten_query for ALL in_scope queries, not just ambiguous ones — it improves retrieval
- For multi-statute queries, include ALL relevant citations — e.g. a juvenile tried as an adult implicates 705 ILCS 405/5-805 (discretionary transfer), 705 ILCS 405/5-130 (mandatory transfer), 730 ILCS 5/5-4.5-105 (juvenile sentencing in adult court), and the underlying offense statute

Key citation mappings (use these to avoid common retrieval errors):
- "preliminary hearing" / "probable cause hearing" / "preliminary examination" → 725 ILCS 5/109-3 (NOT 725 ILCS 5/110-6.1, which is pretrial detention)
- "pretrial detention" / "detention hearing" / "bail" → 725 ILCS 5/110-6.1
- "expungement" / "sealing" of felony records → 20 ILCS 2630/5.2(b) for expungement eligibility, 20 ILCS 2630/5.2(d) for sealing; note that most felony convictions are NOT expungeable in Illinois — sealing is the primary remedy
- "Terry stop" / "investigatory stop" → 725 ILCS 5/107-14
- "search and seizure" / "suppression" → 725 ILCS 5/108-1 (search incident to arrest), 725 ILCS 5/114-12 (suppression motion)
- "discovery" by prosecution → 725 ILCS 5/114-9 (witness list), 725 ILCS 5/114-13 (exculpatory material), Illinois Supreme Court Rule 412
- "good time" / "sentence credit" / "day-for-day" → 730 ILCS 5/3-6-3
- "post-conviction" / "PCR" → 725 ILCS 5/122-1 (Post-Conviction Hearing Act)
- "mandatory transfer" of juvenile → 705 ILCS 405/5-130
- "discretionary transfer" of juvenile → 705 ILCS 405/5-805

Respond ONLY with this JSON:
{
  "intent": "in_scope" | "out_of_scope" | "ambiguous",
  "reasoning": "one sentence",
  "rewritten_query": "rewritten query with ILCS citations where known, else null"
}"""


def reflect(query: str) -> ReflectionResult:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        temperature=0,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Process this query: {query}"}],
    )

    raw = response.content[0].text.strip()
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
