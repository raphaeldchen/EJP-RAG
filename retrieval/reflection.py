import json
from enum import Enum
from dataclasses import dataclass
import anthropic
import ollama

from retrieval.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, OLLAMA_BASE_URL


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
- Include the most specific ILCS or Rule citation you know based on your knowledge of Illinois law
- Expand colloquial language into precise legal terminology (e.g. "beat a charge" → "affirmative defenses and grounds for suppression")
- Make the rewritten query specific enough to retrieve the right statutory sections
- Provide rewritten_query for ALL in_scope queries, not just ambiguous ones — it improves retrieval
- For multi-statute queries, include ALL relevant citations you know

Respond ONLY with this JSON:
{
  "intent": "in_scope" | "out_of_scope" | "ambiguous",
  "reasoning": "one sentence",
  "rewritten_query": "rewritten query with ILCS citations where known, else null"
}"""


def reflect(query: str, use_local: bool = False) -> ReflectionResult:
    if use_local:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Process this query: {query}"},
            ],
            format="json",
            options={"temperature": 0},
        )
        raw = response.message.content.strip()
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
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
