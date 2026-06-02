import json
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import CritiqueResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a skeptical senior analyst reviewing a colleague's findings BEFORE they reach a stakeholder.
You are given the business question, the proposed answer + insights, and the supporting data and statistics.

Your job:
1. Check that EVERY claim is actually supported by the provided numbers/statistics. Flag anything asserted
   but not evidenced, any internal contradiction, and any part of the question that was not answered.
   STATISTICAL DISCIPLINE: cross-check claims against the supplied statistical tests. If the answer calls a
   difference/relationship "significant" but no test shows p < 0.05 (or no test was run), demote it to a
   descriptive observation in the refined_answer and note it. If a test shows non-significance, ensure the
   answer reflects that rather than overstating the effect.
2. Produce a `refined_answer` that keeps ONLY what the evidence supports — qualify or remove overreaching
   claims, and never invent new numbers. If the original answer is sound, return it essentially unchanged.
3. List honest `caveats` a decision-maker should know (small sample, proxy metric, correlation is not
   causation, missing segments, non-significant differences).
4. Give a grounded `confidence` (0-1): high ONLY when the claims rest on significant, sufficient data.

Be rigorous but fair: if it holds up, pass it.
"""


def run_critique(
    query: str,
    direct_answer: str,
    insights: List[str],
    evidence: Dict[str, Any],
    provider: str = None,
    model_override: str = None
) -> CritiqueResult:
    """
    Review-and-refine pass: validate the synthesized answer against the evidence,
    tighten unsupported claims, surface caveats, and return a grounded confidence.
    """
    logger.info("Critic reviewing the answer against the evidence...")

    insights_str = "\n".join(f"- {i}" for i in (insights or []))
    evidence_str = json.dumps(evidence)[:6000]  # bound tokens

    prompt = (
        f"Business question: {query}\n\n"
        f"Proposed answer: {direct_answer}\n\n"
        f"Insights:\n{insights_str}\n\n"
        f"Supporting data & statistics:\n{evidence_str}"
    )

    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=CritiqueResult,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
