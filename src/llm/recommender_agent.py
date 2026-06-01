import json
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import RecommenderAgentPlan
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Strategic Recommendation Agent for an AI Analytics Pipeline.
Your ONLY job is to read the synthesized insights AND the supporting data/statistics
from a business query, and formulate 2-4 concrete, actionable business recommendations.

CRITICAL RULES:
1. RECOMMENDATIONS ONLY: Do not summarize the insights. Jump straight to the "So what should we do?"
2. GROUND IN THE DATA: Every recommendation MUST cite specific figures from the Supporting Data
   (segment names, counts, percentages, averages, or statistical significance). Quantify the
   opportunity — e.g. "Target the 86 high-ARPU Metro customers..." not "Target high-value customers".
3. ACTIONABLE: Use strong verbs and name the exact segment/lever. Do not invent numbers that are
   not present in the supporting data.
"""

def run_recommender_agent(
    query: str,
    direct_answer: str,
    insights: List[str],
    evidence: str = None,
    provider: str = None,
    model_override: str = None
) -> RecommenderAgentPlan:
    """
    Formulate recommendations grounded in the actual result data.

    Args:
        evidence: Optional formatted string of the supporting result table and
            statistical findings, so recommendations can cite concrete figures.
    """
    logger.info("Recommendation Agent formulating strategies...")

    prompt = f"User Question: {query}\n\nAnswer: {direct_answer}\n\nInsights:\n" + "\n".join(f"- {i}" for i in insights)
    if evidence:
        prompt += f"\n\nSupporting Data & Statistics (cite these exact figures in every recommendation):\n{evidence}"

    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=RecommenderAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
