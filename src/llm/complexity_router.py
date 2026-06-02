from src.llm.client import llm_client, MODEL_FAST
from src.llm.schemas import ComplexityClassification
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """
You are a query classifier for an AI Analytics Copilot.
Classify the user's input as SIMPLE, COMPLEX, PREDICTIVE, or LEARN_RULE.

SIMPLE queries are single-step operations:
- Direct counts: "How many customers do we have?"
- Single aggregations: "What is the average revenue?"
- Basic lookups: "Show me the top 10 plans by usage"

COMPLEX queries require multi-step reasoning:
- Comparisons across groups: "Compare Plan 349 users vs lower plan users"
- Correlations: "Is there a relationship between data usage and churn?"
- Why/How questions: "Why are users churning?"

PREDICTIVE queries require machine learning classification:
- Recommendations: "Train a system to recommend who should be upgraded"
- Predictive Scoring: "Score users based on their likelihood to churn"
- "Who should we upgrade to Plan 349 and why?"

LEARN_RULE is NOT a data question — the user is teaching you a business rule,
definition, or preference to remember for the future:
- "Active users are those with more than 5GB of data usage"
- "ARPU means total recharge divided by subscriber count"
- "Always use histograms for distributions"
When you choose LEARN_RULE, put the exact, concise rule to memorize in `learned_rule`.

Also set `wants_recommendations`: True ONLY if the user explicitly asks for recommendations,
actions, advice, strategy, or how to improve/fix/increase/reduce/grow something
(e.g. "what should we do", "how can we improve ARPU", "recommend who to upgrade").
Set it False for purely descriptive or analytical questions ("show", "how many",
"compare", "which", "why is X") — for those, the user wants the finding, not advice.
"""

def classify_query_complexity(query: str, provider: str = None) -> ComplexityClassification:
    """
    Uses a fast, lightweight LLM call to classify query complexity.
    Uses Gemini Flash for speed (~1-2 seconds).
    """
    logger.info(f"Classifying query complexity: '{query}'")
    
    prompt = f'Classify this business analytics query:\n"{query}"'
    
    try:
        result = llm_client.generate_structured_output(
            prompt=prompt,
            response_model=ComplexityClassification,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            provider=provider,
            model_override=MODEL_FAST
        )
        logger.info(f"Query classified as {result.complexity}: {result.reasoning}")
        return result
    except Exception as e:
        logger.warning(f"Complexity classification failed ({e}), defaulting to COMPLEX.")
        return ComplexityClassification(
            complexity="COMPLEX",
            reasoning="Classification failed, defaulting to full analysis."
        )
