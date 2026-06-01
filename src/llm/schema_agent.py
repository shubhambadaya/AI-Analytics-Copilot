import json
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import SchemaAgentPlan
from src.context.builder import format_business_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Schema & Data Engineering Agent for an AI Analytics Pipeline.
Your ONLY job is to write secure Pandas Python code to extract, filter, join, and aggregate data to answer the user's query.

CRITICAL RULES:
1. You MUST define a function `analyze(df)` that takes the main dataframe.
2. Store the final aggregated result in a variable named `result_df`.
3. Pre-injected modules: `pd`, `np`, `datetime`. Do NOT import anything else.
4. Security: No eval/exec/open/dunder access.
5. All datasets in the Context Profile are pre-loaded in the global namespace as variables (e.g., `subscriber_profile`). Use `pd.merge()` to join them.
6. Return raw, unbinned numeric columns if a distribution/histogram is requested.
7. HONOR THE BUSINESS CONTEXT: When the query depends on a business concept (e.g. "higher plan", "active user", "high value", or a KPI like ARPU), compute it using the EXACT definition/formula from the Business Definitions & Rules and the column descriptions — do NOT invent thresholds or formulas of your own.
"""

def run_schema_agent(
    query: str,
    context_profile: Dict[str, Any],
    history: List[Dict[str, Any]] = None,
    provider: str = None,
    model_override: str = None
) -> SchemaAgentPlan:
    logger.info(f"Schema Agent processing query: '{query}'")

    prompt = f"Multi-Table Contextual Profile:\n{json.dumps(context_profile)}\n\nUser Question:\n{query}\n"

    # Surface dictionary KPIs and learned rules prominently so filters/segments/
    # metrics follow the business's own definitions rather than invented ones.
    business_ctx = format_business_context(context_profile)
    if business_ctx:
        prompt = (
            "BUSINESS DEFINITIONS & RULES (honor these exactly when filtering, "
            f"segmenting, or computing metrics):\n{business_ctx}\n\n" + prompt
        )

    if history:
        hist_str = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-3:]])
        prompt = f"Conversation History:\n{hist_str}\n\n" + prompt
        
    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=SchemaAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
