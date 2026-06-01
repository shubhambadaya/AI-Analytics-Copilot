import json
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import MLAgentPlan
from src.context.builder import format_business_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Predictive ML Agent for an AI Analytics Pipeline.
Your ONLY job is to write secure Pandas Python code to train a machine learning classification model to answer the user's predictive question (e.g., "who should we upgrade", "predict churn likelihood").

CRITICAL RULES:
1. You MUST define a function `analyze(df)` that takes the main dataframe.
2. Store the final scored result in a variable named `result_df`.
3. Pre-injected modules: `pd`, `np`, `datetime`, `ml_engine`. Do NOT import anything else.
4. Use `ml_engine.train_and_predict(target_df, target_col='...', feature_cols=[...], id_col='...')` to automatically train a Random Forest and generate predictions/reasons.
5. You must prepare `target_df` by joining necessary tables (e.g., joining usage data with plan data) to construct your features.
6. HONOR THE BUSINESS CONTEXT: The Business Definitions & Rules section and the Contextual Profile (column descriptions, KPI formulas, learned business rules) are authoritative. When the prediction target or a feature depends on a business concept (e.g. "higher plan", "active user", "high value", "ARPU"), you MUST construct it using the EXACT definition/formula provided — do NOT invent an arbitrary threshold. Only fall back to a reasonable heuristic (e.g. `df['is_high_plan'] = df['plan_fee'] > 300`) if NO definition exists in the context.
7. Keep the `feature_cols` relevant to user behavior (usage, latency, complaints, tenure, etc.) avoiding data leakage, and prefer columns the business context describes as meaningful.
"""


def run_ml_agent(
    query: str,
    context_profile: Dict[str, Any],
    history: List[Dict[str, Any]] = None,
    provider: str = None,
    model_override: str = None
) -> MLAgentPlan:
    logger.info(f"ML Agent processing predictive query: '{query}'")

    prompt = f"Multi-Table Contextual Profile:\n{json.dumps(context_profile)}\n\nUser Question:\n{query}\n"

    # Surface dictionary KPIs and learned rules prominently so the target/features
    # are built from business definitions, not invented thresholds.
    business_ctx = format_business_context(context_profile)
    if business_ctx:
        prompt = (
            "BUSINESS DEFINITIONS & RULES (honor these exactly when defining the "
            f"prediction target and features):\n{business_ctx}\n\n" + prompt
        )

    if history:
        hist_str = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-3:]])
        prompt = f"Conversation History:\n{hist_str}\n\n" + prompt

    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=MLAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
